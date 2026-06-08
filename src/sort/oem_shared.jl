# oem_shared.jl — column-batched odd-even merge sort, shared-memory kernel.
#
# Sorts each column of a K×M matrix A in place, using shared memory.
# Handles K in roughly 65..4096. Each block sorts one column.
#
# Shared-memory layout:  shared_buf has K_PAD = next_pow2(K) entries.
# Items > K (up to K_PAD) are padded with typemax(T) so they sort to the
# end and can be discarded on write-back.
#
# Algorithm: same bitonic network as oem_warp.jl, applied across the
# entire shared buffer. log₂(K_PAD)·(log₂(K_PAD)+1)/2 stages total.
#
# Threading:
#   workgroup = min(K_PAD/2, 1024)
#   pair_slots_per_thread = K_PAD / 2 / workgroup    ∈ {1, 2}
#   load_slots_per_thread = K_PAD / workgroup        ∈ {2, 4}
# Both inner loops unrolled to a fixed max via @nexprs + compile-time
# guards. For K_PAD ≤ 2048 a single pair-slot per thread suffices; at
# K_PAD = 4096, two slots per thread per stage.

using KernelAbstractions
using KernelIntrinsics
import KernelIntrinsics as KI
using Base.Cartesian: @nexprs

# oem_warp.jl (OEM_WARPSZ, OEM_BLOCK, OEM_COLS_PER_BLOCK, oem_sort_columns_warp!)
# is included by KernelForge.jl immediately before this file — see there for the
# canonical include order. Re-including it here double-defined symbols and broke
# precompilation.


# ── Constants ──────────────────────────────────────────────────────────

const OEM_MAX_SLOTS = 8   # max pair-slots per thread; handles K_PAD ≤ 16384
const OEM_MAX_LOG2  = 12  # max log₂(K_PAD); handles K_PAD ≤ 4096


# ── Kernel ──────────────────────────────────────────────────────────────

@kernel inbounds = true unsafe_indices = true function oem_sort_shared_kernel!(
        A::AbstractMatrix{T},
        ::Val{K_PAD},
        ::Val{K_ACTUAL},
        ::Val{COLS_PER_BLOCK},
) where {T, K_PAD, K_ACTUAL, COLS_PER_BLOCK}
    @uniform begin
        workgroup       = Int(@groupsize()[1])
        threads_per_col = workgroup ÷ COLS_PER_BLOCK
    end
    lid = Int(@index(Local))
    gid = Int(@index(Group))

    # Decompose thread into (col_in_block, lid_in_col).
    col_in_block = (lid - 1) ÷ threads_per_col          # 0..COLS_PER_BLOCK-1
    lid_in_col   = (lid - 1) % threads_per_col + 1      # 1..threads_per_col

    M = size(A, 2)
    col_idx = (gid - 1) * COLS_PER_BLOCK + col_in_block + 1
    in_range = col_idx <= M

    # Sort in `uint_map` space — Ada FP64 cmp throughput is 1/64 of int;
    # mapping at load and unmapping at store turns the bitonic compares
    # into plain integer compares for free. For unsigned T this is a
    # no-op the compiler folds away.
    shared_buf = @localmem typeof(uint_map(zero(T))) (COLS_PER_BLOCK * K_PAD,)
    sh_off = col_in_block * K_PAD                       # base offset into shared

    # ─── Load: each thread reads K_PAD/threads_per_col positions ─────
    @nexprs 8 c -> begin
        pos_c = (c - 1) * threads_per_col + lid_in_col
        if pos_c <= K_PAD
            if in_range && pos_c <= K_ACTUAL
                shared_buf[sh_off + pos_c] = uint_map(A[pos_c, col_idx])
            else
                shared_buf[sh_off + pos_c] = uint_map(typemax(T))
            end
        end
    end
    @synchronize

    # ─── Bitonic sort ─────────────────────────────────────────────────
    @nexprs 12 lvl_p -> begin
        if (1 << lvl_p) <= K_PAD
            @nexprs 12 stg_q -> begin
                if stg_q <= lvl_p
                    q_idx = lvl_p - stg_q + 1
                    d_off = 1 << (q_idx - 1)
                    @nexprs 8 pslot -> begin
                        t_idx = (pslot - 1) * threads_per_col + lid_in_col - 1
                        if t_idx < (K_PAD >> 1)
                            mask_b = d_off - 1
                            i_b = ((t_idx & ~mask_b) << 1) | (t_idx & mask_b)
                            partner_b = i_b + d_off
                            asc_b = ((i_b >> lvl_p) & 1) == 0
                            a_b = shared_buf[sh_off + i_b + 1]
                            b_b = shared_buf[sh_off + partner_b + 1]
                            swap_b = asc_b ⊻ (a_b < b_b)
                            if swap_b
                                shared_buf[sh_off + i_b + 1]       = b_b
                                shared_buf[sh_off + partner_b + 1] = a_b
                            end
                        end
                    end
                    @synchronize
                end
            end
        end
    end

    # ─── Write back ──────────────────────────────────────────────────
    if in_range
        @nexprs 8 c -> begin
            pos_c = (c - 1) * threads_per_col + lid_in_col
            if pos_c <= K_ACTUAL
                A[pos_c, col_idx] = uint_unmap(T, shared_buf[sh_off + pos_c])
            end
        end
    end
end


# ── TAG kernel: shared-mem with valid-tag + custom lt ─────────────────

@kernel inbounds = true unsafe_indices = true function oem_sort_shared_tag_kernel!(
        A::AbstractMatrix{T},
        lt,
        ::Val{K_PAD},
        ::Val{K_ACTUAL},
        ::Val{COLS_PER_BLOCK},
) where {T, K_PAD, K_ACTUAL, COLS_PER_BLOCK}
    @uniform begin
        workgroup       = Int(@groupsize()[1])
        threads_per_col = workgroup ÷ COLS_PER_BLOCK
    end
    lid = Int(@index(Local))
    gid = Int(@index(Group))
    col_in_block = (lid - 1) ÷ threads_per_col
    lid_in_col   = (lid - 1) % threads_per_col + 1

    M = size(A, 2)
    col_idx = (gid - 1) * COLS_PER_BLOCK + col_in_block + 1
    in_range = col_idx <= M

    shared_v  = @localmem T    (COLS_PER_BLOCK * K_PAD,)
    shared_va = @localmem Bool (COLS_PER_BLOCK * K_PAD,)
    sh_off = col_in_block * K_PAD

    # ─── Load ────────────────────────────────────────────────────────
    @nexprs 8 c -> begin
        pos_c = (c - 1) * threads_per_col + lid_in_col
        if pos_c <= K_PAD
            is_valid_pos = in_range && pos_c <= K_ACTUAL
            # Read a clamped (known-valid) index when out of range — value is
            # masked out by `is_valid_pos=false` in the tag-aware compare.
            # Avoids `zero(T)`, which fails to compile for custom isbits T.
            safe_pos = is_valid_pos ? pos_c   : 1
            safe_col = is_valid_pos ? col_idx : 1
            shared_v[sh_off + pos_c]  = A[safe_pos, safe_col]
            shared_va[sh_off + pos_c] = is_valid_pos
        end
    end
    @synchronize

    # ─── Bitonic sort with tag-aware compare ────────────────────────
    @nexprs 12 lvl_p -> begin
        if (1 << lvl_p) <= K_PAD
            @nexprs 12 stg_q -> begin
                if stg_q <= lvl_p
                    q_idx = lvl_p - stg_q + 1
                    d_off = 1 << (q_idx - 1)
                    @nexprs 8 pslot -> begin
                        t_idx = (pslot - 1) * threads_per_col + lid_in_col - 1
                        if t_idx < (K_PAD >> 1)
                            mask_b = d_off - 1
                            i_b = ((t_idx & ~mask_b) << 1) | (t_idx & mask_b)
                            partner_b = i_b + d_off
                            ascending = ((i_b >> lvl_p) & 1) == 0
                            a_v  = shared_v[sh_off + i_b + 1]
                            a_va = shared_va[sh_off + i_b + 1]
                            b_v  = shared_v[sh_off + partner_b + 1]
                            b_va = shared_va[sh_off + partner_b + 1]
                            is_less = (a_va & !b_va) | (a_va & b_va & lt(a_v, b_v))
                            if ascending ⊻ is_less
                                shared_v[sh_off + i_b + 1]       = b_v
                                shared_va[sh_off + i_b + 1]      = b_va
                                shared_v[sh_off + partner_b + 1] = a_v
                                shared_va[sh_off + partner_b + 1] = a_va
                            end
                        end
                    end
                    @synchronize
                end
            end
        end
    end

    # ─── Write back ─────────────────────────────────────────────────
    if in_range
        @nexprs 8 c -> begin
            pos_c = (c - 1) * threads_per_col + lid_in_col
            if pos_c <= K_ACTUAL
                A[pos_c, col_idx] = shared_v[sh_off + pos_c]
            end
        end
    end
end


# ── Driver ─────────────────────────────────────────────────────────────

"""
    oem_sort_columns_shared!(A; backend, lt) -> A

Sort each column of `A` (K×M) in place via the shared-mem kernel.
Handles K up to ~4096; for K ≤ 64 prefer the warp kernel.

If `lt === nothing` (default) and T has `typemax`, uses the fast typemax
sentinel path. Otherwise (custom `lt` or no `typemax`), uses a valid-tag
fallback kernel that doesn't depend on a sentinel value.
"""
function oem_sort_columns_shared!(A::AbstractMatrix{T};
                                  backend = get_backend(A),
                                  lt = nothing) where {T}
    K, M = size(A)

    k_pad = 1
    while k_pad < K
        k_pad <<= 1
    end
    @assert k_pad <= 4096 "shared kernel handles K_PAD ≤ 4096; got K_PAD=$k_pad"

    pair_slots = max(1, (k_pad ÷ 2 + 1023) ÷ 1024)
    threads_per_col = (k_pad ÷ 2) ÷ pair_slots
    if threads_per_col < OEM_WARPSZ
        threads_per_col = OEM_WARPSZ
    end

    target_workgroup = 256
    cols_per_block = max(1, target_workgroup ÷ threads_per_col)
    while cols_per_block * threads_per_col > 1024
        cols_per_block ÷= 2
    end

    workgroup = cols_per_block * threads_per_col
    nblocks   = cld(M, cols_per_block)

    use_tag = (lt !== nothing) || !hasmethod(typemax, Tuple{Type{T}})
    lt_used = lt === nothing ? (<) : lt

    if use_tag
        oem_sort_shared_tag_kernel!(backend, workgroup)(
            A, lt_used, Val(k_pad), Val(K), Val(cols_per_block);
            ndrange = nblocks * workgroup,
        )
    else
        oem_sort_shared_kernel!(backend, workgroup)(
            A, Val(k_pad), Val(K), Val(cols_per_block);
            ndrange = nblocks * workgroup,
        )
    end
    return A
end


# ── Unified dispatcher ─────────────────────────────────────────────────

"""
    oem_sort_columns!(A; backend, lt) -> A

Dispatch the right kernel based on K: warp (K ≤ 64) or shared (K ≤ 4096).

`lt` is the comparator. If `lt === nothing` (default) and `T` has a
`typemax` method, the fast typemax-sentinel path is used. Otherwise a
valid-tag fallback (one `Bool` per item) is used with `lt = something or `<``.
"""
function oem_sort_columns!(A::AbstractMatrix{T};
                           backend = get_backend(A),
                           lt = nothing) where {T}
    K = size(A, 1)
    if K <= 64
        return oem_sort_columns_warp!(A; backend, lt)
    else
        return oem_sort_columns_shared!(A; backend, lt)
    end
end


# ── Smoke ──────────────────────────────────────────────────────────────

