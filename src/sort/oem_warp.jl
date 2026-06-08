# oem_warp.jl — column-batched odd-even merge sort, warp-level kernel.
#
# Sorts each column of an K×M matrix A independently, using ONLY registers
# and warp shuffles. Two regimes:
#
#   K ≤ 32: one lane per item, one warp per column.
#   K ≤ 64: two items per thread, one warp per column.
#
# The sort network is **bitonic** (functionally equivalent to Batcher's
# odd-even mergesort: same O(N log²N) compare-exchange count, both
# data-independent). Bitonic maps cleanly onto `@shfl(Xor, val, mask)`:
# at stage with stride d = 1 << (q-1), each lane's partner is `lane ⊻ d`.
#
# Comparator: native `<` via `min`/`max`. Comparator-clean (no `uint_map`).

using KernelAbstractions
using KernelIntrinsics
import KernelIntrinsics as KI
using Base.Cartesian: @nexprs


# ── Constants ──────────────────────────────────────────────────────────

const OEM_WARPSZ = 32
const OEM_BLOCK  = 128                  # 4 warps × 32 lanes
const OEM_COLS_PER_BLOCK = OEM_BLOCK ÷ OEM_WARPSZ

# For K=33..64 the warp kernel can route to:
#   :twoitem  — 1 warp/col, 2 items/lane          (4 cols/block, no shmem)
#   :dualwarp — 2 warps/col, 1 item/lane          (2 cols/block, 1 KB shmem)
# `:twoitem` wins across every measured dtype on Ada (Float64 K=64
# M=65536: 1257 vs 1341 µs after the branchless-min/max fix below). Kept
# as a per-dtype switch in case a future arch flips the balance.
@inline default_k64_variant(::Type{T}) where {T} = :twoitem


# ── Compare-exchange decision ──────────────────────────────────────────
#
# For a bitonic stage at level p (subnet size k = 1 << p) and sub-stage
# stride d = 1 << (q-1), each lane at position i:
#   - partner is at i ⊻ d (got via @shfl(Xor, ...))
#   - ascending = ((i >> p) & 1) == 0     (low half of size-2^(p+1) chunk)
#   - am_low    = ((i >> (q-1)) & 1) == 0 (low position of the pair)
#   - take_min  = am_low == ascending
#     • am_low + ascending → I should hold the min
#     • !am_low + !ascending → I should hold the min
#     • the other two cases → I should hold the max

@inline cmpex(my_val, partner_val, take_min::Bool) =
    take_min ? _branchless_min(my_val, partner_val) :
               _branchless_max(my_val, partner_val)

# Branchless compare-select that lowers to a single `min.f64` / `max.f64`
# (PTX) or `cmov` (integer). Julia's `Base.min(::Float64, ::Float64)` is
# NaN-aware and branches per call → it serialises the unrolled 21 bitonic
# stages and reduces the warp kernel to ~7 % of peak BW for Float64. The
# input distribution for `sort_columns!` excludes NaNs (random/`randn`),
# so the branchless variant is observationally equivalent. Integer types
# inline trivially.
@inline _branchless_min(a, b) = ifelse(a < b, a, b)
@inline _branchless_max(a, b) = ifelse(a < b, b, a)


# ── Tag-aware compare ──────────────────────────────────────────────────
#
# When T has no `typemax` (or the user supplies a custom `lt`), we carry
# a `valid` bool alongside each value. Invalid items sort to the END
# (above any valid item under the user's `lt`).
#
# is_less((lo_v, lo_va), (hi_v, hi_va)):
#   true   iff lo sorts strictly before hi
#   = (lo_va & !hi_va)                            # valid before invalid
#   | (lo_va & hi_va & lt(lo_v, hi_v))             # both valid → user lt
#
# Per-lane swap decision in bitonic stage:
#   collect (lo_v, lo_va, hi_v, hi_va) by swapping my/partner based on am_low
#   swap = ascending ⊻ is_less(lo, hi)
#   if swap: my_val, valid := partner_val, partner_valid

@inline function tag_swap_decision(am_low::Bool, ascending::Bool,
                                   my_val, my_valid::Bool,
                                   partner_val, partner_valid::Bool,
                                   lt)
    lo_v  = am_low ? my_val       : partner_val
    lo_va = am_low ? my_valid     : partner_valid
    hi_v  = am_low ? partner_val  : my_val
    hi_va = am_low ? partner_valid : my_valid
    is_less = (lo_va & !hi_va) | (lo_va & hi_va & lt(lo_v, hi_v))
    return ascending ⊻ is_less
end


# ── Kernel: K ≤ 32 (multi-column-per-warp packing) ────────────────────
#
# Pack `OEM_WARPSZ / K_PAD` columns per warp. Each column occupies a
# contiguous sub-warp of K_PAD lanes. Bitonic sort within each sub-warp
# uses XOR shuffle with mask ∈ {1, 2, …, K_PAD/2} — all < K_PAD, so
# partners stay inside the sub-warp.
#
# For K_PAD = 32: cols_per_warp = 1 (same as the original 1-warp-per-col).
# For K_PAD = 16:  2 cols / warp ×  4 warps/block =  8 cols/block.
# For K_PAD =  8:  4 cols / warp ×  4 warps/block = 16 cols/block.
# For K_PAD =  4:  8 cols / warp ×  4 warps/block = 32 cols/block.
# For K_PAD =  2: 16 cols / warp ×  4 warps/block = 64 cols/block.
# For K_PAD =  1: trivial — every lane is its own column, just copy.

@kernel inbounds = true unsafe_indices = true function oem_sort_packed_kernel!(
        A::AbstractMatrix{T},
        ::Val{K_PAD},          # power of 2 ≥ K_ACTUAL, ≤ 32
        ::Val{K_ACTUAL},
) where {T, K_PAD, K_ACTUAL}
    @uniform begin
        cols_per_warp  = OEM_WARPSZ ÷ K_PAD
        cols_per_block = OEM_BLOCK   ÷ K_PAD
    end
    lid = Int(@index(Local))
    gid = Int(@index(Group))
    warp_id_in_block = (lid - 1) ÷ OEM_WARPSZ + 1     # 1..4
    lane_in_warp     = (lid - 1) % OEM_WARPSZ         # 0..31
    sub_warp_id      = lane_in_warp ÷ K_PAD           # 0..cols_per_warp-1
    sub_lane         = lane_in_warp % K_PAD           # 0..K_PAD-1 (i.e. position 0..K_PAD-1)

    M = size(A, 2)
    col_idx = (gid - 1) * cols_per_block +
              (warp_id_in_block - 1) * cols_per_warp +
              sub_warp_id + 1
    in_range = col_idx <= M

    # Load: typemax pad for sub_lane ≥ K_ACTUAL (per-column padding).
    # See note in `oem_sort64_kernel!`: sort in `uint_map` space so the
    # bitonic compares hit the int rate on FP types.
    v = (in_range && (sub_lane + 1) <= K_ACTUAL) ?
        uint_map(A[sub_lane + 1, col_idx]) : uint_map(typemax(T))

    if in_range
        # Bitonic over K_PAD positions inside the sub-warp.
        @nexprs 5 lvl_p -> begin
            if (1 << lvl_p) <= K_PAD
                @nexprs 5 stg_q -> begin
                    if stg_q <= lvl_p
                        q_idx = lvl_p - stg_q + 1
                        d_off = 1 << (q_idx - 1)
                        partner_v = @shfl(Xor, v, d_off)
                        ascending = ((sub_lane >> lvl_p) & 1) == 0
                        am_low    = ((sub_lane >> (q_idx - 1)) & 1) == 0
                        take_min  = (am_low == ascending)
                        v = cmpex(v, partner_v, take_min)
                    end
                end
            end
        end

        if (sub_lane + 1) <= K_ACTUAL
            A[sub_lane + 1, col_idx] = uint_unmap(T, v)
        end
    end
end


# ── Kernel: K ≤ 64 (two values per lane, halved layout) ───────────────
#
# Lane t (1-indexed) holds:
#   v_lo = A[t,        col]   at "position" i_lo = t - 1     ∈ 0..31
#   v_hi = A[t + 32,   col]   at "position" i_hi = t + 31    ∈ 32..63
#
# For bitonic-64, stride d = 1 << (q-1) ∈ {1, 2, 4, 8, 16, 32}:
#   - d < 32: partner is in same half. Use @shfl(Xor, v_*, d) on each
#     half independently. v_lo and v_hi share the same shuffle offset.
#   - d = 32: partner is in the OTHER half. Compare v_lo with v_hi
#     intra-thread; swap as needed. No shuffle.
#
# Direction and am-low bits are computed from i_lo / i_hi separately
# because high half flips bit 5.

@kernel inbounds = true unsafe_indices = true function oem_sort64_kernel!(
        A::AbstractMatrix{T},
        ::Val{K_ACTUAL},
) where {T, K_ACTUAL}
    lid = Int(@index(Local))
    gid = Int(@index(Group))
    warp_id_in_block = (lid - 1) ÷ OEM_WARPSZ + 1
    lane             = (lid - 1) % OEM_WARPSZ + 1
    i_lo = lane - 1
    i_hi = lane - 1 + 32

    M = size(A, 2)
    col_idx = (gid - 1) * OEM_COLS_PER_BLOCK + warp_id_in_block
    in_range = col_idx <= M

    # Sort in `uint_map` space so `cmpex`/`<` compile to int compares.
    # NVIDIA Ada (sm_89) consumer GPUs run `fcmp.f64` at 1/64 of `icmp.u64`
    # throughput and `fcmp.f32` at 1/4 — converting before the bitonic
    # network and unconverting at write-back is a pure register-side
    # transform with no extra memory traffic. For unsigned T, `uint_map`
    # is the identity and the compiler folds the transform away.
    v_lo = (in_range && lane <= K_ACTUAL) ?
           uint_map(A[lane, col_idx]) : uint_map(typemax(T))
    v_hi = (in_range && (lane + 32) <= K_ACTUAL) ?
           uint_map(A[lane + 32, col_idx]) : uint_map(typemax(T))

    if in_range
        # Bitonic-64: 6 levels, 21 stages total.
        @nexprs 6 lvl_p -> begin
            @nexprs 6 stg_q -> begin
                if stg_q <= lvl_p
                    q_idx = lvl_p - stg_q + 1
                    d_off = 1 << (q_idx - 1)
                    if d_off == 32
                        # Intra-thread compare-exchange between v_lo and v_hi.
                        # At q=6 (only possible when lvl_p=6), pair is (i_lo, i_lo+32).
                        # v_lo is "low" of pair, v_hi is "high".
                        # ascending = bit lvl_p of i_lo == 0. For lvl_p=6 and
                        # i_lo < 32, asc = T.
                        asc = ((i_lo >> lvl_p) & 1) == 0
                        if asc
                            new_lo = _branchless_min(v_lo, v_hi)
                            new_hi = _branchless_max(v_lo, v_hi)
                        else
                            new_lo = _branchless_max(v_lo, v_hi)
                            new_hi = _branchless_min(v_lo, v_hi)
                        end
                        v_lo = new_lo
                        v_hi = new_hi
                    else
                        plo = @shfl(Xor, v_lo, d_off)
                        phi = @shfl(Xor, v_hi, d_off)
                        # v_lo decision
                        asc_lo = ((i_lo >> lvl_p) & 1) == 0
                        amlow_lo = ((i_lo >> (q_idx - 1)) & 1) == 0
                        v_lo = cmpex(v_lo, plo, amlow_lo == asc_lo)
                        # v_hi decision
                        asc_hi = ((i_hi >> lvl_p) & 1) == 0
                        amlow_hi = ((i_hi >> (q_idx - 1)) & 1) == 0
                        v_hi = cmpex(v_hi, phi, amlow_hi == asc_hi)
                    end
                end
            end
        end

        if lane <= K_ACTUAL
            A[lane, col_idx] = uint_unmap(T, v_lo)
        end
        if (lane + 32) <= K_ACTUAL
            A[lane + 32, col_idx] = uint_unmap(T, v_hi)
        end
    end
end


# ── Kernel: K ≤ 64 (two warps per column, cross-warp merge) ───────────
#
# Each column uses 2 warps (64 lanes total), with 1 value per lane. The
# warps cover positions 0..31 (warp 0 of the pair) and 32..63 (warp 1).
# Workgroup = OEM_BLOCK = 128 → 4 warps → 2 columns per block.
#
# Sort proceeds in standard bitonic order over 64 positions:
#   - Stages with stride d < 32: partner is in same warp → @shfl(Xor).
#   - Stage with stride d = 32: cross-warp → exchange via a small
#     shared-mem buffer (2 cols × 64 entries × sizeof(T)).
#
# Direction (`ascending`) and `am_low` are computed from the GLOBAL
# position (0..63), not the warp-local lane.

@kernel inbounds = true unsafe_indices = true function oem_sort64_dualwarp_kernel!(
        A::AbstractMatrix{T},
        ::Val{K_ACTUAL},
) where {T, K_ACTUAL}
    lid = Int(@index(Local))
    gid = Int(@index(Group))

    # 4 warps × 32 lanes = 128 = workgroup.
    # 2 warps per column ⇒ 2 columns per block.
    warp_id_in_block = (lid - 1) ÷ OEM_WARPSZ + 1     # 1..4
    lane_in_warp     = (lid - 1) % OEM_WARPSZ + 1     # 1..32
    col_in_block     = (warp_id_in_block - 1) ÷ 2     # 0 or 1
    warp_in_col      = (warp_id_in_block - 1) % 2     # 0 or 1
    position         = warp_in_col * 32 + (lane_in_warp - 1)   # 0..63

    M = size(A, 2)
    col_idx = (gid - 1) * 2 + col_in_block + 1
    in_range = col_idx <= M

    # Shared buffer carries the `uint_map`'d value so the cross-warp
    # exchange and the bitonic compares operate in int space.
    shared_buf = @localmem typeof(uint_map(zero(T))) (128,)

    v = (in_range && (position + 1) <= K_ACTUAL) ?
        uint_map(A[position + 1, col_idx]) : uint_map(typemax(T))

    @nexprs 6 lvl_p -> begin
        @nexprs 6 stg_q -> begin
            if stg_q <= lvl_p
                q_idx = lvl_p - stg_q + 1
                d_off = 1 << (q_idx - 1)
                if d_off < 32
                    partner_v = @shfl(Xor, v, d_off)
                    ascending = ((position >> lvl_p) & 1) == 0
                    am_low    = ((position >> (q_idx - 1)) & 1) == 0
                    v = cmpex(v, partner_v, am_low == ascending)
                else
                    # Cross-warp exchange via shared memory.
                    sh_off = col_in_block * 64
                    shared_buf[sh_off + position + 1] = v
                    @synchronize
                    partner_pos = position ⊻ 32
                    partner_v = shared_buf[sh_off + partner_pos + 1]
                    @synchronize
                    ascending = ((position >> lvl_p) & 1) == 0
                    am_low    = ((position >> (q_idx - 1)) & 1) == 0
                    v = cmpex(v, partner_v, am_low == ascending)
                end
            end
        end
    end

    if in_range && (position + 1) <= K_ACTUAL
        A[position + 1, col_idx] = uint_unmap(T, v)
    end
end


# ── TAG kernel: K ≤ 32 (multi-column-per-warp packing, valid-tag) ─────

@kernel inbounds = true unsafe_indices = true function oem_sort_packed_tag_kernel!(
        A::AbstractMatrix{T},
        lt,
        ::Val{K_PAD},
        ::Val{K_ACTUAL},
) where {T, K_PAD, K_ACTUAL}
    @uniform begin
        cols_per_warp  = OEM_WARPSZ ÷ K_PAD
        cols_per_block = OEM_BLOCK   ÷ K_PAD
    end
    lid = Int(@index(Local))
    gid = Int(@index(Group))
    warp_id_in_block = (lid - 1) ÷ OEM_WARPSZ + 1
    lane_in_warp     = (lid - 1) % OEM_WARPSZ
    sub_warp_id      = lane_in_warp ÷ K_PAD
    sub_lane         = lane_in_warp % K_PAD

    M = size(A, 2)
    col_idx = (gid - 1) * cols_per_block +
              (warp_id_in_block - 1) * cols_per_warp +
              sub_warp_id + 1
    in_range = col_idx <= M
    is_valid_lane = in_range && (sub_lane + 1) <= K_ACTUAL

    # Clamped read for out-of-range lanes — `valid` flag masks it later.
    # Avoids `zero(T)`, which fails to compile for custom isbits T.
    sl_pos = is_valid_lane ? sub_lane + 1 : 1
    sl_col = is_valid_lane ? col_idx      : 1
    v      = A[sl_pos, sl_col]
    valid  = is_valid_lane

    if in_range
        @nexprs 5 lvl_p -> begin
            if (1 << lvl_p) <= K_PAD
                @nexprs 5 stg_q -> begin
                    if stg_q <= lvl_p
                        q_idx = lvl_p - stg_q + 1
                        d_off = 1 << (q_idx - 1)
                        partner_v     = @shfl(Xor, v, d_off)
                        partner_valid = @shfl(Xor, valid, d_off)
                        ascending = ((sub_lane >> lvl_p) & 1) == 0
                        am_low    = ((sub_lane >> (q_idx - 1)) & 1) == 0
                        if tag_swap_decision(am_low, ascending,
                                             v, valid,
                                             partner_v, partner_valid, lt)
                            v     = partner_v
                            valid = partner_valid
                        end
                    end
                end
            end
        end

        if is_valid_lane
            A[sub_lane + 1, col_idx] = v
        end
    end
end


# ── TAG kernel: K ≤ 64 (dualwarp + valid-tag) ─────────────────────────

@kernel inbounds = true unsafe_indices = true function oem_sort64_dualwarp_tag_kernel!(
        A::AbstractMatrix{T},
        lt,
        ::Val{K_ACTUAL},
) where {T, K_ACTUAL}
    lid = Int(@index(Local))
    gid = Int(@index(Group))
    warp_id_in_block = (lid - 1) ÷ OEM_WARPSZ + 1
    lane_in_warp     = (lid - 1) % OEM_WARPSZ + 1
    col_in_block     = (warp_id_in_block - 1) ÷ 2
    warp_in_col      = (warp_id_in_block - 1) % 2
    position         = warp_in_col * 32 + (lane_in_warp - 1)

    M = size(A, 2)
    col_idx = (gid - 1) * 2 + col_in_block + 1
    in_range = col_idx <= M
    is_valid_pos = in_range && (position + 1) <= K_ACTUAL

    shared_buf_v  = @localmem T    (128,)   # 2 cols × 64 positions
    shared_buf_va = @localmem Bool (128,)

    # Clamped read for out-of-range slots — `valid` flag masks it later.
    sp_pos = is_valid_pos ? position + 1 : 1
    sp_col = is_valid_pos ? col_idx      : 1
    v      = A[sp_pos, sp_col]
    valid  = is_valid_pos

    @nexprs 6 lvl_p -> begin
        @nexprs 6 stg_q -> begin
            if stg_q <= lvl_p
                q_idx = lvl_p - stg_q + 1
                d_off = 1 << (q_idx - 1)
                if d_off < 32
                    partner_v     = @shfl(Xor, v, d_off)
                    partner_valid = @shfl(Xor, valid, d_off)
                    ascending = ((position >> lvl_p) & 1) == 0
                    am_low    = ((position >> (q_idx - 1)) & 1) == 0
                    if tag_swap_decision(am_low, ascending,
                                         v, valid,
                                         partner_v, partner_valid, lt)
                        v     = partner_v
                        valid = partner_valid
                    end
                else
                    # Cross-warp exchange via shared.
                    sh_off = col_in_block * 64
                    shared_buf_v[sh_off + position + 1]  = v
                    shared_buf_va[sh_off + position + 1] = valid
                    @synchronize
                    partner_pos = position ⊻ 32
                    partner_v     = shared_buf_v[sh_off + partner_pos + 1]
                    partner_valid = shared_buf_va[sh_off + partner_pos + 1]
                    @synchronize
                    ascending = ((position >> lvl_p) & 1) == 0
                    am_low    = ((position >> (q_idx - 1)) & 1) == 0
                    if tag_swap_decision(am_low, ascending,
                                         v, valid,
                                         partner_v, partner_valid, lt)
                        v     = partner_v
                        valid = partner_valid
                    end
                end
            end
        end
    end

    if is_valid_pos
        A[position + 1, col_idx] = v
    end
end


# ── Driver ─────────────────────────────────────────────────────────────

"""
    oem_sort_columns_warp!(A; backend) -> A

Sort each column of `A` (a K×M matrix) in place using the warp-level
kernel. Requires K ≤ 64. For K > 64, use the shared-memory kernel.
"""
function oem_sort_columns_warp!(A::AbstractMatrix{T};
                                backend = get_backend(A),
                                lt = nothing,
                                k64_variant::Symbol = default_k64_variant(T)) where {T}
    K, M = size(A)
    @assert K <= 64 "warp kernel only handles K ≤ 64; got K=$K"

    # Next power of 2 ≥ K
    k_pad = 1
    while k_pad < K
        k_pad <<= 1
    end

    # Dispatch fast (typemax) vs tag (valid-bit) path.
    use_tag = (lt !== nothing) || !hasmethod(typemax, Tuple{Type{T}})
    lt_used = lt === nothing ? (<) : lt

    if K <= 32
        cols_per_block = OEM_BLOCK ÷ k_pad
        nblocks = cld(M, cols_per_block)
        if use_tag
            oem_sort_packed_tag_kernel!(backend, OEM_BLOCK)(
                A, lt_used, Val(k_pad), Val(K); ndrange = nblocks * OEM_BLOCK,
            )
        else
            oem_sort_packed_kernel!(backend, OEM_BLOCK)(
                A, Val(k_pad), Val(K); ndrange = nblocks * OEM_BLOCK,
            )
        end
    else
        # K = 33..64.
        if use_tag
            # Tag path: only dualwarp variant is implemented.
            nblocks = cld(M, 2)
            oem_sort64_dualwarp_tag_kernel!(backend, OEM_BLOCK)(
                A, lt_used, Val(K); ndrange = nblocks * OEM_BLOCK,
            )
        elseif k64_variant === :dualwarp
            nblocks = cld(M, 2)
            oem_sort64_dualwarp_kernel!(backend, OEM_BLOCK)(
                A, Val(K); ndrange = nblocks * OEM_BLOCK,
            )
        else
            nblocks = cld(M, OEM_COLS_PER_BLOCK)
            oem_sort64_kernel!(backend, OEM_BLOCK)(
                A, Val(K); ndrange = nblocks * OEM_BLOCK,
            )
        end
    end
    return A
end


# ── Smoke ──────────────────────────────────────────────────────────────

