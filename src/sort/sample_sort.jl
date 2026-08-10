# sample_sort.jl — host-side wrapper API for the iterative hierarchical
# GPU sample sort.  The `@kernel` definitions + device-side helpers +
# shared `const`s live in sample_sort_kernel.jl (included FIRST by
# KernelForge.jl, so its names are visible here).  This file holds the
# workspace struct, host accessors, per-arch default hooks, and the host
# driver functions.

# The leaf cap (max bucket handed to the batched bitonic leaf) is a per-arch TUNABLE
# knob — `default_leaf_max(arch, T)`, autotune-overridable like `default_nitem` — but
# HARD-BOUNDED by the static-`@localmem` ceiling: an L-element leaf needs L·sizeof(T) B
# of static LDS (+L B on the tag path) and the bitonic ladder tops at `Val(8192)`, so
# the only valid caps are {4096, 8192} for 4-byte and {4096} for 8-byte (8192·8 B =
# 64 KB blows the 48 KB static cap). `leaf_max_cap(T)` is that ceiling; every
# `default_leaf_max` method MUST stay within it and `sample_sort` asserts LEAF_MAX ≤
# leaf_max ≤ cap.
#
# Why a bigger 4-byte leaf helps: a bucket of 8192 (32 KB LDS, under the 48 KB static
# cap and the 64 KB gfx942 LDS) lets the partition loop STOP one level earlier. At
# N≈1e6 the mean post-level-0 bucket is ~3906 with an overflow tail ~5500 ≤ 8192 ⇒ ONE
# level suffices instead of two (a whole partition pass + ~23 launches + a 1.6 GB
# round-trip removed). At 1e8 the 8192 class never fires (post-level-1 buckets ~1526),
# so it is a pure small-N lever. The workspace stays sized on LEAF_MAX=4096 — an
# over-provision for any larger cap, never an under-provision (active buckets ≤
# cld(N, leaf_max) ≤ cld(N, 4096) = max_active), which is WHY the tuned cap must be
# ≥ LEAF_MAX.
@inline leaf_max_cap(::Type{T}) where {T} = sizeof(T) <= 4 ? 8192 : 4096
@inline default_leaf_max(::AbstractArch, ::Type{T}) where {T} = sizeof(T) <= 4 ? 8192 : 4096
# ════════════════════════════════════════════════════════════════════════
#  WORKSPACE PATH (O(N), allocated once) — active-only, variable expansion.
#
#  Removes the per-level allocation churn (measured: U64 1e7 wall 1152 ms
#  vs ~26 ms GPU). Key idea: each level only the ACTIVE buckets
#  (size>TERMINAL & !done & !const) are sampled/partitioned; FROZEN buckets
#  are identity-copied. Active buckets are partitioned WITHIN their own
#  [off[b],off[b+1]] range ⇒ per-parent-local offsets (no global child-size
#  scan). #active ≤ N/TERMINAL ⇒ every buffer is O(N) and preallocated once.
#  Comparator policy: partition + leaf both use the user's `lt`;
#  `reverse=true` is a single global pass applied to the final output.
# ════════════════════════════════════════════════════════════════════════

# Concretely typed: VT=vec-of-T, VU=vec-UInt32, VB=vec-Bool.  ::Any fields
# would make every `ws.field` access type-unstable ⇒ every kernel launch
# recompiles/dynamic-dispatches (GPUCompiler JIT inside the warm loop —
# the real cause of the slow 1e8 wall).  Concrete types ⇒ launches cache.
struct SampleSortWorkspace{T, VecT, VecU32, VecBool, VecU16, ScanTmp}
    N::Int
    max_active::Int
    max_buckets::Int
    buf_a::VecT; buf_b::VecT
    samples::VecT                           # pivots alias view(samples, 1:R, :)
    offsets::VecU32; new_offsets::VecU32
    done::VecBool; new_done::VecBool; const_mask::VecBool
    is_candidate::VecU32; candidate_prefix::VecU32; active_ids::VecU32
    is_active::VecU32; active_prefix::VecU32; child_count::VecU32; child_base::VecU32
    scan_out::VecU32                        # A4: active_scan aliased onto region_offsets
    histogram::VecU32                       # A2: also holds child_prefix (in-place)
    max_size::VecU32
    block_counts::VecU32; region_offsets::VecU32        # stable-scatter:
                                                # flat per-(parent,blk,s_local)
                                                # counts/prefix + region offsets
                                                # (region sizes aliased onto
                                                # child_count — A3)
    slot_cache::VecU16                          # O(N) UInt16 cached slot
                                                # (0=frozen): histogram→scatter
    bc_chunks::VecU32                           # per-(parent,chunk,slot) partials
                                                # for the parallel block-count scan
    scan_tmp::ScanTmp
end

function sample_sort_workspace(::Type{T}, N::Int; backend) where {T}
    max_active  = cld(N, LEAF_MAX)               # max #active ever (disjoint, >LEAF_MAX)
    max_buckets = 2 * FANOUT * max_active         # generous bound on total buckets
    al(S, n) = KernelAbstractions.allocate(backend, S, n)
    samples_len   = SAMPLES_PER_BUCKET * max_active
    histogram_len = FANOUT * max_active           # R counts per active parent
    buf_a = al(T, N);            buf_b = al(T, N)
    samples = al(T, samples_len)    # pivots = view(samples, 1:R, :) — A1 alias
    offsets = al(UInt32, max_buckets + 1); new_offsets = al(UInt32, max_buckets + 1)
    done = al(Bool, max_buckets);      new_done = al(Bool, max_buckets);     const_mask = al(Bool, max_active)
    is_candidate = al(UInt32, max_buckets);  candidate_prefix = al(UInt32, max_buckets); active_ids = al(UInt32, max_active)
    is_active = al(UInt32, max_buckets);   active_prefix = al(UInt32, max_buckets);  child_count = al(UInt32, max_buckets); child_base = al(UInt32, max_buckets)
    scan_out = al(UInt32, max_buckets)             # A4: active_scan aliased onto region_offsets
    histogram = al(UInt32, histogram_len)    # A2: also serves as child_prefix (in-place)
    max_size = al(UInt32, 1)
    num_tiles = cld(N, PARTITION_GROUP * ITEMS_PER_THREAD)        # #partition tiles
    block_counts_len = 2 * num_tiles * FANOUT     # Σ nblocks_p ≤ 2*num_tiles, ×FANOUT
    block_counts   = al(UInt32, block_counts_len)
    region_offsets = al(UInt32, max_buckets + 1)   # A3: region_sizes aliases child_count
    slot_cache = al(UInt16, N)
    # Parallel block-count scan partials. `_bc_scan_kt` only returns KT ≥ 2 while
    # B ≤ BC_SCAN_TARGET_WG, and KT ≤ cld(BC_SCAN_TARGET_WG, B) + 1 there, so the
    # workgroup count B*KT never exceeds 2*BC_SCAN_TARGET_WG. Sizing on that bound
    # keeps this a fixed ~1 MB regardless of N (KT==1 doesn't touch it at all).
    bc_chunks = al(UInt32, 2 * BC_SCAN_TARGET_WG * FANOUT)
    scan_tmp = get_allocation(Scan1D, identity, +, scan_out)
    SampleSortWorkspace{T, typeof(buf_a), typeof(offsets), typeof(done), typeof(slot_cache), typeof(scan_tmp)}(
        N, max_active, max_buckets,
        buf_a, buf_b, samples,
        offsets, new_offsets,
        done, new_done, const_mask,
        is_candidate, candidate_prefix, active_ids,
        is_active, active_prefix, child_count, child_base,
        scan_out,
        histogram,
        max_size,
        block_counts, region_offsets,
        slot_cache,
        bc_chunks,
        scan_tmp,
    )
end
@inline samples_matrix(ws::SampleSortWorkspace) = reshape(ws.samples, SAMPLES_PER_BUCKET, ws.max_active)
# A1: pivots alias `view(samples_matrix, 1:R, :)` (strided 2D view).
# Safe because pick_pivots_kernel! reads sorted samples then writes
# pivots only after a workgroup-wide @synchronize; samples is not
# read again that level after the pick.
@inline pivots_matrix(ws::SampleSortWorkspace) = view(samples_matrix(ws), 1:FANOUT, 1:ws.max_active)


function device_max_active_size(ws::SampleSortWorkspace, B::Int; backend)
    fill!(view(ws.max_size, 1:1), UInt32(0))
    wg = 256
    max_active_size_kernel!(backend, wg)(ws.max_size, ws.offsets, ws.done, B; ndrange = cld(B, wg) * wg)
    return Int(Array(view(ws.max_size, 1:1))[1])
end
# Driver: pick KT, then either the original single kernel (KT==1) or the 3 phases.
@inline function _block_count_scan!(ws, B::Int, mx::Int, backend)
    KT = _bc_scan_kt(B, mx)
    if KT == 1
        block_count_scan_kernel!(backend, FANOUT)(
            ws.block_counts, ws.offsets, ws.is_active, ws.region_offsets, B;
            ndrange = B * FANOUT)
    else
        nwg = B * KT
        block_count_reduce_kernel!(backend, FANOUT)(
            ws.bc_chunks, ws.block_counts, ws.offsets, ws.is_active, ws.region_offsets,
            B, KT; ndrange = nwg * FANOUT)
        block_count_chunk_scan_kernel!(backend, FANOUT)(
            ws.bc_chunks, B, KT; ndrange = B * FANOUT)
        block_count_apply_kernel!(backend, FANOUT)(
            ws.block_counts, ws.bc_chunks, ws.offsets, ws.is_active, ws.region_offsets,
            B, KT; ndrange = nwg * FANOUT)
    end
    return nothing
end
# ── driver loop (workspace, ping-pong, zero per-level alloc) ───────────

function partition_loop!(ws::SampleSortWorkspace{T}, src::AbstractVector{T};
                                 lt_used = (<),
                                 leaf_max_::Int = default_leaf_max(detect_arch(src), T),
                                 backend = get_backend(src)) where {T}
    N = ws.N
    # K2 sample sort: keep oem's fast typemax path for the default
    # comparator; use oem's tag path for a custom `lt`.
    lt_k2 = lt_used === (<) ? nothing : lt_used
    @assert length(src) == N
    cur = ws.buf_a; oth = ws.buf_b
    copyto!(view(cur, 1:N), src)
    copyto!(view(ws.offsets, 1:2), UInt32[0, N])
    fill!(view(ws.done, 1:1), false)
    B = 1; level = 0
    wg = 256
    sampm = samples_matrix(ws); pivm = pivots_matrix(ws)
    while level < MAX_LEVELS
        # The `mx = device_max_active_size(...); mx <= LEAF_MAX && break` that used to
        # stand here was REDUNDANT with the `A == 0 && break` below, and it cost a
        # kernel + a fill! + a full D2H drain EVERY level:
        #   mx > LEAF_MAX  ⟺  ∃b: !done[b] ∧ sz(b) > LEAF_MAX  ⟺  A > 0
        # because `max_active_size_kernel!` maxes sz over !done[b], and
        # `flag_candidates_kernel!` sets iscand[b] = (sz > LEAF_MAX && !done[b]) with
        # A = Σ iscand. The two predicates are literally the same set being tested.
        # (`mx_final` after the loop still feeds the leaf; only the per-level call is
        # gone.) `mx` was otherwise used only for `_bc_scan_kt(B, mx)` — see the N
        # substitution note at that call.
        flag_candidates_kernel!(backend, wg)(
            ws.is_candidate, ws.offsets, ws.done, B, leaf_max_; ndrange = cld(B, wg) * wg)
        scan!(identity, +, view(ws.candidate_prefix, 1:B), view(ws.is_candidate, 1:B);
                 tmp = ws.scan_tmp)
        A = Int(Array(view(ws.candidate_prefix, B:B))[1])
        A == 0 && break
        @assert A <= ws.max_active "active count $A exceeds Amax=$(ws.max_active)"

        compact_candidates_kernel!(backend, wg)(
            ws.active_ids, ws.is_candidate, ws.candidate_prefix, B; ndrange = cld(B, wg) * wg)
        sample_kernel!(backend, wg)(
            sampm, cur, ws.offsets, ws.active_ids, level, A;
            ndrange = cld(SAMPLES_PER_BUCKET * A, wg) * wg)
        oem_sort_columns!(view(sampm, :, 1:A); backend, lt = lt_k2)
        pick_pivots_kernel!(backend, wg)(
            pivm, ws.const_mask, sampm, A, Val(OVERSAMPLE);
            ndrange = cld(FANOUT * A, wg) * wg)

        finalize_active_kernel!(backend, wg)(
            ws.is_active, ws.child_count, ws.is_candidate, ws.candidate_prefix, ws.const_mask, B;
            ndrange = cld(B, wg) * wg)
        scan!(identity, +, view(ws.scan_out, 1:B), view(ws.child_count, 1:B);
                 tmp = ws.scan_tmp)
        scan!(identity, +, view(ws.region_offsets, 1:B), view(ws.is_active, 1:B);
                 tmp = ws.scan_tmp)
        newB = Int(Array(view(ws.scan_out, B:B))[1])
        @assert newB <= ws.max_buckets "newB=$newB exceeds MAXB=$(ws.max_buckets)"
        finalize_prefixes_kernel!(backend, wg)(
            ws.child_base, ws.active_prefix, ws.scan_out, ws.region_offsets, ws.child_count, ws.is_active, B;
            ndrange = cld(B, wg) * wg)

        # O-B: bc region offsets BEFORE localhist so the fused localhist
        # writes per-(parent,block,s_local) counts directly (no separate
        # bc_count binsearch pass).  scan_out is free after
        # finalize_prefixes (cb/actpref derived, newB already read).
        block_region_sizes_kernel!(backend, wg)(
            ws.child_count, ws.offsets, ws.is_active, B; ndrange = cld(B, wg) * wg)
        scan!(identity, +, view(ws.scan_out, 1:B), view(ws.child_count, 1:B);
                 tmp = ws.scan_tmp)
        block_region_offsets_kernel!(backend, wg)(
            ws.region_offsets, ws.scan_out, ws.child_count, B; ndrange = cld(B, wg) * wg)

        fill!(view(ws.histogram, 1:FANOUT * A), UInt32(0))
        nblk = cld(N, PARTITION_GROUP * ITEMS_PER_THREAD)
        partition_histogram_kernel!(backend, PARTITION_GROUP)(
            ws.histogram, ws.block_counts, ws.slot_cache, cur, ws.offsets, pivm, ws.is_active, ws.active_prefix,
            ws.child_base, ws.region_offsets, lt_used, B; ndrange = nblk * PARTITION_GROUP)
        child_prefix_kernel!(backend, FANOUT)(
            ws.histogram, A; ndrange = A * FANOUT)        # A2 in-place
        build_offsets_kernel!(backend, wg)(
            ws.new_offsets, ws.new_done, ws.offsets, ws.done, ws.is_candidate, ws.candidate_prefix,
            ws.const_mask, ws.is_active, ws.active_prefix, ws.child_base, ws.histogram, B, N, newB;
            ndrange = cld(B, wg) * wg)

        # N, not `mx`: substituting the array length here is BIT-IDENTICAL. _bc_scan_kt
        # returns clamp(min(cld(512,B), cld(mx,TILE)), 1, 128), and B is only ever 1 or
        # ≥256 (newB = FANOUT*A + (B-A) ≥ FANOUT once A ≥ 1). At B==1 the single bucket
        # IS the whole array, so mx == N. At B ≥ 256, cld(512,B) ≤ 2 while reaching this
        # line implies A > 0 ⟹ some active bucket exceeds LEAF_MAX ⟹ cld(mx,TILE) ≥ 2,
        # so the `min` picks cld(512,B) either way. Hence no per-level max is needed.
        _block_count_scan!(ws, B, N, backend)
        stable_scatter_kernel!(backend, PARTITION_GROUP)(
            oth, cur, ws.offsets, ws.is_active, ws.active_prefix, ws.child_base,
            ws.new_offsets, ws.block_counts, ws.region_offsets, ws.slot_cache, B; ndrange = nblk * PARTITION_GROUP)

        cur, oth = oth, cur
        copyto!(view(ws.offsets, 1:newB + 1), view(ws.new_offsets, 1:newB + 1))
        copyto!(view(ws.done, 1:newB), view(ws.new_done, 1:newB))
        B = newB
        level += 1
    end
    mx_final = device_max_active_size(ws, B; backend)
    return cur, oth, B, level, mx_final
end
# Sort every non-const bucket with size > TERMINAL (rare).  off is already
# on host-readable; loop the (few) oversized buckets, pow2-pad each into a
# fresh temp, run the multi-pass network, copy back in place.
function bigbucket_sort!(array::AbstractVector{T},
                              hoff::Vector{UInt32}, B::Int;
                              lt_used, padval::Union{Nothing,T},
                              use_tag::Bool,
                              leaf_max_::Int = default_leaf_max(detect_arch(array), T),
                              backend) where {T}
    nfb = 0
    lmax = leaf_max_
    @inbounds for b in 1:B
        n = Int(hoff[b + 1]) - Int(hoff[b])
        n > lmax || continue
        nfb += 1
        base = Int(hoff[b])                       # 0-based start
        P = 1
        while P < n
            P <<= 1
        end
        seg = view(array, base + 1 : base + n)
        if use_tag
            tmp = KernelAbstractions.allocate(backend, T, P)
            tva = KernelAbstractions.allocate(backend, Bool, P)
            copyto!(view(tmp, 1:n), seg)
            fill!(view(tva, 1:n), true)
            n < P && fill!(view(tva, n + 1 : P), false)
            wg = 256
            for lvl_p in 1:Int(log2(P))
                for stg_q in 1:lvl_p
                    q_idx = lvl_p - stg_q + 1
                    bigbucket_step_tag_kernel!(backend, wg)(
                        tmp, tva, lt_used, P, lvl_p, q_idx;
                        ndrange = cld(P >> 1, wg) * wg)
                end
            end
            copyto!(seg, view(tmp, 1:n))
        else
            pv = padval::T
            tmp = KernelAbstractions.allocate(backend, T, P)
            copyto!(view(tmp, 1:n), seg)
            n < P && fill!(view(tmp, n + 1 : P), pv)
            wg = 256
            for lvl_p in 1:Int(log2(P))
                for stg_q in 1:lvl_p
                    q_idx = lvl_p - stg_q + 1
                    bigbucket_step_kernel!(backend, wg)(
                        tmp, lt_used, P, lvl_p, q_idx;
                        ndrange = cld(P >> 1, wg) * wg)
                end
            end
            copyto!(seg, view(tmp, 1:n))
        end
    end
    return nfb
end

# Leaf routing.  `use_tag` (custom `lt` w/o `tmax`, or no-`typemax` T
# w/o `tmax`) uses the validity-Bool kernels (pad value irrelevant);
# otherwise the fast `(lt_used, padval)` kernels (padval = typemax(T)
# or the user sentinel).  Both strategies get the 2-tier speed:
# ≤BMAX → register Stage-C bitonic; 257..K_PAD → uniform shared kernel
# (minsz=BMAX so it skips the ones the reg kernel already did).
function leaf_sort!(array::AbstractVector{T}, off::AbstractVector{UInt32},
                    B::Int, max_bucket::Int, ws::SampleSortWorkspace{T};
                    lt_used = (<),
                    padval::Union{Nothing,T} = nothing,
                    use_tag::Bool = false,
                    leaf_max_::Int = default_leaf_max(detect_arch(array), T),
                    backend = get_backend(array)) where {T}
    # Fallback: sort any non-const bucket still > leaf cap (would overflow K_PAD).
    # Rare/adversarial.  After this every remaining bucket ≤ leaf_max_; the larger
    # ones are sorted in place and skipped below (bsize > K_PAD).
    lmax = leaf_max_
    if max_bucket > lmax
        hoff = Array(view(off, 1:B + 1))
        bigbucket_sort!(array, hoff, B;
                             lt_used, padval, use_tag, leaf_max_, backend)
        mx_small = 0
        @inbounds for b in 1:B
            n = Int(hoff[b + 1]) - Int(hoff[b])
            n <= lmax && n > mx_small && (mx_small = n)
        end
        max_bucket = mx_small
        max_bucket == 0 && return array        # nothing ≤ leaf cap left
    end

    @assert max_bucket <= lmax "unreachable: bigbucket fallback did not cap max bucket (max=$max_bucket)"

    # The ≤BMAX register Stage-C kernels use warp `@shfl`, which is only
    # safe for the standard numeric types (covers the default + any
    # numeric `lt`).  For an exotic bitstype @shfl is unsupported → route
    # ALL buckets through the uniform shared kernel (no shuffle, minsz=0).
    regok = reg_leaf_eligible(T)
    if regok
        wgf = 256
        leaf_oversize_flag_kernel!(backend, wgf)(
            ws.is_candidate, off, 0, REG_LEAF_MAX, B; ndrange = cld(B, wgf) * wgf)
        scan!(identity, +, view(ws.candidate_prefix, 1:B), view(ws.is_candidate, 1:B);
                 tmp = ws.scan_tmp)
        nsmall = Int(Array(view(ws.candidate_prefix, B:B))[1])
        if nsmall > 0
            compact_candidates_kernel!(backend, wgf)(
                ws.scan_out, ws.is_candidate, ws.candidate_prefix, B;
                ndrange = cld(B, wgf) * wgf)
        end
    else
        nsmall = 0
    end

    # ≤BMAX register Stage-C (numeric only).
    if nsmall > 0
        if use_tag
            leaf_register_tag_kernel!(backend, 64)(
                array, off, view(ws.scan_out, 1:nsmall), nsmall, lt_used;
                ndrange = nsmall * 64)
        else
            leaf_register_kernel!(backend, 64)(
                array, off, view(ws.scan_out, 1:nsmall), nsmall, lt_used,
                padval::T; ndrange = nsmall * 64)
        end
    end

    # W2: right-size K_PAD per bucket-size class.  A ~1300-elem bucket
    # padded to 2048 instead of the global-max 4096 ⇒ ½ shared mem
    # (≈2× occupancy) + fewer bitonic stages.  Same proven bitonic body
    # (`minsz<bsize≤K_PAD` selects exactly one disjoint class), one launch
    # per class.  `>4096` survivors were sorted by the fallback (skipped:
    # bsize>K_PAD).  `minsz` (=BMAX if regok else 0) is the first class's
    # lower bound so ≤BMAX is covered by the reg kernel (regok) or the
    # K=256 class (!regok).
    @inline function _leafclass!(lob::Int, ::Val{K}) where {K}
        lob >= max_bucket && return nothing
        wg = clamp(K ÷ 2, OEM_WARPSZ, 1024)
        if use_tag
            leaf_bitonic_tag_kernel!(backend, wg)(
                array, off, lt_used, lob, Val(K); ndrange = B * wg)
        else
            leaf_bitonic_kernel!(backend, wg)(
                array, off, lt_used, padval::T, lob, Val(K); ndrange = B * wg)
        end
        return nothing
    end
    regok || _leafclass!(0, Val(256))   # ≤256 (reg covers this if regok)
    _leafclass!(256,  Val(512))
    _leafclass!(512,  Val(1024))
    _leafclass!(1024, Val(2048))
    _leafclass!(2048, Val(4096))
    # 4-byte keys only: an 8192 bucket = 32 KB LDS (tag path +8 KB Bool = 40 KB),
    # both under the 48 KB A100 static cap. 8-byte keys never reach here (leaf_max
    # keeps them ≤ 4096). `@nexprs 13` in the bitonic kernels covers log2(8192)=13.
    leaf_max_ > 4096 && _leafclass!(4096, Val(8192))
    return array
end


# ── Public entry ───────────────────────────────────────────────────────

"""
    sample_sort(src; backend, lt=nothing, tmax=nothing, reverse=false) -> sorted

Iterative hierarchical sample sort.  No host sort, no recursion: partition
into globally-ordered buckets (K1 sample → K2 oem batched sort → K3 pivot
pick → K4 anchored partition) until every bucket ≤ TERMINAL, then one
batched per-bucket bitonic leaf (K6).

Sorts by an arbitrary strict-weak-order `lt` (default `isless`/`<`).  The
SAME comparator drives the K2 sample sort, the pivot binsearch, and the
leaf, so buckets are globally ordered under `lt` (not raw bit order) —
correct for any `lt` and any bitstype.  `reverse` is a global index mirror
applied after the ascending sort (composes with any `lt`).

Leaf padding strategy (`tmax` = an optional user max-sentinel, a value or
a `()`/`Type→value` function):

  * `lt===nothing && tmax===nothing && typemax(T)` exists → fast path,
    `typemax(T)` pad, `<` comparator (byte-identical to the legacy path).
  * `tmax` given → fast path, the sentinel pad, `lt` comparator.  Contract:
    `lt(x, sentinel)` must hold for every real `x`.
  * otherwise (custom `lt` w/o `tmax`, or no-`typemax` T w/o `tmax`) →
    validity-Bool tag leaf (no sentinel needed).  Note: for a custom `lt`,
    `typemax(T)` is NOT a valid sentinel unless `lt` agrees with bit order,
    so this case takes the tag path even when `typemax(T)` exists.

!!! warning "The return value ALIASES the workspace"
    `sorted` is a `view` into one of the workspace ping-pong buffers, NOT a
    fresh array.  When you pass your own `ws`, the next `sample_sort(...; ws)`
    call (or any mutation of `ws`'s buffers) CLOBBERS a previously-returned
    result.  If you need the sorted output to outlive the next reuse of `ws`,
    `copy(sorted)` (or `collect`) it first.  With the default (no `ws`) each
    call allocates its own workspace, so the view stays valid for the caller —
    but the workspace as a whole is retained by that view and only freed when
    the view is.
"""
function sample_sort(src::AbstractVector{T};
                         backend = get_backend(src),
                         lt = nothing,
                         tmax = nothing,
                         reverse::Bool = false,
                         leaf_max = nothing,
                         arch = nothing,
                         ws::Union{Nothing,SampleSortWorkspace{T}} = nothing) where {T}
    N = length(src)
    w = ws === nothing ? sample_sort_workspace(T, N; backend) : ws
    @assert w.N == N "workspace built for N=$(w.N), got N=$N"

    # Leaf cap: autotune-overridable per-arch (`default_leaf_max`), or a caller
    # override. HARD invariant — must sit in [LEAF_MAX, leaf_max_cap(T)] so it
    # (a) never under-provisions the LEAF_MAX-sized workspace and (b) never blows
    # the static-`@localmem` ceiling / overflows the bitonic ladder.
    arch_ = arch === nothing ? detect_arch(src) : arch
    leaf_max_ = something(leaf_max, default_leaf_max(arch_, T))
    @assert LEAF_MAX <= leaf_max_ <= leaf_max_cap(T) "leaf_max=$leaf_max_ out of " *
        "[$(LEAF_MAX), $(leaf_max_cap(T))] for $(T) (sizeof $(sizeof(T)))"

    lt_used = lt === nothing ? (<) : lt
    has_tmax = tmax !== nothing
    if has_tmax
        raw = tmax isa Function ?
            (hasmethod(tmax, Tuple{}) ? tmax() : tmax(T)) : tmax
        sentinel = convert(T, raw)
    end
    if lt === nothing && !has_tmax && hasmethod(typemax, Tuple{Type{T}})
        use_tag = false
        padval  = typemax(T)            # legacy fast path (byte-identical)
    elseif has_tmax
        use_tag = false
        padval  = sentinel              # user sentinel, any `lt`
    else
        use_tag = true
        padval  = nothing               # tag path: validity-Bool, no pad
    end

    cur, oth, B, _level, mx =
        partition_loop!(w, src; lt_used, leaf_max_, backend)
    if mx > 0
        leaf_sort!(view(cur, 1:N), view(w.offsets, 1:B + 1), B, mx, w;
                   lt_used, padval, use_tag, leaf_max_, backend)
    end
    if reverse
        wg = 256
        reverse_kernel!(backend, wg)(
            view(oth, 1:N), view(cur, 1:N); ndrange = cld(N, wg) * wg)
        return view(oth, 1:N)
    end
    return view(cur, 1:N)
end


# ── Smoke (M1 + M2: K1 → K2 → K3 at B=1) ───────────────────────────────

