# sample_sort.jl — iterative hierarchical GPU sample sort.
# Fully on-device: no recursion, no host sort, deterministic.
#
# Each partition pass leaves buckets GLOBALLY ORDERED, so the next pass
# only samples/sorts WITHIN each bucket — many independent ≤ LEAF_MAX
# sorts run in parallel via the batched oem_sort_columns!, never one
# large sort. Iterating (while-loop) instead of recursing and never
# materializing one big flat pivot array removes the host-sort bridge
# and the recursion overhead of earlier designs.
#
# Algorithm (FANOUT=256 / level, OVERSAMPLE=16, SAMPLES_PER_BUCKET=4096):
#   while max_bucket > LEAF_MAX (4096):
#       samples = SAMPLES_PER_BUCKET random elems / bucket    → Matrix(S,B)
#       oem_sort_columns!(samples)                            batched, GPU
#       pivots  = samples[OVERSAMPLE:OVERSAMPLE:S, :]          → (R,B) + const_mask
#       array, offsets, B = partition(array, offsets, pivots) anchored hist+scatter
#   leaf_sort(array, offsets)                                 per-bucket, off-indexed
#
# Key invariant: the loop only partitions buckets > LEAF_MAX=4096; the
# block tile is also 4096 ⇒ a contiguous tile spans ≤ 2 adjacent
# buckets (the anchored-partition invariant), so the histogram/scatter
# pair uses `offsets` as the coarse cumulative and the 256 per-bucket
# pivots as the sub-bucket splitters.
#
# Comparator policy: the partition binsearch + sample-sort + leaf all
# use the user's `lt` (default `<`), so sub-buckets are globally
# lt-ordered and the leaf sorts each by lt ⇒ output is exact-lt-sorted.
# `reverse=true` is applied as a single global pass at the very end.

using KernelAbstractions
using KernelAbstractions: @atomic
using KernelIntrinsics
import KernelIntrinsics as KI
using Base.Cartesian: @nexprs

# oem_shared.jl (oem_sort_columns! K2 + leaf bitonic) is included by
# KernelForge.jl *before* this file — re-including it here double-defined
# `cmpex` and broke precompilation.


# ── Constants ──────────────────────────────────────────────────────────

const FANOUT        = 256              # fan-out per level
const OVERSAMPLE     = 16               # oversample factor
const SAMPLES_PER_BUCKET        = FANOUT * OVERSAMPLE # = 4096 samples / bucket (≤ oem max)
const LEAF_MAX = 4096             # max bucket handed to the leaf (8-byte / default)
# 4-byte keys get a bigger leaf: a bucket of 8192 fits in 32 KB of LDS — under the
# 48 KB A100 static-@localmem cap and the 64 KB gfx942 LDS — so the partition loop
# can STOP one level earlier. At N≈1e6 the mean post-level-0 bucket is ~3906 and its
# overflow tail sits ~5500 ≤ 8192 ⇒ ONE level suffices instead of two (a whole
# partition pass + ~23 launches + a 1.6 GB round-trip removed). 8-byte keys keep
# 4096 (8192×8 B = 64 KB would blow the A100 static cap). At 1e8 the 8192 class never
# fires (post-level-1 buckets ~1526), so this is a pure small-N lever. The workspace
# stays sized on LEAF_MAX=4096 — an over-provision for the 8192 case, never an
# under-provision (active buckets ≤ cld(N, leaf_max) ≤ cld(N, 4096) = max_active).
@inline leaf_max(::Type{T}) where {T} = sizeof(T) <= 4 ? 8192 : 4096
const PARTITION_GROUP       = 256              # partition workgroup
const ITEMS_PER_THREAD    = 16               # items/thread → tile = 4096
const LOG2_FANOUT    = 8                # log2(R): per-item sub-bucket binsearch depth
const MAX_LEVELS = 6                # safety cap on partition levels


# ── Stateless hash (cross-arch, reproducible). ─────────────────────────

@inline function splitmix_hash(x0::UInt64)
    x = x0 * 0x9E3779B97F4A7C15
    x ⊻= x >> 30; x *= 0xBF58476D1CE4E5B9
    x ⊻= x >> 27; x *= 0x94D049BB133111EB
    x ⊻= x >> 31
    return x
end



# ════════════════════════════════════════════════════════════════════════
#  K4 — anchored partition.  Each bucket b (size > TERMINAL) is split into
#  R=256 sub-buckets by its own 256 quantile pivots piv[:,b].  Buckets stay
#  GLOBALLY ORDERED.  Invariant: every active bucket > TERMINAL = tile, so a
#  contiguous tile spans ≤2 adjacent buckets (bucket_lo, bucket_hi).
#  Bucket of an element is decided by POSITION (array is contiguous), the
#  sub-bucket by an 8-step binsearch of its value in piv[:,bucket].
#  Global sub-bucket index = (bucket-1)*R + sub.
# ════════════════════════════════════════════════════════════════════════

# Bucket containing 1-based position `pos`: largest k∈1..B with off[k] < pos.
@inline function bucket_of(off, B::Int, pos::Int)
    lo = 1
    hi = B + 1
    while hi - lo > 1
        m = (lo + hi) >> 1
        if off[m] < UInt32(pos)
            lo = m
        else
            hi = m
        end
    end
    return lo
end


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


# ── max size over non-done buckets → ws.max_size (one 4-byte D2H) ─────────

@kernel inbounds = true unsafe_indices = true function max_active_size_kernel!(
        maxsz::AbstractVector{UInt32},
        @Const(off::AbstractVector{UInt32}),
        @Const(dn::AbstractVector{Bool}),
        B::Int,
) where {}
    lid = Int(@index(Local)); gid = Int(@index(Group))
    g = (gid - 1) * Int(@groupsize()[1]) + lid
    if g <= B
        if !dn[g]
            sz = off[g + 1] - off[g]
            @atomic maxsz[1] max sz
        end
    end
end

function device_max_active_size(ws::SampleSortWorkspace, B::Int; backend)
    fill!(view(ws.max_size, 1:1), UInt32(0))
    wg = 256
    max_active_size_kernel!(backend, wg)(ws.max_size, ws.offsets, ws.done, B; ndrange = cld(B, wg) * wg)
    return Int(Array(view(ws.max_size, 1:1))[1])
end


# ── active-set construction ────────────────────────────────────────────

@kernel inbounds = true unsafe_indices = true function flag_candidates_kernel!(
        iscand::AbstractVector{UInt32},
        @Const(off::AbstractVector{UInt32}),
        @Const(dn::AbstractVector{Bool}),
        B::Int,
        leaf_max_::Int,                          # per-T leaf cap (8192 for 4-byte)
) where {}
    lid = Int(@index(Local)); gid = Int(@index(Group))
    b = (gid - 1) * Int(@groupsize()[1]) + lid
    if b <= B
        sz = Int(off[b + 1]) - Int(off[b])
        iscand[b] = (sz > leaf_max_ && !dn[b]) ? UInt32(1) : UInt32(0)
    end
end

# active_ids[candpref[b]] = b   for candidate b   (candpref = inclusive scan)
@kernel inbounds = true unsafe_indices = true function compact_candidates_kernel!(
        active_ids::AbstractVector{UInt32},
        @Const(iscand::AbstractVector{UInt32}),
        @Const(candpref::AbstractVector{UInt32}),
        B::Int,
) where {}
    lid = Int(@index(Local)); gid = Int(@index(Group))
    b = (gid - 1) * Int(@groupsize()[1]) + lid
    if b <= B && iscand[b] == UInt32(1)
        active_ids[candpref[b]] = UInt32(b)
    end
end

# samp[:,a] sampled from bucket active_ids[a]
@kernel inbounds = true unsafe_indices = true function sample_kernel!(
        samp::AbstractMatrix{T},
        @Const(array::AbstractVector{T}),
        @Const(off::AbstractVector{UInt32}),
        @Const(active_ids::AbstractVector{UInt32}),
        level::Int, A::Int,
) where {T}
    lid = Int(@index(Local)); gid = Int(@index(Group))
    i = (gid - 1) * Int(@groupsize()[1]) + lid
    S = size(samp, 1)
    if i <= S * A
        j = (i - 1) % S + 1
        a = (i - 1) ÷ S + 1
        b = Int(active_ids[a])
        base = off[b]; size_b = off[b + 1] - base
        if size_b > UInt32(0)
            seed = (UInt64(b) << 32) ⊻ UInt64(j) ⊻ (UInt64(level) << 52)
            idx  = Int(base) + Int(splitmix_hash(seed) % UInt64(size_b)) + 1
            samp[j, a] = array[idx]
        end
    end
end

@kernel inbounds = true unsafe_indices = true function pick_pivots_kernel!(
        piv::AbstractMatrix{T},
        cmask::AbstractVector{Bool},
        @Const(samp::AbstractMatrix{T}),
        A::Int, ::Val{OVER},
) where {T, OVER}
    lid = Int(@index(Local)); gid = Int(@index(Group))
    i = (gid - 1) * Int(@groupsize()[1]) + lid
    R = size(piv, 1); S = size(samp, 1)
    # A1: piv may alias view(samp, 1:R, :) — read into a register and
    # @synchronize before writing so the workgroup's reads of
    # samp[r*OVER, a] (rows 16..R*OVER) complete before any thread's
    # write to piv[r, a] (rows 1..R) corrupts those positions in samp.
    # Layout: one workgroup of size R per active column ⇒ workgroup-
    # wide barrier suffices (no cross-column race).
    inr = i <= R * A
    r = inr ? ((i - 1) % R + 1) : 1
    a = inr ? ((i - 1) ÷ R + 1) : 1
    # Clamped read (always valid index) — avoids `zero(T)` for custom isbits T.
    val = samp[r * OVER, a]
    cflag = (inr && r == 1) && (samp[1, a] == samp[S, a])
    @synchronize
    if inr
        piv[r, a] = val
        if r == 1
            cmask[a] = cflag
        end
    end
end

# isact[b]=1 iff candidate & not const; childcnt = active ? R : 1
@kernel inbounds = true unsafe_indices = true function finalize_active_kernel!(
        isact::AbstractVector{UInt32},
        childcnt::AbstractVector{UInt32},
        @Const(iscand::AbstractVector{UInt32}),
        @Const(candpref::AbstractVector{UInt32}),
        @Const(cmask::AbstractVector{Bool}),
        B::Int,
) where {}
    lid = Int(@index(Local)); gid = Int(@index(Group))
    b = (gid - 1) * Int(@groupsize()[1]) + lid
    if b <= B
        if iscand[b] == UInt32(1) && !cmask[candpref[b]]
            isact[b] = UInt32(1); childcnt[b] = UInt32(FANOUT)
        else
            isact[b] = UInt32(0); childcnt[b] = UInt32(1)
        end
    end
end

# cb = exclusive prefix of childcnt; actpref = exclusive prefix of isact.
# Inputs are the INCLUSIVE scans (cc_inc, act_inc).
@kernel inbounds = true unsafe_indices = true function finalize_prefixes_kernel!(
        cb::AbstractVector{UInt32},
        actpref::AbstractVector{UInt32},
        @Const(cc_inc::AbstractVector{UInt32}),
        @Const(act_inc::AbstractVector{UInt32}),
        @Const(childcnt::AbstractVector{UInt32}),
        @Const(isact::AbstractVector{UInt32}),
        B::Int,
) where {}
    lid = Int(@index(Local)); gid = Int(@index(Group))
    b = (gid - 1) * Int(@groupsize()[1]) + lid
    if b <= B
        cb[b]      = cc_inc[b]  - childcnt[b]
        actpref[b] = act_inc[b] - isact[b]
    end
end


# ── anchored active-only: classify a tile's ≤2 active buckets ──────────
# Active buckets are >TILE ⇒ at most {bucket_of(first), bucket_of(last)}.
# Shared layout: [1]blo [2]bhi [3]ob_lo1 [4]ob_hi [5]alo [6]ahi
#                [7]ar_lo [8]ar_hi [9]cb_lo [10]cb_hi [11]ob_lo [12]ob_hi1

@inline function load_anchor!(shared_anc, off, isact, actpref, cb, B, bf, bl)
    blo = bucket_of(off, B, bf)
    bhi = bucket_of(off, B, bl)
    shared_anc[1] = blo
    shared_anc[2] = bhi
    shared_anc[3] = Int(off[blo + 1])
    shared_anc[4] = Int(off[bhi])
    shared_anc[5] = Int(isact[blo])
    shared_anc[6] = Int(isact[bhi])
    shared_anc[7] = Int(actpref[blo]) + 1        # active rank (1-based)
    shared_anc[8] = Int(actpref[bhi]) + 1
    shared_anc[9]  = Int(cb[blo])
    shared_anc[10] = Int(cb[bhi])
    shared_anc[11] = Int(off[blo])
    shared_anc[12] = Int(off[bhi + 1])
    return nothing
end

# Fused K4a: builds the global per-active-child `hist` AND the
# per-(parent,block-row k,s_local) counts `bc` from the SAME
# shared_hist (one pass — no separate bc_count) AND caches each
# element's slot in `slotc` for the scatter.  Writes ALL R bc entries
# (incl 0) for the block's active parent(s) ⇒ no stale, no pre-zeroing.
@kernel inbounds = true unsafe_indices = true function partition_histogram_kernel!(
        hist::AbstractVector{UInt32},            # R*A, zeroed
        bc::AbstractVector{UInt32},              # stable: per-(p,k,sl)
        slotc::AbstractVector{UInt16},           # stable: cached slot/elem
        @Const(array::AbstractVector{T}),
        @Const(off::AbstractVector{UInt32}),
        @Const(piv::AbstractMatrix{T}),          # (R, A)
        @Const(isact::AbstractVector{UInt32}),
        @Const(actpref::AbstractVector{UInt32}),
        @Const(cb::AbstractVector{UInt32}),
        @Const(bcoff::AbstractVector{UInt32}),
        lt,
        B::Int,
) where {T}
    @uniform begin
        N = length(array); R = FANOUT; TILE = PARTITION_GROUP * ITEMS_PER_THREAD
    end
    lid = Int(@index(Local)); gid = Int(@index(Group))
    shared_piv  = @localmem T      (2 * FANOUT,)
    shared_hist = @localmem UInt32 (2 * FANOUT,)
    shared_anc  = @localmem Int    (12,)

    block_first = (gid - 1) * TILE + 1
    block_last  = min(gid * TILE, N)
    if lid == 1
        load_anchor!(shared_anc, off, isact, actpref, cb, B, block_first, block_last)
    end
    @nexprs 2 z -> begin
        s_z = (z - 1) * PARTITION_GROUP + lid
        if s_z <= 2 * R
            shared_hist[s_z] = UInt32(0)
        end
    end
    @synchronize

    blo = shared_anc[1]; bhi = shared_anc[2]
    ob_lo1 = shared_anc[3]; ob_hi = shared_anc[4]
    alo = shared_anc[5] != 0; ahi = shared_anc[6] != 0
    ar_lo = shared_anc[7]; ar_hi = shared_anc[8]
    ob_lo = shared_anc[11]
    g0lo = ob_lo ÷ TILE + 1
    g0hi = ob_hi ÷ TILE + 1
    klo = gid - g0lo
    khi = gid - g0hi
    bo_lo = alo ? Int(bcoff[blo]) : 0
    bo_hi = ahi ? Int(bcoff[bhi]) : 0

    k = lid
    while k <= R
        # Clamped read so the load compiles for types without `zero`;
        # pivots at non-active sides are masked downstream by in_lo/in_hi.
        shared_piv[k]     = piv[k, alo ? ar_lo : 1]
        shared_piv[R + k] = piv[k, ahi ? ar_hi : 1]
        k += PARTITION_GROUP
    end
    @synchronize

    # NOTE — a 16-wide MLP split of this loop (all 16 loads → all 16 binsearches →
    # all 16 atomics, the restructure that paid twice on the radix onesweep in
    # 817851b/99591b0) was tried and REGRESSED: MI300A 1e8 F32 21409→23417 µs, F64
    # 26227→28599 µs (−9%), while small N was unaffected. 16 live values of T plus
    # 16 pos/slot/predicate sets is past this kernel's register budget — the same
    # cliff the radix hit at 24 items (2.5× slower) and avoided at 8. Do not re-try
    # 16-wide; an 8-wide split is the only shape worth re-testing here.
    @nexprs 16 j -> begin
        pos_j = block_first + (j - 1) * PARTITION_GROUP + lid - 1
        if pos_j <= block_last
            in_lo = alo && (pos_j <= ob_lo1)
            in_hi = (!in_lo) && ahi && (pos_j > ob_hi)
            if in_lo || in_hi
                v_j = array[pos_j]
                base_j = in_hi ? R : 0
                lo_j = 0; hi_j = R
                @nexprs 8 st -> begin
                    m_j  = (lo_j + hi_j) >> 1
                    pk_j = shared_piv[base_j + m_j]
                    if lt(v_j, pk_j)
                        hi_j = m_j
                    else
                        lo_j = m_j
                    end
                end
                slot_j = base_j + lo_j + 1
                @atomic shared_hist[slot_j] += UInt32(1)
                slotc[pos_j] = UInt16(slot_j)            # cache for scatter
            else
                slotc[pos_j] = UInt16(0)                 # frozen marker
            end
        end
    end
    @synchronize

    s = lid
    while s <= R
        if alo
            v = shared_hist[s]
            if v != UInt32(0)
                @atomic hist[(ar_lo - 1) * R + s] += v
            end
            bc[bo_lo + klo * R + (s - 1) + 1] = v
        end
        if ahi && (blo != bhi)
            v2 = shared_hist[R + s]
            if v2 != UInt32(0)
                @atomic hist[(ar_hi - 1) * R + s] += v2
            end
            bc[bo_hi + khi * R + (s - 1) + 1] = v2
        end
        s += PARTITION_GROUP
    end
end

# per active column: exclusive prefix of its R hist bins → locpre
@kernel inbounds = true unsafe_indices = true function child_prefix_kernel!(
        hist::AbstractVector{UInt32},                    # in/out (A2 in-place)
        A::Int,
)
    @uniform R = FANOUT
    lid = Int(@index(Local)); gid = Int(@index(Group))   # gid = active col
    sh = @localmem UInt32 (FANOUT,)
    if gid <= A
        base = (gid - 1) * R
        # A2: register-save the input so we can write the exclusive
        # prefix back into the SAME hist slot without losing the value
        # we still need for the inclusive→exclusive conversion.
        mine = hist[base + lid]
        sh[lid] = mine
        @synchronize
        off = 1
        while off < R
            other = (lid > off) ? sh[lid - off] : UInt32(0)
            @synchronize
            sh[lid] += other
            @synchronize
            off <<= 1
        end
        hist[base + lid] = sh[lid] - mine                # inclusive → exclusive
    end
end

# build new_off (child starts, 0-based) + new_dn for every parent b.
@kernel inbounds = true unsafe_indices = true function build_offsets_kernel!(
        new_off::AbstractVector{UInt32},
        new_dn::AbstractVector{Bool},
        @Const(off::AbstractVector{UInt32}),
        @Const(dn::AbstractVector{Bool}),
        @Const(iscand::AbstractVector{UInt32}),
        @Const(candpref::AbstractVector{UInt32}),
        @Const(cmask::AbstractVector{Bool}),
        @Const(isact::AbstractVector{UInt32}),
        @Const(actpref::AbstractVector{UInt32}),
        @Const(cb::AbstractVector{UInt32}),
        @Const(locpre::AbstractVector{UInt32}),
        B::Int, N::Int, newB::Int,
) where {}
    @uniform R = FANOUT
    lid = Int(@index(Local)); gid = Int(@index(Group))
    b = (gid - 1) * Int(@groupsize()[1]) + lid
    if b <= B
        g0 = Int(cb[b])
        ob = off[b]
        if isact[b] == UInt32(1)
            ar = Int(actpref[b]) + 1
            @inbounds for s in 1:R
                new_off[g0 + s] = ob + locpre[(ar - 1) * R + s]
                new_dn[g0 + s]  = false
            end
        else
            new_off[g0 + 1] = ob
            isdone = dn[b] || (iscand[b] == UInt32(1) && cmask[candpref[b]])
            new_dn[g0 + 1] = isdone
        end
        if b == B
            new_off[newB + 1] = UInt32(N)
        end
    end
end

# ════════════════════════════════════════════════════════════════════════
#  Deterministic stable counting-sort scatter.
#  See xp/sample/stable_scatter_DESIGN.md.  P0 region sizes → P1 per-(parent,
#  block,s_local) counts → P2 parent-segmented exclusive scan (=crossblk
#  base) → P3 stable scatter (= new_off[c] + crossblk + intra-block
#  stable rank).  TILE = PARTITION_GROUP*ITEMS_PER_THREAD.  k(g,p) = g - g0(p),
#  g0(p) = off[p]÷TILE + 1.  bc Julia idx = bcoff[p] + k*R + (sl-1) + 1.
# ════════════════════════════════════════════════════════════════════════

# P0: regsz[p] = active(p) ? nblocks_p*R : 0   (nblocks_p over off).
@kernel inbounds = true unsafe_indices = true function block_region_sizes_kernel!(
        bcreg::AbstractVector{UInt32},
        @Const(off::AbstractVector{UInt32}),
        @Const(isact::AbstractVector{UInt32}),
        B::Int,
) where {}
    @uniform TILE = PARTITION_GROUP * ITEMS_PER_THREAD
    lid = Int(@index(Local)); gid = Int(@index(Group))
    p = (gid - 1) * Int(@groupsize()[1]) + lid
    if p <= B
        if isact[p] == UInt32(1)
            o0 = Int(off[p]); o1 = Int(off[p + 1])
            g0 = o0 ÷ TILE + 1
            g1 = (o1 - 1) ÷ TILE + 1
            bcreg[p] = UInt32((g1 - g0 + 1) * FANOUT)
        else
            bcreg[p] = UInt32(0)
        end
    end
end

# bcoff = EXCLUSIVE prefix of regsz (bc_inc is the inclusive scan).
@kernel inbounds = true unsafe_indices = true function block_region_offsets_kernel!(
        bcoff::AbstractVector{UInt32},
        @Const(bc_inc::AbstractVector{UInt32}),
        @Const(bcreg::AbstractVector{UInt32}),
        B::Int,
) where {}
    lid = Int(@index(Local)); gid = Int(@index(Group))
    p = (gid - 1) * Int(@groupsize()[1]) + lid
    if p <= B
        bcoff[p] = bc_inc[p] - bcreg[p]
    end
end

# P2: parent-segmented EXCLUSIVE scan of bc down the block-row dim.
# One launch-block per parent p (workgroup = R; lid = s_local column).
@kernel inbounds = true unsafe_indices = true function block_count_scan_kernel!(
        bc::AbstractVector{UInt32},
        @Const(off::AbstractVector{UInt32}),
        @Const(isact::AbstractVector{UInt32}),
        @Const(bcoff::AbstractVector{UInt32}),
        B::Int,
) where {}
    @uniform begin
        R = FANOUT; TILE = PARTITION_GROUP * ITEMS_PER_THREAD
    end
    lid = Int(@index(Local)); p = Int(@index(Group))
    if p <= B && isact[p] == UInt32(1)
        o0 = Int(off[p]); o1 = Int(off[p + 1])
        g0 = o0 ÷ TILE + 1
        g1 = (o1 - 1) ÷ TILE + 1
        n  = g1 - g0 + 1
        base = Int(bcoff[p])
        run = UInt32(0)
        kk = 0
        while kk < n
            idx = base + kk * R + (lid - 1) + 1
            x = bc[idx]
            bc[idx] = run
            run += x
            kk += 1
        end
    end
end

# --- Parallel block-count scan (reduce → scan-chunks → apply) ----------------
# `block_count_scan_kernel!` above launches ONE workgroup per parent, and each
# workgroup serially walks that parent's n block-rows carrying `run`. At level 0
# there is exactly ONE parent (B=1), so a single workgroup walks n = cld(N, TILE)
# rows — 24414 dependent iterations at N=1e8, measured at 8.68 ms for that single
# call and 25.8% of the whole sample_sort (rocprofv3, MI300A, 2026-07-17). All the
# parallelism lives in the block-row (k) dimension; splitting it is the same fix
# shape as the histogram-combine launch bug ([[project_sort_mi300a_combine_parallelize]]),
# in its scan form: reduce each k-slab, exclusive-scan the slab totals, re-apply.
#
# KT (chunks per parent) is ADAPTIVE and host-computed from B and `mx` (the max
# active bucket size, already known host-side each level): KT ≈ 512/B, capped by
# how many rows a parent can even have. So level 0 (B=1) gets 128 chunks, mid
# levels get 2, and once B is large enough on its own KT collapses to 1 and we
# take the ORIGINAL single-kernel path — zero extra launches, zero extra buffer.
const BC_SCAN_TARGET_WG = 512     # workgroups we want the block-count scan to fill
const BC_SCAN_KT_MAX    = 128     # cap on chunks per parent (bounds `bc_chunks`)

# Chunks per parent. `mx` = max active bucket size (host-known). Returns 1 when the
# plain kernel already has enough parallelism, which routes to the original path.
@inline function _bc_scan_kt(B::Int, mx::Int)
    TILE = PARTITION_GROUP * ITEMS_PER_THREAD
    nmax = cld(max(mx, 1), TILE)               # no parent has more rows than this
    kt   = cld(BC_SCAN_TARGET_WG, max(B, 1))
    return clamp(min(kt, nmax), 1, BC_SCAN_KT_MAX)
end

# Half-open row range [k0, k1) of chunk `c`. MUST be identical in reduce and apply
# or the seeds won't line up with the rows they seed.
@inline function _bc_chunk_range(n::Int, KT::Int, c::Int)
    per = cld(n, KT)
    return (c - 1) * per, min(c * per, n)
end

# Row count / base for parent p (shared by both kernels; mirrors the plain kernel).
@inline function _bc_parent_rows(off, bcoff, p::Int)
    TILE = PARTITION_GROUP * ITEMS_PER_THREAD
    o0 = Int(off[p]); o1 = Int(off[p + 1])
    g0 = o0 ÷ TILE + 1
    g1 = (o1 - 1) ÷ TILE + 1
    return g1 - g0 + 1, Int(bcoff[p])
end

# Phase 1: per-(parent, chunk, slot) sum of that chunk's rows → csum.
@kernel inbounds = true unsafe_indices = true function block_count_reduce_kernel!(
        csum::AbstractVector{UInt32},
        @Const(bc::AbstractVector{UInt32}),
        @Const(off::AbstractVector{UInt32}),
        @Const(isact::AbstractVector{UInt32}),
        @Const(bcoff::AbstractVector{UInt32}),
        B::Int, KT::Int,
) where {}
    @uniform R = FANOUT
    lid = Int(@index(Local)); w = Int(@index(Group))
    p = (w - 1) ÷ KT + 1
    c = (w - 1) % KT + 1
    acc = UInt32(0)
    if p <= B && isact[p] == UInt32(1)
        n, base = _bc_parent_rows(off, bcoff, p)
        k0, k1 = _bc_chunk_range(n, KT, c)
        kk = k0
        while kk < k1
            acc += bc[base + kk * R + (lid - 1) + 1]
            kk += 1
        end
    end
    csum[(w - 1) * R + lid] = acc          # always written (inactive ⇒ 0), no fill!
end

# Phase 2: exclusive scan of the KT chunk totals, per (parent, slot). One workgroup
# per parent, KT (≤128) iterations — the serial walk that used to be n rows long.
@kernel inbounds = true unsafe_indices = true function block_count_chunk_scan_kernel!(
        csum::AbstractVector{UInt32}, B::Int, KT::Int,
) where {}
    @uniform R = FANOUT
    lid = Int(@index(Local)); p = Int(@index(Group))
    if p <= B
        run = UInt32(0)
        c = 1
        while c <= KT
            i = ((p - 1) * KT + (c - 1)) * R + lid
            x = csum[i]
            csum[i] = run
            run += x
            c += 1
        end
    end
end

# Phase 3: local exclusive scan of each chunk's rows, seeded by its scanned total.
@kernel inbounds = true unsafe_indices = true function block_count_apply_kernel!(
        bc::AbstractVector{UInt32},
        @Const(csum::AbstractVector{UInt32}),
        @Const(off::AbstractVector{UInt32}),
        @Const(isact::AbstractVector{UInt32}),
        @Const(bcoff::AbstractVector{UInt32}),
        B::Int, KT::Int,
) where {}
    @uniform R = FANOUT
    lid = Int(@index(Local)); w = Int(@index(Group))
    p = (w - 1) ÷ KT + 1
    c = (w - 1) % KT + 1
    if p <= B && isact[p] == UInt32(1)
        n, base = _bc_parent_rows(off, bcoff, p)
        k0, k1 = _bc_chunk_range(n, KT, c)
        run = csum[(w - 1) * R + lid]
        kk = k0
        while kk < k1
            idx = base + kk * R + (lid - 1) + 1
            x = bc[idx]
            bc[idx] = run
            run += x
            kk += 1
        end
    end
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


# P3: deterministic stable scatter.  dst_pos = new_off[c] +
# crossblk(bc) + intra-block stable rank (rank in tile-position order
# among same-slot elements of THIS block).  Intra-block rank via
# NITEM sequential phases: each phase a deterministic triangular
# same-slot lower-lane count + a per-slot cursor carried across
# phases (cursor updated by atomic COUNT — value order-independent ⇒
# deterministic).  Correct-first (O(WG) per element); ballot-match
# optimisation deferred until the pipeline is gated.
# O-A slot-cache: scatter reads the slot cached by the fused localhist
# (same tile/anchor ⇒ same classification) instead of re-loading pivots
# and re-running the 8-step binsearch.  Drops shared_piv + its load loop
# + one @synchronize + the per-element binsearch.  Determinism-critical
# rank machinery below is UNCHANGED.
@kernel inbounds = true unsafe_indices = true function stable_scatter_kernel!(
        dst::AbstractVector{T},
        @Const(array::AbstractVector{T}),
        @Const(off::AbstractVector{UInt32}),
        @Const(isact::AbstractVector{UInt32}),
        @Const(actpref::AbstractVector{UInt32}),
        @Const(cb::AbstractVector{UInt32}),
        @Const(new_off::AbstractVector{UInt32}),     # child base (0-based)
        @Const(bc::AbstractVector{UInt32}),          # post-P2 crossblk prefix
        @Const(bcoff::AbstractVector{UInt32}),
        @Const(slotc::AbstractVector{UInt16}),       # cached slot (0=frozen)
        B::Int,
) where {T}
    @uniform begin
        N = length(array); R = FANOUT; TILE = PARTITION_GROUP * ITEMS_PER_THREAD
    end
    lid = Int(@index(Local)); gid = Int(@index(Group))
    shared_anc = @localmem Int (12,)
    cursor     = @localmem UInt32 (2 * FANOUT,)
    ls         = @localmem UInt32 (PARTITION_GROUP,)
    wcnt       = @localmem UInt32 ((PARTITION_GROUP ÷ 32) * 2 * FANOUT,)  # [warp,slot]

    block_first = (gid - 1) * TILE + 1
    block_last  = min(gid * TILE, N)
    if lid == 1
        load_anchor!(shared_anc, off, isact, actpref, cb, B, block_first, block_last)
    end
    @nexprs 2 z -> begin
        s_z = (z - 1) * PARTITION_GROUP + lid
        if s_z <= 2 * R
            cursor[s_z] = UInt32(0)
        end
    end
    @synchronize

    blo = shared_anc[1]; bhi = shared_anc[2]
    ob_hi = shared_anc[4]
    alo = shared_anc[5] != 0; ahi = shared_anc[6] != 0
    cb_lo = shared_anc[9]; cb_hi = shared_anc[10]
    ob_lo = shared_anc[11]

    g0lo = ob_lo ÷ TILE + 1
    g0hi = ob_hi ÷ TILE + 1
    bo_lo = alo ? Int(bcoff[blo]) : 0
    bo_hi = ahi ? Int(bcoff[bhi]) : 0
    klo = gid - g0lo
    khi = gid - g0hi

    j = 1
    while j <= ITEMS_PER_THREAD
        pos_j = block_first + (j - 1) * PARTITION_GROUP + lid - 1
        inr_j = pos_j <= block_last
        v_j   = array[inr_j ? pos_j : 1]   # clamped read; used only if act_j
        slot_j = inr_j ? Int(slotc[pos_j]) : 0       # cached; 0 = frozen
        act_j = slot_j != 0
        ls[lid] = UInt32(slot_j)                       # 0 = frozen/none
        # deterministic same-slot lower-position rank this phase, warp-
        # decomposed: per-(warp,slot) count via atomic (COUNT only ⇒
        # value order-independent ⇒ deterministic); intra-warp lower-
        # lane via the (deterministic, lid-written) ls scan (≤WS iters);
        # cross-warp = Σ earlier warps' counts.
        WS = Int(@warpsize())
        nwarp = PARTITION_GROUP ÷ WS
        warpid = (lid - 1) ÷ WS                          # 0-based
        zz = lid
        while zz <= nwarp * 2 * R
            wcnt[zz] = UInt32(0)
            zz += PARTITION_GROUP
        end
        @synchronize
        if act_j
            @atomic wcnt[warpid * 2 * R + slot_j] += UInt32(1)
        end
        @synchronize
        if act_j
            wfirst = warpid * WS + 1                      # 1-based
            dw = UInt32(0)
            l = wfirst
            while l < lid
                if ls[l] == UInt32(slot_j)
                    dw += UInt32(1)
                end
                l += 1
            end
            dc = UInt32(0)
            w2 = 0
            while w2 < warpid
                dc += wcnt[w2 * 2 * R + slot_j]
                w2 += 1
            end
            d = dc + dw
            cur = cursor[slot_j]
            if slot_j <= R
                p_local = slot_j
                c = cb_lo + slot_j
                cross = bc[bo_lo + klo * R + (p_local - 1) + 1]
            else
                p_local = slot_j - R
                c = cb_hi + p_local
                cross = bc[bo_hi + khi * R + (p_local - 1) + 1]
            end
            dst[Int(new_off[c]) + Int(cross) + Int(cur) + Int(d) + 1] = v_j
        elseif inr_j
            dst[pos_j] = v_j
        end
        @synchronize
        # carry per-slot cursor += Σ_w wcnt[w,slot]  (deterministic sum)
        zz2 = lid
        while zz2 <= 2 * R
            tot = UInt32(0)
            w3 = 0
            while w3 < nwarp
                tot += wcnt[w3 * 2 * R + zz2]
                w3 += 1
            end
            cursor[zz2] += tot
            zz2 += PARTITION_GROUP
        end
        @synchronize
        j += 1
    end
end


# ── driver loop (workspace, ping-pong, zero per-level alloc) ───────────

function partition_loop!(ws::SampleSortWorkspace{T}, src::AbstractVector{T};
                                 lt_used = (<),
                                 dbg::Union{Nothing,Vector} = nothing,
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
        # substitution note at that call — and for the debug hook, which now pays for
        # its own value so production doesn't.
        mx_dbg = dbg === nothing ? 0 : device_max_active_size(ws, B; backend)

        flag_candidates_kernel!(backend, wg)(
            ws.is_candidate, ws.offsets, ws.done, B, leaf_max(T); ndrange = cld(B, wg) * wg)
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
        if dbg !== nothing
            push!(dbg, (level = level, A = A, B = B, mx = mx_dbg,
                        coff = sum(UInt64.(view(ws.offsets, 1:B + 1))),
                        csamp = sum(UInt64.(view(ws.samples, 1:SAMPLES_PER_BUCKET * A))),
                        cpiv = sum(UInt64.(vec(view(samples_matrix(ws), 1:FANOUT, 1:A)))),
                        chist = sum(UInt64.(view(ws.histogram, 1:FANOUT * A))),
                        ccur = sum(UInt64.(view(cur, 1:N)))))
        end
    end
    mx_final = device_max_active_size(ws, B; backend)
    return cur, oth, B, level, mx_final
end


# ════════════════════════════════════════════════════════════════════════
#  K6 — leaf sort.  Adapted from oem_sort_shared_kernel! (xp/sample/oem/
#  oem_shared.jl): one block per bucket, load array[off[b]+1 : off[b+1]]
#  into shared, pad to K_PAD=next_pow2(max_bucket)≤4096 with typemax,
#  bitonic network, write back in place.  REVERSE = Option-2 mirror
#  (reversed write-back index).
# ════════════════════════════════════════════════════════════════════════

# Leaf sorts each bucket ASCENDING in place (read + write the SAME bucket
# region ⇒ race-free per block).  `reverse` is applied afterwards by a
# single global-reverse kernel on the fully-sorted array (no cross-bucket
# writes here).  Sort EVERY bucket ≤ K_PAD (don't trust the sampled const
# flag here: it false-positives on near-constant buckets; skipping leaves
# them unsorted).  Only > K_PAD buckets are skipped — excluded from the
# K_PAD max via `dn`, so genuinely (near-)constant ⇒ already sorted.
# `active` is uniform per block (one bucket/block) ⇒ wrapping the body
# (incl. @synchronize) is barrier-safe.

@kernel inbounds = true unsafe_indices = true function leaf_bitonic_kernel!(
        array::AbstractVector{T},
        @Const(off::AbstractVector{UInt32}),     # length B+1
        lt,
        padval::T,
        minsz::Int,                              # skip buckets ≤ minsz (reg kernel did them)
        ::Val{K_PAD},
) where {T, K_PAD}
    @uniform workgroup = Int(@groupsize()[1])
    lid = Int(@index(Local))
    gid = Int(@index(Group))

    bstart = Int(off[gid]) + 1
    bsize  = Int(off[gid + 1]) - Int(off[gid])
    active = (bsize > minsz) & (bsize <= K_PAD)

    shared_buf = @localmem T (K_PAD,)

    if active
        @nexprs 8 c -> begin
            pos_c = (c - 1) * workgroup + lid
            if pos_c <= K_PAD
                shared_buf[pos_c] = (pos_c <= bsize) ?
                    array[bstart + pos_c - 1] : padval
            end
        end
        @synchronize

        @nexprs 13 lvl_p -> begin
            if (1 << lvl_p) <= K_PAD
                @nexprs 13 stg_q -> begin
                    if stg_q <= lvl_p
                        q_idx = lvl_p - stg_q + 1
                        d_off = 1 << (q_idx - 1)
                        @nexprs 8 pslot -> begin
                            t_idx = (pslot - 1) * workgroup + lid - 1
                            if t_idx < (K_PAD >> 1)
                                mask_b = d_off - 1
                                i_b = ((t_idx & ~mask_b) << 1) | (t_idx & mask_b)
                                partner_b = i_b + d_off
                                asc_b = ((i_b >> lvl_p) & 1) == 0
                                a_b = shared_buf[i_b + 1]
                                b_b = shared_buf[partner_b + 1]
                                swap_b = asc_b ⊻ lt(a_b, b_b)
                                if swap_b
                                    shared_buf[i_b + 1]       = b_b
                                    shared_buf[partner_b + 1] = a_b
                                end
                            end
                        end
                        @synchronize
                    end
                end
            end
        end

        @nexprs 8 c -> begin
            pos_c = (c - 1) * workgroup + lid
            if pos_c <= bsize
                array[bstart + pos_c - 1] = shared_buf[pos_c]
            end
        end
    end
end

# Tag-Bool fallback (custom `lt` or T has no `typemax`) — mirrors
# oem_sort_shared_tag_kernel!.  Invalid pad slots sort to the end via
# (a_va & !b_va); comparator is `lt`.  In-place ascending (race-free).
@kernel inbounds = true unsafe_indices = true function leaf_bitonic_tag_kernel!(
        array::AbstractVector{T},
        @Const(off::AbstractVector{UInt32}),
        lt,
        minsz::Int,
        ::Val{K_PAD},
) where {T, K_PAD}
    @uniform workgroup = Int(@groupsize()[1])
    lid = Int(@index(Local))
    gid = Int(@index(Group))

    bstart = Int(off[gid]) + 1
    bsize  = Int(off[gid + 1]) - Int(off[gid])
    active = (bsize > minsz) & (bsize <= K_PAD)

    shared_v  = @localmem T    (K_PAD,)
    shared_va = @localmem Bool (K_PAD,)

    if active
        @nexprs 8 c -> begin
            pos_c = (c - 1) * workgroup + lid
            if pos_c <= K_PAD
                ok = pos_c <= bsize
                # Clamped read for out-of-range slots — masked by va flag.
                shared_v[pos_c]  = array[bstart + (ok ? pos_c : 1) - 1]
                shared_va[pos_c] = ok
            end
        end
        @synchronize

        @nexprs 13 lvl_p -> begin
            if (1 << lvl_p) <= K_PAD
                @nexprs 13 stg_q -> begin
                    if stg_q <= lvl_p
                        q_idx = lvl_p - stg_q + 1
                        d_off = 1 << (q_idx - 1)
                        @nexprs 8 pslot -> begin
                            t_idx = (pslot - 1) * workgroup + lid - 1
                            if t_idx < (K_PAD >> 1)
                                mask_b = d_off - 1
                                i_b = ((t_idx & ~mask_b) << 1) | (t_idx & mask_b)
                                partner_b = i_b + d_off
                                asc_b = ((i_b >> lvl_p) & 1) == 0
                                a_v  = shared_v[i_b + 1]
                                a_va = shared_va[i_b + 1]
                                b_v  = shared_v[partner_b + 1]
                                b_va = shared_va[partner_b + 1]
                                is_less = (a_va & !b_va) |
                                          (a_va & b_va & lt(a_v, b_v))
                                if asc_b ⊻ is_less
                                    shared_v[i_b + 1]        = b_v
                                    shared_va[i_b + 1]       = b_va
                                    shared_v[partner_b + 1]  = a_v
                                    shared_va[partner_b + 1] = a_va
                                end
                            end
                        end
                        @synchronize
                    end
                end
            end
        end

        @nexprs 8 c -> begin
            pos_c = (c - 1) * workgroup + lid
            if pos_c <= bsize
                array[bstart + pos_c - 1] = shared_v[pos_c]
            end
        end
    end
end

# Full descending = reverse a fully-ascending-sorted array.  out[i] =
# array[N+1-i]  (disjoint write per thread ⇒ race-free, separate buffer).
@kernel inbounds = true unsafe_indices = true function reverse_kernel!(
        out::AbstractVector{T},
        @Const(array::AbstractVector{T}),
) where {T}
    lid = Int(@index(Local))
    gid = Int(@index(Group))
    i = (gid - 1) * Int(@groupsize()[1]) + lid
    n = length(array)
    if i <= n
        out[i] = array[n + 1 - i]
    end
end

# ── C1: register-resident bitonic for buckets ≤256.
#  4 items/thread, workgroup=64 (2 warps), 2 @synchronize total (only
#  d=128 crosses warps).  Processes bucket ids[gid] (compacted list of
#  size-≤256 buckets); ascending in place; typemax pad; `<` comparator
#  (fast path only — use_tag stays on the uniform tag kernel).
const REG_LEAF_MAX = 256

# The register Stage-C leaf kernels use warp `@shfl`, only safe for the
# standard numeric bit-types (the default fast path + any numeric `lt`).
# Exotic user bitstypes route through the shared (no-shuffle) kernels.
@inline reg_leaf_eligible(::Type{T}) where {T} =
    (T <: Integer) || (T <: AbstractFloat)

# Value compare-exchange under a general comparator.  `lt===(<)` (the
# default) monomorphizes to identical code to the old `(v<p)` form.
@inline compare_exchange(v, p, take_lo::Bool, lt) = (take_lo ⊻ lt(v, p)) ? p : v

# Tag (value, valid) compare-exchange — invalid pads sort to the END
# (above any valid item under `lt`).  Mirrors oem tag_swap_decision.
@inline function compare_exchange_tag(v, va::Bool, p, pva::Bool, take_lo::Bool, lt)
    less = (va & !pva) | (va & pva & lt(v, p))
    return (take_lo ⊻ less) ? (p, pva) : (v, va)
end

@kernel inbounds = true unsafe_indices = true function leaf_register_kernel!(
        array::AbstractVector{T},
        @Const(off::AbstractVector{UInt32}),
        @Const(ids::AbstractVector{UInt32}),     # bucket ids, size ≤ BMAX
        nids::Int,
        lt,
        padval::T,
) where {T}
    lid = Int(@index(Local))
    gid = Int(@index(Group))
    warp_0 = (lid - 1) ÷ 32
    lane_0 = (lid - 1) % 32

    inrange = gid <= nids
    b       = inrange ? Int(ids[gid]) : 1
    bstart  = Int(off[b]) + 1
    bsize   = inrange ? (Int(off[b + 1]) - Int(off[b])) : 0
    nonempty = bsize > 0

    slot_pos_1 = warp_0 * 128 + lane_0 + 1
    slot_pos_2 = slot_pos_1 + 32
    slot_pos_3 = slot_pos_2 + 32
    slot_pos_4 = slot_pos_3 + 32
    i_1 = slot_pos_1 - 1; i_2 = slot_pos_2 - 1
    i_3 = slot_pos_3 - 1; i_4 = slot_pos_4 - 1

    v_1 = (nonempty && slot_pos_1 <= bsize) ? array[bstart + slot_pos_1 - 1] : padval
    v_2 = (nonempty && slot_pos_2 <= bsize) ? array[bstart + slot_pos_2 - 1] : padval
    v_3 = (nonempty && slot_pos_3 <= bsize) ? array[bstart + slot_pos_3 - 1] : padval
    v_4 = (nonempty && slot_pos_4 <= bsize) ? array[bstart + slot_pos_4 - 1] : padval

    shared_buf = @localmem T (REG_LEAF_MAX,)

    @nexprs 8 lvl_p -> begin
        if (1 << lvl_p) <= REG_LEAF_MAX
            @nexprs 8 stg_q -> begin
                if stg_q <= lvl_p
                    q_idx = lvl_p - stg_q + 1
                    d = 1 << (q_idx - 1)
                    if d < 32
                        p_1 = @shfl(Xor, v_1, d)
                        p_2 = @shfl(Xor, v_2, d)
                        p_3 = @shfl(Xor, v_3, d)
                        p_4 = @shfl(Xor, v_4, d)
                        asc_1 = ((i_1 >> lvl_p) & 1) == 0
                        asc_2 = ((i_2 >> lvl_p) & 1) == 0
                        asc_3 = ((i_3 >> lvl_p) & 1) == 0
                        asc_4 = ((i_4 >> lvl_p) & 1) == 0
                        am_lo_1 = ((i_1 >> (q_idx - 1)) & 1) == 0
                        am_lo_2 = ((i_2 >> (q_idx - 1)) & 1) == 0
                        am_lo_3 = ((i_3 >> (q_idx - 1)) & 1) == 0
                        am_lo_4 = ((i_4 >> (q_idx - 1)) & 1) == 0
                        v_1 = compare_exchange(v_1, p_1, am_lo_1 == asc_1, lt)
                        v_2 = compare_exchange(v_2, p_2, am_lo_2 == asc_2, lt)
                        v_3 = compare_exchange(v_3, p_3, am_lo_3 == asc_3, lt)
                        v_4 = compare_exchange(v_4, p_4, am_lo_4 == asc_4, lt)
                    elseif d == 32
                        asc_12 = ((i_1 >> lvl_p) & 1) == 0
                        asc_34 = ((i_3 >> lvl_p) & 1) == 0
                        new_v_1 = compare_exchange(v_1, v_2, asc_12, lt)
                        new_v_2 = compare_exchange(v_1, v_2, !asc_12, lt)
                        new_v_3 = compare_exchange(v_3, v_4, asc_34, lt)
                        new_v_4 = compare_exchange(v_3, v_4, !asc_34, lt)
                        v_1 = new_v_1; v_2 = new_v_2
                        v_3 = new_v_3; v_4 = new_v_4
                    elseif d == 64
                        asc_13 = ((i_1 >> lvl_p) & 1) == 0
                        asc_24 = ((i_2 >> lvl_p) & 1) == 0
                        new_v_1 = compare_exchange(v_1, v_3, asc_13, lt)
                        new_v_3 = compare_exchange(v_1, v_3, !asc_13, lt)
                        new_v_2 = compare_exchange(v_2, v_4, asc_24, lt)
                        new_v_4 = compare_exchange(v_2, v_4, !asc_24, lt)
                        v_1 = new_v_1; v_2 = new_v_2
                        v_3 = new_v_3; v_4 = new_v_4
                    else
                        shared_buf[slot_pos_1] = v_1
                        shared_buf[slot_pos_2] = v_2
                        shared_buf[slot_pos_3] = v_3
                        shared_buf[slot_pos_4] = v_4
                        @synchronize
                        p_1 = shared_buf[(i_1 ⊻ d) + 1]
                        p_2 = shared_buf[(i_2 ⊻ d) + 1]
                        p_3 = shared_buf[(i_3 ⊻ d) + 1]
                        p_4 = shared_buf[(i_4 ⊻ d) + 1]
                        asc_1 = ((i_1 >> lvl_p) & 1) == 0
                        asc_2 = ((i_2 >> lvl_p) & 1) == 0
                        asc_3 = ((i_3 >> lvl_p) & 1) == 0
                        asc_4 = ((i_4 >> lvl_p) & 1) == 0
                        am_lo_1 = (i_1 & d) == 0
                        am_lo_2 = (i_2 & d) == 0
                        am_lo_3 = (i_3 & d) == 0
                        am_lo_4 = (i_4 & d) == 0
                        v_1 = compare_exchange(v_1, p_1, am_lo_1 == asc_1, lt)
                        v_2 = compare_exchange(v_2, p_2, am_lo_2 == asc_2, lt)
                        v_3 = compare_exchange(v_3, p_3, am_lo_3 == asc_3, lt)
                        v_4 = compare_exchange(v_4, p_4, am_lo_4 == asc_4, lt)
                        @synchronize
                    end
                end
            end
        end
    end

    if nonempty
        slot_pos_1 <= bsize && (array[bstart + slot_pos_1 - 1] = v_1)
        slot_pos_2 <= bsize && (array[bstart + slot_pos_2 - 1] = v_2)
        slot_pos_3 <= bsize && (array[bstart + slot_pos_3 - 1] = v_3)
        slot_pos_4 <= bsize && (array[bstart + slot_pos_4 - 1] = v_4)
    end
end

# TAG variant of the ≤256 register Stage-C bitonic (custom `lt` or
# no-`typemax` T).  Carries a parallel valid-Bool register set; invalid
# pad slots sort to the END via compare_exchange_tag.  `@shfl(Xor, va, d)`
# shuffles the Bool partner-valid (oem_warp tag pattern).
@kernel inbounds = true unsafe_indices = true function leaf_register_tag_kernel!(
        array::AbstractVector{T},
        @Const(off::AbstractVector{UInt32}),
        @Const(ids::AbstractVector{UInt32}),     # bucket ids, size ≤ BMAX
        nids::Int,
        lt,
) where {T}
    lid = Int(@index(Local))
    gid = Int(@index(Group))
    warp_0 = (lid - 1) ÷ 32
    lane_0 = (lid - 1) % 32

    inrange = gid <= nids
    b       = inrange ? Int(ids[gid]) : 1
    bstart  = Int(off[b]) + 1
    bsize   = inrange ? (Int(off[b + 1]) - Int(off[b])) : 0
    nonempty = bsize > 0

    slot_pos_1 = warp_0 * 128 + lane_0 + 1
    slot_pos_2 = slot_pos_1 + 32
    slot_pos_3 = slot_pos_2 + 32
    slot_pos_4 = slot_pos_3 + 32
    i_1 = slot_pos_1 - 1; i_2 = slot_pos_2 - 1
    i_3 = slot_pos_3 - 1; i_4 = slot_pos_4 - 1

    va_1 = nonempty && slot_pos_1 <= bsize
    va_2 = nonempty && slot_pos_2 <= bsize
    va_3 = nonempty && slot_pos_3 <= bsize
    va_4 = nonempty && slot_pos_4 <= bsize
    # Clamped reads for invalid lanes — values masked by va flags.
    v_1 = array[bstart + (va_1 ? slot_pos_1 : 1) - 1]
    v_2 = array[bstart + (va_2 ? slot_pos_2 : 1) - 1]
    v_3 = array[bstart + (va_3 ? slot_pos_3 : 1) - 1]
    v_4 = array[bstart + (va_4 ? slot_pos_4 : 1) - 1]

    shared_buf = @localmem T    (REG_LEAF_MAX,)
    shared_va  = @localmem Bool (REG_LEAF_MAX,)

    @nexprs 8 lvl_p -> begin
        if (1 << lvl_p) <= REG_LEAF_MAX
            @nexprs 8 stg_q -> begin
                if stg_q <= lvl_p
                    q_idx = lvl_p - stg_q + 1
                    d = 1 << (q_idx - 1)
                    if d < 32
                        p_1 = @shfl(Xor, v_1, d)
                        p_2 = @shfl(Xor, v_2, d)
                        p_3 = @shfl(Xor, v_3, d)
                        p_4 = @shfl(Xor, v_4, d)
                        pa_1 = @shfl(Xor, va_1, d)
                        pa_2 = @shfl(Xor, va_2, d)
                        pa_3 = @shfl(Xor, va_3, d)
                        pa_4 = @shfl(Xor, va_4, d)
                        asc_1 = ((i_1 >> lvl_p) & 1) == 0
                        asc_2 = ((i_2 >> lvl_p) & 1) == 0
                        asc_3 = ((i_3 >> lvl_p) & 1) == 0
                        asc_4 = ((i_4 >> lvl_p) & 1) == 0
                        am_lo_1 = ((i_1 >> (q_idx - 1)) & 1) == 0
                        am_lo_2 = ((i_2 >> (q_idx - 1)) & 1) == 0
                        am_lo_3 = ((i_3 >> (q_idx - 1)) & 1) == 0
                        am_lo_4 = ((i_4 >> (q_idx - 1)) & 1) == 0
                        v_1, va_1 = compare_exchange_tag(v_1, va_1, p_1, pa_1, am_lo_1 == asc_1, lt)
                        v_2, va_2 = compare_exchange_tag(v_2, va_2, p_2, pa_2, am_lo_2 == asc_2, lt)
                        v_3, va_3 = compare_exchange_tag(v_3, va_3, p_3, pa_3, am_lo_3 == asc_3, lt)
                        v_4, va_4 = compare_exchange_tag(v_4, va_4, p_4, pa_4, am_lo_4 == asc_4, lt)
                    elseif d == 32
                        asc_12 = ((i_1 >> lvl_p) & 1) == 0
                        asc_34 = ((i_3 >> lvl_p) & 1) == 0
                        nv_1, na_1 = compare_exchange_tag(v_1, va_1, v_2, va_2, asc_12, lt)
                        nv_2, na_2 = compare_exchange_tag(v_1, va_1, v_2, va_2, !asc_12, lt)
                        nv_3, na_3 = compare_exchange_tag(v_3, va_3, v_4, va_4, asc_34, lt)
                        nv_4, na_4 = compare_exchange_tag(v_3, va_3, v_4, va_4, !asc_34, lt)
                        v_1 = nv_1; va_1 = na_1; v_2 = nv_2; va_2 = na_2
                        v_3 = nv_3; va_3 = na_3; v_4 = nv_4; va_4 = na_4
                    elseif d == 64
                        asc_13 = ((i_1 >> lvl_p) & 1) == 0
                        asc_24 = ((i_2 >> lvl_p) & 1) == 0
                        nv_1, na_1 = compare_exchange_tag(v_1, va_1, v_3, va_3, asc_13, lt)
                        nv_3, na_3 = compare_exchange_tag(v_1, va_1, v_3, va_3, !asc_13, lt)
                        nv_2, na_2 = compare_exchange_tag(v_2, va_2, v_4, va_4, asc_24, lt)
                        nv_4, na_4 = compare_exchange_tag(v_2, va_2, v_4, va_4, !asc_24, lt)
                        v_1 = nv_1; va_1 = na_1; v_2 = nv_2; va_2 = na_2
                        v_3 = nv_3; va_3 = na_3; v_4 = nv_4; va_4 = na_4
                    else
                        shared_buf[slot_pos_1] = v_1
                        shared_buf[slot_pos_2] = v_2
                        shared_buf[slot_pos_3] = v_3
                        shared_buf[slot_pos_4] = v_4
                        shared_va[slot_pos_1] = va_1
                        shared_va[slot_pos_2] = va_2
                        shared_va[slot_pos_3] = va_3
                        shared_va[slot_pos_4] = va_4
                        @synchronize
                        p_1 = shared_buf[(i_1 ⊻ d) + 1]
                        p_2 = shared_buf[(i_2 ⊻ d) + 1]
                        p_3 = shared_buf[(i_3 ⊻ d) + 1]
                        p_4 = shared_buf[(i_4 ⊻ d) + 1]
                        pa_1 = shared_va[(i_1 ⊻ d) + 1]
                        pa_2 = shared_va[(i_2 ⊻ d) + 1]
                        pa_3 = shared_va[(i_3 ⊻ d) + 1]
                        pa_4 = shared_va[(i_4 ⊻ d) + 1]
                        asc_1 = ((i_1 >> lvl_p) & 1) == 0
                        asc_2 = ((i_2 >> lvl_p) & 1) == 0
                        asc_3 = ((i_3 >> lvl_p) & 1) == 0
                        asc_4 = ((i_4 >> lvl_p) & 1) == 0
                        am_lo_1 = (i_1 & d) == 0
                        am_lo_2 = (i_2 & d) == 0
                        am_lo_3 = (i_3 & d) == 0
                        am_lo_4 = (i_4 & d) == 0
                        v_1, va_1 = compare_exchange_tag(v_1, va_1, p_1, pa_1, am_lo_1 == asc_1, lt)
                        v_2, va_2 = compare_exchange_tag(v_2, va_2, p_2, pa_2, am_lo_2 == asc_2, lt)
                        v_3, va_3 = compare_exchange_tag(v_3, va_3, p_3, pa_3, am_lo_3 == asc_3, lt)
                        v_4, va_4 = compare_exchange_tag(v_4, va_4, p_4, pa_4, am_lo_4 == asc_4, lt)
                        @synchronize
                    end
                end
            end
        end
    end

    if nonempty
        slot_pos_1 <= bsize && (array[bstart + slot_pos_1 - 1] = v_1)
        slot_pos_2 <= bsize && (array[bstart + slot_pos_2 - 1] = v_2)
        slot_pos_3 <= bsize && (array[bstart + slot_pos_3 - 1] = v_3)
        slot_pos_4 <= bsize && (array[bstart + slot_pos_4 - 1] = v_4)
    end
end

# flag[b] = 1 iff lo < (off[b+1]-off[b]) ≤ hi  (lo=0,hi=BMAX → ≤256 small
# class; lo=TERMINAL,hi=typemax → oversized survivors for the fallback)
@kernel inbounds = true unsafe_indices = true function leaf_oversize_flag_kernel!(
        flag::AbstractVector{UInt32},
        @Const(off::AbstractVector{UInt32}),
        lo::Int, hi::Int, B::Int,
) where {}
    lid = Int(@index(Local)); gid = Int(@index(Group))
    b = (gid - 1) * Int(@groupsize()[1]) + lid
    if b <= B
        sz = Int(off[b + 1]) - Int(off[b])
        flag[b] = (sz > lo && sz <= hi) ? UInt32(1) : UInt32(0)
    end
end

# ── Large-bucket fallback ──────────────────────────────────────────────
# A NON-const bucket still > TERMINAL after MAXLEVEL (pathological pivot
# failure) would overflow the leaf's K_PAD ≤ 4096.  Rare / adversarial
# only.  Robust size-independent fix: copy the bucket into a pow2-padded
# temp and run the SAME bitonic compare-exchange network as the leaf, but
# multi-pass (one (lvl_p,q_idx) step per launch) so it works for any size.
# `padval`/validity-Bool handled exactly like the leaf so a custom `lt` /
# no-`typemax` T stay correct.

# One bitonic step over a pow2 temp (fast: value + padval).
@kernel inbounds = true unsafe_indices = true function bigbucket_step_kernel!(
        tmp::AbstractVector{T},
        lt, P::Int, lvl_p::Int, q_idx::Int,
) where {T}
    lid = Int(@index(Local)); gid = Int(@index(Group))
    t = (gid - 1) * Int(@groupsize()[1]) + lid - 1     # 0-based pair id
    if t < (P >> 1)
        d_off = 1 << (q_idx - 1)
        mask  = d_off - 1
        i_b   = ((t & ~mask) << 1) | (t & mask)
        partner = i_b + d_off
        asc = ((i_b >> lvl_p) & 1) == 0
        a = tmp[i_b + 1]
        b = tmp[partner + 1]
        if asc ⊻ lt(a, b)
            tmp[i_b + 1]     = b
            tmp[partner + 1] = a
        end
    end
end

# One bitonic step (tag: value + validity-Bool; invalid sorts high).
@kernel inbounds = true unsafe_indices = true function bigbucket_step_tag_kernel!(
        tmp::AbstractVector{T},
        tva::AbstractVector{Bool},
        lt, P::Int, lvl_p::Int, q_idx::Int,
) where {T}
    lid = Int(@index(Local)); gid = Int(@index(Group))
    t = (gid - 1) * Int(@groupsize()[1]) + lid - 1
    if t < (P >> 1)
        d_off = 1 << (q_idx - 1)
        mask  = d_off - 1
        i_b   = ((t & ~mask) << 1) | (t & mask)
        partner = i_b + d_off
        asc = ((i_b >> lvl_p) & 1) == 0
        a_v = tmp[i_b + 1];     a_va = tva[i_b + 1]
        b_v = tmp[partner + 1]; b_va = tva[partner + 1]
        is_less = (a_va & !b_va) | (a_va & b_va & lt(a_v, b_v))
        if asc ⊻ is_less
            tmp[i_b + 1]     = b_v;  tva[i_b + 1]     = b_va
            tmp[partner + 1] = a_v;  tva[partner + 1] = a_va
        end
    end
end

# Sort every non-const bucket with size > TERMINAL (rare).  off is already
# on host-readable; loop the (few) oversized buckets, pow2-pad each into a
# fresh temp, run the multi-pass network, copy back in place.
function bigbucket_sort!(array::AbstractVector{T},
                              hoff::Vector{UInt32}, B::Int;
                              lt_used, padval::Union{Nothing,T},
                              use_tag::Bool, backend) where {T}
    nfb = 0
    lmax = leaf_max(T)
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
                    B::Int, max_bucket::Int, ws::SampleSortWorkspace;
                    lt_used = (<),
                    padval::Union{Nothing,T} = nothing,
                    use_tag::Bool = false,
                    backend = get_backend(array)) where {T}
    # Fallback: sort any non-const bucket still > leaf cap (would overflow K_PAD).
    # Rare/adversarial.  After this every remaining bucket ≤ leaf_max(T); the larger
    # ones are sorted in place and skipped below (bsize > K_PAD).
    lmax = leaf_max(T)
    if max_bucket > lmax
        hoff = Array(view(off, 1:B + 1))
        bigbucket_sort!(array, hoff, B;
                             lt_used, padval, use_tag, backend)
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
    leaf_max(T) > 4096 && _leafclass!(4096, Val(8192))
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
"""
function sample_sort(src::AbstractVector{T};
                         backend = get_backend(src),
                         lt = nothing,
                         tmax = nothing,
                         reverse::Bool = false,
                         ws::Union{Nothing,SampleSortWorkspace{T}} = nothing) where {T}
    N = length(src)
    w = ws === nothing ? sample_sort_workspace(T, N; backend) : ws
    @assert w.N == N "workspace built for N=$(w.N), got N=$N"

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
        partition_loop!(w, src; lt_used, backend)
    if mx > 0
        leaf_sort!(view(cur, 1:N), view(w.offsets, 1:B + 1), B, mx, w;
                   lt_used, padval, use_tag, backend)
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

