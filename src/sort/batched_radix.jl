# ============================================================================
# Batched LSD radix sort for K × M matrices.
#
# Sorts each column of a `K × M` matrix independently, in parallel, with a
# single launch per stage (vs M sequential `KF.sort!` calls). Generic over
# any uint_map-eligible type (UInt{8,16,32,64} / Int{8,16,32,64} /
# Float32 / Float64).
#
# Two architectures, picked by K relative to `BATCH_RADIX_SHARED_MAX(T)`:
#
#   smallK (K ≤ shared budget):
#     One block per column. Loads the whole column into shared memory,
#     runs all `npasses` byte-passes via ping-pong shared_a ↔ shared_b.
#     ONE kernel launch sorts all M columns.
#
#   largeK (K > shared budget):
#     Multi-block per column with per-column decoupled-lookback. Pipeline
#     of (batched hist) + (per-pass scan) + (`npasses` batched onesweeps).
#     Each onesweep launches all blocks of all columns together.
#
# Architecture cloned from src/sort/onesweep_kernel.jl (+ bucket_histogram
# + scan_histogram), with a `col_idx` dim added to every per-column buffer
# and a `(g_in_col, col_idx)` decode of the block index. Tail tiles are
# bounds-checked so K can be any positive integer.
# ============================================================================

using KernelAbstractions
using KernelIntrinsics
using KernelAbstractions: @atomic
using Base.Cartesian: @nexprs


# ── Constants ──────────────────────────────────────────────────────────

const BATCH_RADIX_NBUCKETS = 256
const BATCH_RADIX_WG       = 256
const BATCH_RADIX_WARPSZ_NV = 32     # NVIDIA assumed; AMD code path uses warpsz=64
const BATCH_RADIX_NITEM    = 16

# Per-T choices: for ≤4-byte types we use Nchunks=2 (block_size_max = 8192);
# for 8-byte types Nchunks=1 (block_size_max = 4096) — both yield 32 KB
# shared_sorted, fitting in the 48 KB static-shared budget.
@inline _batch_radix_nchunks(::Type{T}) where T = sizeof(T) <= 4 ? 2 : 1
@inline _batch_radix_block_size_max(::Type{T}) where T =
    BATCH_RADIX_WG * _batch_radix_nchunks(T) * BATCH_RADIX_NITEM

# Largest K the small-K path can handle for a given T (2 × K_PAD × sizeof
# ping-pong + ~9 KB hist/aux ≤ 48 KB):
@inline _batch_radix_smallK_max(::Type{T}) where T =
    sizeof(T) <= 4 ? 4096 : 2048


# ── Kernel 1: small-K (single block per column) ─────────────────────────

const _batch_radix_smallK_cache = Dict{Tuple{Int,Int}, Any}()

function _build_batch_radix_smallK_def(name::Symbol, items::Int, npasses::Int)
    nbuckets    = BATCH_RADIX_NBUCKETS
    wg          = BATCH_RADIX_WG
    warpsz      = BATCH_RADIX_WARPSZ_NV
    wpb         = wg ÷ warpsz
    nb_div_wsz  = nbuckets ÷ warpsz

    body = quote
        K_real = size(A, 1)
        gid = Int(@index(Group))    # 1..M (one block per column)
        lid = Int(@index(Local))    # 1..WG

        warp_id = (lid - 1) ÷ $warpsz + 1
        lane    = (lid - 1) % $warpsz + 1
        bucket  = lid

        items_per_warp    = $items * $warpsz
        warp_first_in_col = (warp_id - 1) * items_per_warp + 1

        shared_a    = @localmem T      (K_PAD,)
        shared_b    = @localmem T      (K_PAD,)
        shared_hist = @localmem UInt32 ($nbuckets, $wpb)
        shared_aux  = @localmem UInt32 ($nbuckets + $wpb,)

        # Load column into shared_a. `pos_r` and `in_bounds_r` are constant
        # per (lane, r) across all passes — compute once here and reuse.
        # OOB slots get a dummy in-bounds load (A[1, gid]); they're gated
        # out of every later atomic-add and every scatter via `in_bounds_r`,
        # so the dummy value is never used. No `typemax(T)` needed.
        @nexprs $items r -> begin
            pos_r = warp_first_in_col + (r - 1) * $warpsz + (lane - 1)
            in_bounds_r = pos_r <= K_real
            shared_a[pos_r] = in_bounds_r ? A[pos_r, gid] : A[1, gid]
        end
        @synchronize

        for pass in 0:($npasses - 1)
            shift    = Int32(8 * pass)
            src_is_a = (pass & 1) == 0

            # zero hist
            for s in 1:$nb_div_wsz
                b_init = lane + (s - 1) * $warpsz
                shared_hist[b_init, warp_id] = UInt32(0)
            end
            @synchronize

            # Phase 1: read, compute key via uint_map, atomic-add rank.
            # OOB items still read (the dummy from load phase) but DO NOT
            # contribute to the histogram.
            @nexprs $items r -> begin
                if src_is_a
                    item_r = shared_a[pos_r]
                else
                    item_r = shared_b[pos_r]
                end
                if in_bounds_r
                    key_r  = uint_map(by(item_r))
                    d_r    = Int((key_r >> shift) & 0xFF) + 1
                    rank_r = (@atomic shared_hist[d_r, warp_id] += UInt32(1)) - UInt32(1)
                else
                    d_r    = 1
                    rank_r = UInt32(0)
                end
            end
            @synchronize

            # Phase 2: per-bucket exclusive scan over warps.
            prefix_acc = UInt32(0)
            for w in 1:$wpb
                v_pa = shared_hist[bucket, w]
                shared_hist[bucket, w] = prefix_acc
                prefix_acc += v_pa
            end
            block_total_b = prefix_acc

            # Phase 3 (warp-totals + block_d_start).
            val_inc = block_total_b
            @warpreduce(val_inc, +, lane, $warpsz)
            val_exc_within = val_inc - block_total_b

            if lane == $warpsz
                shared_aux[$nbuckets + warp_id] = val_inc
            end
            @synchronize

            if warp_id == 1
                wt = lane <= $wpb ? shared_aux[$nbuckets + lane] : UInt32(0)
                inc_v = wt
                @warpreduce(inc_v, +, lane, $wpb)
                excl_v = inc_v - wt
                if lane <= $wpb
                    shared_aux[$nbuckets + lane] = excl_v
                end
            end
            @synchronize

            warp_prefix = shared_aux[$nbuckets + warp_id]
            own_bds_reg = warp_prefix + val_exc_within
            shared_aux[bucket] = own_bds_reg
            @synchronize

            # Phase 4: warp_block_local_base into shared_hist.
            for s in 1:$nb_div_wsz
                b_seed = lane + (s - 1) * $warpsz
                my_excl_inline = shared_hist[b_seed, warp_id]
                shared_hist[b_seed, warp_id] = shared_aux[b_seed] + my_excl_inline
            end
            @synchronize

            # Phase 5: scatter into the OTHER ping-pong buffer.
            # OOB items do NOT scatter — positions K_real+1..K_PAD in the
            # destination buffer are left at whatever they were (initially
            # the load-phase dummy in shared_a; nothing in shared_b for the
            # first scatter pass). They're never read for digit decisions
            # because phase 1 of the next pass gates atomic-add on
            # in_bounds_r too.
            @nexprs $items r -> begin
                if in_bounds_r
                    block_local_pos_r = shared_hist[d_r, warp_id] + rank_r
                    dst_pos_r = Int(block_local_pos_r) + 1
                    if src_is_a
                        shared_b[dst_pos_r] = item_r
                    else
                        shared_a[dst_pos_r] = item_r
                    end
                end
            end
            @synchronize
        end

        # Store sorted column back. Result in shared_a if npasses even,
        # else in shared_b. Reuse cached `pos_r` / `in_bounds_r`.
        final_in_a = ($npasses & 1) == 0
        @nexprs $items r -> begin
            if in_bounds_r
                if final_in_a
                    A[pos_r, gid] = shared_a[pos_r]
                else
                    A[pos_r, gid] = shared_b[pos_r]
                end
            end
        end
    end

    return quote
        @kernel inbounds=true unsafe_indices=true function $(name)(
                A::AbstractMatrix{T},
                by::F, uint_map::UM,
                ::Val{K_PAD}
        ) where {T, F, UM, K_PAD}
            $body
        end
    end
end

function _define_batch_radix_smallK!(items::Int, npasses::Int)
    name = Symbol("batch_radix_smallK_", items, "_", npasses, "!")
    fn = Core.eval(@__MODULE__, quote
        $(_build_batch_radix_smallK_def(name, items, npasses))
        $name
    end)
    Core.eval(@__MODULE__,
        :(@inline get_batch_radix_smallK_kernel(::Val{$items}, ::Val{$npasses}) = $fn))
    _batch_radix_smallK_cache[(items, npasses)] = fn
    return fn
end

# Lazy fallback for non-default (items, npasses).
function get_batch_radix_smallK_kernel(::Val{items}, ::Val{npasses}) where {items, npasses}
    key = (items, npasses)
    haskey(_batch_radix_smallK_cache, key) && return _batch_radix_smallK_cache[key]
    return _define_batch_radix_smallK!(items, npasses)
end


# ── Kernel 2: large-K batched histogram ─────────────────────────────────

const _batch_radix_hist_cache = Dict{Tuple{Int,Int,Int}, Any}()

function _build_batch_radix_hist_def(name::Symbol, nitem::Int, npasses::Int, items_per_hist_block::Int)
    nbuckets = BATCH_RADIX_NBUCKETS
    wg       = BATCH_RADIX_WG

    body = quote
        lid = Int(@index(Local))
        gid = Int(@index(Group))

        # Decode (g_in_col, col_idx).
        g_in_col = (gid - 1) % nblocks_per_col_hist + 1
        col_idx  = (gid - 1) ÷ nblocks_per_col_hist + 1
        col_offset = (col_idx - 1) * K_per_col

        block_last_pos_in_col = g_in_col * $items_per_hist_block
        is_full_tile = block_last_pos_in_col <= K_per_col

        shared_hist = @localmem UInt32 ($nbuckets, $npasses)

        # Zero shared_hist (nbuckets * npasses entries).
        let total_init = $nbuckets * $npasses
            for k in lid:$wg:total_init
                shared_hist[k] = UInt32(0)
            end
        end
        @synchronize

        # Scalar lid-strided loads. (vload requires col_offset alignment to
        # nitem — not true when K is not a multiple of nitem.)
        if is_full_tile
            @nexprs $nitem i -> begin
                pos_i_local  = (i - 1) * $wg + lid
                pos_i_in_col = (g_in_col - 1) * $items_per_hist_block + pos_i_local
                item_i = v[col_offset + pos_i_in_col]
                key_i  = uint_map(by(item_i))
                @nexprs $npasses p -> begin
                    shift_pi = Int32((p - 1) * 8)
                    d_pi     = Int((key_i >> shift_pi) & 0xFF) + 1
                    @atomic shared_hist[d_pi, p] += UInt32(1)
                end
            end
        else
            @nexprs $nitem i -> begin
                pos_i_local  = (i - 1) * $wg + lid
                pos_i_in_col = (g_in_col - 1) * $items_per_hist_block + pos_i_local
                if pos_i_in_col <= K_per_col
                    item_i = v[col_offset + pos_i_in_col]
                    key_i  = uint_map(by(item_i))
                    @nexprs $npasses p -> begin
                        shift_pi = Int32((p - 1) * 8)
                        d_pi     = Int((key_i >> shift_pi) & 0xFF) + 1
                        @atomic shared_hist[d_pi, p] += UInt32(1)
                    end
                end
            end
        end
        @synchronize

        # Flush shared_hist → global hist[:, :, col_idx] (atomic-add).
        for p in 1:$npasses
            for b in lid:$wg:$nbuckets
                cnt = shared_hist[b, p]
                if cnt != UInt32(0)
                    @atomic hist[b, p, col_idx] += cnt
                end
            end
        end
    end

    return quote
        @kernel inbounds=true unsafe_indices=true function $(name)(
                hist,
                v::AbstractVector{T},
                by::F, uint_map::UM,
                ::Val{K_per_col},
                ::Val{nblocks_per_col_hist},
        ) where {T, F, UM, K_per_col, nblocks_per_col_hist}
            $body
        end
    end
end

function _define_batch_radix_hist!(nitem::Int, npasses::Int, items_per_hist_block::Int)
    name = Symbol("batch_radix_hist_", nitem, "_", npasses, "_", items_per_hist_block, "!")
    fn = Core.eval(@__MODULE__, quote
        $(_build_batch_radix_hist_def(name, nitem, npasses, items_per_hist_block))
        $name
    end)
    Core.eval(@__MODULE__,
        :(@inline get_batch_radix_hist_kernel(
            ::Val{$nitem}, ::Val{$npasses}, ::Val{$items_per_hist_block}) = $fn))
    _batch_radix_hist_cache[(nitem, npasses, items_per_hist_block)] = fn
    return fn
end

function get_batch_radix_hist_kernel(::Val{nitem}, ::Val{npasses}, ::Val{ipb}
                                     ) where {nitem, npasses, ipb}
    key = (nitem, npasses, ipb)
    haskey(_batch_radix_hist_cache, key) && return _batch_radix_hist_cache[key]
    return _define_batch_radix_hist!(nitem, npasses, ipb)
end


# ── Kernel 3: large-K batched onesweep ──────────────────────────────────

const _batch_radix_onesweep_cache = Dict{Tuple{Int,Int}, Any}()

function _build_batch_radix_onesweep_def(name::Symbol, nitem::Int, nchunks::Int)
    nbuckets = BATCH_RADIX_NBUCKETS
    wg       = BATCH_RADIX_WG
    warpsz   = BATCH_RADIX_WARPSZ_NV
    wpb      = wg ÷ warpsz
    niter_5b = nchunks * nitem

    body = quote
        N = length(src)
        digit_mask = UInt32($nbuckets - 1)
        T = eltype(dst)
        block_size_max = $wpb * $nchunks * $warpsz * $nitem

        lid = Int(@index(Local))
        gid = Int(@index(Group))

        g_in_col = (gid - 1) % nblocks_per_col + 1
        col_idx  = (gid - 1) ÷ nblocks_per_col + 1
        col_offset = (col_idx - 1) * K_per_col

        warp_id = (lid - 1) ÷ $warpsz + 1
        lane    = (lid - 1) % $warpsz + 1
        global_warp_in_col = (g_in_col - 1) * $wpb + warp_id

        warp_first_pos  = col_offset + (global_warp_in_col - 1) * $nchunks * $warpsz * $nitem + 1
        block_first_pos = col_offset + (g_in_col - 1) * block_size_max + 1
        block_last_real_pos = col_offset + min(g_in_col * block_size_max, K_per_col)
        block_size_actual   = block_last_real_pos - block_first_pos + 1
        is_full_tile = (g_in_col * block_size_max) <= K_per_col

        shared_hist   = @localmem UInt32 ($nbuckets, $wpb)
        shared_aux    = @localmem UInt32 ($nbuckets + $wpb,)
        shared_sorted = @localmem T (block_size_max,)

        # Phase 1a: zero shared_hist (per-warp).
        for s in 1:($nbuckets ÷ $warpsz)
            b_init = lane + (s - 1) * $warpsz
            shared_hist[b_init, warp_id] = UInt32(0)
        end
        @synchronize

        # Phase 1b: strided load + atomic-add. Tail-tile bounds checks.
        if is_full_tile
            @nexprs $nchunks c -> begin
                chunk_base_c = warp_first_pos + (c - 1) * $warpsz * $nitem
                @nexprs $nitem i -> begin
                    pos_c_i  = chunk_base_c + (i - 1) * $warpsz + (lane - 1)
                    item_c_i = src[pos_c_i]
                    in_bounds_c_i = true
                    key_c_i  = uint_map(by(item_c_i))
                    d_c_i    = Int((key_c_i >> shift) & digit_mask) + 1
                    rank_c_i = (@atomic shared_hist[d_c_i, warp_id] += UInt32(1)) - UInt32(1)
                end
            end
        else
            @nexprs $nchunks c -> begin
                chunk_base_c = warp_first_pos + (c - 1) * $warpsz * $nitem
                @nexprs $nitem i -> begin
                    pos_c_i = chunk_base_c + (i - 1) * $warpsz + (lane - 1)
                    in_bounds_c_i = pos_c_i <= block_last_real_pos
                    item_c_i = in_bounds_c_i ? src[pos_c_i] : src[col_offset + 1]
                    if in_bounds_c_i
                        key_c_i  = uint_map(by(item_c_i))
                        d_c_i    = Int((key_c_i >> shift) & digit_mask) + 1
                        rank_c_i = (@atomic shared_hist[d_c_i, warp_id] += UInt32(1)) - UInt32(1)
                    else
                        d_c_i    = 1
                        rank_c_i = UInt32(0)
                    end
                end
            end
        end
        @synchronize

        # Phase 2: per-bucket exclusive scan over warps.
        bucket = lid
        prefix_acc = UInt32(0)
        for w in 1:$wpb
            v_pa = shared_hist[bucket, w]
            shared_hist[bucket, w] = prefix_acc
            prefix_acc += v_pa
        end
        block_total_b = prefix_acc

        # Phase 3a: publish aggregate to partial1 / partial2 (this column).
        @access partial1[bucket, g_in_col, col_idx] = block_total_b
        @access partial2[bucket, g_in_col, col_idx] = block_total_b
        @synchronize
        if lid == 1
            @access flag[g_in_col, col_idx] = 0x01
        end

        # Phase 4.5: warp-totals reduction.
        val_inc = block_total_b
        @warpreduce(val_inc, +, lane, $warpsz)
        val_exc_within = val_inc - block_total_b

        if lane == $warpsz
            shared_aux[$nbuckets + warp_id] = val_inc
        end
        @synchronize

        if warp_id == 1
            wt = lane <= $wpb ? shared_aux[$nbuckets + lane] : UInt32(0)
            inc_v = wt
            @warpreduce(inc_v, +, lane, $wpb)
            excl_v = inc_v - wt
            if lane <= $wpb
                shared_aux[$nbuckets + lane] = excl_v
            end
        end
        @synchronize

        warp_prefix = shared_aux[$nbuckets + warp_id]
        own_bds_reg = warp_prefix + val_exc_within
        shared_aux[bucket] = own_bds_reg
        @synchronize

        # Phase 4 first: warp_block_local_base into shared_hist.
        for s in 1:($nbuckets ÷ $warpsz)
            b_seed = lane + (s - 1) * $warpsz
            my_excl_inline = shared_hist[b_seed, warp_id]
            warp_block_local_base = shared_aux[b_seed] + my_excl_inline
            shared_hist[b_seed, warp_id] = warp_block_local_base
        end
        @synchronize

        # Phase 5a: shuffle into shared_sorted.
        if is_full_tile
            @nexprs $nchunks c -> begin
                @nexprs $nitem i -> begin
                    key_5a_c_i = uint_map(by(item_c_i))
                    d_5a_c_i = Int((key_5a_c_i >> shift) & digit_mask) + 1
                    block_local_pos_c_i = shared_hist[d_5a_c_i, warp_id] + rank_c_i
                    shared_sorted[Int(block_local_pos_c_i) + 1] = item_c_i
                end
            end
        else
            @nexprs $nchunks c -> begin
                @nexprs $nitem i -> begin
                    if in_bounds_c_i
                        key_5a_c_i = uint_map(by(item_c_i))
                        d_5a_c_i = Int((key_5a_c_i >> shift) & digit_mask) + 1
                        block_local_pos_c_i = shared_hist[d_5a_c_i, warp_id] + rank_c_i
                        shared_sorted[Int(block_local_pos_c_i) + 1] = item_c_i
                    end
                end
            end
        end
        @synchronize

        # Phase 3b: per-column decoupled lookback within THIS column.
        if lid == 1
            shared_hist[1, 1] = UInt32(g_in_col - 1) << 8
        end
        @synchronize

        cross_block_prefix = UInt32(0)
        if g_in_col >= 2
            contains_prefix = false
            while !contains_prefix
                if warp_id == 1 && lane == 1
                    comb = shared_hist[1, 1]
                    local_b = Int(comb >> 8)
                    local_flg = UInt32(0)
                    while local_flg == UInt32(0)
                        @access tmp_flg = flag[local_b, col_idx]
                        local_flg = UInt32(tmp_flg)
                    end
                    shared_hist[1, 1] = (UInt32(local_b) << 8) | local_flg
                end
                @synchronize

                comb = shared_hist[1, 1]
                flg_val = comb & UInt32(0xFF)
                b_val   = Int(comb >> 8)

                if flg_val == UInt32(0x02)
                    @access pval = partial2[bucket, b_val, col_idx]
                    cross_block_prefix += pval
                    contains_prefix = true
                else
                    @access pval = partial1[bucket, b_val, col_idx]
                    cross_block_prefix += pval
                    if b_val == 1
                        contains_prefix = true
                    else
                        if lid == 1
                            shared_hist[1, 1] = UInt32(b_val - 1) << 8
                        end
                        @synchronize
                    end
                end
            end
        end

        @access partial2[bucket, g_in_col, col_idx] = cross_block_prefix + block_total_b
        @synchronize
        if lid == 1
            @access flag[g_in_col, col_idx] = 0x02
        end

        # Phase 4 second: dst offset.
        shared_hist[bucket, 1] = global_excl_hist[bucket, col_idx] + cross_block_prefix - own_bds_reg
        @synchronize

        # Phase 5b: stream-out. Bounds-cap at block_size_actual for tail tiles.
        @nexprs $niter_5b c -> begin
            p_c = (c - 1) * ($wpb * $warpsz) + lid
            if p_c <= block_size_actual
                item_5b_c = shared_sorted[p_c]
                key_5b_c  = uint_map(by(item_5b_c))
                d_5b_c    = Int((key_5b_c >> shift) & digit_mask) + 1
                dst[col_offset + Int(shared_hist[d_5b_c, 1]) + p_c] = item_5b_c
            end
        end
    end

    return quote
        @kernel inbounds=true unsafe_indices=true function $(name)(
                dst::AbstractVector,
                src::AbstractVector,
                by::F, uint_map::UM,
                shift::Int32,
                global_excl_hist::AbstractMatrix{UInt32},
                partial1::AbstractArray{UInt32, 3},
                partial2::AbstractArray{UInt32, 3},
                flag::AbstractMatrix{UInt8},
                ::Val{K_per_col},
                ::Val{nblocks_per_col},
        ) where {F, UM, K_per_col, nblocks_per_col}
            $body
        end
    end
end

function _define_batch_radix_onesweep!(nitem::Int, nchunks::Int)
    name = Symbol("batch_radix_onesweep_", nitem, "_", nchunks, "!")
    fn = Core.eval(@__MODULE__, quote
        $(_build_batch_radix_onesweep_def(name, nitem, nchunks))
        $name
    end)
    Core.eval(@__MODULE__,
        :(@inline get_batch_radix_onesweep_kernel(::Val{$nitem}, ::Val{$nchunks}) = $fn))
    _batch_radix_onesweep_cache[(nitem, nchunks)] = fn
    return fn
end

function get_batch_radix_onesweep_kernel(::Val{nitem}, ::Val{nchunks}
                                         ) where {nitem, nchunks}
    key = (nitem, nchunks)
    haskey(_batch_radix_onesweep_cache, key) && return _batch_radix_onesweep_cache[key]
    return _define_batch_radix_onesweep!(nitem, nchunks)
end


# ── Pre-define common specs ─────────────────────────────────────────────

# Small-K covers items ∈ {1, 2, 4, 8, 16} × npasses ∈ {1, 2, 4, 8} —
# 20 specs that span every (K_PAD ∈ {256..4096}, sizeof(T) ∈ {1, 2, 4, 8})
# combination. Lazy compilation is correct but trips world-age on the
# first call; pre-defining at file load avoids that path.
for ip in (1, 2, 4, 8)
    for items in (1, 2, 4, 8, 16)
        _define_batch_radix_smallK!(items, ip)
    end
end

# Large-K: hist items_per_hist_block = WG*NITEM = 4096.
for np in (1, 2, 4, 8)
    _define_batch_radix_hist!(BATCH_RADIX_NITEM, np, BATCH_RADIX_WG * BATCH_RADIX_NITEM)
end
_define_batch_radix_onesweep!(BATCH_RADIX_NITEM, 1)
_define_batch_radix_onesweep!(BATCH_RADIX_NITEM, 2)


# ── Helpers ─────────────────────────────────────────────────────────────

@inline _next_pow2_at_least(x::Int, lo::Int) = max(lo, 1 << (x <= 1 ? 0 : (sizeof(Int)*8 - leading_zeros(x - 1))))


# ── Workspace ───────────────────────────────────────────────────────────

"""
    get_allocation(SortColumns, A::AbstractMatrix; by=identity, uint_map=KF.uint_map)

Pre-allocate the workspace used by `batched_radix_sort_columns!`.
Sized for `A`'s shape and `uint_map ∘ by`'s output type.
Workspace fields (NamedTuple):

- `scratch :: AbstractMatrix{T}` — ping-pong buffer same shape as `A`.
- `hist    :: AbstractArray{UInt32, 3}` — `(256, npasses, M)` per-pass
  exclusive global histogram per column.
- `partial1`, `partial2 :: AbstractArray{UInt32, 4}` —
  `(256, nblocks_per_col, M, npasses)` decoupled-lookback partials.
- `flag    :: AbstractArray{UInt8, 3}` — `(nblocks_per_col, M, npasses)`.

For columns short enough to use the small-K single-block path
(`K ≤ _batch_radix_smallK_max(T)`) the multi-block buffers are sized 1
along the per-block axis — still allocated so the same workspace works
for any future `A` of compatible shape.
"""
function get_allocation(::Type{SortColumns}, A::AbstractMatrix{T};
                        by::F = identity,
                        uint_map::UM = uint_map) where {T, F, UM}
    K = size(A, 1); M = size(A, 2)
    K_uint = Base.promote_op(uint_map ∘ by, T)
    K_uint <: Unsigned || error("`uint_map ∘ by` must return Unsigned; got $K_uint")
    npasses = sizeof(K_uint)
    bsmax = _batch_radix_block_size_max(T)
    nblocks_per_col = max(1, cld(K, bsmax))
    backend = get_backend(A)
    scratch  = KernelAbstractions.allocate(backend, T,      K, M)
    hist     = KernelAbstractions.allocate(backend, UInt32, BATCH_RADIX_NBUCKETS, npasses, M)
    partial1 = KernelAbstractions.allocate(backend, UInt32, BATCH_RADIX_NBUCKETS, nblocks_per_col, M, npasses)
    partial2 = KernelAbstractions.allocate(backend, UInt32, BATCH_RADIX_NBUCKETS, nblocks_per_col, M, npasses)
    flag     = KernelAbstractions.allocate(backend, UInt8,  nblocks_per_col, M, npasses)
    return KernelBuffer((; scratch, hist, partial1, partial2, flag))
end


# ── Driver: small-K (single-block per column) ───────────────────────────

function _batched_radix_smallK!(A::AbstractMatrix{T},
                                by::F, uint_map::UM) where {T, F, UM}
    K = size(A, 1); M = size(A, 2)
    K_PAD = _next_pow2_at_least(K, BATCH_RADIX_WG)
    shared_bytes = 2 * K_PAD * sizeof(T) + 9 * 1024
    @assert shared_bytes <= 48 * 1024 "small-K path needs ≤ 48 KB shared; K_PAD=$K_PAD T=$T bytes=$shared_bytes"
    items = K_PAD ÷ BATCH_RADIX_WG

    K_uint = Base.promote_op(uint_map ∘ by, T)
    npasses = sizeof(K_uint)

    backend = get_backend(A)
    ker = get_batch_radix_smallK_kernel(Val(items), Val(npasses))
    ker(backend, BATCH_RADIX_WG, M * BATCH_RADIX_WG)(A, by, uint_map, Val(K_PAD))
    return A
end


# ── Driver: large-K (multi-block per column) ────────────────────────────

function _batched_radix_largeK!(A::AbstractMatrix{T}, tmp::KernelBuffer,
                                by::F, uint_map::UM) where {T, F, UM}
    K = size(A, 1); M = size(A, 2)
    K_uint = Base.promote_op(uint_map ∘ by, T)
    npasses = sizeof(K_uint)
    nchunks_T            = _batch_radix_nchunks(T)
    bsmax                = _batch_radix_block_size_max(T)
    nblocks_per_col      = max(1, cld(K, bsmax))
    items_per_hist_block = BATCH_RADIX_WG * BATCH_RADIX_NITEM
    nblocks_per_col_hist = max(1, cld(K, items_per_hist_block))

    backend = get_backend(A)
    scratch  = tmp.arrays.scratch
    hist     = tmp.arrays.hist
    partial1 = tmp.arrays.partial1
    partial2 = tmp.arrays.partial2
    flag     = tmp.arrays.flag

    fill!(hist, UInt32(0))

    # 1) batched bucket histogram
    v_src = vec(A)
    hist_ker = get_batch_radix_hist_kernel(
        Val(BATCH_RADIX_NITEM), Val(npasses), Val(items_per_hist_block))
    hist_ker(backend, BATCH_RADIX_WG, nblocks_per_col_hist * M * BATCH_RADIX_WG)(
        hist, v_src, by, uint_map, Val(K), Val(nblocks_per_col_hist))

    # 2) per-pass exclusive prefix scan over each column (reuse single-vec scan)
    for p in 1:npasses
        view_p = @view hist[:, p, :]
        scan_histogram_kernel!(backend, BATCH_RADIX_NBUCKETS,
                               BATCH_RADIX_NBUCKETS * M)(view_p, Val(BATCH_RADIX_NBUCKETS))
    end

    # 3) per-pass onesweep (Option A: zero partials/flag once upfront).
    fill!(partial1, UInt32(0))
    fill!(partial2, UInt32(0))
    fill!(flag,     UInt8(0))

    a, b = vec(A), vec(scratch)
    onesweep_ker = get_batch_radix_onesweep_kernel(
        Val(BATCH_RADIX_NITEM), Val(nchunks_T))
    inst = onesweep_ker(backend, BATCH_RADIX_WG,
                        nblocks_per_col * M * BATCH_RADIX_WG)
    for pass in 1:npasses
        p1_view   = @view partial1[:, :, :, pass]
        p2_view   = @view partial2[:, :, :, pass]
        fl_view   = @view flag[:, :, pass]
        hist_view = @view hist[:, pass, :]
        inst(b, a, by, uint_map, Int32(8 * (pass - 1)),
             hist_view, p1_view, p2_view, fl_view,
             Val(K), Val(nblocks_per_col))
        a, b = b, a
    end
    if isodd(npasses)
        # Result is in `a` (= vec(scratch) after an odd number of swaps).
        copyto!(vec(A), a)
    end
    return A
end


# ── Public driver ───────────────────────────────────────────────────────

"""
    batched_radix_sort_columns!(A::AbstractMatrix; tmp=nothing,
                                by=identity, uint_map=KF.uint_map) -> A

In-place batched LSD radix sort: each column of `A` is sorted independently
and ascending under `uint_map(by(x))`. ONE kernel launch per stage covers
all columns simultaneously.

Type support: any `T` with a defined `uint_map(::T) <: Unsigned` and a
`typemax(::Type{T})` — i.e. `UInt{8,16,32,64}`, `Int{8,16,32,64}`,
`Float32`, `Float64` out of the box.

K and M may be any positive integers (tail tiles bounds-checked on load
and store).

Use [`get_allocation(SortColumns, A)`](@ref) to preallocate `tmp` and
reuse across calls.
"""
function batched_radix_sort_columns!(A::AbstractMatrix{T};
                                     tmp::Union{Nothing,KernelBuffer} = nothing,
                                     by::F = identity,
                                     uint_map::UM = uint_map) where {T, F, UM}
    K = size(A, 1)
    if K <= _batch_radix_smallK_max(T)
        return _batched_radix_smallK!(A, by, uint_map)
    else
        tmp_ = tmp === nothing ?
               get_allocation(SortColumns, A; by, uint_map) : tmp
        return _batched_radix_largeK!(A, tmp_, by, uint_map)
    end
end
