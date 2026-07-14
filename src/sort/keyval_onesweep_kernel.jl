# Key-value variant of the radix onesweep kernel.
#
# Given two parallel arrays `src` (values) and `src_keys` (keys), this kernel
# sorts BOTH arrays together using `uint_map(by(src_keys[i]))` as the digit
# for src[i]. After all passes, the items appear in dst in ascending key
# order (stable), and the keys themselves end up in dst_keys in the same
# permuted order.
#
# Diff vs onesweep_kernel.jl:
#   - Two extra arguments: dst_keys and src_keys.
#   - Phase 1b reads BOTH src[pos] and src_keys[pos]; the digit comes from
#     the KEY (not the value).
#   - Phase 5a stages two parallel buffers in shared mem:
#       shared_sorted[block_local_pos] = src_value
#       shared_keys[block_local_pos]   = src_key
#   - Phase 5b streams BOTH buffers out at the same offset; digit is
#     recomputed from the staged key.
#
# Shared budget: extra `block_size_max * sizeof(KT)` bytes for shared_keys.
# Compared to the value-only onesweep, default Nchunks is HALVED for keyval
# to stay inside the 48 KB static-shared budget. See
# `default_nchunks_keyval(...)` in sort1d.jl for the per-(T,KT) defaults.
#
# Stability: same as plain onesweep — strided phase-1b loads + decoupled-
# lookback in phase 3b give globally stable LSD radix.
#
# Memory layouts (must be preserved):
#   partial1, partial2 :: (Nbuckets, nblocks) UInt32 — TRANSPOSED relative
#       to the obvious (nblocks, Nbuckets). Threads write
#       `partial[bucket=lid, gid]` so 256 threads coalesce into 8 cache
#       lines per warp instead of 32 strided lines.
#   global_excl_hist   :: AbstractVector{UInt32} of length Nbuckets — the
#       exclusive prefix of the global byte-histogram for THIS pass's byte.
#   flag               :: AbstractVector{UInt8} of length nblocks

using KernelAbstractions
using KernelIntrinsics
using Base.Cartesian: @nexprs
using KernelAbstractions: @atomic


const _keyval_onesweep_kernel_cache = Dict{Tuple{Int,Int},Any}()


function build_keyval_kernel_def(name::Symbol, Nitem::Int, Nchunks::Int)
    Niter_5b = Nchunks * Nitem
    body = quote
        @uniform begin
            N = length(src)
            digit_mask = UInt32(Nbuckets - 1)
            T  = eltype(dst)
            KT = eltype(dst_keys)
            block_size_max = wpb * $Nchunks * warpsz * $Nitem
        end

        lid = Int(@index(Local))
        gid = Int(@index(Group))
        warp_id = (lid - 1) ÷ warpsz + 1
        lane    = (lid - 1) % warpsz + 1
        global_warp = (gid - 1) * wpb + warp_id

        warp_first_pos = (global_warp - 1) * $Nchunks * warpsz * $Nitem + 1

        block_first_pos_1 = (gid - 1) * block_size_max + 1
        block_last_pos_1  = min(gid * block_size_max, N)
        block_size_actual = max(0, block_last_pos_1 - block_first_pos_1 + 1)

        shared_hist   = @localmem UInt32 (Nbuckets, wpb)
        shared_aux    = @localmem UInt32 (Nbuckets + wpb,)
        shared_sorted = @localmem T  (block_size_max,)
        shared_keys   = @localmem KT (block_size_max,)

        is_full_tile = gid * block_size_max <= N

        # Phase 1a: zero shared_hist.
        for s in 1:(Nbuckets ÷ warpsz)
            b_init = lane + (s - 1) * warpsz
            shared_hist[b_init, warp_id] = UInt32(0)
        end
        @synchronize

        # Phase 1b: STRIDED per-item load + atomic-add histogram.
        # Read item AND key. Digit comes from the KEY.
        if is_full_tile
            @nexprs $Nchunks c -> begin
                chunk_base_c = warp_first_pos + (c - 1) * warpsz * $Nitem
                @nexprs $Nitem i -> begin
                    pos_c_i = chunk_base_c + (i - 1) * warpsz + (lane - 1)
                    item_c_i  = src[pos_c_i]
                    keyit_c_i = src_keys[pos_c_i]
                    key_c_i = uint_map(by(keyit_c_i))
                    d_c_i = Int((key_c_i >> shift) & digit_mask) + 1
                    rank_c_i = (@atomic shared_hist[d_c_i, warp_id] += UInt32(1)) - UInt32(1)
                end
            end
        else
            @nexprs $Nchunks c -> begin
                chunk_base_c = warp_first_pos + (c - 1) * warpsz * $Nitem
                @nexprs $Nitem i -> begin
                    pos_c_i = chunk_base_c + (i - 1) * warpsz + (lane - 1)
                    in_bounds_c_i = pos_c_i <= N
                    item_c_i  = in_bounds_c_i ? src[pos_c_i]      : src[N]
                    keyit_c_i = in_bounds_c_i ? src_keys[pos_c_i] : src_keys[N]
                    if in_bounds_c_i
                        key_c_i = uint_map(by(keyit_c_i))
                        d_c_i = Int((key_c_i >> shift) & digit_mask) + 1
                        rank_c_i = (@atomic shared_hist[d_c_i, warp_id] += UInt32(1)) - UInt32(1)
                    else
                        rank_c_i = UInt32(0)
                    end
                end
            end
        end
        @synchronize

        # Phase 2: per-bucket exclusive scan over warps.
        bucket = lid
        prefix_acc = UInt32(0)
        for w in 1:wpb
            v = shared_hist[bucket, w]
            shared_hist[bucket, w] = prefix_acc
            prefix_acc += v
        end
        block_total_b = prefix_acc

        # Phase 3a: publish aggregate (transposed layout).
        # `Bypass` (AMD): see sort_desc_bypass. The release fence runs on EVERY thread
        # BEFORE the barrier — vmcnt is per-wave, so fencing on lid==1 alone would not
        # drain the partial stores issued by the other waves.
        if Bypass
            @access Device Relaxed partial1[bucket, gid] = block_total_b
            @access Device Relaxed partial2[bucket, gid] = block_total_b
            @fence Workgroup Release
        else
            @access partial1[bucket, gid] = block_total_b
            @access partial2[bucket, gid] = block_total_b
        end
        @synchronize
        if lid == 1
            if Bypass
                @access Device Relaxed flag[gid] = 0x01
            else
                @access flag[gid] = 0x01
            end
        end

        # Phase 4.5: warp-totals reduction.
        val_inc = block_total_b
        @warpreduce(val_inc, +, lane, warpsz)
        val_exc_within = val_inc - block_total_b

        if lane == warpsz
            shared_aux[Nbuckets + warp_id] = val_inc
        end
        @synchronize

        if warp_id == 1
            wt = lane <= wpb ? shared_aux[Nbuckets + lane] : UInt32(0)
            inc_v = wt
            @warpreduce(inc_v, +, lane, wpb)
            excl_v = inc_v - wt
            if lane <= wpb
                shared_aux[Nbuckets + lane] = excl_v
            end
        end
        @synchronize

        warp_prefix = shared_aux[Nbuckets + warp_id]
        own_bds_reg = warp_prefix + val_exc_within
        shared_aux[bucket] = own_bds_reg
        @synchronize

        # Phase 4 first.
        for s in 1:(Nbuckets ÷ warpsz)
            b_seed = lane + (s - 1) * warpsz
            my_excl_inline = shared_hist[b_seed, warp_id]
            warp_block_local_base = shared_aux[b_seed] + my_excl_inline
            shared_hist[b_seed, warp_id] = warp_block_local_base
        end
        @synchronize

        # Phase 5a: shuffle BOTH item and key into shared.
        if is_full_tile
            @nexprs $Nchunks c -> begin
                @nexprs $Nitem i -> begin
                    key_5a_c_i = uint_map(by(keyit_c_i))
                    d_5a_c_i = Int((key_5a_c_i >> shift) & digit_mask) + 1
                    block_local_pos_c_i = shared_hist[d_5a_c_i, warp_id] + rank_c_i
                    shared_sorted[Int(block_local_pos_c_i) + 1] = item_c_i
                    shared_keys[Int(block_local_pos_c_i) + 1]   = keyit_c_i
                end
            end
        else
            @nexprs $Nchunks c -> begin
                chunk_base_5a_c = warp_first_pos + (c - 1) * warpsz * $Nitem
                @nexprs $Nitem i -> begin
                    pos_5a_c_i = chunk_base_5a_c + (i - 1) * warpsz + (lane - 1)
                    if pos_5a_c_i <= N
                        key_5a_c_i = uint_map(by(keyit_c_i))
                        d_5a_c_i = Int((key_5a_c_i >> shift) & digit_mask) + 1
                        block_local_pos_c_i = shared_hist[d_5a_c_i, warp_id] + rank_c_i
                        shared_sorted[Int(block_local_pos_c_i) + 1] = item_c_i
                        shared_keys[Int(block_local_pos_c_i) + 1]   = keyit_c_i
                    end
                end
            end
        end
        @synchronize

        # Phase 3b: decoupled lookback, BARRIER-FREE.
        #
        # Each of the Nbuckets bucket-carrying threads walks back on its own. They all
        # poll the SAME flag[b], so `flg` and `b` advance identically across them: the
        # loop is warp-uniform BY CONSTRUCTION and needs no barrier, no shared slot,
        # and no descriptor hand-off. The 256 threads polling one address coalesce into
        # a single request.
        #
        # The previous version centralised the poll on one lane and published through
        # shared_hist[1, 1], which cost TWO block barriers per step of the walk -- and
        # at wg=512/1024 a barrier is expensive. Measured on A100: -2.8% to -8.6% end
        # to end. It also removes the read-then-overwrite race on shared_hist[1, 1]
        # (fixed in 4a6848d) at the source: there is no shared descriptor left to race on.
        #
        # This kernel is NOT workgroup-decoupled (bucket = lid, workgroup == Nbuckets),
        # so every thread carries a bucket and takes part in the walk.
        cross_block_prefix = UInt32(0)
        if gid >= 2
            b_lb = gid - 1
            done_lb = false
            while !done_lb
                flg_lb = UInt32(0)
                while flg_lb == UInt32(0)
                    # Device-scope RELAXED (Bypass): coherent, but no per-round acquire.
                    # That acquire is what made this walk 66-68% of the sort on gfx942.
                    f_lb = Bypass ? (@access Device Relaxed flag[b_lb]) :
                                    (@access flag[b_lb])        # Device Acquire
                    flg_lb = UInt32(f_lb)
                end
                if Bypass
                    @fence Workgroup Acquire   # once per FOUND block, not per spin
                end
                if flg_lb == UInt32(0x02)
                    # inclusive prefix published -> the walk ends here
                    pv_lb = Bypass ? (@access Device Relaxed partial2[bucket, b_lb]) :
                                     (@access partial2[bucket, b_lb])
                    cross_block_prefix += pv_lb
                    done_lb = true
                else
                    pv_lb = Bypass ? (@access Device Relaxed partial1[bucket, b_lb]) :
                                     (@access partial1[bucket, b_lb])
                    cross_block_prefix += pv_lb
                    if b_lb == 1
                        done_lb = true
                    else
                        b_lb -= 1
                    end
                end
            end
        end
        @synchronize

        if Bypass
            @access Device Relaxed partial2[bucket, gid] = cross_block_prefix + block_total_b
            @fence Workgroup Release   # per-wave drain, BEFORE the barrier
        else
            @access partial2[bucket, gid] = cross_block_prefix + block_total_b
        end
        @synchronize
        if lid == 1
            if Bypass
                @access Device Relaxed flag[gid] = 0x02
            else
                @access flag[gid] = 0x02
            end
        end

        # Phase 4 second.
        shared_hist[bucket, 1] = global_excl_hist[bucket] + cross_block_prefix - own_bds_reg
        @synchronize

        # Phase 5b: stream out BOTH dst and dst_keys at the same offset.
        @nexprs $Niter_5b c -> begin
            p_c = (c - 1) * (wpb * warpsz) + lid
            if p_c <= block_size_actual
                item_5b_c   = shared_sorted[p_c]
                keyit_5b_c  = shared_keys[p_c]
                key_5b_c = uint_map(by(keyit_5b_c))
                d_5b_c = Int((key_5b_c >> shift) & digit_mask) + 1
                idx_5b_c = Int(shared_hist[d_5b_c, 1]) + p_c
                dst[idx_5b_c]      = item_5b_c
                dst_keys[idx_5b_c] = keyit_5b_c
            end
        end
    end

    return quote
        @kernel inbounds = true unsafe_indices = true function $(name)(
            dst::AbstractVector,
            src::AbstractVector,
            dst_keys::AbstractVector,
            src_keys::AbstractVector,
            by::F,
            uint_map::M,
            shift::Int32,
            global_excl_hist::AbstractVector{UInt32},
            partial1::AbstractMatrix{UInt32},
            partial2::AbstractMatrix{UInt32},
            flag::AbstractVector{UInt8},
            ::Val{Nbuckets},
            ::Val{warpsz},
            ::Val{wpb},
            ::Val{Bypass},
        ) where {F,M,Nbuckets,warpsz,wpb,Bypass}
            $body
        end
    end
end


function _define_keyval_kernel!(Nitem::Int, Nchunks::Int)
    name = Symbol("keyval_onesweep_kernel_", Nitem, "_", Nchunks, "!")
    fn = Core.eval(@__MODULE__, quote
        $(build_keyval_kernel_def(name, Nitem, Nchunks))
        $name
    end)
    Core.eval(@__MODULE__,
        :(@inline get_keyval_kernel(::Val{$Nitem}, ::Val{$Nchunks}) = $fn))
    _keyval_onesweep_kernel_cache[(Nitem, Nchunks)] = fn
    return fn
end


# Generic fallback for exotic specs.
function get_keyval_kernel(::Val{Nitem}, ::Val{Nchunks}) where {Nitem,Nchunks}
    key = (Nitem, Nchunks)
    haskey(_keyval_onesweep_kernel_cache, key) && return _keyval_onesweep_kernel_cache[key]
    return _define_keyval_kernel!(Nitem, Nchunks)
end


# Pre-compile the default specs at package load. Default Nchunks for keyval
# is half of the value-only sort to keep shared_sorted + shared_keys within
# the 48 KB budget.
_define_keyval_kernel!(16, 1)   # default for sizeof(T)+sizeof(KT) ≤ 8 (e.g. UInt32+UInt32)
_define_keyval_kernel!(8,  1)   # default for sizeof(T)+sizeof(KT) ≤ 16 (e.g. Float64+UInt32)
