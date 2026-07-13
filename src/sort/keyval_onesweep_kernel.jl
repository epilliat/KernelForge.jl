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
        @access partial1[bucket, gid] = block_total_b
        @access partial2[bucket, gid] = block_total_b
        @synchronize
        if lid == 1
            @access flag[gid] = 0x01
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

        # Phase 3b: warp-specialized lookback.
        if lid == 1
            shared_hist[1, 1] = UInt32(gid - 1) << 8
        end
        @synchronize

        cross_block_prefix = UInt32(0)
        if gid >= 2
            contains_prefix = false
            while !contains_prefix
                if warp_id == 1 && lane == 1
                    comb = shared_hist[1, 1]
                    local_b = Int(comb >> 8)
                    local_flg = UInt32(0)
                    while local_flg == UInt32(0)
                        @access tmp_flg = flag[local_b]
                        local_flg = UInt32(tmp_flg)
                    end
                    shared_hist[1, 1] = (UInt32(local_b) << 8) | local_flg
                end
                @synchronize

                comb = shared_hist[1, 1]
                flg_val = comb & UInt32(0xFF)
                b_val = Int(comb >> 8)
                # Every thread must finish reading shared_hist[1, 1] before the
                # `lid == 1` store below overwrites it with the next block index.
                # Without this barrier a fast warp clobbers the slot while a slow
                # warp is still loading it: the warps then disagree on `b_val` and
                # on `contains_prefix`, which both corrupts the prefix and makes the
                # trailing `@synchronize` divergent.
                @synchronize

                if flg_val == UInt32(0x02)
                    @access pval = partial2[bucket, b_val]
                    cross_block_prefix += pval
                    contains_prefix = true
                else
                    @access pval = partial1[bucket, b_val]
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

        @access partial2[bucket, gid] = cross_block_prefix + block_total_b
        @synchronize
        if lid == 1
            @access flag[gid] = 0x02
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
        ) where {F,M,Nbuckets,warpsz,wpb}
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
