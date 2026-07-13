# Radix onesweep kernel factory (stable, multi-pass-correct, generic over T).
#
# The kernel body uses `@nexprs` for unrolled phase-1/phase-5a/phase-5b loops,
# which requires the loop count as a literal at parse time. To make Nitem and
# Nchunks tunable, we build the @kernel definition as an `Expr` with literal
# numeric constants substituted into `@nexprs` counts and `@eval` it under a
# uniquely-named function (one per (Nitem, Nchunks) spec). `get_radix_kernel(
# Val(Nitem), Val(Nchunks))` caches the compiled function.
#
# Generic over T: src and dst hold arbitrary T values. The sort key is
# computed as `uint_map(by(x))` — `by` extracts the field to sort by
# (e.g. `first` for tuples), `uint_map` projects to an order-preserving
# unsigned integer (UInt8/16/32/64). The kernel reads/scatters T values and
# only computes the key for digit extraction. Number of byte passes is
# `sizeof(K)` where K is the key type — scheduled by the wrapper, not the
# kernel itself (the kernel just does ONE pass on the byte indicated by
# `shift`).
#
# Stability: phase 1b uses STRIDED loads (lane L's i-th item = src[base +
# (i-1)*warpsz + (L-1)]) instead of contiguous vload, so within a warp the
# src-position order is (c, i, lane) — matching the @nexprs (c, i) time
# order plus HW lane-order ranking inside each step. Per-(digit, warp)
# ranks are src-position-stable, which is required for chained multi-pass
# LSD radix sort.
#
# Memory layouts (must be preserved):
#   partial1, partial2 :: (Nbuckets, nblocks) UInt32 — TRANSPOSED relative
#       to the obvious (nblocks, Nbuckets). Threads write
#       `partial[bucket=lid, gid]` so 256 threads coalesce into 8 cache
#       lines per warp instead of 32 strided lines.
#   global_excl_hist   :: AbstractVector{UInt32} of length Nbuckets — the
#       exclusive prefix of the global byte-histogram for THIS pass's byte.
#       Pass a view (e.g. `@view full_excl_hist[:, p]`) when calling.
#   flag               :: AbstractVector{UInt8} of length nblocks
#
# Shared memory is DYNAMIC (`KI.@dynlocalmem`), not `@localmem`. `@localmem`
# allocates statically, and ptxas caps a static allocation at 48 KB on every
# NVIDIA architecture — which capped `block_size_max`, which capped the tile.
# The dynamic region reaches the real per-block limit (99 KB Ada, 163 KB A100),
# so the tile is now bounded by `KI.max_dynamic_localmem(dev)` instead. The
# three arrays are carved out of ONE flat blob by `onesweep_shared_layout`,
# which is called from the host (to size the launch) and from the kernel (to
# place each array) — never derive the two sides separately.
#
# The workgroup is DECOUPLED from Nbuckets. It used to be pinned to 256 by
# `bucket = lid`, so the only way to grow the tile was to grow items/thread,
# which exploded registers (tile 32768 at wg=256 => 128 items/thread => 255 regs
# and 488 B/thread spilled to local memory). With `has_bucket = lid <= Nbuckets`
# the same tile can be bought at wg=1024 with 32 items/thread, CUB-like and
# spill-free. Only the first `nwg = Nbuckets ÷ warpsz` warps carry a bucket.
#
# NOT a single block-wide histogram, though that would make the histogram
# workgroup-independent: it would BREAK STABILITY. Block-wide atomics hand out
# ranks in ARRIVAL order rather than source order, and an LSD radix sort needs
# every pass to be stable or it destroys the order the previous passes
# established. The per-warp histogram is exactly what makes this kernel stable.
#
# Stable champions on RTX 1000 Ada Laptop (sm_89), UInt32 keys:
#   N = 10M : (Nitem=16, Nchunks=2) ≈ 532 µs/pass
#   N = 100M: (Nitem=16, Nchunks=2) ≈ 4830 µs/pass

using KernelAbstractions
using KernelIntrinsics
using Base.Cartesian: @nexprs
using KernelAbstractions: @atomic


"""
    onesweep_shared_layout(T, Nbuckets, wpb, nwg, block_size_max)

Byte offsets of the three shared arrays inside the kernel's dynamic workgroup
region, plus the total size of that region:

    (off_hist, off_aux, off_sorted, total)

`shared_sorted` is padded up to a 16-byte boundary, which is at least the
alignment of any isbits `T`.

Call this from BOTH the host (`total` sizes the launch) and the device (the
offsets place each array). The host and the device disagreeing about this layout
is the one way to corrupt the kernel silently, so there is exactly one copy of it.
"""
@inline function onesweep_shared_layout(
        ::Type{T}, Nbuckets::Integer, wpb::Integer, nwg::Integer, block_size_max::Integer
    ) where {T}
    off_hist = 0
    off_aux = off_hist + Nbuckets * wpb * sizeof(UInt32)
    off_sorted = cld(off_aux + (Nbuckets + nwg) * sizeof(UInt32), 16) * 16
    return (off_hist, off_aux, off_sorted, off_sorted + block_size_max * sizeof(T))
end

"""
    onesweep_shmem(T, Nbuckets, warpsz, workgroup, Nchunks, Nitem) -> Int

Bytes of dynamic workgroup memory one onesweep block needs for this launch
configuration. Compare against `KI.max_dynamic_localmem(dev)` before launching;
`KI.launch!` throws (catchably) if it does not fit.
"""
@inline function onesweep_shmem(
        ::Type{T}, Nbuckets::Integer, warpsz::Integer, workgroup::Integer,
        Nchunks::Integer, Nitem::Integer
    ) where {T}
    wpb = workgroup ÷ warpsz
    nwg = Nbuckets ÷ warpsz
    return Int(onesweep_shared_layout(T, Nbuckets, wpb, nwg, workgroup * Nchunks * Nitem)[4])
end


# Cache: (Nitem, Nchunks) → compiled @kernel function object.
const _onesweep_kernel_cache = Dict{Tuple{Int,Int},Any}()


function build_kernel_def(name::Symbol, Nitem::Int, Nchunks::Int)
    # All numeric constants below get interpolated as LITERALS into @nexprs
    # counts and Val(...) — so the @kernel macro sees plain @nexprs Nc/Ni
    # forms, identical to a hand-written hardcoded body.
    Niter_5b = Nchunks * Nitem

    body = quote
        @uniform begin
            N = length(src)
            digit_mask = UInt32(Nbuckets - 1)
            T = eltype(dst)
            workgroup = wpb * warpsz
            block_size_max = wpb * $Nchunks * warpsz * $Nitem
            # Bucket-carrying warps. NOT `wpb`: buckets are spread over the
            # first Nbuckets THREADS, so they occupy Nbuckets ÷ warpsz warps
            # regardless of how many warps the block has.
            nwg = Nbuckets ÷ warpsz
            # One flat dynamic region, carved by the SAME layout function the
            # host calls to size `shmem`. See onesweep_shared_layout.
            (off_hist, off_aux, off_sorted, _) =
                onesweep_shared_layout(T, Nbuckets, wpb, nwg, block_size_max)
        end

        lid = Int(@index(Local))
        gid = Int(@index(Group))
        warp_id = (lid - 1) ÷ warpsz + 1
        lane = (lid - 1) % warpsz + 1
        global_warp = (gid - 1) * wpb + warp_id

        # Workgroup decoupling: at workgroup > Nbuckets the trailing threads
        # carry no bucket. They still load, rank and scatter items — they just
        # take no part in the per-bucket scan, the aggregate publish, or the
        # lookback. `bucket` is clamped to 1 so every index stays in range; the
        # `has_bucket` guard is what keeps those threads from writing anything.
        has_bucket = lid <= Nbuckets
        bucket = has_bucket ? lid : 1

        # Position-based addressing (strided load). Lane L's i-th item in
        # chunk c is at src position:
        #   pos = warp_first_pos + (c-1)*warpsz*Nitem + (i-1)*warpsz + (L-1)
        warp_first_pos = (global_warp - 1) * $Nchunks * warpsz * $Nitem + 1

        block_first_pos_1 = (gid - 1) * block_size_max + 1
        block_last_pos_1 = min(gid * block_size_max, N)
        block_size_actual = max(0, block_last_pos_1 - block_first_pos_1 + 1)

        shared_hist = KernelIntrinsics.@dynlocalmem UInt32 (Nbuckets, wpb) off_hist
        # shared_aux: scratch for warp_totals[1..nwg] and block_digit_start[1..Nbuckets].
        # In v19/v22 these aliased into the front of shared_sorted (then UInt32),
        # which broke once shared_sorted started carrying T-typed values (an
        # InexactError if T is narrower than UInt32). Costs ~1 KB extra shared.
        shared_aux = KernelIntrinsics.@dynlocalmem UInt32 (Nbuckets + nwg,) off_aux
        shared_sorted = KernelIntrinsics.@dynlocalmem T (block_size_max,) off_sorted

        is_full_tile = gid * block_size_max <= N

        # Phase 1a: zero shared_hist. Iteration count is Nbuckets ÷ warpsz —
        # 8 on NVIDIA (warpsz=32), 4 on AMD MI300X (warpsz=64). Hardcoding 8
        # writes 8×warpsz=512 slots into a 256-slot tile on AMD and corrupts
        # shared_aux/shared_sorted. The compiler unrolls this fully since
        # both bounds are Val-known.
        for s in 1:(Nbuckets ÷ warpsz)
            b_init = lane + (s - 1) * warpsz
            shared_hist[b_init, warp_id] = UInt32(0)
        end
        @synchronize

        # Phase 1b: STRIDED per-item load + atomic-add histogram.
        # Each lane loads Nitem items at stride warpsz from `chunk_base`. Atomic
        # OLD value gives rank, src-stable: within (digit, warp), items receive
        # ranks in src-position order (= the @nexprs (c, i, lane) time order).
        if is_full_tile
            @nexprs $Nchunks c -> begin
                chunk_base_c = warp_first_pos + (c - 1) * warpsz * $Nitem
                @nexprs $Nitem i -> begin
                    pos_c_i = chunk_base_c + (i - 1) * warpsz + (lane - 1)
                    item_c_i = src[pos_c_i]
                    key_c_i = uint_map(by(item_c_i))
                    d_c_i = Int((key_c_i >> shift) & digit_mask) + 1
                    # @atomic returns the NEW value; subtract 1 to get OLD = rank.
                    rank_c_i = (@atomic shared_hist[d_c_i, warp_id] += UInt32(1)) - UInt32(1)
                end
            end
        else
            @nexprs $Nchunks c -> begin
                chunk_base_c = warp_first_pos + (c - 1) * warpsz * $Nitem
                @nexprs $Nitem i -> begin
                    pos_c_i = chunk_base_c + (i - 1) * warpsz + (lane - 1)
                    in_bounds_c_i = pos_c_i <= N
                    item_c_i = in_bounds_c_i ? src[pos_c_i] : src[N]
                    if in_bounds_c_i
                        key_c_i = uint_map(by(item_c_i))
                        d_c_i = Int((key_c_i >> shift) & digit_mask) + 1
                        rank_c_i = (@atomic shared_hist[d_c_i, warp_id] += UInt32(1)) - UInt32(1)
                    else
                        rank_c_i = UInt32(0)
                    end
                end
            end
        end
        @synchronize

        # Phase 2: per-bucket exclusive scan over warps. Iterates over wpb
        # warps (8 on NVIDIA workgroup=256/warpsz=32, 4 on AMD warpsz=64).
        block_total_b = UInt32(0)
        if has_bucket
            prefix_acc = UInt32(0)
            for w in 1:wpb
                v = shared_hist[bucket, w]
                shared_hist[bucket, w] = prefix_acc
                prefix_acc += v
            end
            block_total_b = prefix_acc
        end

        # Phase 3a: publish aggregate (TRANSPOSED layout for coalescing).
        if has_bucket
            @access partial1[bucket, gid] = block_total_b
            @access partial2[bucket, gid] = block_total_b
        end
        @synchronize
        if lid == 1
            @access flag[gid] = 0x01
        end

        # Phase 4.5: exclusive scan of the Nbuckets block totals, which live in
        # threads 1..Nbuckets, i.e. in warps 1..nwg. The guard is warp-uniform,
        # so the shuffles inside @warpreduce see a converged warp.
        val_exc_within = UInt32(0)
        if warp_id <= nwg
            val_inc = block_total_b
            @warpreduce(val_inc, +, lane, warpsz)
            val_exc_within = val_inc - block_total_b
            if lane == warpsz
                shared_aux[Nbuckets + warp_id] = val_inc
            end
        end
        @synchronize

        if warp_id == 1
            wt = lane <= nwg ? shared_aux[Nbuckets + lane] : UInt32(0)
            inc_v = wt
            @warpreduce(inc_v, +, lane, nwg)
            excl_v = inc_v - wt
            if lane <= nwg
                shared_aux[Nbuckets + lane] = excl_v
            end
        end
        @synchronize

        own_bds_reg = UInt32(0)
        if has_bucket
            warp_prefix = shared_aux[Nbuckets + warp_id]
            own_bds_reg = warp_prefix + val_exc_within
            shared_aux[bucket] = own_bds_reg
        end
        @synchronize

        # Phase 4 first: write warp_block_local_base into shared_hist.
        # Same Nbuckets÷warpsz iteration count as Phase 1a.
        for s in 1:(Nbuckets ÷ warpsz)
            b_seed = lane + (s - 1) * warpsz
            my_excl_inline = shared_hist[b_seed, warp_id]
            warp_block_local_base = shared_aux[b_seed] + my_excl_inline
            shared_hist[b_seed, warp_id] = warp_block_local_base
        end
        @synchronize

        # Phase 5a: shuffle to shared_sorted. Bounds check uses the same strided
        # position formula as phase 1b.
        #
        # RR ("re-read"): re-load the item from `src` instead of carrying it in a
        # register from phase 1b. `src` is not written during a pass, so the value
        # is identical. It trades a second (already-cached) load for roughly half
        # the live values, which is what stops the register allocator spilling at
        # high items/thread. It wins on 4-byte keys and loses on 8-byte ones, so
        # it is a tuning knob, not a default.
        if is_full_tile
            @nexprs $Nchunks c -> begin
                chunk_base_5a_c = warp_first_pos + (c - 1) * warpsz * $Nitem
                @nexprs $Nitem i -> begin
                    pos_5a_c_i = chunk_base_5a_c + (i - 1) * warpsz + (lane - 1)
                    it_5a_c_i = RR ? src[pos_5a_c_i] : item_c_i
                    key_5a_c_i = uint_map(by(it_5a_c_i))
                    d_5a_c_i = Int((key_5a_c_i >> shift) & digit_mask) + 1
                    block_local_pos_c_i = shared_hist[d_5a_c_i, warp_id] + rank_c_i
                    shared_sorted[Int(block_local_pos_c_i) + 1] = it_5a_c_i
                end
            end
        else
            @nexprs $Nchunks c -> begin
                chunk_base_5a_c = warp_first_pos + (c - 1) * warpsz * $Nitem
                @nexprs $Nitem i -> begin
                    pos_5a_c_i = chunk_base_5a_c + (i - 1) * warpsz + (lane - 1)
                    if pos_5a_c_i <= N
                        it_5a_c_i = RR ? src[pos_5a_c_i] : item_c_i
                        key_5a_c_i = uint_map(by(it_5a_c_i))
                        d_5a_c_i = Int((key_5a_c_i >> shift) & digit_mask) + 1
                        block_local_pos_c_i = shared_hist[d_5a_c_i, warp_id] + rank_c_i
                        shared_sorted[Int(block_local_pos_c_i) + 1] = it_5a_c_i
                    end
                end
            end
        end
        @synchronize

        # Phase 3b: warp-specialized lookback (warp 1 spins, others wait at sync).
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

                # `flg_val` / `b_val` are block-uniform (every thread just read the
                # same slot behind a barrier), so `contains_prefix` is uniform too
                # and the loop trip count is the same for every thread — which is
                # what makes the @synchronize below legal.
                if flg_val == UInt32(0x02)
                    if has_bucket
                        @access pval = partial2[bucket, b_val]
                        cross_block_prefix += pval
                    end
                    contains_prefix = true
                else
                    if has_bucket
                        @access pval = partial1[bucket, b_val]
                        cross_block_prefix += pval
                    end
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

        if has_bucket
            @access partial2[bucket, gid] = cross_block_prefix + block_total_b
        end
        @synchronize
        if lid == 1
            @access flag[gid] = 0x02
        end

        # Phase 4 second: write block_global_offset[d] into shared_hist[d, 1].
        if has_bucket
            shared_hist[bucket, 1] = global_excl_hist[bucket] + cross_block_prefix - own_bds_reg
        end
        @synchronize

        # Phase 5b: stream-out. Recompute the key from the staged value
        # (re-applying `by` and `uint_map`) — single bit-twiddle for the
        # built-in `uint_map` definitions, negligible cost.
        @nexprs $Niter_5b c -> begin
            p_c = (c - 1) * (wpb * warpsz) + lid
            if p_c <= block_size_actual
                item_5b_c = shared_sorted[p_c]
                key_5b_c = uint_map(by(item_5b_c))
                d_5b_c = Int((key_5b_c >> shift) & digit_mask) + 1
                dst[Int(shared_hist[d_5b_c, 1]) + p_c] = item_5b_c
            end
        end
    end

    return quote
        @kernel inbounds = true unsafe_indices = true function $(name)(
            dst::AbstractVector,
            src::AbstractVector,
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
            ::Val{RR},
        ) where {F,M,Nbuckets,warpsz,wpb,RR}
            $body
        end
    end
end


function _define_kernel!(Nitem::Int, Nchunks::Int)
    name = Symbol("onesweep_kernel_", Nitem, "_", Nchunks, "!")
    # Define the kernel AND retrieve it in the same Core.eval. The trailing
    # `$name` evaluates the freshly-defined binding in the latest world,
    # avoiding the world-age warning a separate `getfield` would trigger.
    fn = Core.eval(@__MODULE__, quote
        $(build_kernel_def(name, Nitem, Nchunks))
        $name
    end)
    # Install a typed dispatch method whose body returns `fn` as a literal
    # singleton — no global-binding access at any point in the call chain.
    Core.eval(@__MODULE__,
        :(@inline get_radix_kernel(::Val{$Nitem}, ::Val{$Nchunks}) = $fn))
    _onesweep_kernel_cache[(Nitem, Nchunks)] = fn
    return fn
end


# Generic fallback for exotic (Nitem, Nchunks). Dict short-circuit
# prevents a second @eval if the typed method is not yet visible
# (it was installed in a world that hasn't propagated to our caller).
function get_radix_kernel(::Val{Nitem}, ::Val{Nchunks}) where {Nitem,Nchunks}
    key = (Nitem, Nchunks)
    haskey(_onesweep_kernel_cache, key) && return _onesweep_kernel_cache[key]
    return _define_kernel!(Nitem, Nchunks)
end


# Pre-compile the two default specs at package load. We reuse
# `_define_kernel!`, which both defines the @kernel and installs the
# typed `Val` dispatch method, so the default code path needs neither
# @eval nor invokelatest at runtime.
_define_kernel!(16, 1)
_define_kernel!(16, 2)
