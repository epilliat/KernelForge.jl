# Specialized one-pass scatter for byte-wide keys (sizeof(K) == 1).
#
# When the radix key fits in a single byte (UInt8 / Int8 / Bool plus any T
# whose `uint_map ∘ by` lands in 0..255), full LSD radix degenerates to a
# single byte pass. We don't need decoupled-lookback: block ordering only
# matters across blocks for the SAME byte, so a per-block atomic-add into a
# 256-slot global counter (initialized to the global exclusive prefix)
# claims a unique destination range per (block, byte). No partial1/2/flag,
# no warp-specialized spin loop. ~25% faster than the 1-pass onesweep on
# UInt8 inputs.
#
# Pipeline (drives from sort! with sizeof(K) == 1):
#   1. bucket_histogram_kernel! → hist :: (256, 1) UInt32
#   2. scan_histogram_kernel!   → hist holds exclusive prefix
#   3. byte_sort_kernel_<Nitem> → reads hist as `global_counter` (mutated to
#      cumulative running offsets), writes sorted dst.
#
# Same factory pattern as onesweep_kernel: the kernel body uses `@nexprs $Nitem`
# which requires Nitem as a literal at parse time, so we build the @kernel def
# as an Expr and `@eval` it under a uniquely-named function per Nitem value.

using KernelAbstractions
using KernelIntrinsics
using Base.Cartesian: @nexprs
using KernelAbstractions: @atomic


# Cache: Nitem → compiled @kernel function object.
const _byte_sort_kernel_cache = Dict{Int, Any}()


function build_byte_sort_kernel_def(name::Symbol, Nitem::Int)
    body = quote
        @uniform begin
            N = length(src)
            workgroup = Int(@groupsize()[1])
            wpb = workgroup ÷ warpsz
            block_size_max = workgroup * $Nitem
            T = eltype(dst)
        end

        lid = Int(@index(Local))
        gid = Int(@index(Group))
        warp_id = (lid - 1) ÷ warpsz + 1
        lane = (lid - 1) % warpsz + 1
        global_warp = (gid - 1) * wpb + warp_id

        # Strided load: lane L's i-th item at src[base + (i-1)*warpsz + (L-1)].
        # Same pattern as onesweep_kernel — gives src-stable per-(byte, block)
        # rank under HW-compaction-of-atomic-add. Stability among same-byte
        # items doesn't matter for radix sort with a 1-byte key (same key →
        # same value for UInt8), but the strided pattern is what makes the
        # @nexprs (i, lane) order match src order, which lets the per-warp
        # scatter into shared_sorted produce a contiguous-by-byte arrangement
        # for the coalesced stream-out in phase 5.
        warp_first_pos = (global_warp - 1) * $Nitem * warpsz + 1
        block_first_pos = (gid - 1) * block_size_max + 1
        block_last_pos = min(gid * block_size_max, N)
        block_size_actual = max(0, block_last_pos - block_first_pos + 1)

        shared_hist    = @localmem UInt32 (Nbuckets,)   # block-local count of each byte
        shared_starts  = @localmem UInt32 (Nbuckets,)   # block-local exclusive prefix
        shared_dst_off = @localmem UInt32 (Nbuckets,)   # global-offset minus block-local-offset (combined)
        shared_sorted  = @localmem T      (block_size_max,)

        # Phase 0: zero shared_hist (one entry per thread, since wg == Nbuckets).
        bucket = lid
        if bucket <= Nbuckets
            shared_hist[bucket] = UInt32(0)
        end
        @synchronize

        # Phase 1: strided load + per-byte atomic-add to capture ranks.
        @nexprs $Nitem i -> begin
            pos_i = warp_first_pos + (i - 1) * warpsz + (lane - 1)
            in_bounds_i = pos_i <= N
            item_i = in_bounds_i ? src[pos_i] : src[N]
            if in_bounds_i
                d_i = Int(uint_map(by(item_i))) + 1
                rank_i = (@atomic shared_hist[d_i] += UInt32(1)) - UInt32(1)
            else
                d_i = 1
                rank_i = UInt32(0)
            end
        end
        @synchronize

        # Phase 2: block-local exclusive prefix scan over shared_hist
        # (Hillis-Steele, log2(Nbuckets) steps). Result lands in shared_starts.
        # Each thread owns one bucket (workgroup == Nbuckets).
        if bucket <= Nbuckets
            shared_starts[bucket] = shared_hist[bucket]
        end
        @synchronize

        offset = 1
        while offset < Nbuckets
            other = (bucket > offset && bucket <= Nbuckets) ?
                    shared_starts[bucket - offset] : UInt32(0)
            @synchronize
            if bucket <= Nbuckets
                shared_starts[bucket] += other
            end
            @synchronize
            offset <<= 1
        end
        # shared_starts now holds INCLUSIVE prefix; convert to exclusive.
        if bucket <= Nbuckets
            shared_starts[bucket] -= shared_hist[bucket]
        end
        @synchronize

        # Phase 3: re-arrange items in shared_sorted, sorted by byte.
        # Position within shared_sorted (1-indexed) = shared_starts[d] + rank + 1.
        @nexprs $Nitem i -> begin
            if in_bounds_i
                local_pos_i = Int(shared_starts[d_i]) + Int(rank_i) + 1
                shared_sorted[local_pos_i] = item_i
            end
        end
        @synchronize

        # Phase 4: claim a destination range per byte via global atomic-add.
        # `global_counter[bucket]` starts at the GLOBAL exclusive prefix of
        # byte (bucket-1) (i.e. `hist[bucket, 1]` after scan_histogram_kernel).
        # `(@atomic counter += count) - count` returns the OLD value = the
        # 0-indexed dst start for this block's byte-(bucket-1) items.
        # Combine with the within-block start so phase 5 writes via a
        # single offset (analogous to onesweep's phase 4 second).
        if bucket <= Nbuckets
            count_b = shared_hist[bucket]
            if count_b > UInt32(0)
                old = (@atomic global_counter[bucket] += count_b) - count_b
                # Combined offset: dst[shared_dst_off[d] + p] = item, with p the
                # 1-indexed shared_sorted position. UInt32 underflow is OK —
                # modular arithmetic recovers the right result on add-back.
                shared_dst_off[bucket] = old - shared_starts[bucket]
            else
                shared_dst_off[bucket] = UInt32(0)   # unused
            end
        end
        @synchronize

        # Phase 5: coalesced stream-out from shared_sorted to dst.
        @nexprs $Nitem c -> begin
            p_c = (c - 1) * workgroup + lid
            if p_c <= block_size_actual
                item_5_c = shared_sorted[p_c]
                d_5_c = Int(uint_map(by(item_5_c))) + 1
                dst[Int(shared_dst_off[d_5_c]) + p_c] = item_5_c
            end
        end
    end

    return quote
        @kernel inbounds = true unsafe_indices = true function $(name)(
            dst::AbstractVector,
            src::AbstractVector,
            by::F, uint_map::M,
            global_counter::AbstractVector{UInt32},   # init = global excl prefix; mutated
            ::Val{Nbuckets},
            ::Val{warpsz},
        ) where {F,M,Nbuckets,warpsz}
            $body
        end
    end
end


function _define_byte_sort_kernel!(Nitem::Int)
    name = Symbol("byte_sort_kernel_", Nitem, "!")
    # Define the kernel AND retrieve it in the same Core.eval block. The
    # trailing `$name` evaluates the freshly-defined symbol in the latest
    # world, avoiding the world-age warning that `getfield(@__MODULE__,
    # name)` would trigger after Core.eval bumps the world.
    fn = Core.eval(@__MODULE__, quote
        $(build_byte_sort_kernel_def(name, Nitem))
        $name
    end)
    # Install a typed dispatch method whose body returns `fn` as a literal
    # singleton embedded in the AST — no global-binding access at any point.
    Core.eval(@__MODULE__, :(@inline get_byte_sort_kernel(::Val{$Nitem}) = $fn))
    _byte_sort_kernel_cache[Nitem] = fn
    return fn
end


# Generic fallback for exotic Nitem. The Dict short-circuit prevents a
# second @eval if `_define_byte_sort_kernel!` ran in a world we can't yet
# see (typed method invisible until invokelatest unwinds).
function get_byte_sort_kernel(::Val{Nitem}) where {Nitem}
    haskey(_byte_sort_kernel_cache, Nitem) && return _byte_sort_kernel_cache[Nitem]
    return _define_byte_sort_kernel!(Nitem)
end


# Pre-compile the default spec at package load time so the default code
# path needs neither @eval nor invokelatest at runtime. We reuse
# `_define_byte_sort_kernel!`, which both defines the @kernel and installs
# the typed `Val{16}` dispatch method.
_define_byte_sort_kernel!(16)
