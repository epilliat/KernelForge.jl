# Global byte-histogram of a source array, computed in TWO STAGES to avoid the
# global-atomic flush that dominated the single-stage version.
#
# Output: `hist :: (Nbuckets, Npasses)` UInt32 — column p holds the Nbuckets-bucket
# count of digit `(uint_map(by(x)) >> 8*(p-1)) & 0xff` over all `x` in `src`.
#
# WHY TWO STAGES. The single-stage kernel had every block atomic-add its
# block-wide shared histogram into ONE global `hist`. Profiled on MI300A (1e8
# UInt32) that flush made the histogram 6.28 ms = 47% of the whole sort, at
# 64 GB/s — the cost was the GLOBAL-ATOMIC CONTENTION, which scales with the
# number of blocks (all blocks hit the same 1024 global slots). NOT the LDS
# atomics: per-warp privatization of the shared histogram was measured at only
# ~10%, while cutting the block count (or removing the global flush) was 5-13x.
#
# So: STAGE 1 (`bucket_histogram_count_kernel!`) writes each block's partial
# histogram to a DISTINCT global slice `hist_g[:, :, gid]` — plain stores, no
# cross-block atomics. STAGE 2 (`bucket_histogram_combine_kernel!`) tree-sums the
# `nhblk` slices into `hist`. MI300A 1e8: 6.28 -> 0.49 ms (12.8x). The launch
# uses a dedicated (wg, Nitem) chosen to keep `nhblk` modest (fewer/fatter blocks
# ⇒ cheaper combine) — see `_hist_launch_config` in sort1d.jl.
#
# Generic over the source eltype: `by(x)` extracts the field to sort by, and
# `uint_map(...)` projects it to an unsigned key. `Npasses == sizeof(K)`.
#
# Layout note (column-major): `hist[bucket, pass]` keeps a fixed-pass column
# contiguous — 256 threads writing `hist[lid, pass]` coalesce.

using KernelAbstractions
using KernelIntrinsics
using KernelIntrinsics: vload
using KernelAbstractions: @atomic
using Base.Cartesian: @nexprs


# Bounded item loader for the last partial chunk. OOB slots get a dummy load of
# `src[N]` (a valid read); the per-item bounds check skips them before counting.
@inline @generated function load_items_bounded(
    src::AbstractVector,
    block_idx::Int,
    N::Int,
    ::Val{Nitem},
) where {Nitem}
    elems = [
        quote
            let pos = (block_idx - 1) * $Nitem + $i
                pos <= N ? src[pos] : src[N]
            end
        end
        for i in 1:Nitem
    ]
    Expr(:tuple, elems...)
end


# `@nexprs` needs Npasses as an integer literal at macro-expansion time.
@inline @generated function bump_all_passes!(
    shared_hist, key,
    ::Val{Nbuckets}, ::Val{Npasses},
) where {Nbuckets,Npasses}
    quote
        digit_mask = UInt32($Nbuckets - 1)
        @nexprs $Npasses p -> begin
            shift_p = Int32((p - 1) * 8)
            digit_p = Int((key >> shift_p) & digit_mask)
            @atomic shared_hist[digit_p + 1, p] += UInt32(1)
        end
        nothing
    end
end


# SINGLE-STAGE (the original): every block atomic-adds its block-wide shared
# histogram into ONE global `hist`. On NVIDIA the shared+global atomics are fast
# (dedicated units), so this is NOT a bottleneck and BEATS the 2-stage — measured
# on A100: 2-stage regressed UInt64/Float64 1e8 by +21%. So the 2-stage is gated
# to CDNA3 (`sort_hist_2stage`) and every other arch keeps THIS path, byte-identical.
@kernel inbounds = true unsafe_indices = true function bucket_histogram_kernel!(
    hist,                                  # (Nbuckets, Npasses) UInt32
    src::AbstractVector,
    by::F,
    uint_map::M,
    ::Val{Nitem},
    ::Val{Nbuckets},
    ::Val{warpsz},
    ::Val{Npasses},
) where {F,M,Nitem,Nbuckets,warpsz,Npasses}

    @uniform begin
        N = length(src)
        block_total = N ÷ Nitem
        workgroup = Int(@groupsize()[1])
        warps_per_block = workgroup ÷ warpsz
    end

    lid = Int(@index(Local))
    gid = Int(@index(Group))
    warp_id = (lid - 1) ÷ warpsz + 1
    lane = (lid - 1) % warpsz + 1
    global_warp = (gid - 1) * warps_per_block + warp_id

    block_idx = (global_warp - 1) * warpsz + lane
    warp_last_block = global_warp * warpsz

    if warp_last_block <= block_total
        items = vload(src, block_idx, Val(Nitem))
        all_in_bounds = true
    else
        items = load_items_bounded(src, block_idx, N, Val(Nitem))
        all_in_bounds = false
    end

    shared_hist = @localmem UInt32 (Nbuckets, Npasses)

    let total = Nbuckets * Npasses
        for k in lid:workgroup:total
            shared_hist[k] = UInt32(0)
        end
    end
    @synchronize

    if all_in_bounds
        for i in 1:Nitem
            key_i = uint_map(by(items[i]))
            bump_all_passes!(shared_hist, key_i, Val(Nbuckets), Val(Npasses))
        end
    else
        for i in 1:Nitem
            pos = (block_idx - 1) * Nitem + i
            if pos <= N
                key_i = uint_map(by(items[i]))
                bump_all_passes!(shared_hist, key_i, Val(Nbuckets), Val(Npasses))
            end
        end
    end
    @synchronize

    for p in 1:Npasses
        for b in lid:workgroup:Nbuckets
            v = shared_hist[b, p]
            if v != UInt32(0)
                @atomic hist[b, p] += v
            end
        end
    end
end


# STAGE 1 (2-stage, CDNA3): per-block partial histograms. Each block accumulates
# a block-wide shared histogram (LDS atomics — cheap, uncontended enough), then
# writes it — EVERY (bucket, pass) entry, including zeros — to its own slice
# `hist_g[:,:,gid]`. No cross-block atomics, so no fill! of `hist_g` downstream.
@kernel inbounds = true unsafe_indices = true function bucket_histogram_count_kernel!(
    hist_g,                                # (Nbuckets, Npasses, nhblk) UInt32
    src::AbstractVector,                   # any T
    by::F,                                 # x -> field to sort by
    uint_map::M,                           # value -> Unsigned key
    ::Val{Nitem},
    ::Val{Nbuckets},
    ::Val{warpsz},
    ::Val{Npasses},
) where {F,M,Nitem,Nbuckets,warpsz,Npasses}

    @uniform begin
        N = length(src)
        block_total = N ÷ Nitem
        workgroup = Int(@groupsize()[1])
        warps_per_block = workgroup ÷ warpsz
    end

    lid = Int(@index(Local))
    gid = Int(@index(Group))
    warp_id = (lid - 1) ÷ warpsz + 1
    lane = (lid - 1) % warpsz + 1
    global_warp = (gid - 1) * warps_per_block + warp_id

    block_idx = (global_warp - 1) * warpsz + lane
    warp_last_block = global_warp * warpsz

    if warp_last_block <= block_total
        items = vload(src, block_idx, Val(Nitem))
        all_in_bounds = true
    else
        items = load_items_bounded(src, block_idx, N, Val(Nitem))
        all_in_bounds = false
    end

    shared_hist = @localmem UInt32 (Nbuckets, Npasses)

    let total = Nbuckets * Npasses
        for k in lid:workgroup:total
            shared_hist[k] = UInt32(0)
        end
    end
    @synchronize

    if all_in_bounds
        for i in 1:Nitem
            key_i = uint_map(by(items[i]))
            bump_all_passes!(shared_hist, key_i, Val(Nbuckets), Val(Npasses))
        end
    else
        for i in 1:Nitem
            pos = (block_idx - 1) * Nitem + i
            if pos <= N
                key_i = uint_map(by(items[i]))
                bump_all_passes!(shared_hist, key_i, Val(Nbuckets), Val(Npasses))
            end
        end
    end
    @synchronize

    # Plain store of this block's partial to its own slice. Every (b, p) is
    # written (zeros included) so `hist_g` is fully defined without a fill!.
    for p in 1:Npasses
        for b in lid:workgroup:Nbuckets
            hist_g[b, p, gid] = shared_hist[b, p]
        end
    end
end


# STAGE 2: sum the `nhblk` partial slices into the final histogram. One thread
# per (bucket, pass); each sums the block dimension. `hist` is fully written, so
# no fill! is needed before this either.
@kernel inbounds = true unsafe_indices = true function bucket_histogram_combine_kernel!(
    hist,                                  # (Nbuckets, Npasses) UInt32
    hist_g,                                # (Nbuckets, Npasses, nhblk) UInt32
    ::Val{Nbuckets},
    ::Val{Npasses},
    ::Val{nhblk},
) where {Nbuckets,Npasses,nhblk}
    idx = Int(@index(Global))              # 1 .. Nbuckets*Npasses
    if idx <= Nbuckets * Npasses
        b = (idx - 1) % Nbuckets + 1
        p = (idx - 1) ÷ Nbuckets + 1
        acc = UInt32(0)
        for k in 1:nhblk
            acc += hist_g[b, p, k]
        end
        hist[b, p] = acc
    end
end
