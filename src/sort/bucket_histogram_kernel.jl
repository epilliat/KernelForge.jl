# Global byte-histogram of a source array. Output: `(Nbuckets, Npasses)`
# UInt32 matrix; column p holds the Nbuckets-bucket count of digit
# `(uint_map(by(x)) >> 8*(p-1)) & 0xff` over all `x` in `src`.
#
# Generic over the source eltype: `by(x)` extracts the field to sort by,
# and `uint_map(...)` projects it to an unsigned integer (UInt8/16/32/64).
# `Npasses == sizeof(K)` where K is the result type of `uint_map ∘ by` —
# the wrapper computes it and passes via `Val(Npasses)`.
#
# Layout note (column-major): with `hist[bucket, pass]`, a fixed-pass column
# is contiguous in memory — 256 threads writing `hist[lid, pass]` coalesce.
#
# Per block (all Npasses fused into one inner loop):
#   - zero shared_hist :: (Nbuckets, Npasses) once + @synchronize.
#   - atomic-shared: workgroup * Nitem * Npasses ops; each item bumps all
#     Npasses passes' bins (different shared addresses → different banks,
#     pipelines well).
#   - @synchronize.
#   - atomic-global: Nbuckets * Npasses ops total per block (one per
#     non-zero bin per pass).
#
# Items are loaded once via `vload`; `src` is read exactly once for all
# Npasses passes. Sync count and zero-init writes are constant in Npasses
# (versus the original per-pass version which paid them Npasses times).

using KernelAbstractions
using KernelIntrinsics
using KernelIntrinsics: vload
using Atomix: @atomic
using Base.Cartesian: @nexprs


# Bounded item loader for the last partial chunk. OOB slots get a dummy
# load of `src[N]` (a valid memory read); the per-item bounds check
# (`pos <= N`) skips them before they're counted.
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
# Inside the kernel body Npasses is a Val type parameter; @generated lets
# us substitute it as a literal (`$Npasses`) into the @nexprs count.
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


@kernel inbounds = true unsafe_indices = true function bucket_histogram_kernel!(
    hist,                                  # (Nbuckets, Npasses) UInt32
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

    # Zero the whole (Nbuckets, Npasses) tile in one pass.
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

    # Flush shared_hist → global hist. For each fixed pass, threads at
    # stride `workgroup` write hist[b, pass] — column-major layout means
    # stride-1 across threads → coalesced.
    for p in 1:Npasses
        for b in lid:workgroup:Nbuckets
            v = shared_hist[b, p]
            if v != UInt32(0)
                @atomic hist[b, p] += v
            end
        end
    end
end
