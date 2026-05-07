# Per-column exclusive prefix scan over a fixed-height histogram matrix.
#
# Input/Output: hist :: (Nrows, ncols), modified in place. After the kernel,
# hist[lid, gid] = sum_{r < lid} hist_input[r, gid] (exclusive prefix down
# the column gid).
#
# This is the form expected by `onesweep_kernel.jl` as `global_excl_hist`.
#
# Layout note: one workgroup per column (gid), workgroup size == Nrows. Each
# thread holds one row. With `hist[lid, gid]` and column-major order, a fixed
# gid puts rows contiguously → 256 threads access stride-1 → coalesced.
#
# Implementation: stage `hist[lid, gid]` into shared memory, run a Hillis-Steele
# inclusive scan (log2(Nrows) steps), then write back hist[lid, gid] = inclusive
# prefix - own value (= exclusive).

using KernelAbstractions


@kernel inbounds = true unsafe_indices = true function scan_histogram_kernel!(
    hist::AbstractMatrix{T},               # (Nrows, ncols), in-place
    ::Val{Nrows},
) where {T,Nrows}

    lid = Int(@index(Local))               # row,    1..Nrows
    gid = Int(@index(Group))               # column, 1..ncols

    shared = @localmem T Nrows

    own = hist[lid, gid]
    shared[lid] = own
    @synchronize

    offset = 1
    while offset < Nrows
        other = lid > offset ? shared[lid - offset] : zero(T)
        @synchronize
        shared[lid] += other
        @synchronize
        offset <<= 1
    end

    # shared[lid] is now the INCLUSIVE prefix at row lid. Subtract own value
    # to get the exclusive prefix.
    hist[lid, gid] = shared[lid] - own
end
