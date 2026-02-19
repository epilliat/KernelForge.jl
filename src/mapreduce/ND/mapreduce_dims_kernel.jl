@kernel function mapreduce_dims_kernel!(
    f, op, dst, src, g,
    ::Val{reduce_dims}, ::Val{iters}, ::Val{keep_size}
) where {reduce_dims,iters,keep_size}
    linear_idx = @index(Global, Linear)
    idx_keep = Tuple(CartesianIndices(keep_size)[linear_idx])

    first_iters = map(first, iters)
    first_idx = ntuple(Val(ndims(src))) do d
        pos_reduce = findfirst(==(d), reduce_dims)
        if pos_reduce !== nothing
            first_iters[pos_reduce]
        else
            idx_keep[d]  # use d directly, not keep_pos
        end
    end

    s = f(src[first_idx...])

    for idx_reduce in CartesianIndices(iters)
        Tuple(idx_reduce) == first_iters && continue
        full_idx = ntuple(Val(ndims(src))) do d
            pos_reduce = findfirst(==(d), reduce_dims)
            if pos_reduce !== nothing
                idx_reduce[pos_reduce]
            else
                idx_keep[d]  # use d directly, not keep_pos
            end
        end
        s = op(s, f(src[full_idx...]))
    end

    dst[idx_keep...] = g(s)
end