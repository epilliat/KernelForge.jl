@kernel function mapreducedims_kernel!(
    f, op, dst, src::AbstractArray{T,Ndims}, g,
    ::Val{reduce_dims}, ::Val{iters}, ::Val{keep_size}, ::Val{dim_map}
) where {T,Ndims,reduce_dims,iters,keep_size,dim_map}
    linear_idx = @index(Global, Linear)
    idx_keep = Tuple(CartesianIndices(keep_size)[linear_idx])
    first_iters = map(first, iters)

    first_idx = ntuple(Val(Ndims)) do d
        dim_map[d] == 0 ? idx_keep[d] : first_iters[dim_map[d]]
    end

    s = f(src[first_idx...])

    for idx_reduce in CartesianIndices(iters)
        Tuple(idx_reduce) == first_iters && continue
        full_idx = ntuple(Val(Ndims)) do d
            dim_map[d] == 0 ? idx_keep[d] : idx_reduce[dim_map[d]]
        end
        s = op(s, f(src[full_idx...]))
    end

    dst[idx_keep...] = g(s)
end