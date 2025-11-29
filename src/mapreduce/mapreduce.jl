
function mapreduce!(
    f, op,
    dst,
    src::AbstractGPUVector{T};
    #
    g=identity,
    Nitem=nothing,
    tmp::Union{AbstractGPUVector{UInt8},Nothing}=nothing,
    dims=nothing,
    config=nothing,
    FlagType=UInt8,
) where {T}
    if !isnothing(dims) && dims ∉ (1, (1,))
        error("dims !")
    end
    mapreduce1d!(f, op, dst, src; g, Nitem, tmp, config, FlagType)
end

function mapreduce!(
    f, op,
    dst,
    src::AbstractGPUArray{T};
    #
    g=identity,
    Nitem=nothing,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    dims=nothing,
    config=nothing,
    FlagType=UInt8,
) where {T}
    nonflat_axes = (i for (i, x) in size(src) if x > 1)
    flat_axes = (i for (i, x) in size(src) if x == 1)
    if isnothing(dims) || issubset(nonflat_axes, dims)
        mapreduce1d!(f, op, dst, src; g, Nitem, tmp, config, FlagType)
        dst = reshape(dst, ntuple(i -> 1, Val(length(size(src)))))
    elseif issubset(flat_axes, dims)
        dst = copy(src)
    end
end

function mapreduce(
    f, op,
    src::AbstractGPUArray{T};
    #
    g=identity,
    Nitem=nothing,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    config=nothing,
    FlagType=UInt8,
    to_cpu=false
) where {T}
    S = Base.promote_op(g ∘ f, T)

    dst = KernelAbstractions.allocate(get_backend(src), S, 1)
    mapreduce1d!(f, op, dst, (src,); g=g, Nitem=Nitem, tmp=tmp, config=config, FlagType=FlagType)
    if to_cpu
        return @allowscalar dst[1]
    else
        return dst
    end
end


function reduce!(
    op,
    dst::AbstractGPUArray{S},
    src::AbstractGPUArray{T};
    #
    Nitem=nothing,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    config=nothing,
    FlagType=UInt8
) where {S,T}
    return mapreduce1d!(identity, op, dst, (src,); Nitem=Nitem, tmp=tmp, config=config, FlagType=FlagType)
end