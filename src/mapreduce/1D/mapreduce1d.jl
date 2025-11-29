function mapreduce1d! end

#const DEFAULT_MAPREDUCE_NITEM = Dict(
#    UInt8 => 16
#)

@inline function default_mapreduce_nitem(::typeof(mapreduce1d!), T::Type)
    if sizeof(T) == 1
        return 16
    elseif sizeof(T) == 2
        return 8
    else
        return 1
    end
end

function get_allocation(
    ::typeof(mapreduce1d!),
    f, op,
    dst::AbstractGPUArray{S},
    srcs::NTuple{U,AbstractGPUArray{T}};
    #
    g=identity,
    Nitem=nothing,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    config::Union{NamedTuple,Nothing}=nothing,
    FlagType=UInt8
) where {U,S,T}
    backend = get_backend(dst)
    if isnothing(Nitem)
        Nitem = default_mapreduce_nitem(mapreduce1d!, T)
    end
    if isnothing(config)
        kernel = mapreduce1d_kernel!(backend, 10000, 100000) # dumy high values for launch config
        dummy_flag_array = KernelAbstractions.allocate(backend, FlagType, 0)
        config = get_default_config_cached(kernel, f, op, dst, srcs, g, Val(Nitem), dst, dummy_flag_array, FlagType(0))
    end

    H = Base.promote_op(f, T)
    sz = sum(get_partition_sizes(config.blocks, H, FlagType))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

function mapreduce1d!(
    f, op,
    dst::AbstractGPUArray{S},
    srcs::NTuple{U,AbstractGPUArray{T}};
    #
    g=identity,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    config=nothing,
    FlagType=UInt8
) where {U,S,T}

    n = length(srcs[1])
    backend = get_backend(srcs[1])

    H = Base.promote_op(f, T)
    if isnothing(Nitem)
        Nitem = default_mapreduce_nitem(mapreduce1d!, T)
    end
    if isnothing(config)
        kernel = mapreduce1d_kernel!(backend, 10000, 100000) # dumy high values for launch config
        dummy_flag_array = KernelAbstractions.allocate(backend, FlagType, 0)
        dummy_partial = dst # we could put KernelAbstractions.allocate(backend, H, 0) for more accuracy
        config = get_default_config_cached(kernel, f, op, dst, srcs, g, Val(Nitem), dummy_partial, dummy_flag_array, FlagType(0)) #time costly with @eval only the first time, then cached
        #println((MemoryAccess.InteractiveUtils.@which get_default_config_cached(kernel, f, op, dst, srcs, dst, dummy_flag_array, FlagType(0))))
    end

    workgroup, blocks = config

    workgroup = min(workgroup, n)
    ndrange = min(blocks * workgroup, max((fld(n, workgroup)) * workgroup, 1))

    # ensure that ndrange * Nitem <= N. Take a smaller Nitem if necessary (take power of two for alignment safety)
    Nitem = min(Nitem, prevpow(2, max(fld(n, ndrange), 1)))


    if isnothing(tmp)
        tmp = get_allocation(mapreduce1d!, f, op, dst, srcs; g=g, FlagType=FlagType, config=config, Nitem=Nitem)
    end

    partial, flag = partition(tmp, blocks, H, FlagType)
    if FlagType == UInt8
        setvalue!(flag, 0x00; Nitem=8)
        targetflag = 0x01
    else
        targetflag = rand(FlagType)
    end
    #KernelAbstractions.synchronize(backend)
    mapreduce1d_kernel!(backend, workgroup, ndrange)(f, op, dst, srcs, g, Val(Nitem), partial, flag, targetflag)

end

function mapreduce1d!(
    f, op,
    dst::AbstractGPUArray{S},
    src::AbstractGPUArray{T};
    #
    g=identity,
    Nitem=nothing,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    config=nothing,
    FlagType=UInt8
) where {S,T}
    return mapreduce1d!(f, op, dst, (src,); g=g, tmp=tmp, config=config, FlagType=FlagType, Nitem=Nitem)
end
