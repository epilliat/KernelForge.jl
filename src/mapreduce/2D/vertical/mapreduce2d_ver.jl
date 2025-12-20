using ArgCheck
function mapreduce2d_ver! end

#const DEFAULT_MAPREDUCE_NITEM = Dict(
#    UInt8 => 16
#)

@inline function default_nitem(::typeof(mapreduce2d_ver!), T::DataType)
    if sizeof(T) == 1
        return 16
    elseif sizeof(T) == 2
        return 8
    else
        return 1
    end
end
@inline function default_nblocks(::typeof(mapreduce2d_ver!), src)
    if sizeof(T) == 1
        return 16
    elseif sizeof(T) == 2
        return 8
    elseif sizeof(T) == 4
        return 4
    else
        return 1
    end
end
function _get_allocation( # Temporary allocation needed only for splitgrid, when there are several blocks per column doing reductions (n>>p).
    ::typeof(mapreduce2d_ver!),
    srcs::NTuple{U,AbstractArray{T}};
    #
    Nblocks::Integer, #number of blocks per column, must be > 1
    H::DataType,
    FlagType::DataType #necessary to get the size of the flags
) where {U,T}
    backend = get_backend(srcs[1])
    n, p = size(srcs[1])
    if Nblocks > 1
        sz = sum(get_partition_sizes(Nblocks * p, H, FlagType))
    else
        error("Nblocks must be > 1, otherwise tmp allocation is unnecessary")
    end
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

function mapreduce2d_ver!(
    f, op,
    dst::AbstractArray{S},
    srcs::NTuple{U,AbstractArray{T}};
    #
    g=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    Nthreads=nothing,
    Nblocks=nothing,
    config=nothing,
    FlagType=UInt8
) where {U,S,T}
    n, p = size(srcs[1])
    backend = get_backend(srcs[1])

    H = Base.promote_op(f, T)

    def_nitem = default_nitem(mapreduce2d_ver!, T)
    if isnothing(config) # Maybe we should be able to query the number of SMs instead
        kernel = mapreduce2d_ver_splitgrid_kernel!(backend, 10000, 100000) # dummy high values for launch config
        dummy_flag_array = KernelAbstractions.allocate(backend, FlagType, 0)
        dummy_partial = dst # we could put KernelAbstractions.allocate(backend, H, 0) for more accuracy
        config = get_default_config(kernel, f, op, dst, srcs, g, Val(def_nitem), Val(1), dummy_partial, dummy_flag_array, FlagType(0)) #time costly with @eval only the first time, then cached
        #@show kernel, config
        workgroup, blocks = 256, config.blocks # take a power of 2 for workgroup
    else
        workgroup, blocks = config
    end

    workgroup = min(workgroup, prevpow(2, n * p))
    blocks = config.blocks
    ####### Default fine tuning !!
    if isnothing(Nthreads) && isnothing(Nitem)
        Nthreads = 256
        thresh = prevpow(2, max(fld(n, 4), 1)) # power of 2 of order n/4
        if thresh >= 256 #n/4 >~ 256
            Nitem = def_nitem
        else # horizontal rectangular case
            Nitem = min(thresh, def_nitem * 4)
            Nthreads = cld(thresh, Nitem)
        end
        Nitem = min(Nitem, prevpow(2, max(fld(n, Nthreads), 1)))
    elseif !isnothing(Nthreads) && !isnothing(Nitem)
        @argcheck Nthreads * Nitem <= n
    end
    if isnothing(Nblocks)
        Nblocks = min(cld(blocks, p), max(fld(n, workgroup), 1))
    else
        @argcheck Nblocks * Nthreads * Nitem <= n
    end
    if Nblocks > 1
        @argcheck Nthreads == workgroup
    end
    ndrange = min(Nthreads * Nblocks * p)

    # ensure that ndrange * Nitem <= N. Take a smaller Nitem if necessary (take power of two for alignment safety)
    @show Nitem, Nthreads, Nblocks
    @show workgroup, blocks, ndrange

    if Nblocks == 1
        if Nthreads <= warpsz
            mapreduce2d_ver_splitwarp_kernel!(backend, workgroup, ndrange)(f, op, dst, srcs, g, Val(Nitem), Val(Nthreads), H)
        else
            mapreduce2d_ver_splitblock_kernel!(backend, workgroup, ndrange)(f, op, dst, srcs, g, Val(Nitem), Val(Nthreads), H)
        end
    else
        if isnothing(tmp)
            tmp = _get_allocation(mapreduce2d_ver!, srcs; Nblocks=Nblocks, H=H, FlagType=FlagType)
        end
        partial, flag = partition(tmp, Nblocks * p, H, FlagType)
        if FlagType == UInt8
            setvalue!(flag, 0x00; Nitem=8)
            targetflag = 0x01
        else
            targetflag = rand(FlagType)
        end
        mapreduce2d_ver_splitgrid_kernel!(backend, workgroup, ndrange)(f, op, dst, srcs, g, Val(Nitem), Val(Nblocks), partial, flag, targetflag)
    end
    #KernelAbstractions.synchronize(backend)

end

function mapreduce2d_ver!(
    f, op,
    dst::AbstractArray{S},
    src::AbstractArray{T};
    #
    g=identity,
    Nitem=nothing,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    config=nothing,
    FlagType=UInt8
) where {S,T}
    return mapreduce2d_ver!(f, op, dst, (src,); g=g, tmp=tmp, config=config, FlagType=FlagType, Nitem=Nitem)
end
