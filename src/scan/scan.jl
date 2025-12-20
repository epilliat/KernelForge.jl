
function scan! end

const DEFAULT_SCAN_NITEM = Dict(
    Float32 => 8,
    Float64 => 8,
    Int => 4,
    UInt => 4,
    UInt8 => 32,
)
@inline function default_scan_nitem(::typeof(scan!), Outf::Type)
    if Outf in keys(DEFAULT_SCAN_NITEM)
        return DEFAULT_SCAN_NITEM[Outf]
    elseif sizeof(Outf) in (1, 2, 4, 8)
        return 32 ÷ sizeof(Outf)
    else
        return 4
    end
end


function _get_allocation(
    ::typeof(scan!),
    f, op,
    dst::AbstractGPUVector{Outf},
    src::AbstractGPUVector{T};
    workgroup=256,
    Nitem::Union{Integer,Nothing}=nothing,
    FlagType=UInt8
) where {Outf,T}

    if isnothing(Nitem)
        Nitem = default_scan_nitem(scan!, Outf)
    end
    ndrange = cld(length(src), Nitem)
    blocks = cld(ndrange, workgroup)
    backend = get_backend(dst)
    sz = sum(get_partition_sizes(blocks, Outf, Outf, FlagType))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end



function scan!(
    f, op,
    dst::AbstractGPUVector{Outf},
    src::AbstractGPUVector{T};
    tmp::Union{AbstractGPUVector{UInt8},Nothing}=nothing,
    workgroup=256,
    Nitem::Union{Integer,Nothing}=nothing,
    FlagType=UInt8
) where {Outf,T}

    if isnothing(Nitem)
        Nitem = default_scan_nitem(scan!, Outf)
    end
    if isnothing(tmp)
        tmp = _get_allocation(scan!, f, op, dst, src; workgroup=workgroup, Nitem=Nitem, FlagType=FlagType)
    end

    n = length(src)
    backend = get_backend(src)
    ndrange = cld(n, Nitem)
    blocks = cld(ndrange, workgroup)

    partial1, partial2, flag = partition(tmp, blocks, Outf, Outf, FlagType)

    if FlagType == UInt8
        setvalue!(flag, 0x00; Nitem=8)
        targetflag = 0x01
    else
        targetflag = rand(FlagType)
    end
    scan_kernel!(backend, workgroup)(f, op, dst, src, Val(Nitem), partial1, partial2, flag, targetflag; ndrange=ndrange)
end



function scan!(
    op,
    dst::AbstractGPUVector{Outf},
    src::AbstractGPUVector{T};
    tmp::Union{AbstractGPUVector{UInt8},Nothing}=nothing,
    workgroup=256,
    Nitem::Union{Integer,Nothing}=nothing,
    FlagType=UInt8
) where {Outf,T}

    scan!(identity, op, dst, src; tmp=tmp, workgroup=workgroup, Nitem=Nitem, FlagType=FlagType)
end