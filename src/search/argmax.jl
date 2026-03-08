
# ===========================================================================
# argmax1d
# ============================================================================

@inline default_nitem(arch::AbstractArch, ::Type{Argmax1D}, ::Type{T}) where T = default_nitem(arch, MapReduce1D, T)

"""
    argmax1d(f, rel, src; kwargs...) -> Int or GPU array
    argmax1d(f, rel, srcs::NTuple; kwargs...) -> Int or GPU array

GPU parallel argmax/argmin operation.

Applies `f` to each element, finds the extremum according to `rel`, and returns
the index of the first extremal element. Ties are broken by smallest index.

# Arguments
- `f`: Map function applied to each element
- `rel`: Comparison relation (`>` for argmax, `<` for argmin)
- `src` or `srcs`: Input GPU array(s)

# Keyword Arguments
- `tmp=nothing`: Pre-allocated `KernelBuffer` (or `nothing` to allocate automatically)
- `Nitem=nothing`: Items per thread (auto-selected if `nothing`)
- `workgroup=nothing`: Workgroup size (auto-selected if nothing)
- `blocks=nothing`: Number of blocks (auto-selected if nothing)
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)
- `to_cpu=true`: If `true`, return scalar `Int`; otherwise return 1-element GPU array

# Examples
```julia
x = CUDA.rand(Float32, 10_000)

# Argmax returning scalar index
idx = argmax1d(identity, >, x)

# Argmax returning 1-element GPU array
idx_gpu = argmax1d(identity, >, x; to_cpu=false)

# Argmin of absolute values
idx = argmax1d(abs, <, x)
```

See also: [`KernelForge.argmax1d!`](@ref) for the in-place version.
"""
function argmax1d end

"""
    argmax1d!(f, rel, dst, src; kwargs...)
    argmax1d!(f, rel, dst, srcs::NTuple; kwargs...)

In-place GPU parallel argmax/argmin, writing the index to `dst[1]`.

Ties are broken by smallest index.

# Arguments
- `f`: Map function applied to each element
- `rel`: Comparison relation (`>` for argmax, `<` for argmin)
- `dst`: Output array (index written to first element)
- `src` or `srcs`: Input GPU array(s)

# Keyword Arguments
- `tmp=nothing`: Pre-allocated `KernelBuffer` (or `nothing` to allocate automatically)
- `Nitem=nothing`: Items per thread (auto-selected if `nothing`)
- `workgroup=nothing`: Workgroup size (auto-selected if nothing)
- `blocks=nothing`: Number of blocks (auto-selected if nothing)
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
dst = CUDA.zeros(Int, 1)

# Argmax index
argmax1d!(identity, >, dst, x)

# With pre-allocated temporary for repeated calls
tmp = KernelForge.get_allocation(Argmax1D, identity, x)
for i in 1:100
    argmax1d!(identity, >, dst, x; tmp)
end
```

See also: [`KernelForge.argmax1d`](@ref) for the allocating version.
"""
function argmax1d! end

# ============================================================================
# Buffer allocation
# ============================================================================

"""
    get_allocation(::Type{Argmax1D}, f, src, blocks=nothing, arch=nothing)
    get_allocation(::Type{Argmax1D}, f, srcs::NTuple, blocks=nothing, arch=nothing)

Allocate a `KernelBuffer` for `argmax1d!`. Useful for repeated reductions.

The intermediate type is `Tuple{H, Int}` where `H = promote_op(f, T)`,
tracking both value and index.

# Arguments
- `f`: Map function (used to infer intermediate eltype)
- `src` or `srcs`: Input GPU array(s) (used for backend and element type)
- `blocks=nothing`: Number of blocks (auto-selected if nothing)
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
tmp = KernelForge.get_allocation(Argmax1D, identity, x)
dst = CUDA.zeros(Int, 1)

for i in 1:100
    argmax1d!(identity, >, dst, x; tmp)
end
```
"""
function get_allocation(
    ::Type{Argmax1D},
    f::F,
    src::AT,
    blocks=nothing,
    arch=nothing
) where {F<:Function,AT<:AbstractArray}
    return get_allocation(Argmax1D, f, (src,), blocks, arch)
end

function get_allocation(
    ::Type{Argmax1D},
    f::F,
    srcs::NTuple{U,AT},
    blocks=nothing,
    arch=nothing
) where {F<:Function,U,AT<:AbstractArray}
    arch = something(arch, detect_arch(srcs[1]))
    blocks = something(blocks, default_blocks(arch))
    T = eltype(AT)
    H = Base.promote_op(f, ntuple(_ -> T, Val(U))...)
    backend = get_backend(srcs[1])
    partial = KernelAbstractions.allocate(backend, Tuple{H,Int}, blocks)
    flag = KernelAbstractions.allocate(backend, UInt8, blocks)
    return KernelBuffer((; partial, flag))
end

get_allocation(::Type{Argmax}, f::F, src_or_srcs, blocks=nothing, arch=nothing) where {F<:Function} =
    get_allocation(Argmax1D, f, src_or_srcs, blocks, arch)

# ============================================================================
# Allocating API
# ============================================================================

# Single array
function argmax1d(
    f::F, rel::R,
    src::AT;
    tmp::TMP=nothing,
    Nitem=nothing,
    workgroup=nothing,
    blocks=nothing,
    arch=nothing,
    to_cpu::Bool=true,
) where {AT<:AbstractArray,F<:Function,R<:Function,TMP<:Union{KernelBuffer,Nothing}}
    return argmax1d(f, rel, (src,); tmp, Nitem, workgroup, blocks, arch, to_cpu)
end

# Tuple of arrays
function argmax1d(
    f::F, rel::R,
    srcs::NTuple{U,AT};
    tmp::TMP=nothing,
    Nitem=nothing,
    workgroup=nothing,
    blocks=nothing,
    arch=nothing,
    to_cpu::Bool=true,
) where {U,AT<:AbstractArray,F<:Function,R<:Function,TMP<:Union{KernelBuffer,Nothing}}
    T = eltype(AT)
    H = Base.promote_op(f, ntuple(_ -> T, Val(U))...)
    backend = get_backend(srcs[1])
    dst = KernelAbstractions.allocate(backend, Int, 1)
    arch = something(arch, detect_arch(srcs[1]))
    Nitem = something(Nitem, default_nitem(arch, Argmax1D, T))
    workgroup = something(workgroup, default_workgroup(arch))
    blocks = something(blocks, default_blocks(arch))
    _argmax1d_impl!(f, rel, dst, srcs, Nitem, workgroup, blocks, tmp, H, length(srcs[1]), backend, arch)
    return to_cpu ? (Array(dst)[1]) : dst
end

# ============================================================================
# In-place API
# ============================================================================

# Single array
function argmax1d!(
    f::F, rel::R,
    dst::DS,
    src::AT;
    kwargs...
) where {AT<:AbstractArray,DS<:AbstractArray,F<:Function,R<:Function}
    return argmax1d!(f, rel, dst, (src,); kwargs...)
end

# Main in-place entry point
function argmax1d!(
    f::F, rel::R,
    dst::DS,
    srcs::NTuple{U,AT};
    tmp::TMP=nothing,
    Nitem=nothing,
    workgroup=nothing,
    blocks=nothing,
    arch=nothing,
) where {U,AT<:AbstractArray,DS<:AbstractArray,F<:Function,R<:Function,TMP<:Union{KernelBuffer,Nothing}}
    T = eltype(AT)
    n = length(srcs[1])
    backend = get_backend(srcs[1])
    H = Base.promote_op(f, ntuple(_ -> T, Val(U))...)
    arch = something(arch, detect_arch(srcs[1]))
    Nitem = something(Nitem, default_nitem(arch, Argmax1D, T))
    workgroup = something(workgroup, default_workgroup(arch))
    blocks = something(blocks, default_blocks(arch))
    _argmax1d_impl!(f, rel, dst, srcs, Nitem, workgroup, blocks, tmp, H, n, backend, arch)
end

# ============================================================================
# Core implementation
# ============================================================================

# Nothing dispatch: allocate buffer then forward to KernelBuffer dispatch
function _argmax1d_impl!(
    f::F, rel::R,
    dst::DS,
    srcs::NTuple{U,AT},
    Nitem::Int,
    workgroup::Int,
    blocks::Int,
    ::Nothing,
    ::Type{H},
    n::Int,
    backend,
    arch
) where {U,AT<:AbstractArray,DS<:AbstractArray,F,R,H}
    tmp = get_allocation(Argmax1D, f, srcs, blocks, arch)
    _argmax1d_impl!(f, rel, dst, srcs, Nitem, workgroup, blocks, tmp, H, n, backend, arch)
end

# KernelBuffer dispatch: run the kernel
function _argmax1d_impl!(
    f::F, rel::R,
    dst::DS,
    srcs::NTuple{U,AT},
    Nitem::Int,
    workgroup::Int,
    blocks::Int,
    tmp::KernelBuffer,
    ::Type{H},
    n::Int,
    backend,
    arch
) where {U,AT<:AbstractArray,DS<:AbstractArray,F,R,H}
    workgroup = min(workgroup, n)
    ndrange = min(blocks * workgroup, max(fld(n, workgroup) * workgroup, 1))
    Nitem = min(Nitem, prevpow(2, max(fld(n, ndrange), 1)))

    fill!(tmp.arrays.flag, 0x00)
    warpsz = get_warpsize(arch)
    argmax_kernel!(backend, workgroup, ndrange)(
        f, rel, dst, srcs, Val(Nitem), tmp.arrays.partial, tmp.arrays.flag, Val(warpsz)
    )
end