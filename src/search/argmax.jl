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
- `tmp=nothing`: Pre-allocated temporary buffer (see [`get_allocation`](@ref))
- `Nitem=nothing`: Items per thread (auto-selected if `nothing`)
- `workgroup=256`: Workgroup size
- `blocks=100`: Number of blocks
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
- `tmp=nothing`: Pre-allocated temporary buffer (see [`get_allocation`](@ref))
- `Nitem=nothing`: Items per thread (auto-selected if `nothing`)
- `workgroup=256`: Workgroup size
- `blocks=100`: Number of blocks

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
dst = CUDA.zeros(Int, 1)

# Argmax index
argmax1d!(identity, >, dst, x)

# With pre-allocated temporary for repeated calls
tmp = KernelForge.get_allocation(Argmax1D, x)
for i in 1:100
    argmax1d!(identity, >, dst, x; tmp)
end
```

See also: [`KernelForge.argmax1d`](@ref) for the allocating version.
"""
function argmax1d! end

# ============================================================================
# Temporary buffer allocation
# ============================================================================

"""
    get_allocation(::Type{Argmax1D}, src; blocks=100, out_eltype=nothing)

Allocate temporary buffer for `argmax1d!`. Useful for repeated reductions.

The intermediate type is `Tuple{out_eltype, Int}` to track both value and index.

# Arguments
- `src` or `srcs`: Input GPU array(s) (used for backend and default element type)

# Keyword Arguments
- `blocks=100`: Number of blocks (must match the `blocks` used in `argmax1d!`)
- `out_eltype=nothing`: Element type for intermediate values. If `nothing`, defaults to
  the element type of `src`. For proper type inference, pass `promote_op(f, T, ...)`.

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
tmp = KernelForge.get_allocation(Argmax1D, x)
dst = CUDA.zeros(Int, 1)

for i in 1:100
    argmax1d!(identity, >, dst, x; tmp)
end
```
"""
function get_allocation(
    ::Type{Argmax1D},
    src::AbstractGPUArray{T};
    blocks::Integer=DEFAULT_BLOCKS,
    out_eltype::Type=T,
) where {T}
    return get_allocation(Argmax1D, (src,); blocks, out_eltype)
end

function get_allocation(
    ::Type{Argmax1D},
    srcs::NTuple{U,AbstractGPUArray{T}};
    blocks::Integer=DEFAULT_BLOCKS,
    out_eltype::Type=T,
) where {U,T}
    H = out_eltype
    backend = get_backend(srcs[1])
    sz = sum(get_partition_sizes(blocks, Tuple{H,Int}, UInt8))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

# ============================================================================
# Allocating API
# ============================================================================

# Single array
function argmax1d(
    f, rel,
    src::AbstractGPUArray{T};
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_WORKGROUP,
    blocks::Int=DEFAULT_BLOCKS,
    to_cpu::Bool=true,
) where {T}
    return argmax1d(f, rel, (src,); tmp, Nitem, workgroup, blocks, to_cpu)
end

# Tuple of arrays
function argmax1d(
    f::F, rel::R,
    srcs::NTuple{U,AbstractGPUArray{T}};
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_WORKGROUP,
    blocks::Int=DEFAULT_BLOCKS,
    to_cpu::Bool=true,
) where {U,T,F<:Function,R<:Function}
    H = Base.promote_op(f, ntuple(_ -> T, Val(U))...)
    backend = get_backend(srcs[1])
    dst = KernelAbstractions.allocate(backend, Int, 1)
    _Nitem = something(Nitem, default_nitem(Argmax1D, T))
    _tmp = something(tmp, get_allocation(Argmax1D, srcs; blocks, out_eltype=H))
    _argmax1d_impl!(f, rel, dst, srcs, _Nitem, workgroup, blocks, _tmp, H, length(srcs[1]), backend)
    return to_cpu ? (@allowscalar dst[1]) : dst
end

# ============================================================================
# In-place API
# ============================================================================

# Single array convenience wrapper
function argmax1d!(
    f, rel,
    dst::AbstractGPUArray{<:Integer},
    src::AbstractGPUArray{T};
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_WORKGROUP,
    blocks::Int=DEFAULT_BLOCKS,
) where {T}
    return argmax1d!(f, rel, dst, (src,); tmp, Nitem, workgroup, blocks)
end

# Main in-place entry point
function argmax1d!(
    f::F, rel::R,
    dst::AbstractGPUArray{<:Integer},
    srcs::NTuple{U,AbstractGPUArray{T}};
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_WORKGROUP,
    blocks::Int=DEFAULT_BLOCKS,
) where {U,T,F<:Function,R<:Function}
    n = length(srcs[1])
    backend = get_backend(srcs[1])
    H = Base.promote_op(f, ntuple(_ -> T, Val(U))...)
    _Nitem = something(Nitem, default_nitem(Argmax1D, T))
    _tmp = something(tmp, get_allocation(Argmax1D, srcs; blocks, out_eltype=H))
    _argmax1d_impl!(f, rel, dst, srcs, _Nitem, workgroup, blocks, _tmp, H, n, backend)
end

# ============================================================================
# Core implementation
# ============================================================================

function _argmax1d_impl!(
    f::F, rel::R,
    dst::AbstractGPUArray{<:Integer},
    srcs::NTuple{U,AbstractGPUArray{T}},
    Nitem::Int,
    workgroup::Int,
    blocks::Int,
    tmp::AbstractGPUArray{UInt8},
    ::Type{H},
    n::Int,
    backend,
) where {U,T,F,R,H}
    workgroup = min(workgroup, n)
    ndrange = min(blocks * workgroup, max(fld(n, workgroup) * workgroup, 1))
    Nitem = min(Nitem, prevpow(2, max(fld(n, ndrange), 1)))

    partial, flag = partition(tmp, blocks, Tuple{H,Int}, UInt8)
    fill!(flag, 0x00)

    argmax_kernel!(backend, workgroup, ndrange)(
        f, rel, dst, srcs, Val(Nitem), partial, flag
    )
end