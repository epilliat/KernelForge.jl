# ============================================================================
# Buffer type
# ============================================================================

"""
    MapReduceBuffer{A1,A2}

Pre-allocated buffer for `mapreduce1d!`, holding typed intermediate arrays
instead of a raw `UInt8` byte buffer.

- `A1`: array type for partial reduction values (eltype = output of `f`)
- `A2`: array type for synchronization flags (eltype = `UInt8`)
"""
struct MapReduceBuffer{A1,A2}
    partial::A1
    flag::A2
end

# ============================================================================
# Public API docstrings
# ============================================================================

"""
    mapreduce1d(f, op, src; kwargs...) -> scalar or GPU array
    mapreduce1d(f, op, srcs::NTuple; kwargs...) -> scalar or GPU array

GPU parallel map-reduce operation.

Applies `f` to each element, reduces with `op`, and optionally applies `g` to the final result.

# Arguments
- `f`: Map function applied to each element
- `op`: Associative binary reduction operator
- `src` or `srcs`: Input GPU array(s)

# Keyword Arguments
- `g=identity`: Post-reduction transformation applied to final result
- `tmp=nothing`: Pre-allocated `MapReduceBuffer` (or `nothing` to allocate automatically)
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=$(DEFAULT_WORKGROUP)`: Workgroup size
- `blocks=$(DEFAULT_BLOCKS)`: Number of blocks
- `to_cpu=true`: If true, return scalar; otherwise return 1-element GPU array

# Examples
```julia
# Sum of squares
x = CUDA.rand(Float32, 10_000)
result = mapreduce1d(x -> x^2, +, x)

# Return GPU array instead of scalar
result = mapreduce1d(x -> x^2, +, x; to_cpu=false)

# Dot product of two arrays
x, y = CUDA.rand(Float32, 10_000), CUDA.rand(Float32, 10_000)
result = mapreduce1d((a, b) -> a * b, +, (x, y))
```

See also: [`KernelForge.mapreduce1d!`](@ref) for the in-place version.
"""
function mapreduce1d end

"""
    mapreduce1d!(f, op, dst, src; kwargs...)
    mapreduce1d!(f, op, dst, srcs::NTuple; kwargs...)

In-place GPU parallel map-reduce, writing result to `dst[1]`.

# Arguments
- `f`: Map function applied to each element
- `op`: Associative binary reduction operator
- `dst`: Output array (result written to first element)
- `src` or `srcs`: Input GPU array(s)

# Keyword Arguments
- `g=identity`: Post-reduction transformation applied to final result
- `tmp=nothing`: Pre-allocated `MapReduceBuffer` (or `nothing` to allocate automatically)
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=$(DEFAULT_WORKGROUP)`: Workgroup size
- `blocks=$(DEFAULT_BLOCKS)`: Number of blocks

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
dst = CUDA.zeros(Float32, 1)

# Sum
mapreduce1d!(identity, +, dst, x)

# With pre-allocated buffer for repeated calls
tmp = KernelForge.get_allocation(MapReduce1D, x -> x^2, +, x)
for i in 1:100
    mapreduce1d!(x -> x^2, +, dst, x; tmp)
end
```

See also: [`KernelForge.mapreduce1d`](@ref) for the allocating version.
"""
function mapreduce1d! end

# ============================================================================
# Buffer allocation
# ============================================================================

"""
    get_allocation(::Type{MapReduce1D}, f, op, src_or_srcs, blocks=DEFAULT_BLOCKS)

Allocate a `MapReduceBuffer` for `mapreduce1d!`. Useful for repeated reductions.

# Arguments
- `f`: Map function (used to infer intermediate eltype)
- `op`: Reduction operator
- `src_or_srcs`: Input GPU array or NTuple of arrays (used to determine backend and eltype)
- `blocks`: Number of blocks (must match `blocks` used in `mapreduce1d!`)

# Returns
A `MapReduceBuffer{A1,A2}` holding typed partial and flag arrays (flags are `UInt8`).

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
tmp = KernelForge.get_allocation(MapReduce1D, x -> x^2, +, x)
dst = CUDA.zeros(Float32, 1)

for i in 1:100
    mapreduce1d!(x -> x^2, +, dst, x; tmp)
end
```
"""
function get_allocation(
    ::Type{MapReduce1D},
    f::F,
    op::O,
    src::AT,
    blocks::Integer=DEFAULT_BLOCKS
) where {F<:Function,O<:Function,AT<:AbstractArray}
    return get_allocation(MapReduce1D, f, op, (src,), blocks)
end

function get_allocation(
    ::Type{MapReduce1D},
    f::F,
    op::O,
    srcs::NTuple{U,AT},
    blocks::Integer=DEFAULT_BLOCKS
) where {U,F<:Function,O<:Function,AT<:AbstractArray}
    T = eltype(AT)
    H = Base.promote_op(f, ntuple(_ -> T, Val(U))...)
    backend = get_backend(srcs[1])
    partial = KernelAbstractions.allocate(backend, H, blocks)
    flag = KernelAbstractions.allocate(backend, UInt8, blocks)
    return MapReduceBuffer(partial, flag)
end

# ============================================================================
# Allocating API
# ============================================================================

# Single array: forward to tuple method
function mapreduce1d(
    f::F, op::O,
    src::AT;
    kwargs...
) where {F<:Function,O<:Function,AT<:AbstractArray}
    return mapreduce1d(f, op, (src,); kwargs...)
end

# Tuple of arrays: allocate dst then delegate to mapreduce1d!
function mapreduce1d(
    f::F, op::O,
    srcs::NTuple{U,AT};
    g::G=identity,
    tmp::TMP=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_WORKGROUP,
    blocks::Int=DEFAULT_BLOCKS,
    to_cpu::Bool=true
) where {U,AT<:AbstractArray,F<:Function,O<:Function,G<:Function,TMP<:Union{MapReduceBuffer,Nothing}}
    T = eltype(AT)
    H = Base.promote_op(f, ntuple(_ -> T, Val(U))...)
    S = Base.promote_op(g, H)
    backend = get_backend(srcs[1])
    dst = KernelAbstractions.allocate(backend, S, 1)
    mapreduce1d!(f, op, dst, srcs; g, tmp, Nitem, workgroup, blocks)
    return to_cpu ? Array(dst)[1] : dst
end

# ============================================================================
# In-place API
# ============================================================================

# Single array: forward to tuple method
function mapreduce1d!(
    f::F, op::O,
    dst::DS,
    src::AT;
    kwargs...
) where {F<:Function,O<:Function,DS<:AbstractArray,AT<:AbstractArray}
    return mapreduce1d!(f, op, dst, (src,); kwargs...)
end

# Main in-place entry point
function mapreduce1d!(
    f::F, op::O,
    dst::DS,
    srcs::NTuple{U,AT};
    g::G=identity,
    tmp::TMP=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_WORKGROUP,
    blocks::Int=DEFAULT_BLOCKS
) where {U,DS<:AbstractArray,AT<:AbstractArray,F<:Function,O<:Function,G<:Function,TMP<:Union{MapReduceBuffer,Nothing}}
    T = eltype(AT)
    n = length(srcs[1])
    backend = get_backend(srcs[1])
    H = Base.promote_op(f, ntuple(_ -> T, Val(U))...)
    _Nitem = something(Nitem, default_nitem(MapReduce1D, T))
    _mapreduce1d_impl!(f, op, g, dst, srcs, _Nitem, workgroup, blocks, tmp, H, n, backend)
end

# ============================================================================
# Core implementation
# ============================================================================

# Nothing dispatch: allocate buffer then forward to MapReduceBuffer method.
# Avoids passing a Union{MapReduceBuffer,Nothing} into the impl, which
# would cause type instability and spurious allocations.
function _mapreduce1d_impl!(
    f::F, op::O, g::G,
    dst::DS,
    srcs::NTuple{U,AT},
    Nitem::Int,
    workgroup::Int,
    blocks::Int,
    ::Nothing,
    ::Type{H},
    n::Int,
    backend
) where {U,DS<:AbstractArray,AT<:AbstractArray,F,O,G,H}
    tmp = get_allocation(MapReduce1D, f, op, srcs, blocks)
    _mapreduce1d_impl!(f, op, g, dst, srcs, Nitem, workgroup, blocks, tmp, H, n, backend)
end

function _mapreduce1d_impl!(
    f::F, op::O, g::G,
    dst::DS,
    srcs::NTuple{U,AT},
    Nitem::Int,
    workgroup::Int,
    blocks::Int,
    tmp::MapReduceBuffer,
    ::Type{H},
    n::Int,
    backend
) where {U,DS<:AbstractArray,AT<:AbstractArray,F,O,G,H}
    flag = tmp.flag
    fill!(flag, 0x00)
    partial = tmp.partial

    workgroup = min(workgroup, n)
    ndrange = min(blocks * workgroup, max(fld(n, workgroup) * workgroup, 1))
    Nitem = min(Nitem, prevpow(2, max(fld(n, ndrange), 1)))

    T = eltype(AT)
    Alignment = if U == 1
        (Int(pointer(srcs[1])) ÷ sizeof(T)) % Nitem + 1
    else
        aligns = ntuple(i -> (Int(pointer(srcs[i])) ÷ sizeof(T)) % Nitem, Val(U))
        allequal(aligns) ? aligns[1] + 1 : -1
    end
    mapreduce1d_kernel!(backend, workgroup)(
        f, op, g, dst, srcs, Val(Nitem), partial, flag, Val(Alignment); ndrange
    )
end