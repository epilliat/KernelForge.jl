@inline function default_nitem(::Type{Scan1D}, ::Type{T}) where {T}
    sz = sizeof(T)
    if sz == 1
        return 16
    elseif sz == 2
        return 16
    elseif sz == 4
        return 8
    elseif sz == 8
        return 8
    else
        return 4
    end
end

# ============================================================================
# Public API docstrings
# ============================================================================

"""
    scan(f, op, src; kwargs...) -> GPU array
    scan(op, src; kwargs...) -> GPU array

GPU parallel prefix scan (cumulative reduction) using a decoupled lookback algorithm.

Applies `f` to each element, computes inclusive prefix scan with `op`, and
optionally applies `g` to each output element.

# Arguments
- `f`: Map function applied to each element (defaults to `identity`)
- `op`: Associative binary scan operator
- `src`: Input GPU array

# Keyword Arguments
- `g=identity`: Post-scan transformation applied to each output element
- `tmp=nothing`: Pre-allocated `KernelBuffer` (or `nothing` to allocate automatically)
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=$(DEFAULT_WORKGROUP)`: Workgroup size

# Examples
```julia
# Cumulative sum
x = CUDA.rand(Float32, 10_000)
result = scan(+, x)

# Cumulative sum of squares
result = scan(x -> x^2, +, x)

# With post-scan transformation
result = scan(+, x; g = sqrt)

# With pre-allocated buffer for repeated calls
tmp = KernelForge.get_allocation(Scan1D, identity, +, x)
result = scan(+, x; tmp)
```

See also: [`KernelForge.scan!`](@ref) for the in-place version.
"""
function scan end

"""
    scan!(f, op, dst, src; kwargs...)
    scan!(op, dst, src; kwargs...)

In-place GPU parallel prefix scan using a decoupled lookback algorithm.

Applies `f` to each element, computes inclusive prefix scan with `op`, and
optionally applies `g` to each output element, writing results to `dst`.

# Arguments
- `f`: Map function applied to each element (defaults to `identity`)
- `op`: Associative binary scan operator
- `dst`: Output array for scan results
- `src`: Input GPU array

# Keyword Arguments
- `g=identity`: Post-scan transformation applied to each output element
- `tmp=nothing`: Pre-allocated `KernelBuffer` (or `nothing` to allocate automatically)
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=$(DEFAULT_WORKGROUP)`: Workgroup size

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
dst = similar(x)

# Cumulative sum
scan!(+, dst, x)

# With pre-allocated buffer for repeated calls
tmp = KernelForge.get_allocation(Scan1D, identity, +, x)
for i in 1:100
    scan!(+, dst, x; tmp)
end
```

See also: [`KernelForge.scan`](@ref) for the allocating version.
"""
function scan! end

# ============================================================================
# Buffer allocation
# ============================================================================

"""
    get_allocation(::Type{Scan1D}, f, op, src[, blocks])

Allocate a `KernelBuffer` for `scan!`. Useful for repeated scans.

# Arguments
- `f`: Map function (used to infer intermediate eltype)
- `op`: Reduction operator
- `src`: Input GPU array (used to determine backend and eltype)
- `blocks`: Number of blocks (auto-computed using default `Nitem` and `DEFAULT_WORKGROUP` if omitted)

# Returns
A `KernelBuffer` with named fields `partial1`, `partial2`, and `flag` (flags are `UInt8`).

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
tmp = KernelForge.get_allocation(Scan1D, identity, +, x)

dst = similar(x)
for i in 1:100
    scan!(+, dst, x; tmp)
end
```
"""
function get_allocation(
    ::Type{Scan1D},
    f::F,
    op::O,
    src::AT,
    blocks::Int
) where {F<:Function,O<:Function,AT<:AbstractArray}
    T = eltype(AT)
    H = Base.promote_op(f, T)
    backend = get_backend(src)
    partial1 = KernelAbstractions.allocate(backend, H, blocks)
    partial2 = KernelAbstractions.allocate(backend, H, blocks)
    flag = KernelAbstractions.allocate(backend, UInt8, blocks)
    return KernelBuffer((; partial1, partial2, flag))
end

function get_allocation(
    ::Type{Scan1D},
    f::F,
    op::O,
    src::AT
) where {F<:Function,O<:Function,AT<:AbstractArray}
    T = eltype(AT)
    H = Base.promote_op(f, T)
    Nitem = default_nitem(Scan1D, H)
    n = length(src)
    ndrange = cld(n, Nitem)
    blocks = cld(ndrange, DEFAULT_WORKGROUP)
    return get_allocation(Scan1D, f, op, src, blocks)
end

# ============================================================================
# Allocating API
# ============================================================================

# Without map function: forward to f=identity method
function scan(
    op::O,
    src::AT;
    kwargs...
) where {O<:Function,AT<:AbstractArray}
    return scan(identity, op, src; kwargs...)
end

# Main allocating entry point
function scan(
    f::F, op::O,
    src::AT;
    g::G=identity,
    tmp::TMP=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_WORKGROUP
) where {AT<:AbstractArray,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    T = eltype(AT)
    H = Base.promote_op(f, T)
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    dst = KernelAbstractions.allocate(backend, S, length(src))
    scan!(f, op, dst, src; g, tmp, Nitem, workgroup)
    return dst
end

# ============================================================================
# In-place API
# ============================================================================

# Without map function: forward to f=identity method
function scan!(
    op::O,
    dst::DS,
    src::AT;
    kwargs...
) where {O<:Function,DS<:AbstractArray,AT<:AbstractArray}
    return scan!(identity, op, dst, src; kwargs...)
end

# Main in-place entry point
function scan!(
    f::F, op::O,
    dst::DS,
    src::AT;
    g::G=identity,
    tmp::TMP=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_WORKGROUP
) where {DS<:AbstractArray,AT<:AbstractArray,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    n = length(src)
    n == 0 && return dst
    T = eltype(AT)
    H = Base.promote_op(f, T)
    backend = get_backend(src)
    Nitem_int = Nitem === nothing ? default_nitem(Scan1D, H) : Int(Nitem)
    ndrange = cld(n, Nitem_int)
    blocks = cld(ndrange, workgroup)
    _scan_impl!(f, op, g, dst, src, Nitem_int, workgroup, ndrange, blocks, tmp, n, backend)
end

# ============================================================================
# Core implementation
# ============================================================================

# Nothing dispatch: allocate buffer then forward to KernelBuffer dispatch
function _scan_impl!(
    f::F, op::O, g::G,
    dst::DS,
    src::AT,
    Nitem::Int,
    workgroup::Int,
    ndrange::Int,
    blocks::Int,
    ::Nothing,
    n::Int,
    backend
) where {DS<:AbstractArray,AT<:AbstractArray,F,O,G}
    tmp = get_allocation(Scan1D, f, op, src, blocks)
    _scan_impl!(f, op, g, dst, src, Nitem, workgroup, ndrange, blocks, tmp, n, backend)
end

# KernelBuffer dispatch: run the kernel
function _scan_impl!(
    f::F, op::O, g::G,
    dst::DS,
    src::AT,
    Nitem::Int,
    workgroup::Int,
    ndrange::Int,
    blocks::Int,
    tmp::KernelBuffer,
    n::Int,
    backend
) where {DS<:AbstractArray,AT<:AbstractArray,F,O,G}
    fill!(tmp.arrays.flag, 0x00)
    T = eltype(AT)
    S = eltype(DS)
    src_align = (Int(pointer(src)) ÷ sizeof(T)) % Nitem + 1
    dst_align = (Int(pointer(dst)) ÷ sizeof(S)) % Nitem + 1
    Alignment = src_align == dst_align ? src_align : -1
    scan_kernel!(backend, workgroup, ndrange)(
        f, op, g, dst, src, Val(Nitem),
        tmp.arrays.partial1, tmp.arrays.partial2, tmp.arrays.flag,
        Val(Alignment)
    )
    return dst
end