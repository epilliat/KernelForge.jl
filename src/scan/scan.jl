"""
    scan(f, op, src; kwargs...) -> GPU array
    scan(op, src; kwargs...) -> GPU array

GPU parallel prefix scan (cumulative reduction) using a decoupled lookback algorithm.

Applies `f` to each element, then computes inclusive prefix scan with `op`.

# Arguments
- `f`: Map function applied to each element (defaults to `identity`)
- `op`: Associative binary scan operator
- `src`: Input GPU array

# Keyword Arguments
- `tmp=nothing`: Pre-allocated temporary buffer
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=256`: Workgroup size
- `FlagType=UInt8`: Synchronization flag type

# Examples
```julia
# Cumulative sum
x = CUDA.rand(Float32, 10_000)
result = scan(+, x)

# Cumulative sum of squares
result = scan(x -> x^2, +, x)

# With pre-allocated temporary for repeated calls
tmp = Luma.get_allocation(scan!, similar(x), x)
result = scan(+, x; tmp)
```

See also: [`Luma.scan!`](@ref) for the in-place version.
"""
function scan end

"""
    scan!(f, op, dst, src; kwargs...)
    scan!(op, dst, src; kwargs...)

In-place GPU parallel prefix scan using a decoupled lookback algorithm.

Applies `f` to each element, then computes inclusive prefix scan with `op`,
writing results to `dst`.

# Arguments
- `f`: Map function applied to each element (defaults to `identity`)
- `op`: Associative binary scan operator
- `dst`: Output array for scan results
- `src`: Input GPU array

# Keyword Arguments
- `tmp=nothing`: Pre-allocated temporary buffer
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=256`: Workgroup size
- `FlagType=UInt8`: Synchronization flag type

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
dst = similar(x)

# Cumulative sum
scan!(+, dst, x)

# With pre-allocated temporary for repeated calls
tmp = Luma.get_allocation(scan!, dst, x)
for i in 1:100
    scan!(+, dst, x; tmp)
end
```

See also: [`Luma.scan`](@ref) for the allocating version.
"""
function scan! end

# ============================================================================
# Configuration helpers
# ============================================================================

@inline function default_nitem(::typeof(scan!), ::Type{T}) where {T}
    sz = sizeof(T)
    if sz == 1
        return 32
    elseif sz == 2
        return 16
    elseif sz == 4
        return 8
    elseif sz == 8  # Float64
        return 8
    else
        return 4
    end
end

const DEFAULT_SCAN_CONFIG = (workgroup=256,)

@inline function get_scan_config(n::Int, Nitem::Int, workgroup::Int)
    ndrange = cld(n, Nitem)
    blocks = cld(ndrange, workgroup)
    return ndrange, blocks
end

# ============================================================================
# Temporary buffer allocation
# ============================================================================

"""
    get_allocation(::typeof(scan!), dst, src; kwargs...)

Allocate temporary buffer for `scan!`. Useful for repeated scans.

# Arguments
- `dst`: Output GPU array (used for element type of intermediates)
- `src`: Input GPU array (used for backend)

# Keyword Arguments
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=256`: Workgroup size (must match the `workgroup` used in `scan!`)
- `FlagType=UInt8`: Synchronization flag type

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
dst = similar(x)
tmp = Luma.get_allocation(scan!, dst, x)

for i in 1:100
    scan!(+, dst, x; tmp)
end
```
"""
function get_allocation(
    ::typeof(scan!),
    dst::AbstractGPUVector{Outf},
    src::AbstractGPUVector{T};
    Nitem::Union{Integer,Nothing}=nothing,
    workgroup::Int=DEFAULT_SCAN_CONFIG.workgroup,
    FlagType::Type{FT}=UInt8
) where {Outf,T,FT}
    _Nitem = Nitem === nothing ? default_nitem(scan!, Outf) : Int(Nitem)
    n = length(src)
    _, blocks = get_scan_config(n, _Nitem, workgroup)
    backend = get_backend(dst)
    sz = sum(get_partition_sizes(blocks, Outf, Outf, FT))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

# ============================================================================
# Allocating API
# ============================================================================

# Without map function (identity)
function scan(
    op::O,
    src::AbstractGPUVector{T};
    tmp::Union{AbstractGPUVector{UInt8},Nothing}=nothing,
    Nitem::Union{Integer,Nothing}=nothing,
    workgroup::Int=DEFAULT_SCAN_CONFIG.workgroup,
    FlagType::Type{FT}=UInt8
) where {T,O<:Function,FT}
    return scan(identity, op, src; tmp, Nitem, workgroup, FlagType)
end

# With map function
function scan(
    f::F, op::O,
    src::AbstractGPUVector{T};
    tmp::Union{AbstractGPUVector{UInt8},Nothing}=nothing,
    Nitem::Union{Integer,Nothing}=nothing,
    workgroup::Int=DEFAULT_SCAN_CONFIG.workgroup,
    FlagType::Type{FT}=UInt8
) where {T,F<:Function,O<:Function,FT}
    H = Base.promote_op(f, T)  # Output type after applying f
    backend = get_backend(src)
    dst = KernelAbstractions.allocate(backend, H, length(src))
    scan!(f, op, dst, src; tmp, Nitem, workgroup, FlagType)
    return dst
end

# ============================================================================
# In-place API
# ============================================================================

# Without map function (identity)
function scan!(
    op::O,
    dst::AbstractGPUVector{Outf},
    src::AbstractGPUVector{T};
    tmp::Union{AbstractGPUVector{UInt8},Nothing}=nothing,
    Nitem::Union{Integer,Nothing}=nothing,
    workgroup::Int=DEFAULT_SCAN_CONFIG.workgroup,
    FlagType::Type{FT}=UInt8
) where {Outf,T,O<:Function,FT}
    return scan!(identity, op, dst, src; tmp, Nitem, workgroup, FlagType)
end

# Main in-place entry point
function scan!(
    f::F, op::O,
    dst::AbstractGPUVector{Outf},
    src::AbstractGPUVector{T};
    tmp::Union{AbstractGPUVector{UInt8},Nothing}=nothing,
    Nitem::Union{Integer,Nothing}=nothing,
    workgroup::Int=DEFAULT_SCAN_CONFIG.workgroup,
    FlagType::Type{FT}=UInt8
) where {Outf,T,F<:Function,O<:Function,FT}
    n = length(src)
    n == 0 && return dst
    backend = get_backend(src)

    # Resolve defaults (avoiding `something` for type stability)
    _Nitem = Nitem === nothing ? default_nitem(scan!, Outf) : Int(Nitem)

    # Compute launch configuration
    ndrange, blocks = get_scan_config(n, _Nitem, workgroup)

    # Allocate temporaries if not provided
    _tmp = if tmp === nothing
        sz = sum(get_partition_sizes(blocks, Outf, Outf, FT))
        KernelAbstractions.allocate(backend, UInt8, sz)
    else
        tmp
    end

    _scan_impl!(f, op, dst, src, Val(_Nitem), _tmp, ndrange, blocks, workgroup, Outf, FT, n, backend)
end

# ============================================================================
# Core implementation
# ============================================================================

function _scan_impl!(
    f::F, op::O,
    dst::AbstractGPUVector{Outf},
    src::AbstractGPUVector{T},
    ::Val{Nitem},
    tmp::AbstractGPUVector{UInt8},
    ndrange::Int,
    blocks::Int,
    workgroup::Int,
    ::Type{H},
    ::Type{FT},
    n::Int,
    backend
) where {Outf,T,F,O,Nitem,H,FT}
    partial1, partial2, flag = partition(tmp, blocks, H, H, FT)

    # Initialize flags and select target value (type-stable)
    targetflag = if FT === UInt8
        fill!(flag, 0x00)
        0x01
    else
        rand(FT)
    end
    scan_kernel!(backend, workgroup, ndrange)(
        f, op, dst, src, Val(Nitem), partial1, partial2, flag, targetflag, H
    )

    return dst
end