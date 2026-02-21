"""
    vecmat!(dst, x, A; kwargs...)
    vecmat!(f, op, dst, x, A; kwargs...)

GPU parallel vector-matrix multiplication: `dst = g(op(f(x .* A), dims=1))`.

For standard matrix-vector product: `vecmat!(dst, x, A)` computes `dst[j] = sum(x[i] * A[i,j])`.
When `x = nothing`, computes column reductions: `dst[j] = sum(A[i,j])`.

# Arguments
- `f=identity`: Element-wise transformation applied to `x[i] * A[i,j]` (or `A[i,j]` if `x=nothing`)
- `op=+`: Reduction operator
- `dst`: Output vector of length `p` (number of columns)
- `x`: Input vector of length `n` (number of rows), or `nothing` for pure column reduction
- `A`: Input matrix of size `(n, p)`

# Keyword Arguments
- `g=identity`: Optional post-reduction transformation
- `tmp=nothing`: Pre-allocated temporary buffer (from `get_allocation`)
- `Nitem=nothing`: Number of items per thread (auto-selected if nothing)
- `Nthreads=nothing`: Number of threads per column reduction
- `workgroup=nothing`: Workgroup size
- `blocks=nothing`: Maximum number of blocks
- `FlagType=UInt8`: Type for synchronization flags
"""
function vecmat end, function vecmat! end

# ============================================================================
# Configuration helpers
# ============================================================================

@inline function default_nitem(::Type{VecMat}, ::Type{T}) where {T}
    if sizeof(T) == 1
        return 16
    elseif sizeof(T) == 2
        return 8
    elseif sizeof(T) == 4
        return 8
    else
        return 4
    end
end

# ============================================================================
# Allocation
# ============================================================================

# Main public entry point for allocation
function get_allocation(
    ::Type{VecMat},
    f, op,
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T};
    Nblocks::Integer,
    out_eltype::Union{Type,Nothing}=nothing,
    FlagType::Type{FT}=UInt8
) where {T,FT}
    H = if !isnothing(out_eltype)
        out_eltype
    else
        isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, Base.eltype(x))
    end
    return _get_allocation(VecMat, x, src, Nblocks, H, FT)
end

# Core implementation (positional args)
function _get_allocation(
    ::Type{VecMat},
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T},
    Nblocks::Integer,
    ::Type{H},
    ::Type{FT}
) where {T,H,FT}
    Nblocks > 1 || error("Nblocks = cld(workgroup, Nthreads) must be > 1, otherwise tmp allocation is unnecessary")
    backend = get_backend(src)
    _, p = size(src)
    sz = sum(get_partition_sizes(Nblocks * p, H, FT))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

# ============================================================================
# Parameter resolution
# ============================================================================

function _resolve_parameters(
    ::Type{VecMat},
    Nthreads::Union{Int,Nothing},
    Nitem::Union{Int,Nothing},
    workgroup::Int,
    blocks::Int,
    def_nitem::Int,
    n::Int,
    p::Int
)
    workgroup = min(workgroup, prevpow(2, n * p))
    if isnothing(Nthreads) && isnothing(Nitem)
        thresh = prevpow(2, max(fld(n, 4), 1))
        Nitem = def_nitem
        if thresh >= workgroup
            Nblocks = min(blocks, max(fld(n, workgroup * Nitem), 1))
            Nthreads = workgroup * Nblocks
        else
            Nthreads = cld(thresh, Nitem)
        end
        Nitem = min(Nitem, prevpow(2, max(fld(n, Nthreads), 1)))
    end
    @assert !isnothing(Nthreads) && !isnothing(Nitem) "Must provide both Nthreads and Nitem, or neither"
    @assert Nthreads * Nitem <= n "Nthreads * Nitem must be <= n"
    return Nthreads, Nitem, workgroup
end

# ============================================================================
# Public API
# ============================================================================

# Simplified interface (standard vec-mat multiply) - allocating version
function vecmat(
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T};
    f::F=*,
    op::O=+,
    g::G=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    Nthreads=nothing,
    workgroup=nothing,
    blocks=nothing,
    FlagType::Type{FT}=UInt8
) where {T,F<:Function,O<:Function,G<:Function,FT}
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, Base.eltype(x))
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    p = size(src, 2)
    dst = KernelAbstractions.allocate(backend, S, p)
    _vecmat_entry!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, blocks, tmp, FT)
    return dst
end

# Full interface with f and op - allocating version
function vecmat(
    f::F, op::O,
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T};
    g::G=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    Nthreads=nothing,
    workgroup=nothing,
    blocks=nothing,
    FlagType::Type{FT}=UInt8
) where {T,F<:Function,O<:Function,G<:Function,FT}
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, Base.eltype(x))
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    p = size(src, 2)
    dst = KernelAbstractions.allocate(backend, S, p)
    _vecmat_entry!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, blocks, tmp, FT)
    return dst
end

# Simplified interface (standard vec-mat multiply) - in-place version
function vecmat!(
    dst::AbstractArray{S},
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T};
    f::F=*,
    op::O=+,
    g::G=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    Nthreads=nothing,
    workgroup=nothing,
    blocks=nothing,
    FlagType::Type{FT}=UInt8
) where {S,T,F<:Function,O<:Function,G<:Function,FT}
    _vecmat_entry!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, blocks, tmp, FT)
end

# Full interface with f and op - in-place version
function vecmat!(
    f::F, op::O,
    dst::AbstractArray{S},
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T};
    g::G=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    Nthreads=nothing,
    workgroup=nothing,
    blocks=nothing,
    FlagType::Type{FT}=UInt8
) where {S,T,F<:Function,O<:Function,G<:Function,FT}
    _vecmat_entry!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, blocks, tmp, FT)
end

# ============================================================================
# Entry point (validation and parameter resolution)
# ============================================================================

function _vecmat_entry!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T},
    Nitem::Union{Int,Nothing},
    Nthreads::Union{Int,Nothing},
    workgroup::Union{Int,Nothing},
    blocks::Union{Int,Nothing},
    tmp::Union{AbstractArray{UInt8},Nothing},
    ::Type{FT}
) where {S,T,F,O,G,FT}
    n, p = size(src)
    if !isnothing(x)
        @assert length(x) == n "Vector length must match matrix rows"
    end
    @assert length(dst) == p "Output length must match matrix columns"

    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, Base.eltype(x))

    workgroup_ = something(workgroup, DEFAULT_WORKGROUP)
    blocks_ = something(blocks, DEFAULT_BLOCKS)
    def_nitem = default_nitem(VecMat, T)

    Nthreads_, Nitem_, workgroup__ = _resolve_parameters(
        VecMat, Nthreads, Nitem, workgroup_, blocks_, def_nitem, n, p
    )

    _vecmat_dispatch!(f, op, g, dst, x, src, Nitem_, Nthreads_, workgroup__, tmp, H, FT)
end

# ============================================================================
# Dispatch (function barrier for type stability)
# ============================================================================

# Dispatch when user provides tmp buffer
function _vecmat_dispatch!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T},
    Nitem::Int,
    Nthreads::Int,
    workgroup::Int,
    tmp::AbstractArray{UInt8},
    ::Type{H},
    ::Type{FT}
) where {S,T,F,O,G,H,FT}
    if Nthreads <= workgroup
        _vecmat_impl_single!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, H, FT)
    else
        _vecmat_impl_multi!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, tmp, H, FT)
    end
end

# Dispatch when no tmp buffer provided
function _vecmat_dispatch!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T},
    Nitem::Int,
    Nthreads::Int,
    workgroup::Int,
    tmp::Nothing,
    ::Type{H},
    ::Type{FT}
) where {S,T,F,O,G,H,FT}
    if Nthreads <= workgroup
        _vecmat_impl_single!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, H, FT)
    else
        Nblocks = cld(Nthreads, workgroup)
        tmp_ = _get_allocation(VecMat, x, src, Nblocks, H, FT)
        _vecmat_impl_multi!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, tmp_, H, FT)
    end
end

# ============================================================================
# Core implementations
# ============================================================================

# Single-block case (no synchronization needed)
function _vecmat_impl_single!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T},
    Nitem::Int,
    Nthreads::Int,
    workgroup::Int,
    ::Type{H},
    ::Type{FT},
) where {S,T,F,O,G,H,FT}
    p = size(src, 2)
    backend = get_backend(src)
    ndrange = Nthreads * p
    vecmat_kernel!(backend, workgroup, ndrange)(
        f, op, g, dst, x, src, Val(Nitem), Val(Nthreads), nothing, nothing, one(FT), H
    )
end

# Multi-block case (with synchronization)
@inline function _vecmat_impl_multi!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T},
    Nitem::Int,
    Nthreads::Int,
    workgroup::Int,
    tmp::AbstractArray{UInt8},
    ::Type{H},
    ::Type{FT},
) where {S,T,F,O,G,H,FT}
    p = size(src, 2)
    backend = get_backend(src)
    ndrange = Nthreads * p
    Nblocks = cld(Nthreads, workgroup)

    partial, flag = partition(tmp, Nblocks * p, H, FT)
    FT === UInt8 && fill!(flag, zero(FT))
    targetflag = FT === UInt8 ? one(FT) : rand(FT)

    vecmat_kernel!(backend, workgroup, ndrange)(
        f, op, g, dst, x, src, Val(Nitem), Val(Nthreads), partial, flag, targetflag, H
    )
end