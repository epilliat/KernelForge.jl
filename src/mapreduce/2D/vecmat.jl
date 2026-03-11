# ============================================================================
# Public API docstrings
# ============================================================================

"""
    vecmat([f, op,] x, src; kwargs...) -> GPU array
    vecmat!([f, op,] dst, x, src; kwargs...)

GPU parallel vector-matrix multiplication: `dst = g(op(f(x .* A), dims=1))`.

For standard matrix-vector product: `vecmat!(dst, x, A)` computes `dst[j] = sum(x[i] * A[i,j])`.
When `x = nothing`, computes column reductions: `dst[j] = sum(A[i,j])`.

# Arguments
- `f=*`: Element-wise transformation applied to `x[i] * A[i,j]` (or `A[i,j]` if `x=nothing`)
- `op=+`: Reduction operator
- `dst`: Output vector of length `p` (number of columns)
- `x`: Input vector of length `n` (number of rows), or `nothing` for pure column reduction
- `src`: Input matrix of size `(n, p)`

# Keyword Arguments
- `g=identity`: Optional post-reduction transformation
- `tmp=nothing`: Pre-allocated `KernelBuffer` (or `nothing` to allocate automatically)
- `Nitem=nothing`: Number of items per thread (auto-selected if nothing)
- `Nthreads=nothing`: Number of threads per column reduction (auto-selected if nothing)
- `workgroup=nothing`: Workgroup size (auto-selected if nothing)
- `blocks=nothing`: Maximum number of blocks (auto-selected if nothing)
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)

# Examples
```julia
A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 1000)

# Standard vector-matrix multiply: y = x' * A
y = vecmat(x, A)

# Column-wise sum: y[j] = sum(A[:, j])
y = vecmat(nothing, A)

# With pre-allocated buffer for repeated calls
tmp = KernelForge.get_allocation(VecMat, *, +, x, A)
dst = CUDA.zeros(Float32, 500)
for i in 1:100
    vecmat!(dst, x, A; tmp)
end
```
"""
function vecmat end, function vecmat! end

# ============================================================================
# Configuration helpers
# ============================================================================
@inline default_workgroup(::AbstractArch, ::Type{VecMat}, n, p, ::Type{T}) where T = 256
@inline default_workgroup(::RTX1000, ::Type{VecMat}, n, p, ::Type{T}) where T = 256

@inline default_blocks(::AbstractArch, ::Type{VecMat}, n, p, ::Type{T}) where T = 128
@inline default_blocks(::RTX1000, ::Type{VecMat}, n, p, ::Type{T}) where T = 80
@inline default_blocks(::AMDArch, ::Type{VecMat}, n, p, ::Type{T}) where T = 256


@inline function default_nitem(::AbstractArch, ::Type{VecMat}, n, p, ::Type{T}) where {T}
    Nitem = prevpow(2, cld(16, cld(sizeof(T), 2)))
    return Nitem
end

@inline function default_nitem(::Ampere, ::Type{VecMat}, n, p, ::Type{T}) where {T} #A40
    Nitem = prevpow(2, cld(16, cld(sizeof(T), 2)))
    return Nitem
end
@inline function default_nitem(::RTX1000, ::Type{VecMat}, n, p, ::Type{T}) where {T} #A40
    Nitem = prevpow(2, cld(16, cld(sizeof(T), 2)))
    return Nitem
end

@inline function default_nitem(::A40, ::Type{VecMat}, n, p, ::Type{T}) where {T} #A40
    if n * p * sizeof(T) >= 4 * 10^8 && p * sizeof(T) >= 4 * 1000
        Nitem = prevpow(2, cld(4, cld(sizeof(T), 2)))
    elseif p * sizeof(T) <= 4 * 100
        Nitem = min(16, prevpow(2, cld(64, sizeof(T))))
    else
        Nitem = prevpow(2, cld(16, cld(sizeof(T), 2)))
    end
    return Nitem
end

# @inline function default_nitem(::RTX1000, ::Type{VecMat}, n, p, ::Type{T}) where {T}
#     Nitem = prevpow(2, cld(16, sizeof(T)))
#     return min(Nitem, prevpow(2, n))
# end



param1(::AbstractArch, ::Type{VecMat}) = 2 # tuned for A40
param1(::RTX1000, ::Type{VecMat}) = 4

param2(::AbstractArch, ::Type{VecMat}) = 100 # large enough so cld = 1
param2(::RTX1000, ::Type{VecMat}) = 1

@inline function default_nthreads(arch::AbstractArch, ::Type{VecMat}, n, p, ::Type{T}, Nitem, workgroup, blocks) where T
    p1 = param1(arch, VecMat)
    p2 = param2(arch, VecMat)
    thresh = prevpow(2, max(fld(n, p1), 1))
    if thresh >= workgroup # n large (tall matrix, reduction regime)
        Nblocks = min(max(fld(blocks, p), 1), max(fld(n, workgroup * cld(Nitem, p2)), 1))
        Nthreads = workgroup * Nblocks ÷ 2
    else # n small (large matrix, copy regime)
        Nthreads = cld(thresh, cld(Nitem, p2))
    end
    return Nthreads
end
# ============================================================================
# Parameter resolution
# ============================================================================


function resolve_parameters(
    arch::AbstractArch,
    ::Type{VecMat},
    src::AbstractArray{T},
    Nitem=nothing,
    Nthreads=nothing,
    workgroup=nothing,
    blocks=nothing
) where {T}
    n, p = size(src)
    workgroup = something(workgroup, default_workgroup(arch, VecMat, n, p, T))
    blocks = something(blocks, default_blocks(arch, VecMat, n, p, T))
    Nitem = something(Nitem, default_nitem(arch, VecMat, n, p, T))
    Nthreads = something(Nthreads, default_nthreads(arch, VecMat, n, p, T, Nitem, workgroup, blocks))

    Nitem = min(Nitem, prevpow(2, n))
    Nthreads = min(Nthreads, prevpow(2, max(fld(n, Nitem), 1)))
    workgroup = min(workgroup, prevpow(2, n * p))

    @assert Nthreads * Nitem <= n "Nthreads * Nitem must be <= n"
    return (; Nitem, Nthreads, workgroup, blocks)
end
# ============================================================================
# Buffer allocation
# ============================================================================

"""
    get_allocation(::Type{VecMat}, f, op, x, src, Nblocks=nothing, arch=nothing) -> KernelBuffer

Allocate a `KernelBuffer` for `vecmat!`. Useful for repeated calls.

# Arguments
- `f`: Map function (used to infer intermediate eltype)
- `op`: Reduction operator
- `x`: Input vector or `nothing`
- `src`: Input GPU matrix (used to determine backend and eltype)
- `Nblocks=nothing`: Number of blocks (auto-computed if nothing)
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)

# Returns
A `KernelBuffer` with named fields `partial` and `flag`.

# Examples
```julia
A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 1000)
tmp = KernelForge.get_allocation(VecMat, *, +, x, A)
dst = CUDA.zeros(Float32, 500)

for i in 1:100
    vecmat!(dst, x, A; tmp)
end
```
"""
function get_allocation(
    ::Type{VecMat},
    f::F,
    op::O,
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T},
    Nitem=nothing,
    Nthreads=nothing,
    workgroup=nothing,
    blocks=nothing,
    arch=nothing
) where {T,F<:Function,O<:Function}
    arch = something(arch, detect_arch(src))
    params = resolve_parameters(arch, VecMat, src, Nitem, Nthreads, workgroup, blocks)
    Nblocks = cld(params.Nthreads, params.workgroup)
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))
    _, p = size(src)
    backend = get_backend(src)
    partial = KernelAbstractions.allocate(backend, H, Nblocks * p)
    flag = KernelAbstractions.allocate(backend, UInt8, Nblocks * p)
    return KernelBuffer((; partial, flag))
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
    tmp::TMP=nothing,
    Nitem=nothing,
    Nthreads=nothing,
    workgroup=nothing,
    blocks=nothing,
    arch=nothing
) where {T,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    p = size(src, 2)
    dst = KernelAbstractions.allocate(backend, S, p)
    _vecmat_entry!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, blocks, tmp, arch)
    return dst
end

# Full interface with f and op - allocating version
function vecmat(
    f::F, op::O,
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T};
    g::G=identity,
    tmp::TMP=nothing,
    Nitem=nothing,
    Nthreads=nothing,
    workgroup=nothing,
    blocks=nothing,
    arch=nothing
) where {T,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    p = size(src, 2)
    dst = KernelAbstractions.allocate(backend, S, p)
    _vecmat_entry!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, blocks, tmp, arch)
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
    tmp::TMP=nothing,
    Nitem=nothing,
    Nthreads=nothing,
    workgroup=nothing,
    blocks=nothing,
    arch=nothing
) where {S,T,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    _vecmat_entry!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, blocks, tmp, arch)
end

# Full interface with f and op - in-place version
function vecmat!(
    f::F, op::O,
    dst::AbstractArray{S},
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T};
    g::G=identity,
    tmp::TMP=nothing,
    Nitem=nothing,
    Nthreads=nothing,
    workgroup=nothing,
    blocks=nothing,
    arch=nothing
) where {S,T,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    _vecmat_entry!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, blocks, tmp, arch)
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
    tmp::Union{KernelBuffer,Nothing},
    arch
) where {S,T,F,O,G}
    n, p = size(src)
    if !isnothing(x)
        @assert length(x) == n "Vector length must match matrix rows"
    end
    @assert length(dst) == p "Output length must match matrix columns"

    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))

    arch = something(arch, detect_arch(src))::AbstractArch
    params = resolve_parameters(arch, VecMat, src, Nitem, Nthreads, workgroup, blocks)

    if params.Nthreads > params.workgroup
        tmp = something(tmp, get_allocation(VecMat, f, op, x, src, params.Nitem, params.Nthreads, params.workgroup, params.blocks, arch))
    end

    _vecmat_impl!(f, op, g, dst, x, src, params.Nitem, params.Nthreads, params.workgroup, tmp, H, n, p, arch)
end

# ============================================================================
# Core implementation
# ============================================================================


# KernelBuffer dispatch
function _vecmat_impl!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T},
    Nitem::Int,
    Nthreads::Int,
    workgroup::Int,
    tmp::Union{Nothing,KernelBuffer},
    ::Type{H},
    n::Int,
    p::Int,
    arch::AbstractArch
) where {S,T,F,O,G,H}
    if Nthreads <= workgroup
        _vecmat_impl_single!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, H, arch)
    else
        _vecmat_impl_multi!(f, op, g, dst, x, src, Nitem, Nthreads, workgroup, tmp, H, arch)
    end
end

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
    arch
) where {S,T,F,O,G,H}
    p = size(src, 2)
    backend = get_backend(src)
    ndrange = Nthreads * p
    warpsz = get_warpsize(arch)
    vecmat_kernel!(backend, workgroup, ndrange)(
        f, op, g, dst, x, src, Val(Nitem), Val(Nthreads), nothing, nothing, H, Val(warpsz)
    )
end

# Multi-block case (with synchronization)
function _vecmat_impl_multi!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    x::Union{AbstractArray,Nothing},
    src::AbstractMatrix{T},
    Nitem::Int,
    Nthreads::Int,
    workgroup::Int,
    tmp::KernelBuffer,
    ::Type{H},
    arch
) where {S,T,F,O,G,H}
    p = size(src, 2)
    backend = get_backend(src)
    ndrange = Nthreads * p
    fill!(tmp.arrays.flag, 0x00)
    warpsz = get_warpsize(arch)
    vecmat_kernel!(backend, workgroup, ndrange)(
        f, op, g, dst, x, src, Val(Nitem), Val(Nthreads), tmp.arrays.partial, tmp.arrays.flag, H, Val(warpsz)
    )
end