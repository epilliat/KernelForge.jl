## TODO: understand relationship of defaults with cache sizes of arch

@inline function default_nblocks(arch::AbstractArch, ::Type{MatVec}, n, p, ::Type{T}) where T
    n * p <= cld(4 * 10^6, sizeof(T)) && n >= 10^4 && return 1 # micro optim
    return nextpow(2, cld(128, floor(Int, sqrt(n))))
end
@inline function default_nblocks(arch::A40, ::Type{MatVec}, n, p, ::Type{T}) where T
    if n <= 10
        Nblocks = cld(2048, n)
    elseif n <= 100
        Nblocks = cld(512, floor(Int, sqrt(n)))
    else
        Nblocks = cld(2^8, floor(Int, sqrt(n)))
    end
    return prevpow(2, Nblocks)
end

@inline function default_workgroup(arch::A40, ::Type{MatVec}, n, p, ::Type{T}) where T
    return 256
end
@inline function default_workgroup(arch::AbstractArch, ::Type{MatVec}, n, p, ::Type{T}) where T
    sizeof(T) > 8 && return 256 # safe guard
    (n * p <= cld(4 * 10^6, sizeof(T)) || (n <= cld(4 * 10^2, sizeof(T)))) && return 128 # small matrices or large matrices
    (n < cld(4 * 10^4, sizeof(T))) && return 256 # intermediary regime
    p <= 10 && return 256 # big tall matrices
    return 512 # big large matrices
end

@inline function default_nitem(arch::AbstractArch, ::Type{MatVec}, n, p, ::Type{T}) where T
    sizeof(T) > 8 && return 1
    n == 1 && return 1
    n <= 10 && return min(max(1, fld(n, 4)), prevpow(2, cld(16, sizeof(T))))
    return prevpow(2, cld(16, sizeof(T)))
end
@inline function default_nitem(arch::A40, ::Type{MatVec}, n, p, ::Type{T}) where T
    if n * p <= 10^7
        return min(2, n)
    end
    if n >= 10^4
        Nitem = 2#cld(sizeof(T), 2)
    elseif n <= 10
        Nitem = cld(n, 4)
    else
        Nitem = cld(16, sizeof(T))
    end
    return prevpow(2, Nitem)
end



@inline function default_chunksz(arch::AbstractArch, ::Type{MatVec}, n, p, ::Type{T}, Nitem, workgroup) where T
    p == 1 && return workgroup
    n <= cld(4 * 10, sizeof(T)) && return nextpow(2, cld(n, 2 * Nitem))
    n <= cld(4 * 100, sizeof(T)) && return 4#prevpow(2, cld(n, 2 * Nitem))
    if n * p <= cld(4 * 10^6, sizeof(T))
        p <= 10 && return 64
        p <= 1000 && return 16
    end
    p <= 100 && return 64

    return cld(workgroup, get_warpsize(arch))
end

@inline function default_chunksz(arch::A40, ::Type{MatVec}, n, p, ::Type{T}, Nitem, workgroup) where T
    p == 1 && return workgroup
    p <= 10 && return 128
    p <= 100 && return 64
    p <= 1000 && n*p > 10^6 && return 32
    p <= 1000 && n*p <= 10^6 && return 16

    p <= 10^4 && n * p >= 10^8 && return 32
    p <= 10^4 && 10^6 < n * p < 10^8 && return 16
    p <= 10^4 && n*p <= 10^6 && return 8

    p <= 10^4 && n * p < 10^8 && return 16

    p <= 10^5 && n*p > 10^6 && return 8
    p <= 10^5 && n*p <= 10^6 && return 4

    if n <= 10
            chunksz=prevpow(2, max(fld(n, Nitem), 1))
        elseif n <= 100
            chunksz=4
        else
            chunksz=8
        end
    if n*p <= 10^6
        chunksz = cld(chunksz,2)
    end

    return chunksz
end





# ============================================================================
# Public API docstrings
# ============================================================================

"""
    matvec([f, op,] src::AbstractMatrix, x; kwargs...) -> GPU array
    matvec!([f, op,] dst, src, x; kwargs...)

Generalized matrix-vector operation with customizable element-wise and reduction operations.

Computes `dst[i] = g(op_j(f(src[i,j], x[j])))` for each row `i`, where `op_j` denotes
reduction over columns. For standard matrix-vector multiplication, this is
`dst[i] = sum_j(src[i,j] * x[j])`.

The allocating version `matvec` returns a newly allocated result vector.
The in-place version `matvec!` writes to `dst`.

# Arguments
- `f`: Binary operation applied element-wise (default: `*`)
- `op`: Reduction operation across columns (default: `+`)
- `dst`: Output vector (in-place versions only)
- `src`: Input matrix
- `x`: Input vector, or `nothing` for row-wise reduction of `src` alone

# Keyword Arguments
- `g=identity`: Unary transformation applied to each reduced row
- `tmp=nothing`: Pre-allocated `KernelBuffer` (or `nothing` to allocate automatically)
- `chunksz=nothing`: Elements per thread (auto-tuned if `nothing`)
- `Nblocks=nothing`: Number of thread blocks (auto-tuned if `nothing`)
- `workgroup=nothing`: Threads per block (auto-tuned if `nothing`)
- `blocks_row=nothing`: Number of blocks used to process a single row; relevant only
  for wide matrices (many columns, few rows) where parallelizing across columns is
  beneficial. Auto-tuned if `nothing`.
- `Nitem=nothing`: Number of rows loaded per thread via vectorised loads. Defaults to 1.
  When `Nitem > 1`, `Nblocks` must be 1 and `chunksz` is set to `workgroup`.
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)

# Examples
```julia
A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)

# Standard matrix-vector multiply: y = A * x
y = matvec(A, x)

# Row-wise sum: y[i] = sum(A[i, :])
y = matvec(A, nothing)

# Row-wise maximum: y[i] = max_j(A[i, j])
y = matvec(identity, max, A, nothing)

# Softmax numerator: y[i] = sum_j(exp(A[i,j] - x[j]))
y = matvec((a, b) -> exp(a - b), +, A, x)

# In-place version
dst = CUDA.zeros(Float32, 1000)
matvec!(dst, A, x)

# With pre-allocated buffer for repeated calls
tmp = KernelForge.get_allocation(MatVec, *, +, A, x)
for i in 1:100
    matvec!(dst, A, x; tmp)
end
```
"""
function matvec end, function matvec! end

# ============================================================================
# Buffer allocation
# ============================================================================

"""
    get_allocation(::Type{MatVec}, f, op, src, x, Nblocks=nothing, arch=nothing) -> KernelBuffer

Allocate a `KernelBuffer` for `matvec!`. Useful for repeated calls.

# Arguments
- `f`: Map function (used to infer intermediate eltype)
- `op`: Reduction operator
- `src`: Input GPU matrix (used to determine backend and eltype)
- `x`: Input vector or `nothing`
- `Nblocks=nothing`: Number of blocks (auto-computed if nothing)
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)

# Returns
A `KernelBuffer` with named fields `partial` and `flag`.

# Examples
```julia
A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)
tmp = KernelForge.get_allocation(MatVec, *, +, A, x)
dst = CUDA.zeros(Float32, 1000)

for i in 1:100
    matvec!(dst, A, x; tmp)
end
```
"""

function get_allocation(
    ::Type{MatVec},
    f::F,
    op::O,
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing,
    Nitem=nothing,
    arch=nothing
) where {T,F<:Function,O<:Function}
    n, p = size(src)
    arch = something(arch, detect_arch(src))
    params = resolve_parameters(
        arch, MatVec, src, chunksz, Nblocks, workgroup, blocks_row, Nitem
    )
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))
    backend = get_backend(src)
    nbatch = cld(n, params.Nitem)
    partial = KernelAbstractions.allocate(backend, NTuple{params.Nitem,H}, nbatch * params.Nblocks)
    flag = KernelAbstractions.allocate(backend, UInt8, nbatch * params.Nblocks)
    return KernelBuffer((; partial, flag))
end

# ============================================================================
# Parameter resolution
# ============================================================================
factor_matvec(::AbstractArch) = 4
factor_matvec(::Ampere) = 2
factor_matvec(::RTX1000) = 8

function resolve_parameters(
    arch::AbstractArch,
    ::Type{MatVec},
    src::AbstractArray{T},
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing,
    Nitem=nothing
) where T
    n, p = size(src)
    warpsz = get_warpsize(arch)
    blocks_row = something(blocks_row, default_blocks(arch))
    workgroup = something(workgroup, default_workgroup(arch, MatVec, n, p, T))
    Nblocks = something(Nblocks, default_nblocks(arch, MatVec, n, p, T))
    Nitem = something(Nitem, default_nitem(arch, MatVec, n, p, T))
    chunksz = something(chunksz, default_chunksz(arch, MatVec, n, p, T, Nitem, workgroup))

    workgroup = max(min(workgroup, prevpow(2, n * p)), warpsz)
    Nblocks = min(Nblocks, prevpow(2, max(fld(p, cld(workgroup, chunksz)), 1)))
    chunksz = min(max(chunksz, nextpow(2, cld(workgroup * Nblocks, p))), workgroup)

    if workgroup == warpsz
        chunksz = workgroup
    end

    @assert cld(workgroup, chunksz) * Nblocks <= p
    @assert ispow2(Nblocks) || chunksz * Nblocks >= workgroup || chunksz * Nblocks <= warpsz
    return (; chunksz, Nblocks, workgroup, blocks_row, Nitem)
end

# ============================================================================
# Public API
# ============================================================================

# Simplified interface (standard mat-vec multiply) - allocating version
function matvec(
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing};
    f::F=*,
    op::O=+,
    g::G=identity,
    tmp::TMP=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing,
    Nitem=nothing,
    arch=nothing
) where {T,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    n = size(src, 1)
    dst = KernelAbstractions.allocate(backend, S, n)
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, Nitem, tmp, arch)
    return dst
end

# Full interface with f and op - allocating version
function matvec(
    f::F, op::O,
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing};
    g::G=identity,
    tmp::TMP=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing,
    Nitem=nothing,
    arch=nothing
) where {T,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    n = size(src, 1)
    dst = KernelAbstractions.allocate(backend, S, n)
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, Nitem, tmp, arch)
    return dst
end

# Simplified interface (standard mat-vec multiply) - in-place version
function matvec!(
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing};
    f::F=*,
    op::O=+,
    g::G=identity,
    tmp::TMP=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing,
    Nitem=nothing,
    arch=nothing
) where {S,T,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, Nitem, tmp, arch)
end

# Full interface with f and op - in-place version
function matvec!(
    f::F, op::O,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing};
    g::G=identity,
    tmp::TMP=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing,
    Nitem=nothing,
    arch=nothing
) where {S,T,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, Nitem, tmp, arch)
end

# ============================================================================
# Entry point (validation and parameter resolution)
# ============================================================================

function _matvec_entry!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Union{Int,Nothing},
    Nblocks::Union{Int,Nothing},
    workgroup::Union{Int,Nothing},
    blocks_row::Union{Int,Nothing},
    Nitem::Union{Int,Nothing},
    tmp::Union{KernelBuffer,Nothing},
    arch
) where {S,T,F,O,G}
    n, p = size(src)
    if !isnothing(x)
        @assert length(x) == p "Vector length must match matrix columns"
    end
    @assert length(dst) == n "Output length must match matrix rows"

    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))

    arch = something(arch, detect_arch(src))::AbstractArch
    params = resolve_parameters(
        arch, MatVec, src, chunksz, Nblocks, workgroup, blocks_row, Nitem
    )
    if params.Nblocks > 1
        tmp = something(tmp, get_allocation(MatVec, f, op, src, x, params.chunksz, params.Nblocks, params.workgroup, params.blocks_row, params.Nitem, arch))
    end
    #@show params
    _matvec_impl!(f, op, g, dst, src, x, params.chunksz, params.Nblocks, params.workgroup, params.Nitem, tmp, H, n, p, arch)
end

# ============================================================================
# Core implementation
# ============================================================================

# KernelBuffer dispatch
function _matvec_impl!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Int,
    Nblocks::Int,
    workgroup::Int,
    Nitem::Int,
    tmp::Union{Nothing,KernelBuffer},
    ::Type{H},
    n::Int,
    p::Int,
    arch::AbstractArch
) where {S,T,F,O,G,H}
    if Nblocks == 1
        _matvec_impl_single!(f, op, g, dst, src, x, chunksz, workgroup, Nitem, H, arch)
    else
        _matvec_impl_multi!(f, op, g, dst, src, x, chunksz, Nblocks, Nitem, workgroup, tmp, H, arch)
    end
end

# Single-block case: dispatch to scalar or vectorised kernel
function _matvec_impl_single!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Int,
    workgroup::Int,
    Nitem::Int,
    ::Type{H},
    arch
) where {S,T,F,O,G,H}
    n, p = size(src)
    backend = get_backend(src)
    warpsz = get_warpsize(arch)
    ndrange = cld(n, chunksz * Nitem) * workgroup
    matvec_kernel!(backend, workgroup, ndrange)(
        f, op, g, dst, src, x, Val(chunksz), Val(1), Val(Nitem), nothing, nothing, H, Val(warpsz)
    )
    # else
    #     ndrange = cld(n, Nitem)
    #     matvec_vload_kernel!(backend, workgroup, ndrange)(
    #         f, op, g, dst, src, x, Val(Nitem), H
    #     )
    # end
end

# Multi-block case (with synchronization)
function _matvec_impl_multi!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Int,
    Nblocks::Int,
    Nitem::Int,
    workgroup::Int,
    tmp::KernelBuffer,
    ::Type{H},
    arch
) where {S,T,F,O,G,H}
    n, p = size(src)
    backend = get_backend(src)
    ndrange = cld(n, chunksz * Nitem) * Nblocks * workgroup
    fill!(tmp.arrays.flag, 0x00)
    warpsz = get_warpsize(arch)
    matvec_kernel!(backend, workgroup, ndrange)(
        f, op, g, dst, src, x, Val(chunksz), Val(Nblocks), Val(Nitem), tmp.arrays.partial, tmp.arrays.flag, H, Val(warpsz)
    )
end