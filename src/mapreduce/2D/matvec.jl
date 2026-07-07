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
    matvec([f, op,] src::AbstractMatrix, x; kwargs...) -> GPU vector

Generalized matrix-vector operation with customizable element-wise and reduction operations.

Computes `y[i] = g(op_j(f(src[i,j], x[j])))` for each row `i`, where `op_j` denotes
reduction over columns. For standard matrix-vector multiplication, this is
`y[i] = sum_j(src[i,j] * x[j])`. Returns a newly allocated result vector.

# Arguments
- `f`: Binary operation applied element-wise (default: `*`)
- `op`: Reduction operation across columns (default: `+`)
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
```

See also: [`matvec!`](@ref).
"""
function matvec end

"""
    matvec!([f, op,] dst, src::AbstractMatrix, x; kwargs...)

In-place form of [`matvec`](@ref): writes `dst[i] = g(op_j(f(src[i,j], x[j])))`.

# Examples
```julia
A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)
dst = CUDA.zeros(Float32, 1000)

# Standard matrix-vector multiply
matvec!(dst, A, x)

# With pre-allocated buffer for repeated calls
tmp = KernelForge.get_allocation(MatVec, *, +, A, x)
for i in 1:100
    matvec!(dst, A, x; tmp)
end
```

See [`matvec`](@ref) for the full keyword-argument list.
"""
function matvec! end

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
    # Tuning lookup: returns (; Nitem, chunksz, Nblocks, workgroup) if this
    # arch has a `data/tuning/<arch>.jl` entry covering (T, n, p), else nothing.
    # Used to fill any knob the caller didn't explicitly override; falls
    # through to default_* heuristics if there's no tuning data.
    tuned = lookup_matvec(arch, T, n, p)
    # Only GENERIC tuned cells fill the generic knobs below. A row-thread cell
    # is consumed earlier in `_matvec_entry!`; if we reach here with one (e.g.
    # the caller forced an explicit knob), ignore it and use the heuristics.
    tuned = (tuned !== nothing && tuned.kernel === :generic) ? tuned : nothing
    workgroup = something(workgroup,
                          tuned === nothing ? nothing : tuned.workgroup,
                          default_workgroup(arch, MatVec, n, p, T))
    Nblocks = something(Nblocks,
                        tuned === nothing ? nothing : tuned.Nblocks,
                        default_nblocks(arch, MatVec, n, p, T))
    Nitem = something(Nitem,
                      tuned === nothing ? nothing : tuned.Nitem,
                      default_nitem(arch, MatVec, n, p, T))
    chunksz = something(chunksz,
                        tuned === nothing ? nothing : tuned.chunksz,
                        default_chunksz(arch, MatVec, n, p, T, Nitem, workgroup))

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
# Row-thread fast path (small-n..square-band, large-p) — dispatch + launch.
# ============================================================================
# See the "Row-thread path" note in matvec_kernel.jl (matvec_rowthread_kernel! /
# matvec_rowthread_reduce!). The tiled kernel maps 1 thread = 1 row and streams
# coalesced columns; the grid tiles both dims (row_tiles × ncb column-splits).
#
# `_matvec_use_rowthread` is the UNTUNED-ARCH fallback heuristic (used only when
# there's no autotune cell selecting a kernel): fire for the wide/square band
# where the row-thread kernel was validated to beat the generic one — n ≥ 64 to
# fill ≥2 warps, p wide enough to amortize the split reduce. On a tuned arch the
# autotune's per-cell `kernel` tag decides instead (see _matvec_entry!). Default
# heuristic OFF for every arch except A100 (measured 49%→76% of peak,
# 2.28×→1.05× cuBLAS at n=100,p=1e7; square/middle at parity). The `ncb` formula
# here is a rough fill-the-GPU estimate; the autotune finds the true per-shape
# optimum (spans a 256× range, non-monotonic → not formula-friendly).
_matvec_use_rowthread(::AbstractArch, n::Int, p::Int, ::Type{T}) where {T} = false
_matvec_use_rowthread(::A100, n::Int, p::Int, ::Type{T}) where {T} =
    n >= 64 && p >= 65536 && p >= 8n

# Rough fallback column-split for the untuned heuristic path: enough row_tiles ×
# ncb blocks to fill the GPU, clamped so partial[ncb·n] stays modest. ~512 at
# small n down to a few at n≈1e5.
_rowthread_default_ncb(n::Int, p::Int) = clamp(cld(3 * 10^5, n), 1, 512)

function _matvec_rowthread_impl!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    ::Type{H}, n::Int, p::Int, arch::AbstractArch;
    U::Int=16, wg::Int=256, ncb::Int=_rowthread_default_ncb(n, p)
) where {S,T,F,O,G,H}
    backend = get_backend(src)
    warpsz  = get_warpsize(arch)
    wg      = clamp(cld(min(wg, n), warpsz) * warpsz, warpsz, 1024)  # warp-multiple, ≥ enough for n
    ncol    = cld(p, ncb)
    ncbe    = cld(p, ncol)                                  # active column-blocks (each ≥1 column)
    rtiles  = cld(n, wg)                                    # row-tiles to cover all n rows
    partial = KernelAbstractions.allocate(backend, H, ncbe * n)
    matvec_rowthread_kernel!(backend, wg)(
        f, op, partial, src, x, n, Val(U), ncol, p, wg, ncbe; ndrange = wg * rtiles * ncbe)
    rwg = clamp(cld(min(n, 256), warpsz) * warpsz, warpsz, 256)
    matvec_rowthread_reduce!(backend, rwg)(op, g, dst, partial, n, ncbe; ndrange = n)
    return dst
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

    # Row-thread fast path — only when the caller left the launch knobs at their
    # defaults (explicit knobs mean "use the generic kernel"). Two sources:
    #   (1) a tuned cell whose autotune-picked family is :rowthread — use its
    #       (U, ncb, workgroup) directly (data-driven, per exact shape);
    #   (2) no tuned rowthread cell, but the fallback heuristic fires for the
    #       wide/square band on an untuned arch.
    if isnothing(chunksz) && isnothing(Nblocks) && isnothing(workgroup) &&
       isnothing(Nitem)
        tuned = lookup_matvec(arch, T, n, p)
        if tuned !== nothing && tuned.kernel === :rowthread
            return _matvec_rowthread_impl!(f, op, g, dst, src, x, H, n, p, arch;
                                           U = tuned.U, wg = tuned.workgroup, ncb = tuned.ncb)
        elseif !_arch_has_rowthread_tuning(arch) && _matvec_use_rowthread(arch, n, p, T)
            # Transition / untuned-arch fallback: the arch has no row-thread cells
            # yet, so the wide/square heuristic overrides any stale generic cell
            # in that band (validated to beat it). Goes quiet once re-tuned.
            return _matvec_rowthread_impl!(f, op, g, dst, src, x, H, n, p, arch)
        end
    end

    params = resolve_parameters(
        arch, MatVec, src, chunksz, Nblocks, workgroup, blocks_row, Nitem
    )
    if params.Nblocks > 1
        tmp = something(tmp, get_allocation(MatVec, f, op, src, x, params.chunksz, params.Nblocks, params.workgroup, params.blocks_row, params.Nitem, arch))
    end
    #@show params
    _matvec_impl!(f, op, g, dst, src, x, params.chunksz, params.Nblocks, params.workgroup, params.Nitem, tmp, H, n, p, arch)
    return dst   # `!` returns the mutated dst (matches the row-thread fast-path return)
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
    matvec_kernel!(backend, workgroup)(
        f, op, g, dst, src, x, Val(chunksz), Val(1), Val(Nitem), nothing, nothing, H, Val(warpsz);
        ndrange = ndrange,
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
    matvec_kernel!(backend, workgroup)(
        f, op, g, dst, src, x, Val(chunksz), Val(Nblocks), Val(Nitem), tmp.arrays.partial, tmp.arrays.flag, H, Val(warpsz);
        ndrange = ndrange,
    )
end