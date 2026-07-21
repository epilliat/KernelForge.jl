# ============================================================================
# Generalized GEMM — wrapper API, dispatch, tiling defaults
# ============================================================================
#
# Public surface mirrors matvec's four-method shape: allocating + in-place, each
# in a simplified (`A*B`) and a full generalized-op (`f`, `op`, `g`) form. The
# `tA`/`tB` transpose flags ∈ `{:N, :T}` select the per-operand KI layout tag
# (ColMajor / RowMajor) driving the layout-parametric tiled core — all four
# states NN / NT / TN / TT. `:C` (conjugate) is deferred to complex eltypes.
#
# Two-family dispatch (mirrors matvec `generic`/`rowthread`, vecmat `generic`/`mlp`):
#   :generic — K1 register-blocked shared-mem core (`gemm_kernel!`). Handles ANY
#              isbits eltype + ANY (f, op, g); the reduction is seeded from a real
#              value (no operator identity needed).
#   :mma     — K2 tensor-core core (`gemm_mma_kernel!`). PLAIN (*, +) only, HW MMA
#              (WMMA/MFMA), CT ∈ {Float16, BFloat16} accumulating in Float32 on this
#              device. Chosen by `_resolve_family` iff the HW path exists; a forced
#              `family=:mma` on an unsupported config is a loud error (never silent).

# ---------------------------------------------------------------------------
# Generic (K1) per-arch tiling defaults. Returns (BM, BN, BK, TM, TN).
# Invariants the launcher asserts: BM%TM==0, BN%TN==0, wg=(BM÷TM)*(BN÷TN) a warp
# multiple ≤1024, shared tile (BM·BK + BK·BN) elements fits the 48 KB cap.
# ---------------------------------------------------------------------------
@inline function default_gemm_tile(::AbstractArch, M, N, K, ::Type{T}) where {T}
    if sizeof(T) > 16
        return (32, 32, 8, 4, 4)     # wg = 64
    end
    return (64, 64, 8, 4, 4)         # wg = 256
end

# ---------------------------------------------------------------------------
# MMA (K2) accumulation-type default + tile selection.
# ---------------------------------------------------------------------------
# Default accumulation type: bump low-precision floats to Float32 (F16/BF16 MMA
# accumulate in F32; also fixes K1's F16-accumulate quality). Everything else keeps
# its natural map type. `accT` kwarg overrides this for BOTH families.
@inline _default_acc_type(::Type{Float16})      = Float32
@inline _default_acc_type(::Type{Core.BFloat16}) = Float32
@inline _default_acc_type(::Type{H}) where {H}   = H

# Pick an MMA (Mt,Nt,Kt) for (CT, AccT) from the device's `mma_shapes` — prefer the
# square 16×16×16 (most portable). Returns `nothing` when no HW path exists. Never
# a hardcoded shape table — queried on the live backend.
function _mma_pick_shape(backend, ::Type{CT}, ::Type{AccT}) where {CT,AccT}
    best = nothing
    for s in KI.MMA.mma_shapes(backend)
        (s.compute === CT && s.acc === AccT) || continue
        KI.MMA.mma_supported(backend, KI.MMA.MMAConfig{s.M,s.N,s.K,CT,AccT}()) || continue
        if s.M == 16 && s.N == 16 && s.K == 16
            return (Mt=s.M, Nt=s.N, Kt=s.K)      # square — take immediately
        end
        best === nothing && (best = (Mt=s.M, Nt=s.N, Kt=s.K))
    end
    return best
end

# MMA (K2) launch: v1 is ONE warp per block, ONE Mt×Nt MMA tile per block, one
# `mma` per Kt-wide K panel. This is the proven-safe configuration on the current
# toolchain (warp/register tiling miscompiles — see the wall note in
# gemm_mma_kernel.jl). Only wg = warpsize; the block tile IS the MMA tile.
@inline default_gemm_mma_wg(arch::AbstractArch) = get_warpsize(arch)

# ============================================================================
# Public API docstrings
# ============================================================================

"""
    gemm([f, op,] A::AbstractMatrix, B::AbstractMatrix; kwargs...) -> C

Generalized matrix–matrix product. Computes
`C[m,n] = g(op_k f(A[m,k], B[k,n]))` — for standard GEMM (`f=*`, `op=+`,
`g=identity`) this is `C = A * B`. Returns a newly allocated result matrix.

Supports arbitrary isbits element types (custom structs) and arbitrary operators
(the reduction is seeded from a real product, so `op` needs no identity element).
On NVIDIA/AMD with a HW MMA path, plain (`*`,`+`) `Float16`/`BFloat16` inputs
auto-route to a tensor-core kernel accumulating in `Float32`.

# Keyword Arguments
- `g=identity`: epilogue map applied to each reduced `C[m,n]`.
- `tA=:N`, `tB=:N`: per-operand transpose flags ∈ `{:N, :T}` → all four states
  NN / NT / TN / TT (real transpose; `:C` conjugate-transpose deferred to complex).
- `accT=nothing`: accumulation-type override (both families). Defaults to `Float32`
  for `Float16`/`BFloat16` inputs, else the natural map type.
- `family=nothing`: `:generic` (K1) | `:mma` (K2 tensor-core) | `nothing` (auto).
  A forced `:mma` on an unsupported config errors. **`:mma` is currently OPT-IN
  ONLY** — auto never selects it, because the WMMA path returns silently wrong
  results under `--check-bounds=yes` (bounds-check branches diverge the warp inside
  the lockstep MMA region). Use it only with bounds checks off, and validate.
- `BM,BN,BK,TM,TN=nothing`: generic (K1) tile knobs (auto per-arch if `nothing`;
  supplying any of them forces the generic family).
- `arch=nothing`: architecture (auto-detected from `A`).

See also: [`gemm!`](@ref).
"""
function gemm end

"""
    gemm!([f, op,] C, A, B; kwargs...)

In-place form of [`gemm`](@ref): writes `C[m,n] = g(op_k f(A[m,k], B[k,n]))`.
"""
function gemm! end

# ============================================================================
# Buffer allocation — GEMM needs no scratch (no split-K). Empty KernelBuffer so
# `KernelForge.@allocate gemm(A, B)` resolves; the `tmp=` contract stays uniform.
# ============================================================================
function get_allocation(
    ::Type{GEMM},
    f::F, op::O,
    A::AbstractMatrix, B::AbstractMatrix,
    arch=nothing
) where {F<:Function,O<:Function}
    return KernelBuffer((;))
end

# ============================================================================
# Type + parameter resolution
# ============================================================================
# (natural map type, accumulation type, output type) for the given f/g/accT.
@inline function _gemm_types(f::F, g::G, ::Type{TA}, ::Type{TB}, accT) where {F,G,TA,TB}
    Hnat = Base.promote_op(f, TA, TB)
    AccT = accT === nothing ? _default_acc_type(Hnat) : accT
    S = Base.promote_op(g, AccT)
    return (Hnat, AccT, S)
end

function resolve_gemm_parameters(
    arch::AbstractArch, M, N, K, ::Type{T},
    BM, BN, BK, TM, TN
) where {T}
    dBM, dBN, dBK, dTM, dTN = default_gemm_tile(arch, M, N, K, T)
    BM = something(BM, dBM); BN = something(BN, dBN); BK = something(BK, dBK)
    TM = something(TM, dTM); TN = something(TN, dTN)
    @assert BM % TM == 0 "BM ($BM) must be divisible by TM ($TM)"
    @assert BN % TN == 0 "BN ($BN) must be divisible by TN ($TN)"
    wg = (BM ÷ TM) * (BN ÷ TN)
    warpsz = get_warpsize(arch)
    @assert wg <= 1024 "workgroup ($wg) exceeds 1024; shrink BM/BN or grow TM/TN"
    @assert wg % warpsz == 0 "workgroup ($wg) must be a multiple of the warp size ($warpsz)"
    return (; BM, BN, BK, TM, TN, wg)
end

# ============================================================================
# Public API — four methods (simplified / full-op) × (allocating / in-place).
# ============================================================================

# Simplified (A*B), allocating.
function gemm(
    A::AbstractMatrix{TA}, B::AbstractMatrix{TB};
    f::F=*, op::O=+, g::G=identity,
    tA::Symbol=:N, tB::Symbol=:N, accT=nothing, family=nothing,
    BM=nothing, BN=nothing, BK=nothing, TM=nothing, TN=nothing,
    arch=nothing
) where {TA,TB,F<:Function,O<:Function,G<:Function}
    _, _, S = _gemm_types(f, g, TA, TB, accT)
    backend = get_backend(A)
    M, N = _gemm_out_dims(A, B, tA, tB)
    C = KernelAbstractions.allocate(backend, S, M, N)
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family, BM, BN, BK, TM, TN, arch)
    return C
end

# Full generalized-op, allocating.
function gemm(
    f::F, op::O,
    A::AbstractMatrix{TA}, B::AbstractMatrix{TB};
    g::G=identity, tA::Symbol=:N, tB::Symbol=:N, accT=nothing, family=nothing,
    BM=nothing, BN=nothing, BK=nothing, TM=nothing, TN=nothing,
    arch=nothing
) where {TA,TB,F<:Function,O<:Function,G<:Function}
    _, _, S = _gemm_types(f, g, TA, TB, accT)
    backend = get_backend(A)
    M, N = _gemm_out_dims(A, B, tA, tB)
    C = KernelAbstractions.allocate(backend, S, M, N)
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family, BM, BN, BK, TM, TN, arch)
    return C
end

# Simplified (A*B), in-place.
function gemm!(
    C::AbstractMatrix{S},
    A::AbstractMatrix{TA}, B::AbstractMatrix{TB};
    f::F=*, op::O=+, g::G=identity,
    tA::Symbol=:N, tB::Symbol=:N, accT=nothing, family=nothing,
    BM=nothing, BN=nothing, BK=nothing, TM=nothing, TN=nothing,
    arch=nothing
) where {S,TA,TB,F<:Function,O<:Function,G<:Function}
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family, BM, BN, BK, TM, TN, arch)
    return C
end

# Full generalized-op, in-place.
function gemm!(
    f::F, op::O,
    C::AbstractMatrix{S},
    A::AbstractMatrix{TA}, B::AbstractMatrix{TB};
    g::G=identity, tA::Symbol=:N, tB::Symbol=:N, accT=nothing, family=nothing,
    BM=nothing, BN=nothing, BK=nothing, TM=nothing, TN=nothing,
    arch=nothing
) where {S,TA,TB,F<:Function,O<:Function,G<:Function}
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family, BM, BN, BK, TM, TN, arch)
    return C
end

# ============================================================================
# Entry point (validation, family resolution, launch)
# ============================================================================
# Public transpose flag → KI per-operand layout tag. `:N` = natural col-major
# (ColMajor); `:T` = operand stored transposed, read as RowMajor on the parent.
@inline function _gemm_layout(s::Symbol)
    s === :N && return ColMajor()
    s === :T && return RowMajor()
    error("gemm: transpose flag must be :N or :T (got :$s); :C/conjugate-transpose " *
          "is not supported yet (reals only)")
end

# Output (M, N) of the logical product opA(A) * opB(B), honouring transpose flags.
@inline function _gemm_out_dims(A, B, tA::Symbol, tB::Symbol)
    M = tA === :N ? size(A, 1) : size(A, 2)
    N = tB === :N ? size(B, 2) : size(B, 1)
    return (M, N)
end

# Pick the compute family. `:mma` iff plain (*,+), matching CT=TA=TB with a HW MMA
# path for (CT, AccT), and the output is at least one MMA tile. Explicit generic
# tile knobs force `:generic`. A forced `family=:mma` that fails the gate errors.
function _resolve_family(family, backend, f::F, op::O,
                         ::Type{TA}, ::Type{TB}, ::Type{AccT},
                         M, N, K, tA::Symbol, tB::Symbol, user_generic_knobs::Bool) where {F,O,TA,TB,AccT}
    family === :generic && return (:generic, nothing)
    # ⚠ `:mma` is OPT-IN ONLY (never auto-selected) — but no longer for the reason
    # this comment used to give. The old rationale (bounds-check branches diverging
    # the warp inside the WMMA region) was REFUTED: the real cause was in
    # KernelIntrinsics, where overlay-installed MMA entry points left the K-loop
    # accumulator phi initialised to `undef`, and it is FIXED in KI 0.1.14 (which
    # this package now requires). Correctness is no longer the blocker: the suite is
    # green with `:mma` active under `--check-bounds=yes` on both RTX1000 Ada (WMMA)
    # and MI300A gfx942 (MFMA).
    #
    # It stays opt-in purely because it is not yet FASTER: K2 measures 0.85–1.07×
    # K1 on F16 squares, since one 16×16 tile per warp plus a block barrier per K
    # panel leaves the kernel staging-bound with no register reuse. Auto-routing a
    # slower path would be a pessimisation. Flip this to auto once warp/register
    # tiling clears the ≥2× gate — it is a one-line change here.
    family === nothing && return (:generic, nothing)
    plain = (f === Base.:*) && (op === Base.:+)
    # NOTE: the TT (both-RowMajor) MMA specialization miscompiles on the current
    # toolchain (WMMA + KA, see gemm_mma_kernel.jl) → excluded from :mma; it routes
    # to the correct generic K1 (accumulating in AccT). NN/NT/TN are HW MMA.
    tt = (tA === :T && tB === :T)
    shape = (plain && TA === TB && !tt) ? _mma_pick_shape(backend, TA, AccT) : nothing
    ok = shape !== nothing && M >= shape.Mt && N >= shape.Nt && K >= 1
    if family === :mma
        user_generic_knobs && error("gemm: family=:mma does not accept generic tile " *
                                     "knobs (BM/BN/BK/TM/TN)")
        tt && error("gemm: family=:mma is not supported for the TT layout " *
                    "(tA=:T, tB=:T) on this toolchain; use the generic family")
        ok || error("gemm: family=:mma requested but no HW tensor-core path for " *
                    "(compute=$TA, acc=$AccT, f=$f, op=$op) on this device (or output < one MMA tile)")
        return (:mma, shape)
    end
    # auto: explicit generic knobs → generic; else :mma when available.
    (user_generic_knobs || !ok) && return (:generic, nothing)
    return (:mma, shape)
end

function _gemm_entry!(
    f::F, op::O, g::G,
    C::AbstractMatrix{S},
    A::AbstractMatrix{TA},
    B::AbstractMatrix{TB},
    tA::Symbol, tB::Symbol,
    accT, family,
    BM, BN, BK, TM, TN,
    arch
) where {S,TA,TB,F,O,G}
    layA = _gemm_layout(tA)
    layB = _gemm_layout(tB)
    M  = tA === :N ? size(A, 1) : size(A, 2)
    Ka = tA === :N ? size(A, 2) : size(A, 1)
    Kb = tB === :N ? size(B, 1) : size(B, 2)
    N  = tB === :N ? size(B, 2) : size(B, 1)
    @assert Ka == Kb "inner dimensions must match: Aop k=$Ka vs Bop k=$Kb (tA=:$tA, tB=:$tB)"
    K = Ka
    @assert size(C) == (M, N) "C has size $(size(C)), expected ($M, $N)"

    _, AccT, _ = _gemm_types(f, g, TA, TB, accT)
    arch = something(arch, detect_arch(A))::AbstractArch
    backend = get_backend(A)

    user_generic_knobs = !(BM === nothing && BN === nothing && BK === nothing &&
                           TM === nothing && TN === nothing)
    fam, shape = _resolve_family(family, backend, f, op, TA, TB, AccT,
                                 M, N, K, tA, tB, user_generic_knobs)

    if fam === :mma
        return _gemm_launch_mma!(g, C, A, B, layA, layB, shape, TA, AccT, M, N, K, arch)
    end
    return _gemm_launch_generic!(f, op, g, C, A, B, layA, layB, AccT,
                                 BM, BN, BK, TM, TN, M, N, K, arch)
end

function _gemm_launch_generic!(
    f::F, op::O, g::G, C, A, B, layA, layB, ::Type{AccT},
    BM, BN, BK, TM, TN, M, N, K, arch
) where {F,O,G,AccT}
    backend = get_backend(A)
    p = resolve_gemm_parameters(arch, M, N, K, eltype(A), BM, BN, BK, TM, TN)
    nbx = cld(M, p.BM); nby = cld(N, p.BN)
    ndrange = nbx * nby * p.wg      # exact workgroup multiple → no bare-@index OOB
    gemm_kernel!(backend, p.wg)(
        f, op, g, C, A, B, layA, layB,
        Val(p.BM), Val(p.BN), Val(p.BK), Val(p.TM), Val(p.TN),
        AccT, M, N, K;                # AccT is the accumulation type H for K1
        ndrange = ndrange,
    )
    return C
end

function _gemm_launch_mma!(
    g::G, C, A, B, layA, layB, shape, ::Type{CT}, ::Type{AccT}, M, N, K, arch
) where {G,CT,AccT}
    backend = get_backend(A)
    Mt, Nt, Kt = shape.Mt, shape.Nt, shape.Kt
    cfg = KI.MMA.MMAConfig{Mt,Nt,Kt,CT,AccT}()
    ws = default_gemm_mma_wg(arch)     # one warp per block (v1)
    nbx = cld(M, Mt); nby = cld(N, Nt)
    ndrange = nbx * nby * ws
    gemm_mma_kernel!(backend, ws)(
        g, C, A, B, layA, layB, cfg,
        Val(Mt), Val(Nt), Val(Kt), Val(ws),
        AccT, M, N, K;
        ndrange = ndrange,
    )
    return C
end
