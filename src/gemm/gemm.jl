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
#              (WMMA/MFMA), for whatever (compute, acc) pairs the device announces
#              via `KI.MMA.mma_shapes` — F16/BF16→F32, plus F64 (CDNA3 MFMA 16×16×4)
#              and Int8→Int32 where the hardware has them. OPT-IN: `_resolve_family`
#              never auto-selects it (see the comment there); a forced `family=:mma`
#              on an unsupported config is a loud error (never a silent fallback).

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
# Int8 accumulates in Int32 — the same widening every BLAS and the Int8 MMA
# hardware itself does. An Int8 accumulator overflows at 127, i.e. after a
# handful of terms, so `Int8` was the wrong default for any real GEMM; it also
# made `_mma_pick_shape(Int8, Int8)` return nothing and silently miss the
# Int8→Int32 tensor-core path (16×16×32 / 32×32×16 on gfx942).
# NOTE: deliberately NOT extended to Int16/UInt8/UInt16 — the same overflow
# argument applies, but widening them is a visible output-eltype change with no
# hardware path to justify it, so it stays a maintainer decision.
# `Bool` must keep mapping to itself: the boolean-semiring path (f=&, op=|)
# depends on the accumulator staying Bool.
@inline _default_acc_type(::Type{Int8})         = Int32
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

# MMA (K2) warp/register tiling knobs. Returns (WM, WN, NWM, NWN, WK) — MMA tiles
# per warp (WM×WN), warps per block (NWM×NWN), and k-steps per staged panel (WK).
# The block tile follows by construction: BM = NWM·WM·Mt, BN = NWN·WN·Nt, BK = WK·Kt.
#
# Registers per lane scale as WM·WN·(acc frag) + (WM+WN)·(operand frag); shared
# bytes as `_gemm_mma_shmem`. The tiers below are the measured RTX1000 Ada winners
# (no autotune this pass — a hand sweep of ~50 configs at n=512/1024/2048).
# WM=2, WN=4 is a broad plateau: growing either past that spills (WM=8 costs 3×)
# and non-power-of-two WM/WN wrecks the shared leading dimension. Small outputs
# step down so a block is not mostly padding, and so the grid still fills the
# device.
function default_gemm_mma_tile(::AbstractArch, M, N, K, ::Type{CT}, ::Type{AccT},
                               Mt, Nt, Kt) where {CT,AccT}
    W4 = max(1, 64 ÷ Kt)                    # 64-deep K panel
    W2 = max(1, 32 ÷ Kt)                    # 32-deep K panel
    if M >= 64Mt && N >= 64Nt
        return (2, 4, 8, 2, W4)             # 256×128×64, 16 warps, 32×64 per warp
    elseif M >= 16Mt && N >= 16Nt
        return (2, 4, 4, 2, W4)             # 128×128×64, 8 warps
    elseif M >= 4Mt && N >= 4Nt
        return (2, 2, 2, 2, W2)             # 64×64×32, 4 warps, 32×32 per warp
    elseif M >= 2Mt && N >= 2Nt
        return (1, 1, 2, 2, W2)             # 32×32×32, 4 warps
    end
    return (1, 1, 1, 1, W2)                 # one warp, one MMA tile
end

# ============================================================================
# Public API docstrings
# ============================================================================

"""
    gemm([f, op,] A::AbstractMatrix, B::AbstractMatrix; kwargs...) -> C

Generalized matrix-matrix operation with customizable element-wise and reduction
operations.

Computes `C[m,n] = g(op_k(f(A[m,k], B[k,n])))`, the reduction running over the
inner dimension `k`. For standard matrix multiplication (`f=*`, `op=+`,
`g=identity`) this is `C = A * B`. Returns a newly allocated result matrix.

Arbitrary isbits element types (custom structs included) and arbitrary operators
are supported: `f` need not be a multiplication and may change type
(`f: TA × TB → H`). The accumulator is seeded from a real computed value
`f(A[m,1], B[1,n])` rather than `zero(H)`, so `op` needs no identity element; and
because there is no split-K the reduction is a strictly ordered left fold, so `op`
need not be associative or commutative either.

# Arguments
- `f`: Binary operation applied element-wise (default: `*`)
- `op`: Reduction operation across the inner dimension (default: `+`)
- `A`, `B`: Input matrices — logically `M×K` and `K×N` after the transpose flags

# Keyword Arguments
- `g=identity`: epilogue map applied to each reduced `C[m,n]`; the output element
  type is `Base.promote_op(g, accT)`.
- `tA=:N`, `tB=:N`: per-operand transpose flags ∈ `{:N, :T}` → all four states
  NN / NT / TN / TT (real transpose; `:C` conjugate-transpose deferred to complex).
- `accT=nothing`: accumulation-type override (both families). Defaults to `Float32`
  for `Float16`/`BFloat16` and `Int32` for `Int8` (so an `Int8` × `Int8` product
  returns an `Int32` matrix), else the natural map type — in particular `Bool` stays
  `Bool`, which the boolean semiring relies on.
- `family=nothing`: `:generic` (K1, the universal path) | `:mma` (K2 tensor-core) |
  `nothing` (auto). `:mma` is **opt-in** — auto always picks `:generic` — and a
  forced `:mma` on an unsupported config is a loud error, never a silent fallback.
  K2 needs plain (`*`,`+`), `eltype(A) === eltype(B)`, and a (compute, accumulate)
  pair the device announces via `KI.MMA.mma_shapes`: `Float16`/`BFloat16` → `Float32`,
  plus `Float64` (CDNA3 MFMA 16×16×4) and `Int8` → `Int32` where the hardware has
  them. The silicon implements one fixed multiply-add, so a custom `(f, op)` can never
  be hardware-accelerated — every such call correctly runs K1. K2 vs K1 on RTX1000 Ada,
  F16 square: 0.90× @256, 2.97× @512, 5.81× @1024, 6.53× @2048, 7.49× @4096 (~10.3
  TFLOP/s at 2048³ ≈ 54% of cuBLAS on the same part) — that small-size crossover is
  why auto does not select it.
- `BM,BN,BK,TM,TN=nothing`: generic (K1) tile knobs (auto per-arch if `nothing`;
  supplying any of them forces the generic family, and is an error together with
  `family=:mma`).
- `WM,WN,NWM,NWN,WK=nothing`: MMA (K2) warp/register tiling knobs — MMA tiles per
  warp (`WM×WN`), warps per block (`NWM×NWN`), k-steps per staged panel (`WK`).
  The block tile follows: `BM = NWM·WM·Mt`, `BN = NWN·WN·Nt`, `BK = WK·Kt`.
- `PC,PA=nothing`: shared-tile leading-dimension PADDING, in elements, for the
  compute-type (As/Bs) and accumulator (Ds) tiles. Defaults to one 16-byte segment
  (8 for `Float16`, 4 for `Float32`); `0` disables. Breaks the bank conflicts of the
  column-strided fragment loads — worth ~1.33× on RTX1000 Ada (F16 2048³: 6.9 → 9.1
  TFLOP/s). `PC` carries essentially all of it; `PA` (the epilogue tile) is noise.
- `arch=nothing`: architecture (auto-detected from `A`).

# Examples
```julia
A = CUDA.rand(Float32, 512, 256)
B = CUDA.rand(Float32, 256, 128)

# Standard product: C = A * B
C = gemm(A, B)

# Transposed operand: C = Aᵀ * B, with A stored 256×512
C = gemm(CUDA.rand(Float32, 256, 512), B; tA=:T)

# Tropical (max, +) semiring: C[m,n] = max_k(A[m,k] + B[k,n])
C = gemm(+, max, A, B)

# Boolean semiring (reachability): C[m,n] = |_k (A[m,k] & B[k,n])
R = gemm(&, |, CuArray(rand(Bool, 512, 256)), CuArray(rand(Bool, 256, 128)))

# Tensor cores, opt-in: Float16 inputs accumulating in Float32
A16 = CuArray(rand(Float16, 2048, 2048)); B16 = CuArray(rand(Float16, 2048, 2048))
C16 = gemm(A16, B16; family=:mma)
```

See also: [`gemm!`](@ref).
"""
function gemm end

"""
    gemm!([f, op,] C, A, B; kwargs...)

In-place form of [`gemm`](@ref): writes `C[m,n] = g(op_k(f(A[m,k], B[k,n])))` into
the caller's `C`, which must have size `(M, N)` and hold the epilogue's output type.

# Examples
```julia
A = CUDA.rand(Float32, 512, 256)
B = CUDA.rand(Float32, 256, 128)
C = CUDA.zeros(Float32, 512, 128)

gemm!(C, A, B)                       # C = A * B
gemm!(+, max, C, A, B)               # tropical product, in place
```

See [`gemm`](@ref) for the full keyword-argument list.
"""
function gemm! end

# ============================================================================
# Buffer allocation
# ============================================================================

"""
    get_allocation(::Type{GEMM}, f, op, A, B, arch=nothing) -> KernelBuffer

Allocate a `KernelBuffer` for `gemm!`. GEMM needs **no scratch**: every block owns
a full tile of `C` and there is no split-K, so this returns an EMPTY buffer. It
exists only so `KernelForge.@allocate gemm(*, +, A, B)` resolves and the allocation
contract stays uniform across operations — `gemm!` has no `tmp` keyword, and
repeated calls allocate nothing beyond `C` itself.
"""
function get_allocation(
    ::Type{GEMM},
    f::F, op::O,
    A::AbstractMatrix, B::AbstractMatrix,
    arch=nothing
) where {F<:Function,O<:Function}
    return KernelBuffer((;))
end

# Convenience arity matching the simplified `gemm(A, B)` entry point, so
# `KernelForge.@allocate gemm(A, B)` resolves like the call it mirrors. No
# ambiguity with the method above: an `AbstractMatrix` is never a `Function`.
function get_allocation(
    ::Type{GEMM},
    A::AbstractMatrix, B::AbstractMatrix,
    arch=nothing
)
    return get_allocation(GEMM, *, +, A, B, arch)
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
    WM=nothing, WN=nothing, NWM=nothing, NWN=nothing, WK=nothing,
    PC=nothing, PA=nothing,
    arch=nothing
) where {TA,TB,F<:Function,O<:Function,G<:Function}
    _, _, S = _gemm_types(f, g, TA, TB, accT)
    backend = get_backend(A)
    M, N = _gemm_out_dims(A, B, tA, tB)
    C = KernelAbstractions.allocate(backend, S, M, N)
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family,
                 BM, BN, BK, TM, TN, WM, WN, NWM, NWN, WK, PC, PA, arch)
    return C
end

# Full generalized-op, allocating.
function gemm(
    f::F, op::O,
    A::AbstractMatrix{TA}, B::AbstractMatrix{TB};
    g::G=identity, tA::Symbol=:N, tB::Symbol=:N, accT=nothing, family=nothing,
    BM=nothing, BN=nothing, BK=nothing, TM=nothing, TN=nothing,
    WM=nothing, WN=nothing, NWM=nothing, NWN=nothing, WK=nothing,
    PC=nothing, PA=nothing,
    arch=nothing
) where {TA,TB,F<:Function,O<:Function,G<:Function}
    _, _, S = _gemm_types(f, g, TA, TB, accT)
    backend = get_backend(A)
    M, N = _gemm_out_dims(A, B, tA, tB)
    C = KernelAbstractions.allocate(backend, S, M, N)
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family,
                 BM, BN, BK, TM, TN, WM, WN, NWM, NWN, WK, PC, PA, arch)
    return C
end

# Simplified (A*B), in-place.
function gemm!(
    C::AbstractMatrix{S},
    A::AbstractMatrix{TA}, B::AbstractMatrix{TB};
    f::F=*, op::O=+, g::G=identity,
    tA::Symbol=:N, tB::Symbol=:N, accT=nothing, family=nothing,
    BM=nothing, BN=nothing, BK=nothing, TM=nothing, TN=nothing,
    WM=nothing, WN=nothing, NWM=nothing, NWN=nothing, WK=nothing,
    PC=nothing, PA=nothing,
    arch=nothing
) where {S,TA,TB,F<:Function,O<:Function,G<:Function}
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family,
                 BM, BN, BK, TM, TN, WM, WN, NWM, NWN, WK, PC, PA, arch)
    return C
end

# Full generalized-op, in-place.
function gemm!(
    f::F, op::O,
    C::AbstractMatrix{S},
    A::AbstractMatrix{TA}, B::AbstractMatrix{TB};
    g::G=identity, tA::Symbol=:N, tB::Symbol=:N, accT=nothing, family=nothing,
    BM=nothing, BN=nothing, BK=nothing, TM=nothing, TN=nothing,
    WM=nothing, WN=nothing, NWM=nothing, NWN=nothing, WK=nothing,
    PC=nothing, PA=nothing,
    arch=nothing
) where {S,TA,TB,F<:Function,O<:Function,G<:Function}
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family,
                 BM, BN, BK, TM, TN, WM, WN, NWM, NWN, WK, PC, PA, arch)
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
    # Validate FIRST: an unrecognised symbol (a typo like `:mmma`) used to fall
    # through to the auto branch with `:mma` enabled, silently giving behaviour the
    # caller did not ask for. A wrong knob must fail loudly, like every other
    # unsupported request here.
    family === nothing || family === :generic || family === :mma ||
        error("gemm: unknown family=$(repr(family)); expected :generic, :mma, or nothing (auto)")
    family === :generic && return (:generic, nothing)
    # ⚠ `:mma` is OPT-IN ONLY (never auto-selected). Not for correctness — the old
    # rationale (bounds-check branches diverging the warp inside the WMMA region)
    # was REFUTED; the real cause was KernelIntrinsics leaving the K-loop
    # accumulator phi `undef` under overlay dispatch, FIXED in KI 0.1.14. The suite
    # is green with `:mma` under `--check-bounds=yes`.
    #
    # Nor is it speed any more. Warp/register tiling (`gemm_mma_kernel!`) took K2
    # from 0.85–1.07× K1 to (RTX1000 Ada, F16 square, xp screen — NOT a deposited
    # benchmark) 2.97× @512, 5.81× @1024, 6.53× @2048, 7.49× @4096, i.e. 10.3
    # TFLOP/s and ~54% of cuBLAS `gemmEx`.
    #
    # What is left is the CROSSOVER: at n=256 K2 is still 0.90× K1 (both are
    # launch-overhead bound there), so flipping auto would regress small F16
    # products. Auto-selection wants a measured size gate, ideally from autotune,
    # across more than one arch — a maintainer decision, not a silent default.
    # When it is taken, it is this one line.
    family === nothing && return (:generic, nothing)
    plain = (f === Base.:*) && (op === Base.:+)
    # All FOUR transpose states are HW MMA. TT used to be excluded here because it
    # "miscompiled" — that was the KI `undef`-accumulator bug wearing another mask,
    # and TT is exact (rel err ~1e-6) since KI 0.1.14. The Phase-2 staging already
    # normalises every layout into col-major-logical shared tiles, so there is no
    # per-layout MMA variant to get wrong.
    shape = (plain && TA === TB) ? _mma_pick_shape(backend, TA, AccT) : nothing
    ok = shape !== nothing && M >= shape.Mt && N >= shape.Nt && K >= 1
    if family === :mma
        user_generic_knobs && error("gemm: family=:mma does not accept generic tile " *
                                     "knobs (BM/BN/BK/TM/TN)")
        ok || error("gemm: family=:mma requested but no HW tensor-core path for " *
                    "(compute=$TA, acc=$AccT, f=$f, op=$op) on this device (or output < one MMA tile)")
        return (:mma, shape)
    end
    # Anything else (an unrecognised `family` symbol): explicit generic knobs →
    # generic; else :mma when available. `family === nothing` returned above, so
    # AUTO never reaches this line — flipping auto on is the single return above.
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
    WM, WN, NWM, NWN, WK, PC, PA,
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
        return _gemm_launch_mma!(g, C, A, B, layA, layB, shape, TA, AccT,
                                 WM, WN, NWM, NWN, WK, PC, PA, M, N, K, arch)
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

# Resolve the MMA warp/register tiling. Returns everything the launcher needs;
# `shmem` comes from the ONE pure sizing fn the kernel also uses.
function resolve_gemm_mma_parameters(
    arch::AbstractArch, backend, M, N, K, ::Type{CT}, ::Type{AccT},
    Mt, Nt, Kt, WM, WN, NWM, NWN, WK, PC, PA
) where {CT,AccT}
    dWM, dWN, dNWM, dNWN, dWK = default_gemm_mma_tile(arch, M, N, K, CT, AccT, Mt, Nt, Kt)
    WM = something(WM, dWM); WN = something(WN, dWN)
    NWM = something(NWM, dNWM); NWN = something(NWN, dNWN); WK = something(WK, dWK)
    PC = something(PC, default_mma_pad(CT)); PA = something(PA, default_mma_pad(AccT))
    @assert WM >= 1 && WN >= 1 && NWM >= 1 && NWN >= 1 && WK >= 1 "MMA tile knobs must be ≥ 1"
    ws = get_warpsize(arch)
    wg = NWM * NWN * ws
    @assert wg <= 1024 "workgroup ($wg) exceeds 1024; shrink NWM/NWN"
    BM = NWM * WM * Mt; BN = NWN * WN * Nt; BK = WK * Kt
    shmem = _gemm_mma_shmem(CT, AccT, BM, BN, BK, NWN * Nt, PC, PA)
    cap = KI.max_dynamic_localmem(backend)
    @assert shmem <= cap "gemm :mma tile needs $shmem B of workgroup memory, device cap is $cap B; shrink WM/WN/NWM/NWN/WK"
    return (; WM, WN, NWM, NWN, WK, PC, PA, ws, wg, BM, BN, BK, shmem)
end

function _gemm_launch_mma!(
    g::G, C, A, B, layA, layB, shape, ::Type{CT}, ::Type{AccT},
    WM, WN, NWM, NWN, WK, PC, PA, M, N, K, arch
) where {G,CT,AccT}
    backend = get_backend(A)
    Mt, Nt, Kt = shape.Mt, shape.Nt, shape.Kt
    cfg = KI.MMA.MMAConfig{Mt,Nt,Kt,CT,AccT}()
    p = resolve_gemm_mma_parameters(arch, backend, M, N, K, CT, AccT,
                                    Mt, Nt, Kt, WM, WN, NWM, NWN, WK, PC, PA)
    nbx = cld(M, p.BM); nby = cld(N, p.BN)
    ndrange = nbx * nby * p.wg          # exact workgroup multiple → no bare-@index OOB
    KI.launch!(
        gemm_mma_kernel!(backend, p.wg),
        g, C, A, B, layA, layB, cfg,
        Val(Mt), Val(Nt), Val(Kt), Val(p.ws),
        Val(p.WM), Val(p.WN), Val(p.NWM), Val(p.NWN), Val(p.WK), Val(p.PC), Val(p.PA),
        AccT, M, N, K;
        ndrange = ndrange, shmem = p.shmem,
    )
    return C
end
