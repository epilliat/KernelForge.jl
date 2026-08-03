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

# Does the (compute-type) tensor path reduce precision vs the generic kernel on the
# SAME eltype? True for Float32 → the NVIDIA tensor path is TF32 (mantissa truncated
# to 10 bits, ~1e-3 rel err) — so `family=nothing` (auto) must NOT pick it; it stays
# opt-in behind `family=:mma`. F16/BF16/F64/Int8 tensor paths match their eltype's
# precision (F32/Int32 accumulate) → not lossy. A pure predicate on the input eltype.
@inline _mma_lossy(::Type{Float32}) = true
@inline _mma_lossy(::Type{T}) where {T} = false

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
    # TF32 path (Float32 in, m16n8k8 → Mt=16, Nt=8, Kt=8) has a DIFFERENT sweet spot
    # than F16 (m16n16k16): the scalar tf32 fragment load (no ldmatrix — the b16 wall)
    # makes it load-bound, so the generic-tile default (WK=8, 256×64) is poor. A hand
    # sweep at RTX1000 Ada square 2048 found 128×128 / 4-warp / WK=4 (+DB) best:
    # 2.6 → 4.8 TFLOP/s (2× the generic-F32 kernel; still ~3.5× behind cuBLAS TF32 —
    # the residual is the load path, an M3 rewrite). Autotune can refine per shape.
    # `WK=32÷Kt=4`, a 32-deep panel that keeps the doubled (DB) tile inside the cap.
    if CT === Float32
        if M >= 16Mt && N >= 16Nt
            return (4, 8, 2, 2, W2)         # 128×128×32, 4 warps  (square/large)
        elseif M >= 4Mt && N >= 4Nt
            return (2, 4, 2, 2, W2)         # 64×64×32, 4 warps
        elseif M >= 2Mt && N >= 2Nt
            return (1, 2, 2, 2, W2)         # 32×32×32, 4 warps
        end
        return (1, 1, 1, 1, W2)
    end
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

# Double-buffered shared staging (see the DOUBLE BUFFERING note in
# gemm_mma_kernel.jl). Correct and worth 1.08–1.29× at a FIXED WK=2 tile, but
# MEASURED a wash on the shipped default path (RTX1000 Ada, F16 square: DB-auto vs
# DB-off = 0.98–1.02×). The reason is structural: the default tile tiers pick
# WK=4 deep-K panels, which already amortize staging (so DB helps them least) and
# whose doubled As/Bs exceed the device cap (256×128×64 → 104 KB > 99 KB on Ada),
# so DB cannot even engage there; the tiles where DB wins are non-default WK=2
# wide-warp configs an autotuner would pick. And the real gap to cuBLAS is the
# async global→shared copy (`cp.async`), which KI does not yet expose — our
# staging goes global→register→shared, capping the overlap any DB can extract.
# So DB stays an OPT-IN knob (pass `DB=true`, or let a future autotune set it per
# tile); the DEFAULT is OFF, making the shipped path identical to the pre-DB one.
# The fit rule below is what a caller/autotune gets when it asks for `DB=true` via
# the resolver default only where sensible — here we simply return false.
function default_gemm_mma_db(::AbstractArch, M, N, K, ::Type{CT}, ::Type{AccT},
                             BM, BN, BK, DN, PC, PA, cap) where {CT,AccT}
    # TF32 (Float32) is load-bound (scalar fragment loads), so the cp.async + DB
    # overlap of global→shared staging pays — unlike the F16 default tiers, whose
    # deep-K panels already amortize staging. Enable DB for the tf32 path only when
    # the DOUBLED tile fits the device cap (else the resolver assert would fire);
    # every other path keeps the measured-wash OFF default.
    CT === Float32 || return false
    return _gemm_mma_shmem(CT, AccT, BM, BN, BK, DN, PC, PA, 2) <= cap
end

# ── cp.async global→shared staging (`ASYNC` knob) ───────────────────────────
# On sm_80+ NVIDIA, `KI.async_copy!` emits `cp.async` (LDGSTS): the tile moves
# global→shared WITHOUT a register hop, so a double-buffered K-loop can overlap the
# copy with the tensor-core math instead of stalling on a global→register→shared
# round trip. It is an OPT-IN KNOB, defaulted OFF, for the same reasons DB is: the
# win only materialises paired with DB at a WK=2 tile that fits the doubled As/Bs
# (the default WK=4 tiers already amortize staging AND overflow the cap when
# doubled), and the register path must stay reachable for an honest A/B. When OFF —
# or on any device without a hardware cp.async path, or on a misaligned/transposed
# operand the host gate rejects — staging is byte-identical to the register kernel.
# `default_gemm_mma_async` returns the arch default (ON), then `_gemm_launch_mma!`
# ANDs the shape gate (`_staging_worthwhile`) and computes the per-operand ENABLE
# (layout + alignment + `async_copy_supported`), never here.
function default_gemm_mma_async(::AbstractArch, M, N, K, ::Type{CT}, ::Type{AccT}) where {CT,AccT}
    # Arch-level default ON — but NOT unconditional: the launch site ANDs this with
    # the shape gate `_staging_worthwhile`, which turns staging OFF in the shallow-K +
    # saturated corner where it regressed (MI300A `tall`, buf/reg 1.22 F16 / 1.35 F64).
    # This method only sets the arch FLAG; the authoritative per-operand ENABLE is
    # `_gemm_async_flags` (requires `async_copy_supported(backend)` — NVIDIA sm_80+
    # cp.async OR AMD gfx942 buffer-DMA — AND per-operand alignment). On any device or
    # operand that can't do it, staging is byte-identical to the register path.
    # Measured wins: RTX1000 Ada cp.async (1024 62%→88%, 2048 43%→73% cuBLAS); MI300A
    # buffer DMA (square/deep-K F16 up to 2×). Gate details at `_staging_worthwhile`.
    return true
end

# ── L2-aware threadblock swizzle group width (GSW) ──────────────────────────
# `gemm_mma_kernel!` iterates GSW consecutive block-ROWS while sweeping ALL block-COLS,
# so a group's B column-tiles stay resident in L2 across those rows — cutting DRAM
# re-reads (the classic CUTLASS threadblock swizzle). But it is NOT a free win: it helps
# ONLY when the A/B working set exceeds L2 AND the grid is 2-D balanced (both dims large).
# MEASURED (RTX1000 Ada, F16, GSW=8 vs 1): square 4096 (A/B 34 MB > 25 MB L2, 32×32 grid)
# 0.49× — 2× FASTER, beats cuBLAS; but it HURTS the other regimes — inner (512², tiny 4×4
# grid) 1.54×, tall (nby=2 skewed) 1.97×, outer (A/B 4 MB < L2) 1.15×. So gate on BOTH the
# footprint and grid balance; GSW=1 (no grouping) everywhere else. `l2_cache_bytes` is the
# cross-arch device L2 query (unknown backend → typemax ⇒ swizzle off, safe). The `16`
# balance bound = 2·GSW (need several full groups in each dim). A future autotune can tune
# GSW/the thresholds per arch.
function default_gemm_swizzle(::AbstractArch, nbx::Int, nby::Int, ab_bytes::Int, l2::Int)
    (ab_bytes > l2 && nbx >= 16 && nby >= 16) ? 8 : 1
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
- `outT=nothing`: OUTPUT (storage) eltype override, independent of the accumulator.
  Defaults to `Base.promote_op(g, accT)` (the precision-first default: `Float16` ×
  `Float16` returns a `Float32` matrix). Pass `outT=Float16` for the standard
  mixed-precision GEMM — F16/BF16 in, `Float32` accumulate, `Float16` store — which
  keeps the F32-accumulate quality but HALVES the C-write traffic; on memory-bound
  shapes (short-K `outer`, tall-skinny `tall`) that is a large end-to-end win (RTX1000
  Ada F16 `outer` 16384²×64: F32-out 2.5× vs cuBLAS → F16-out 1.3×, cuBLAS itself
  storing F16). Allocating entry points only; `gemm!` takes the eltype from `C`.
- `family=nothing`: `:generic` (K1, the universal path) | `:mma` (K2 tensor-core) |
  `nothing` (auto). `:mma` is **opt-in** — auto always picks `:generic` — and a
  forced `:mma` on an unsupported config is a loud error, never a silent fallback.
  K2 needs plain (`*`,`+`), `eltype(A) === eltype(B)`, and a (compute, accumulate)
  pair the device announces via `KI.MMA.mma_shapes`: `Float16`/`BFloat16` → `Float32`,
  plus `Float64` (CDNA3 MFMA 16×16×4) and `Int8` → `Int32` where the hardware has
  them, and `Float32` → **TF32** (NVIDIA sm_80+, m16n8k8, needs an LLVM≥9.8 toolchain).
  ⚠️ The F32 tensor path is TF32 — a 10-bit mantissa (~1e-3 rel err), LOWER precision
  than the generic F32 kernel — so it is **opt-in ONLY**: `family=nothing` (auto) never
  selects it (it stays full-precision K1), and you get tf32 only by asking for
  `family=:mma` explicitly. On RTX1000 Ada it is ~1.9× the generic F32 kernel (2.3→4.3
  TFLOP/s at 2048³, closing KF-vs-cuBLAS-SGEMM from 3.0× to 1.4×; still behind cuBLAS's
  own TF32 — the residual is the scalar tf32 fragment load, tf32 cannot use `ldmatrix`).
  The silicon implements one fixed multiply-add, so a custom `(f, op)` can never
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

# Mixed-precision store: F32 accumulate, F16 output (matches cuBLAS `A*B`; half the
# C-write traffic — the win on memory-bound shapes). `C16b` is a Float16 matrix.
C16b = gemm(A16, B16; family=:mma, outT=Float16)
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
# (natural map type, accumulation type, output type) for the given f/g/accT/outT.
# `outT` forces the OUTPUT (storage) eltype independently of the accumulator: the
# standard mixed-precision GEMM stages — F16/BF16 in, F32 accumulate (`accT`), F16
# store (`outT`) — halves the C-write traffic on memory-bound shapes (outer/tall)
# while keeping the F32-accumulate quality. `nothing` ⇒ derive from `g(AccT)` (the
# precision-first default; e.g. F16×F16 ⇒ F32 out). The epilogue already casts the
# AccT accumulator to whatever eltype C carries (`S(g(...))`), so no kernel change.
@inline function _gemm_types(f::F, g::G, ::Type{TA}, ::Type{TB}, accT, outT=nothing) where {F,G,TA,TB}
    Hnat = Base.promote_op(f, TA, TB)
    AccT = accT === nothing ? _default_acc_type(Hnat) : accT
    S = outT === nothing ? Base.promote_op(g, AccT) : outT
    return (Hnat, AccT, S)
end

function resolve_gemm_parameters(
    arch::AbstractArch, M, N, K, ::Type{T},
    BM, BN, BK, TM, TN
) where {T}
    dBM, dBN, dBK, dTM, dTN = default_gemm_tile(arch, M, N, K, T)
    # Autotuned K1 tile (F32 & any non-tensor dtype) overrides the tiny default; a knob
    # the caller passed explicitly still wins. The 5 tuned knobs form a coherent tuple
    # validated against the same asserts below, so a full miss (all nothing) or a full
    # hit is always consistent — mirrors `resolve_gemm_mma_parameters`/`lookup_gemm`.
    tuned = lookup_gemm_k1(arch, T, M, N, K)
    _t(f) = tuned === nothing ? nothing : getfield(tuned, f)
    BM = something(BM, _t(:BM), dBM); BN = something(BN, _t(:BN), dBN); BK = something(BK, _t(:BK), dBK)
    TM = something(TM, _t(:TM), dTM); TN = something(TN, _t(:TN), dTN)
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
    tA::Symbol=:N, tB::Symbol=:N, accT=nothing, outT=nothing, family=nothing,
    BM=nothing, BN=nothing, BK=nothing, TM=nothing, TN=nothing,
    WM=nothing, WN=nothing, NWM=nothing, NWN=nothing, WK=nothing,
    PC=nothing, PA=nothing, DB=nothing, ASYNC=nothing, splitk=nothing,
    arch=nothing
) where {TA,TB,F<:Function,O<:Function,G<:Function}
    _, _, S = _gemm_types(f, g, TA, TB, accT, outT)
    backend = get_backend(A)
    M, N = _gemm_out_dims(A, B, tA, tB)
    C = KernelAbstractions.allocate(backend, S, M, N)
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family,
                 BM, BN, BK, TM, TN, WM, WN, NWM, NWN, WK, PC, PA, DB, ASYNC, splitk, arch)
    return C
end

# Full generalized-op, allocating.
function gemm(
    f::F, op::O,
    A::AbstractMatrix{TA}, B::AbstractMatrix{TB};
    g::G=identity, tA::Symbol=:N, tB::Symbol=:N, accT=nothing, outT=nothing, family=nothing,
    BM=nothing, BN=nothing, BK=nothing, TM=nothing, TN=nothing,
    WM=nothing, WN=nothing, NWM=nothing, NWN=nothing, WK=nothing,
    PC=nothing, PA=nothing, DB=nothing, ASYNC=nothing, splitk=nothing,
    arch=nothing
) where {TA,TB,F<:Function,O<:Function,G<:Function}
    _, _, S = _gemm_types(f, g, TA, TB, accT, outT)
    backend = get_backend(A)
    M, N = _gemm_out_dims(A, B, tA, tB)
    C = KernelAbstractions.allocate(backend, S, M, N)
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family,
                 BM, BN, BK, TM, TN, WM, WN, NWM, NWN, WK, PC, PA, DB, ASYNC, splitk, arch)
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
    PC=nothing, PA=nothing, DB=nothing, ASYNC=nothing, splitk=nothing,
    arch=nothing
) where {S,TA,TB,F<:Function,O<:Function,G<:Function}
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family,
                 BM, BN, BK, TM, TN, WM, WN, NWM, NWN, WK, PC, PA, DB, ASYNC, splitk, arch)
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
    PC=nothing, PA=nothing, DB=nothing, ASYNC=nothing, splitk=nothing,
    arch=nothing
) where {S,TA,TB,F<:Function,O<:Function,G<:Function}
    _gemm_entry!(f, op, g, C, A, B, tA, tB, accT, family,
                 BM, BN, BK, TM, TN, WM, WN, NWM, NWN, WK, PC, PA, DB, ASYNC, splitk, arch)
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
# NOTE: a lazy `Adjoint`/`Transpose` wrapper reports its LOGICAL dims (`size(A',1)`
# already accounts for the flip), so this is correct on wrappers too — the C
# allocation in the `gemm` methods stays right even before `_norm_operand` unwraps.
@inline function _gemm_out_dims(A, B, tA::Symbol, tB::Symbol)
    M = tA === :N ? size(A, 1) : size(A, 2)
    N = tB === :N ? size(B, 2) : size(B, 1)
    return (M, N)
end

# ── Adjoint/Transpose wrapper support ───────────────────────────────────────
# `gemm(A', B)` / `gemm(transpose(A), B)` must be correct: a lazy wrapper cannot be
# fed to the kernel as-is (it would be read as its col-major PARENT, i.e. wrong).
# `_norm_operand` unwraps the wrapper to its bare parent and COMPOSES the implied
# transpose with the caller's explicit `tA`/`tB` flag (so a wrapper AND a flag stack
# correctly). Reals: adjoint ≡ transpose. Complex adjoint would need conjugation
# (`:C`), which `_gemm_layout` rejects until complex is supported — a clean error.
@inline _flip_flag(s::Symbol) = s === :N ? :T : (s === :T ? :N : s)   # :C passes through
@inline _norm_operand(A::AbstractMatrix, flag::Symbol) = (A, flag)
@inline _norm_operand(A::Transpose, flag::Symbol) = (parent(A), _flip_flag(flag))
@inline _norm_operand(A::Adjoint, flag::Symbol) =
    eltype(A) <: Real ? (parent(A), _flip_flag(flag)) :
                        (parent(A), flag === :N ? :C : :C)   # conj-transpose → :C (errors)

# Pick the compute family. `:mma` iff plain (*,+), matching CT=TA=TB with a HW MMA
# path for (CT, AccT), and the output is at least one MMA tile. Explicit generic
# tile knobs force `:generic`. A forced `family=:mma` that fails the gate errors.
# Total-work threshold (M·N·K, in madds) at/above which AUTO picks the tensor-core
# kernel over K1. Coarse cross-arch heuristic: the measured RTX1000 F16 square
# crossover is between 256³ (≈1.7e7, K1 0.90× faster) and 512³ (≈1.3e8, K2 5.8×
# faster); 2^26 sits in that band. A per-arch autotune gate can refine it later.
const MMA_AUTO_FLOP_GATE = 1 << 26      # ≈ 6.7e7

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
    plain = (f === Base.:*) && (op === Base.:+)
    # All FOUR transpose states are HW MMA. TT used to be excluded here because it
    # "miscompiled" — that was the KI `undef`-accumulator bug wearing another mask,
    # and TT is exact (rel err ~1e-6) since KI 0.1.14. The Phase-2 staging already
    # normalises every layout into col-major-logical shared tiles, so there is no
    # per-layout MMA variant to get wrong.
    shape = (plain && TA === TB) ? _mma_pick_shape(backend, TA, AccT) : nothing
    ok = shape !== nothing && M >= shape.Mt && N >= shape.Nt && K >= 1
    # AUTO (`family === nothing`): pick the tensor-core kernel when a HW path exists
    # and the product is big enough to amortize its setup — else K1. `:mma` used to be
    # opt-in only; the block correctness ban (KI `undef`-accumulator phi) was FIXED in
    # KI 0.1.14, and warp/register tiling made K2 3–7× K1 on RTX1000 F16 (n≥512),
    # while split-K (see `_default_splitk`) closed the deep-K occupancy hole. The one
    # regime K1 still wins is TINY launch-overhead-bound products (RTX1000 F16 square
    # n=256 → K2 0.90× K1), so gate on total work `M·N·K`. The threshold is a coarse
    # cross-arch heuristic (crossover sits between 256³ and 512³); a future per-arch
    # autotune gate can refine it. Generic tile knobs (BM/BN/…) still force generic.
    if family === nothing
        # NEVER auto-select a LOSSY tensor path. For Float32 the only NVIDIA tensor
        # path is TF32 (10-bit mantissa, ~1e-3 rel err) — genuinely lower precision
        # than the generic F32 kernel, so it must stay OPT-IN (`family=:mma`), matching
        # the precision-first default the rest of the API follows. F16/BF16 tensor is
        # NOT lossy (F32 accumulate = the natural full path) so auto stays on for them.
        use_mma = ok && !user_generic_knobs && !_mma_lossy(TA) &&
                  (Float64(M) * N * K >= MMA_AUTO_FLOP_GATE)
        return use_mma ? (:mma, shape) : (:generic, nothing)
    end
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

# Normalise lazy Adjoint/Transpose operands into bare parents + composed flags, THEN
# dispatch to the impl on the bare array types (so the impl stays type-specialised).
function _gemm_entry!(
    f::F, op::O, g::G, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix,
    tA::Symbol, tB::Symbol, accT, family,
    BM, BN, BK, TM, TN, WM, WN, NWM, NWN, WK, PC, PA, DB, ASYNC, splitk, arch
) where {F,O,G}
    A2, tA2 = _norm_operand(A, tA)
    B2, tB2 = _norm_operand(B, tB)
    _gemm_entry_impl!(f, op, g, C, A2, B2, tA2, tB2, accT, family,
                      BM, BN, BK, TM, TN, WM, WN, NWM, NWN, WK, PC, PA, DB, ASYNC, splitk, arch)
end

function _gemm_entry_impl!(
    f::F, op::O, g::G,
    C::AbstractMatrix{S},
    A::AbstractMatrix{TA},
    B::AbstractMatrix{TB},
    tA::Symbol, tB::Symbol,
    accT, family,
    BM, BN, BK, TM, TN,
    WM, WN, NWM, NWN, WK, PC, PA, DB, ASYNC, splitk,
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
        # TT NN-DUALITY (L1 FUSED, opt-in `KF_GEMM_DUAL=1`). Both operands RowMajor ⇒ the
        # passed arrays are plain col-major physicals (A is K×M, B is N×K) and Cᵀ = B·A is
        # an NN product of two COLMAJOR operands — the fast path (free fragment loads +
        # cp.async on both), no RowMajor staging at all. Launch NN(B,A) on the SWAPPED shape
        # (M,N)→(N,M) and write the result TRANSPOSED into C via the fused RowMajor-store_d!
        # epilogue (`TC=true`) — no scratch, no copy, parity at ALL shapes. Split-K composes:
        # the transposed atomic index `(m-1)·N+n` addresses valid linear positions in `Cacc`
        # and the linear `Cacc→C` copy matches (needed for deep-K/small-MN — `inner`). Duality
        # proven bit-identical: xp/gemm/rewrite/tt_duality_check.jl. NN/NT/TN untouched
        # (TC compile-time-dead there). DEFAULT-ON: it is a strict win over the RowMajor-
        # staged TT path on every measured shape (F16 L2048 du/cB 0.87-1.72 vs 2.14-3.46),
        # allocates nothing beyond what split-K already does, and TT is the COMMON case for
        # row-major (Python/NumPy) callers. `KF_GEMM_DUAL=0` forces the old staged path (A/B).
        if (layA isa RowMajor) && (layB isa RowMajor) && get(ENV, "KF_GEMM_DUAL", "") != "0"
            return _gemm_launch_mma!(g, C, B, A, ColMajor(), ColMajor(), shape, TA, AccT,
                                     WM, WN, NWM, NWN, WK, PC, PA, DB, ASYNC, splitk, N, M, K, arch;
                                     TC = true)
        end
        return _gemm_launch_mma!(g, C, A, B, layA, layB, shape, TA, AccT,
                                 WM, WN, NWM, NWN, WK, PC, PA, DB, ASYNC, splitk, M, N, K, arch)
    end
    return _gemm_launch_generic!(f, op, g, C, A, B, layA, layB, AccT,
                                 BM, BN, BK, TM, TN, splitk, M, N, K, arch)
end

# Split-K is valid for the generic kernel ONLY when the k-reduction is a plain SUM
# (`op === +`) over an atomic-add-able accumulator — then each slice's partial is a
# sub-sum and the cross-slice combine is `+`. Any other `op` (tropical max, boolean,
# a custom monoid) or a non-primitive AccT (struct eltype) has no global atomic, so
# it keeps the single-workgroup-per-tile path. `f`/`g` are unrestricted: `g` applies
# once in the epilogue, and `+`-splitting holds for any map `f`.
@inline _generic_splitk_ok(op, ::Type{AccT}) where {AccT} =
    op === Base.:+ && (AccT <: AbstractFloat || AccT <: Integer)

function _gemm_launch_generic!(
    f::F, op::O, g::G, C, A, B, layA, layB, ::Type{AccT},
    BM, BN, BK, TM, TN, splitk, M, N, K, arch
) where {F,O,G,AccT}
    backend = get_backend(A)
    p = resolve_gemm_parameters(arch, M, N, K, eltype(A), BM, BN, BK, TM, TN)
    nbx = cld(M, p.BM); nby = cld(N, p.BN)
    NK = _generic_splitk_ok(op, AccT) ?
         something(splitk, _default_splitk(M, N, K, p.BM, p.BN, p.BK, num_sms(backend)))::Int : 1
    @assert NK >= 1 "splitk must be ≥ 1"

    if NK == 1
        ndrange = nbx * nby * p.wg      # exact workgroup multiple → no bare-@index OOB
        gemm_kernel!(backend, p.wg)(
            f, op, g, C, A, B, layA, layB,
            Val(p.BM), Val(p.BN), Val(p.BK), Val(p.TM), Val(p.TN), Val(1),
            AccT, M, N, K;                # AccT is the accumulation type H for K1
            ndrange = ndrange,
        )
        return C
    end

    # Split-K: each (tile, slice) sums its K-slice and atomic-adds the raw AccT
    # partial into a zeroed scratch; the epilogue then applies `g` + casts into C.
    Cacc = KernelAbstractions.allocate(backend, AccT, M, N)
    fill!(Cacc, zero(AccT))
    ndrange = nbx * nby * NK * p.wg
    gemm_kernel!(backend, p.wg)(
        f, op, g, Cacc, A, B, layA, layB,
        Val(p.BM), Val(p.BN), Val(p.BK), Val(p.TM), Val(p.TN), Val(NK),
        AccT, M, N, K;
        ndrange = ndrange,
    )
    nC = M * N
    epi_wg = 256
    gemm_splitk_epilogue!(backend, epi_wg)(g, C, Cacc, nC; ndrange = cld(nC, epi_wg) * epi_wg)
    return C
end

# Resolve the MMA warp/register tiling. Returns everything the launcher needs;
# `shmem` comes from the ONE pure sizing fn the kernel also uses.
function resolve_gemm_mma_parameters(
    arch::AbstractArch, backend, M, N, K, ::Type{CT}, ::Type{AccT},
    Mt, Nt, Kt, WM, WN, NWM, NWN, WK, PC, PA, DB, ldm::Bool=false, dualDs::Bool=false
) where {CT,AccT}
    dWM, dWN, dNWM, dNWN, dWK = default_gemm_mma_tile(arch, M, N, K, CT, AccT, Mt, Nt, Kt)
    # Autotuned knobs (per shape-regime × dtype from `<arch>_gemm.jl`) sit BETWEEN the
    # caller and the heuristic default: caller-passed > tuned > `default_*`. A tuned
    # config was found feasible on THIS arch and its shmem is a pure function of the
    # knobs (independent of M,N,K), so it always fits at runtime — trust it fully.
    tuned = lookup_gemm(arch, CT, M, N, K)
    _t(f) = tuned === nothing ? nothing : getfield(tuned, f)
    WM = something(WM, _t(:WM), dWM); WN = something(WN, _t(:WN), dWN)
    NWM = something(NWM, _t(:NWM), dNWM); NWN = something(NWN, _t(:NWN), dNWN)
    WK = something(WK, _t(:WK), dWK)
    PC = something(PC, _t(:PC), default_mma_pad(CT))
    PA = something(PA, _t(:PA), default_mma_pad(AccT))
    @assert WM >= 1 && WN >= 1 && NWM >= 1 && NWN >= 1 && WK >= 1 "MMA tile knobs must be ≥ 1"
    ws = get_warpsize(arch)
    wg = NWM * NWN * ws
    @assert wg <= 1024 "workgroup ($wg) exceeds 1024; shrink NWM/NWN"
    BM = NWM * WM * Mt; BN = NWN * WN * Nt; BK = WK * Kt
    DN = NWN * Nt
    # Double buffering: the default is per-tier, but it must also FIT. A tile whose
    # doubled As/Bs blows the device cap silently falls back to single-buffered —
    # the same tile is then still legal, just without the overlap. An EXPLICIT
    # `DB=true` that does not fit is a loud error instead of a silent downgrade.
    dDB = default_gemm_mma_db(arch, M, N, K, CT, AccT, BM, BN, BK, DN, PC, PA,
                              KI.max_dynamic_localmem(backend))
    DB = something(DB, _t(:DB), dDB)::Bool
    shmem = _gemm_mma_shmem(CT, AccT, BM, BN, BK, DN, PC, PA, DB ? 2 : 1, ldm;
                            dualDs = dualDs, DM = NWM * Mt)
    cap = KI.max_dynamic_localmem(backend)
    @assert shmem <= cap "gemm :mma tile needs $shmem B of workgroup memory (DB=$DB), device cap is $cap B; shrink WM/WN/NWM/NWN/WK or pass DB=false"
    return (; WM, WN, NWM, NWN, WK, PC, PA, DB, ws, wg, BM, BN, BK, shmem)
end

# Per-operand cp.async enable. A 16-byte chunk is `EPB = 16÷sizeof(CT)` contiguous
# elements, so cp.async serves an operand ONLY when it is ColMajor (staged-contiguous
# ⇒ global-contiguous), the leading global dimension is a multiple of EPB (source
# 16-byte alignment: A's is `M`, B's is `K`), and the padded shared ld is too (dest
# alignment). Everything here is host-known; the per-CHUNK edge in-bounds test is the
# kernel's. `false` whenever the device lacks a hardware cp.async path — the fallback
# would still be correct but pointless, so we keep the register kernel instead.
# Per-operand staging STYLE: `:cpasync` (NVIDIA, per-thread 16-byte cp.async),
# `:buffer` (AMD, wave-cooperative dword raw.buffer.load.lds), or `:none` (register).
# The backend picks the hardware style via `KI.async_copy_style`; the alignment
# preconditions differ by granularity — cp.async needs 16-byte-aligned ColMajor
# chunks, the buffer DMA needs 4-byte (dword) alignment (columns contiguous & the
# column start dword-aligned). RowMajor or any misaligned operand stays `:none` and
# takes the byte-identical register path.
@inline function _gemm_async_flags(backend, ::Type{CT}, layA, layB, M, N, K,
                                   BM, BK, PC, ASYNC::Bool) where {CT}
    ASYNC || return (:none, :none)
    style = KI.async_copy_style(backend)
    ldA = BM + PC; ldB = BK + PC
    if style === :cpasync
        EPB = 16 ÷ sizeof(CT)
        sA = (layA isa ColMajor) && (M % EPB == 0) && (ldA % EPB == 0) && (BM % EPB == 0)
        sB = (layB isa ColMajor) && (K % EPB == 0) && (ldB % EPB == 0) && (BK % EPB == 0)
        return (sA ? :cpasync : :none, sB ? :cpasync : :none)
    elseif style === :buffer
        al = x -> (x * sizeof(CT)) % 4 == 0        # 4-byte (dword) alignment
        sA = (layA isa ColMajor) && al(BM) && al(M) && al(ldA)
        sB = (layB isa ColMajor) && al(BK) && al(K) && al(ldB)
        return (sA ? :buffer : :none, sB ? :buffer : :none)
    else
        return (:none, :none)
    end
end

# SPLIT-K factor. `inner`/deep-K shapes launch only a handful of output tiles
# (cld(M,BM)·cld(N,BN)) so the device sits idle while each workgroup grinds the
# full K serially. Splitting K into NK slices multiplies the grid by NK (each
# slice reduces its partial atomically) — recovering occupancy without shrinking
# the (efficient) MMA tile. Engage ONLY when there are few tiles AND enough K to
# split; the common (well-filled) case returns 1 → byte-identical to no split.
@inline function _default_splitk(M, N, K, BM, BN, BK, ncu)
    tiles = cld(M, BM) * cld(N, BN)
    nkt   = cld(K, BK)
    # Engage only when the output-tile grid can't fill the device (few tiles vs the
    # CU/SM count) AND K is deep enough that a slice still does real work. The
    # threshold MUST scale with the device: a fixed cut regressed either square 1024
    # on a 24-CU RTX (64 tiles wrongly split) OR F64 inner-512 on a 228-CU MI300A
    # (64 tiles left unsplit though the device was starved). `2·ncu` ≈ light
    # oversubscription; target the same in workgroups. Cap NK at nkt÷4 (≥4 K-panels
    # per slice) and 64 (bound atomic contention + the AccT scratch).
    (tiles >= 2 * ncu || nkt < 8) && return 1
    want = cld(2 * ncu, tiles)
    return clamp(want, 1, min(nkt ÷ 4, 64))
end

# Shallow-K floor below which global→shared staging stops paying once the device is
# saturated. Calibrated on MI300A (xp/gemm/amd_buffer_ab.jl, 2026-07-27): the ONLY
# shape where buffer staging regressed was `tall` 65536×512×512 (K=512) — everything
# with K ≥ 1024 either won or washed. A conservative floor; refine with a K-sweep if
# the boundary ever matters. Backend-generic (also gates NVIDIA cp.async — validated
# non-regressing on RTX1000 Ada before shipping).
const GEMM_STAGING_KMIN = 1024

# Gate for the coalesced (`:coal`) staging of a RowMajor **B** operand. `:coal` trades
# B's scattered global read for a strided SHARED write (bank conflict); the conflict is
# paid over B's whole staging footprint ~ N·K (M-independent — verified: a tinyM..M=512
# sweep at N=K=8192 all regressed equally). So coal-B WINS unless N AND K are BOTH large
# and LOSES when they are. RTX1000 Ada (xp/gemm/rewrite/phase1_coal*, 2026-08-03):
#   WIN  square(N=K=2048), tall(128-256), band(N≤16384 @ K=2048), deepK(K=16384 @
#        N=2048), inner(N≥128, K≤131072), outer(N=8192, K=64)
#   LOSS N=K=8192 (1.65-1.99× over scatter, any M), N=K=16384, N=64 @ K≥8192 (A_thin
#        1.29-1.37×), N=32 (tiny-N masking).
# ⇒ ON iff N ≥ GEMM_COAL_NMIN AND min(N,K) ≤ GEMM_COAL_MINKN. The N floor (128) rejects
# the small-N × large-K masking corner (N=64 wins at K=2048 but loses at K≥8192, N=128
# wins even at K=65536); the min(N,K) cap rejects the both-large footprint corner. coal-A
# has no such corner (safe unconditionally) so only B is gated. Kin to `_default_splitk`.
const GEMM_COAL_NMIN  = 128
const GEMM_COAL_MINKN = 4096

# Is global→shared staging worth its fixed per-K-panel setup cost for this shape? The
# answer is STYLE-DEPENDENT — the two hardware primitives have very different cost
# curves (measured, not assumed):
#   • `:buffer` (AMD gfx942 raw.buffer.load.lds) is wave-cooperative with a HEAVY fixed
#     cost (hand SRD build + LDS bounce + s_waitcnt drain). It LOSES in the shallow-K +
#     saturated corner — the grid is already full (no idle latency to overlap) AND K is
#     too short to amortize the setup: MI300A `tall` 65536×512×512 regressed buf/reg
#     1.22 F16 / 1.35 F64. Square/deep-K stayed a win (F16 up to 2×) or wash (F64).
#   • `:cpasync` (NVIDIA sm_80+ LDGSTS) is per-thread and hardware-pipelined with a
#     LIGHT fixed cost. On RTX1000 Ada it still WON that same corner (tall async/reg
#     0.67–0.73) — it never reaches the loss regime at these sizes.
# So the gate fires for `:buffer` ONLY: disable staging when the grid saturates AND K
# is shallow; keep it ON for cp.async and (harmlessly) for `:none`. Mirror of
# `_default_splitk`'s tiles-vs-ncu regime test — a pure predicate, white-box testable
# without relying on numeric agreement between the staged and register paths.
@inline function _staging_worthwhile(style::Symbol, M, N, K, BM, BN, ncu)
    style === :buffer || return true          # cp.async's fixed cost is low → no shallow-K loss
    tiles = cld(M, BM) * cld(N, BN)
    return !(tiles >= ncu && K < GEMM_STAGING_KMIN)
end

function _gemm_launch_mma!(
    g::G, C, A, B, layA, layB, shape, ::Type{CT}, ::Type{AccT},
    WM, WN, NWM, NWN, WK, PC, PA, DB, ASYNC, splitk, M, N, K, arch; TC::Bool=false
) where {G,CT,AccT}
    backend = get_backend(A)
    Mt, Nt, Kt = shape.Mt, shape.Nt, shape.Kt
    cfg = KI.MMA.MMAConfig{Mt,Nt,Kt,CT,AccT}()
    # ldmatrix tf32 fragment-load path (m16n8k8) — OPT-IN, OFF by default. MEASURED a
    # net LOSS for NN (ColMajor A): ldmatrix needs a K-contiguous A tile, which forces a
    # TRANSPOSED stage (strided shared writes) whose cost exceeds the ldmatrix load
    # saving. The scalar path stages A M-contiguously (coalesced) and wins. The load
    # lever IS real in isolation (1.7×) and would help TN (RowMajor A, coalesced K-contig
    # stage — no transpose), so the validated path is kept behind `KF_LDM` for that
    # future case, not enabled as a default. See project_gemm_mma_sync_ldmatrix memory.
    ldm = (CT === Float32) && (Mt == 16 && Nt == 8 && Kt == 8) && KI.ldmatrix_supported(backend) &&
          haskey(ENV, "KF_LDM")
    p = resolve_gemm_mma_parameters(arch, backend, M, N, K, CT, AccT,
                                    Mt, Nt, Kt, WM, WN, NWM, NWN, WK, PC, PA, DB, ldm, TC)
    style = KI.async_copy_style(backend)
    dASYNC = default_gemm_mma_async(arch, M, N, K, CT, AccT) &&
             _staging_worthwhile(style, M, N, K, p.BM, p.BN, num_sms(backend))
    asyncflag = ldm ? false : something(ASYNC, dASYNC)::Bool
    styleA, styleB = _gemm_async_flags(backend, CT, layA, layB, M, N, K,
                                       p.BM, p.BK, p.PC, asyncflag)
    # Transpose staging gate (Phase 1, RTX1000-validated 2026-08-03). A RowMajor operand
    # can't be served by cp.async (it is transposed in global) so it falls to register
    # staging; the `:coal` variant coalesces its global read into a BIT-IDENTICAL shared
    # tile (k/n-inner loop) — a clean win except where measured to regress:
    #   • coal-A (RowMajor A): unconditionally safe (win/wash across M,N,K) → always on.
    #   • coal-B (RowMajor B): loses when N,K are BOTH large (footprint bound) or N tiny →
    #     on iff N ≥ GEMM_COAL_NMIN AND min(N,K) ≤ GEMM_COAL_MINKN.
    # ColMajor operands are untouched (keep their `_gemm_async_flags` style), so the
    # NN / ColMajor staging path stays byte-identical. `KF_GEMM_STAGE` overrides for
    # benchmarking: `coal` forces coal on both RowMajor operands, `scatter` forces the
    # old scatter path on both.
    coalmode = get(ENV, "KF_GEMM_STAGE", "")
    if coalmode != "scatter"
        forceB = coalmode == "coal"
        coalBok = N >= GEMM_COAL_NMIN && min(N, K) <= GEMM_COAL_MINKN
        styleA = (layA isa RowMajor) ? :coal : styleA
        styleB = (layB isa RowMajor && (forceB || coalBok)) ? :coal : styleB
    end
    NK = something(splitk, _default_splitk(M, N, K, p.BM, p.BN, p.BK, num_sms(backend)))::Int
    @assert NK >= 1 "splitk must be ≥ 1"
    nbx = cld(M, p.BM); nby = cld(N, p.BN)
    gsw = default_gemm_swizzle(arch, nbx, nby, (M * K + K * N) * sizeof(CT),
                               l2_cache_bytes(backend))::Int

    if NK == 1
        ndrange = nbx * nby * p.wg      # exact workgroup multiple → no bare-@index OOB
        KI.launch!(
            gemm_mma_kernel!(backend, p.wg),
            g, C, A, B, layA, layB, cfg,
            Val(Mt), Val(Nt), Val(Kt), Val(p.ws),
            Val(p.WM), Val(p.WN), Val(p.NWM), Val(p.NWN), Val(p.WK), Val(p.PC), Val(p.PA),
            Val(p.DB), Val(styleA), Val(styleB), Val(1), Val(ldm), Val(TC), AccT, M, N, K, gsw;
            ndrange = ndrange, shmem = p.shmem,
        )
        return C
    end

    # Split-K: each (tile, slice) workgroup atomic-adds its raw AccT partial into a
    # zeroed scratch accumulator; a second kernel then applies `g` + casts into C.
    Cacc = KernelAbstractions.allocate(backend, AccT, M, N)
    fill!(Cacc, zero(AccT))
    ndrange = nbx * nby * NK * p.wg
    KI.launch!(
        gemm_mma_kernel!(backend, p.wg),
        g, Cacc, A, B, layA, layB, cfg,
        Val(Mt), Val(Nt), Val(Kt), Val(p.ws),
        Val(p.WM), Val(p.WN), Val(p.NWM), Val(p.NWN), Val(p.WK), Val(p.PC), Val(p.PA),
        Val(p.DB), Val(styleA), Val(styleB), Val(NK), Val(ldm), Val(TC), AccT, M, N, K, gsw;
        ndrange = ndrange, shmem = p.shmem,
    )
    nC = M * N
    epi_wg = 256
    gemm_splitk_epilogue!(backend, epi_wg)(g, C, Cacc, nC; ndrange = cld(nC, epi_wg) * epi_wg)
    return C
end
