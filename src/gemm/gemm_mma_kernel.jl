# ============================================================================
# Generalized GEMM — K2: tensor-core (WMMA/MFMA) kernel  [Phase 3, Steps 1-4]
# ============================================================================
#
# HW MMA engine for the PLAIN (*, +) path: C = A*B accumulated in AccT, epilogue
# `g`. Consumes KI's `MMA` layer verbatim (WMMA on NVIDIA / MFMA on AMD / portable
# fallback). `:mma` is MulAdd-only; the family gate in gemm.jl enforces `f===* &&
# op===+`. CT ∈ {Float16, BFloat16}, AccT = Float32 on this device.
#
# ── v1 tiling: ONE warp per block, ONE MMA tile per block ───────────────────
# Each block = one warp = one Mt×Nt output tile (BM=Mt, BN=Nt, BK=Kt), streaming K
# in Kt-wide panels staged in shared memory, a SINGLE `mma` per panel accumulating
# one fragment (the KI `mma_kloop` pattern). Correct and deterministic on RTX1000
# for all shapes/layouts/edges (verified xp/gemm/*).
#
# ⚠ MISCOMPILATION WALL (Steps 5+ blocker, fully diagnosed — see the Phase-3
# report): on THIS toolchain (Julia 1.12.6 + CUDA.jl jsgxH + KI 0.1.13 WMMA),
# combining a warp-resident WMMA fragment with EITHER (a) >1 warp per block, or
# (b) >1 `mma` per staged panel (NKS>1), miscompiles to NaN/garbage — regardless
# of indexing (2D vs linear) or loop count. A related trap: the `e=lid-1; while
# e<N; e+=wg` (0-based) staging-loop form miscompiles in the WMMA-fragment kernel
# where the equivalent `p=lid; while p<=N; p+=wg` (1-based) form is correct — so
# every loop here is the 1-based form. The single-warp / single-mma / 1-based
# configuration below is the proven-safe island. Recovering warp/register tiling
# (the real perf lever) needs a KI/CUDA.jl/KA codegen fix, NOT a KF workaround.
#
# Reuses the Phase-1/2 skeleton: block-grid decode + full-workgroup ndrange; the
# Phase-2 layout normalisation ⇒ shared tiles are col-major-logical and the
# fragment loads use the `MMA.ColMajor` type LITERAL ⇒ NN/NT/TN/TT for free.
# Walls: W2-ld (2D shared tiles, ld=size(As,1)); W4b (zero(CT) edge pad, MMA
# consumes full slices); W4c (masked/`g` epilogue via a shared D-tile).

# ── Zero-padding shared staging (K2-only; 1-based loops — see the wall note) ──
@inline function _stage_a_mma!(As, A, ::ColMajor, brow, kt, M, K, lid, wg,
                               ::Val{BM}, ::Val{BK}) where {BM,BK}
    z = zero(eltype(As))
    p = lid
    while p <= BM * BK
        e = p - 1
        row = e % BM
        kk  = e ÷ BM
        m = brow * BM + row + 1
        k = kt * BK + kk + 1
        @inbounds As[p] = (m <= M && k <= K) ? @inbounds(A[(k - 1) * M + m]) : z
        p += wg
    end
end
@inline function _stage_a_mma!(As, A, ::RowMajor, brow, kt, M, K, lid, wg,
                               ::Val{BM}, ::Val{BK}) where {BM,BK}
    # Contiguous SHARED write `As[p]` (the WMMA-safe form — a scattered shared
    # write miscompiles); the transposed source makes the GLOBAL read scattered
    # (A stored K×M, Aop[m,k] = A[(m-1)*K+k]).
    z = zero(eltype(As))
    p = lid
    while p <= BM * BK
        e = p - 1
        row = e % BM
        kk  = e ÷ BM
        m = brow * BM + row + 1
        k = kt * BK + kk + 1
        @inbounds As[p] = (m <= M && k <= K) ? @inbounds(A[(m - 1) * K + k]) : z
        p += wg
    end
end
@inline function _stage_b_mma!(Bs, B, ::ColMajor, bcol, kt, K, N, lid, wg,
                               ::Val{BK}, ::Val{BN}) where {BK,BN}
    z = zero(eltype(Bs))
    p = lid
    while p <= BK * BN
        e = p - 1
        kk  = e % BK
        col = e ÷ BK
        k = kt * BK + kk + 1
        n = bcol * BN + col + 1
        @inbounds Bs[p] = (k <= K && n <= N) ? @inbounds(B[(n - 1) * K + k]) : z
        p += wg
    end
end
@inline function _stage_b_mma!(Bs, B, ::RowMajor, bcol, kt, K, N, lid, wg,
                               ::Val{BK}, ::Val{BN}) where {BK,BN}
    # Contiguous SHARED write `Bs[p]` (WMMA-safe); scattered GLOBAL read (B stored
    # N×K, Bop[k,n] = B[(k-1)*N+n]).
    z = zero(eltype(Bs))
    p = lid
    while p <= BK * BN
        e = p - 1
        kk  = e % BK
        col = e ÷ BK
        k = kt * BK + kk + 1
        n = bcol * BN + col + 1
        @inbounds Bs[p] = (k <= K && n <= N) ? @inbounds(B[(k - 1) * N + n]) : z
        p += wg
    end
end

@kernel unsafe_indices = true function gemm_mma_kernel!(
    g::G,
    C::AbstractMatrix{S},
    A::AbstractMatrix{CT},
    B::AbstractMatrix{CT},
    layA::LA, layB::LB,
    cfg::CFG,
    ::Val{Mt}, ::Val{Nt}, ::Val{Kt}, ::Val{WS},
    ::Type{AccT},
    M::Int, N::Int, K::Int
) where {G<:Function,S,CT,LA<:FragLayout,LB<:FragLayout,CFG,Mt,Nt,Kt,WS,AccT}
    @uniform begin
        nbx = cld(M, Mt)          # block-rows across the C grid
        nkt = cld(K, Kt)          # K panels (one Mt×Nt block tile per block)
        wg  = WS                  # one warp per block
    end

    As = @localmem CT (Mt, Kt)    # 2D — ld = size(As,1) = Mt (W2-ld)
    Bs = @localmem CT (Kt, Nt)    # 2D — ld = Kt
    Ds = @localmem AccT (Mt, Nt)  # 2D D-tile for the masked/`g` epilogue

    lid  = Int(@index(Local))
    gid  = Int(@index(Group))
    brow = (gid - 1) % nbx        # 0-based block-row → owns rows brow*Mt .. +Mt
    bcol = (gid - 1) ÷ nbx        # 0-based block-col

    c = KI.MMA.fill_c(cfg, KI.MMA.acc_identity(cfg))
    kt = 0
    while kt < nkt
        _stage_a_mma!(As, A, layA, brow, kt, M, K, lid, wg, Val(Mt), Val(Kt))
        _stage_b_mma!(Bs, B, layB, bcol, kt, K, N, lid, wg, Val(Kt), Val(Nt))
        @synchronize
        a = KI.MMA.load_a(cfg, As, 1, 1, KI.MMA.ColMajor)
        b = KI.MMA.load_b(cfg, Bs, 1, 1, KI.MMA.ColMajor)
        c = KI.MMA.mma(cfg, a, b, c)
        @synchronize
        kt += 1
    end

    # Epilogue: fragment → shared D-tile, then masked elementwise `g` copy to C.
    KI.MMA.store_d!(cfg, Ds, 1, 1, c, KI.MMA.ColMajor)
    @synchronize
    p = lid
    while p <= Mt * Nt
        e   = p - 1
        r   = e % Mt
        col = e ÷ Mt
        m = brow * Mt + r + 1
        n = bcol * Nt + col + 1
        if m <= M && n <= N
            @inbounds C[(n - 1) * M + m] = g(@inbounds Ds[p])
        end
        p += wg
    end
end
