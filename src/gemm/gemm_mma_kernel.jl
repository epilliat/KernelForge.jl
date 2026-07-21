# ============================================================================
# Generalized GEMM — K2: tensor-core (WMMA/MFMA) kernel  [Phase 3]
# ============================================================================
#
# HW MMA engine for the PLAIN (*, +) path: C = A*B accumulated in AccT, epilogue
# `g`. Consumes KI's `MMA` layer verbatim (WMMA on NVIDIA / MFMA on AMD / portable
# fallback). `:mma` is MulAdd-only; the family gate in gemm.jl enforces `f===* &&
# op===+`. CT ∈ {Float16, BFloat16}, AccT = Float32 on NVIDIA.
#
# ── Warp / register tiling ──────────────────────────────────────────────────
# A block owns a BM×BN tile of C and streams K in BK-deep panels through shared
# memory. The block's warps form an NWM×NWN grid; warp (wm,wn) owns a
# `WM·Mt × WN·Nt` SUB-TILE held as a WM×WN grid of MMA accumulator fragments in
# REGISTERS, so
#     BM = NWM·WM·Mt      BN = NWN·WN·Nt      BK = WK·Kt
# holds by construction (no divisibility assertions needed). Per k-step a warp
# loads WM A-fragments and WN B-fragments and issues WM·WN `mma`s: every loaded
# fragment feeds a whole row/column of the accumulator grid, so staging traffic
# per FLOP falls as 1/WM + 1/WN. The v1 kernel was WM=WN=NWM=NWN=1 — one fragment
# load per mma, zero reuse, staging-bound. That is the entire perf story.
#
# ⚠ THE CLOSURE-BOXING TRAP (this cost the previous attempt the whole lever).
# The accumulator grid is an `NTuple` of fragments carried across the K-loop. If
# the tuple is built by a closure (`ntuple(...) do l ... end`) that CAPTURES a
# variable which is ALSO reassigned in an enclosing loop, Julia boxes that
# variable (`Core.Box`) → the fragment arrives as `Any` → `unsupported dynamic
# function invocation (call to getindex)` at codegen. It is a Julia
# closure-capture rule, NOT an MMA/codegen defect. The cure is structural and is
# why `_mma_tile_step` / `_mma_tile_store_col!` are TOP-LEVEL functions taking the
# tuples as ARGUMENTS: nothing they capture is ever reassigned. Verified clean
# (`xp/gemm/tile_probe.jl`): 32 accumulator phis all initialised to 0.0 (no
# `undef` — the KI 0.1.14 fix), zero dynamic dispatch, WM·WN·WK `wmma.mma` sites.
#
# Reuses the Phase-1/2 skeleton: block-grid decode + full-workgroup ndrange; the
# Phase-2 layout normalisation ⇒ shared tiles are col-major-logical and the
# fragment loads use the `MMA.ColMajor` type LITERAL ⇒ NN/NT/TN for free.
# Walls: W2-ld (2D shared tiles, ld=size(As,1)); W4b (zero(CT) edge pad, MMA
# consumes full slices); W4c (masked/`g` epilogue via a shared D-tile).
#
# ── Shared memory ───────────────────────────────────────────────────────────
# One dynamic blob (`KI.@dynlocalmem` + `KI.launch!(...; shmem)`, the sort-onesweep
# precedent) sized by the ONE pure function `_gemm_mma_shmem`, called host- AND
# device-side. `Ds` ALIASES `As`/`Bs` at offset 0: both are dead after the K-loop's
# trailing `@synchronize`, so the union costs max(...) not sum(...). Combined with
# the CHUNKED epilogue (Ds is BM×NWN·Nt, not BM×BN) this keeps the request at the
# K-loop's own footprint instead of the output tile's — the difference between a
# 256×128×64 tile fitting the device and not. The host checks the total against
# `KI.max_dynamic_localmem` at RUNTIME (never an arch table).

# ── Zero-padding shared staging (K2-only; 1-based loops) ────────────────────
@inline function _stage_a_mma!(As, A, ::ColMajor, brow, kt, M, K, lid, wg, ldA,
                               ::Val{BM}, ::Val{BK}) where {BM,BK}
    z = zero(eltype(As))
    p = lid
    while p <= BM * BK
        e = p - 1
        row = e % BM
        kk  = e ÷ BM
        m = brow * BM + row + 1
        k = kt * BK + kk + 1
        @inbounds As[kk * ldA + row + 1] = (m <= M && k <= K) ? @inbounds(A[(k - 1) * M + m]) : z
        p += wg
    end
end
@inline function _stage_a_mma!(As, A, ::RowMajor, brow, kt, M, K, lid, wg, ldA,
                               ::Val{BM}, ::Val{BK}) where {BM,BK}
    # Contiguous SHARED write `As[p]`; the transposed source makes the GLOBAL read
    # scattered (A stored K×M, Aop[m,k] = A[(m-1)*K+k]).
    z = zero(eltype(As))
    p = lid
    while p <= BM * BK
        e = p - 1
        row = e % BM
        kk  = e ÷ BM
        m = brow * BM + row + 1
        k = kt * BK + kk + 1
        @inbounds As[kk * ldA + row + 1] = (m <= M && k <= K) ? @inbounds(A[(m - 1) * K + k]) : z
        p += wg
    end
end
@inline function _stage_b_mma!(Bs, B, ::ColMajor, bcol, kt, K, N, lid, wg, ldB,
                               ::Val{BK}, ::Val{BN}) where {BK,BN}
    z = zero(eltype(Bs))
    p = lid
    while p <= BK * BN
        e = p - 1
        kk  = e % BK
        col = e ÷ BK
        k = kt * BK + kk + 1
        n = bcol * BN + col + 1
        @inbounds Bs[col * ldB + kk + 1] = (k <= K && n <= N) ? @inbounds(B[(n - 1) * K + k]) : z
        p += wg
    end
end
@inline function _stage_b_mma!(Bs, B, ::RowMajor, bcol, kt, K, N, lid, wg, ldB,
                               ::Val{BK}, ::Val{BN}) where {BK,BN}
    z = zero(eltype(Bs))
    p = lid
    while p <= BK * BN
        e = p - 1
        kk  = e % BK
        col = e ÷ BK
        k = kt * BK + kk + 1
        n = bcol * BN + col + 1
        @inbounds Bs[col * ldB + kk + 1] = (k <= K && n <= N) ? @inbounds(B[(k - 1) * N + n]) : z
        p += wg
    end
end

# ── Register-tile primitives (TOP-LEVEL — see the closure-boxing note) ──────
# One k-step for one warp: WM A-fragments × WN B-fragments → WM·WN `mma`s.
@inline function _mma_tile_step(cfg, As, Bs, wrow0, wcol0, koff, accs,
                                ::Val{WM}, ::Val{WN}, ::Val{Mt}, ::Val{Nt}) where {WM,WN,Mt,Nt}
    af = ntuple(i -> KI.MMA.load_a(cfg, As, wrow0 + (i - 1) * Mt, koff, KI.MMA.ColMajor), Val(WM))
    bf = ntuple(j -> KI.MMA.load_b(cfg, Bs, koff, wcol0 + (j - 1) * Nt, KI.MMA.ColMajor), Val(WN))
    return ntuple(Val(WM * WN)) do l
        i = (l - 1) % WM + 1
        j = (l - 1) ÷ WM + 1
        KI.MMA.mma(cfg, af[i], bf[j], accs[l])
    end
end

# Spill ONE accumulator COLUMN (`jsel`) of the warp's grid into the shared D-tile.
#
# The epilogue runs in WN column CHUNKS so the D-tile is `BM × NWN·Nt`, not
# `BM × BN`. That matters a lot: `Ds` sets the workgroup-memory request (it unions
# over the much smaller As/Bs), so a full-BN D-tile costs WN× the shared memory the
# K-loop actually needs and throttles blocks/SM. Measured on RTX1000 Ada: chunking
# is what lets a 128×128 tile run at all and lifts the best config's throughput.
#
# `jsel` is a RUNTIME chunk index, but it is never used to INDEX the tuple — the
# tuple stays fully unrolled and a warp-UNIFORM predicate picks the column. Indexing
# an NTuple of fragments with a runtime value would force an `alloca`, risking the
# whole accumulator grid landing in local memory across the K-loop.
@inline function _mma_tile_store_col!(cfg, Ds, wrow0, wcolD, accs, jsel,
                                      ::Val{WM}, ::Val{WN}, ::Val{Mt}) where {WM,WN,Mt}
    ntuple(Val(WM * WN)) do l
        i = (l - 1) % WM + 1
        j = (l - 1) ÷ WM + 1
        if j == jsel
            KI.MMA.store_d!(cfg, Ds, wrow0 + (i - 1) * Mt, wcolD, accs[l], KI.MMA.ColMajor)
        end
        0
    end
    return nothing
end

# ── Shared-tile leading dimensions: PADDING kills the bank conflicts ────────
# The fragment loads read a Mt×Kt (resp. Kt×Nt) sub-block COLUMN-STRIDED out of a
# col-major shared tile. With an unpadded ld the stride is a power of two, so the
# lanes of a warp land on a handful of the 32 banks and every fragment load
# serialises. Padding the leading dimension by `PAD` elements de-phases the rows.
# `PAD` is a knob (0 disables) whose default is one 16-byte segment worth of
# elements — the largest pad that keeps the ld a multiple of the 16-byte alignment
# WMMA requires of a fragment load, so it is legal on every announced shape:
#   F16  → 8 elements (ld stays ≡ 0 mod 8)
#   F32  → 4 elements
# Measured on RTX1000 Ada, F16 2048³ at the best tile: 6.9 → 9.1 TFLOP/s (~1.33×),
# essentially all of it from `PC` (As/Bs); padding the epilogue tile is noise.
@inline default_mma_pad(::Type{T}) where {T} = 16 ÷ sizeof(T)

# `Ds` (BM × NWN·Nt, one epilogue chunk) unions with `As`/`Bs` at offset 0 — both
# are dead after the K-loop's trailing `@synchronize` ⇒ max, not sum. `_align16`
# keeps the Bs sub-blob 16-byte aligned inside the one dynamic allocation.
@inline _align16(nbytes) = ((nbytes + 15) ÷ 16) * 16
@inline function _gemm_mma_shmem(::Type{CT}, ::Type{AccT}, BM, BN, BK, DN, PC, PA) where {CT,AccT}
    ab = _gemm_mma_off_b(CT, BM, BK, PC) + (BK + PC) * BN * sizeof(CT)
    d  = (BM + PA) * DN * sizeof(AccT)
    return max(ab, d)
end
@inline _gemm_mma_off_b(::Type{CT}, BM, BK, PC) where {CT} = _align16((BM + PC) * BK * sizeof(CT))

@kernel unsafe_indices = true function gemm_mma_kernel!(
    g::G,
    C::AbstractMatrix{S},
    A::AbstractMatrix{CT},
    B::AbstractMatrix{CT},
    layA::LA, layB::LB,
    cfg::CFG,
    ::Val{Mt}, ::Val{Nt}, ::Val{Kt}, ::Val{WS},
    ::Val{WM}, ::Val{WN}, ::Val{NWM}, ::Val{NWN}, ::Val{WK}, ::Val{PC}, ::Val{PA},
    ::Type{AccT},
    M::Int, N::Int, K::Int
) where {G<:Function,S,CT,LA<:FragLayout,LB<:FragLayout,CFG,
         Mt,Nt,Kt,WS,WM,WN,NWM,NWN,WK,PC,PA,AccT}
    @uniform begin
        BM  = NWM * WM * Mt
        BN  = NWN * WN * Nt
        BK  = WK * Kt
        DN  = NWN * Nt            # D-tile width = one epilogue column chunk
        wg  = NWM * NWN * WS
        nbx = cld(M, BM)          # block-rows across the C grid
        nkt = cld(K, BK)          # K panels
        ldA = BM + PC             # padded leading dims — see `default_mma_pad`
        ldB = BK + PC
        ldD = BM + PA
        offB = _gemm_mma_off_b(CT, BM, BK, PC)
    end

    # 2D tiles — ld = size(·,1) is what the fragment loads consume (W2-ld).
    # Ds aliases As/Bs at offset 0 (both dead after the K-loop).
    As = KI.@dynlocalmem CT (ldA, BK) 0
    Bs = KI.@dynlocalmem CT (ldB, BN) offB
    Ds = KI.@dynlocalmem AccT (ldD, DN) 0

    lid  = Int(@index(Local))
    gid  = Int(@index(Group))
    brow = (gid - 1) % nbx        # 0-based block-row
    bcol = (gid - 1) ÷ nbx        # 0-based block-col
    warp = (lid - 1) ÷ WS         # 0-based warp id within the block
    wm   = warp % NWM
    wn   = warp ÷ NWM
    wrow0 = wm * (WM * Mt) + 1    # 1-based sub-tile origin inside As / Ds
    wcol0 = wn * (WN * Nt) + 1

    accs = ntuple(_ -> KI.MMA.fill_c(cfg, KI.MMA.acc_identity(cfg)), Val(WM * WN))

    kt = 0
    while kt < nkt
        _stage_a_mma!(As, A, layA, brow, kt, M, K, lid, wg, ldA, Val(BM), Val(BK))
        _stage_b_mma!(Bs, B, layB, bcol, kt, K, N, lid, wg, ldB, Val(BK), Val(BN))
        @synchronize
        ks = 0
        while ks < WK
            accs = _mma_tile_step(cfg, As, Bs, wrow0, wcol0, ks * Kt + 1, accs,
                                  Val(WM), Val(WN), Val(Mt), Val(Nt))
            ks += 1
        end
        @synchronize                # also fences As/Bs before Ds reuses the blob
        kt += 1
    end

    # Epilogue, in WN column chunks: register grid → shared D-tile → masked
    # elementwise `g` copy to C. Chunk j holds, for every warp-column `wn`, that
    # warp's j-th accumulator column — so Ds is COMPACTED to NWN·Nt columns and the
    # global column is re-expanded on the copy-out. `j` and `wn` are warp-uniform,
    # so no MMA call ever sits under a divergent branch.
    wcolD = wn * Nt + 1
    j = 1
    while j <= WN
        _mma_tile_store_col!(cfg, Ds, wrow0, wcolD, accs, j, Val(WM), Val(WN), Val(Mt))
        @synchronize
        p = lid
        while p <= BM * DN
            e   = p - 1
            r   = e % BM
            cD  = e ÷ BM              # 0-based compacted D column
            wnc = cD ÷ Nt             # which warp-column slot it belongs to
            t   = cD % Nt             # offset inside that Nt-wide tile
            m = brow * BM + r + 1
            n = bcol * BN + wnc * (WN * Nt) + (j - 1) * Nt + t + 1
            if m <= M && n <= N
                @inbounds C[(n - 1) * M + m] = g(@inbounds Ds[cD * ldD + r + 1])
            end
            p += wg
        end
        @synchronize                  # before the next chunk overwrites Ds
        j += 1
    end
end
