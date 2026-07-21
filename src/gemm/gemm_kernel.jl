# ============================================================================
# Generalized GEMM — layout-parametric register-blocked, shared-mem-staged core
# ============================================================================
#
# Computes  C[m,n] = g( op_k f(Aop[m,k], Bop[k,n]) )   (reduction over k), where
# Aop / Bop are the LOGICAL operands (M×K and K×N) after applying the per-operand
# transpose flag. Arbitrary elementwise `f`, reduction `op`, epilogue `g`, and
# arbitrary isbits element types + operators (the reduction is SEEDED from a real
# loaded value — never `zero(H)` — so `op` needs no identity), mirroring matvec.
#
# ── Layout parametrisation (Phase 2: NN / NT / TN / TT) ─────────────────────
# ONE kernel covers all four transpose states. The per-operand layout tag is
# KI's `MMA.ColMajor` / `MMA.RowMajor` (the SAME singleton the Phase-3 tensor-core
# path passes to `MMA.load_a/load_b`), so this core is reused verbatim there.
# Mapping from the public `tA`/`tB` flags:
#   :N  →  ColMajor  (operand stored in its natural M×K / K×N col-major form)
#   :T  →  RowMajor  (operand stored transposed → load the parent as RowMajor;
#                     KI's identity load_a(A,col,row,RowMajor) ≡ load_a(A',row,col,ColMajor))
# Only the GLOBAL→SHARED staging load branches on the tag: it transposes the read
# into the shared tile so the shared tiles (`As` BM×BK, `Bs` BK×BN) are ALWAYS
# col-major logical operands and the register compute loop stays layout-agnostic
# and contiguous. The staging thread→element decode is chosen per layout to keep
# the global read COALESCED along whichever axis is unit-stride in the source.
#
# Invariants (unchanged from Phase 1): each block owns a full C tile (no split-K,
# no scratch, no allocation hazard); ndrange = nbx·nby full workgroups + every
# global load/store edge-masked ⇒ clean under `--check-bounds=yes`; shared tile
# (BM·BK + BK·BN) elements kept under the 48 KB static-@localmem cap by the
# conservative per-arch defaults in gemm.jl.

import KernelIntrinsics.MMA: RowMajor, ColMajor, FragLayout

# ── Shared-tile staging (dispatched on the KI layout tag) ────────────────────
# Both write the SAME col-major shared slot (`As[row + kk*BM + 1]`,
# `Bs[kk + col*BK + 1]`); only the source index formula and the thread→element
# decode order differ, so the compute loop below never sees the layout.

# A panel, ColMajor: A is M×K, unit-stride in m ⇒ decode m(row)-fastest (coalesced).
@inline function _stage_a!(As, A, ::ColMajor, brow, kt, M, K, lid, wg,
                           ::Val{BM}, ::Val{BK}) where {BM,BK}
    e = lid - 1
    while e < BM * BK
        row = e % BM
        kk  = e ÷ BM
        m = brow * BM + row + 1
        k = kt * BK + kk + 1
        As[e+1] = (m <= M && k <= K) ? @inbounds(A[(k - 1) * M + m]) : @inbounds(A[1])
        e += wg
    end
end
# A panel, RowMajor: A stored K×M (Aᵀ), unit-stride in k ⇒ decode k-fastest.
@inline function _stage_a!(As, A, ::RowMajor, brow, kt, M, K, lid, wg,
                           ::Val{BM}, ::Val{BK}) where {BM,BK}
    e = lid - 1
    while e < BM * BK
        kk  = e % BK
        row = e ÷ BK
        m = brow * BM + row + 1
        k = kt * BK + kk + 1
        As[row + kk * BM + 1] = (m <= M && k <= K) ? @inbounds(A[(m - 1) * K + k]) : @inbounds(A[1])
        e += wg
    end
end

# B panel, ColMajor: B is K×N, unit-stride in k ⇒ decode k-fastest (coalesced).
@inline function _stage_b!(Bs, B, ::ColMajor, bcol, kt, K, N, lid, wg,
                           ::Val{BK}, ::Val{BN}) where {BK,BN}
    e = lid - 1
    while e < BK * BN
        kk  = e % BK
        col = e ÷ BK
        k = kt * BK + kk + 1
        n = bcol * BN + col + 1
        Bs[e+1] = (k <= K && n <= N) ? @inbounds(B[(n - 1) * K + k]) : @inbounds(B[1])
        e += wg
    end
end
# B panel, RowMajor: B stored N×K (Bᵀ), unit-stride in n ⇒ decode n(col)-fastest.
@inline function _stage_b!(Bs, B, ::RowMajor, bcol, kt, K, N, lid, wg,
                           ::Val{BK}, ::Val{BN}) where {BK,BN}
    e = lid - 1
    while e < BK * BN
        col = e % BN
        kk  = e ÷ BN
        k = kt * BK + kk + 1
        n = bcol * BN + col + 1
        Bs[kk + col * BK + 1] = (k <= K && n <= N) ? @inbounds(B[(k - 1) * N + n]) : @inbounds(B[1])
        e += wg
    end
end

# One k-slice update of a thread's TM×TN register micro-tile. `@generated` so the
# TM/TN unroll and the seed-vs-fold branch resolve at compile time (like matvec's
# `_rowthread_chunk`). Returns a fresh NTuple{TM*TN,H}. Micro-tile is flattened
# col-major: element (ti,tj) lives at index ti + tj*TM + 1 (0-based ti,tj).
@inline @generated function _gemm_micro(
    f::F, op::O, acc::NTuple{L,H}, As, Bs,
    trow::Int, tcol::Int, kl::Int, seeded::Bool,
    ::Val{TM}, ::Val{TN}, ::Val{BM}, ::Val{BK}
) where {F,O,L,H,TM,TN,BM,BK}
    body = Expr(:block)
    for ti in 0:TM-1
        push!(body.args, :($(Symbol(:a_, ti)) =
            @inbounds As[(trow * TM + $ti) + kl * BM + 1]))
    end
    for tj in 0:TN-1
        push!(body.args, :($(Symbol(:b_, tj)) =
            @inbounds Bs[kl + (tcol * TN + $tj) * BK + 1]))
    end
    elems = Any[]
    for tj in 0:TN-1, ti in 0:TM-1
        a = ti + tj * TM + 1
        prod = :(f($(Symbol(:a_, ti)), $(Symbol(:b_, tj))))
        # Unseeded (first k): coerce the product to the accumulator element type
        # (`oftype`) so an `accT` override (e.g. F16 inputs → F32 acc) stays type
        # stable; identity when acc already matches (the default case).
        push!(elems, :(seeded ? op(@inbounds(acc[$a]), $prod) : oftype(@inbounds(acc[$a]), $prod)))
    end
    push!(body.args, Expr(:tuple, elems...))
    body
end

# Write a thread's micro-tile to C (col-major, linear index), edge-masked, with
# the epilogue `g` applied per element. `m0`/`n0` are 1-based global tile origins.
@inline @generated function _gemm_store!(
    g::G, C, acc::NTuple{L,H}, m0::Int, n0::Int, M::Int, N::Int,
    ::Val{TM}, ::Val{TN}
) where {G,L,H,TM,TN}
    body = Expr(:block)
    for tj in 0:TN-1, ti in 0:TM-1
        a = ti + tj * TM + 1
        push!(body.args, quote
            m = m0 + $ti
            n = n0 + $tj
            if m <= M && n <= N
                @inbounds C[(n - 1) * M + m] = g(acc[$a])
            end
        end)
    end
    push!(body.args, nothing)
    body
end

@kernel unsafe_indices = true inbounds = true function gemm_kernel!(
    f::F, op::O, g::G,
    C::AbstractMatrix{S},
    A::AbstractMatrix{TA},
    B::AbstractMatrix{TB},
    layA::LA, layB::LB,
    ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{TM}, ::Val{TN},
    ::Type{H},
    M::Int, N::Int, K::Int
) where {F<:Function,O<:Function,G<:Function,S,TA,TB,
         LA<:FragLayout,LB<:FragLayout,BM,BN,BK,TM,TN,H}
    @uniform begin
        RThreads = BM ÷ TM            # thread-rows per block
        wg       = (BM ÷ TM) * (BN ÷ TN)
        nbx      = cld(M, BM)         # block-rows across the C grid
        nkt      = cld(K, BK)         # number of K panels
    end

    As = @localmem TA (BM * BK)
    Bs = @localmem TB (BK * BN)

    lid  = Int(@index(Local))
    gid  = Int(@index(Group))
    brow = (gid - 1) % nbx            # 0-based block-row
    bcol = (gid - 1) ÷ nbx            # 0-based block-col
    trow = (lid - 1) % RThreads       # 0-based thread-row
    tcol = (lid - 1) ÷ RThreads       # 0-based thread-col

    m0 = brow * BM + trow * TM + 1
    n0 = bcol * BN + tcol * TN + 1

    # Typed accumulator seed WITHOUT zero(H): a real f-value coerced to the
    # accumulation type H (= AccT, possibly wider than the map type for an `accT`
    # override). Only the TYPE matters — the `seeded=false` branch overwrites each
    # element at the first k (via `oftype` in `_gemm_micro`).
    seed0  = convert(H, f(@inbounds(A[1]), @inbounds(B[1])))
    acc    = ntuple(_ -> seed0, Val(TM * TN))
    seeded = false

    kt = 0
    while kt < nkt
        _stage_a!(As, A, layA, brow, kt, M, K, lid, wg, Val(BM), Val(BK))
        _stage_b!(Bs, B, layB, bcol, kt, K, N, lid, wg, Val(BK), Val(BN))
        @synchronize

        # All threads share (kt, BK, K) → the `break` is uniform, so the trailing
        # @synchronize is reached by every thread.
        kl = 0
        while kl < BK
            k = kt * BK + kl + 1
            k > K && break
            acc = _gemm_micro(f, op, acc, As, Bs, trow, tcol, kl, seeded,
                              Val(TM), Val(TN), Val(BM), Val(BK))
            seeded = true
            kl += 1
        end
        @synchronize
        kt += 1
    end

    _gemm_store!(g, C, acc, m0, n0, M, N, Val(TM), Val(TN))
end
