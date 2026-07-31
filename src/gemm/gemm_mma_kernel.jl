# ============================================================================
# Generalized GEMM — K2: tensor-core (WMMA/MFMA) kernel  [Phase 3]
# ============================================================================
#
# HW MMA engine for the PLAIN (*, +) path: C = A*B accumulated in AccT, epilogue
# `g`. Consumes KI's `MMA` layer verbatim (WMMA on NVIDIA / MFMA on AMD / portable
# fallback). `:mma` is MulAdd-only; the family gate in gemm.jl enforces `f===* &&
# op===+`. (CT, AccT) is any pair the device announces through `KI.MMA.mma_shapes`
# — F16/BF16→F32 everywhere, plus F64 (MFMA 16×16×4) and Int8→Int32 on CDNA3.
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
#
# `sh0` is the LINEAR element offset of the destination buffer inside the (possibly
# double-width) shared tile — 0 for the single-buffered path, `p*BK*ldA` (resp.
# `p*BN*ldB`) to address buffer `p` when `DB` is on. It shifts only the destination:
# the global read is identical, so the double-buffered staging costs nothing extra.
@inline function _stage_a_mma!(As, A, ::ColMajor, brow, kt, M, K, lid, wg, ldA, sh0,
                               ::Val{BM}, ::Val{BK}) where {BM,BK}
    z = zero(eltype(As))
    p = lid
    while p <= BM * BK
        e = p - 1
        row = e % BM
        kk  = e ÷ BM
        m = brow * BM + row + 1
        k = kt * BK + kk + 1
        @inbounds As[sh0 + kk * ldA + row + 1] = (m <= M && k <= K) ? @inbounds(A[(k - 1) * M + m]) : z
        p += wg
    end
end
@inline function _stage_a_mma!(As, A, ::RowMajor, brow, kt, M, K, lid, wg, ldA, sh0,
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
        @inbounds As[sh0 + kk * ldA + row + 1] = (m <= M && k <= K) ? @inbounds(A[(m - 1) * K + k]) : z
        p += wg
    end
end
@inline function _stage_b_mma!(Bs, B, ::ColMajor, bcol, kt, K, N, lid, wg, ldB, sh0,
                               ::Val{BK}, ::Val{BN}) where {BK,BN}
    z = zero(eltype(Bs))
    p = lid
    while p <= BK * BN
        e = p - 1
        kk  = e % BK
        col = e ÷ BK
        k = kt * BK + kk + 1
        n = bcol * BN + col + 1
        @inbounds Bs[sh0 + col * ldB + kk + 1] = (k <= K && n <= N) ? @inbounds(B[(n - 1) * K + k]) : z
        p += wg
    end
end
@inline function _stage_b_mma!(Bs, B, ::RowMajor, bcol, kt, K, N, lid, wg, ldB, sh0,
                               ::Val{BK}, ::Val{BN}) where {BK,BN}
    z = zero(eltype(Bs))
    p = lid
    while p <= BK * BN
        e = p - 1
        kk  = e % BK
        col = e ÷ BK
        k = kt * BK + kk + 1
        n = bcol * BN + col + 1
        @inbounds Bs[sh0 + col * ldB + kk + 1] = (k <= K && n <= N) ? @inbounds(B[(k - 1) * N + n]) : z
        p += wg
    end
end

# ── cp.async staging (CUDA sm_80+; `ASYNC` knob) ────────────────────────────
# Move the tile global→shared WITHOUT a register hop, via `KI.async_copy!`
# (`cp.async.ca.shared.global.16` on NVIDIA). A 16-byte chunk is `EPB = 16÷sizeof(CT)`
# contiguous elements (8 for F16/BF16), which cp.async copies with NO per-element
# masking and NO OOB zero-fill — so a chunk is async-copyable ONLY when all EPB
# elements are in-bounds AND both the global source and the shared dest are 16-byte
# aligned. The HOST gate (`_gemm_launch_mma!`) guarantees the *alignment*
# precondition (ColMajor operand, `M%EPB==0`/`K%EPB==0`, `ld%EPB==0`); what stays a
# per-chunk RUNTIME decision here is only the in-bounds test at the block/K edges.
# So this is HYBRID: fully-in-bounds chunks go async; edge chunks fall back to the
# byte-identical zero-filling register element path (`_stage_*_mma!`'s inner body).
#
# ColMajor ONLY. Async needs the staged-contiguous direction (rows of As, k-rows of
# Bs) to match the global-contiguous direction; a RowMajor operand is transposed in
# global, so its chunk is strided and cp.async cannot serve it — the host never sets
# the async flag for a RowMajor operand, and the RowMajor method here delegates to
# the register path as a defensive guarantee.
@inline function _stage_a_mma_async!(As, A, ::ColMajor, brow, kt, M, K, lid, wg, ldA, sh0,
                                     ::Val{BM}, ::Val{BK}) where {BM,BK}
    z = zero(eltype(As))
    EPB = 16 ÷ sizeof(eltype(As))       # elements per 16-byte cp.async chunk
    nr = BM ÷ EPB                        # EPB-row chunks down a column
    p = lid
    while p <= nr * BK
        e = p - 1
        rc  = e % nr
        kk  = e ÷ nr
        row0 = rc * EPB
        k  = kt * BK + kk + 1
        m0 = brow * BM + row0 + 1
        if k <= K && m0 + (EPB - 1) <= M
            KI.async_copy!(pointer(As, sh0 + kk * ldA + row0 + 1),
                           pointer(A, (k - 1) * M + m0), Val(16))
        else
            r = 0
            while r < EPB
                row = row0 + r
                m = brow * BM + row + 1
                @inbounds As[sh0 + kk * ldA + row + 1] = (m <= M && k <= K) ? @inbounds(A[(k - 1) * M + m]) : z
                r += 1
            end
        end
        p += wg
    end
end
@inline _stage_a_mma_async!(As, A, lay::RowMajor, brow, kt, M, K, lid, wg, ldA, sh0,
                            vBM::Val, vBK::Val) =
    _stage_a_mma!(As, A, lay, brow, kt, M, K, lid, wg, ldA, sh0, vBM, vBK)

@inline function _stage_b_mma_async!(Bs, B, ::ColMajor, bcol, kt, K, N, lid, wg, ldB, sh0,
                                     ::Val{BK}, ::Val{BN}) where {BK,BN}
    z = zero(eltype(Bs))
    EPB = 16 ÷ sizeof(eltype(Bs))
    nk = BK ÷ EPB                        # EPB-deep chunks down a Bs column
    p = lid
    while p <= nk * BN
        e = p - 1
        kc  = e % nk
        col = e ÷ nk
        k0off = kc * EPB
        n  = bcol * BN + col + 1
        k0 = kt * BK + k0off + 1
        if n <= N && k0 + (EPB - 1) <= K
            KI.async_copy!(pointer(Bs, sh0 + col * ldB + k0off + 1),
                           pointer(B, (n - 1) * K + k0), Val(16))
        else
            r = 0
            while r < EPB
                kk = k0off + r
                k = kt * BK + kk + 1
                @inbounds Bs[sh0 + col * ldB + kk + 1] = (k <= K && n <= N) ? @inbounds(B[(n - 1) * K + k]) : z
                r += 1
            end
        end
        p += wg
    end
end
@inline _stage_b_mma_async!(Bs, B, lay::RowMajor, bcol, kt, K, N, lid, wg, ldB, sh0,
                            vBK::Val, vBN::Val) =
    _stage_b_mma!(Bs, B, lay, bcol, kt, K, N, lid, wg, ldB, sh0, vBK, vBN)

# Compile-time dispatch between register and async staging. `Val{true}` only ever
# reaches a ColMajor operand (host gate); with `Val{false}` the async code is not
# generated at all and staging is byte-identical to the pre-async kernel.
# ── raw.buffer.load.lds staging (AMD CDNA; `:buffer` style) ─────────────────
# Wave-cooperative global→LDS DMA — the AMD analogue of cp.async, dword-granular with
# a HARDWARE per-lane scatter (lane l → lds_base + l*4 bytes). A tile column (BM resp.
# BK CONTIGUOUS elements = `Dw = elems*sizeof÷4` dwords) is loaded by the whole
# workgroup: wave w covers dwords [w*WS, w*WS+WS), rounds stride by `wg`. Dword math
# verified on MI300A (xp/gemm/amd_stage_buf.jl). ColMajor ONLY (the host sets `:buffer`
# only for a dword-aligned ColMajor operand). The DMA runs only when the tile is FULLY
# in-bounds and `Dw` is a whole number of waves (else a partial wave would need a
# divergent scatter) — every edge/odd tile takes the byte-identical register path.
# NOTE: `Core.LLVMPtr + n` advances n BYTES, so dword `d` sits at `+ d*4`.
@inline function _stage_a_mma_buf!(As, A, ::ColMajor, brow, kt, M, K, lid, wg, ldA, sh0,
                                   ::Val{BM}, ::Val{BK}) where {BM,BK}
    T = eltype(As)
    WS = KI.@warpsize()
    Dw = (BM * sizeof(T)) ÷ 4
    if brow * BM + BM <= M && kt * BK + BK <= K && Dw % WS == 0
        wave = (lid - 1) ÷ WS
        lane = (lid - 1) % WS
        kk = 0
        while kk < BK
            k = kt * BK + kk + 1
            srccol = reinterpret(Core.LLVMPtr{UInt32,1}, pointer(A, (k - 1) * M + brow * BM + 1))
            dstcol = reinterpret(Core.LLVMPtr{UInt32,3}, pointer(As, sh0 + kk * ldA + 1))
            base = wave * WS
            while base < Dw
                jj = base + lane
                KI.async_copy!(dstcol + base * 4, srccol, Int32(jj * 4), Val(4))
                base += wg
            end
            kk += 1
        end
    else
        _stage_a_mma!(As, A, ColMajor(), brow, kt, M, K, lid, wg, ldA, sh0, Val(BM), Val(BK))
    end
end
@inline _stage_a_mma_buf!(As, A, lay::RowMajor, brow, kt, M, K, lid, wg, ldA, sh0,
                          vBM::Val, vBK::Val) =
    _stage_a_mma!(As, A, lay, brow, kt, M, K, lid, wg, ldA, sh0, vBM, vBK)

@inline function _stage_b_mma_buf!(Bs, B, ::ColMajor, bcol, kt, K, N, lid, wg, ldB, sh0,
                                   ::Val{BK}, ::Val{BN}) where {BK,BN}
    T = eltype(Bs)
    WS = KI.@warpsize()
    Dw = (BK * sizeof(T)) ÷ 4
    if bcol * BN + BN <= N && kt * BK + BK <= K && Dw % WS == 0
        wave = (lid - 1) ÷ WS
        lane = (lid - 1) % WS
        col = 0
        while col < BN
            n = bcol * BN + col + 1
            srccol = reinterpret(Core.LLVMPtr{UInt32,1}, pointer(B, (n - 1) * K + kt * BK + 1))
            dstcol = reinterpret(Core.LLVMPtr{UInt32,3}, pointer(Bs, sh0 + col * ldB + 1))
            base = wave * WS
            while base < Dw
                jj = base + lane
                KI.async_copy!(dstcol + base * 4, srccol, Int32(jj * 4), Val(4))
                base += wg
            end
            col += 1
        end
    else
        _stage_b_mma!(Bs, B, ColMajor(), bcol, kt, K, N, lid, wg, ldB, sh0, Val(BK), Val(BN))
    end
end
@inline _stage_b_mma_buf!(Bs, B, lay::RowMajor, bcol, kt, K, N, lid, wg, ldB, sh0,
                          vBK::Val, vBN::Val) =
    _stage_b_mma!(Bs, B, lay, bcol, kt, K, N, lid, wg, ldB, sh0, vBK, vBN)

# Compile-time dispatch between register (`:none`), cp.async (`:cpasync`) and
# buffer_load_lds (`:buffer`) staging; the host picks the per-operand style.
@inline _stage_a(::Val{:none},    args...) = _stage_a_mma!(args...)
@inline _stage_a(::Val{:cpasync}, args...) = _stage_a_mma_async!(args...)
@inline _stage_a(::Val{:buffer},  args...) = _stage_a_mma_buf!(args...)
@inline _stage_b(::Val{:none},    args...) = _stage_b_mma!(args...)
@inline _stage_b(::Val{:cpasync}, args...) = _stage_b_mma_async!(args...)
@inline _stage_b(::Val{:buffer},  args...) = _stage_b_mma_buf!(args...)

# ── Register-tile primitives (TOP-LEVEL — see the closure-boxing note) ──────
# One k-step for one warp: WM A-fragments × WN B-fragments → WM·WN `mma`s.
#
# `koffA` and `koffB` are the SAME k-step in the two tiles but are NOT the same
# number under double buffering: buffer `p` is a COLUMN offset in both tiles, and k
# is the column of As but the ROW of Bs. So As shifts by `p·BK` in `koffA` while
# Bs shifts by `p·BN` in `wcol0` and its row index `koffB` stays put. They coincide
# (`ks·Kt+1`) on the single-buffered path.
@inline function _mma_tile_step(cfg, As, Bs, wrow0, wcol0, koffA, koffB, accs,
                                ::Val{WM}, ::Val{WN}, ::Val{Mt}, ::Val{Nt}) where {WM,WN,Mt,Nt}
    af = ntuple(i -> KI.MMA.load_a(cfg, As, wrow0 + (i - 1) * Mt, koffA, KI.MMA.ColMajor), Val(WM))
    bf = ntuple(j -> KI.MMA.load_b(cfg, Bs, koffB, wcol0 + (j - 1) * Nt, KI.MMA.ColMajor), Val(WN))
    return ntuple(Val(WM * WN)) do l
        i = (l - 1) % WM + 1
        j = (l - 1) ÷ WM + 1
        KI.MMA.mma(cfg, af[i], bf[j], accs[l])
    end
end

# ── ldmatrix fragment-load path (tf32 m16n8k8 only; `LDM` compile-time branch) ──
# Replaces the scalar per-lane fragment packing with a warp-collective `ldmatrix.x4`
# (KI.MMA.load_a_ldm/load_b_ldm), the 1.7× load lever. It needs a K-CONTIGUOUS A tile
# — `_stage_a_mma_ldm!` stages A that way (ld = BK, no pad) INTO the same `As` blob
# (BM·BK ≤ (BM+PC)·BK, so it fits with no shmem/decl change). B's existing ColMajor
# `Bs` is already K-contiguous ⇒ reused as-is (ld = ldB). Both fragment lanes are
# bit-exact vs the scalar oracle (KI test/tests/mma_sync_ldm.jl).
#
# `Aptr`/`Bptr` are the buffer-base shared pointers; (m0w,n0w) the warp's 0-based
# sub-tile origin, k0 the 0-based k-step column. ldkA = BK (A, no pad), ldkB = ldB (B).
@inline function _mma_tile_step_ldm(cfg, Aptr, Bptr, m0w, n0w, k0, ldkA, ldkB, accs,
                                    ::Val{WM}, ::Val{WN}, ::Val{Mt}, ::Val{Nt}) where {WM,WN,Mt,Nt}
    af = ntuple(i -> KI.MMA.load_a_ldm(cfg, Aptr, m0w + (i - 1) * Mt, k0, ldkA), Val(WM))
    bf = ntuple(j -> KI.MMA.load_b_ldm(cfg, Bptr, k0, n0w + (j - 1) * Nt, ldkB), Val(WN))
    return ntuple(Val(WM * WN)) do l
        i = (l - 1) % WM + 1
        j = (l - 1) ÷ WM + 1
        KI.MMA.mma(cfg, af[i], bf[j], accs[l])
    end
end

# K-contiguous A staging for the ldmatrix path: destination element (m,k) at
# `sh0 + m*BK + k` (ld = BK). ColMajor A (NN): iterate m-inner ⇒ COALESCED global
# read `A[(k-1)M+m]` (strided shared write). RowMajor A (TN): iterate k-inner ⇒
# coalesced global read `A[(m-1)K+k]`. Edge tiles zero-fill (MMA consumes full slices).
# `ldk` = A row stride in the K-contiguous tile. Padding it to break the power-of-2
# (BK is a power of two for the tf32 tile ⇒ a BK stride puts consecutive M-rows in the
# SAME shared bank → conflicts on both this write and the ldmatrix read). ldk = BK+1
# de-phases the rows and still fits the As blob exactly (BM·(BK+1) = (BM+PC)·BK here).
@inline function _stage_a_mma_ldm!(As, A, ::ColMajor, brow, kt, M, K, lid, wg, sh0, ldk,
                                   ::Val{BM}, ::Val{BK}) where {BM,BK}
    z = zero(eltype(As))
    p = lid
    while p <= BM * BK
        e = p - 1
        row = e % BM           # m (inner) → coalesced global read
        kk  = e ÷ BM           # k
        m = brow * BM + row + 1
        k = kt * BK + kk + 1
        @inbounds As[sh0 + row * ldk + kk + 1] = (m <= M && k <= K) ? @inbounds(A[(k - 1) * M + m]) : z
        p += wg
    end
end
@inline function _stage_a_mma_ldm!(As, A, ::RowMajor, brow, kt, M, K, lid, wg, sh0, ldk,
                                   ::Val{BM}, ::Val{BK}) where {BM,BK}
    z = zero(eltype(As))
    p = lid
    while p <= BM * BK
        e = p - 1
        kk  = e % BK           # k (inner) → coalesced global read (RowMajor A is K-contig)
        row = e ÷ BK           # m
        m = brow * BM + row + 1
        k = kt * BK + kk + 1
        @inbounds As[sh0 + row * ldk + kk + 1] = (m <= M && k <= K) ? @inbounds(A[(m - 1) * K + k]) : z
        p += wg
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

# Epilogue write, split-K-aware. NK==1 (the default, non-split path): apply the
# output map `g` and store into C::S directly — byte-identical to the pre-split
# kernel. NK>1: this workgroup owns only a K-SLICE, so its accumulator is a
# PARTIAL; atomic-add the raw AccT value into the shared C accumulator (C::AccT,
# zero-initialised host-side) and let the separate epilogue kernel apply `g`
# after all slices have reduced. Dispatch on Val{NK} so the AccT-atomic branch is
# never type-checked against an S-typed C (F16 out / F32 acc would not compile).
@inline _mma_epi_store!(::Val{1}, g::G, C, idx, v) where {G} =
    (@inbounds C[idx] = g(v); nothing)
@inline _mma_epi_store!(::Val{NK}, g::G, C, idx, v) where {NK,G} =
    (@atomic C[idx] += v; nothing)

# ── Epilogue copy Ds→C: VECTORISED interior fast path + scalar masked fallback ──
# The Ds→C copy is ~80% of a rank-short (short-K) GEMM's time (measured, RTX1000 Ada
# outer 16384²×64: epilogue ≈ 5.4/6.7 ms) and the scalar per-element copy ran at only
# ~60% of the device write-BW floor. The interior path writes `EPB = 16÷sizeof(S)`
# contiguous outputs per `KI.vstore!` (one wide store instead of EPB scalar stores),
# lifting the epilogue toward full write BW — so it helps EVERY epilogue-heavy shape,
# most on rank-short. Selected per block, warp-uniformly: taken only when the whole
# block tile is in-bounds (no masking), `M % EPB == 0` (16-byte-aligned column runs),
# and `NK == 1` (direct write). Edges / misalignment / split-K (NK>1 atomic-add) fall
# to the byte-identical scalar path. CROSS-ARCH: plain KI vload/vstore! + array
# indexing over the F32 `Ds` tile and the S-typed `C` — no WMMA/MFMA specifics, so the
# same win lands on the NVIDIA and AMD MMA paths. Top-level (not a kernel closure) so
# nothing `g` captures is reassigned — the boxing trap the register-tile note documents.
@inline function _mma_epi_copy!(::Val{NK}, ::Val{EPB}, g::G, C, Ds, ::Type{S},
                                brow, bcol, j, BM, BN, DN, Nt, WN, ldD, wg, lid, M, N) where {NK,EPB,G,S}
    if NK == 1 && EPB > 1 && (M % EPB == 0) && (brow * BM + BM <= M) && (bcol * BN + BN <= N)
        nrb = BM ÷ EPB
        p = lid
        while p <= nrb * DN
            e   = p - 1
            rb  = e % nrb
            cD  = e ÷ nrb
            wnc = cD ÷ Nt
            t   = cD % Nt
            r0  = rb * EPB
            m0  = brow * BM + r0
            n   = bcol * BN + wnc * (WN * Nt) + (j - 1) * Nt + t + 1
            # Read Ds with SCALAR shared loads (fast, and `vload` can't cast the shared
            # address space) — the win is the ONE wide GLOBAL `vstore!`, not the read.
            base = cD * ldD + r0
            vs = ntuple(q -> S(g(@inbounds Ds[base + q])), Val(EPB))   # g + cast to S
            # Val(false) = idx is the literal element offset, NOT a rebased Nitem-block index.
            KI.vstore!(C, (n - 1) * M + m0 + 1, vs, Val(false))        # one wide store to C
            p += wg
        end
    else
        p = lid
        while p <= BM * DN
            e   = p - 1
            r   = e % BM
            cD  = e ÷ BM
            wnc = cD ÷ Nt
            t   = cD % Nt
            m = brow * BM + r + 1
            n = bcol * BN + wnc * (WN * Nt) + (j - 1) * Nt + t + 1
            if m <= M && n <= N
                _mma_epi_store!(Val(NK), g, C, (n - 1) * M + m, @inbounds Ds[cD * ldD + r + 1])
            end
            p += wg
        end
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

# ── DOUBLE BUFFERING (`DB`) ────────────────────────────────────────────────
# Single-buffered, staging and compute strictly ALTERNATE: the tensor cores idle
# while the memory pipes fill As/Bs, and vice versa. The second `@synchronize`
# exists only for the write-after-read hazard — the next panel overwrites As/Bs
# while warps may still be reading them.
#
# With `DB` the shared tiles are TWICE as wide (2·BK columns of As, 2·BN of Bs) and
# the loop computes on buffer `p` while staging panel `kt+1` into buffer `1-p`.
# The loads for the next panel are therefore in flight across the whole MMA
# sequence of the current one. It also HALVES the barrier count: one barrier per
# panel instead of two, because within an iteration the writes (buffer q) and the
# reads (buffer p) touch disjoint memory, and every cross-iteration hazard on a
# given buffer is separated by exactly one barrier. See the K-loop for the proof.
#
# WHY IT IS AFFORDABLE HERE (measured, RTX1000 Ada — `xp/gemm/db_occupancy.jl`):
# forcing 2 blocks/SM → 1 with the K-loop bit-identical costs 1.41×, so the kernel
# is LATENCY-bound and leans on inter-block overlap. But the fast tiers run 32
# warps = 1024 threads/block, and Ada caps an SM at 1536 threads, so those tiers
# are already 1 block/SM for a reason that has NOTHING to do with shared memory —
# doubling As/Bs there costs no occupancy at all, as long as the total still fits
# `KI.max_dynamic_localmem`. On the tiers where `Ds` dominates the union it is
# free outright. `DB` is a KNOB, defaulted per tier by `default_gemm_mma_db`, so
# the single-buffered path stays available for an honest A/B and for tiers (and
# devices) where the doubled footprint would in fact cost a block.

# `Ds` (BM × NWN·Nt, one epilogue chunk) unions with `As`/`Bs` at offset 0 — both
# are dead after the K-loop's trailing `@synchronize` ⇒ max, not sum. `_align16`
# keeps the Bs sub-blob 16-byte aligned inside the one dynamic allocation.
@inline _align16(nbytes) = ((nbytes + 15) ÷ 16) * 16
# `NB` = number of shared buffers (1 single-, 2 double-buffered). With NB=1 both
# expressions below reduce EXACTLY to the pre-DB layout, byte for byte.
# `_gemm_mma_ldm_lda(BK)` = the K-contiguous A leading dim for the ldmatrix path:
# BK + 4 (Float32 tile). +4 keeps the row-start 16-byte aligned (ldmatrix.x4 needs it
# — a non-multiple-of-4 stride device-traps MISALIGNED) AND de-phases the shared banks
# (a pow2 BK stride serialises the reads). A K-contig A tile (BM rows) at this ld is
# LARGER than the M-contig (BM+PC)×BK tile, so the shmem sizing must know `ldm`.
@inline _gemm_mma_ldm_lda(BK) = BK + 4
@inline function _gemm_mma_shmem(::Type{CT}, ::Type{AccT}, BM, BN, BK, DN, PC, PA,
                                 NB::Int=1, ldm::Bool=false) where {CT,AccT}
    # As and Bs are each ONE array of NB·(BK resp. BN) columns, so the buffers are
    # separated by a plain column stride — the 16-byte round-up applies once, to
    # the whole As blob, never between buffers (padding between them would desync
    # the element-offset `sh0` the kernel uses to address buffer `p`).
    ab = _gemm_mma_off_b(CT, BM, BK, PC, NB, ldm) + (BK + PC) * BN * NB * sizeof(CT)
    d  = (BM + PA) * DN * sizeof(AccT)
    return max(ab, d)
end
@inline _gemm_mma_off_b(::Type{CT}, BM, BK, PC, NB::Int=1, ldm::Bool=false) where {CT} =
    _align16((ldm ? _gemm_mma_ldm_lda(BK) * BM : (BM + PC) * BK) * NB * sizeof(CT))

@kernel unsafe_indices = true function gemm_mma_kernel!(
    g::G,
    C::AbstractMatrix{S},
    A::AbstractMatrix{CT},
    B::AbstractMatrix{CT},
    layA::LA, layB::LB,
    cfg::CFG,
    ::Val{Mt}, ::Val{Nt}, ::Val{Kt}, ::Val{WS},
    ::Val{WM}, ::Val{WN}, ::Val{NWM}, ::Val{NWN}, ::Val{WK}, ::Val{PC}, ::Val{PA},
    ::Val{DB}, ::Val{ASYNCA}, ::Val{ASYNCB}, ::Val{NK}, ::Val{LDM},
    ::Type{AccT},
    M::Int, N::Int, K::Int, GSW::Int
) where {G<:Function,S,CT,LA<:FragLayout,LB<:FragLayout,CFG,
         Mt,Nt,Kt,WS,WM,WN,NWM,NWN,WK,PC,PA,DB,ASYNCA,ASYNCB,NK,LDM,AccT}
    @uniform begin
        # `ASYNC` gates the cp.async commit/wait scaffolding: emit it only when at
        # least one operand actually stages asynchronously, else the pipeline ops
        # would fire around a purely register-staged loop. `va`/`vb` are the
        # per-operand staging selectors (see `_stage_a`/`_stage_b`).
        ASYNC = (ASYNCA !== :none) || (ASYNCB !== :none)
        va = Val(ASYNCA)
        vb = Val(ASYNCB)
        BM  = NWM * WM * Mt
        BN  = NWN * WN * Nt
        BK  = WK * Kt
        DN  = NWN * Nt            # D-tile width = one epilogue column chunk
        wg  = NWM * NWN * WS
        nbx = cld(M, BM)          # block-rows across the C grid
        nby = cld(N, BN)          # block-cols across the C grid
        nkt = cld(K, BK)          # K panels
        pps = cld(nkt, NK)        # K panels per split-K slice (NK==1 ⇒ pps==nkt)
        ldA = BM + PC             # padded leading dims — see `default_mma_pad`
        ldB = BK + PC
        ldD = BM + PA
        # LDM (ldmatrix tf32) A tile is K-contiguous, ld = BK+4 (16-byte aligned for
        # ldmatrix.x4 + bank-de-phased); its blob is sized by `_gemm_mma_off_b(…, LDM)`.
        ldLdmA = _gemm_mma_ldm_lda(BK)
        NB  = DB ? 2 : 1          # shared buffers (see the DOUBLE BUFFERING note)
        offB = _gemm_mma_off_b(CT, BM, BK, PC, NB, LDM)
    end

    # 2D tiles — ld = size(·,1) is what the fragment loads consume (W2-ld), and it
    # is INDEPENDENT of NB: widening the tile adds columns, so the fragment loads
    # are bit-identical and buffer `p` is reached by a pure column offset.
    # Ds aliases As/Bs at offset 0 (both dead after the K-loop). LDM stages A
    # K-contiguously (ld = ldLdmA) and reads it via a raw pointer, so it declares As
    # 1-D (bigger blob, sized into `offB`); the scalar path keeps the 2-D tile its
    # `load_a` indexes. LDM is a compile-time `Val` ⇒ exactly one branch is emitted.
    As = LDM ? KI.@dynlocalmem(CT, (ldLdmA * BM * NB,), 0) :
               KI.@dynlocalmem(CT, (ldA, BK * NB), 0)
    Bs = KI.@dynlocalmem CT (ldB, BN * NB) offB
    Ds = KI.@dynlocalmem AccT (ldD, DN) 0

    lid  = Int(@index(Local))
    gid  = Int(@index(Group))
    blk  = gid - 1                # 0-based (block-row, block-col, k-slice) index
    ksl  = blk ÷ (nbx * nby)     # 0-based K-slice (always 0 when NK==1)
    b2   = blk % (nbx * nby)     # block within this k-slice's C plane
    # L2-aware threadblock swizzle: iterate GSW consecutive block-ROWS, sweeping ALL
    # block-COLS within the group before the next group — so the group's B column-tiles
    # stay resident in L2 across the GSW rows (a large win once A/B exceed L2, neutral
    # when they fit). GSW=1 reduces to a plain bcol-fastest raster. Set by
    # `default_gemm_swizzle`.
    bpg  = GSW * nby
    grp  = b2 ÷ bpg
    br0  = grp * GSW
    rows = min(GSW, nbx - br0)
    iig  = b2 - grp * bpg
    brow = br0 + iig % rows       # 0-based block-row
    bcol = iig ÷ rows             # 0-based block-col
    ktlo = ksl * pps             # first K panel this slice owns …
    kthi = min(ktlo + pps, nkt)  # … one past its last (NK==1 ⇒ 0 … nkt)
    warp = (lid - 1) ÷ WS         # 0-based warp id within the block
    wm   = warp % NWM
    wn   = warp ÷ NWM
    wrow0 = wm * (WM * Mt) + 1    # 1-based sub-tile origin inside As / Ds
    wcol0 = wn * (WN * Nt) + 1

    accs = ntuple(_ -> KI.MMA.fill_c(cfg, KI.MMA.acc_identity(cfg)), Val(WM * WN))

    if DB
        # ── Double-buffered K-loop ──────────────────────────────────────────
        # Prologue stages panel 0 into buffer 0; iteration `kt` computes on buffer
        # `pb` while staging panel `kt+1` into `qb = 1-pb`.
        #
        # ONE barrier per panel is sufficient, and this is the whole correctness
        # argument. Per iteration the order is: barrier — write qb — read pb.
        #   • write(qb)@kt vs read(qb)@kt+1  → separated by the barrier at kt+1 ✓
        #   • read(pb)@kt  vs write(pb)@kt+1 → separated by the barrier at kt+1 ✓
        #   • write(qb) vs read(pb) within one iteration → DISJOINT buffers ✓
        # (Single-buffered needs two because those first two hazards collapse onto
        # the same memory.) The barrier AFTER the loop is not optional: the last
        # iteration's fragment reads of As/Bs must be fenced before the epilogue
        # writes `Ds`, which ALIASES the blob at offset 0.
        if ASYNC
            # cp.async variant: mirror the register hazard argument but overlap the
            # global→shared copy with the tensor-core math. Ordering per iteration is
            # ISSUE(qb) → COMMIT → COMPUTE(pb) → WAIT(0) → barrier → swap: the next
            # panel's copies are in flight across the whole MMA sequence of the
            # current one (the overlap), and the `wait(0)` before the barrier drains
            # them so buffer `qb` is fully landed before it is read next iteration.
            # `wait(Val(0))` (drain-all) is the conservative granularity the prototype
            # proved; `Val(1)` to keep the next tile in flight is a later refinement.
            # The HYBRID staging also emits register stores for edge/misaligned
            # chunks — synchronous, made visible by the same barrier.
            _stage_a(va, As, A, layA, brow, ktlo, M, K, lid, wg, ldA, 0, Val(BM), Val(BK))
            _stage_b(vb, Bs, B, layB, bcol, ktlo, K, N, lid, wg, ldB, 0, Val(BK), Val(BN))
            KI.async_commit()
            KI.async_wait(Val(0))
            pb = 0
            kt = ktlo
            while kt < kthi
                @synchronize
                qb = 1 - pb
                if kt + 1 < kthi
                    _stage_a(va, As, A, layA, brow, kt + 1, M, K, lid, wg, ldA,
                             qb * BK * ldA, Val(BM), Val(BK))
                    _stage_b(vb, Bs, B, layB, bcol, kt + 1, K, N, lid, wg, ldB,
                             qb * BN * ldB, Val(BK), Val(BN))
                    KI.async_commit()
                end
                koffA0 = pb * BK + 1
                wcolp  = wcol0 + pb * BN
                ks = 0
                while ks < WK
                    accs = _mma_tile_step(cfg, As, Bs, wrow0, wcolp,
                                          koffA0 + ks * Kt, ks * Kt + 1, accs,
                                          Val(WM), Val(WN), Val(Mt), Val(Nt))
                    ks += 1
                end
                if kt + 1 < kthi
                    KI.async_wait(Val(0))   # buffer qb landed before next-iter read
                end
                pb = qb
                kt += 1
            end
            @synchronize                # fences As/Bs before Ds reuses the blob
        else
            if LDM
                # tf32 ldmatrix DB path: K-contiguous A buffers at linear offset
                # `qb*BM*BK` (no pad — fits As), B reused ColMajor at `qb*BN*ldB`;
                # warp-collective ldmatrix loads read buffer `pb` via a base pointer.
                _stage_a_mma_ldm!(As, A, layA, brow, ktlo, M, K, lid, wg, 0, ldLdmA, Val(BM), Val(BK))
                _stage_b_mma!(Bs, B, layB, bcol, ktlo, K, N, lid, wg, ldB, 0, Val(BK), Val(BN))
                pb = 0
                kt = ktlo
                while kt < kthi
                    @synchronize
                    qb = 1 - pb
                    if kt + 1 < kthi
                        _stage_a_mma_ldm!(As, A, layA, brow, kt + 1, M, K, lid, wg,
                                          qb * BM * ldLdmA, ldLdmA, Val(BM), Val(BK))
                        _stage_b_mma!(Bs, B, layB, bcol, kt + 1, K, N, lid, wg, ldB,
                                      qb * BN * ldB, Val(BK), Val(BN))
                    end
                    Aptr = pointer(As, pb * BM * ldLdmA + 1)
                    Bptr = pointer(Bs, pb * BN * ldB + 1)
                    ks = 0
                    while ks < WK
                        accs = _mma_tile_step_ldm(cfg, Aptr, Bptr, wrow0 - 1, wcol0 - 1, ks * Kt,
                                                  ldLdmA, ldB, accs, Val(WM), Val(WN), Val(Mt), Val(Nt))
                        ks += 1
                    end
                    pb = qb
                    kt += 1
                end
                @synchronize
            else
                _stage_a_mma!(As, A, layA, brow, ktlo, M, K, lid, wg, ldA, 0, Val(BM), Val(BK))
                _stage_b_mma!(Bs, B, layB, bcol, ktlo, K, N, lid, wg, ldB, 0, Val(BK), Val(BN))
                pb = 0
                kt = ktlo
                while kt < kthi
                    @synchronize
                    qb = 1 - pb
                    if kt + 1 < kthi
                        _stage_a_mma!(As, A, layA, brow, kt + 1, M, K, lid, wg, ldA,
                                      qb * BK * ldA, Val(BM), Val(BK))
                        _stage_b_mma!(Bs, B, layB, bcol, kt + 1, K, N, lid, wg, ldB,
                                      qb * BN * ldB, Val(BK), Val(BN))
                    end
                    koffA0 = pb * BK + 1        # buffer offset lives in As's COLUMN index
                    wcolp  = wcol0 + pb * BN    # ... and in Bs's COLUMN index
                    ks = 0
                    while ks < WK
                        accs = _mma_tile_step(cfg, As, Bs, wrow0, wcolp,
                                              koffA0 + ks * Kt, ks * Kt + 1, accs,
                                              Val(WM), Val(WN), Val(Mt), Val(Nt))
                        ks += 1
                    end
                    pb = qb
                    kt += 1
                end
                @synchronize                # fences As/Bs before Ds reuses the blob
            end
        end
    else
        if ASYNC
            kt = ktlo
            while kt < kthi
                _stage_a(va, As, A, layA, brow, kt, M, K, lid, wg, ldA, 0, Val(BM), Val(BK))
                _stage_b(vb, Bs, B, layB, bcol, kt, K, N, lid, wg, ldB, 0, Val(BK), Val(BN))
                KI.async_commit()
                KI.async_wait(Val(0))
                @synchronize
                ks = 0
                while ks < WK
                    accs = _mma_tile_step(cfg, As, Bs, wrow0, wcol0, ks * Kt + 1, ks * Kt + 1,
                                          accs, Val(WM), Val(WN), Val(Mt), Val(Nt))
                    ks += 1
                end
                @synchronize            # also fences As/Bs before Ds reuses the blob
                kt += 1
            end
        else
            # NOTE: nested `if LDM` (NOT `elseif`) — KA's kernel-split can't handle an
            # `elseif` whose body contains `@synchronize` (the block-wrapped condition →
            # `unblock` → bare Symbol → `is_scope_construct(::Symbol)`). Keep it nested.
            if LDM
                # tf32 ldmatrix path (single-buffered): K-contiguous A stage (ld=BK, no
                # pad, fits the As blob), B reused ColMajor, warp-collective ldmatrix loads.
                kt = ktlo
                while kt < kthi
                    _stage_a_mma_ldm!(As, A, layA, brow, kt, M, K, lid, wg, 0, ldLdmA, Val(BM), Val(BK))
                    _stage_b_mma!(Bs, B, layB, bcol, kt, K, N, lid, wg, ldB, 0, Val(BK), Val(BN))
                    @synchronize
                    Aptr = pointer(As, 1)
                    Bptr = pointer(Bs, 1)
                    ks = 0
                    while ks < WK
                        accs = _mma_tile_step_ldm(cfg, Aptr, Bptr, wrow0 - 1, wcol0 - 1, ks * Kt,
                                                  ldLdmA, ldB, accs, Val(WM), Val(WN), Val(Mt), Val(Nt))
                        ks += 1
                    end
                    @synchronize            # also fences As/Bs before Ds reuses the blob
                    kt += 1
                end
            else
                kt = ktlo
                while kt < kthi
                    _stage_a_mma!(As, A, layA, brow, kt, M, K, lid, wg, ldA, 0, Val(BM), Val(BK))
                    _stage_b_mma!(Bs, B, layB, bcol, kt, K, N, lid, wg, ldB, 0, Val(BK), Val(BN))
                    @synchronize
                    ks = 0
                    while ks < WK
                        accs = _mma_tile_step(cfg, As, Bs, wrow0, wcol0, ks * Kt + 1, ks * Kt + 1,
                                              accs, Val(WM), Val(WN), Val(Mt), Val(Nt))
                        ks += 1
                    end
                    @synchronize            # also fences As/Bs before Ds reuses the blob
                    kt += 1
                end
            end
        end
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
        _mma_epi_copy!(Val(NK), Val(16 ÷ sizeof(S)), g, C, Ds, S,
                       brow, bcol, j, BM, BN, DN, Nt, WN, ldD, wg, lid, M, N)
        @synchronize                  # before the next chunk overwrites Ds
        j += 1
    end
end

# Split-K epilogue: after the split-K main kernel has atomic-reduced every slice's
# partial into the AccT accumulator `Cacc`, apply the output map `g` and cast into
# the S-typed output `C`. One thread per output element; ndrange rounded up to a
# workgroup multiple, so the bare global index is masked against `n`.
@kernel unsafe_indices = true function gemm_splitk_epilogue!(
    g::G, C::AbstractMatrix{S}, Cacc::AbstractMatrix{AccT}, n::Int
) where {G<:Function,S,AccT}
    i = Int(@index(Global))
    if i <= n
        @inbounds C[i] = g(Cacc[i])
    end
end
