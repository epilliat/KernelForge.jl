# Generalized GEMM — Phase 3: tensor-core (MMA) family (F16/BF16 → F32).
#
# Separate file so gemm_test.jl's exact generic checks stay untouched. MMA cases use
# relaxed tolerances (KI's measured values: ~1e-2 F16, ~5e-2 BF16) and are pinned
# BOTH against a Float32 CPU reference AND against KF's own generic kernel on the
# SAME low-precision inputs (cross-family agreement).
#
# HISTORY — this file used to SKIP itself under `--check-bounds=yes` (the Pkg.test
# default), blaming bounds-check branches for diverging the warp inside the
# warp-cooperative WMMA region: 14/18 shapes came out silently wrong. That diagnosis
# was WRONG. The real cause was in KernelIntrinsics: the MMA entry points were
# installed through overlay method tables, inference could not see through them, and
# the K-loop accumulator degenerated to an `undef` phi — which materialises as 0 or
# as garbage depending on unrelated code shape, which is exactly why bounds checks
# looked like the trigger. Fixed in KI 0.1.14 (backend token, see KI src/mma.jl,
# section « Jeton matériel »). These tests now run and pass in BOTH check-bounds
# modes; do not re-add the skip.
#
# The HW MMA path covers NN/NT/TN; the TT layout (and F32/F64) route to the generic
# K1 — `_resolve_family` rejects a forced `:mma` for TT, so TT is checked on generic.

if !KF.KI.MMA.mma_supported(backend, KF.KI.MMA.MMAConfig{16,16,16,Float16,Float32}())
    @info "gemm MMA tests SKIPPED — no HW F16→F32 MMA path on this device"
else

@testset "gemm MMA (Phase 3, F16/BF16 → F32)" begin
    mma_tol(::Type{Float16}) = 1f-2
    mma_tol(::Type{Core.BFloat16}) = 5f-2

    # `Core.BFloat16` is a bare primitive type in Base: usable only if some
    # package supplies its conversions. The CUDA test env gets them transitively;
    # the ROC env does NOT (`Core.BFloat16(::Float32)` is a MethodError on
    # MI300A), so the reference values cannot even be built there. The GPU path
    # is fine — gfx942 announces BF16 MFMA shapes — this is a host-side gap, so
    # skip loudly rather than fail or pretend the coverage exists.
    _bf16_usable = hasmethod(Core.BFloat16, Tuple{Float32})
    _bf16_usable || @info "BFloat16 GEMM tests SKIPPED: Core.BFloat16 conversions \
                           unavailable in this environment (host-side, not a kernel gap)"
    _mma_eltypes = _bf16_usable ? (Float16, Core.BFloat16) : (Float16,)

    @testset "$CT MMA layouts vs CPU F32 + cross-family" for CT in _mma_eltypes
        rng = Xoshiro(30)
        tol = mma_tol(CT)
        for (tA, tB) in [(:N, :N), (:T, :N), (:N, :T)],       # HW MMA layouts
            # all ≥ one 16×16 MMA tile (smaller outputs route to generic by design);
            # deliberately non-multiples of the tile to exercise the edge masking.
            (M, N, K) in [(16, 16, 16), (64, 48, 32), (100, 80, 50),
                          (200, 137, 71), (512, 256, 128), (17, 19, 33)]
            # Draw in Float32 then convert: `rand(rng, Core.BFloat16, …)` needs a
            # Random.Sampler that is NOT defined in every backend environment
            # (present under the CUDA test env, absent under the ROC one — it
            # errored on MI300A). Converting is portable and equally random.
            _draw(::Type{T}, dims...) where {T} = T.(rand(rng, Float32, dims...))
            Astore = tA === :N ? _draw(CT, M, K) : _draw(CT, K, M)
            Bstore = tB === :N ? _draw(CT, K, N) : _draw(CT, N, K)
            opA = tA === :N ? Astore : permutedims(Astore)
            opB = tB === :N ? Bstore : permutedims(Bstore)
            ref = Float32.(opA) * Float32.(opB)               # Float32 CPU reference
            dA = AT(Astore); dB = AT(Bstore)
            C = KF.gemm(dA, dB; tA, tB, family=:mma)          # explicit opt-in
            @test eltype(C) === Float32                        # accT bumps F16/BF16 → F32
            @test maximum(abs.(Array(C) .- ref) ./ (abs.(ref) .+ 1f-2)) < tol
            # cross-family: same inputs through the generic kernel (F32 accumulation).
            Cg = KF.gemm(dA, dB; tA, tB, family=:generic, accT=Float32)
            @test isapprox(Array(C), Array(Cg); rtol=tol)
        end
    end

    @testset "TT layout routes to generic and is correct" begin
        rng = Xoshiro(33)
        for (M, N, K) in [(64, 48, 32), (200, 137, 71)]
            Astore = rand(rng, Float16, K, M); Bstore = rand(rng, Float16, N, K)
            ref = Float32.(permutedims(Astore)) * Float32.(permutedims(Bstore))
            C = KF.gemm(AT(Astore), AT(Bstore); tA=:T, tB=:T, accT=Float32)
            @test maximum(abs.(Array(C) .- ref) ./ (abs.(ref) .+ 1f-2)) < 1f-2
        end
    end

    @testset "g epilogue on :mma" begin
        rng = Xoshiro(31)
        M, N, K = 128, 96, 64
        A = rand(rng, Float16, M, K); B = rand(rng, Float16, K, N)
        C = KF.gemm(AT(A), AT(B); g = x -> 2f0 * x + 1f0, family=:mma)
        ref = 2 .* (Float32.(A) * Float32.(B)) .+ 1
        @test maximum(abs.(Array(C) .- ref) ./ (abs.(ref) .+ 1f-2)) < 2f-2
    end

    @testset "family routing + forced-family errors" begin
        A16 = AT(rand(Float16, 32, 32)); B16 = AT(rand(Float16, 32, 32))
        A32 = AT(rand(Float32, 32, 32)); B32 = AT(rand(Float32, 32, 32))
        # auto must NOT select :mma (opt-in only) → agrees with forced generic.
        @test isapprox(Array(KF.gemm(A16, B16)),
                       Array(KF.gemm(A16, B16; family=:generic, accT=Float32)); rtol=1f-3)
        # forced :mma agrees with generic on the same inputs.
        @test isapprox(Array(KF.gemm(A16, B16; family=:mma)),
                       Array(KF.gemm(A16, B16; family=:generic, accT=Float32)); rtol=1f-2)
        # forced :mma on an unsupported config errors (never silent).
        @test_throws ErrorException KF.gemm(A32, B32; family=:mma)               # F32: no HW path
        @test_throws ErrorException KF.gemm(A16, B16; tA=:T, tB=:T, family=:mma) # TT excluded
        @test_throws ErrorException KF.gemm(A16, B16; family=:mma, BM=32)        # generic knobs
    end

    @testset "mma_shapes sweep — announced F16 shapes" begin
        rng = Xoshiro(32)
        for s in KF.KI.MMA.mma_shapes(backend)
            s.compute === Float16 && s.acc === Float32 || continue
            M, N, K = 3 * s.M, 2 * s.N, 4 * s.K
            A = rand(rng, Float16, M, K); B = rand(rng, Float16, K, N)
            C = KF.gemm(AT(A), AT(B); family=:mma)
            ref = Float32.(A) * Float32.(B)
            @test maximum(abs.(Array(C) .- ref) ./ (abs.(ref) .+ 1f-2)) < 1f-2
        end
    end
end

end  # mma_supported guard
