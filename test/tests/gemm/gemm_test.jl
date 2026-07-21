# Generalized GEMM — Phase 1 (NN generic register-blocked tiled core).
#
# Correctness matrix: vs CPU A*B (F32 reference promoted to Float64 per the
# "F32 stress → Float64" invariant), a generalized-op reference (tropical max-plus,
# boolean semiring), a custom isbits struct eltype, Float32 AND Float64, and a
# battery of non-tile-multiple / degenerate edge shapes exercised under
# `--check-bounds=yes` (Pkg.test default). Phase 1 = NN only.

# ── local isbits struct eltype (top-level so `struct` is legal) ──────────────
struct GVec2
    a::Float32
    b::Float32
end
gvec_f(x::Float32, y::Float32) = GVec2(x * y, x + y)
gvec_op(u::GVec2, v::GVec2) = GVec2(u.a + v.a, u.b + v.b)

# CPU reference: C[m,n] = op_k f(A[m,k], B[k,n]), seeded at k=1 (no identity).
function gemm_ref(f, op, A::AbstractMatrix, B::AbstractMatrix)
    M, K = size(A); Kb, N = size(B)
    @assert K == Kb
    T = typeof(f(A[1, 1], B[1, 1]))
    C = Matrix{T}(undef, M, N)
    for n in 1:N, m in 1:M
        acc = f(A[m, 1], B[1, n])
        for k in 2:K
            acc = op(acc, f(A[m, k], B[k, n]))
        end
        C[m, n] = acc
    end
    return C
end

# A representative set of shapes: square, tall, wide, tile-multiples, prime/odd
# (non-multiples of the 64×64×8 default tile), K=1, and the 1×1 / M×1 / 1×N edges.
const GEMM_SHAPES = [
    (1, 1, 1), (1, 1, 8), (3, 3, 3), (4, 1, 7), (1, 5, 9),
    (64, 64, 8), (65, 63, 9), (100, 100, 100), (128, 96, 64),
    (200, 137, 71), (256, 256, 32), (127, 129, 130), (300, 50, 500),
    (50, 300, 500), (513, 257, 33), (1000, 8, 8), (8, 1000, 8),
]

@testset "gemm (Phase 1 NN)" begin

    @testset "plain A*B vs CPU — Float32" begin
        rng = Xoshiro(1)
        for (M, N, K) in GEMM_SHAPES
            A = rand(rng, Float32, M, K)
            B = rand(rng, Float32, K, N)
            dA = AT(A); dB = AT(B)
            C = KF.gemm(dA, dB)
            ref = Float64.(A) * Float64.(B)          # Float64 reference (F32 stress → F64)
            @test isapprox(Array(C), ref; rtol=1f-3)
        end
    end

    @testset "plain A*B vs CPU — Float64" begin
        rng = Xoshiro(2)
        for (M, N, K) in GEMM_SHAPES
            A = rand(rng, Float64, M, K)
            B = rand(rng, Float64, K, N)
            dA = AT(A); dB = AT(B)
            C = KF.gemm(dA, dB)
            @test isapprox(Array(C), A * B; rtol=1e-10)
        end
    end

    @testset "in-place gemm! + simplified/full API" begin
        rng = Xoshiro(3)
        M, N, K = 200, 137, 71
        A = rand(rng, Float32, M, K); B = rand(rng, Float32, K, N)
        dA = AT(A); dB = AT(B)
        ref = Float64.(A) * Float64.(B)
        C = KA.allocate(backend, Float32, M, N)
        KF.gemm!(C, dA, dB)                          # simplified in-place
        @test isapprox(Array(C), ref; rtol=1f-3)
        fill!(C, 0f0)
        KF.gemm!(*, +, C, dA, dB)                    # full-op in-place
        @test isapprox(Array(C), ref; rtol=1f-3)
        C2 = KF.gemm(*, +, dA, dB)                   # full-op allocating
        @test isapprox(Array(C2), ref; rtol=1f-3)
    end

    @testset "custom epilogue g" begin
        rng = Xoshiro(4)
        M, N, K = 128, 96, 64
        A = rand(rng, Float32, M, K); B = rand(rng, Float32, K, N)
        dA = AT(A); dB = AT(B)
        C = KF.gemm(dA, dB; g = x -> 2f0 * x + 1f0)
        ref = 2 .* (Float64.(A) * Float64.(B)) .+ 1
        @test isapprox(Array(C), ref; rtol=1f-3)
    end

    @testset "generalized op — tropical (max, +)" begin
        rng = Xoshiro(5)
        for (M, N, K) in [(3, 3, 3), (100, 80, 40), (65, 63, 9), (257, 129, 33)]
            A = rand(rng, Float32, M, K); B = rand(rng, Float32, K, N)
            dA = AT(A); dB = AT(B)
            C = KF.gemm((a, b) -> a + b, max, dA, dB)
            ref = gemm_ref((a, b) -> a + b, max, A, B)
            @test isapprox(Array(C), ref; rtol=1f-4)
        end
    end

    @testset "generalized op — boolean semiring (|, &)" begin
        rng = Xoshiro(6)
        for (M, N, K) in [(3, 3, 3), (100, 80, 40), (127, 65, 50)]
            A = rand(rng, Bool, M, K); B = rand(rng, Bool, K, N)
            dA = AT(A); dB = AT(B)
            C = KF.gemm(&, |, dA, dB)
            ref = gemm_ref(&, |, A, B)
            @test Array(C) == ref
        end
    end

    @testset "custom isbits struct eltype (GVec2)" begin
        rng = Xoshiro(7)
        for (M, N, K) in [(3, 3, 3), (100, 80, 40), (200, 137, 71)]
            A = rand(rng, Float32, M, K); B = rand(rng, Float32, K, N)
            dA = AT(A); dB = AT(B)
            C = KF.gemm(gvec_f, gvec_op, dA, dB)
            ref = gemm_ref(gvec_f, gvec_op, A, B)
            Cc = Array(C)
            @test all(isapprox(Cc[i].a, ref[i].a; rtol=1f-3) &&
                      isapprox(Cc[i].b, ref[i].b; rtol=1f-3) for i in eachindex(ref))
        end
    end

    @testset "explicit tile knobs + non-multiple shapes" begin
        rng = Xoshiro(8)
        for (M, N, K) in [(150, 90, 45), (257, 257, 17)]
            A = rand(rng, Float32, M, K); B = rand(rng, Float32, K, N)
            dA = AT(A); dB = AT(B)
            C = KF.gemm(dA, dB; BM=32, BN=32, BK=4, TM=2, TN=2)
            ref = Float64.(A) * Float64.(B)
            @test isapprox(Array(C), ref; rtol=1f-3)
        end
    end

    # ── Phase 2: transpose states NT / TN / TT ───────────────────────────────
    # For flags (tA,tB), the STORED arrays have shapes:
    #   A: :N → M×K, :T → K×M     B: :N → K×N, :T → N×K
    # and the logical product is  opA(A) * opB(B), opX = transpose iff flag :T.
    @testset "all layouts vs CPU — Float32 & Float64" begin
        rng = Xoshiro(20)
        layouts = [(:N, :N), (:T, :N), (:N, :T), (:T, :T)]
        shapes = [(1, 1, 1), (3, 3, 3), (65, 63, 9), (100, 80, 40),
                  (128, 96, 64), (200, 137, 71), (257, 129, 33), (300, 50, 71)]
        for S in (Float32, Float64), (tA, tB) in layouts, (M, N, K) in shapes
            Astore = tA === :N ? rand(rng, S, M, K) : rand(rng, S, K, M)
            Bstore = tB === :N ? rand(rng, S, K, N) : rand(rng, S, N, K)
            opA = tA === :N ? Astore : permutedims(Astore)   # → M×K
            opB = tB === :N ? Bstore : permutedims(Bstore)   # → K×N
            ref = (S === Float32 ? Float64.(opA) * Float64.(opB) : opA * opB)
            C = KF.gemm(AT(Astore), AT(Bstore); tA, tB)
            rtol = S === Float32 ? 1f-3 : 1e-10
            @test isapprox(Array(C), ref; rtol)
        end
    end

    @testset "generalized op (tropical) per non-NN layout" begin
        rng = Xoshiro(21)
        for (tA, tB) in [(:T, :N), (:N, :T), (:T, :T)], (M, N, K) in [(65, 63, 9), (100, 80, 40)]
            Astore = tA === :N ? rand(rng, Float32, M, K) : rand(rng, Float32, K, M)
            Bstore = tB === :N ? rand(rng, Float32, K, N) : rand(rng, Float32, N, K)
            opA = tA === :N ? Astore : permutedims(Astore)
            opB = tB === :N ? Bstore : permutedims(Bstore)
            C = KF.gemm((a, b) -> a + b, max, AT(Astore), AT(Bstore); tA, tB)
            ref = gemm_ref((a, b) -> a + b, max, opA, opB)
            @test isapprox(Array(C), ref; rtol=1f-4)
        end
    end

    @testset "custom struct eltype per non-NN layout" begin
        rng = Xoshiro(22)
        for (tA, tB) in [(:T, :N), (:N, :T), (:T, :T)], (M, N, K) in [(127, 65, 50), (200, 137, 71)]
            Astore = tA === :N ? rand(rng, Float32, M, K) : rand(rng, Float32, K, M)
            Bstore = tB === :N ? rand(rng, Float32, K, N) : rand(rng, Float32, N, K)
            opA = tA === :N ? Astore : permutedims(Astore)
            opB = tB === :N ? Bstore : permutedims(Bstore)
            C = KF.gemm(gvec_f, gvec_op, AT(Astore), AT(Bstore); tA, tB)
            ref = gemm_ref(gvec_f, gvec_op, opA, opB)
            Cc = Array(C)
            @test all(isapprox(Cc[i].a, ref[i].a; rtol=1f-3) &&
                      isapprox(Cc[i].b, ref[i].b; rtol=1f-3) for i in eachindex(ref))
        end
    end

    @testset "transposed layouts consistency (A'B == (Aᵀ)·B)" begin
        # Cross-check: gemm(A,B;tA=:T) must equal gemm(Aᵀ_materialised, B).
        rng = Xoshiro(23)
        M, N, K = 150, 90, 45
        Astore = rand(rng, Float32, K, M)          # stored K×M
        B = rand(rng, Float32, K, N)
        Ct = KF.gemm(AT(Astore), AT(B); tA=:T)
        Cn = KF.gemm(AT(permutedims(Astore)), AT(B))
        @test isapprox(Array(Ct), Array(Cn); rtol=1f-4)
    end

    @testset "bad transpose flag rejected" begin
        A = AT(rand(Float32, 8, 8)); B = AT(rand(Float32, 8, 8))
        @test_throws ErrorException KF.gemm(A, B; tA=:C)
        @test_throws ErrorException KF.gemm(A, B; tB=:Z)
    end

    @testset "unknown family rejected" begin
        # A typo'd family symbol used to fall through to the auto branch with
        # :mma enabled — silently giving behaviour the caller never asked for.
        A = AT(rand(Float32, 8, 8)); B = AT(rand(Float32, 8, 8))
        @test_throws ErrorException KF.gemm(A, B; family=:mmma)
        @test_throws ErrorException KF.gemm(A, B; family=:genric)
        @test KF.gemm(A, B; family=:generic) isa AbstractMatrix   # valid ones still work
        @test KF.gemm(A, B; family=nothing)  isa AbstractMatrix
    end

    @testset "@allocate resolves at the simplified arity" begin
        # `@allocate gemm(A, B)` must resolve like the `gemm(A, B)` call it
        # mirrors, not just the explicit `gemm(*, +, A, B)` form.
        A = AT(rand(Float32, 16, 16)); B = AT(rand(Float32, 16, 16))
        @test (KF.@allocate gemm(A, B))       isa KF.KernelBuffer
        @test (KF.@allocate gemm(*, +, A, B)) isa KF.KernelBuffer
    end
end
