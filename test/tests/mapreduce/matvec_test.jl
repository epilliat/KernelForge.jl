function make_test_arrays2(rng, n::Int, p::Int; S=Float32)
    #AT = BACKEND_ARRAY_TYPES[backend]
    x = AT(rand(rng, S, p))
    src = AT(rand(rng, S, n, p))
    dst = AT(zeros(S, n))
    return src, x, dst
end

# ============================================================================
# matvec! tests
# ============================================================================

@testset "matvec!" begin

    @testset "square matrices" begin
        rng = Xoshiro(1)
        for (n, p) in [(100, 100), (256, 256), (512, 512), (1000, 1000)]
            src, x, dst = make_test_arrays2(rng, n, p)
            KF.matvec!(*, +, dst, src, x)
            @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)
        end
    end

    @testset "tall matrices (n >> p)" begin
        rng = Xoshiro(2)
        for (n, p) in [(1024, 10), (4096, 32), (8192, 64), (16384, 100), (65536, 16)]
            src, x, dst = make_test_arrays2(rng, n, p)
            KF.matvec!(*, +, dst, src, x)
            @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)
        end
    end

    @testset "wide matrices (p >> n)" begin
        rng = Xoshiro(3)
        for (n, p) in [(3, 10_000), (16, 10_000), (32, 50_000), (64, 100_000), (128, 100_000), (256, 100_000)]
            src, x, dst = make_test_arrays2(rng, n, p)
            KF.matvec!(*, +, dst, src, x)
            @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)
        end
    end

    @testset "small n" begin
        rng = Xoshiro(4)
        for (n, p) in [(4, 1000), (8, 1000), (16, 1000), (32, 1000), (48, 1000), (63, 1000)]
            src, x, dst = make_test_arrays2(rng, n, p)
            KF.matvec!(*, +, dst, src, x)
            @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)
        end
    end

    @testset "large n" begin
        rng = Xoshiro(5)
        for (n, p) in [(2048, 1000), (4096, 500), (8192, 200), (16384, 100), (32768, 50)]
            src, x, dst = make_test_arrays2(rng, n, p)
            KF.matvec!(*, +, dst, src, x)
            @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)
        end
    end

    @testset "edge cases" begin
        rng = Xoshiro(6)

        src, x, dst = make_test_arrays2(rng, 256, 1)
        KF.matvec!(*, +, dst, src, x)
        @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)

        src, x, dst = make_test_arrays2(rng, 1, 1000)
        KF.matvec!(*, +, dst, src, x)
        @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)

        src, x, dst = make_test_arrays2(rng, 1, 1)
        KF.matvec!(*, +, dst, src, x)
        @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)

        for n in [31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
            src, x, dst = make_test_arrays2(rng, n, 100)
            KF.matvec!(*, +, dst, src, x)
            @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)
        end
    end

    @testset "non-power-of-2" begin
        rng = Xoshiro(7)
        for (n, p) in [(100, 100), (300, 500), (777, 333), (1234, 5678)]
            src, x, dst = make_test_arrays2(rng, n, p)
            KF.matvec!(*, +, dst, src, x)
            @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)
        end
    end

    @testset "stress test" begin
        rng = Xoshiro(8)

        src, x, dst = make_test_arrays2(rng, 1024, 100_000)
        KF.matvec!(*, +, dst, src, x)
        @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)

        src, x, dst = make_test_arrays2(rng, 50_000, 1000)
        KF.matvec!(*, +, dst, src, x)
        @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)
    end

    @testset "simplified API" begin
        rng = Xoshiro(9)
        for (n, p) in [(256, 256), (1024, 100), (64, 10_000)]
            src, x, dst = make_test_arrays2(rng, n, p)
            KF.matvec!(dst, src, x)
            @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)
        end
    end

    @testset "auto allocating matvec" begin
        rng = Xoshiro(10)
        for (n, p) in [(256, 256), (1024, 100), (64, 10_000)]
            src, x, _ = make_test_arrays2(rng, n, p)
            dst = KF.matvec(src, x)
            @test isapprox(Array(dst), Array(vec(src * x)); rtol=1f-3)
        end
    end

    @testset "custom struct with 3 Float32" begin
        f_custom(a::Float32, b::Float32) = Vec3(a * b, a + b, a - b)
        op_custom(v1::Vec3, v2::Vec3) = Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)

        rng = Xoshiro(11)
        n, p = 200, 500
        src_cpu = rand(rng, Float32, n, p)
        x_cpu = rand(rng, Float32, p)
        src = AT(src_cpu)
        x = AT(x_cpu)
        dst = KA.allocate(backend, Vec3, n)

        KF.matvec!(f_custom, op_custom, dst, src, x)

        expected = Vector{Vec3}(undef, n)
        for i in 1:n
            acc = Vec3(0f0, 0f0, 0f0)
            for j in 1:p
                acc = op_custom(acc, f_custom(src_cpu[i, j], x_cpu[j]))
            end
            expected[i] = acc
        end

        dst_cpu = Array(dst)
        @test all(1:n) do i
            isapprox(dst_cpu[i].x, expected[i].x) &&
                isapprox(dst_cpu[i].y, expected[i].y) &&
                isapprox(dst_cpu[i].z, expected[i].z)
        end
    end

end