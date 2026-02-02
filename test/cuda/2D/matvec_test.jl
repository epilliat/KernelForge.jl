# ============================================================================
# Helpers
# ============================================================================
using Test
using Random
using CUDA
import Luma

function make_test_arrays2(n::Int, p::Int; seed::Int=42, S=Float32)
    Random.seed!(seed)
    x = CuArray{S}(rand(S, p)) # x has column size of A
    src = CuArray{S}(rand(S, n, p))
    dst = CuArray{S}(zeros(S, n))
    return src, x, dst
end

# ============================================================================
# matvec! tests
# ============================================================================

@testset "matvec!" begin

    @testset "square matrices" begin
        for (n, p) in [(100, 100), (256, 256), (512, 512), (1000, 1000)]
            src, x, dst = make_test_arrays2(n, p)
            Luma.matvec!(*, +, dst, src, x)
            expected = vec(src * x)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "tall matrices (n >> p)" begin
        for (n, p) in [(1024, 10), (4096, 32), (8192, 64), (16384, 100), (65536, 16)]
            src, x, dst = make_test_arrays2(n, p)
            Luma.matvec!(*, +, dst, src, x)
            expected = vec(src * x)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "wide matrices (p >> n)" begin
        for (n, p) in [(3, 10_000), (16, 10_000), (32, 50_000), (64, 100_000), (128, 100_000), (256, 100_000)]
            src, x, dst = make_test_arrays2(n, p)
            Luma.matvec!(*, +, dst, src, x)
            expected = vec(src * x)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "small n" begin
        for (n, p) in [(4, 1000), (8, 1000), (16, 1000), (32, 1000), (48, 1000), (63, 1000)]
            src, x, dst = make_test_arrays2(n, p)
            Luma.matvec!(*, +, dst, src, x)
            expected = vec(src * x)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "large n" begin
        for (n, p) in [(2048, 1000), (4096, 500), (8192, 200), (16384, 100), (32768, 50)]
            src, x, dst = make_test_arrays2(n, p)
            Luma.matvec!(*, +, dst, src, x)
            expected = vec(src * x)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "edge cases" begin
        # Single column
        src, x, dst = make_test_arrays2(256, 1)
        Luma.matvec!(*, +, dst, src, x)
        @test isapprox(Array(dst), Array(vec(src * x)))

        # Single row
        src, x, dst = make_test_arrays2(1, 1000)
        Luma.matvec!(*, +, dst, src, x)
        @test isapprox(Array(dst), Array(vec(src * x)))

        # Minimal
        src, x, dst = make_test_arrays2(1, 1)
        Luma.matvec!(*, +, dst, src, x)
        @test isapprox(Array(dst), Array(vec(src * x)))

        # Power of 2 boundaries
        for n in [31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
            src, x, dst = make_test_arrays2(n, 100)
            Luma.matvec!(*, +, dst, src, x)
            @test isapprox(Array(dst), Array(vec(src * x)))
        end
    end

    @testset "non-power-of-2" begin
        for (n, p) in [(100, 100), (300, 500), (777, 333), (1234, 5678)]
            src, x, dst = make_test_arrays2(n, p)
            Luma.matvec!(*, +, dst, src, x)
            expected = vec(src * x)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "stress test" begin
        src, x, dst = make_test_arrays2(1024, 100_000)
        Luma.matvec!(*, +, dst, src, x)
        @test isapprox(Array(dst), Array(vec(src * x)))

        src, x, dst = make_test_arrays2(50_000, 1000)
        Luma.matvec!(*, +, dst, src, x)
        @test isapprox(Array(dst), Array(vec(src * x)))
    end

    @testset "simplified API" begin
        for (n, p) in [(256, 256), (1024, 100), (64, 10_000)]
            src, x, dst = make_test_arrays2(n, p)
            Luma.matvec!(dst, src, x)
            expected = vec(src * x)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "auto allocating matvec function" begin
        for (n, p) in [(256, 256), (1024, 100), (64, 10_000)]
            src, x, _ = make_test_arrays2(n, p)
            dst = Luma.matvec(src, x)
            expected = vec(src * x)
            @test isapprox(Array(dst), Array(expected))
        end
    end
end

@testset "custom struct with 3 Float32" begin
    struct Vec3
        x::Float32
        y::Float32
        z::Float32
    end

    # Custom f: combine matrix element and vector element into Vec3
    f_custom(a::Float32, b::Float32) = Vec3(a * b, a + b, a - b)

    # Custom op: component-wise addition
    op_custom(v1::Vec3, v2::Vec3) = Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)

    n, p = 200, 500
    Random.seed!(123)
    src = CuArray{Float32}(rand(Float32, n, p))
    x = CuArray{Float32}(rand(Float32, p))
    dst = CuArray{Vec3}(undef, n)

    Luma.matvec!(f_custom, op_custom, dst, src, x)

    # Compute expected on CPU
    src_cpu = Array(src)
    x_cpu = Array(x)
    expected = Vector{Vec3}(undef, n)
    for i in 1:n
        acc = Vec3(0f0, 0f0, 0f0)
        for j in 1:p
            acc = op_custom(acc, f_custom(src_cpu[i, j], x_cpu[j]))
        end
        expected[i] = acc
    end

    dst_cpu = Array(dst)
    @test all(i -> isapprox(dst_cpu[i].x, expected[i].x) &&
                       isapprox(dst_cpu[i].y, expected[i].y) &&
                       isapprox(dst_cpu[i].z, expected[i].z), 1:n)
end