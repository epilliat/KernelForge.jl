# ============================================================================
# Helpers
# ============================================================================
using Test
using Random
using CUDA
using Luma

function make_test_arrays(n::Int, p::Int; seed::Int=42, S=Float32)
    Random.seed!(seed)
    x = CuArray{S}(rand(S, n))
    src = CuArray{S}(rand(S, n, p))
    dst = CuArray{S}(zeros(S, p))
    return x, src, dst
end

# ============================================================================
# vecmat! tests
# ============================================================================

@testset "vecmat!" begin

    @testset "square matrices" begin
        for (n, p) in [(100, 100), (256, 256), (512, 512), (1000, 1000)]
            x, src, dst = make_test_arrays(n, p)
            Luma.vecmat!(*, +, dst, x, src)
            expected = vec(x' * src)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "tall matrices (n >> p)" begin
        for (n, p) in [(1024, 10), (4096, 32), (8192, 64), (16384, 100), (65536, 16)]
            x, src, dst = make_test_arrays(n, p)
            Luma.vecmat!(*, +, dst, x, src)
            expected = vec(x' * src)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "wide matrices (p >> n)" begin
        for (n, p) in [(3, 10_000), (16, 10_000), (32, 50_000), (64, 100_000), (128, 100_000), (256, 100_000)]
            x, src, dst = make_test_arrays(n, p)
            Luma.vecmat!(*, +, dst, x, src)
            expected = vec(x' * src)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "small n" begin
        for (n, p) in [(4, 1000), (8, 1000), (16, 1000), (32, 1000), (48, 1000), (63, 1000)]
            x, src, dst = make_test_arrays(n, p)
            Luma.vecmat!(*, +, dst, x, src)
            expected = vec(x' * src)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "large n" begin
        for (n, p) in [(2048, 1000), (4096, 500), (8192, 200), (16384, 100), (32768, 50)]
            x, src, dst = make_test_arrays(n, p)
            Luma.vecmat!(*, +, dst, x, src)
            expected = vec(x' * src)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "edge cases" begin
        # Single column
        x, src, dst = make_test_arrays(256, 1)
        Luma.vecmat!(*, +, dst, x, src)
        @test isapprox(Array(dst), Array(vec(x' * src)))

        # Single row
        x, src, dst = make_test_arrays(1, 1000)
        Luma.vecmat!(*, +, dst, x, src)
        @test isapprox(Array(dst), Array(vec(x' * src)))

        # Minimal
        x, src, dst = make_test_arrays(1, 1)
        Luma.vecmat!(*, +, dst, x, src)
        @test isapprox(Array(dst), Array(vec(x' * src)))

        # Power of 2 boundaries
        for n in [31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
            x, src, dst = make_test_arrays(n, 100)
            Luma.vecmat!(*, +, dst, x, src)
            @test isapprox(Array(dst), Array(vec(x' * src)))
        end
    end

    @testset "non-power-of-2" begin
        for (n, p) in [(100, 100), (300, 500), (777, 333), (1234, 5678)]
            x, src, dst = make_test_arrays(n, p)
            Luma.vecmat!(*, +, dst, x, src)
            expected = vec(x' * src)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "stress test" begin
        x, src, dst = make_test_arrays(1024, 100_000)
        Luma.vecmat!(*, +, dst, x, src)
        @test isapprox(Array(dst), Array(vec(x' * src)))

        x, src, dst = make_test_arrays(50_000, 1000)
        Luma.vecmat!(*, +, dst, x, src)
        @test isapprox(Array(dst), Array(vec(x' * src)))
    end

    @testset "simplified API" begin
        for (n, p) in [(256, 256), (1024, 100), (64, 10_000)]
            x, src, dst = make_test_arrays(n, p)
            Luma.vecmat!(dst, x, src)
            expected = vec(x' * src)
            @test isapprox(Array(dst), Array(expected))
        end
    end

    @testset "custom struct with 3 Float32" begin
        struct Vec3v
            x::Float32
            y::Float32
            z::Float32
        end

        # Custom f: combine vector element and matrix element into Vec3v
        f_custom(a::Float32, b::Float32) = Vec3v(a * b, a + b, a - b)

        # Custom op: component-wise addition
        op_custom(v1::Vec3v, v2::Vec3v) = Vec3v(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)

        n, p = 200, 500
        Random.seed!(123)
        x = CuArray{Float32}(rand(Float32, n))
        src = CuArray{Float32}(rand(Float32, n, p))
        dst = CuArray{Vec3v}(undef, p)

        Luma.vecmat!(f_custom, op_custom, dst, x, src)

        # Compute expected on CPU
        x_cpu = Array(x)
        src_cpu = Array(src)
        expected = Vector{Vec3v}(undef, p)
        for j in 1:p
            acc = Vec3v(0f0, 0f0, 0f0)
            for i in 1:n
                acc = op_custom(acc, f_custom(x_cpu[i], src_cpu[i, j]))
            end
            expected[j] = acc
        end

        dst_cpu = Array(dst)
        @test all(j -> isapprox(dst_cpu[j].x, expected[j].x) &&
                           isapprox(dst_cpu[j].y, expected[j].y) &&
                           isapprox(dst_cpu[j].z, expected[j].z), 1:p)
    end

end