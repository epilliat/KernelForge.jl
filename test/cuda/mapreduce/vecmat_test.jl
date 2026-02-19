# ============================================================================
# Helpers
# ============================================================================
using Test
using Random
using CUDA
using KernelForge

function make_test_arrays(rng, n::Int, p::Int; S=Float32)
    x = CuArray{S}(rand(rng, S, n))
    src = CuArray{S}(rand(rng, S, n, p))
    dst = CuArray{S}(zeros(S, p))
    return x, src, dst
end

# ============================================================================
# vecmat! tests
# ============================================================================

@testset "vecmat!" begin

    @testset "square matrices" begin
        rng = Xoshiro(1)
        for (n, p) in [(100, 100), (256, 256), (512, 512), (1000, 1000)]
            x, src, dst = make_test_arrays(rng, n, p)
            KernelForge.vecmat!(*, +, dst, x, src)
            @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)
        end
    end

    @testset "tall matrices (n >> p)" begin
        rng = Xoshiro(2)
        for (n, p) in [(1024, 10), (4096, 32), (8192, 64), (16384, 100), (65536, 16)]
            x, src, dst = make_test_arrays(rng, n, p)
            KernelForge.vecmat!(*, +, dst, x, src)
            @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)
        end
    end

    @testset "wide matrices (p >> n)" begin
        rng = Xoshiro(3)
        for (n, p) in [(3, 10_000), (16, 10_000), (32, 50_000), (64, 100_000), (128, 100_000), (256, 100_000)]
            x, src, dst = make_test_arrays(rng, n, p)
            KernelForge.vecmat!(*, +, dst, x, src)
            @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)
        end
    end

    @testset "small n" begin
        rng = Xoshiro(4)
        for (n, p) in [(4, 1000), (8, 1000), (16, 1000), (32, 1000), (48, 1000), (63, 1000)]
            x, src, dst = make_test_arrays(rng, n, p)
            KernelForge.vecmat!(*, +, dst, x, src)
            @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)
        end
    end

    @testset "large n" begin
        rng = Xoshiro(5)
        for (n, p) in [(2048, 1000), (4096, 500), (8192, 200), (16384, 100), (32768, 50)]
            x, src, dst = make_test_arrays(rng, n, p)
            KernelForge.vecmat!(*, +, dst, x, src)
            @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)
        end
    end

    @testset "edge cases" begin
        rng = Xoshiro(6)

        # Single column
        x, src, dst = make_test_arrays(rng, 256, 1)
        KernelForge.vecmat!(*, +, dst, x, src)
        @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)

        # Single row
        x, src, dst = make_test_arrays(rng, 1, 1000)
        KernelForge.vecmat!(*, +, dst, x, src)
        @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)

        # Minimal
        x, src, dst = make_test_arrays(rng, 1, 1)
        KernelForge.vecmat!(*, +, dst, x, src)
        @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)

        # Power of 2 boundaries
        for n in [31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
            x, src, dst = make_test_arrays(rng, n, 100)
            KernelForge.vecmat!(*, +, dst, x, src)
            @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)
        end
    end

    @testset "non-power-of-2" begin
        rng = Xoshiro(7)
        for (n, p) in [(100, 100), (300, 500), (777, 333), (1234, 5678)]
            x, src, dst = make_test_arrays(rng, n, p)
            KernelForge.vecmat!(*, +, dst, x, src)
            @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)
        end
    end

    @testset "stress test" begin
        rng = Xoshiro(8)

        x, src, dst = make_test_arrays(rng, 1024, 100_000)
        KernelForge.vecmat!(*, +, dst, x, src)
        @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)

        x, src, dst = make_test_arrays(rng, 50_000, 1000)
        KernelForge.vecmat!(*, +, dst, x, src)
        @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)
    end

    @testset "simplified API" begin
        rng = Xoshiro(9)
        for (n, p) in [(256, 256), (1024, 100), (64, 10_000)]
            x, src, dst = make_test_arrays(rng, n, p)
            KernelForge.vecmat!(dst, x, src)
            @test isapprox(Array(dst), Array(vec(x' * src)); rtol=1f-3)
        end
    end

    @testset "custom struct with 3 Float32" begin
        struct Vec3v
            x::Float32
            y::Float32
            z::Float32
        end

        # f: combine vector element and matrix element into Vec3v
        f_custom(a::Float32, b::Float32) = Vec3v(a * b, a + b, a - b)
        # op: component-wise addition (Vec3v(0,0,0) is the identity)
        op_custom(v1::Vec3v, v2::Vec3v) = Vec3v(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)

        rng = Xoshiro(10)
        n, p = 200, 500
        x_cpu = rand(rng, Float32, n)
        src_cpu = rand(rng, Float32, n, p)
        x = CuArray(x_cpu)
        src = CuArray(src_cpu)
        dst = CuArray{Vec3v}(undef, p)

        KernelForge.vecmat!(f_custom, op_custom, dst, x, src)

        # CPU reference
        expected = Vector{Vec3v}(undef, p)
        for j in 1:p
            acc = Vec3v(0f0, 0f0, 0f0)
            for i in 1:n
                acc = op_custom(acc, f_custom(x_cpu[i], src_cpu[i, j]))
            end
            expected[j] = acc
        end

        dst_cpu = Array(dst)
        @test all(1:p) do j
            isapprox(dst_cpu[j].x, expected[j].x) &&
                isapprox(dst_cpu[j].y, expected[j].y) &&
                isapprox(dst_cpu[j].z, expected[j].z)
        end
    end

end