
backend = CUDABackend()

@testset "mapreduce2d API" begin
    n, p = 128, 64
    A = CUDA.rand(Float32, n, p)

    @testset "dim=1 (column-wise)" begin
        # Allocating
        result = KernelForge.mapreduce2d(identity, +, A, 1)
        @test size(result) == (p,)
        @test result ≈ vec(sum(A; dims=1))

        # In-place
        dst = CUDA.zeros(Float32, p)
        KernelForge.mapreduce2d!(identity, +, dst, A, 1)
        @test dst ≈ vec(sum(A; dims=1))

        # With g: check value and output eltype
        result_g = KernelForge.mapreduce2d(identity, +, A, 1; g=x -> x^2)
        @test result_g ≈ vec(sum(A; dims=1)) .^ 2
        result_g_f16 = KernelForge.mapreduce2d(identity, +, A, 1; g=x -> Float16(x))
        @test eltype(result_g_f16) == Float16
    end

    @testset "dim=2 (row-wise)" begin
        # Allocating
        result = KernelForge.mapreduce2d(identity, +, A, 2)
        @test size(result) == (n,)
        @test result ≈ vec(sum(A; dims=2))

        # In-place
        dst = CUDA.zeros(Float32, n)
        KernelForge.mapreduce2d!(identity, +, dst, A, 2)
        @test dst ≈ vec(sum(A; dims=2))

        # With f and g
        result_fg = KernelForge.mapreduce2d(abs2, +, A, 2; g=sqrt)
        @test result_fg ≈ vec(sqrt.(sum(abs2, A; dims=2)))
    end

    @testset "agrees with Base, various sizes" begin
        @testset "($m, $k)" for (m, k) in [(1, 64), (128, 1), (1024, 256), (3, 7)]
            B = CUDA.rand(Float32, m, k)
            @test KernelForge.mapreduce2d(identity, +, B, 1) ≈ vec(sum(B; dims=1))
            @test KernelForge.mapreduce2d(identity, +, B, 2) ≈ vec(sum(B; dims=2))
        end
    end

    @testset "invalid dim" begin
        @test_throws ArgumentError KernelForge.mapreduce2d(identity, +, A, 0)
        @test_throws ArgumentError KernelForge.mapreduce2d(identity, +, A, 3)
    end
end