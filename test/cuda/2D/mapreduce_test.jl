using Test
using CUDA

# Assuming Luma is loaded and exports mapreduce, mapreduce!, mapreduce1d, mapreduce2d, etc.
# using Luma

@testset "mapreduce" begin

    @testset "1D arrays" begin
        x = CUDA.rand(Float32, 1000)
        x_cpu = Array(x)

        @testset "full reduction" begin
            # dims=nothing
            result = Luma.mapreduce(identity, +, x; to_cpu=true)
            @test result ≈ sum(x_cpu)

            # dims=:
            result = Luma.mapreduce(identity, +, x; dims=:, to_cpu=true)
            @test result ≈ sum(x_cpu)

            # with map function
            result = Luma.mapreduce(abs2, +, x; to_cpu=true)
            @test result ≈ sum(abs2, x_cpu)

            # with post-reduction g
            result = Luma.mapreduce(identity, +, x; g=x -> sqrt(x), to_cpu=true)
            @test result ≈ sqrt(sum(x_cpu))
        end

        @testset "dims=1" begin
            result = Luma.mapreduce(identity, +, x; dims=1, to_cpu=true)
            @test result ≈ sum(x_cpu)

            result = Luma.mapreduce(identity, +, x; dims=(1,), to_cpu=true)
            @test result ≈ sum(x_cpu)
        end

        @testset "invalid dims" begin
            @test_throws ArgumentError Luma.mapreduce(identity, +, x; dims=2)
            @test_throws ArgumentError Luma.mapreduce(identity, +, x; dims=(2,))
            @test_throws ArgumentError Luma.mapreduce(identity, +, x; dims=(1, 2))
        end
    end

    @testset "2D arrays" begin
        A = CUDA.rand(Float32, 100, 50)
        A_cpu = Array(A)

        @testset "full reduction" begin
            result = Luma.mapreduce(identity, +, A; to_cpu=true)
            @test result ≈ sum(A_cpu)

            result = Luma.mapreduce(identity, +, A; dims=(1, 2), to_cpu=true)
            @test result ≈ sum(A_cpu)

            result = Luma.mapreduce(identity, +, A; dims=(2, 1), to_cpu=true)
            @test result ≈ sum(A_cpu)
        end

        @testset "dims=1 (column reduction)" begin
            result = Luma.mapreduce(identity, +, A; dims=1)
            @test Array(result) ≈ vec(sum(A_cpu; dims=1))

            result = Luma.mapreduce(identity, +, A; dims=(1,))
            @test Array(result) ≈ vec(sum(A_cpu; dims=1))

            # with map function
            result = Luma.mapreduce(abs2, +, A; dims=1)
            @test Array(result) ≈ vec(sum(abs2, A_cpu; dims=1))

            # max reduction
            result = Luma.mapreduce(identity, max, A; dims=1)
            @test Array(result) ≈ vec(maximum(A_cpu; dims=1))
        end

        @testset "dims=2 (row reduction)" begin
            result = Luma.mapreduce(identity, +, A; dims=2)
            @test Array(result) ≈ vec(sum(A_cpu; dims=2))

            result = Luma.mapreduce(identity, +, A; dims=(2,))
            @test Array(result) ≈ vec(sum(A_cpu; dims=2))

            # with map function
            result = Luma.mapreduce(abs2, +, A; dims=2)
            @test Array(result) ≈ vec(sum(abs2, A_cpu; dims=2))

            # min reduction
            result = Luma.mapreduce(identity, min, A; dims=2)
            @test Array(result) ≈ vec(minimum(A_cpu; dims=2))
        end

        @testset "invalid dims" begin
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=3)
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=(3,))
        end
    end

    @testset "3D arrays" begin
        A = CUDA.rand(Float32, 20, 30, 40)
        A_cpu = Array(A)

        @testset "full reduction" begin
            result = Luma.mapreduce(identity, +, A; to_cpu=true)
            @test result ≈ sum(A_cpu)
        end

        @testset "dims=1 (reduce first dim)" begin
            result = Luma.mapreduce(identity, +, A; dims=1)
            @test size(result) == (30, 40)
            @test Array(result) ≈ dropdims(sum(A_cpu; dims=1); dims=1)
        end

        @testset "dims=(1,2) (reduce first two dims)" begin
            result = Luma.mapreduce(identity, +, A; dims=(1, 2))
            @test size(result) == (40,)
            @test Array(result) ≈ vec(sum(A_cpu; dims=(1, 2)))
        end

        @testset "dims=3 (reduce last dim)" begin
            result = Luma.mapreduce(identity, +, A; dims=3)
            @test size(result) == (20, 30)
            @test Array(result) ≈ dropdims(sum(A_cpu; dims=3); dims=3)
        end

        @testset "dims=(2,3) (reduce last two dims)" begin
            result = Luma.mapreduce(identity, +, A; dims=(2, 3))
            @test size(result) == (20,)
            @test Array(result) ≈ vec(sum(A_cpu; dims=(2, 3)))
        end

        @testset "invalid dims" begin
            # Non-contiguous
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=(1, 3))

            # Not from start or end
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=2)
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=(2,))

            # Out of range
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=4)
        end
    end

    @testset "4D arrays" begin
        A = CUDA.rand(Float32, 10, 12, 14, 16)
        A_cpu = Array(A)

        @testset "reduce first dims" begin
            # dims=1
            result = Luma.mapreduce(identity, +, A; dims=1)
            @test size(result) == (12, 14, 16)
            @test Array(result) ≈ dropdims(sum(A_cpu; dims=1); dims=1)

            # dims=(1,2)
            result = Luma.mapreduce(identity, +, A; dims=(1, 2))
            @test size(result) == (14, 16)
            @test Array(result) ≈ dropdims(sum(A_cpu; dims=(1, 2)); dims=(1, 2))

            # dims=(1,2,3)
            result = Luma.mapreduce(identity, +, A; dims=(1, 2, 3))
            @test size(result) == (16,)
            @test Array(result) ≈ vec(sum(A_cpu; dims=(1, 2, 3)))
        end

        @testset "reduce last dims" begin
            # dims=4
            result = Luma.mapreduce(identity, +, A; dims=4)
            @test size(result) == (10, 12, 14)
            @test Array(result) ≈ dropdims(sum(A_cpu; dims=4); dims=4)

            # dims=(3,4)
            result = Luma.mapreduce(identity, +, A; dims=(3, 4))
            @test size(result) == (10, 12)
            @test Array(result) ≈ dropdims(sum(A_cpu; dims=(3, 4)); dims=(3, 4))

            # dims=(2,3,4)
            result = Luma.mapreduce(identity, +, A; dims=(2, 3, 4))
            @test size(result) == (10,)
            @test Array(result) ≈ vec(sum(A_cpu; dims=(2, 3, 4)))
        end

        @testset "full reduction" begin
            result = Luma.mapreduce(identity, +, A; dims=(1, 2, 3, 4), to_cpu=true)
            @test result ≈ sum(A_cpu)
        end

        @testset "invalid dims" begin
            # Middle dims only
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=2)
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=3)
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=(2, 3))

            # Non-contiguous
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=(1, 3))
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=(1, 4))
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=(2, 4))
            @test_throws ArgumentError Luma.mapreduce(identity, +, A; dims=(1, 2, 4))
        end
    end

    @testset "different element types" begin
        for T in (Float32, Float64)
            x = CUDA.rand(T, 500)
            x_cpu = Array(x)
            result = Luma.mapreduce(identity, +, x; to_cpu=true)
            @test result ≈ sum(x_cpu)
        end

        for T in (Int32, Int64)
            x = CuArray(rand(1:10, 1000))
            x_cpu = Array(x)
            result = Luma.mapreduce(identity, +, x; to_cpu=true)
            @test result == sum(x_cpu)
        end
    end

    @testset "different reduction operators" begin
        x = CUDA.rand(Float32, 1000)
        x_cpu = Array(x)

        @test Luma.mapreduce(identity, +, x; to_cpu=true) ≈ sum(x_cpu)
        @test Luma.mapreduce(identity, *, x; to_cpu=true) ≈ prod(x_cpu) rtol = 1e-3
        @test Luma.mapreduce(identity, max, x; to_cpu=true) ≈ maximum(x_cpu)
        @test Luma.mapreduce(identity, min, x; to_cpu=true) ≈ minimum(x_cpu)
    end
end

@testset "mapreduce!" begin

    @testset "1D arrays" begin
        x = CUDA.rand(Float32, 1000)
        x_cpu = Array(x)
        dst = CUDA.zeros(Float32, 1)

        Luma.mapreduce!(identity, +, dst, x)
        @test Array(dst)[1] ≈ sum(x_cpu)

        Luma.mapreduce!(identity, +, dst, x; dims=1)
        @test Array(dst)[1] ≈ sum(x_cpu)
    end

    @testset "2D arrays" begin
        A = CUDA.rand(Float32, 100, 50)
        A_cpu = Array(A)

        @testset "dims=1" begin
            dst = CUDA.zeros(Float32, 50)
            Luma.mapreduce!(identity, +, dst, A; dims=1)
            @test Array(dst) ≈ vec(sum(A_cpu; dims=1))
        end

        @testset "dims=2" begin
            dst = CUDA.zeros(Float32, 100)
            Luma.mapreduce!(identity, +, dst, A; dims=2)
            @test Array(dst) ≈ vec(sum(A_cpu; dims=2))
        end

        @testset "full reduction" begin
            dst = CUDA.zeros(Float32, 1)
            Luma.mapreduce!(identity, +, dst, A)
            @test Array(dst)[1] ≈ sum(A_cpu)
        end
    end

    @testset "3D arrays" begin
        A = CUDA.rand(Float32, 20, 30, 40)
        A_cpu = Array(A)

        @testset "dims=1" begin
            dst = CUDA.zeros(Float32, 30, 40)
            Luma.mapreduce!(identity, +, dst, A; dims=1)
            @test Array(dst) ≈ dropdims(sum(A_cpu; dims=1); dims=1)
        end

        @testset "dims=(1,2)" begin
            dst = CUDA.zeros(Float32, 40)
            Luma.mapreduce!(identity, +, dst, A; dims=(1, 2))
            @test Array(dst) ≈ vec(sum(A_cpu; dims=(1, 2)))
        end

        @testset "dims=3" begin
            dst = CUDA.zeros(Float32, 20, 30)
            Luma.mapreduce!(identity, +, dst, A; dims=3)
            @test Array(dst) ≈ dropdims(sum(A_cpu; dims=3); dims=3)
        end

        @testset "dims=(2,3)" begin
            dst = CUDA.zeros(Float32, 20)
            Luma.mapreduce!(identity, +, dst, A; dims=(2, 3))
            @test Array(dst) ≈ vec(sum(A_cpu; dims=(2, 3)))
        end
    end

    @testset "with post-reduction g" begin
        A = CUDA.rand(Float32, 100, 50)
        A_cpu = Array(A)
        n = size(A, 1)

        dst = CUDA.zeros(Float32, 50)
        Luma.mapreduce!(identity, +, dst, A; dims=1, g=x -> x / n)
        expected = vec(sum(A_cpu; dims=1)) ./ n
        @test Array(dst) ≈ expected
    end
end

@testset "dimension validation" begin

    @testset "_normalize_dims" begin
        # Single int
        @test Luma._normalize_dims(1, 3) == (1,)
        @test Luma._normalize_dims(3, 3) == (3,)

        # Negative indexing
        @test Luma._normalize_dims(-1, 3) == (3,)
        @test Luma._normalize_dims(-2, 4) == (3,)

        # Tuple
        @test Luma._normalize_dims((1, 2), 3) == (1, 2)
        @test Luma._normalize_dims((2, 1), 3) == (1, 2)  # sorted
        @test Luma._normalize_dims((3, 2, 1), 3) == (1, 2, 3)  # sorted
    end

    @testset "_validate_dims" begin
        # Valid: contiguous from start
        @test Luma._validate_dims((1,), 3) === nothing
        @test Luma._validate_dims((1, 2), 3) === nothing
        @test Luma._validate_dims((1, 2, 3), 3) === nothing

        # Valid: contiguous from end
        @test Luma._validate_dims((3,), 3) === nothing
        @test Luma._validate_dims((2, 3), 3) === nothing
        @test Luma._validate_dims((1, 2, 3), 3) === nothing

        # Invalid: middle only
        @test_throws ArgumentError Luma._validate_dims((2,), 3)
        @test_throws ArgumentError Luma._validate_dims((2,), 4)
        @test_throws ArgumentError Luma._validate_dims((3,), 4)

        # Invalid: non-contiguous
        @test_throws ArgumentError Luma._validate_dims((1, 3), 3)
        @test_throws ArgumentError Luma._validate_dims((1, 3), 4)
        @test_throws ArgumentError Luma._validate_dims((1, 4), 4)

        # Invalid: out of range
        @test_throws ArgumentError Luma._validate_dims((4,), 3)
        @test_throws ArgumentError Luma._validate_dims((0,), 3)

        # Invalid: duplicates
        @test_throws ArgumentError Luma._validate_dims((1, 1), 3)
    end
end

@testset "edge cases" begin

    @testset "small arrays" begin
        # Single element
        x = CUDA.ones(Float32, 1)
        @test Luma.mapreduce(identity, +, x; to_cpu=true) ≈ 1.0f0

        # Very small 2D
        A = CUDA.ones(Float32, 2, 3)
        @test Array(Luma.mapreduce(identity, +, A; dims=1)) ≈ [2.0f0, 2.0f0, 2.0f0]
        @test Array(Luma.mapreduce(identity, +, A; dims=2)) ≈ [3.0f0, 3.0f0]
    end

    @testset "large arrays" begin
        # Large 1D
        x = CUDA.rand(Float32, 10_000_000)
        x_cpu = Array(x)
        @test Luma.mapreduce(identity, +, x; to_cpu=true) ≈ sum(x_cpu) rtol = 1e-4

        # Large 2D
        A = CUDA.rand(Float32, 5000, 2000)
        A_cpu = Array(A)
        result = Luma.mapreduce(identity, +, A; dims=1)
        @test Array(result) ≈ vec(sum(A_cpu; dims=1)) rtol = 1e-4
    end

    @testset "non-square matrices" begin
        # Tall matrix
        A = CUDA.rand(Float32, 1000, 10)
        A_cpu = Array(A)
        @test Array(Luma.mapreduce(identity, +, A; dims=1)) ≈ vec(sum(A_cpu; dims=1))
        @test Array(Luma.mapreduce(identity, +, A; dims=2)) ≈ vec(sum(A_cpu; dims=2))

        # Wide matrix
        A = CUDA.rand(Float32, 10, 1000)
        A_cpu = Array(A)
        @test Array(Luma.mapreduce(identity, +, A; dims=1)) ≈ vec(sum(A_cpu; dims=1))
        @test Array(Luma.mapreduce(identity, +, A; dims=2)) ≈ vec(sum(A_cpu; dims=2))
    end
end