@testset "mapreduce2d API" begin
    AT = BACKEND_ARRAY_TYPES[backend]
    n, p = 128, 64
    A = AT(rand(Float32, n, p))

    @testset "dim=1 (column-wise)" begin
        # Allocating
        result = KF.mapreduce2d(identity, +, A, 1)
        @test size(result) == (p,)
        @test result ≈ vec(sum(A; dims=1))
        # In-place
        dst = AT(zeros(Float32, p))
        KF.mapreduce2d!(identity, +, dst, A, 1)
        @test dst ≈ vec(sum(A; dims=1))
        # With g: check value and output eltype
        result_g = KF.mapreduce2d(identity, +, A, 1; g=x -> x^2)
        @test result_g ≈ vec(sum(A; dims=1)) .^ 2
        result_g_f16 = KF.mapreduce2d(identity, +, A, 1; g=x -> Float16(x))
        @test eltype(result_g_f16) == Float16
    end

    @testset "dim=2 (row-wise)" begin
        # Allocating
        result = KF.mapreduce2d(identity, +, A, 2)
        @test size(result) == (n,)
        @test result ≈ vec(sum(A; dims=2))
        # In-place
        dst = AT(zeros(Float32, n))
        KF.mapreduce2d!(identity, +, dst, A, 2)
        @test dst ≈ vec(sum(A; dims=2))
        # With f and g
        result_fg = KF.mapreduce2d(abs2, +, A, 2; g=sqrt)
        @test result_fg ≈ vec(sqrt.(sum(abs2, A; dims=2)))
    end

    @testset "agrees with Base, various sizes" begin
        @testset "($m, $k)" for (m, k) in [(1, 64), (128, 1), (1024, 256), (3, 7)]
            B = AT(rand(Float32, m, k))
            @test KF.mapreduce2d(identity, +, B, 1) ≈ vec(sum(B; dims=1))
            @test KF.mapreduce2d(identity, +, B, 2) ≈ vec(sum(B; dims=2))
        end
    end

    @testset "invalid dim" begin
        @test_throws ArgumentError KF.mapreduce2d(identity, +, A, 0)
        @test_throws ArgumentError KF.mapreduce2d(identity, +, A, 3)
    end
end