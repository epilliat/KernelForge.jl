@testset "mapreduce_dims" begin
    AT = BACKEND_ARRAY_TYPES[backend]

    # ----------------------------------------------------------------
    # Basic shapes and dims
    # ----------------------------------------------------------------

    @testset "1D reduce all" begin
        x = AT(Float32[1, 2, 3, 4])
        result = KF.mapreduce_dims(identity, +, x, 1)
        @test size(result) == (1,)
        @test Array(result)[1] ≈ 10f0
    end

    @testset "2D reduce dim 1 (sum along columns)" begin
        x = AT(Float32[1 2 3; 4 5 6])  # shape (2, 3)
        result = KF.mapreduce_dims(identity, +, x, 1)
        @test size(result) == (1, 3)
        @test Array(result) ≈ Float32[5 7 9]
    end

    @testset "2D reduce dim 2 (sum along rows)" begin
        x = AT(Float32[1 2 3; 4 5 6])  # shape (2, 3)
        result = KF.mapreduce_dims(identity, +, x, 2)
        @test size(result) == (2, 1)
        @test Array(result) ≈ Float32[6; 15;;]
    end

    @testset "3D reduce dim 1" begin
        x = AT(ones(Float32, 4, 5, 6))
        result = KF.mapreduce_dims(identity, +, x, 1)
        @test size(result) == (1, 5, 6)
        @test all(Array(result) .≈ 4f0)
    end

    @testset "3D reduce dim 2" begin
        x = AT(ones(Float32, 4, 5, 6))
        result = KF.mapreduce_dims(identity, +, x, 2)
        @test size(result) == (4, 1, 6)
        @test all(Array(result) .≈ 5f0)
    end

    @testset "3D reduce dim 3" begin
        x = AT(ones(Float32, 4, 5, 6))
        result = KF.mapreduce_dims(identity, +, x, 3)
        @test size(result) == (4, 5, 1)
        @test all(Array(result) .≈ 6f0)
    end

    @testset "3D reduce dims (1, 3)" begin
        x = AT(ones(Float32, 4, 5, 6))
        result = KF.mapreduce_dims(identity, +, x, (1, 3))
        @test size(result) == (1, 5, 1)
        @test all(Array(result) .≈ 24f0)
    end

    @testset "3D reduce all dims" begin
        x = AT(ones(Float32, 4, 5, 6))
        result = KF.mapreduce_dims(identity, +, x, (1, 2, 3))
        @test size(result) == (1, 1, 1)
        @test Array(result)[1] ≈ 120f0
    end

    @testset "5D reduce dim 5" begin
        x_cpu = rand(Float32, 3, 4, 5, 6, 7)
        x_gpu = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=5)
        result = Array(KF.mapreduce_dims(identity, +, x_gpu, 5))
        @test size(result) == (3, 4, 5, 6, 1)
        @test result ≈ ref
    end

    @testset "5D reduce dims (2, 4)" begin
        x_cpu = rand(Float32, 3, 4, 5, 6, 7)
        x_gpu = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=(2, 4))
        result = Array(KF.mapreduce_dims(identity, +, x_gpu, (2, 4)))
        @test size(result) == (3, 1, 5, 1, 7)
        @test result ≈ ref
    end

    # ----------------------------------------------------------------
    # Non-trivial f and op
    # ----------------------------------------------------------------

    @testset "sum of squares" begin
        x = AT(Float32[1 2; 3 4])
        result = KF.mapreduce_dims(x -> x^2, +, x, 1)
        @test size(result) == (1, 2)
        @test Array(result) ≈ Float32[10 20]
    end

    @testset "max reduction" begin
        x = AT(Float32[3 1; 2 4; 0 5])  # shape (3, 2)
        result = KF.mapreduce_dims(identity, max, x, 1)
        @test size(result) == (1, 2)
        @test Array(result) ≈ Float32[3 5]
    end

    @testset "min reduction" begin
        x = AT(Float32[3 1; 2 4; 0 5])
        result = KF.mapreduce_dims(identity, min, x, 2)
        @test size(result) == (3, 1)
        @test Array(result) ≈ Float32[1; 2; 0;;]
    end

    # ----------------------------------------------------------------
    # g post-transform
    # ----------------------------------------------------------------

    @testset "g post-transform (mean)" begin
        x = AT(ones(Float32, 4, 6))
        n = Float32(size(x, 1))
        result = KF.mapreduce_dims(identity, +, x, 1; g=s -> s / n)
        @test size(result) == (1, 6)
        @test all(Array(result) .≈ 1f0)
    end

    @testset "g post-transform (negate)" begin
        x = AT(Float32[1 2; 3 4])
        result = KF.mapreduce_dims(identity, +, x, 2; g=s -> -s)
        @test Array(result) ≈ Float32[-3; -7;;]
    end

    # ----------------------------------------------------------------
    # In-place API
    # ----------------------------------------------------------------

    @testset "mapreduce_dims! basic" begin
        x = AT(Float32[1 2 3; 4 5 6])
        dst = AT(zeros(Float32, 1, 3))
        KF.mapreduce_dims!(identity, +, dst, x, 1)
        @test Array(dst) ≈ Float32[5 7 9]
    end

    @testset "mapreduce_dims! does not allocate dst" begin
        x = AT(ones(Float32, 8, 8))
        dst = AT(zeros(Float32, 8, 1))
        KF.mapreduce_dims!(identity, +, dst, x, 2)
        @test all(Array(dst) .≈ 8f0)
    end

    # ----------------------------------------------------------------
    # Agreement with Base.mapreduce
    # ----------------------------------------------------------------

    @testset "agrees with Base.mapreduce, 2D dim 1" begin
        x_cpu = rand(Float32, 16, 32)
        x_gpu = AT(x_cpu)
        ref = mapreduce(x -> x^2, +, x_cpu; dims=1)
        result = Array(KF.mapreduce_dims(x -> x^2, +, x_gpu, 1))
        @test result ≈ ref
    end

    @testset "agrees with Base.mapreduce, 2D dim 2" begin
        x_cpu = rand(Float32, 16, 32)
        x_gpu = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=2)
        result = Array(KF.mapreduce_dims(identity, +, x_gpu, 2))
        @test result ≈ ref
    end

    @testset "agrees with Base.mapreduce, 3D dims (1,3)" begin
        x_cpu = rand(Float32, 8, 5, 6)
        x_gpu = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=(1, 3))
        result = Array(KF.mapreduce_dims(identity, +, x_gpu, (1, 3)))
        @test result ≈ ref
    end

    # ----------------------------------------------------------------
    # Edge cases
    # ----------------------------------------------------------------

    @testset "size-1 dimension" begin
        x = AT(reshape(Float32[1, 2, 3], 1, 3))
        result = KF.mapreduce_dims(identity, +, x, 1)
        @test size(result) == (1, 3)
        @test Array(result) ≈ Float32[1 2 3]
    end

    @testset "single element" begin
        x = AT(Float32[42])
        result = KF.mapreduce_dims(identity, +, x, 1)
        @test Array(result)[1] ≈ 42f0
    end

end  # mapreduce_dims testset