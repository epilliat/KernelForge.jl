const BACKEND_ARRAY_TYPES = Dict(
    CUDABackend() => CuArray,
)

backend = CUDABackend()

# ============================================================================
# mapreduce_test.jl
# Tests the unified mapreduce API (routing logic).
# Low-level kernel correctness is tested in mapreduce1d_test.jl,
# mapreduce2d_test.jl, and mapreduce_dims_test.jl.
# ============================================================================

@testset "mapreduce unified API" begin
    AT = BACKEND_ARRAY_TYPES[backend]

    # ----------------------------------------------------------------
    # Full reduction routes to mapreduce1d
    # ----------------------------------------------------------------

    @testset "full reduction (dims=nothing)" begin
        x_cpu = rand(Float32, 4, 5, 6)
        x = AT(x_cpu)
        @test KF.mapreduce(identity, +, x; to_cpu=true) ≈ sum(x_cpu)
    end

    @testset "full reduction (dims=:)" begin
        x_cpu = rand(Float32, 4, 5, 6)
        x = AT(x_cpu)
        @test KF.mapreduce(identity, +, x; dims=:, to_cpu=true) ≈ sum(x_cpu)
    end

    @testset "full reduction (all dims explicit)" begin
        x_cpu = rand(Float32, 4, 5, 6)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=(1, 2, 3))
        @test Array(KF.mapreduce(identity, +, x; dims=(1, 2, 3))) ≈ ref
    end

    # ----------------------------------------------------------------
    # Leading dims route to mapreduce2d on dim 1
    # ----------------------------------------------------------------

    @testset "leading dim (3D, dims=1)" begin
        x_cpu = rand(Float32, 8, 5, 6)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=1)
        @test Array(KF.mapreduce(identity, +, x; dims=1)) ≈ ref
    end

    @testset "leading dims (3D, dims=(1,2))" begin
        x_cpu = rand(Float32, 8, 5, 6)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=(1, 2))
        @test Array(KF.mapreduce(identity, +, x; dims=(1, 2))) ≈ ref
    end

    @testset "leading dims (4D, dims=(1,2,3))" begin
        x_cpu = rand(Float32, 4, 5, 6, 7)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=(1, 2, 3))
        @test Array(KF.mapreduce(identity, +, x; dims=(1, 2, 3))) ≈ ref
    end

    # ----------------------------------------------------------------
    # Trailing dims route to mapreduce2d on dim 2
    # ----------------------------------------------------------------

    @testset "trailing dim (3D, dims=3)" begin
        x_cpu = rand(Float32, 8, 5, 6)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=3)
        @test Array(KF.mapreduce(identity, +, x; dims=3)) ≈ ref
    end

    @testset "trailing dims (3D, dims=(2,3))" begin
        x_cpu = rand(Float32, 8, 5, 6)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=(2, 3))
        @test Array(KF.mapreduce(identity, +, x; dims=(2, 3))) ≈ ref
    end

    @testset "trailing dims (4D, dims=(2,3,4))" begin
        x_cpu = rand(Float32, 4, 5, 6, 7)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=(2, 3, 4))
        @test Array(KF.mapreduce(identity, +, x; dims=(2, 3, 4))) ≈ ref
    end

    # ----------------------------------------------------------------
    # Two contiguous blocks route to two mapreduce2d passes
    # ----------------------------------------------------------------

    @testset "two blocks (4D, dims=(1,3,4))" begin
        x_cpu = rand(Float32, 4, 5, 6, 7)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=(1, 3, 4))
        @test Array(KF.mapreduce(identity, +, x; dims=(1, 3, 4))) ≈ ref
    end

    @testset "two blocks (5D, dims=(1,2,4,5))" begin
        x_cpu = rand(Float32, 3, 4, 5, 6, 7)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=(1, 2, 4, 5))
        @test Array(KF.mapreduce(identity, +, x; dims=(1, 2, 4, 5))) ≈ ref
    end

    # ----------------------------------------------------------------
    # General fallback routes to mapreduce_dims
    # ----------------------------------------------------------------

    @testset "general fallback (4D, dims=(2,3))" begin
        x_cpu = rand(Float32, 4, 5, 6, 7)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=(2, 3))
        @test Array(KF.mapreduce(identity, +, x; dims=(2, 3))) ≈ ref
    end

    @testset "general fallback (5D, dims=(2,4))" begin
        x_cpu = rand(Float32, 3, 4, 5, 6, 7)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=(2, 4))
        @test Array(KF.mapreduce(identity, +, x; dims=(2, 4))) ≈ ref
    end

    # ----------------------------------------------------------------
    # g post-transform propagates correctly through all paths
    # ----------------------------------------------------------------

    @testset "g with full reduction" begin
        x_cpu = rand(Float32, 100)
        x = AT(x_cpu)
        ref = sqrt(sum(x_cpu))
        @test KF.mapreduce(identity, +, x; g=sqrt, to_cpu=true) ≈ ref
    end

    @testset "g with leading dims" begin
        x_cpu = rand(Float32, 8, 6)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=1) ./ 8
        @test Array(KF.mapreduce(identity, +, x; dims=1, g=s -> s / 8f0)) ≈ ref
    end

    @testset "g with trailing dims" begin
        x_cpu = rand(Float32, 8, 6)
        x = AT(x_cpu)
        ref = mapreduce(identity, +, x_cpu; dims=2) ./ 6
        @test Array(KF.mapreduce(identity, +, x; dims=2, g=s -> s / 6f0)) ≈ ref
    end

    # ----------------------------------------------------------------
    # In-place API
    # ----------------------------------------------------------------

    @testset "mapreduce! full reduction" begin
        x_cpu = rand(Float32, 100)
        x = AT(x_cpu)
        dst = AT(zeros(Float32, 1))
        KF.mapreduce!(identity, +, dst, x)
        @test Array(dst)[1] ≈ sum(x_cpu)
    end

    @testset "mapreduce! leading dims" begin
        x_cpu = rand(Float32, 8, 6)
        x = AT(x_cpu)
        dst = AT(zeros(Float32, 1, 6))
        KF.mapreduce!(identity, +, dst, x; dims=1)
        ref = mapreduce(identity, +, x_cpu; dims=1)
        @test Array(dst) ≈ ref
    end

    @testset "mapreduce! trailing dims" begin
        x_cpu = rand(Float32, 8, 6)
        x = AT(x_cpu)
        dst = AT(zeros(Float32, 8, 1))
        KF.mapreduce!(identity, +, dst, x; dims=2)
        ref = mapreduce(identity, +, x_cpu; dims=2)
        @test Array(dst) ≈ ref
    end

    # ----------------------------------------------------------------
    # Error handling
    # ----------------------------------------------------------------

    @testset "out of range dim throws" begin
        x = AT(rand(Float32, 4, 5))
        @test_throws ArgumentError KF.mapreduce(identity, +, x; dims=3)
    end

    @testset "duplicate dims throws" begin
        x = AT(rand(Float32, 4, 5, 6))
        @test_throws ArgumentError KF.mapreduce(identity, +, x; dims=(1, 1))
    end

    @testset "multi-array with dims throws" begin
        x = AT(rand(Float32, 10))
        y = AT(rand(Float32, 10))
        @test_throws ArgumentError KF.mapreduce((a, b) -> a * b, +, (x, y); dims=1)
    end

end