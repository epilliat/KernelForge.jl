@testset "argmax/argmin 1D" begin
    # --- argmax (rel = >) ---
    @testset "argmax basic" begin
        x = CuArray(Float32[42])
        @test KF.argmax(>, x) == 1
        x = CuArray(Float32[3, 7])
        @test KF.argmax(>, x) == 2
        x = CuArray(Float32[-5, -1, -3, -2])
        @test KF.argmax(>, x) == 2
        x = CuArray(Float32[100, 1, 2, 3, 4])
        @test KF.argmax(>, x) == 1
        x = CuArray(Float32[1, 2, 3, 4, 100])
        @test KF.argmax(>, x) == 5
    end

    # --- argmin (rel = <) ---
    @testset "argmin basic" begin
        x = CuArray(Float32[-1, -2, -3, -10])
        @test KF.argmax(<, x) == 4
        x = CuArray(Float32[-99, 0, 1, 2])
        @test KF.argmax(<, x) == 1
    end

    # --- Tie-breaking: first index wins ---
    @testset "tie-breaking (first index)" begin
        x = CuArray(Float32[5, 5, 5, 5, 5])
        @test KF.argmax(>, x) == 1
        @test KF.argmax(<, x) == 1
        x = CuArray(Float32[10, 10, 1, 2, 3])
        @test KF.argmax(>, x) == 1
        x = CuArray(Float32[1, 2, 10, 10, 10])
        @test KF.argmax(>, x) == 3
        x = CuArray(Float32[5, 1, 3, 1, 5])
        @test KF.argmax(<, x) == 2
        n = 10_000
        x = CUDA.fill(Float32(7), n)
        @test KF.argmax(>, x) == 1
        @test KF.argmax(<, x) == 1
        n = 100_000
        x = CUDA.fill(Float32(1), n)
        x[10:end] .= Float32(10)
        @test KF.argmax(>, x) == 10
        n = 100_000
        x = CUDA.fill(Float32(99), n)
        x[50:end] .= Float32(-1)
        @test KF.argmax(<, x) == 50
    end

    # --- Custom struct with custom relation ---
    @testset "custom struct and rel" begin
        # Priority queue item: higher priority wins, ties broken by lower cost
        struct PriorityItem
            priority::Int32
            cost::Float32
        end

        # rel(a, b) = true means "a is strictly better than b"
        function better(a::PriorityItem, b::PriorityItem)
            a.priority > b.priority || (a.priority == b.priority && a.cost < b.cost)
        end

        items = CuArray([
            PriorityItem(1, 10.0f0),
            PriorityItem(3, 5.0f0),
            PriorityItem(3, 2.0f0),  # same priority, lower cost → best
            PriorityItem(3, 2.0f0),  # tied on both → first index wins
            PriorityItem(2, 1.0f0),
        ])
        @test KF.argmax(better, items) == 3

        # All identical — first index wins
        items = CuArray(fill(PriorityItem(1, 5.0f0), 100))
        @test KF.argmax(better, items) == 1

        # Best item at the end
        items = CuArray([
            PriorityItem(1, 10.0f0),
            PriorityItem(1, 8.0f0),
            PriorityItem(1, 6.0f0),
            PriorityItem(5, 1.0f0),
        ])
        @test KF.argmax(better, items) == 4

        # Large array, best item buried
        n = 50_000
        items = CuArray(fill(PriorityItem(1, 99.0f0), n))
        CUDA.@allowscalar items[7777] = PriorityItem(10, 0.5f0)
        @test KF.argmax(better, items) == 7777
    end
end