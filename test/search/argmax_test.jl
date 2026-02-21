@testset "argmax/argmin 1D" begin
    #AT = BACKEND_ARRAY_TYPES[backend]

    @testset "argmax basic" begin
        x = AT(Float32[42])
        @test KF.argmax(>, x) == 1
        x = AT(Float32[3, 7])
        @test KF.argmax(>, x) == 2
        x = AT(Float32[-5, -1, -3, -2])
        @test KF.argmax(>, x) == 2
        x = AT(Float32[100, 1, 2, 3, 4])
        @test KF.argmax(>, x) == 1
        x = AT(Float32[1, 2, 3, 4, 100])
        @test KF.argmax(>, x) == 5
    end

    @testset "argmin basic" begin
        x = AT(Float32[-1, -2, -3, -10])
        @test KF.argmax(<, x) == 4
        x = AT(Float32[-99, 0, 1, 2])
        @test KF.argmax(<, x) == 1
    end

    @testset "tie-breaking (first index)" begin
        x = AT(Float32[5, 5, 5, 5, 5])
        @test KF.argmax(>, x) == 1
        @test KF.argmax(<, x) == 1
        x = AT(Float32[10, 10, 1, 2, 3])
        @test KF.argmax(>, x) == 1
        x = AT(Float32[1, 2, 10, 10, 10])
        @test KF.argmax(>, x) == 3
        x = AT(Float32[5, 1, 3, 1, 5])
        @test KF.argmax(<, x) == 2

        n = 10_000
        x = KA.allocate(backend, Float32, n)
        fill!(x, Float32(7))
        @test KF.argmax(>, x) == 1
        @test KF.argmax(<, x) == 1

        n = 100_000
        x = KA.allocate(backend, Float32, n)
        fill!(x, Float32(1))
        x[10:end] .= Float32(10)
        @test KF.argmax(>, x) == 10

        n = 100_000
        x = KA.allocate(backend, Float32, n)
        fill!(x, Float32(99))
        x[50:end] .= Float32(-1)
        @test KF.argmax(<, x) == 50
    end

    @testset "custom struct and rel" begin
        items = AT([
            PriorityItem(1, 10.0f0),
            PriorityItem(3, 5.0f0),
            PriorityItem(3, 2.0f0),
            PriorityItem(3, 2.0f0),
            PriorityItem(2, 1.0f0),
        ])
        @test KF.argmax(better, items) == 3

        items = AT(fill(PriorityItem(1, 5.0f0), 100))
        @test KF.argmax(better, items) == 1

        items = AT([
            PriorityItem(1, 10.0f0),
            PriorityItem(1, 8.0f0),
            PriorityItem(1, 6.0f0),
            PriorityItem(5, 1.0f0),
        ])
        @test KF.argmax(better, items) == 4

        n = 50_000
        items_cpu = fill(PriorityItem(1, 99.0f0), n)
        items_cpu[7777] = PriorityItem(10, 0.5f0)
        items = AT(items_cpu)
        @test KF.argmax(better, items) == 7777
    end
end