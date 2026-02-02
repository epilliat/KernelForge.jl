@testset "mapreduce correctness" begin
    ns = reverse([33, 100, 10_001, 100_000])
    types = [Float32, Float64, Int32]
    f = identity
    op = +

    for T in types
        @testset "Type $T" begin
            for n in ns
                @testset "n=$n" begin
                    for trial in 1:5
                        src = cu(T.([i for i in 1:n]))
                        dst = cu([T(0)])

                        # Get allocation - H is T since f=identity
                        tmp = Luma.get_allocation(Luma.mapreduce1d!, (src,);
                            blocks=400, eltype=T)

                        # Warm up
                        CUDA.@sync Luma.mapreduce!(f, op, dst, src)

                        # Check module ownership
                        if trial == 1  # Only check once per n
                            mb = @which mapreduce(identity, op, Array(src))
                            ml = @which Luma.mapreduce(identity, op, src)
                            @test mb.module == Base
                            @test ml.module == Luma
                        end

                        # Test correctness
                        luma_result = CUDA.@sync Luma.mapreduce(f, op, src, to_cpu=true)
                        base_result = mapreduce(identity, op, Array(src))
                        if T <: AbstractFloat
                            @test isapprox(luma_result, base_result, rtol=1e-4, atol=1e-5)
                        elseif T <: Integer
                            @test luma_result == base_result
                        end
                    end
                end
            end
        end
    end
end

#%%
@testset "mapreduce with custom struct" begin
    # Define input and output structures
    struct Input6
        a::Float32
        b::Float32
        c::Float32
        d::Float32
        e::Float32
        f::Float32
    end

    struct Output3
        x::Float32
        y::Float32
        z::Float32
    end

    # Map function: sum pairs of fields
    function map_func(input::Input6)
        return Output3(
            input.a + input.b,  # x = a + b
            input.c + input.d,  # y = c + d
            input.e + input.f   # z = e + f
        )
    end

    # Reduce function: coordinate-wise sum
    function reduce_func(acc::Output3, val::Output3)
        return Output3(
            acc.x + val.x,
            acc.y + val.y,
            acc.z + val.z
        )
    end

    #reduce_func(acc::Output3) = acc
    ns = [105, 100_000, 1_000_001]

    for n in ns
        @testset "n=$n" begin
            for trial in 1:5
                # Create input data
                src = cu([Input6(
                    rand(Float32),
                    rand(Float32),
                    rand(Float32),
                    rand(Float32),
                    rand(Float32),
                    rand(Float32)
                ) for i in 1:n])

                dst = cu([Output3(Float32(0), Float32(0), Float32(0))])

                # Get allocation - H is Output3 (result type of map_func)
                tmp = Luma.get_allocation(Luma.mapreduce1d!, (src,);
                    blocks=400, eltype=Output3)

                # Warm up
                CUDA.@sync Luma.mapreduce!(map_func, reduce_func, dst, src)

                # Check module ownership (once per n)
                if trial == 1
                    mb = @which mapreduce(map_func, reduce_func, Array(src))
                    ml = @which Luma.mapreduce(map_func, reduce_func, src)
                    @test mb.module == Base
                    @test ml.module == Luma
                end

                # Test correctness
                luma_result = CUDA.@sync Luma.mapreduce(map_func, reduce_func, src,
                    to_cpu=true)
                base_result = mapreduce(map_func, reduce_func, Array(src))

                # Test each coordinate
                @test isapprox(luma_result.x, base_result.x)
                @test isapprox(luma_result.y, base_result.y)
                @test isapprox(luma_result.z, base_result.z)
            end
        end
    end
end
@testset "mapreduce dot product (tuple inputs)" begin
    T = Float64

    @testset "Standard dot product" begin
        n = 10_001
        x = cu(rand(T, n))
        y = cu(rand(T, n))

        luma_result = CUDA.@sync Luma.mapreduce1d(
            (a, b) -> a * b, +, (x, y); to_cpu=true
        )

        base_result = sum(Array(x) .* Array(y))
        @test isapprox(luma_result, base_result)
    end

    @testset "Weighted sum of squares: sum(w * x^2)" begin
        n = 50_000
        w = cu(rand(T, n))
        x = cu(rand(T, n))

        luma_result = CUDA.@sync Luma.mapreduce1d(
            (wi, xi) -> wi * xi * xi, +, (w, x); to_cpu=true
        )

        base_result = sum(Array(w) .* Array(x) .^ 2)
        @test isapprox(luma_result, base_result)
    end

    @testset "Custom binary function: sum(exp(a - b))" begin
        n = 10_000
        a = cu(rand(T, n) .* 2.0)
        b = cu(rand(T, n) .* 2.0)

        luma_result = CUDA.@sync Luma.mapreduce1d(
            (ai, bi) -> exp(ai - bi), +, (a, b); to_cpu=true
        )

        base_result = sum(exp.(Array(a) .- Array(b)))
        @test isapprox(luma_result, base_result)
    end

    @testset "Dot product with output function g: sqrt(sum(x * y))" begin
        n = 20_000
        x = cu(rand(T, n))
        y = cu(rand(T, n))

        luma_result = CUDA.@sync Luma.mapreduce1d(
            (a, b) -> a * b, +, (x, y);
            g=sqrt, to_cpu=true
        )

        base_result = sqrt(sum(Array(x) .* Array(y)))
        @test isapprox(luma_result, base_result)
    end

    @testset "Euclidean distance: sqrt(sum((a - b)^2))" begin
        n = 15_000
        a = cu(rand(T, n))
        b = cu(rand(T, n))

        luma_result = CUDA.@sync Luma.mapreduce1d(
            (ai, bi) -> (ai - bi)^2, +, (a, b);
            g=sqrt, to_cpu=true
        )

        base_result = sqrt(sum((Array(a) .- Array(b)) .^ 2))
        @test isapprox(luma_result, base_result)
    end

    @testset "Normalized dot product: sum(x * y) / n" begin
        n = 25_000
        x = cu(rand(T, n))
        y = cu(rand(T, n))

        luma_result = CUDA.@sync Luma.mapreduce1d(
            (a, b) -> a * b, +, (x, y);
            g=s -> s / n,
            FlagType=UInt8, to_cpu=true
        )

        base_result = sum(Array(x) .* Array(y)) / n
        @test isapprox(luma_result, base_result)
    end

    @testset "In-place dot product with pre-allocated tmp" begin
        n = 50_000
        x = cu(rand(T, n))
        y = cu(rand(T, n))
        dst = cu([T(0)])

        # H = promote_op((a,b) -> a*b, T, T) = T
        tmp = Luma.get_allocation(Luma.mapreduce1d!, (x, y);
            blocks=100, eltype=T)

        for trial in 1:3
            copyto!(x, rand(T, n))
            copyto!(y, rand(T, n))

            CUDA.@sync Luma.mapreduce1d!(
                (a, b) -> a * b, +, dst, (x, y);
                tmp=tmp
            )

            luma_result = CUDA.@allowscalar dst[1]
            base_result = sum(Array(x) .* Array(y))
            @test isapprox(luma_result, base_result)
        end
    end

    @testset "Three-array reduction: sum(a * b * c)" begin
        n = 12_000
        a = cu(rand(T, n))
        b = cu(rand(T, n))
        c = cu(rand(T, n))

        luma_result = CUDA.@sync Luma.mapreduce1d(
            (ai, bi, ci) -> ai * bi * ci, +, (a, b, c); to_cpu=true
        )

        base_result = sum(Array(a) .* Array(b) .* Array(c))
        @test isapprox(luma_result, base_result)
    end

    @testset "Three-array with g: cbrt(sum(a * b * c))" begin
        n = 8_000
        a = cu(rand(T, n))
        b = cu(rand(T, n))
        c = cu(rand(T, n))

        luma_result = CUDA.@sync Luma.mapreduce1d(
            (ai, bi, ci) -> ai * bi * ci, +, (a, b, c);
            g=cbrt, to_cpu=true
        )

        base_result = cbrt(sum(Array(a) .* Array(b) .* Array(c)))
        @test isapprox(luma_result, base_result)
    end
end