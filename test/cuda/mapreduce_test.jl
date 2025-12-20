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
                            blocks=400, eltype=T, FlagType=UInt64)

                        # Warm up
                        CUDA.@sync Luma.mapreduce!(f, op, dst, src, FlagType=UInt64)

                        # Check module ownership
                        if trial == 1  # Only check once per n
                            mb = @which mapreduce(identity, op, Array(src))
                            ml = @which Luma.mapreduce(identity, op, src)
                            @test mb.module == Base
                            @test ml.module == Luma
                        end

                        # Test correctness
                        luma_result = CUDA.@sync Luma.mapreduce(f, op, src, FlagType=UInt64, to_cpu=true)
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
                    blocks=400, eltype=Output3, FlagType=UInt64)

                # Warm up
                CUDA.@sync Luma.mapreduce!(map_func, reduce_func, dst, src, FlagType=UInt64)

                # Check module ownership (once per n)
                if trial == 1
                    mb = @which mapreduce(map_func, reduce_func, Array(src))
                    ml = @which Luma.mapreduce(map_func, reduce_func, src)
                    @test mb.module == Base
                    @test ml.module == Luma
                end

                # Test correctness
                luma_result = CUDA.@sync Luma.mapreduce(map_func, reduce_func, src,
                    FlagType=UInt64, to_cpu=true)
                base_result = mapreduce(map_func, reduce_func, Array(src))

                # Test each coordinate
                @test isapprox(luma_result.x, base_result.x)
                @test isapprox(luma_result.y, base_result.y)
                @test isapprox(luma_result.z, base_result.z)
            end
        end
    end
end