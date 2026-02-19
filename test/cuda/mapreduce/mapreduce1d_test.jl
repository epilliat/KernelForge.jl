backend = CUDABackend()

# ============================================================================
# mapreduce1d_test.jl
# Tests the mapreduce1d low-level API directly.
# ============================================================================

@testset "mapreduce1d correctness" begin
    AT = BACKEND_ARRAY_TYPES[backend]

    ns = reverse([33, 100, 10_001, 100_000])
    types = [Float32, Float64, Int32]

    @testset "Type $T" for T in types
        @testset "n=$n" for n in ns
            @testset "trial $trial" for trial in 1:5
                src = AT(T.([i for i in 1:n]))
                dst = AT([T(0)])
                tmp = KF.get_allocation(KF.mapreduce1d!, (src,); blocks=400, eltype=T)

                # Warm up
                CUDA.@sync KF.mapreduce1d!(identity, +, dst, src; tmp)

                result = CUDA.@sync KF.mapreduce1d(identity, +, src; tmp, to_cpu=true)
                ref = mapreduce(identity, +, Array(src))

                if T <: AbstractFloat
                    @test isapprox(result, ref; rtol=1e-4, atol=1e-5)
                else
                    @test result == ref
                end
            end
        end
    end
end

# ============================================================================

@testset "mapreduce1d custom struct" begin
    AT = BACKEND_ARRAY_TYPES[backend]

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

    map_func(v::Input6) = Output3(v.a + v.b, v.c + v.d, v.e + v.f)
    reduce_func(a::Output3, b::Output3) = Output3(a.x + b.x, a.y + b.y, a.z + b.z)

    @testset "n=$n" for n in [105, 100_000, 1_000_001]
        @testset "trial $trial" for trial in 1:5
            src = AT([Input6(rand(Float32), rand(Float32), rand(Float32),
                rand(Float32), rand(Float32), rand(Float32)) for _ in 1:n])
            dst = AT([Output3(0f0, 0f0, 0f0)])
            tmp = KF.get_allocation(KF.mapreduce1d!, (src,); blocks=400, eltype=Output3)

            CUDA.@sync KF.mapreduce1d!(map_func, reduce_func, dst, src; tmp)

            result = CUDA.@sync KF.mapreduce1d(map_func, reduce_func, src; tmp, to_cpu=true)
            ref = mapreduce(map_func, reduce_func, Array(src))

            @test isapprox(result.x, ref.x)
            @test isapprox(result.y, ref.y)
            @test isapprox(result.z, ref.z)
        end
    end
end

# ============================================================================

@testset "mapreduce1d tuple inputs (dot product and variants)" begin
    AT = BACKEND_ARRAY_TYPES[backend]
    T = Float64

    @testset "dot product" begin
        n = 10_001
        x, y = AT(rand(T, n)), AT(rand(T, n))
        result = CUDA.@sync KF.mapreduce1d((a, b) -> a * b, +, (x, y); to_cpu=true)
        @test isapprox(result, sum(Array(x) .* Array(y)))
    end

    @testset "weighted sum of squares: sum(w * x^2)" begin
        n = 50_000
        w, x = AT(rand(T, n)), AT(rand(T, n))
        result = CUDA.@sync KF.mapreduce1d((wi, xi) -> wi * xi * xi, +, (w, x); to_cpu=true)
        @test isapprox(result, sum(Array(w) .* Array(x) .^ 2))
    end

    @testset "sum(exp(a - b))" begin
        n = 10_000
        a, b = AT(rand(T, n) .* 2), AT(rand(T, n) .* 2)
        result = CUDA.@sync KF.mapreduce1d((ai, bi) -> exp(ai - bi), +, (a, b); to_cpu=true)
        @test isapprox(result, sum(exp.(Array(a) .- Array(b))))
    end

    @testset "g=sqrt: sqrt(sum(x * y))" begin
        n = 20_000
        x, y = AT(rand(T, n)), AT(rand(T, n))
        result = CUDA.@sync KF.mapreduce1d((a, b) -> a * b, +, (x, y); g=sqrt, to_cpu=true)
        @test isapprox(result, sqrt(sum(Array(x) .* Array(y))))
    end

    @testset "g=sqrt: euclidean distance" begin
        n = 15_000
        a, b = AT(rand(T, n)), AT(rand(T, n))
        result = CUDA.@sync KF.mapreduce1d((ai, bi) -> (ai - bi)^2, +, (a, b); g=sqrt, to_cpu=true)
        @test isapprox(result, sqrt(sum((Array(a) .- Array(b)) .^ 2)))
    end

    @testset "g=s->s/n: normalized dot product" begin
        n = 25_000
        x, y = AT(rand(T, n)), AT(rand(T, n))
        result = CUDA.@sync KF.mapreduce1d((a, b) -> a * b, +, (x, y); g=s -> s / n, to_cpu=true)
        @test isapprox(result, sum(Array(x) .* Array(y)) / n)
    end

    @testset "in-place with pre-allocated tmp" begin
        n = 50_000
        x, y = AT(rand(T, n)), AT(rand(T, n))
        dst = AT([T(0)])
        tmp = KF.get_allocation(KF.mapreduce1d!, (x, y); blocks=100, eltype=T)

        @testset "trial $trial" for trial in 1:3
            copyto!(x, rand(T, n))
            copyto!(y, rand(T, n))
            CUDA.@sync KF.mapreduce1d!((a, b) -> a * b, +, dst, (x, y); tmp)
            result = CUDA.@allowscalar dst[1]
            @test isapprox(result, sum(Array(x) .* Array(y)))
        end
    end

    @testset "three-array: sum(a * b * c)" begin
        n = 12_000
        a, b, c = AT(rand(T, n)), AT(rand(T, n)), AT(rand(T, n))
        result = CUDA.@sync KF.mapreduce1d((ai, bi, ci) -> ai * bi * ci, +, (a, b, c); to_cpu=true)
        @test isapprox(result, sum(Array(a) .* Array(b) .* Array(c)))
    end

    @testset "three-array g=cbrt: cbrt(sum(a * b * c))" begin
        n = 8_000
        a, b, c = AT(rand(T, n)), AT(rand(T, n)), AT(rand(T, n))
        result = CUDA.@sync KF.mapreduce1d((ai, bi, ci) -> ai * bi * ci, +, (a, b, c); g=cbrt, to_cpu=true)
        @test isapprox(result, cbrt(sum(Array(a) .* Array(b) .* Array(c))))
    end
end