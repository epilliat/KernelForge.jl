using Test, CUDA, KernelForge

# Custom struct with 3 Float32 fields
struct Vec3
    x::Float32
    y::Float32
    z::Float32
end

Base.:+(a::Vec3, b::Vec3) = Vec3(a.x + b.x, a.y + b.y, a.z + b.z)
Base.:*(a::Float32, b::Vec3) = Vec3(a * b.x, a * b.y, a * b.z)
Base.:*(a::Vec3, b::Float32) = Vec3(a.x * b, a.y * b, a.z * b)
Base.zero(::Type{Vec3}) = Vec3(0f0, 0f0, 0f0)
Base.isapprox(a::Vec3, b::Vec3; kwargs...) = isapprox(a.x, b.x; kwargs...) && isapprox(a.y, b.y; kwargs...) && isapprox(a.z, b.z; kwargs...)

n = 33


src_cpu = rand(Float64, n)
src = CuArray(src_cpu)

# Plain array
dst = CUDA.zeros(Float64, n)
CUDA.@sync KernelForge.scan!(+, dst, src)
@test isapprox(Array(dst), accumulate(+, src_cpu))

# Contiguous view (offset)
v = view(src, 2:n)
v_dst = CUDA.zeros(Float64, n - 1)
CUDA.@sync KernelForge.scan!(+, v_dst, v)
@test isapprox(Array(v_dst), accumulate(+, src_cpu[2:n]))

# Output is also a view
dst_full = CUDA.zeros(Float64, n)
CUDA.@sync KernelForge.scan!(+, view(dst_full, 3:n), view(src, 3:n))
@test isapprox(Array(dst_full)[3:n], accumulate(+, src_cpu[3:n]))


src_cpu = [Vec3(rand(Float32, 3)...) for _ in 1:n]
src = CuArray(src_cpu)

dst = CuArray(fill(zero(Vec3), n))
CUDA.@sync KernelForge.scan!(+, dst, src)
expected = accumulate(+, src_cpu)
result = Array(dst)
@test all(isapprox.(result, expected))

# Contiguous view
v = view(src, 2:n)
v_dst = CuArray(fill(zero(Vec3), n - 1))
CUDA.@sync KernelForge.scan!(+, v_dst, v)
@test all(isapprox.(Array(v_dst), accumulate(+, src_cpu[2:n])))



src_cpu = [Vec3(rand(Float32, 3)...) for _ in 1:n]
src = CuArray(src_cpu)

result = KernelForge.mapreduce(identity, +, src)
expected = reduce(+, src_cpu)
@test isapprox(result, expected)

# Contiguous view
v = view(src, 2:n)
result2 = KernelForge.mapreduce(identity, +, v)
@test isapprox(result2, reduce(+, src_cpu[2:n]))
@testset "scan! with views" begin
    @testset "Float64" begin
        src_cpu = rand(Float64, n)
        src = CuArray(src_cpu)

        # Plain array
        dst = CUDA.zeros(Float64, n)
        CUDA.@sync KernelForge.scan!(+, dst, src)
        @test isapprox(Array(dst), accumulate(+, src_cpu))

        # Contiguous view (offset)
        v = view(src, 2:n)
        v_dst = CUDA.zeros(Float64, n - 1)
        CUDA.@sync KernelForge.scan!(+, v_dst, v)
        @test isapprox(Array(v_dst), accumulate(+, src_cpu[2:n]))

        # Output is also a view
        dst_full = CUDA.zeros(Float64, n)
        CUDA.@sync KernelForge.scan!(+, view(dst_full, 3:n), view(src, 3:n))
        @test isapprox(Array(dst_full)[3:n], accumulate(+, src_cpu[3:n]))
    end

    @testset "Vec3" begin
        src_cpu = [Vec3(rand(Float32, 3)...) for _ in 1:n]
        src = CuArray(src_cpu)

        dst = CuArray(fill(zero(Vec3), n))
        CUDA.@sync KernelForge.scan!(+, dst, src)
        expected = accumulate(+, src_cpu)
        result = Array(dst)
        @test all(isapprox.(result, expected))

        # Contiguous view
        v = view(src, 2:n)
        v_dst = CuArray(fill(zero(Vec3), n - 1))
        CUDA.@sync KernelForge.scan!(+, v_dst, v)
        @test all(isapprox.(Array(v_dst), accumulate(+, src_cpu[2:n])))
    end
end

@testset "mapreduce with views" begin
    @testset "Float64 - 1D" begin
        src_cpu = rand(Float64, n)
        src = CuArray(src_cpu)

        # Plain
        result = KernelForge.mapreduce(identity, +, src)
        @test isapprox(result, sum(src_cpu))

        # Contiguous view
        v = view(src, 3:n)
        result2 = KernelForge.mapreduce(identity, +, v)
        @test isapprox(result2, sum(src_cpu[3:n]))

        # With map function
        result3 = KernelForge.mapreduce(x -> x^2, +, v)
        @test isapprox(result3, sum(x -> x^2, src_cpu[3:n]))

        # Min/max
        result4 = KernelForge.mapreduce(identity, min, v)
        @test isapprox(result4, minimum(src_cpu[3:n]))
    end

    @testset "Float64 - 2D dims" begin
        A_cpu = rand(Float64, 5, 7)
        A = CuArray(A_cpu)

        # Reduce along dim 1
        result_d1 = KernelForge.mapreduce(identity, +, A; dims=1)
        @test isapprox(Array(result_d1), sum(A_cpu; dims=1))

        # Reduce along dim 2
        result_d2 = KernelForge.mapreduce(identity, +, A; dims=2)
        @test isapprox(Array(result_d2), sum(A_cpu; dims=2))

        # View of 2D array
        vA = view(A, :, 2:6)
        result_v = KernelForge.mapreduce(identity, +, vA; dims=1)
        @test isapprox(Array(result_v), sum(A_cpu[:, 2:6]; dims=1))
    end

    @testset "Vec3" begin
        src_cpu = [Vec3(rand(Float32, 3)...) for _ in 1:n]
        src = CuArray(src_cpu)

        result = KernelForge.mapreduce(identity, +, src)
        expected = reduce(+, src_cpu)
        @test isapprox(result, expected)

        # Contiguous view
        v = view(src, 2:n)
        result2 = KernelForge.mapreduce(identity, +, v)
        @test isapprox(result2, reduce(+, src_cpu[2:n]))
    end
end

@testset "matvec with views" begin
    @testset "Float64" begin
        A_cpu = rand(Float64, 5, n)
        x_cpu = rand(Float64, n)
        A = CuArray(A_cpu)
        x = CuArray(x_cpu)

        # Plain
        result = KernelForge.matvec(A, x)
        @test isapprox(Array(result), A_cpu * x_cpu)

        # View of matrix columns
        vA = view(A, :, 2:n)
        vx = view(x, 2:n)
        result2 = KernelForge.matvec(vA, vx)
        @test isapprox(Array(result2), A_cpu[:, 2:n] * x_cpu[2:n])

        # View of matrix rows
        vA2 = view(A, 2:5, :)
        result3 = KernelForge.matvec(vA2, x)
        @test isapprox(Array(result3), A_cpu[2:5, :] * x_cpu)
    end

    @testset "Float64 - no vector (row sums)" begin
        A_cpu = rand(Float64, 5, n)
        A = CuArray(A_cpu)

        result = KernelForge.matvec(A, nothing)
        @test isapprox(Array(result), sum(A_cpu; dims=2)[:])
    end
end


@testset "vecmat with views" begin
    @testset "Float64" begin
        A_cpu = rand(Float64, n, 7)
        x_cpu = rand(Float64, n)
        A = CuArray(A_cpu)
        x = CuArray(x_cpu)

        # Plain
        result = KernelForge.vecmat(x, A)
        @test isapprox(Array(result), x_cpu' * A_cpu |> vec)

        # View of matrix rows
        vA = view(A, 2:n, :)
        vx = view(x, 2:n)
        result2 = KernelForge.vecmat(vx, vA)
        @test isapprox(Array(result2), x_cpu[2:n]' * A_cpu[2:n, :] |> vec)

        # View of matrix columns
        vA2 = view(A, :, 2:7)
        result3 = KernelForge.vecmat(x, vA2)
        @test isapprox(Array(result3), x_cpu' * A_cpu[:, 2:7] |> vec)
    end

    @testset "Float64 - no vector (column sums)" begin
        A_cpu = rand(Float64, n, 7)
        A = CuArray(A_cpu)

        result = KernelForge.vecmat(nothing, A)
        @test isapprox(Array(result), sum(A_cpu; dims=1)[:])
    end
end

@testset "matvec with views" begin
    @testset "Float64" begin
        A_cpu = rand(Float64, 7, n)
        x_cpu = rand(Float64, n)
        A = CuArray(A_cpu)
        x = CuArray(x_cpu)

        # Plain
        result = KernelForge.matvec(A, x)
        @test isapprox(Array(result), A_cpu * x_cpu)

        # View of matrix columns (contiguous)
        vA = view(A, :, 2:n)
        vx = view(x, 2:n)
        result2 = KernelForge.matvec(vA, vx)
        @test isapprox(Array(result2), A_cpu[:, 2:n] * x_cpu[2:n])

        # View of matrix rows (strided)
        vA2 = view(A, 2:7, :)
        result3 = KernelForge.matvec(vA2, x)
        @test isapprox(Array(result3), A_cpu[2:7, :] * x_cpu)

        # Both views
        vA3 = view(A, 2:7, 2:n)
        vx3 = view(x, 2:n)
        result4 = KernelForge.matvec(vA3, vx3)
        @test isapprox(Array(result4), A_cpu[2:7, 2:n] * x_cpu[2:n])
    end

    @testset "Float64 - no vector (row sums)" begin
        A_cpu = rand(Float64, 7, n)
        A = CuArray(A_cpu)

        result = KernelForge.matvec(A, nothing)
        @test isapprox(Array(result), sum(A_cpu; dims=2)[:])
    end
end

@testset "vcopy with views" begin
    @testset "Float64" begin
        src_cpu = rand(Float64, n)
        src = CuArray(src_cpu)

        # Plain
        dst = CUDA.zeros(Float64, n)
        KernelForge.vcopy!(dst, src)
        @test isapprox(Array(dst), src_cpu)

        # Contiguous view source
        v_src = view(src, 3:n)
        dst2 = CUDA.zeros(Float64, n - 2)
        KernelForge.vcopy!(dst2, v_src)
        @test isapprox(Array(dst2), src_cpu[3:n])

        # Both views
        dst_full = CUDA.zeros(Float64, n)
        KernelForge.vcopy!(view(dst_full, 3:n), view(src, 3:n))
        @test isapprox(Array(dst_full)[3:n], src_cpu[3:n])
    end

    @testset "Vec3" begin
        src_cpu = [Vec3(rand(Float32, 3)...) for _ in 1:n]
        src = CuArray(src_cpu)

        dst = CuArray(fill(zero(Vec3), n))
        KernelForge.vcopy!(dst, src)
        @test all(isapprox.(Array(dst), src_cpu))

        # Contiguous view
        dst2 = CuArray(fill(zero(Vec3), n - 1))
        KernelForge.vcopy!(dst2, view(src, 2:n))
        @test all(isapprox.(Array(dst2), src_cpu[2:n]))
    end
end
