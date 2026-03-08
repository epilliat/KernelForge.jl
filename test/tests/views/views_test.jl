@testset "scan! with views" begin
    #AT = BACKEND_ARRAY_TYPES[backend]
    n = 33

    @testset "Float64" begin
        src_cpu = rand(Float64, n)
        src = AT(src_cpu)

        dst = AT(zeros(Float64, n))
        KF.scan!(+, dst, src)
        KA.synchronize(backend)
        @test isapprox(Array(dst), accumulate(+, src_cpu))

        v = view(src, 2:n)
        v_dst = AT(zeros(Float64, n - 1))
        KF.scan!(+, v_dst, v)
        KA.synchronize(backend)
        @test isapprox(Array(v_dst), accumulate(+, src_cpu[2:n]))

        dst_full = AT(zeros(Float64, n))
        KF.scan!(+, view(dst_full, 3:n), view(src, 3:n))
        KA.synchronize(backend)
        @test isapprox(Array(dst_full)[3:n], accumulate(+, src_cpu[3:n]))
    end

    @testset "Vec3" begin
        src_cpu = [Vec3(rand(Float32, 3)...) for _ in 1:n]
        src = AT(src_cpu)

        dst = AT(fill(zero(Vec3), n))
        KF.scan!(+, dst, src)
        KA.synchronize(backend)
        expected = accumulate(+, src_cpu)
        @test all(isapprox.(Array(dst), expected))

        v = view(src, 2:n)
        v_dst = AT(fill(zero(Vec3), n - 1))
        KF.scan!(+, v_dst, v)
        KA.synchronize(backend)
        @test all(isapprox.(Array(v_dst), accumulate(+, src_cpu[2:n])))
    end
end

@testset "mapreduce with views" begin
    #AT = BACKEND_ARRAY_TYPES[backend]
    n = 33

    @testset "Float64 - 1D" begin
        src_cpu = rand(Float64, n)
        src = AT(src_cpu)

        result = KF.mapreduce(identity, +, src)
        @test isapprox(result, sum(src_cpu))

        v = view(src, 3:n)
        result2 = KF.mapreduce(identity, +, v)
        @test isapprox(result2, sum(src_cpu[3:n]))

        result3 = KF.mapreduce(x -> x^2, +, v)
        @test isapprox(result3, sum(x -> x^2, src_cpu[3:n]))

        result4 = KF.mapreduce(identity, min, v)
        @test isapprox(result4, minimum(src_cpu[3:n]))
    end

    @testset "Float64 - 2D dims" begin
        A_cpu = rand(Float64, 5, 7)
        A = AT(A_cpu)

        result_d1 = KF.mapreduce(identity, +, A; dims=1)
        @test isapprox(Array(result_d1), sum(A_cpu; dims=1))

        result_d2 = KF.mapreduce(identity, +, A; dims=2)
        @test isapprox(Array(result_d2), sum(A_cpu; dims=2))

        vA = view(A, :, 2:6)
        result_v = KF.mapreduce(identity, +, vA; dims=1)
        @test isapprox(Array(result_v), sum(A_cpu[:, 2:6]; dims=1))
    end

    @testset "Vec3" begin
        src_cpu = [Vec3(rand(Float32, 3)...) for _ in 1:n]
        src = AT(src_cpu)

        result = KF.mapreduce(identity, +, src)
        expected = reduce(+, src_cpu)
        @test isapprox(result, expected)

        v = view(src, 2:n)
        result2 = KF.mapreduce(identity, +, v)
        @test isapprox(result2, reduce(+, src_cpu[2:n]))
    end
end

@testset "matvec with views" begin
    #AT = BACKEND_ARRAY_TYPES[backend]
    n = 33

    @testset "Float64" begin
        A_cpu = rand(Float64, 7, n)
        x_cpu = rand(Float64, n)
        A = AT(A_cpu)
        x = AT(x_cpu)

        result = KF.matvec(A, x)
        @test isapprox(Array(result), A_cpu * x_cpu)

        vA = view(A, :, 2:n)
        vx = view(x, 2:n)
        result2 = KF.matvec(vA, vx)
        @test isapprox(Array(result2), A_cpu[:, 2:n] * x_cpu[2:n])

        vA2 = view(A, 2:7, :)
        result3 = KF.matvec(vA2, x)
        @test isapprox(Array(result3), A_cpu[2:7, :] * x_cpu)

        vA3 = view(A, 2:7, 2:n)
        vx3 = view(x, 2:n)
        result4 = KF.matvec(vA3, vx3)
        @test isapprox(Array(result4), A_cpu[2:7, 2:n] * x_cpu[2:n])
    end

    @testset "Float64 - no vector (row sums)" begin
        A_cpu = rand(Float64, 7, n)
        A = AT(A_cpu)

        result = KF.matvec(A, nothing)
        @test isapprox(Array(result), sum(A_cpu; dims=2)[:])
    end
end

@testset "vecmat with views" begin
    #AT = BACKEND_ARRAY_TYPES[backend]
    n = 33

    @testset "Float64" begin
        A_cpu = rand(Float64, n, 7)
        x_cpu = rand(Float64, n)
        A = AT(A_cpu)
        x = AT(x_cpu)

        result = KF.vecmat(x, A)
        @test isapprox(Array(result), x_cpu' * A_cpu |> vec)

        vA = view(A, 2:n, :)
        vx = view(x, 2:n)
        result2 = KF.vecmat(vx, vA)
        @test isapprox(Array(result2), x_cpu[2:n]' * A_cpu[2:n, :] |> vec)

        vA2 = view(A, :, 2:7)
        result3 = KF.vecmat(x, vA2)
        @test isapprox(Array(result3), x_cpu' * A_cpu[:, 2:7] |> vec)
    end

    @testset "Float64 - no vector (column sums)" begin
        A_cpu = rand(Float64, n, 7)
        A = AT(A_cpu)

        result = KF.vecmat(nothing, A)
        @test isapprox(Array(result), sum(A_cpu; dims=1)[:])
    end
end

@testset "vcopy with views" begin
    #AT = BACKEND_ARRAY_TYPES[backend]
    n = 33

    @testset "Float64" begin
        src_cpu = rand(Float64, n)
        src = AT(src_cpu)

        dst = AT(zeros(Float64, n))
        KF.vcopy!(dst, src)
        @test isapprox(Array(dst), src_cpu)

        v_src = view(src, 3:n)
        dst2 = AT(zeros(Float64, n - 2))
        KF.vcopy!(dst2, v_src)
        @test isapprox(Array(dst2), src_cpu[3:n])

        dst_full = AT(zeros(Float64, n))
        KF.vcopy!(view(dst_full, 3:n), view(src, 3:n))
        @test isapprox(Array(dst_full)[3:n], src_cpu[3:n])
    end

    @testset "Vec3" begin
        src_cpu = [Vec3(rand(Float32, 3)...) for _ in 1:n]
        src = AT(src_cpu)

        dst = AT(fill(zero(Vec3), n))
        KF.vcopy!(dst, src)
        @test all(isapprox.(Array(dst), src_cpu))

        dst2 = AT(fill(zero(Vec3), n - 1))
        KF.vcopy!(dst2, view(src, 2:n))
        @test all(isapprox.(Array(dst2), src_cpu[2:n]))
    end
end