@testset "KernelForge.scan! basic tests" begin
    #AT = BACKEND_ARRAY_TYPES[backend]
    n = 100_000
    T = Float32
    src_cpu = rand(T, n)
    src = AT(src_cpu)
    dst = AT(zeros(T, n))
    KF.scan!(*, dst, src)
    KA.synchronize(backend)
    @test isapprox(Array(dst), accumulate(*, src_cpu))

    n = 1005
    T = Float64
    src_cpu = rand(T, n)
    src = AT(src_cpu)
    dst = AT(zeros(T, n))
    KF.scan!(identity, *, dst, src)
    KA.synchronize(backend)
    @test isapprox(Array(dst), accumulate(*, src_cpu))
end

@testset "KernelForge.scan! with Quaternions (non-commutative)" begin
    using Quaternions
    #AT = BACKEND_ARRAY_TYPES[backend]
    n = 1_000_000
    T = QuaternionF64
    src_cpu = [QuaternionF64((x ./ sqrt(sum(x .^ 2)))...) for x in eachcol(randn(4, n))]
    src = AT(src_cpu)
    dst = AT(zeros(T, n))
    KF.scan!(identity, *, dst, src)
    KA.synchronize(backend)
    @test isapprox(Array(dst), accumulate(*, src_cpu))
end

@testset "KernelForge.scan! comprehensive tests" begin
    #AT = BACKEND_ARRAY_TYPES[backend]
    test_sizes = [1, 5, 10, 100, 1_000, 1_000_000]
    test_types = [Float64, Int32]
    test_ops = [
        (+, "addition"),
        (min, "minimum")
    ]

    for T in test_types
        for n in test_sizes
            for (op, op_name) in test_ops
                @testset "T=$T, n=$n, op=$op_name" begin
                    for trial in 1:5
                        src_cpu = T <: AbstractFloat ? rand(T, n) : T[rand(1:10) for _ in 1:n]
                        src = AT(src_cpu)
                        dst = AT(zeros(T, n))
                        KF.scan!(op, dst, src)
                        KA.synchronize(backend)
                        expected = accumulate(op, src_cpu)
                        if T <: AbstractFloat
                            @test isapprox(Array(dst), expected)
                        else
                            @test Array(dst) == expected
                        end
                    end
                end
            end
        end
    end
end

@testset "UInt8 scan tests" begin
    #AT = BACKEND_ARRAY_TYPES[backend]
    test_sizes = [10_001]
    test_ops = [(+, "addition"), (max, "maximum"), (min, "minimum")]

    for n in test_sizes
        for (op, op_name) in test_ops
            @testset "n=$n, op=$op_name" begin
                for trial in 1:10
                    src_cpu = rand(UInt8, n)
                    src = AT(src_cpu)
                    dst = AT(zeros(UInt8, n))
                    KF.scan!(op, dst, src)
                    KA.synchronize(backend)
                    @test Array(dst) == accumulate(op, src_cpu)
                end
            end
        end
    end
end

@testset "KernelForge.scan! edge sizes (Float64, +)" begin
    #AT = BACKEND_ARRAY_TYPES[backend]
    edge_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for n in edge_sizes
        @testset "n=$n" begin
            src_cpu = rand(Float64, n)
            src = AT(src_cpu)
            dst = AT(zeros(Float64, n))
            KF.scan!(+, dst, src)
            KA.synchronize(backend)
            @test isapprox(Array(dst), accumulate(+, src_cpu))
        end
    end
end