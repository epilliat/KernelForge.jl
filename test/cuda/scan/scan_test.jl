using Test, CUDA, KernelForge

@testset "KernelForge.scan! basic tests" begin
    # Basic multiplication test
    n = 100_000
    T = Float32
    src_cpu = rand(T, n)
    src = CuArray(src_cpu)
    dst = CUDA.zeros(T, n)
    CUDA.@sync KernelForge.scan!(*, dst, src)
    @test isapprox(dst, CuArray(accumulate(*, src_cpu)))

    # Small array test
    n = 1005
    T = Float64
    src_cpu = rand(T, n)
    src = CuArray(src_cpu)
    dst = CUDA.zeros(T, n)
    CUDA.@sync KernelForge.scan!(identity, *, dst, src)
    @test isapprox(dst, CuArray(accumulate(*, src_cpu)))
end

@testset "KernelForge.scan! with Quaternions (non-commutative)" begin
    using Quaternions
    n = 1_000_000
    T = QuaternionF64
    # Normalized quaternions
    src_cpu = [QuaternionF64((x ./ sqrt(sum(x .^ 2)))...) for x in eachcol(randn(4, n))]
    src = CuArray(src_cpu)
    dst = CuArray(zeros(T, n))  # zeros(T, n) works for Quaternions
    CUDA.@sync KernelForge.scan!(identity, *, dst, src)
    @test isapprox(dst, CuArray(accumulate(*, src_cpu)))
end

@testset "KernelForge.scan! comprehensive tests" begin
    test_sizes = [1, 5, 10, 100, 1_000, 10_000, 1_000_000]
    test_types = [Float64, Int32]
    test_ops = [
        (+, "addition"),
        (max, "maximum"),
        (min, "minimum"),
    ]

    for T in test_types
        for n in test_sizes
            for (op, op_name) in test_ops
                @testset "T=$T, n=$n, op=$op_name" begin
                    for trial in 1:5
                        if T <: AbstractFloat
                            src_cpu = rand(T, n)
                        else
                            src_cpu = T[rand(1:10) for _ in 1:n]
                        end
                        src = CuArray(src_cpu)
                        dst = CUDA.zeros(T, n)
                        CUDA.@sync KernelForge.scan!(op, dst, src)
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
    test_sizes = [10_001]
    test_ops = [(+, "addition"), (max, "maximum"), (min, "minimum")]

    for n in test_sizes
        for (op, op_name) in test_ops
            @testset "n=$n, op=$op_name" begin
                for trial in 1:10
                    src_cpu = rand(UInt8, n)
                    src = CuArray(src_cpu)
                    dst = CUDA.zeros(UInt8, n)
                    CUDA.@sync KernelForge.scan!(op, dst, src)
                    @test Array(dst) == accumulate(op, src_cpu)
                end
            end
        end
    end
end

@testset "KernelForge.scan! edge sizes (Float64, +)" begin
    # Edge sizes around common GPU boundaries
    edge_sizes = [
        # Tiny sizes
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        # Around warp size (32)
        # 31, 32, 33,
        # # Around 64 (common block dimension)
        # 63, 64, 65,
        # # Around 128
        # 127, 128, 129,
        # # Around 256 (common block size)
        # 255, 256, 257,
        # # Around 512
        # 511, 512, 513,
        # # Around 1024 (max threads per block)
        # 1023, 1024, 1025,
        # # Around 2048
        # 2047, 2048, 2049,
        # # Around 4096
        # 4095, 4096, 4097,
        # # Around 8192
        # 8191, 8192, 8193,
        # # Larger boundaries
        # 16383, 16384, 16385,
        # 32767, 32768, 32769,
        # 65535, 65536, 65537,
    ]

    for n in edge_sizes
        @testset "n=$n" begin
            src_cpu = rand(Float64, n)
            src = CuArray(src_cpu)
            dst = CUDA.zeros(Float64, n)
            CUDA.@sync KernelForge.scan!(+, dst, src)
            @test isapprox(Array(dst), accumulate(+, src_cpu))
        end
    end
end