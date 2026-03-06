@testset "vcopy! correctness" begin
    ns = [100_000_007, 100_000_001, 100_005, 100_003, 33, 1, 1_000_000]

    @testset "Float32, Nitem=1" begin
        n = 100_000_007
        T = Float32
        src_cpu = rand(T, n)
        src = AT(src_cpu)
        dst = AT(zeros(T, n))
        KF.vcopy!(dst, src; Nitem=1)
        KA.synchronize(backend)
        @test all(Array(dst) .== Array(src))
    end

    @testset "UInt8, Nitem=4" begin
        n = 100_000_001
        T = UInt8
        src = AT(fill(0x03, n))
        dst = AT(zeros(T, n))
        KF.vcopy!(dst, src; Nitem=4)
        KA.synchronize(backend)
        @test all(Array(dst) .== Array(src))
    end

    @testset "custom struct U, Nitem=2" begin
        struct UCopy
            x::UInt8
            y::UInt8
            z::UInt8
        end
        Base.:(==)(a::UCopy, b::UCopy) = a.x == b.x && a.y == b.y && a.z == b.z

        n = 100_005
        src = AT([UCopy(1, 1, 1) for _ in 1:n])
        dst = AT([UCopy(0, 0, 0) for _ in 1:n])
        KF.vcopy!(dst, src; Nitem=2)
        KA.synchronize(backend)
        @test all(Array(dst) .== Array(src))
    end

    @testset "setvalue! UInt8, Nitem=4" begin
        n = 100_003
        T = UInt8
        dst = AT(fill(0xff, n))
        KF.setvalue!(dst, 0x00; Nitem=4)
        KA.synchronize(backend)
        @test all(Array(dst) .== 0x00)
    end

    @testset "Type $T, n=$n, Nitem=$Nitem" for (T, n, Nitem) in [
        (Float32, 33, 1),
        (Float64, 100_000, 1),
        (UInt8, 100_000_001, 4),
        (UInt8, 33, 1),
        (Int32, 100_001, 2),
    ]
        src_cpu = T == UInt8 ? fill(0x03, n) : rand(T, n)
        src = AT(src_cpu)
        dst = AT(zeros(T, n))
        KF.vcopy!(dst, src; Nitem)
        KA.synchronize(backend)
        @test all(Array(dst) .== Array(src))
    end

    @testset "setvalue! Type $T, n=$n" for (T, val, n) in [
        (UInt8, 0x00, 100_003),
        (Float32, 0f0, 100_007),
        (Int32, Int32(0), 99_999),
    ]
        dst = AT(fill(typemax(T), n))
        KF.setvalue!(dst, val)
        KA.synchronize(backend)
        @test all(Array(dst) .== val)
    end
end