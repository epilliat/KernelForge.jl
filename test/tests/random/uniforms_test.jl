# Uniform-stream properties: determinism, length-independence, seed
# sensitivity, CPU↔GPU bit-equality, statistical moments.

using Statistics

const KFR = KernelForge.Random

@testset "uniforms — determinism (CPU)" begin
    for D in (KFR.Uniform(0f0, 1f0), KFR.Uniform(0.0, 1.0))
        T = eltype(D)
        a = zeros(T, 10_000); KFR.rand!(a, D, UInt64(42))
        b = zeros(T, 10_000); KFR.rand!(b, D, UInt64(42))
        @test a == b
    end
end

@testset "uniforms — length-independence" begin
    a = zeros(Float32, 1_000); KFR.rand!(a, UInt64(7))
    b = zeros(Float32, 5_000); KFR.rand!(b, UInt64(7))
    @test a == b[1:1_000]

    a64 = zeros(Float64, 1_234); KFR.rand!(a64, UInt64(7))
    b64 = zeros(Float64, 9_999); KFR.rand!(b64, UInt64(7))
    @test a64 == b64[1:1_234]
end

@testset "uniforms — seed sensitivity" begin
    a = zeros(Float32, 1024); KFR.rand!(a, UInt64(1))
    b = zeros(Float32, 1024); KFR.rand!(b, UInt64(2))
    @test a != b
    @test count(a .== b) ≤ 3
end

@testset "uniforms — CPU↔GPU bit-equality" begin
    N = 1 << 16
    for D in (KFR.Uniform(0f0, 1f0), KFR.Uniform(-2f0, 5f0),
              KFR.Uniform(0.0, 1.0))
        T = eltype(D)
        cpu = zeros(T, N); KFR.rand!(cpu, D, UInt64(123))
        gpu = AT(zeros(T, N)); KFR.rand!(gpu, D, UInt64(123))
        KA.synchronize(backend)
        @test Array(gpu) == cpu
    end
end

@testset "uniforms — tail handling (N % 4 ≠ 0)" begin
    # CPU↔GPU must still be bit-equal for non-divisible lengths.
    for N in (1, 3, 7, 13, 1023, 4097)
        cpu = zeros(Float32, N); KFR.rand!(cpu, UInt64(99))
        gpu = AT(zeros(Float32, N)); KFR.rand!(gpu, UInt64(99))
        KA.synchronize(backend)
        @test Array(gpu) == cpu
    end
end

@testset "uniforms — statistical moments (1M samples)" begin
    N = 1_000_000
    x = zeros(Float64, N); KFR.rand!(x, UInt64(0xC0FFEE))
    m, v = mean(x), var(x)
    @test 0.499 < m < 0.501
    @test 0.0830 < v < 0.0837    # Var(U[0,1)) = 1/12 ≈ 0.0833
end

@testset "uniforms — algo invariance" begin
    # `(seed, i) → out[i]` must hold no matter which kernel template runs.
    # StandardKernel = Val{SPP, NPC} per-thread mapping;
    # PersistentKernel = fixed thread pool looping over output.
    seed = UInt64(0xC0FFEE)
    for D in (KFR.Uniform(0f0, 1f0), KFR.Uniform(-2f0, 5f0), KFR.Uniform(0.0, 1.0))
        T = eltype(D)
        for N in (1_000, 100_000, 1_000_000, 10_000_000)
            std_out = AT(zeros(T, N)); KFR.rand!(std_out, D, seed; algo=KFR.StandardKernel())
            per_out = AT(zeros(T, N)); KFR.rand!(per_out, D, seed; algo=KFR.PersistentKernel())
            KA.synchronize(backend)
            @test Array(std_out) == Array(per_out)
        end
    end
    # Normal (SPP=2 post-v10) — same invariance check.
    for N in (1_000, 100_000, 1_000_000)
        std_out = AT(zeros(Float32, N)); KFR.rand!(std_out, KFR.Normal(0f0, 1f0), UInt64(7); algo=KFR.StandardKernel())
        per_out = AT(zeros(Float32, N)); KFR.rand!(per_out, KFR.Normal(0f0, 1f0), UInt64(7); algo=KFR.PersistentKernel())
        KA.synchronize(backend)
        @test Array(std_out) == Array(per_out)
    end
end

@testset "uniforms — Nitem invariance" begin
    # The (seed, i) → out[i] contract must hold no matter how the work is
    # partitioned across threads. Different Nitem values exercise different
    # NPC = Nitem ÷ samples_per_philox(dist) per-thread block sizes.
    seed = UInt64(0xC0FFEE)
    for D in (KFR.Uniform(0f0, 1f0), KFR.Uniform(-2f0, 5f0))
        T = eltype(D)
        for N in (1_000, 100_000, 1_000_000)
            ref = nothing
            for Nitem in (4, 8, 16)
                gpu = AT(zeros(T, N))
                KFR.rand!(gpu, D, seed; Nitem=Nitem)
                KA.synchronize(backend)
                got = Array(gpu)
                if ref === nothing
                    ref = got
                else
                    @test got == ref
                end
            end
        end
    end

    # Normal (SPP=4 post-v10) — same invariance.
    for N in (1_000, 100_000, 1_000_000)
        ref = nothing
        for Nitem in (4, 8, 16)
            gpu = AT(zeros(Float32, N))
            KFR.rand!(gpu, KFR.Normal(0f0, 1f0), UInt64(7); Nitem=Nitem)
            KA.synchronize(backend)
            got = Array(gpu)
            if ref === nothing
                ref = got
            else
                @test got == ref
            end
        end
    end
end
