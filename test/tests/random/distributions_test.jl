# Distribution-specific tests: bit-equality / ≈, theoretical moments.

using Statistics

const KFR = KernelForge.Random

@testset "Uniform[a, b]" begin
    N = 1_000_000
    cpu = zeros(Float32, N); KFR.rand!(cpu, KFR.Uniform(2f0, 5f0), UInt64(0xC0FFEE))
    gpu = AT(zeros(Float32, N)); KFR.rand!(gpu, KFR.Uniform(2f0, 5f0), UInt64(0xC0FFEE))
    KA.synchronize(backend)
    @test Array(gpu) == cpu
    @test all(2 .≤ cpu .≤ 5)
    @test 3.49 < mean(cpu) < 3.51                  # E = (a+b)/2
    @test 0.74 < var(cpu)  < 0.76                  # Var = (b-a)²/12
end

@testset "Bernoulli(p)" begin
    N = 1_000_000
    cpu = zeros(Bool, N); KFR.rand!(cpu, KFR.Bernoulli(0.3f0), UInt64(0xC0FFEE))
    gpu = AT(zeros(Bool, N)); KFR.rand!(gpu, KFR.Bernoulli(0.3f0), UInt64(0xC0FFEE))
    KA.synchronize(backend)
    @test Array(gpu) == cpu
    @test 0.297 < mean(cpu) < 0.303                # 5σ band on √(p(1-p)/N)
end

@testset "Exponential(λ)" begin
    N = 1_000_000
    cpu = zeros(Float32, N); KFR.rand!(cpu, KFR.Exponential(2f0), UInt64(0xC0FFEE))
    gpu = AT(zeros(Float32, N)); KFR.rand!(gpu, KFR.Exponential(2f0), UInt64(0xC0FFEE))
    KA.synchronize(backend)
    @test isapprox(Array(gpu), cpu; rtol=1f-5)     # ≈ (uses log)
    @test all(cpu .≥ 0)
    @test 0.49 < mean(cpu) < 0.51                  # E = 1/λ
    @test 0.24 < var(cpu)  < 0.26                  # Var = 1/λ²
end

@testset "Categorical(weights)" begin
    N = 1_000_000
    w_cpu = Float32[0.1, 0.5, 1.0]                 # cumulative
    w_gpu = AT(w_cpu)
    cpu = zeros(Int32, N); KFR.rand!(cpu, KFR.Categorical(w_cpu), UInt64(0xC0FFEE))
    gpu = AT(zeros(Int32, N)); KFR.rand!(gpu, KFR.Categorical(w_gpu), UInt64(0xC0FFEE))
    KA.synchronize(backend)
    @test Array(gpu) == cpu
    counts = [count(==(k), cpu)/N for k in 1:3]
    @test 0.099 < counts[1] < 0.101
    @test 0.399 < counts[2] < 0.401
    @test 0.499 < counts[3] < 0.501
end

@testset "Normal(μ, σ)" begin
    N = 1_000_000
    cpu = zeros(Float32, N); KFR.rand!(cpu, KFR.Normal(10f0, 2f0), UInt64(0xC0FFEE))
    gpu = AT(zeros(Float32, N)); KFR.rand!(gpu, KFR.Normal(10f0, 2f0), UInt64(0xC0FFEE))
    KA.synchronize(backend)
    @test isapprox(Array(gpu), cpu; atol=1f-3)     # ≈ (Box–Muller log/cos)
    @test 9.99 < mean(cpu) < 10.01
    @test 1.99 < std(cpu)  < 2.01

    # randn! shortcut.
    z = AT(zeros(Float32, 10_000)); KFR.randn!(z, UInt64(7))
    KA.synchronize(backend)
    @test abs(mean(Array(z))) < 0.05
    @test 0.95 < std(Array(z)) < 1.05
end

@testset "Normal — tail handling (odd N)" begin
    # 2u/sample kernel needs to handle N % 2 ≠ 0.
    for N in (1, 3, 7, 13, 1025)
        cpu = zeros(Float32, N); KFR.randn!(cpu, UInt64(42))
        gpu = AT(zeros(Float32, N)); KFR.randn!(gpu, UInt64(42))
        KA.synchronize(backend)
        @test isapprox(Array(gpu), cpu; atol=1f-3)
    end
end
