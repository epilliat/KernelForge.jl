# Reproducibility — same (seed, N) produces byte-identical bytes across
# repeated calls in the same process. Inter-process reproducibility was
# verified during exploration (xp/philox/repro_check.jl).

using SHA

const KFR = KernelForge.Random

digest(v::AbstractArray) =
    SHA.bytes2hex(SHA.sha256(reinterpret(UInt8, vec(Array(v)))))

@testset "reproducibility — within-process" begin
    seed = UInt64(0xDEADBEEFCAFEBABE)
    N = 10_000

    # Each filler must produce the same bytes when called twice.
    for (T, dist, ka_args) in (
        (Float32, KFR.Uniform(0f0, 1f0),    nothing),
        (Float64, KFR.Uniform(0.0, 1.0),    nothing),
        (Float32, KFR.Exponential(2f0),     nothing),
        (Float32, KFR.Normal(0f0, 1f0),     nothing),
        (Bool,    KFR.Bernoulli(0.3f0),     nothing),
    )
        cpu_a = zeros(T, N); KFR.rand!(cpu_a, dist, seed)
        cpu_b = zeros(T, N); KFR.rand!(cpu_b, dist, seed)
        @test cpu_a == cpu_b

        gpu_a = AT(zeros(T, N)); KFR.rand!(gpu_a, dist, seed)
        gpu_b = AT(zeros(T, N)); KFR.rand!(gpu_b, dist, seed)
        KA.synchronize(backend)
        @test Array(gpu_a) == Array(gpu_b)

        @test digest(gpu_a) == digest(gpu_b)
    end
end

@testset "reproducibility — independent of vector length" begin
    seed = UInt64(0xC0FFEE)
    a = zeros(Float32, 1_000); KFR.rand!(a, seed)
    b = zeros(Float32, 5_000); KFR.rand!(b, seed)
    @test digest(a) == digest(b[1:1_000])
end

@testset "reproducibility — independent of backend" begin
    # CPU↔GPU bit-equality for uniform streams (asserted again here so that
    # this file is self-contained as a reproducibility check).
    seed = UInt64(2026)
    for D in (KFR.Uniform(0f0, 1f0), KFR.Uniform(0.0, 1.0))
        T = eltype(D)
        cpu = zeros(T, 4_000); KFR.rand!(cpu, D, seed)
        gpu = AT(zeros(T, 4_000)); KFR.rand!(gpu, D, seed)
        KA.synchronize(backend)
        @test digest(gpu) == digest(cpu)
    end
end
