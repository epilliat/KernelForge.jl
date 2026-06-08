# randperm! — random permutation via Float32 uniform keys + stable sortperm.
# Tests the contract: out is a permutation of 1..N, deterministic from seed,
# bit-identical across CPU/GPU.

const KFR = KernelForge.Random

@testset "randperm — basic correctness" begin
    for N in (1, 7, 1023, 100_000)
        perm = AT(zeros(Int32, N)); KFR.randperm!(perm, UInt64(42))
        KA.synchronize(backend)
        p = Array(perm)
        @test sort(p) == collect(Int32(1):Int32(N))
    end
end

@testset "randperm — determinism" begin
    a = AT(zeros(Int32, 10_000)); KFR.randperm!(a, UInt64(42))
    b = AT(zeros(Int32, 10_000)); KFR.randperm!(b, UInt64(42))
    KA.synchronize(backend)
    @test Array(a) == Array(b)
end

@testset "randperm — seed sensitivity" begin
    a = AT(zeros(Int32, 1024)); KFR.randperm!(a, UInt64(1))
    b = AT(zeros(Int32, 1024)); KFR.randperm!(b, UInt64(2))
    KA.synchronize(backend)
    @test Array(a) != Array(b)
end

@testset "randperm — Int64 perm" begin
    perm = AT(zeros(Int64, 10_000)); KFR.randperm!(perm, UInt64(7))
    KA.synchronize(backend)
    @test sort(Array(perm)) == collect(Int64(1):Int64(10_000))
end

# NOTE: no CPU↔GPU bit-equality test — KernelForge's sort/sortperm
# family is GPU-only (shared-memory-heavy radix kernels), so randperm!
# inherits that constraint.

@testset "randperm — first-position uniformity" begin
    # Histogram of perm[1] across many seeds. Each of N positions should
    # appear roughly trials/N times — check via χ²-style 4σ band.
    N = 10_000; trials = 4_000; nbuckets = 100
    buckets = zeros(Int, nbuckets)
    perm = AT(zeros(Int32, N))
    for t in 1:trials
        KFR.randperm!(perm, UInt64(t))
        KA.synchronize(backend)
        v = Array(perm)[1]
        b = 1 + ((Int(v) - 1) * nbuckets) ÷ N
        buckets[b] += 1
    end
    expected = trials ÷ nbuckets
    σ = sqrt(expected * (1 - 1/nbuckets))
    @test all(abs.(buckets .- expected) .< 4σ)
end
