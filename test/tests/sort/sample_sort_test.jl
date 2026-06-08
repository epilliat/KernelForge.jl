# Backend-portable tests for the sample-sort fallback inside
# KernelForge.sort / sort!. Exercises:
#   • auto-pick (radix for numeric T + default <; sample for any custom
#     `lt`, `tmax`, or `reverse`),
#   • explicit `algorithm = :radix` / `:sample` overrides,
#   • error paths (radix-only constraints, keyval radix-only).
# Uses `AT`, `backend`, `KF`, `KA` from test/runtests.jl.

# User-defined exotic bitstype (no `typemax` method) ⇒ forces the
# tag-path of sample sort. Guarded so re-running in one session is OK.
if !@isdefined(SampleSortPair)
    struct SampleSortPair
        a::UInt32
        b::UInt32
    end
end
@inline _sspair_lt(p::SampleSortPair, q::SampleSortPair) =
    (p.a, p.b) < (q.a, q.b)


@testset "KernelForge.sort dispatcher — auto-pick" begin
    test_sizes = [33, 10_001, 1_000_000]

    @testset "default `<` on numeric T → radix path (smoke)" begin
        # Sample-sort is exercised heavily elsewhere; this confirms the
        # default keeps using radix (which sort_test.jl validates in
        # depth) for built-in numeric types.
        for T in (UInt32, Int32, Float32, UInt64), n in test_sizes
            src_cpu = T <: AbstractFloat ? randn(T, n) : rand(T, n)
            src = AT(src_cpu)
            y = KF.sort(src)                       # algorithm = :auto
            KA.synchronize(backend)
            @test Array(y) == sort(src_cpu)
        end
    end

    @testset "custom `lt` → sample path (auto)" begin
        for T in (UInt32, UInt64, Int32, Float32), n in test_sizes
            src_cpu = T <: AbstractFloat ? randn(T, n) : rand(T, n)
            src = AT(src_cpu)
            # `lt = >` — disagrees with `<`; exact-order check.
            yg = KF.sort(src; lt = >)
            KA.synchronize(backend)
            @test Array(yg) == sort(src_cpu; lt = >)
            # `lt = isless` (== `<` for these types) — sanity.
            yi = KF.sort(src; lt = isless)
            KA.synchronize(backend)
            @test Array(yi) == sort(src_cpu; lt = isless)
        end
    end

    @testset "non-standard `lt` (mod-then-value) → sample (auto)" begin
        modlt = (a, b) -> (a % 0x61, a) < (b % 0x61, b)
        for T in (UInt32, UInt64), n in test_sizes
            src_cpu = rand(T, n)
            src = AT(src_cpu)
            y = KF.sort(src; lt = modlt)
            KA.synchronize(backend)
            @test Array(y) == sort(src_cpu; lt = modlt)
        end
    end

    @testset "`reverse = true` → sample path (auto)" begin
        for T in (UInt32, Float32), n in test_sizes
            src_cpu = T <: AbstractFloat ? randn(T, n) : rand(T, n)
            src = AT(src_cpu)
            y = KF.sort(src; reverse = true)
            KA.synchronize(backend)
            @test Array(y) == sort(src_cpu; rev = true)
        end
    end

    @testset "`tmax = …` (user sentinel) → sample path (auto)" begin
        n = 100_000
        src_cpu = rand(UInt32, n)
        src = AT(src_cpu)
        y = KF.sort(src; tmax = typemax(UInt32))
        KA.synchronize(backend)
        @test Array(y) == sort(src_cpu)
    end

    @testset "explicit `algorithm = :sample` on default-comparator UInt32" begin
        n = 100_000
        src_cpu = rand(UInt32, n)
        src = AT(src_cpu)
        y = KF.sort(src; algorithm = :sample)
        KA.synchronize(backend)
        @test Array(y) == sort(src_cpu)
    end

    @testset "exotic bitstype (no typemax) + custom lt → sample tag path" begin
        n = 50_000
        src_cpu = [SampleSortPair(rand(UInt32), rand(UInt32)) for _ in 1:n]
        src = AT(src_cpu)
        y = KF.sort(src; lt = _sspair_lt)        # auto → :sample (no Unsigned uint_map)
        KA.synchronize(backend)
        @test Array(y) == sort(src_cpu; lt = _sspair_lt)
    end

    @testset "in-place `sort!(src; lt=…)` routes to sample" begin
        n = 50_000
        src_cpu = rand(UInt32, n)
        src = AT(copy(src_cpu))
        KF.sort!(src; lt = >)
        KA.synchronize(backend)
        @test Array(src) == sort(src_cpu; lt = >)
    end

    @testset "two-arg `sort!(dst, src; lt=…)` routes to sample" begin
        n = 50_000
        src_cpu = rand(UInt32, n)
        src = AT(src_cpu)
        dst = AT(zeros(UInt32, n))
        KF.sort!(dst, src; lt = >)
        KA.synchronize(backend)
        @test Array(dst) == sort(src_cpu; lt = >)
    end
end


@testset "KernelForge.sort dispatcher — error paths" begin
    @testset "`algorithm = :radix` rejects `lt`" begin
        src = AT(rand(UInt32, 100))
        @test_throws ErrorException KF.sort(src; algorithm = :radix, lt = >)
    end

    @testset "`algorithm = :radix` rejects `reverse=true`" begin
        src = AT(rand(UInt32, 100))
        @test_throws ErrorException KF.sort(src; algorithm = :radix, reverse = true)
    end

    @testset "`algorithm = :radix` on exotic bitstype (no `uint_map`)" begin
        src_cpu = [SampleSortPair(rand(UInt32), rand(UInt32)) for _ in 1:100]
        src = AT(src_cpu)
        @test_throws ErrorException KF.sort(src; algorithm = :radix)
    end

    @testset "`algorithm = :sample` rejects `keys=...` (keyval is radix-only)" begin
        n = 100
        vals = AT(rand(UInt32, n))
        keys = AT(rand(UInt32, n))
        @test_throws ErrorException KF.sort(vals; algorithm = :sample, keys = keys)
    end

    @testset "bogus `algorithm = :foo` is rejected" begin
        src = AT(rand(UInt32, 100))
        @test_throws ErrorException KF.sort(src; algorithm = :foo)
    end
end
