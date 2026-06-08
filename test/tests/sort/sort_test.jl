# Backend-portable tests for KernelForge.sort / sort!.
#
# Uses the test harness's `AT` (CuArray on CUDA, ROCArray on AMDGPU) and
# `backend` defined by test/runtests.jl.

# User-defined isbits struct used by the "complex structure with by" testset.
# Guarded so re-running tests in the same Julia session doesn't redefine it.
if !@isdefined(SortTestParticle)
    struct SortTestParticle
        x::Float32
        id::UInt32
    end
end

@testset "KernelForge.sort basic tests" begin
    test_sizes = [1, 33, 100, 10_001, 1_000_000]
    test_types = [UInt8, UInt16, UInt32, UInt64,
                  Int8,  Int16,  Int32,  Int64,
                  Float32, Float64]

    for T in test_types
        for n in test_sizes
            @testset "T=$T, n=$n" begin
                src_cpu = T <: AbstractFloat ? randn(T, n) : rand(T, n)
                src = AT(src_cpu)
                y = KF.sort(src)
                KA.synchronize(backend)
                @test Array(y) == sort(src_cpu)

                # In-place form
                dst = AT(zeros(T, n))
                KF.sort!(dst, src)
                KA.synchronize(backend)
                @test Array(dst) == sort(src_cpu)
            end
        end
    end
end


@testset "KernelForge.sort with `by` kwarg (tuple sort)" begin
    @testset "NTuple{2, Float32} by=first" begin
        n = 1_000_000
        pairs = [(rand(Float32), rand(UInt32)) for _ in 1:n]
        src = AT(pairs)
        y = KF.sort(src; by=first)
        KA.synchronize(backend)
        yh = Array(y)
        # Sorted by first element
        @test issorted(yh; by=first)
        # Permutation
        @test sort(yh) == sort(pairs)
        # Stable: matches CPU's stable sort exactly
        @test yh == sort(pairs; by=first)
    end

    @testset "NTuple{2, UInt32} by=last" begin
        n = 100_000
        pairs = [(rand(UInt32), rand(UInt32)) for _ in 1:n]
        src = AT(pairs)
        y = KF.sort(src; by=last)
        KA.synchronize(backend)
        yh = Array(y)
        @test issorted(yh; by=last)
        @test sort(yh) == sort(pairs)
        @test yh == sort(pairs; by=last)
    end
end


@testset "KernelForge.sort with pre-allocated buffer" begin
    n = 1_000_000
    T = Int32
    src_cpu = rand(T, n)
    src = AT(src_cpu)
    dst = AT(zeros(T, n))
    tmp = KF.get_allocation(KF.Sort1D, src)

    for trial in 1:5
        # Re-randomize to ensure flag arrays etc. are reset properly across calls.
        src_cpu = rand(T, n)
        copyto!(src, src_cpu)
        KF.sort!(dst, src; tmp)
        KA.synchronize(backend)
        @test Array(dst) == sort(src_cpu)
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Edge sizes 1..16 — exercises one-block paths and partial-chunk handling
# in both onesweep and byte-sort kernels (n < workgroup*Nchunks*Nitem).
# ─────────────────────────────────────────────────────────────────────────────
@testset "KernelForge.sort edge sizes 1..16" begin
    for T in (UInt32, Float32), n in 1:16
        @testset "T=$T, n=$n" begin
            src_cpu = T <: AbstractFloat ? randn(T, n) : rand(T, n)
            src = AT(src_cpu)
            y = KF.sort(src)
            KA.synchronize(backend)
            @test Array(y) == sort(src_cpu)
        end
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Multi-trial loops — re-randomize each trial; catches intermittent races in
# the per-block atomic-add ranks and decoupled-lookback partials.
# ─────────────────────────────────────────────────────────────────────────────
@testset "KernelForge.sort multi-trial stability" begin
    for T in (Int32, Float64), n in (100, 10_001, 100_000)
        @testset "T=$T, n=$n" begin
            for trial in 1:10
                src_cpu = T <: AbstractFloat ? randn(T, n) : rand(T, n)
                src = AT(src_cpu)
                dst = AT(zeros(T, n))
                KF.sort!(dst, src)
                KA.synchronize(backend)
                @test Array(dst) == sort(src_cpu)
            end
        end
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial distributions — uniform-random tests miss bucket-skew and
# already-ordered fast paths. Each case stresses a different aspect.
# ─────────────────────────────────────────────────────────────────────────────
@testset "KernelForge.sort adversarial distributions" begin
    n = 10_001
    for T in (UInt32, Int32, Float32)
        @testset "T=$T" begin
            cases = Pair{String,Vector{T}}[
                "all-equal"      => fill(T(7), n),
                "sorted"         => sort(T <: AbstractFloat ? randn(T, n) : rand(T, n)),
                "reverse"        => sort(T <: AbstractFloat ? randn(T, n) : rand(T, n); rev=true),
                "two-value"      => T[iseven(i) ? T(0) : T(1) for i in 1:n],
                "typemin/max"    => T[iseven(i) ? typemin(T) : typemax(T) for i in 1:n],
                "single-outlier" => let v = fill(T(3), n); v[end÷2] = T(99); v end,
            ]
            for (name, src_cpu) in cases
                @testset "$name" begin
                    src = AT(src_cpu)
                    y = KF.sort(src)
                    KA.synchronize(backend)
                    @test Array(y) == sort(src_cpu)
                end
            end
        end
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Float boundary values (±Inf, -0.0). NaN intentionally skipped — the
# float `uint_map` does not give NaNs IEEE-total-order placement.
# Use `isequal` so the -0.0 vs 0.0 distinction is checked bit-for-bit.
# ─────────────────────────────────────────────────────────────────────────────
@testset "KernelForge.sort Float32 boundary values" begin
    n = 10_001
    cases = [
        "with ±Inf" => Float32[isodd(i) ? Float32(Inf) : Float32(-Inf) for i in 1:n],
        "with -0.0" => Float32[iseven(i) ? -0.0f0 : 0.0f0 for i in 1:n],
        "mixed"     => [randn(Float32, n - 4); Float32[Inf, -Inf, 0.0, -0.0]],
    ]
    for (name, src_cpu) in cases
        @testset "$name" begin
            src = AT(src_cpu)
            y = KF.sort(src)
            KA.synchronize(backend)
            @test all(isequal.(Array(y), sort(src_cpu)))
        end
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Bool input — sizeof(uint_map(::Bool)) == 1 → exercises byte_sort_kernel,
# which is otherwise only hit by UInt8/Int8 in the basic block.
# ─────────────────────────────────────────────────────────────────────────────
@testset "KernelForge.sort Bool input (byte-sort path)" begin
    for n in (1, 33, 10_001, 1_000_000)
        @testset "n=$n" begin
            src_cpu = rand(Bool, n)
            src = AT(src_cpu)
            y = KF.sort(src)
            KA.synchronize(backend)
            @test Array(y) == sort(src_cpu)
        end
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Custom struct with `by` — closure over user-defined isbits type.
# ─────────────────────────────────────────────────────────────────────────────
@testset "KernelForge.sort custom struct with by" begin
    n = 100_000
    src_cpu = [SortTestParticle(rand(Float32), UInt32(i)) for i in 1:n]
    src = AT(src_cpu)
    y = KF.sort(src; by = p -> p.x)
    KA.synchronize(backend)
    @test Array(y) == sort(src_cpu; by = p -> p.x)
end


# ─────────────────────────────────────────────────────────────────────────────
# Custom `uint_map` — sort by absolute value via |x| bijection.
# Mirrors scan's "non-commutative op" test in spirit.
# ─────────────────────────────────────────────────────────────────────────────
@testset "KernelForge.sort custom uint_map (sort by |x|)" begin
    n = 10_001
    src_cpu = randn(Float32, n)
    src = AT(src_cpu)
    abs_uint = x -> KernelForge.uint_map(abs(x))
    y = KF.sort(src; uint_map = abs_uint)
    KA.synchronize(backend)
    # GPU result is sorted by |x|; compare to CPU sort with by=abs.
    @test sort(Array(y); by=abs) == sort(src_cpu; by=abs)
    @test issorted(Array(y); by=abs)
end


# ─────────────────────────────────────────────────────────────────────────────
# Views — locked-in coverage for SubArray inputs/outputs at offset positions.
# Strided (non-contiguous) views go through a slow getindex fallback and are
# not a primary use case; only contiguous offset views are exercised here.
# ─────────────────────────────────────────────────────────────────────────────
@testset "KernelForge.sort views" begin
    n = 10_001
    T = UInt32
    parent_src = AT(rand(T, n + 200))
    parent_dst = AT(zeros(T, n + 200))

    @testset "src=view, dst=array" begin
        src = view(parent_src, 51:50+n)
        dst = AT(zeros(T, n))
        KF.sort!(dst, src)
        KA.synchronize(backend)
        @test Array(dst) == sort(Array(src))
    end

    @testset "src=array, dst=view" begin
        src = AT(rand(T, n))
        dst = view(parent_dst, 101:100+n)
        KF.sort!(dst, src)
        KA.synchronize(backend)
        @test Array(dst) == sort(Array(src))
    end

    @testset "src=view, dst=view (different offsets)" begin
        src = view(parent_src, 51:50+n)
        dst = view(parent_dst, 101:100+n)
        KF.sort!(dst, src)
        KA.synchronize(backend)
        @test Array(dst) == sort(Array(src))
    end

    @testset "allocating sort on view" begin
        src = view(parent_src, 51:50+n)
        y = KF.sort(src)
        KA.synchronize(backend)
        @test Array(y) == sort(Array(src))
    end
end
