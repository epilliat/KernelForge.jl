# Backend-portable tests for KernelForge.sortperm / sortperm!
# Reference is Julia's stdlib `sortperm` (stable). The GPU radix path is
# also stable, so element-wise equality is the right comparison.

@testset "KernelForge.sortperm" begin

    # ─────────────────────────────────────────────────────────────────────
    # Basic correctness across types × sizes.
    # Every (eltype, n) combo must produce the exact same permutation as
    # CPU's stable sortperm.
    # ─────────────────────────────────────────────────────────────────────
    test_sizes = [1, 33, 10_001, 200_000]
    test_types = [UInt8, UInt16, UInt32, UInt64,
                  Int8,  Int16,  Int32,  Int64,
                  Float32, Float64]

    for T in test_types, n in test_sizes
        @testset "T=$T n=$n" begin
            src_cpu = T <: AbstractFloat ? randn(T, n) : rand(T, n)
            src = AT(src_cpu)
            p = KF.sortperm(src)
            KA.synchronize(backend)
            p_h = Array(p)
            @test p_h == sortperm(src_cpu)
            @test src_cpu[p_h] == sort(src_cpu)
        end
    end


    # ─────────────────────────────────────────────────────────────────────
    # In-place form returns `perm` (not a tuple) and writes the same
    # permutation as the allocating form.
    # ─────────────────────────────────────────────────────────────────────
    @testset "in-place sortperm!" begin
        n = 100_000
        src_cpu = rand(UInt32, n)
        src  = AT(src_cpu)
        perm = AT(zeros(Int32, n))
        ret  = KF.sortperm!(perm, src)
        KA.synchronize(backend)
        @test ret === perm
        @test Array(perm) == sortperm(src_cpu)
    end


    # ─────────────────────────────────────────────────────────────────────
    # IndexType override: returned eltype matches request, perm correct.
    # ─────────────────────────────────────────────────────────────────────
    @testset "IndexType override" begin
        n = 50_000
        src_cpu = rand(Float32, n)
        src = AT(src_cpu)
        @testset "Int64" begin
            p = KF.sortperm(src; IndexType=Int64)
            KA.synchronize(backend)
            @test eltype(p) === Int64
            @test Array(p) == sortperm(src_cpu)
        end
        @testset "Int32" begin
            p = KF.sortperm(src; IndexType=Int32)
            KA.synchronize(backend)
            @test eltype(p) === Int32
            @test Array(p) == sortperm(src_cpu)
        end
        @testset "default IndexType=Int32 for small n" begin
            p = KF.sortperm(src)
            KA.synchronize(backend)
            @test eltype(p) === Int32
        end
    end


    # ─────────────────────────────────────────────────────────────────────
    # Length mismatch + non-Integer perm errors.
    # ─────────────────────────────────────────────────────────────────────
    @testset "error paths" begin
        @testset "length mismatch" begin
            src  = AT(rand(UInt32, 100))
            perm = AT(zeros(Int32, 99))
            @test_throws ErrorException KF.sortperm!(perm, src)
        end
        @testset "non-Integer perm eltype" begin
            src  = AT(rand(UInt32, 100))
            perm = AT(zeros(Float32, 100))
            @test_throws ErrorException KF.sortperm!(perm, src)
        end
    end


    # ─────────────────────────────────────────────────────────────────────
    # @allocate macro forwards to get_allocation(Sortperm, ...).
    # ─────────────────────────────────────────────────────────────────────
    @testset "@allocate sortperm" begin
        n = 50_000
        src = AT(rand(UInt32, n))
        tmp = KF.@allocate sortperm(src)
        @test tmp isa KF.KernelBuffer
        for f in (:hist, :partial1, :partial2, :flag, :scratch, :scratch_keys, :keys_dst)
            @test f in propertynames(tmp.arrays)
        end
        # Buffer is usable end-to-end.
        p = KF.sortperm(src; tmp=tmp)
        KA.synchronize(backend)
        @test Array(p) == sortperm(Array(src))
    end


    # ─────────────────────────────────────────────────────────────────────
    # Pre-allocated buffer reuse across re-randomized inputs.
    # ─────────────────────────────────────────────────────────────────────
    @testset "buffer reuse" begin
        n = 100_000
        src = AT(rand(UInt32, n))
        perm = AT(zeros(Int32, n))
        tmp = KF.get_allocation(KF.Sortperm, src)
        for trial in 1:5
            src_cpu = rand(UInt32, n)
            copyto!(src, src_cpu)
            KF.sortperm!(perm, src; tmp=tmp)
            KA.synchronize(backend)
            @test Array(perm) == sortperm(src_cpu)
        end
    end


    # ─────────────────────────────────────────────────────────────────────
    # `by=first` on tuple input.
    # ─────────────────────────────────────────────────────────────────────
    @testset "by=first on tuples" begin
        n = 100_000
        src_cpu = [(rand(UInt32), rand(UInt32)) for _ in 1:n]
        src = AT(src_cpu)
        p = KF.sortperm(src; by=first)
        KA.synchronize(backend)
        @test Array(p) == sortperm(src_cpu; by=first)
    end


    # ─────────────────────────────────────────────────────────────────────
    # Custom uint_map: sort by absolute value.
    # ─────────────────────────────────────────────────────────────────────
    @testset "custom uint_map (sort by |x|)" begin
        n = 10_001
        src_cpu = randn(Float32, n)
        src = AT(src_cpu)
        abs_uint = x -> KF.uint_map(abs(x))
        p = KF.sortperm(src; uint_map=abs_uint)
        KA.synchronize(backend)
        p_h = Array(p)
        @test p_h == sortperm(src_cpu; by=abs)
        @test issorted(src_cpu[p_h]; by=abs)
    end


    # ─────────────────────────────────────────────────────────────────────
    # Adversarial key distributions — must produce a valid sortperm
    # (i.e. src[p] is sorted) AND match Base.sortperm (since both stable).
    # ─────────────────────────────────────────────────────────────────────
    @testset "adversarial distributions" begin
        n = 10_001
        for T in (UInt32, Int32, Float32)
            @testset "T=$T" begin
                cases = Pair{String,Vector{T}}[
                    "all-equal"      => fill(T(7), n),
                    "two-value"      => T[iseven(i) ? T(0) : T(1) for i in 1:n],
                    "typemin/max"    => T[iseven(i) ? typemin(T) : typemax(T) for i in 1:n],
                    "single-outlier" => let v = fill(T(3), n); v[end÷2] = T(99); v end,
                ]
                for (name, src_cpu) in cases
                    @testset "$name" begin
                        src = AT(src_cpu)
                        p = KF.sortperm(src)
                        KA.synchronize(backend)
                        p_h = Array(p)
                        @test p_h == sortperm(src_cpu)
                        @test src_cpu[p_h] == sort(src_cpu)
                    end
                end
            end
        end
    end


    # ─────────────────────────────────────────────────────────────────────
    # Edge sizes 1..16 — exercises one-block paths and partial-chunk
    # handling identical to sort's edge-size testset.
    # ─────────────────────────────────────────────────────────────────────
    @testset "edge sizes 1..16" begin
        for T in (UInt32, Float32), n in 1:16
            @testset "T=$T n=$n" begin
                src_cpu = T <: AbstractFloat ? randn(T, n) : rand(T, n)
                src = AT(src_cpu)
                p = KF.sortperm(src)
                KA.synchronize(backend)
                @test Array(p) == sortperm(src_cpu)
            end
        end
    end
end
