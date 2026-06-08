# Backend-portable tests for KernelForge.sort / sort! with `keys=`
# (key-value sort: src and a parallel `keys` array sorted together using
# `uint_map(by(keys[i]))` as the sort key).
#
# Uses `AT` and `backend` defined by test/runtests.jl.

@testset "KernelForge.sort keyval (keys kwarg)" begin

    # ─────────────────────────────────────────────────────────────────────
    # Basic correctness across (val_type × key_type × n).
    # The reference is `sortperm(keys_cpu)` (Julia's stable sort), since
    # the GPU radix path is also stable. We compare keys element-wise and
    # values element-wise, both of which only hold under stable sorting.
    # ─────────────────────────────────────────────────────────────────────
    val_types = [UInt32, Int32, Float32, Float64]
    key_types = [UInt8, UInt16, UInt32, UInt64, Int32, Float32]
    test_sizes = [1, 33, 10_001, 200_000]

    for VT in val_types, KT in key_types, n in test_sizes
        @testset "VT=$VT KT=$KT n=$n" begin
            # Unique value tags so a stable sort produces a unique witness
            # for the value permutation (catches non-stability bugs).
            vals_cpu = collect(VT(1):VT(n))
            keys_cpu = KT <: AbstractFloat ? randn(KT, n) : rand(KT, n)
            vals = AT(vals_cpu)
            keys = AT(keys_cpu)

            # Allocating form returns (sorted_src, sorted_keys).
            sv, sk = KF.sort(vals; keys=keys)
            KA.synchronize(backend)
            sv_h = Array(sv); sk_h = Array(sk)

            perm = sortperm(keys_cpu)
            @test sk_h == keys_cpu[perm]
            @test sv_h == vals_cpu[perm]
            @test issorted(sk_h)
        end
    end


    # ─────────────────────────────────────────────────────────────────────
    # In-place form with explicit dst, keys_dst. Returns `(dst, keys_dst)`.
    # ─────────────────────────────────────────────────────────────────────
    @testset "in-place sort! with keys_dst" begin
        n = 100_000
        vals_cpu = rand(UInt32, n)
        keys_cpu = rand(UInt32, n)
        vals = AT(vals_cpu)
        keys = AT(keys_cpu)
        dst       = AT(zeros(UInt32, n))
        keys_dst  = AT(zeros(UInt32, n))
        ret = KF.sort!(dst, vals; keys=keys, keys_dst=keys_dst)
        KA.synchronize(backend)
        # Returns the same buffers the user supplied.
        @test ret isa Tuple
        @test ret[1] === dst
        @test ret[2] === keys_dst
        perm = sortperm(keys_cpu)
        @test Array(dst)      == vals_cpu[perm]
        @test Array(keys_dst) == keys_cpu[perm]
    end


    # ─────────────────────────────────────────────────────────────────────
    # In-place sort! with `keys=` but no `keys_dst=`: auto-allocates and
    # returns the allocated buffer.
    # ─────────────────────────────────────────────────────────────────────
    @testset "in-place sort! auto-allocates keys_dst" begin
        n = 100_000
        vals_cpu = rand(UInt32, n)
        keys_cpu = rand(UInt32, n)
        vals = AT(vals_cpu)
        keys = AT(keys_cpu)
        dst  = AT(zeros(UInt32, n))
        ret = KF.sort!(dst, vals; keys=keys)
        KA.synchronize(backend)
        @test ret isa Tuple && length(ret) == 2
        d_out, k_out = ret
        @test d_out === dst
        @test eltype(k_out) === UInt32
        @test length(k_out) == n
        perm = sortperm(keys_cpu)
        @test Array(d_out) == vals_cpu[perm]
        @test Array(k_out) == keys_cpu[perm]
    end


    # ─────────────────────────────────────────────────────────────────────
    # `keys === nothing` (i.e. neither keys nor keys_dst): existing
    # value-only behavior, returns `dst` (not a tuple).
    # ─────────────────────────────────────────────────────────────────────
    @testset "keys=nothing falls through to value-only sort" begin
        n = 100_000
        src_cpu = rand(UInt32, n)
        src = AT(src_cpu)
        dst = AT(zeros(UInt32, n))
        ret_in_place = KF.sort!(dst, src)            # no keys
        KA.synchronize(backend)
        @test ret_in_place === dst                     # not a tuple
        @test Array(dst) == sort(src_cpu)

        ret_alloc = KF.sort(src)                     # no keys
        KA.synchronize(backend)
        @test ret_alloc isa AbstractArray              # not a tuple
        @test Array(ret_alloc) == sort(src_cpu)
    end


    # ─────────────────────────────────────────────────────────────────────
    # Length mismatch error.
    # ─────────────────────────────────────────────────────────────────────
    @testset "length mismatch errors" begin
        vals = AT(rand(UInt32, 100))
        keys = AT(rand(UInt32, 99))
        @test_throws ErrorException KF.sort(vals; keys=keys)
    end


    # ─────────────────────────────────────────────────────────────────────
    # `by=` applied to the keys (NTuple key sorted by first element).
    # ─────────────────────────────────────────────────────────────────────
    @testset "by=first on tuple keys" begin
        n = 50_000
        vals_cpu = collect(UInt32(1):UInt32(n))
        keys_cpu = [(rand(UInt32), rand(UInt32)) for _ in 1:n]
        vals = AT(vals_cpu)
        keys = AT(keys_cpu)
        sv, sk = KF.sort(vals; keys=keys, by=first)
        KA.synchronize(backend)
        sv_h = Array(sv); sk_h = Array(sk)
        perm = sortperm(keys_cpu; by=first)
        @test issorted(sk_h; by=first)
        @test sv_h == vals_cpu[perm]
        @test sk_h == keys_cpu[perm]
    end


    # ─────────────────────────────────────────────────────────────────────
    # Custom uint_map on keys (sort by |x| via abs).
    # ─────────────────────────────────────────────────────────────────────
    @testset "custom uint_map (sort by |key|)" begin
        n = 10_001
        vals_cpu = collect(UInt32(1):UInt32(n))
        keys_cpu = randn(Float32, n)
        vals = AT(vals_cpu)
        keys = AT(keys_cpu)
        abs_uint = x -> KF.uint_map(abs(x))
        sv, sk = KF.sort(vals; keys=keys, uint_map=abs_uint)
        KA.synchronize(backend)
        sv_h = Array(sv); sk_h = Array(sk)
        perm = sortperm(keys_cpu; by=abs)
        @test issorted(sk_h; by=abs)
        @test sk_h == keys_cpu[perm]
        @test sv_h == vals_cpu[perm]
    end


    # ─────────────────────────────────────────────────────────────────────
    # Adversarial key distributions.
    # ─────────────────────────────────────────────────────────────────────
    @testset "adversarial key distributions" begin
        n = 10_001
        for KT in (UInt32, Int32)
            @testset "KT=$KT" begin
                cases = Pair{String,Vector{KT}}[
                    "all-equal"      => fill(KT(7), n),
                    "two-value"      => KT[iseven(i) ? KT(0) : KT(1) for i in 1:n],
                    "typemin/max"    => KT[iseven(i) ? typemin(KT) : typemax(KT) for i in 1:n],
                    "single-outlier" => let v = fill(KT(3), n); v[end÷2] = KT(99); v end,
                ]
                for (name, keys_cpu) in cases
                    @testset "$name" begin
                        vals_cpu = collect(UInt32(1):UInt32(n))
                        vals = AT(vals_cpu)
                        keys = AT(keys_cpu)
                        sv, sk = KF.sort(vals; keys=keys)
                        KA.synchronize(backend)
                        perm = sortperm(keys_cpu)
                        @test Array(sk) == keys_cpu[perm]
                        @test Array(sv) == vals_cpu[perm]
                    end
                end
            end
        end
    end


    # ─────────────────────────────────────────────────────────────────────
    # @allocate macro forwards keyword arguments correctly.
    # ─────────────────────────────────────────────────────────────────────
    @testset "@allocate macro with keys=" begin
        n = 50_000
        vals = AT(rand(Float32, n))
        keys_arr = AT(rand(UInt32, n))
        tmp_kv = KF.@allocate sort(vals; keys=keys_arr)
        @test :scratch_keys in propertynames(tmp_kv.arrays)
        @test eltype(tmp_kv.arrays.scratch_keys) === UInt32
        @test length(tmp_kv.arrays.scratch_keys) == n
        # And the buffer is usable end-to-end.
        sv, sk = KF.sort(vals; keys=keys_arr, tmp=tmp_kv)
        KA.synchronize(backend)
        @test issorted(Array(sk))
        # Value-only path still produces a buffer without scratch_keys.
        tmp_v = KF.@allocate sort(vals)
        @test !(:scratch_keys in propertynames(tmp_v.arrays))
    end


    # ─────────────────────────────────────────────────────────────────────
    # Pre-allocated buffer reuse.
    # ─────────────────────────────────────────────────────────────────────
    @testset "pre-allocated buffer reuse" begin
        n = 100_000
        VT, KT = Float32, UInt32
        vals     = AT(rand(VT, n))
        keys     = AT(rand(KT, n))
        dst      = AT(zeros(VT, n))
        keys_dst = AT(zeros(KT, n))
        tmp = KF.get_allocation(KF.Sort1D, vals; keys=keys)
        for trial in 1:5
            vals_cpu = rand(VT, n)
            keys_cpu = rand(KT, n)
            copyto!(vals, vals_cpu); copyto!(keys, keys_cpu)
            KF.sort!(dst, vals; keys=keys, keys_dst=keys_dst, tmp=tmp)
            KA.synchronize(backend)
            perm = sortperm(keys_cpu)
            @test Array(dst)      == vals_cpu[perm]
            @test Array(keys_dst) == keys_cpu[perm]
        end
    end
end
