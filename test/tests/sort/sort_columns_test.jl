# Backend-portable tests for KernelForge.sort_columns / sort_columns!.
#
# Verifies the dispatcher routes correctly (custom lt → OEM, uint_map types
# below threshold → OEM, above threshold → batched radix) and that every
# eltype × shape combination produces a per-column-sorted result.

@testset "KernelForge.sort_columns! per-dtype × per-shape" begin
    test_types = [UInt8, UInt16, UInt32, UInt64,
                  Int8,  Int16,  Int32,  Int64,
                  Float32, Float64]

    # Mix of pow-of-2 and non-pow-of-2 K; small/medium/large M.
    test_shapes = [
        (1,    16),       # degenerate K=1
        (64,   256),      # OEM warp regime
        (1000, 64),       # non-pow2, mid-OEM
        (4096, 64),       # OEM/radix boundary
        (8192, 32),       # large-K radix
        (12345, 16),      # large-K, non-pow2
    ]

    for T in test_types
        for (K, M) in test_shapes
            @testset "T=$T K=$K M=$M" begin
                # Float64 single-block path tops out at K_PAD=2048; the
                # dispatcher then routes K=4096..8192 Float64 to the
                # large-K path. All K should still work end-to-end.
                src_cpu = T <: AbstractFloat ? randn(T, K, M) : rand(T, K, M)
                src = AT(src_cpu)

                expected = hcat([sort(src_cpu[:, j]) for j in 1:M]...)

                # In-place form
                A = copy(src)
                KF.sort_columns!(A)
                KA.synchronize(backend)
                @test Array(A) == expected

                # Allocating form
                B = KF.sort_columns(src)
                KA.synchronize(backend)
                @test Array(B) == expected
                # Source unchanged
                @test Array(src) == src_cpu
            end
        end
    end
end

@testset "KernelForge.sort_columns! algorithm=:radix forced" begin
    for T in (UInt32, Float32, Int64)
        K, M = 8192, 32
        src_cpu = T <: AbstractFloat ? randn(T, K, M) : rand(T, K, M)
        src = AT(src_cpu)
        expected = hcat([sort(src_cpu[:, j]) for j in 1:M]...)

        A = copy(src)
        KF.sort_columns!(A; algorithm = :radix)
        KA.synchronize(backend)
        @test Array(A) == expected
    end
end

@testset "KernelForge.sort_columns! algorithm=:oem forced (K ≤ 4096)" begin
    for T in (UInt32, Float32, Float64, Int8)
        K, M = 1024, 256
        src_cpu = T <: AbstractFloat ? randn(T, K, M) : rand(T, K, M)
        src = AT(src_cpu)
        expected = hcat([sort(src_cpu[:, j]) for j in 1:M]...)

        A = copy(src)
        KF.sort_columns!(A; algorithm = :oem)
        KA.synchronize(backend)
        @test Array(A) == expected
    end

    # OEM rejects K > 4096.
    A = AT(rand(UInt32, 8192, 16))
    @test_throws ErrorException KF.sort_columns!(A; algorithm = :oem)
end

@testset "KernelForge.sort_columns! custom lt routes to OEM" begin
    # Descending sort: custom `lt = >` forces the OEM path. K ≤ 4096.
    for T in (UInt32, Float32, Int64)
        K, M = 1024, 64
        src_cpu = T <: AbstractFloat ? randn(T, K, M) : rand(T, K, M)
        src = AT(src_cpu)
        expected = hcat([sort(src_cpu[:, j]; lt = >) for j in 1:M]...)

        A = copy(src)
        KF.sort_columns!(A; lt = >)
        KA.synchronize(backend)
        @test Array(A) == expected
    end

    # Custom lt + K > 4096 must error (no fallback in v1).
    A = AT(rand(UInt32, 8192, 16))
    @test_throws ErrorException KF.sort_columns!(A; lt = >)
end

@testset "KernelForge.sort_columns! workspace reuse" begin
    K, M = 12345, 16   # large-K + non-pow2 (forces tail bounds)
    src_cpu = rand(UInt32, K, M)
    src = AT(src_cpu)
    expected = hcat([sort(src_cpu[:, j]) for j in 1:M]...)

    A = copy(src)
    tmp = KF.get_allocation(KF.SortColumns, A)

    # First call.
    KF.sort_columns!(A; tmp)
    KA.synchronize(backend)
    @test Array(A) == expected

    # Reuse workspace on a fresh copy.
    copyto!(A, src)
    KF.sort_columns!(A; tmp)
    KA.synchronize(backend)
    @test Array(A) == expected
end

# A custom bitstype with `uint_map` defined but *no* `typemax`. The
# dispatcher must route around the small-K path (which would error on
# `typemax(MyKey)`): K ≤ 4096 → OEM-with-tag, K > 4096 → batched-radix
# large-K (bounds-checked, no sentinel).
if !@isdefined(NoTypemaxKey)
    primitive type NoTypemaxKey 32 end
    NoTypemaxKey(x::UInt32) = reinterpret(NoTypemaxKey, x)
    KF.uint_map(x::NoTypemaxKey) = reinterpret(UInt32, x)
    Base.isless(a::NoTypemaxKey, b::NoTypemaxKey) =
        reinterpret(UInt32, a) < reinterpret(UInt32, b)
end

@testset "KernelForge.sort_columns! handles missing typemax" begin
    @test !hasmethod(typemax, Tuple{Type{NoTypemaxKey}})

    # No-typemax T must work at every K via batched radix (bounds-checked,
    # no sentinel). Exercise small-K (single-block) and large-K (multi-block).
    for (K, M) in ((1024, 64), (4096, 16), (5000, 32), (8192, 32), (12345, 8))
        src_cpu = NoTypemaxKey.(rand(UInt32, K, M))
        src = AT(src_cpu)
        expected_u32 = hcat([sort(reinterpret(UInt32, src_cpu[:, j])) for j in 1:M]...)
        A = copy(src)
        KF.sort_columns!(A; algorithm = :radix)
        KA.synchronize(backend)
        @test reinterpret(UInt32, Array(A)) == expected_u32
    end
end
