# Cross-op eltype matrix — guards the "odd-sized / non-primitive isbits eltype
# through a vectorized LOAD path" bug class.
#
# WHY THIS EXISTS: the A100 UnitFloat8→Float32 mapreduce cliff (commit 1c4d511)
# was a load-coalescing bug that the 480/480 suite did NOT catch, because no test
# ever pushed a non-primitive isbits eltype through an op's vload path. A primitive
# eltype never enters the widen branch, and a Type-capturing closure there fails to
# even launch (non-isbits arg) — so only a struct-eltype test can guard it. This is
# the gate any future `_maybe_widen` port (scan, matvec, …) must pass.
#
# Coverage = {copy (vcopy!), scan (scan!), mapreduce (mapreduce1d)} × a set of
# isbits structs spanning the widen predicate:
#   - pow2 sizeof {1,2,4,8}  → the `_widen_aggregate_load` fast path is ACTIVE
#   - non-pow2 sizeof 3      → widening is SKIPPED; the scalar path must still be correct

# ── test-local isbits eltypes (top-level so `struct` is legal) ──────────────

# 4-byte, 2-field, pow2  → widen path; supports +/zero so scan can accumulate it.
struct ELVec2
    a::Float16
    b::Float16
end
Base.:+(x::ELVec2, y::ELVec2) = ELVec2(x.a + y.a, x.b + y.b)
Base.zero(::Type{ELVec2}) = ELVec2(Float16(0), Float16(0))
Base.:(==)(x::ELVec2, y::ELVec2) = x.a == y.a && x.b == y.b

# 8-byte, 2-field, pow2  → widen path (64-bit unsigned reinterpret + field decomp).
struct ELVec2b
    a::Float32
    b::Float32
end
Base.:+(x::ELVec2b, y::ELVec2b) = ELVec2b(x.a + y.a, x.b + y.b)
Base.zero(::Type{ELVec2b}) = ELVec2b(0f0, 0f0)
Base.:(==)(x::ELVec2b, y::ELVec2b) = x.a == y.a && x.b == y.b

# 3-byte, 3-field, NON-pow2  → widening is skipped; exercises the scalar load path.
struct ELByte3
    x::UInt8
    y::UInt8
    z::UInt8
end
Base.:+(a::ELByte3, b::ELByte3) = ELByte3(a.x + b.x, a.y + b.y, a.z + b.z)  # wraps mod 256 (fine for the test)
Base.zero(::Type{ELByte3}) = ELByte3(0, 0, 0)
Base.:(==)(a::ELByte3, b::ELByte3) = a.x == b.x && a.y == b.y && a.z == b.z

@testset "cross-op eltype matrix (load-path bug class)" begin

    # ── copy: vcopy! must round-trip every isbits eltype at Nitem 1 and >1 ──
    @testset "vcopy! $T (Nitem=$ni)" for (T, mk) in (
            (ELVec2,  i -> ELVec2(Float16(i % 13), Float16(-(i % 7)))),
            (ELVec2b, i -> ELVec2b(Float32(i), Float32(-i))),
            (ELByte3, i -> ELByte3(i % 256, (2i) % 256, (3i) % 256)),
        ), ni in (1, 4)
        n = 100_003
        src = AT([mk(i) for i in 1:n])
        dst = AT([zero(T) for _ in 1:n])
        KF.vcopy!(dst, src; Nitem=ni)
        KA.synchronize(backend)
        @test Array(dst) == Array(src)
    end

    # ── scan: inclusive prefix-+ over a struct accumulator ──
    # NB: exact `==` is only valid where reordering the sum is exact — Float32
    # integer sums (< 2^24) and mod-256 UInt8 sums both qualify; Float16 does NOT
    # (order-dependent rounding), so ELVec2 is deliberately excluded here.
    @testset "scan! $T" for (T, mk) in (
            (ELVec2b, i -> ELVec2b(Float32(i % 11), Float32(i % 4))),
            (ELByte3, i -> ELByte3(i % 4, i % 3, i % 2)),
        )
        n = 50_001
        src_cpu = [mk(i) for i in 1:n]
        src = AT(src_cpu)
        dst = AT([zero(T) for _ in 1:n])
        KF.scan!(+, dst, src)
        KA.synchronize(backend)
        ref = accumulate(+, src_cpu)
        @test Array(dst) == ref
    end

    # ── mapreduce: struct→scalar map, +-reduce (the UnitFloat8 cliff's op) ──
    @testset "mapreduce1d $T" for (T, mk, mapf) in (
            (ELVec2,  i -> ELVec2(Float16(i % 7), Float16(-(i % 5))),
                      p -> Float32(p.a) + Float32(p.b)),
            (ELVec2b, i -> ELVec2b(Float32(i % 9), Float32(i % 6)),
                      p -> p.a + p.b),
            (ELByte3, i -> ELByte3(i % 256, i % 128, i % 64),
                      p -> Int(p.x) + Int(p.y) + Int(p.z)),
        )
        n = 50_000
        src_cpu = [mk(i) for i in 1:n]
        src = AT(src_cpu)
        res = KF.mapreduce1d(mapf, +, src; to_cpu=true)
        KA.synchronize(backend)
        ref = mapreduce(mapf, +, src_cpu)
        @test res ≈ ref rtol = 1e-3
    end

    # ── UnitFloat8: the original 1-byte single-field cliff, through mapreduce ──
    @testset "UnitFloat8 mapreduce (widen-load anomaly)" begin
        n = 50_000
        uf = AT([KF.UnitFloat8(Int8(((i * 37) % 255) - 127)) for i in 1:n])
        res = KF.mapreduce1d(x -> Float32(x), +, uf; to_cpu=true)
        KA.synchronize(backend)
        @test res ≈ sum(Float32.(Array(uf))) rtol = 1e-4
    end
end
