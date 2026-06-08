# Philox4x32-10 core: KAT, allocation-free, Random123 cross-validation.
#
# Tests run on the CPU (the core is host-side); the GPU lowering is exercised
# in uniforms_test.jl and distributions_test.jl via CPU↔GPU bit-equality.

using Random123

const KFR = KernelForge.Random
const Z = UInt32(0)

@testset "philox core — KAT" begin
    # Random123-locked reference for (key=0, ctr=0).
    @test KFR.philox4x32((Z, Z), (Z, Z, Z, Z)) ===
          (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)
end

@testset "philox core — Random123 cross-check" begin
    # 256 sequential counters at key=(0,0); every UInt32 must match.
    r = Philox4x(UInt32, (UInt64(0), UInt64(0)), 10)
    for blk in 0:255
        ours = KFR.philox4x32((Z, Z), (UInt32(blk), Z, Z, Z))
        ref  = (rand(r, UInt32), rand(r, UInt32),
                rand(r, UInt32), rand(r, UInt32))
        @test ours === ref
    end

    # Non-trivial key.
    seed = UInt64(0xCAFEBABEDEADBEEF)
    k = KFR.philox_key(seed)
    r2 = Philox4x(UInt32, (UInt64(k[1]), UInt64(k[2])), 10)
    for blk in 0:63
        ours = KFR.philox4x32(k, (UInt32(blk), Z, Z, Z))
        ref  = (rand(r2, UInt32), rand(r2, UInt32),
                rand(r2, UInt32), rand(r2, UInt32))
        @test ours === ref
    end
end

@testset "philox core — allocation-free hot path" begin
    # Warm up.
    KFR.philox4x32((Z, Z), (Z, Z, Z, Z))
    allocs = @allocated KFR.philox4x32(
        (UInt32(1), UInt32(2)),
        (UInt32(3), UInt32(4), UInt32(5), UInt32(6)),
    )
    @test allocs == 0
end
