# Uniform-distribution primitives built on the Philox core.
#
# Two layers:
#   1. Bit-trick conversions UInt32 → Float32 / UInt64 → Float64. Top
#      mantissa bits become a number in [1, 2); subtract 1 → [0, 1).
#      Identical between CPU and GPU; faster than `Float32(u) * 2^-32`.
#   2. Per-index pickers `uniform_<T>_at(seed, i)` — pure functions of
#      `(seed, i)` that pick element i of the implicit uniform stream.
#      Used by the CPU reference path and by tail handling in the kernels.

# ── bit-trick conversions ─────────────────────────────────────────────────
# Mathematically: take the top sizeof(T)*8 - mantissabits leading bits as
# the mantissa of a float in [1, 2), then subtract 1.

@inline u32_to_f32(u::UInt32) =
    reinterpret(Float32, (u >> 9) | 0x3f800000) - 1f0

@inline u64_to_f64(u::UInt64) =
    reinterpret(Float64, (u >> 12) | 0x3ff0000000000000) - 1.0

# ── per-index pickers ────────────────────────────────────────────────────

# Element i of an implicit UInt32 uniform stream. One Philox call per 4
# elements; element i comes from lane `(i-1) % 4` of block `(i-1) ÷ 4`.
@inline function uniform_u32_at(seed::UInt64, i::Integer)
    j = UInt64(i) - UInt64(1)
    blk = j >> 2
    lane = (j & UInt64(3)) % Int + 1
    @inbounds philox_block(seed, blk)[lane]
end

# Element i of a UInt64 stream — combine two consecutive UInt32 lanes from
# the same Philox call. block = (i-1) ÷ 2; lane pair = (2k, 2k+1).
@inline function uniform_u64_at(seed::UInt64, i::Integer)
    j = UInt64(i) - UInt64(1)
    blk = j >> 1
    lane = (j & UInt64(1)) % Int     # 0 or 1
    u4 = philox_block(seed, blk)
    @inbounds lo = u4[2*lane + 1]
    @inbounds hi = u4[2*lane + 2]
    return (UInt64(hi) << 32) | UInt64(lo)
end

@inline uniform_f32_at(seed::UInt64, i::Integer) = u32_to_f32(uniform_u32_at(seed, i))
@inline uniform_f64_at(seed::UInt64, i::Integer) = u64_to_f64(uniform_u64_at(seed, i))
