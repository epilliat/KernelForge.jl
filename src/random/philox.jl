# Philox4x32-10 counter-based RNG core.
#
# Reference: Salmon, Moraes, Dror, Shaw — "Parallel random numbers: as easy
# as 1, 2, 3" (SC 2011).
#
# Stateless: output = F(key, ctr). All arithmetic in UInt32/UInt64. GPU-safe
# (no allocations, no BigInt). Bit-identical to Random123.Philox4x{UInt32,10}.

const _PHILOX_M0 = 0xD2511F53 % UInt32
const _PHILOX_M1 = 0xCD9E8D57 % UInt32
const _PHILOX_W0 = 0x9E3779B9 % UInt32   # golden ratio
const _PHILOX_W1 = 0xBB67AE85 % UInt32   # √3

@inline mulhi32(a::UInt32, b::UInt32) = (UInt32)((UInt64(a) * UInt64(b)) >> 32)
@inline mullo32(a::UInt32, b::UInt32) = (UInt32)((UInt64(a) * UInt64(b)) & 0xFFFFFFFF)

@inline function philox_round(c0::UInt32, c1::UInt32, c2::UInt32, c3::UInt32,
                              k0::UInt32, k1::UInt32)
    hi0 = mulhi32(_PHILOX_M0, c0)
    lo0 = mullo32(_PHILOX_M0, c0)
    hi1 = mulhi32(_PHILOX_M1, c2)
    lo1 = mullo32(_PHILOX_M1, c2)
    return (hi1 ⊻ c1 ⊻ k0,
            lo1,
            hi0 ⊻ c3 ⊻ k1,
            lo0)
end

"""
    philox4x32(key::NTuple{2,UInt32}, ctr::NTuple{4,UInt32}) -> NTuple{4,UInt32}

Single Philox4x32-10 round-block: 10 rounds with key-bumps.
Pure function; same `(key, ctr)` always yields the same 4-tuple output.
"""
@inline function philox4x32(key::NTuple{2,UInt32}, ctr::NTuple{4,UInt32})
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    Base.Cartesian.@nexprs 10 r -> begin
        c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
        k0 += _PHILOX_W0
        k1 += _PHILOX_W1
    end
    return (c0, c1, c2, c3)
end

"""
    philox_key(seed::UInt64) -> NTuple{2,UInt32}

Split a 64-bit seed into the (low32, high32) Philox key tuple.
"""
@inline philox_key(seed::UInt64) = (UInt32(seed & 0xFFFFFFFF), UInt32(seed >> 32))

"""
    philox_block(seed::UInt64, blk::UInt64) -> NTuple{4,UInt32}

Convenience: derive a key from `seed` and run Philox on the 128-bit counter
`(blk_lo, blk_hi, 0, 0)`. Returns 4 raw UInt32 outputs.
"""
@inline function philox_block(seed::UInt64, blk::UInt64)
    philox4x32(philox_key(seed),
               (UInt32(blk & 0xFFFFFFFF),
                UInt32(blk >> 32),
                UInt32(0), UInt32(0)))
end
