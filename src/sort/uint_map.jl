# Order-preserving bijection T → unsigned-integer-of-same-width, matching
# Base.Sort.uint_map for ascending-order radix sort. The forward direction
# is enough for the sort kernels here (we sort the original T values, not
# the UInt representation, so no `unmap` is needed).
#
# Supported: Bool, UInt8/16/32/64, Int8/16/32/64, Float32, Float64.
# Wider/exotic types (Int128, BigInt, Float16) intentionally left out;
# the wrapper checks `Base.promote_op(uint_map ∘ by, T) <: Unsigned` and
# throws a clear error if a user passes something we don't handle.

@inline uint_map(x::Bool)    = UInt8(x)

@inline uint_map(x::UInt8)   = x
@inline uint_map(x::UInt16)  = x
@inline uint_map(x::UInt32)  = x
@inline uint_map(x::UInt64)  = x

@inline uint_map(x::Int8)    = reinterpret(UInt8,  x) ⊻ UInt8(0x80)
@inline uint_map(x::Int16)   = reinterpret(UInt16, x) ⊻ UInt16(0x8000)
@inline uint_map(x::Int32)   = reinterpret(UInt32, x) ⊻ UInt32(0x80000000)
@inline uint_map(x::Int64)   = reinterpret(UInt64, x) ⊻ UInt64(0x8000000000000000)

# Float bijection: positive numbers → flip sign bit (so they sort above
# negatives); negative numbers → flip every bit (so larger-magnitude
# negatives sort below smaller-magnitude negatives, matching IEEE total
# order under ascending). NaNs are not specially handled; they end up at
# whichever end the bit pattern falls.
@inline function uint_map(x::Float32)
    u = reinterpret(UInt32, x)
    ifelse((u & UInt32(0x80000000)) == 0, u | UInt32(0x80000000), ~u)
end
@inline function uint_map(x::Float64)
    u = reinterpret(UInt64, x)
    ifelse((u & UInt64(0x8000000000000000)) == 0, u | UInt64(0x8000000000000000), ~u)
end
