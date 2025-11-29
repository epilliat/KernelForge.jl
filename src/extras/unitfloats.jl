"""
    UnitFloat8 (UF8)

An 8-bit signed fractional type representing values in (-1, 1).
Uses Int8 directly with range [-127/128, 127/128]:
- 0x00 = 0.0
- 0x7F (127) ≈ +0.992
- 0x81 (-127) ≈ -0.992
- 0x80 (-128) is reserved/unused to keep range symmetric

All values strictly within (-1, 1).
"""
struct UnitFloat8 <: AbstractFloat
    val::Int8
end

const NaN_UF8 = UnitFloat8(Int8(-128))

Base.rand(::Type{UnitFloat8}) = UnitFloat8(rand(Int8(-127):Int8(127)))

@inline function UnitFloat8(x::Float64)
    x_clamped = clamp(x, -127.0 / 128.0, 127.0 / 128.0)
    # Scale to [-127, 127]
    scaled = round(Int8, x_clamped * 128.0)
    # Clamp to avoid -128
    clamped_scaled = clamp(scaled, Int8(-127), Int8(127))
    return UnitFloat8(clamped_scaled)
end

@inline function UnitFloat8(x::Float32)
    x_clamped = clamp(x, -127.0f0 / 128.0f0, 127.0f0 / 128.0f0)
    scaled = round(Int8, x_clamped * 128.0f0)
    clamped_scaled = clamp(scaled, Int8(-127), Int8(127))
    return UnitFloat8(clamped_scaled)
end

# Convert to Float
@inline function Base.Float64(x::UnitFloat8)
    return Float64(x.val) / 128.0
end

@inline function Base.Float32(x::UnitFloat8)
    return Float32(x.val) / 128.0f0
end

@inline Base.Float16(x::UnitFloat8) = Float16(Float32(x))
@inline Base.Int8(x::UnitFloat8) = x.val

# Show
Base.show(io::IO, x::UnitFloat8) = print(io, "UF8(", Float32(x), ")")

# Constants
@inline Base.zero(::Type{UnitFloat8}) = UnitFloat8(Int8(0))
@inline Base.one(::Type{UnitFloat8}) = UnitFloat8(Int8(127))
@inline Base.typemin(::Type{UnitFloat8}) = UnitFloat8(Int8(-127))
@inline Base.typemax(::Type{UnitFloat8}) = UnitFloat8(Int8(127))

# Negation
@inline function Base.:-(x::UnitFloat8)
    # Simple negation, -128 becomes 127 (clamped)
    negated = -x.val
    clamped = clamp(negated, Int8(-127), Int8(127))
    return UnitFloat8(clamped)
end

# Addition with saturation
@inline function Base.:+(x::UnitFloat8, y::UnitFloat8)
    x_i16 = Int16(x.val)
    y_i16 = Int16(y.val)

    result = x_i16 + y_i16

    # Saturate to [-127, 127]
    saturated = clamp(result, Int16(-127), Int16(127))

    return UnitFloat8(Int8(saturated))
end

# Subtraction with saturation
@inline function Base.:-(x::UnitFloat8, y::UnitFloat8)
    x_i16 = Int16(x.val)
    y_i16 = Int16(y.val)

    result = x_i16 - y_i16
    saturated = clamp(result, Int16(-127), Int16(127))

    return UnitFloat8(Int8(saturated))
end

# Multiplication (fixed-point)
@inline function Base.:*(x::UnitFloat8, y::UnitFloat8)
    x_i16 = Int16(x.val)
    y_i16 = Int16(y.val)

    # (x/128) * (y/128) * 128 = (x * y) / 128
    product = x_i16 * y_i16
    result = product >> 7  # Divide by 128

    # Saturate to [-127, 127]
    saturated = clamp(result, Int16(-127), Int16(127))

    return UnitFloat8(Int8(saturated))
end

# Division
@inline function Base.:/(x::UnitFloat8, y::UnitFloat8)
    y_i8 = y.val

    # Handle division by zero
    if y_i8 == 0
        return x.val >= 0 ? typemax(UnitFloat8) : typemin(UnitFloat8)
    end

    x_i16 = Int16(x.val)
    # (x/128) / (y/128) * 128 = (x * 128) / y
    result = (x_i16 << 7) ÷ Int16(y_i8)

    # Saturate to [-127, 127]
    saturated = clamp(result, Int16(-127), Int16(127))

    return UnitFloat8(Int8(saturated))
end

# Comparisons
@inline Base.:(==)(x::UnitFloat8, y::UnitFloat8) = x.val == y.val
@inline Base.:<(x::UnitFloat8, y::UnitFloat8) = x.val < y.val
@inline Base.:<=(x::UnitFloat8, y::UnitFloat8) = x.val <= y.val
@inline Base.isless(x::UnitFloat8, y::UnitFloat8) = x.val < y.val

# Absolute value
@inline function Base.abs(x::UnitFloat8)
    abs_val = abs(x.val)
    # Clamp to 127 in case of -128 input
    clamped = min(abs_val, Int8(127))
    return UnitFloat8(clamped)
end

# Sign
@inline function Base.sign(x::UnitFloat8)
    if x.val > 0
        return one(UnitFloat8)
    elseif x.val < 0
        return -one(UnitFloat8)
    else
        return zero(UnitFloat8)
    end
end

# Promotion rules
Base.promote_rule(::Type{UnitFloat8}, ::Type{Float64}) = Float64
Base.promote_rule(::Type{UnitFloat8}, ::Type{Float32}) = Float32
Base.promote_rule(::Type{UnitFloat8}, ::Type{Float16}) = Float16

# Useful functions
@inline Base.eps(::Type{UnitFloat8}) = 1.0 / 128.0