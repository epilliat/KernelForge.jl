"""
    format_number(n::Int) -> String

Format large integers with underscores for readability (e.g. 1_000_000).
"""
function format_number(n::Int)
    s = string(n)
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    pushfirst!(parts, s)
    return join(parts, "_")
end

"""
    format_3digits(x::Real) -> String

Format a number with exactly 3 significant digits.
"""
function format_3digits(x::Real)
    x == 0 && return "0"
    mag = floor(Int, log10(abs(x)))
    decimals = max(0, 2 - mag)
    return @sprintf("%.*f", decimals, x)
end

"""
    format_1digit(x::Real) -> String

Format a number with exactly 1 significant digit (e.g. `3.6 → 4`,
`0.0291 → 0.03`). Useful for above-bar annotations when you only want
a coarse comparison and not exact values.
"""
function format_1digit(x::Real)
    x == 0 && return "0"
    mag = floor(Int, log10(abs(x)))
    decimals = max(0, -mag)
    return @sprintf("%.*f", decimals, x)
end

"""
    format_2digits(x::Real) -> String

Format a number with exactly 2 significant digits (e.g. `3.59 → 3.6`,
`0.0291 → 0.029`, `12.3 → 12`).
"""
function format_2digits(x::Real)
    x == 0 && return "0"
    mag = floor(Int, log10(abs(x)))
    decimals = max(0, 1 - mag)
    return @sprintf("%.*f", decimals, x)
end

"""
    format_compact(x::Real) -> String

Compact formatter that turns large values into k / M / G suffixed
short strings (e.g. `6_597_000 → "6.6M"`, `9342 → "9.3k"`) while
keeping small values 2-digit (`123 → "123"`, `0.45 → "0.45"`).
Designed for above-bar labels where horizontal space is tight.
"""
function format_compact(x::Real)
    ax = abs(x)
    ax == 0 && return "0"
    if ax >= 1e9
        v = x / 1e9; return @sprintf("%.*fG", v >= 10 ? 0 : 1, v)
    elseif ax >= 1e6
        v = x / 1e6; return @sprintf("%.*fM", v >= 10 ? 0 : 1, v)
    elseif ax >= 1e3
        v = x / 1e3; return @sprintf("%.*fk", v >= 10 ? 0 : 1, v)
    else
        return format_2digits(x)
    end
end