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