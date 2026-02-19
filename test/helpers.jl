# test_helpers.jl
# Shared structs and utilities used across test files.
# Include this once at the top of runtests.jl, before any other test files.

# ============================================================================
# Vec3
# ============================================================================

struct Vec3
    x::Float32
    y::Float32
    z::Float32
end

Base.:+(a::Vec3, b::Vec3) = Vec3(a.x + b.x, a.y + b.y, a.z + b.z)
Base.:*(a::Float32, b::Vec3) = Vec3(a * b.x, a * b.y, a * b.z)
Base.:*(a::Vec3, b::Float32) = Vec3(a.x * b, a.y * b, a.z * b)
Base.zero(::Type{Vec3}) = Vec3(0f0, 0f0, 0f0)
Base.isapprox(a::Vec3, b::Vec3; kwargs...) =
    isapprox(a.x, b.x; kwargs...) &&
    isapprox(a.y, b.y; kwargs...) &&
    isapprox(a.z, b.z; kwargs...)

# ============================================================================
# PriorityItem
# ============================================================================

struct PriorityItem
    priority::Int32
    cost::Float32
end

# rel(a, b) = true means "a is strictly better than b"
function better(a::PriorityItem, b::PriorityItem)
    a.priority > b.priority || (a.priority == b.priority && a.cost < b.cost)
end