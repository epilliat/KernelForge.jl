# Distributions for `KernelForge.Random.rand!`.
#
# Protocol (each distribution implements):
#
#   eltype(d) :: Type                            — output element type.
#   samples_per_philox(d) :: Int                 — 4, 2, or 1; selects ndrange.
#                                                  4 = 1 uniform per sample (Float32 dists).
#                                                  2 = 2 uniforms per sample (Normal{Float32}, Uniform{Float64}).
#                                                  1 = 4 uniforms per sample (Normal{Float64}, etc.).
#   sample_block(d, u4) :: NTuple{SPP, eltype(d)} — fast path: one Philox call → SPP samples.
#   sample_at(d, seed, i) :: eltype(d)           — per-index path: CPU + kernel tail.
#
# All routes for a given distribution must agree (modulo transcendental
# differences for Normal — see RESULTS.md).

import Adapt

abstract type AbstractDistribution{T} end

Base.eltype(::AbstractDistribution{T}) where {T} = T
Base.eltype(::Type{<:AbstractDistribution{T}}) where {T} = T

# Default: 4 samples per Philox call (Float32 1u-per-sample). Overrides
# below for 2-spp and 1-spp distributions.
@inline samples_per_philox(::AbstractDistribution) = 4

# ─────────────────────── Uniform[a, b] ──────────────────────────────────

"""
    Uniform([a, b]) <: AbstractDistribution{T}

Continuous Uniform(a, b) over the half-open interval `[a, b)`. Density
`1 / (b - a)` on `[a, b)`, zero elsewhere. `T` is promoted from
`a` and `b` and must be `<: AbstractFloat` (`Float32` or `Float64`).
Default constructor `Uniform()` yields `Uniform{Float32}(0, 1)`.
"""
struct Uniform{T<:AbstractFloat} <: AbstractDistribution{T}
    a::T
    b::T
end
Uniform() = Uniform(0f0, 1f0)
function Uniform(a::Real, b::Real)
    T = promote_type(typeof(float(a)), typeof(float(b)))
    Uniform{T}(T(a), T(b))
end

# Float32: 4 samples per Philox call (1 uniform each).
# Use `muladd` so CPU and GPU both compile to FMA (or both to mul+add)
# — keeps CPU↔GPU bit-equal even for non-trivial (a, b).
@inline samples_per_philox(::Uniform{Float32}) = 4
@inline _u32_to_uniform_f32(d::Uniform{Float32}, u::UInt32) =
    muladd(d.b - d.a, u32_to_f32(u), d.a)
@inline sample_block(d::Uniform{Float32}, u4::NTuple{4,UInt32}) =
    (_u32_to_uniform_f32(d, u4[1]), _u32_to_uniform_f32(d, u4[2]),
     _u32_to_uniform_f32(d, u4[3]), _u32_to_uniform_f32(d, u4[4]))
@inline sample_at(d::Uniform{Float32}, seed::UInt64, i::Integer) =
    muladd(d.b - d.a, uniform_f32_at(seed, i), d.a)

# Float64: 2 samples per Philox call (each sample consumes 2 UInt32 lanes).
@inline samples_per_philox(::Uniform{Float64}) = 2
@inline _pair_to_uniform_f64(d::Uniform{Float64}, lo::UInt32, hi::UInt32) =
    muladd(d.b - d.a, u64_to_f64((UInt64(hi) << 32) | UInt64(lo)), d.a)
@inline sample_block(d::Uniform{Float64}, u4::NTuple{4,UInt32}) =
    (_pair_to_uniform_f64(d, u4[1], u4[2]),
     _pair_to_uniform_f64(d, u4[3], u4[4]))
@inline sample_at(d::Uniform{Float64}, seed::UInt64, i::Integer) =
    muladd(d.b - d.a, uniform_f64_at(seed, i), d.a)

# ─────────────────────── Bernoulli(p) → Bool ────────────────────────────

"""
    Bernoulli(p::Real) <: AbstractDistribution{Bool}

Bernoulli(p) — emits `true` with probability `p`, `false` otherwise.
Samples are `Bool`-typed (use `eltype(::Bernoulli) == Bool` to allocate
the destination). `p` is stored as `Float32`; pass a `Float64` if
you need higher precision at the cost of one extra Philox lane per
sample.
"""
struct Bernoulli{T<:AbstractFloat} <: AbstractDistribution{Bool}
    p::T
end
Bernoulli(p::Real) = Bernoulli{Float32}(Float32(p))

@inline samples_per_philox(::Bernoulli) = 4
@inline _u32_to_bernoulli(d::Bernoulli{Float32}, u::UInt32) = u32_to_f32(u) < d.p
@inline sample_block(d::Bernoulli{Float32}, u4::NTuple{4,UInt32}) =
    (_u32_to_bernoulli(d, u4[1]), _u32_to_bernoulli(d, u4[2]),
     _u32_to_bernoulli(d, u4[3]), _u32_to_bernoulli(d, u4[4]))
@inline sample_at(d::Bernoulli{Float32}, seed::UInt64, i::Integer) =
    uniform_f32_at(seed, i) < d.p

# ─────────────────────── Exponential(λ) ─────────────────────────────────

"""
    Exponential(λ::Real) <: AbstractDistribution{T}

Exponential(λ) — density `λ · exp(-λx)` for `x ≥ 0`. Sampled via the
inverse-CDF `-log(1 - u) / λ` (the `1 - u` form avoids `log(0)` when
the uniform draw rounds down to exactly zero). `λ` stored as `Float32`.
"""
struct Exponential{T<:AbstractFloat} <: AbstractDistribution{T}
    λ::T
end
Exponential(λ::Real) = Exponential{Float32}(Float32(λ))

@inline samples_per_philox(::Exponential{Float32}) = 4
# Use 1 - u to avoid log(0) when u rounds down to 0.
@inline _u32_to_expo(d::Exponential{Float32}, u::UInt32) = -log(1f0 - u32_to_f32(u)) / d.λ
@inline sample_block(d::Exponential{Float32}, u4::NTuple{4,UInt32}) =
    (_u32_to_expo(d, u4[1]), _u32_to_expo(d, u4[2]),
     _u32_to_expo(d, u4[3]), _u32_to_expo(d, u4[4]))
@inline sample_at(d::Exponential{Float32}, seed::UInt64, i::Integer) =
    -log(1f0 - uniform_f32_at(seed, i)) / d.λ

# ─────────────────────── Categorical(cum_p) ─────────────────────────────
# `cum_p` is the *cumulative* distribution (last entry must be 1.0). Linear
# scan with branchless predication — fine for K ≲ 32. The struct holds
# the array reference; Adapt recurses into it so a CuArray-backed
# Categorical works inside the kernel.

"""
    Categorical(cum_p) <: AbstractDistribution{Int32}

Categorical distribution over `1:length(cum_p)`. `cum_p` is the
**cumulative** distribution — `cum_p[k] = P(X ≤ k)`, must be
non-decreasing, and `cum_p[end] == 1.0`. Samples by linear search
with branchless predication (best for `K ≲ 32`). `cum_p` may be a
host or device array; Adapt recurses into it so a `CuArray`-backed
Categorical works inside the kernel.
"""
struct Categorical{V<:AbstractVector{Float32}} <: AbstractDistribution{Int32}
    cum_p::V
end

# Recurse Adapt into the inner array so KA can convert
# Categorical{CuArray} → Categorical{CuDeviceArray} at kernel-launch time.
Adapt.adapt_structure(to, c::Categorical) = Categorical(Adapt.adapt(to, c.cum_p))

@inline samples_per_philox(::Categorical) = 4

@inline function _searchcdf(cum_p::AbstractVector{Float32}, u::Float32)
    k = Int32(1)
    @inbounds for j in eachindex(cum_p)
        k += Int32(u >= cum_p[j])
    end
    return k
end

@inline _u32_to_cat(d::Categorical, u::UInt32) = _searchcdf(d.cum_p, u32_to_f32(u))
@inline sample_block(d::Categorical, u4::NTuple{4,UInt32}) =
    (_u32_to_cat(d, u4[1]), _u32_to_cat(d, u4[2]),
     _u32_to_cat(d, u4[3]), _u32_to_cat(d, u4[4]))
@inline sample_at(d::Categorical, seed::UInt64, i::Integer) =
    _searchcdf(d.cum_p, uniform_f32_at(seed, i))

# ─────────────────────── Normal(μ, σ) ──────────────────────────────────

"""
    Normal([μ, σ]) <: AbstractDistribution{T}

Normal(μ, σ²) — density `exp(-(x-μ)²/(2σ²)) / (σ√(2π))`. Sampled via
Box–Muller: each `(u₁, u₂)` Philox pair yields TWO normals (cos- and
sin-halves), so a single Philox call produces 4 `Normal{Float32}`
samples — +30% throughput vs the cos-only variant at large N.
`T` is promoted from `μ` and `σ`. Default `Normal()` yields
`Normal{Float32}(0, 1)`.
"""
struct Normal{T<:AbstractFloat} <: AbstractDistribution{T}
    μ::T
    σ::T
end
Normal() = Normal(0f0, 1f0)
function Normal(μ::Real, σ::Real)
    T = promote_type(typeof(float(μ)), typeof(float(σ)))
    Normal{T}(T(μ), T(σ))
end

# Float32: 4 samples per Philox call. Box–Muller naturally produces 2
# normals per (u₁, u₂) pair (cos AND sin halves) — using both doubles
# samples per Philox call vs the naive cos-only variant. xp/philox/v10
# measured +30% throughput at large N.
@inline samples_per_philox(::Normal{Float32}) = 4

# Returns (cos-half, sin-half) of one Box–Muller pair.
@inline function _box_muller_pair_f32(μ::Float32, σ::Float32, u1::UInt32, u2::UInt32)
    f1 = max(u32_to_f32(u1), 1.0f-38)         # avoid log(0)
    f2 = u32_to_f32(u2)
    r  = sqrt(-2f0 * log(f1))
    θ  = 2f0 * Float32(π) * f2
    s, c = sincos(θ)                          # one __sincosf on CUDA
    return (μ + σ * r * c, μ + σ * r * s)
end

@inline function sample_block(d::Normal{Float32}, u4::NTuple{4,UInt32})
    n1, n2 = _box_muller_pair_f32(d.μ, d.σ, u4[1], u4[2])
    n3, n4 = _box_muller_pair_f32(d.μ, d.σ, u4[3], u4[4])
    return (n1, n2, n3, n4)
end

# Per-index path. Element i: pair k = (i-1) ÷ 2, lane = (i-1) mod 2 (0 = cos, 1 = sin).
@inline function sample_at(d::Normal{Float32}, seed::UInt64, i::Integer)
    k = (UInt64(i) - UInt64(1)) >> 1
    lane = (UInt64(i) - UInt64(1)) & UInt64(1)
    j = 2*Int(k) + 1
    n_cos, n_sin = _box_muller_pair_f32(d.μ, d.σ,
                                        uniform_u32_at(seed, j),
                                        uniform_u32_at(seed, j+1))
    return lane == 0 ? n_cos : n_sin
end
