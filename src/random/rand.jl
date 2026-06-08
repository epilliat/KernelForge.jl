# Public API for KernelForge.Random.
#
# Element `out[i]` is a deterministic function of `(seed, i)` — independent
# of vector length, parallel decomposition, or device. Same seed →
# byte-identical output across runs and across CPU/GPU backends.

# Pull in package-level dispatch infrastructure (algorithm tag, arch types,
# default_nitem). All `default_nitem` methods below add to the same
# `KernelForge.default_nitem` generic so the rest of the package can also
# dispatch on `Random1D` if needed.
import ..KernelForge: default_nitem, detect_arch, AbstractArch, CUDAArch, AMDArch,
                       Ada, Ampere, Hopper, Blackwell, Turing, Volta, RTX1000,
                       Random1D

# Default workgroup. Single value picked by xp/philox/v7 measurement
# (256 wins on every problem size on RTX 1000 Ada). Per-arch tuning is
# deferred — wire into `KernelForge.default_workgroup(arch, …)` if/when
# AMD numbers warrant it.
const _DEFAULT_WORKGROUP = 256

# ── default_nitem dispatch — total samples per thread ────────────────────
#
# Convention mirrors src/scan/scan.jl: returns "samples per thread", so
# bytes-per-thread = Nitem × sizeof(T). The wrapper computes
# NPC = max(1, Nitem ÷ SPP) where SPP comes from samples_per_philox(dist).
#
# Generic fallback: 4 samples / thread = 1 Philox call for SPP=4 dists
# (matches round-2 behaviour exactly).

@inline default_nitem(::AbstractArch, ::Type{Random1D}, n, ::Type{T}) where {T} =
    cld(16, sizeof(T))

# CPU / no-arch fallback (detect_arch isn't defined for CPU arrays).
@inline default_nitem(::Nothing, ::Type{Random1D}, n, ::Type{T}) where {T} =
    cld(16, sizeof(T))

# CPU detection: return `nothing` so the default_nitem dispatch falls through.
@inline _maybe_detect_arch(::CPU, _out) = nothing
@inline _maybe_detect_arch(_backend, out) = detect_arch(out)

# NVIDIA Ada (RTX 1000, RTX 4090, etc.) — measured per-Nitem on RTX 1000.
#
# Findings (Float32 Uniform): Nitem=16 hurts at N ≥ 100k due to register
# pressure (4 Philox calls / thread → ~40 live registers → occupancy
# drops below 50 %). Nitem=8 gives a small win at N = 1M; otherwise
# Nitem=4 (NPC=1) is at or near the optimum across all sizes.
#
# Default: gentle amortization at very small N, bandwidth-optimal at large N.
@inline function default_nitem(::Ada, ::Type{Random1D}, n, ::Type{T}) where {T}
    base = cld(16, sizeof(T))
    if n < 100_000
        2 * base       # e.g. Float32 → 8 samples / thread
    else
        base           # e.g. Float32 → 4 samples / thread (round-2 default)
    end
end

# AMDArch placeholder — falls through to AbstractArch until measured on an MI box.

# ── default_random_algo dispatch — pick StandardKernel vs PersistentKernel ──
#
# Generic / unknown arch: never use persistent (unmeasured). The
# StandardKernel path is always correct; PersistentKernel is purely an
# optimisation. Same conservatism as default_nitem.

@inline default_random_algo(::Any, ::Type{Random1D}, n, ::Type{T}) where {T} =
    StandardKernel()

# Ada — measured in xp/philox/v18_persistent_kernel.jl.
# Persistent wins between N = 100k and ~4M:
#   - below 100k: kernel is so fast that host overhead dominates; persistent
#     can't help.
#   - above ~4M: standard kernel saturates the GPU naturally; the persistent
#     kernel's loop overhead causes a small regression.
# The 100_000:4_000_000 band is for Float32 SPP=4. For larger T (Float64),
# the same byte budget covers half as many samples, so the band shifts.
@inline function default_random_algo(::Ada, ::Type{Random1D}, n, ::Type{T}) where {T}
    # Scale the upper bound by 1/sizeof(T) so wider types crossover earlier
    # (a 4M-element Float64 fill = 32 MB, same DRAM pressure as 8M Float32).
    hi = 4_000_000 ÷ max(1, sizeof(T) ÷ 4)
    100_000 <= n <= hi ? PersistentKernel() : StandardKernel()
end

# ── default_persistent_blocks_per_sm — persistent kernel tuning knob ──
#
# Threads launched = n_sms × blocks_per_sm × workgroup. blocks_per_sm
# controls occupancy: 1 = 12.5%, 4 = 50%, 8 = 100% on Ada (with 256-thread
# workgroups and 2048 threads/SM cap).
#
# Measured (xp/v18): 4 wins on Ada, 8 is essentially tied, 1 loses ~10%.

@inline default_persistent_blocks_per_sm(::Any, ::Type{Random1D}, ::Type{T}) where {T} = 4
@inline default_persistent_blocks_per_sm(::Ada, ::Type{Random1D}, ::Type{T}) where {T} = 4

# ── public rand! ────────────────────────────────────────────────────────

"""
    rand!(out::AbstractVector, dist::AbstractDistribution, seed::UInt64;
          backend = get_backend(out), workgroup = 256,
          arch = nothing, Nitem = nothing, algo = nothing)

Fill `out` with i.i.d. samples from `dist`, derived from `(seed, i)`.

`out[i]` is a pure function of `(seed, i)`, so the result is byte-identical
across runs, vector lengths, parallel decompositions, devices, AND the
choice of algorithm/Nitem.

Auto-tuned per architecture:
  - `algo`  — `default_random_algo(arch, Random1D, N, T)` picks
              `StandardKernel()` or `PersistentKernel()`.
  - `Nitem` — `default_nitem(arch, Random1D, N, T)`, samples per thread
              for the standard kernel (ignored by the persistent kernel).

Override either for benchmarking; output is invariant.
"""
function rand!(out::AbstractVector, dist::AbstractDistribution, seed::UInt64;
               backend = get_backend(out),
               workgroup = _DEFAULT_WORKGROUP,
               arch = nothing,
               Nitem = nothing,
               algo = nothing)
    N = length(out)
    N == 0 && return out
    # detect_arch only works on GPU arrays; on CPU, skip it and let
    # default_* methods dispatch on `Nothing` → AbstractArch fallback.
    arch = arch === nothing ? _maybe_detect_arch(backend, out) : arch
    algo = something(algo, default_random_algo(arch, Random1D, N, eltype(out)))
    return _rand_dispatch!(algo, out, dist, seed,
                           backend, workgroup, arch, Nitem, N)
end

# ── per-algorithm launchers ─────────────────────────────────────────────

# StandardKernel: Val{SPP, NPC}-parameterised. Round-3 production path.
function _rand_dispatch!(::StandardKernel, out::AbstractVector, dist, seed::UInt64,
                         backend, workgroup::Int, arch, Nitem, N::Int)
    SPP   = samples_per_philox(dist)
    Nitem = something(Nitem, default_nitem(arch, Random1D, N, eltype(out)))
    NPC   = max(1, Nitem ÷ SPP)              # Philox calls per thread
    nthreads = cld(N, SPP * NPC)
    _rand_kernel!(backend, workgroup)(out, dist, seed,
                                      Val(SPP), Val(NPC), N;
                                      ndrange = nthreads)
    return out
end

# PersistentKernel: launches a fixed small thread pool, kernel loops over
# the output. Wins at small N (100k–4M on Ada). The `Nitem` keyword is
# accepted but ignored — the persistent kernel always emits SPP samples
# per iteration. blocks_per_sm comes from `default_persistent_blocks_per_sm`.
function _rand_dispatch!(::PersistentKernel, out::AbstractVector, dist, seed::UInt64,
                         backend, workgroup::Int, arch, Nitem, N::Int)
    SPP  = samples_per_philox(dist)
    bpsm = default_persistent_blocks_per_sm(arch, Random1D, eltype(out))
    nthreads = _n_sms(backend) * bpsm * workgroup
    _rand_persistent_kernel!(backend, workgroup)(out, dist, seed, Val(SPP), N;
                                                 ndrange = nthreads)
    return out
end

# ── convenience shortcuts ────────────────────────────────────────────────

"""
    rand!(out::AbstractVector{<:AbstractFloat}, seed::UInt64) = rand!(out, Uniform(0, 1), seed)
"""
rand!(out::AbstractVector{Float32}, seed::UInt64; kw...) =
    rand!(out, Uniform(0f0, 1f0), seed; kw...)
rand!(out::AbstractVector{Float64}, seed::UInt64; kw...) =
    rand!(out, Uniform(0.0, 1.0), seed; kw...)

"""
    randn!(out::AbstractVector{<:AbstractFloat}, seed::UInt64) = rand!(out, Normal(0, 1), seed)
"""
randn!(out::AbstractVector{Float32}, seed::UInt64; kw...) =
    rand!(out, Normal(0f0, 1f0), seed; kw...)
