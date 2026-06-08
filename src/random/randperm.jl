# Public API for KernelForge.Random.randperm!.
#
# Composition only: fills a Float32 uniform key vector via the Round-5
# Philox path, then runs the package's stable keyval sortperm. The
# returned `perm` is a uniformly random permutation of 1..length(perm).
#
# Stability of the sort means colliding Float32 keys (prob. ≈ N²/2³²)
# resolve in input order — a vanishingly small bias at every practical N.

using KernelAbstractions

# `sortperm!` lives in the parent module; pull it in for use here.
import ..KernelForge: sortperm!

"""
    randperm!(perm::AbstractVector{<:Integer}, seed::UInt64;
              backend = get_backend(perm), workgroup = nothing,
              arch = nothing, keys = nothing) -> perm

Fill `perm` with a uniformly random permutation of `1:length(perm)`,
derived from `seed`. `perm` must have integer eltype.

Method: fill a Float32 scratch vector with `Uniform(0, 1)` samples via
[`rand!`](@ref), then call `KernelForge.sortperm!` — `perm` ends up
holding the permutation that would sort the random keys.

Pass `keys::AbstractVector{Float32}` (same backend, same length as
`perm`) to reuse a scratch buffer across calls — useful in hot loops.
Otherwise a fresh keys buffer is allocated.

Same `(seed, length(perm))` → same permutation on any backend.
"""
function randperm!(perm::AbstractVector{<:Integer}, seed::UInt64;
                   backend = get_backend(perm),
                   workgroup = nothing,
                   arch = nothing,
                   keys::Union{Nothing,AbstractVector{Float32}} = nothing)
    n = length(perm)
    n == 0 && return perm
    if keys === nothing
        keys = KernelAbstractions.allocate(backend, Float32, n)
    else
        length(keys) == n || error("`keys` must have the same length as `perm` (got $(length(keys)) vs $n)")
    end
    rand!(keys, Uniform(0f0, 1f0), seed; backend = backend, arch = arch)
    sortperm!(perm, keys; workgroup = workgroup, arch = arch)
    return perm
end
