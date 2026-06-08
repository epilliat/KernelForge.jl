# RandomAlgorithm — singleton tag types for `rand!` dispatch.
#
# The wrapper picks one via `default_random_algo(arch, Random1D, n, T)`
# (defined in rand.jl). The dispatched algorithm method is `_rand_dispatch!`
# (also in rand.jl), which selects between the standard SPP/NPC-parameterised
# kernel (fast at large N) and the persistent-thread kernel (fast at small N
# — measured in xp/philox/v18_persistent_kernel.jl).
#
# Singleton instances (not Symbols) so dispatch is type-stable and the
# GPU compiler can specialise per-algorithm at compile time. Same idiom
# as the GPU architecture types in src/architectures.jl.

abstract type RandomAlgorithm end

"Standard `Val{SPP, NPC}`-parameterised kernel: 1 thread per output block."
struct StandardKernel  <: RandomAlgorithm end

"""
Persistent-thread kernel: a fixed small number of threads
(= n_sms × blocks_per_sm × workgroup), each looping over its share of
output blocks. Wins when `cld(N, SPP)` would launch many more blocks
than the GPU can run concurrently — typically N ∈ [100k, 4M] on Ada.
"""
struct PersistentKernel <: RandomAlgorithm end

# ── runtime SM count ────────────────────────────────────────────────────
#
# Backends override this in the weak-dep extensions
# (ext/KernelForgeCUDAExt.jl, ext/KernelForgeAMDGPUExt.jl).
# Only consulted when `_rand_dispatch!(::PersistentKernel, ...)` is hit,
# which never happens for CPU arrays (the CPU dispatch always returns
# `StandardKernel`).
function _n_sms end
