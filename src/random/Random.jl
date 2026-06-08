"""
    KernelForge.Random

Counter-based RNG (Philox4x32-10) for GPU/CPU via KernelAbstractions.

Element `out[i]` of every fill is a deterministic function of `(seed, i)`,
so results are byte-identical across runs, vector lengths, parallel
decompositions, and devices (CPU ↔ CUDA ↔ AMDGPU).

# Quick start

```julia
using KernelForge.Random
import KernelForge.Random as KFR

out = CUDA.zeros(Float32, 1_000_000)
KFR.rand!(out, UInt64(42))                       # Uniform[0,1)
KFR.randn!(out, UInt64(42))                      # Normal(0, 1)
KFR.rand!(out, KFR.Uniform(-1f0, 1f0), UInt64(42))
KFR.rand!(out, KFR.Exponential(2f0), UInt64(42))
KFR.rand!(b, KFR.Bernoulli(0.3f0), UInt64(42))   # b::AbstractVector{Bool}
```

# Adding distributions

Define one struct + three @inline methods:

```julia
struct MyDist <: KernelForge.Random.AbstractDistribution{Float32}
    p::Float32
end
KernelForge.Random.uniforms_per_sample(::MyDist) = 1
KernelForge.Random.sample_from_u32(d::MyDist, u::UInt32) = ...      # GPU fast path
KernelForge.Random.sample_at(d::MyDist, seed, i) = ...              # CPU + tail path
```
"""
module Random

include("algorithms.jl")
include("philox.jl")
include("uniforms.jl")
include("distributions.jl")
include("rand_kernel.jl")
include("rand.jl")
include("randperm.jl")

end # module Random
