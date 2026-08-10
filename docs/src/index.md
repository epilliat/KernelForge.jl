# KernelForge.jl
High-performance, portable GPU primitives for Julia. A pure Julia implementation delivering performance competitive with optimized CUDA C++ libraries.

!!! warning "Experimental Status"
    This package is in an experimental phase. Although extensive testing has been performed,
    no bounds checking is performed, which may lead to unexpected behavior with out-of-bounds
    access. Correctness and performance have been validated on NVIDIA (RTX 1000, A40) and
    AMD (MI300X) GPUs.

!!! info "Architecture & Contributions"
    KernelForge.jl builds on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
    for GPU kernel dispatch. However, certain low-level operations—including warp shuffle
    instructions, vectorized memory access, and memory ordering semantics—are not yet available
    in KA.jl, so we use [KernelIntrinsics.jl](https://github.com/epilliat/KernelIntrinsics.jl) for these primitives.
    **The core contribution of this package lies in the GPU kernel implementations themselves**,
    designed to be portable across GPU backends. CUDA and AMDGPU are both supported as weak
    dependencies; Intel GPU support would primarily require work in KernelIntrinsics.jl.

!!! note "Citation"
    A paper describing this work is in preparation. If you use this code, please check back
    for citation details.

## Installation
```julia
using Pkg
Pkg.add("KernelForge")
```

## Features
- **Map-reduce** with custom functions and operators, supporting arbitrary dimensions and
  multidimensional arrays including non-contiguous dimension reduction via `mapreducedims`
- **Prefix scan** supporting non-commutative operations
- **Matrix-vector operations** with customizable element-wise and reduction operations
- **Matrix-matrix product** — `gemm`/`gemm!` with customizable element-wise, reduction
  and epilogue operations (arbitrary isbits eltypes, no operator identity required),
  all four transpose states, and an opt-in tensor-core family (`family=:mma`)
- **Sort** — 1D radix `sort`/`sort!` with optional `keys=` keyval form, `sortperm`
  for permutation indices, `sample_sort` accepting an arbitrary `lt` comparator, and
  `sort_columns` for batched per-column sort of `K × M` matrices
- **Random** — Philox4x32-10 counter-based GPU RNG (deterministic across CPU and GPU
  for a given seed); `rand!`, `randn!`, `randperm!`, plus `Uniform`, `Normal`,
  `Exponential`, `Bernoulli`, `Categorical` distribution drivers
- **Search** — `findfirst`, `findlast`, `argmax`, `argmin` on GPU arrays
- **Vectorized copy** with configurable load/store widths
- **Per-arch tuning** — runtime lookup of pre-tuned kernel parameters from
  `data/tuning/<Arch>.{json,jl}`; graceful fallback to heuristic defaults when
  no tuning ships for the detected arch
- Views and strided arrays supported throughout, enabled by KernelIntrinsics.jl's
  vectorized memory access primitives which correctly handle non-contiguous memory layouts
- Supports CUDA (NVIDIA) and AMDGPU (AMD) backends via weak dependencies
- Includes `UnitFloat8`, a custom 8-bit floating-point type with range (-1, 1) for testing

## Quick Start
```julia
using KernelForge
using CUDA

# Prefix scan
src = CUDA.rand(Float32, 10^6)
dst = similar(src)
KernelForge.scan!(+, dst, src)

# Matrix-vector multiply
A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)
y = KernelForge.matvec(A, x)

# Map-reduce (full reduction)
total = KernelForge.mapreduce(abs2, +, src)

# Map-reduce over specific dimensions
B = CUDA.rand(Float32, 4, 8, 16)
result = KernelForge.mapreduce(identity, +, B; dims=(1, 3))  # shape: (1, 8, 1)

# Views are supported
v = view(src, 1:2:10^6)
total_view = KernelForge.mapreduce(abs2, +, v)

# Search
i = KernelForge.findfirst(>(0.99f0), src)
j = KernelForge.argmax(src)

# Sort (1D radix; ~CUB throughput on Turing)
keys = CUDA.rand(Float32, 10^6)
KernelForge.sort!(keys)                              # in-place
perm = KernelForge.sortperm(CUDA.rand(Float32, 10^6))  # permutation indices

# Random — counter-based RNG; (seed, i) → byte-identical CPU ↔ CUDA ↔ AMDGPU
import KernelForge.Random as KFR
draws = CUDA.zeros(Float32, 10^6)
KFR.rand!(draws, UInt64(42))                          # Uniform[0, 1)
KFR.randn!(draws, UInt64(42))                         # Normal(0, 1)
KFR.rand!(draws, KFR.Exponential(2f0), UInt64(42))    # Exponential(2)
```

## Acknowledgments
This package builds on the foundation provided by
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). The API design draws inspiration from
several packages in the Julia ecosystem. Development of the API and documentation was
assisted by [Claude](https://claude.ai) (Anthropic).

## Sponsors
KernelForge.jl is an open-source project maintained on personal time. If it is
useful to you — especially in a production or HPC setting — you can support its
development via [GitHub Sponsors](https://github.com/sponsors/epilliat).
Corporate sponsors receive priority support on issues and an acknowledgment here
in the documentation.

## License
MIT