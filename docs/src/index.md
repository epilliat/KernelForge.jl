# Luma.jl

High-performance, portable GPU primitives for Julia. A pure Julia implementation delivering performance competitive with optimized CUDA C++ libraries.

!!! warning "Experimental Status"
    This package is in an experimental phase. Although extensive testing has been performed, the current implementation does not support views or strided arrays. No bounds checking is performed, which may lead to unexpected behavior with non-contiguous data. Correctness and performance have been validated only on a small NVIDIA RTX 1000.

!!! info "Architecture & Contributions"
    Luma.jl builds on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) for GPU kernel dispatch. However, certain low-level operations—including warp shuffle instructions, vectorized memory access, and memory ordering semantics—are not yet available in KA.jl, so we use [KernelIntrinsics.jl](https://github.com/...) for these primitives. As KernelIntrinsics.jl currently supports only CUDA, Luma.jl is likewise restricted to CUDA.
    
    **The core contribution of this package lies in the GPU kernel implementations themselves**, designed to be portable once the underlying intrinsics become available on other backends. Extending support to AMD and Intel GPUs would primarily require work in KernelIntrinsics.jl, with minimal adaptations in Luma.jl.

!!! note "Citation"
    A paper describing this work is in preparation. If you use this code, please check back for citation details.

## Installation
```julia
using Pkg
Pkg.add("Luma")
```

## Features

- **Matrix-vector operations** with customizable element-wise and reduction operations
- **Prefix scan** supporting non-commutative operations
- **Map-reduce** with custom functions and operators, supporting 1D and 2D reductions
- **Vectorized copy** with configurable load/store widths
- Currently only for 1D and 2D arrays
- Currently CUDA-only; cross-platform support via KernelAbstractions.jl planned
- Includes `UnitFloat8`, a custom 8-bit floating-point type with range (-1, 1) for testing

## Quick Start
```julia
using Luma
using CUDA

# Prefix scan
src = CUDA.rand(Float32, 10^6)
dst = similar(src)
Luma.scan!(+, dst, src)

# Matrix-vector multiply
A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)
y = Luma.matvec(A, x)

# Map-reduce
total = Luma.mapreduce(abs2, +, src; to_cpu=true)
```

## Acknowledgments

This package builds on the foundation provided by [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). The API design draws inspiration from several packages in the Julia ecosystem. Development of the API and documentation was assisted by [Claude](https://claude.ai) (Anthropic).

## License

MIT