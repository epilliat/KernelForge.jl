# KernelForge.jl

High-performance, portable GPU primitives for Julia. A pure Julia implementation delivering performance competitive with optimized CUDA C++ libraries.

## Documentation

Full documentation, API reference, and examples are available at:
**https://epilliat.github.io/KernelForge.jl/stable/**

## Installation
```julia
using Pkg
Pkg.add("KernelForge")
```

## Features

- **Map-reduce** with custom functions and operators, supporting arbitrary dimensions and multidimensional arrays
- **Prefix scan** supporting non-commutative operations
- **Matrix-vector operations** with customizable element-wise and reduction operations
- **Search** — `findfirst`, `findlast`, `argmax`, `argmin` on GPU arrays
- **Vectorized copy** with configurable load/store widths
- Views and strided arrays supported throughout
- Supports CUDA (NVIDIA) and AMDGPU (AMD) backends via weak dependencies

## License

MIT