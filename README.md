# KernelForge.jl

A pure-Julia GPU primitives library that is **flexible** — every primitive
takes an arbitrary operator and an arbitrary element type, your own structs and
comparators included; **cross-architecture** — one implementation runs on both
NVIDIA and AMD through KernelAbstractions; and **performant** — matching, or
coming close to, the hand-tuned vendor libraries (CUB / cuBLAS on NVIDIA,
rocPRIM / rocBLAS on AMD).

### NVIDIA A100 (Ampere, 80 GB) — vs CUB / cuBLAS

<table>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/A100/matvec_A100_comparison.png" alt="matvec — A100" width="100%"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/A100/vecmat_A100_comparison.png" alt="vecmat — A100" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/A100/scan_A100_comparison.png" alt="scan — A100" width="100%"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/A100/mapreduce_A100_comparison.png" alt="mapreduce — A100" width="100%"></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/A100/sort_A100_comparison.png" alt="sort — A100" width="100%"></td>
  </tr>
</table>

### AMD MI300A (CDNA3, ROCm 6.4) — vs rocPRIM / rocBLAS

<table>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/MI300A/matvec_MI300A_comparison.png" alt="matvec — MI300A" width="100%"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/MI300A/vecmat_MI300A_comparison.png" alt="vecmat — MI300A" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/MI300A/scan_MI300A_comparison.png" alt="scan — MI300A" width="100%"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/MI300A/mapreduce_MI300A_comparison.png" alt="mapreduce — MI300A" width="100%"></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/MI300A/sort_MI300A_comparison.png" alt="sort — MI300A" width="100%"></td>
  </tr>
</table>

<details>
<summary><b>NVIDIA A40</b> (Ampere) — vs CUB / cuBLAS, plus copy bandwidth</summary>

<table>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/A40/matvec_A40_comparison.png" alt="matvec — A40" width="100%"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/A40/vecmat_A40_comparison.png" alt="vecmat — A40" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/A40/scan_A40_comparison.png" alt="scan — A40" width="100%"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/A40/mapreduce_A40_comparison.png" alt="mapreduce — A40" width="100%"></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><img src="https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/A40/copy_A40_bandwidth.png" alt="copy bandwidth — A40" width="50%"></td>
  </tr>
</table>

</details>

<sub>Each figure states its own op, GPU and metric. <b>Forge</b> = KernelForge,
<b>AK</b> = <a href="https://github.com/JuliaGPU/AcceleratedKernels.jl">AcceleratedKernels</a>;
a hatched bar runs off the top of its panel and is labelled with its real
value. More GPUs (A40, RTX 1000, MI300X) and the raw CSV data live in the
<a href="https://github.com/epilliat/KernelForge-benchmarks"><b>KernelForge-benchmarks</b></a>
repo.</sub>

## Links

- 📄 **Paper** — [arXiv:2603.18695](https://arxiv.org/pdf/2603.18695)
- 📖 **Documentation** — [epilliat.github.io/KernelForge.jl](https://epilliat.github.io/KernelForge.jl/stable/) — API reference & examples
- 🌐 **Homepage** — [epilliat.github.io/software](https://epilliat.github.io/software/software.html)
- 📊 **Benchmarks** — [KernelForge-benchmarks](https://github.com/epilliat/KernelForge-benchmarks) — raw results across GPUs

## Installation

```julia
using Pkg
Pkg.add("KernelForge")
```

## Quick start

```julia
using KernelForge, CUDA   # or AMDGPU

x = CUDA.rand(Float32, 10^6)

# Reduction with a custom map + operator
total = KernelForge.mapreduce(abs2, +, x)      # sum of squares

# Prefix scan (supports non-commutative ops)
dst = similar(x)
KernelForge.scan!(+, dst, x)                    # cumulative sum

# Matrix–vector product
A = CUDA.rand(Float32, 1000, 500)
v = CUDA.rand(Float32, 500)
y = KernelForge.matvec(A, v)                    # y ≈ A * v

# Radix sort, in place
KernelForge.sort!(x)
```

## Features

- **Map-reduce** with custom functions and operators, supporting arbitrary dimensions and multidimensional arrays
- **Prefix scan** supporting non-commutative operations
- **Matrix-vector operations** with customizable element-wise and reduction operations
- **Matrix-matrix product** (`gemm`) with customizable element-wise, reduction and
  epilogue operations, all four transpose states, and an opt-in tensor-core path
- **Search** — `findfirst`, `findlast`, `argmax`, `argmin` on GPU arrays
- **Vectorized copy** with configurable load/store widths
- Views and strided arrays supported throughout

## Backends

CUDA (NVIDIA) and AMDGPU (AMD) via weak dependencies; the backend is selected
through KernelAbstractions extensions. Tested on NVIDIA A100, A40, RTX 1000 and
AMD MI300A, MI300X.

## Sponsors

KernelForge.jl is an open-source project maintained in my personal time.
If this package is useful to you — especially in a production or HPC setting —
you can support its development and maintenance via
[GitHub Sponsors](https://github.com/sponsors/epilliat).

Corporate sponsors receive priority support on issues and an acknowledgment
in the documentation.

Thanks to the people who support KernelForge.jl!

- [@PraneethMerugu](https://github.com/PraneethMerugu)

## License

MIT
