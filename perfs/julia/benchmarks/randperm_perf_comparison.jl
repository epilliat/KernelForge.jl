#=
Random Permutation Performance Benchmarking Script
==================================================
Compares KernelForge.Random.randperm! against two GPU baselines that
implement randperm via the same idiom (generate random keys → sortperm):

    1. KernelForge        — Philox rand! + radix sortperm (this package)
    2. CUDA composite     — CUDA-native rand! + bitonic sortperm! (Base.sortperm! on CuArray)
    3. AK composite       — CUDA-native rand! + AcceleratedKernels.sortperm! (merge sort)

Apple-to-apple methodology
--------------------------
- All three paths produce a uniformly random permutation of 1..N in a
  caller-provided `perm` buffer. The output type (`Int32` / `Int64`) is
  the *same* across libraries within a row.
- Each path runs end-to-end: random-key generation + identity init +
  sortperm. Buffers are pre-allocated once and reused across trials
  (mirrors how `scan_perf_comparison.jl` reuses `tmp_kf` / `temp_ak`).
- `bench()` excludes device-memcpy events (`[copy*]`) uniformly for ALL
  libraries — never for KF only. This is the project-wide rule: if a
  copy is excluded for one library, it must be excluded for every
  library on the same row (and vice-versa).

Methodology details (inherited from `bench_utils.jl`):
- 500ms warm-up phase per call
- 100 profiled trials for kernel timing (via `CUDA.@profile`)
- `BenchmarkTools` total wall-clock timing with synchronization
- Results saved to results/<gpu_short>/randperm.csv
=#
include("../init.jl")

using Random
using AcceleratedKernels
const KFR = KernelForge.Random

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

"""
    run_randperm_benchmarks(n, IT; seed, cuda_native_rng)

Three apple-to-apple randperm pipelines on output type `IT`. Each call
produces a randomized permutation of `1:n` written into a pre-allocated
`perm` buffer of element type `IT`.
"""
function run_randperm_benchmarks(n::Int, ::Type{IT}=Int32;
        seed=UInt64(0xC0FFEE),
        cuda_native_rng=nothing) where {IT<:Integer}

    # Pre-allocated, reused across trials.
    perm_kf  = AT(zeros(IT, n))
    perm_cu  = AT(zeros(IT, n))
    perm_ak  = AT(zeros(IT, n))
    keys_kf  = AT(zeros(Float32, n))         # scratch passed via `keys=` kwarg
    keys_cu  = AT(zeros(Float32, n))
    keys_ak  = AT(zeros(Float32, n))
    temp_ak  = AT(zeros(IT, n))              # AK's `temp=` scratch

    rows = NamedTuple[]
    for (name, method, call) in [
        ("KernelForge.Random.randperm! [$IT]", "KernelForge",
            () -> KFR.randperm!(perm_kf, seed; keys=keys_kf)),
        ("CUDA composite (rand! + sortperm!) [$IT]", "CUDA",
            () -> begin
                Random.rand!(cuda_native_rng, keys_cu)
                Base.sortperm!(perm_cu, keys_cu; initialized=false)
            end),
        ("AcceleratedKernels composite [$IT]", "AK",
            () -> begin
                Random.rand!(cuda_native_rng, keys_ak)
                AcceleratedKernels.sortperm!(perm_ak, keys_ak; temp=temp_ak)
            end),
    ]
        s = bench(name, call; backend)
        push!(rows, (; n, type=string(IT), method,
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs,  s.std_total_μs))
    end
    return rows
end

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

sizes      = [10^5, 10^6, 10^7]   # 10^8 OOMs on 6 GB cards
perm_types = [Int32]              # Int64 doubles shuffle-bandwidth; add back when tuning that path

# CUDA-only RNG handle. KernelForge.Random uses its own Philox seed input.
cuda_native_rng = has_cuda() ? CUDA.default_rng() : nothing

# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------

all_rows = NamedTuple[]

for n in sizes, IT in perm_types
    println("\n" * "="^60)
    println("Randperm: n=$n, IT=$IT")
    println("="^60)
    append!(all_rows, run_randperm_benchmarks(n, IT; cuda_native_rng))
end

# ---------------------------------------------------------------------------
# Build DataFrame, save CSV, display
# ---------------------------------------------------------------------------

df = DataFrame(all_rows)
mkpath(RESULT_DIR)
csv_path = joinpath(RESULT_DIR, "randperm.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")

# Throughput (millions of permuted indices per second), handy for at-a-glance.
df.melems_per_sec = round.(df.n ./ (df.mean_kernel_μs .* 1e-6) ./ 1e6; digits=2)

display_results(df, String(GPU_TAG), "Randperm Benchmark Results")
