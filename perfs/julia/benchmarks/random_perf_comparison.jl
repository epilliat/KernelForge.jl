#=
Random Performance Benchmarking Script
======================================
Compares KernelForge.Random.rand! / randn! against CUDA's two RNGs:
    - CUDA native (GPUArrays Philox-flavoured RNG, the default of `Random.rand!`)
    - cuRAND library RNG (curand_rng())
Methodology:
- 500ms warm-up phase (via bench_utils.jl `bench`)
- 100 profiled trials for accurate kernel timing
- benchmark + synchronization for full pipeline timing
- Results saved to results/<gpu_short>/random.csv
- Final DataFrame printed at the end

Notes:
- Throughput at large N is bandwidth-bound. KernelForge's Float32 Uniform
  hits ~96% of card DRAM peak on RTX 1000 Ada (45.7 Gs/s = 183 GB/s).
- curand library is faster at tiny N (persistent state, optimised launch);
  KernelForge wins at sizes ≥ 16M elements.
=#
include("../init.jl")

using Random           # for the cuRAND-backed `Random.rand!` API
const KFR = KernelForge.Random

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

"""
    run_random_uniform_benchmarks(out, label_T, n; cuda_native_rng, cuda_curand_rng)

Runs bench() for KernelForge.Random.rand! and CUDA's two RNGs filling `out`
with Uniform[0, 1) samples. Returns a Vector of row NamedTuples.
"""
function run_random_uniform_benchmarks(out::AT{T}, label_T::String, n::Int;
    seed=UInt64(0xC0FFEE),
    cuda_native_rng=nothing,
    cuda_curand_rng=nothing) where T

    rows = NamedTuple[]
    for (name, method, call) in [
        ("KernelForge.Random.rand! [$label_T]", "KernelForge",
            () -> KFR.rand!(out, seed)),
        ("CUDA native rand! [$label_T]", "CUDA-native",
            () -> Random.rand!(cuda_native_rng, out)),
        ("cuRAND library rand! [$label_T]", "curand",
            () -> Random.rand!(cuda_curand_rng, out)),
    ]
        s = bench(name, call; backend)
        push!(rows, (; n, type=label_T, method,
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end
    return rows
end

"""
    run_random_normal_benchmarks(out, label_T, n; cuda_native_rng, cuda_curand_rng)

Same idea but for standard-normal samples — three RNGs side by side.
"""
function run_random_normal_benchmarks(out::AT{T}, label_T::String, n::Int;
    seed=UInt64(0xC0FFEE),
    cuda_native_rng=nothing,
    cuda_curand_rng=nothing) where T

    rows = NamedTuple[]
    for (name, method, call) in [
        ("KernelForge.Random.randn! [$label_T]", "KernelForge",
            () -> KFR.randn!(out, seed)),
        ("CUDA native randn! [$label_T]", "CUDA-native",
            () -> Random.randn!(cuda_native_rng, out)),
        ("cuRAND library randn! [$label_T]", "curand",
            () -> Random.randn!(cuda_curand_rng, out)),
    ]
        s = bench(name, call; backend)
        push!(rows, (; n, type=label_T, method,
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end
    return rows
end

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

sizes = [10^6, 10^7, 10^8, Int(2^28)]      # last is ~268M — bandwidth-bound regime
uniform_types = [Float32, Float64]
normal_types  = [Float32]

# CUDA-only: pick up RNG handles once. (Skipping the CPU backend cleanly.)
cuda_native_rng = has_cuda() ? CUDA.default_rng() : nothing
cuda_curand_rng = has_cuda() ? CUDA.curand_rng()  : nothing

# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------

all_rows = NamedTuple[]

# Uniform
for n in sizes, T in uniform_types
    println("\n" * "="^60)
    println("Random Uniform: n=$n, T=$T")
    println("="^60)
    out = AT(zeros(T, n))
    append!(all_rows, run_random_uniform_benchmarks(out, string(T), n;
        cuda_native_rng, cuda_curand_rng))
end

# Normal
for n in sizes, T in normal_types
    println("\n" * "="^60)
    println("Random Normal: n=$n, T=$T")
    println("="^60)
    out = AT(zeros(T, n))
    append!(all_rows, run_random_normal_benchmarks(out, "$(T)-normal", n;
        cuda_native_rng, cuda_curand_rng))
end

# ---------------------------------------------------------------------------
# Build DataFrame, save CSV, display
# ---------------------------------------------------------------------------

df = DataFrame(all_rows)
mkpath(RESULT_DIR)
csv_path = joinpath(RESULT_DIR, "random.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")

# Add Gs/s column for at-a-glance comparison.
df.gsamples_per_sec = round.(df.n ./ (df.mean_kernel_μs .* 1e-6) ./ 1e9; digits=2)

display_results(df, String(GPU_TAG), "Random Benchmark Results")
