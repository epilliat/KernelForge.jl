#=
Copy Scaling Benchmarking Script
=================================
Benchmarks copy performance as a function of array size.
Compares KernelForge v1/v4/v8/v16 against CUDA.copyto!

Methodology:
- Single warmup pass per dtype (outside size loop)
- 10 profiled trials per configuration
- Results saved to results/<gpu_short>/copy.csv
=#

include("../init.jl")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

sizes = unique(round.(Int, 10 .^ range(log10(10_000), log10(10^8), length=500)))
types = [Float32, UInt8]
trials = 10

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

function run_copy_benchmarks(sizes::Vector{Int}, types::Vector{DataType}; trials::Int=10)
    rows = NamedTuple[]

    for T in types
        src = CUDA.ones(T, sizes[end])
        dst = CuArray{T}(undef, sizes[end])

        # Single warmup per dtype on largest size
        warmup(() -> copyto!(dst, src))
        warmup(() -> KernelForge.vcopy!(dst, src; Nitem=1))
        warmup(() -> KernelForge.vcopy!(dst, src; Nitem=4))
        warmup(() -> KernelForge.vcopy!(dst, src; Nitem=8))
        warmup(() -> KernelForge.vcopy!(dst, src; Nitem=16))

        for (idx, n) in enumerate(sizes)
            println("[$idx/$(length(sizes))] T=$T, n=$n")

            src = CUDA.ones(T, n)
            dst = CuArray{T}(undef, n)

            for (method, call) in [
                ("CUDA", () -> copyto!(dst, src)),
                ("Forge v1", () -> KernelForge.vcopy!(dst, src; Nitem=1)),
                ("Forge v4", () -> KernelForge.vcopy!(dst, src; Nitem=4)),
                ("Forge v8", () -> KernelForge.vcopy!(dst, src; Nitem=8)),
                ("Forge v16", () -> KernelForge.vcopy!(dst, src; Nitem=16)),
            ]
                s = bench(method, call; trials, duration=-1, exclude_copy=false)
                push!(rows, (; n, T=string(T), method,
                    s.mean_kernel_μs, s.std_kernel_μs,
                    s.mean_total_μs, s.std_total_μs))
            end
        end
    end

    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Run and save
# ---------------------------------------------------------------------------

df = run_copy_benchmarks(sizes, types; trials)

csv_path = joinpath(RESULT_DIR, "copy2.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")