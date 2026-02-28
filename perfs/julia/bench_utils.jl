#=
bench_utils.jl
==============
Benchmarking primitives: warmup, kernel timing, CUB runner, DataFrame helpers.
No plotting dependencies — safe to use in the minimal quick_bench environment.
=#

using Revise
using KernelForge
using KernelForge: UnitFloat8
using CUDA
using KernelAbstractions
using AcceleratedKernels
using BenchmarkTools
using Statistics
using DataFrames
using CSV
using JSON3
using Printf
using PrettyTables
using Quaternions


include("number_format.jl")

# ---------------------------------------------------------------------------
# GPU benchmarking primitives
# ---------------------------------------------------------------------------

"""
    warmup(f; duration=0.5)

Repeatedly call `f()` for `duration` seconds to warm up JIT and GPU state.
"""
function warmup(f; duration=0.5)
    start = time()
    while time() - start < duration
        CUDA.@sync f()
    end
end

"""
    sum_kernel_durations_μs(prof; exclude_copy=true) -> Float64

Sum all kernel durations from a CUDA profile, returning microseconds.
Pass `exclude_copy=false` to include `[copy*]` entries (needed for copy benchmarks).
"""
function sum_kernel_durations_μs(prof; exclude_copy=true)
    device = prof.device
    isempty(device.name) && return 0.0
    total = 0.0
    for i in eachindex(device.name)
        exclude_copy && startswith(device.name[i], "[copy") && continue
        total += device.stop[i] - device.start[i]
    end
    return total * 1e6
end

"""
    bench(name, f; duration=0.5, trials=100, backend=CUDABackend(), exclude_copy=true)
        -> NamedTuple{mean_kernel_μs, std_kernel_μs, mean_total_μs, std_total_μs}

Benchmark `f()` by:
- Warming up for `duration` seconds.
- Profiling `trials` times for kernel time via `CUDA.@profile`.
- Timing `trials` samples with `BenchmarkTools` for total wall-clock time.
"""
function bench(name, f;
    duration=0.5,
    trials=100,
    backend=CUDABackend(),
    exclude_copy=true
)
    warmup(f; duration)
    println("=== $name ===")

    kernel_times = Vector{Float64}(undef, trials)
    for i in 1:trials
        prof = CUDA.@profile f()
        kernel_times[i] = sum_kernel_durations_μs(prof; exclude_copy)
    end

    mean_kernel_μs = mean(kernel_times)
    std_kernel_μs = std(kernel_times)

    result = @benchmark begin
        $f()
        KernelAbstractions.synchronize($backend)
    end samples = trials evals = 1

    mean_total_μs = mean(result).time / 1000
    std_total_μs = std(result).time / 1000

    println("Kernel time: $(mean_kernel_μs) ± $(std_kernel_μs) μs (n=$trials)")
    println("Total time:  $(mean_total_μs) ± $(std_total_μs) μs (n=$trials)")

    return (; mean_kernel_μs, std_kernel_μs, mean_total_μs, std_total_μs)
end

# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

"""
    display_results(df, gpu_tag)

Pretty-print benchmark DataFrame with KernelForge rows highlighted in blue/bold.
"""
function display_results(df::DataFrame, gpu_tag::String, title::String)
    df_display = transform(df,
        [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs] .=>
            (x -> round.(x; digits=2)) .=>
                [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs]
    )
    df_display.n_str = map(n -> "1e$(round(Int, log10(n)))", df_display.n)
    select!(df_display, :n_str, :)
    select!(df_display, Not(:n))

    println("=== $title — GPU: $gpu_tag ===")
    hl = TextHighlighter(
        (data, i, j) -> data[i, :method] == "KernelForge",
        crayon"blue bold"
    )
    pretty_table(df_display; highlighters=[hl])
end

# ---------------------------------------------------------------------------
# CUB benchmark runner
# ---------------------------------------------------------------------------

const _DTYPE_STRINGS = Dict(
    Float32 => "float",
    Float64 => "double",
    Int32 => "int",
    UInt64 => "uint64",
    UInt8 => "uint8",
)

"""
    run_cub_benchmark(exe; N, iterations, warmup_ms, dtype, extra_flags="") -> JSON3 result

Run an external CUB benchmark executable and return parsed JSON.
"""
function run_cub_benchmark(exe::String;
    N::Int=100_000_000,
    iterations::Int=100,
    warmup_ms::Real=500,
    dtype::Type=Float32,
    extra_flags::String=""
)
    dtype_str = get(_DTYPE_STRINGS, dtype, nothing)
    dtype_str === nothing && error("Unsupported dtype: $dtype")

    exe_path = abspath(expanduser(exe))
    base_cmd = `$exe_path -n $N -i $iterations -w $warmup_ms -t $dtype_str -j`
    cmd = isempty(extra_flags) ? base_cmd : Cmd([base_cmd.exec..., split(extra_flags)...])
    return JSON3.read(read(cmd, String))
end

"""
    bench_cub(exe, n, T; trials=100, extra_flags="")
        -> NamedTuple{mean_kernel_μs, std_kernel_μs, mean_total_μs, std_total_μs}
"""
function bench_cub(exe::String, n::Int, ::Type{T}; trials::Int=100, extra_flags::String="") where T
    println("=== CUB ===")
    results = run_cub_benchmark(exe; N=n, iterations=trials, dtype=T, extra_flags)

    if isempty(results)
        @warn "CUB returned empty results for dtype=$T, n=$n — returning NaN"
        return (; mean_kernel_μs=NaN, std_kernel_μs=NaN, mean_total_μs=NaN, std_total_μs=NaN)
    end

    mean_kernel_μs = results[1]["mean_ms"] * 1000
    std_kernel_μs = results[1]["std_ms"] * 1000

    println("Kernel time: $(mean_kernel_μs) ± $(std_kernel_μs) μs (n=$trials)")
    return (; mean_kernel_μs, std_kernel_μs, mean_total_μs=mean_kernel_μs, std_total_μs=std_kernel_μs)
end

"""
    bench_cub_or_nan(exe, n, T; kw...) -> NamedTuple

Like `bench_cub` but returns NaN stats if the executable is missing/empty.
"""
function bench_cub_or_nan(exe::String, n::Int, ::Type{T}; kw...) where T
    if !isempty(exe) && isfile(exe)
        return bench_cub(exe, n, T; kw...)
    else
        isempty(exe) || @warn "CUB executable not found: $exe"
        return (; mean_kernel_μs=NaN, std_kernel_μs=NaN, mean_total_μs=NaN, std_total_μs=NaN)
    end
end

# ---------------------------------------------------------------------------
# DataFrame row builder
# ---------------------------------------------------------------------------

"""
    push_stats!(rows, stats; n, T, method)

Append a benchmark result NamedTuple to `rows` with metadata fields.
"""
function push_stats!(rows, stats; n, T, method)
    push!(rows, (;
        n, T=string(T), method,
        stats.mean_kernel_μs, stats.std_kernel_μs,
        stats.mean_total_μs, stats.std_total_μs,
    ))
end