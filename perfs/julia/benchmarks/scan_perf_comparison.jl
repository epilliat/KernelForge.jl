#=
Scan Performance Benchmarking Script
====================================
Compares KernelForge.scan! against CUDA.accumulate! and AcceleratedKernels for prefix scan.
Methodology:
- 500ms warm-up phase (via bench_utils.jl `bench`)
- 100 profiled trials for accurate kernel timing
- benchmark + synchronization is used for full pipeline timing
- Results saved to results/<gpu_short>/scan.csv
- Final DataFrame printed at the end
=#
include("../init.jl")

const DEFAULT_CUB_EXE = joinpath(@__DIR__, "../../cuda_cpp/cub_nvcc/bin/cub_scan_benchmark")

# Session-level GPU preheat — see comment in mapreduce_perf_comparison.jl.
println("→ GPU preheat …")
gpu_preheat(backend; duration_s = 1.0)
println("  done.")

# ---------------------------------------------------------------------------
# Benchmark runner — returns a vector of NamedTuples
# ---------------------------------------------------------------------------

"""
    run_scan_benchmarks(src, dst, label_T, n; init, cub_exe, cub_T)

Runs bench() for KernelForge, CUDA/generic accumulate!, AcceleratedKernels, and CUB on `src`.
Returns a Vector of row NamedTuples for the results DataFrame.
"""
function run_scan_benchmarks(src::AT{T}, dst::AT{T}, label_T::String, n::Int;
    init=zero(T),
    cub_exe::String=DEFAULT_CUB_EXE, cub_T::Type=T) where T

    rows = NamedTuple[]
    tmp_kf = KF.@allocate scan1d(identity, +, src)
    temp_ak = similar(src)
    for (name, method, call) in [
        (has_cuda() ? "CUDA [$label_T]" : "Base [$label_T]",
            has_cuda() ? "CUDA" : "Base",
            () -> accumulate!(+, dst, src)),
        ("AcceleratedKernels [$label_T]", "AK",
            () -> AcceleratedKernels.accumulate!(+, dst, src; init, temp=temp_ak)),
        ("KernelForge [$label_T]", "KernelForge",
            () -> KernelForge.scan!(+, dst, src; tmp=tmp_kf)),
    ]
        s = bench(name, call; backend)
        push!(rows, (; n, type=label_T, method,
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end

    if has_cuda()
        # Free Julia-side buffers before launching CUB subprocess (see OOM
        # note in run_mapreduce_benchmarks). Scan holds 2 buffers (src,
        # dst) + tmp_kf + temp_ak ≈ 4×n×sizeof(T).
        src = nothing; dst = nothing
        tmp_kf = nothing; temp_ak = nothing
        GC.gc(true); CUDA.reclaim()
        s = bench_cub_or_nan(cub_exe, n, cub_T; safety_factor=1.5,
                             extra_flags="-m inclusive")
        push!(rows, (; n, type=label_T, method="CUB",
            s.mean_kernel_μs, s.std_kernel_μs,
            mean_total_μs=NaN, std_total_μs=NaN))
    end

    return rows
end


n = 10^6
T = Float32
src = fill!(AT{T}(undef, n), one(T))
dst = fill!(AT{T}(undef, n), zero(T))

# tmp=KF.@allocate scan1d(identity, +, src)
# @benchmark (KernelForge.scan!(identity, +, dst, src; tmp);KA.synchronize(backend)) seconds=1
# @benchmark (accumulate!(+, dst, src);KA.synchronize(backend)) seconds=1
# KA.synchronize(backend)
# temp_ak = similar(src)
# @benchmark (AcceleratedKernels.accumulate!(+, dst, src; temp=temp_ak,init=0.0f0);KA.synchronize(backend)) seconds=1

# Simple profiling example (without warmup here which gives slower results)


#%%
# CUDA.@profile accumulate!(+, dst, src)
# CUDA.@profile AcceleratedKernels.accumulate!(+, dst, src; init=0.0f0)
# CUDA.@profile KernelForge.scan!(identity, +, dst, src; Nitem=8)


# ---------------------------------------------------------------------------
# Configuration — edit these to control what gets benchmarked
# ---------------------------------------------------------------------------

sizes = [10^6, 10^7, 10^8, 10^9]
types = [Float32, Float64]

# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------

all_rows = NamedTuple[]

for n in sizes, T in types
    println("\n" * "="^60)
    println("Scan: n=$n, T=$T")
    println("="^60)
    gpu_preheat(backend; duration_s = 0.3)
    try
        if T === QuaternionF64
            src_cpu = [QuaternionF64((x ./ sqrt(sum(x .^ 2)))...) for x in eachcol(randn(4, n))]
            src = AT(src_cpu)
            dst = fill!(AT{T}(undef, n), zero(T))

            for (name, method, call) in [
                (has_cuda() ? "CUDA [QuaternionF64]" : "Base [QuaternionF64]",
                    has_cuda() ? "CUDA" : "Base",
                    () -> accumulate!(*, dst, src)),
                ("AcceleratedKernels [QuaternionF64]", "AK",
                    () -> AcceleratedKernels.accumulate!(*, dst, src; init=one(T))),
                ("KernelForge [QuaternionF64]", "KernelForge",
                    () -> KernelForge.scan!(*, dst, src; Nitem=4)),
            ]
                s = bench(name, call; backend)
                push!(all_rows, (; n, type="QuaternionF64", method,
                    s.mean_kernel_μs, s.std_kernel_μs,
                    s.mean_total_μs, s.std_total_μs))
            end
            push!(all_rows, (; n, type="QuaternionF64", method="CUB",
                mean_kernel_μs=NaN, std_kernel_μs=NaN,
                mean_total_μs=NaN, std_total_μs=NaN))

        else
            src = fill!(AT{T}(undef, n), one(T))
            dst = fill!(AT{T}(undef, n), zero(T))
            append!(all_rows, run_scan_benchmarks(src, dst, string(T), n))
        end
    catch err
        msg = sprint(showerror, err)
        if occursin("Out of GPU memory", msg) || occursin("hipErrorOutOfMemory", msg)
            @warn "Skipping cell — out of GPU memory" n T
            for m in (has_cuda() ? ("CUDA","AK","KernelForge","CUB") : ("Base","AK","KernelForge"))
                push!(all_rows, (; n, type=string(T), method=m,
                                  mean_kernel_μs=NaN, std_kernel_μs=NaN,
                                  mean_total_μs=NaN,  std_total_μs=NaN))
            end
            GC.gc(true); has_cuda() && CUDA.reclaim()
        else
            rethrow(err)
        end
    end
end

# ---------------------------------------------------------------------------
# Build DataFrame, save CSV, display
# ---------------------------------------------------------------------------

df = DataFrame(all_rows)
mkpath(RESULT_DIR)
csv_path = joinpath(RESULT_DIR, "scan.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")

df_display = transform(df,
    [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs] .=>
        (x -> round.(x; digits=2)) .=>
            [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs]
)
df_display.n_str = map(n -> "1e$(round(Int, log10(n)))", df_display.n)
select!(df_display, :n_str, :)
select!(df_display, Not(:n))

println("=== Scan Benchmark Results — GPU: $GPU_TAG ===")
if has_cuda()
    hl = TextHighlighter(
        (data, i, j) -> data[i, :method] == "KernelForge",
        crayon"blue bold"
    )
    pretty_table(df_display; highlighters=[hl])
else
    hl = Highlighter(
        (data, i, j) -> data[i, Symbol("method")] == "KernelForge",
        crayon"blue bold"
    )
    pretty_table(df_display; highlighters=(hl,), backend=Val(:text))
end