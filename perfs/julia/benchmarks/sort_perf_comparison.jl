#=
Single-vector sort benchmark
============================
Compares end-to-end key-sort of a length-N vector across:

    1. KernelForge.sort!         — radix LSD (this package)
    2. Base.sort!  on CuArray    — CUDA.jl's native sort (bitonic)
    3. AcceleratedKernels.sort!  — merge sort
    4. CUB DeviceRadixSort       — via cub_sort_benchmark JSON envelope

Methodology inherits `bench_utils.jl`:
- 500 ms warmup, then two phases: 10-trial kernel timing via CUDA
  events / AMDGPU.@elapsed and 100-trial host wall-clock via time_ns().
- All scratch buffers (KF tmp, AK temp) are pre-allocated once per
  (T, n) and reused.
- The per-trial `copyto!(src, src_init)` reset runs as `bench()`'s
  `reset` kwarg — outside the event bracket and outside time_ns(), so
  the D2D copy is NEVER counted in kernel or total time. This matches
  CUB's standalone binary which measures only the sort kernel.
- `KF_QUICK_BENCH=1` swaps to 100 ms / 5 trials for smoke runs.

Results saved to `perfs/julia/results/<GPU_TAG>/sort.csv` with the
canonical 1D schema:
    n, type, method, mean_kernel_μs, std_kernel_μs,
    mean_total_μs, std_total_μs
=#
include("../init.jl")

using AcceleratedKernels

const DEFAULT_CUB_EXE = cub_exe(joinpath(@__DIR__, "../../cuda_cpp/cub_nvcc/bin/cub_sort_benchmark"))
const DEFAULT_ROCPRIM_EXE = joinpath(@__DIR__, "../../rocm_cpp/hipcub_hipcc/bin/rocprim_sort_benchmark")

# Session-level GPU preheat — lock clock at sustained steady-state
# BEFORE any bench begins. Without this, the first cell pays a ~10 %
# clock-ramp penalty on laptop GPUs (RTX1000 mobile measured).
println("→ GPU preheat …")
gpu_preheat(backend; duration_s = 1.0)
println("  done.")

# ---------------------------------------------------------------------------
# Per-(T, n) runner
# ---------------------------------------------------------------------------

"""
    run_sort_benchmarks(::Type{T}, n) -> Vector{NamedTuple}

Bench KF / CUDA / AK / CUB sort for a single (T, n) cell and return rows
in the canonical 1D schema. CUB rows are NaN when the binary is missing
or the backend is not CUDA.
"""
function run_sort_benchmarks(::Type{T}, n::Int) where T
    label_T = string(T)
    src_cpu = T <: AbstractFloat ? randn(T, n) : rand(T, n)
    src_init = AT(src_cpu)

    # Three independent buffers — each method works in-place on its own
    # copy so neither warmup nor profiling sees an already-sorted input.
    src_kf  = copy(src_init)
    src_cu  = copy(src_init)
    src_ak  = copy(src_init)

    tmp_kf  = KF.@allocate sort!(src_kf)
    temp_ak = similar(src_ak)

    rows = NamedTuple[]
    # Tuple shape: (name, method, reset_fn, call_fn). `reset_fn` runs
    # outside both the event bracket and the wall-clock window so the
    # D2D copy that re-prepares the unsorted buffer doesn't pollute
    # either measurement.
    for (name, method, reset, call) in [
        ("KernelForge.sort! [$label_T]", "KernelForge",
            () -> copyto!(src_kf, src_init),
            () -> KF.sort!(src_kf; tmp = tmp_kf)),
        (has_cuda() ? "Base.sort! (CUDA.jl) [$label_T]" : "Base.sort! [$label_T]",
            has_cuda() ? "CUDA" : "Base",
            () -> copyto!(src_cu, src_init),
            () -> Base.sort!(src_cu)),
        ("AcceleratedKernels.sort! [$label_T]", "AK",
            () -> copyto!(src_ak, src_init),
            () -> AcceleratedKernels.sort!(src_ak; temp = temp_ak)),
    ]
        s = bench(name, call; backend, reset)
        push!(rows, (; n, type = label_T, method,
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end

    if has_cuda()
        # Free Julia-side buffers (src_init + 3 method copies + tmp_kf +
        # temp_ak ≈ 5×n×sizeof(T)) before launching the CUB subprocess
        # to avoid OOM-kill at large N. See OOM note in
        # mapreduce_perf_comparison.jl run_mapreduce_benchmarks().
        src_init = nothing
        src_kf = nothing; src_cu = nothing; src_ak = nothing
        tmp_kf = nothing; temp_ak = nothing
        GC.gc(true); CUDA.reclaim()
        s = bench_cub_or_nan(DEFAULT_CUB_EXE, n, T; safety_factor=1.5)
        push!(rows, (; n, type = label_T, method = "CUB",
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end

    if has_roc()
        # AMD vendor baseline — rocPRIM (via hipCUB) DeviceRadixSort. Mirror of the
        # CUB block above; no free-mem guard (MI300-class cards have ≥128 GB).
        s = bench_rocprim_or_nan(DEFAULT_ROCPRIM_EXE, n, T)
        push!(rows, (; n, type = label_T, method = "rocPRIM",
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end

    return rows
end

# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------

# Fixed across machines: K * sizeof(T) ≤ 1.6 GB worst-case (Float64×10^8)
# — fits even a 6 GB laptop card.
sizes = [10^6, 10^7, 10^8]
types = [UInt32, UInt64, Float32, Float64]

# ---------------------------------------------------------------------------
# Collect, save, display
# ---------------------------------------------------------------------------

all_rows = NamedTuple[]
for n in sizes, T in types
    println("\n" * "=" ^ 60)
    println("Sort: n=$n, T=$T")
    println("=" ^ 60)
    # Quick refresh — the per-cell src allocation + CPU randn can leave
    # the GPU idle long enough to drop clock; 300 ms preheat re-locks it.
    gpu_preheat(backend; duration_s = 0.3)
    try
        append!(all_rows, run_sort_benchmarks(T, n))
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

df = DataFrame(all_rows)
csv_path = joinpath(RESULT_DIR, "sort.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")

display_results(df, String(GPU_TAG), "Sort Benchmark Results")
