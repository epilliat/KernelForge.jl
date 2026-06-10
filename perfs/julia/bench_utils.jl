#=
bench_utils.jl
==============
Benchmarking primitives: warmup, kernel timing, CUB runner, DataFrame helpers.
No plotting dependencies — safe to use in the minimal quick_bench environment.

Canonical CSV schemas — every comparison script writes ONE row per
(cell × method) using `push_stats!` / `push_stats_2d!` so the same
plot helpers can consume any CSV:

  1D ops (sort, scan, mapreduce, copy, random, randperm):
      n, type, method, mean_kernel_μs, std_kernel_μs,
      mean_total_μs, std_total_μs

  2D ops (matvec, vecmat, sort_columns):
      <axis1>, <axis2>, type, method, mean_kernel_μs, std_kernel_μs,
      mean_total_μs, std_total_μs
      where <axis1, axis2> is `n, p` (matvec/vecmat) or `K, M`
      (sort_columns).

Add a new axis only when an existing one cannot describe the shape;
never replace `type/method` — they're keys the plot helpers join on.

Environment knobs:
  KF_QUICK_BENCH=1   100 ms warmup + 5 trials (vs 500 ms + 100). For
                     smoke tests against an existing CSV before commit.
  KF_CUB_EXE=…       Override the CUB executable path used by
                     `bench_cub_or_nan` and benchmark scripts (lets CI
                     point at a pre-built artifact without editing
                     hard-coded `@__DIR__` paths).
=#



using AcceleratedKernels
using Statistics
using DataFrames
using CSV
using Printf
using PrettyTables
using Quaternions
using PrettyTables


include("number_format.jl")

# ---------------------------------------------------------------------------
# Backend detection (mirrors data/tuning/matvec/autotune.jl:71-88).
#
# `try @eval import X; true catch; false end` makes the import side-effect
# happen at top-level (so X is then a usable name) and lets us probe
# whether the dependency is installed without exploding the precompile.
# Each backend branch is wrapped in `@eval begin … end` further down so the
# dead branch's macros (e.g. `AMDGPU.@elapsed`) are never lowered in the
# wrong environment.
# ---------------------------------------------------------------------------

const _CUDA_LOADED = try @eval import CUDA; true catch; false end
const _AMD_LOADED  = try @eval import AMDGPU; true catch; false end

# ---------------------------------------------------------------------------
# Quick-mode toggle (ENV-driven)
#
# Set `KF_QUICK_BENCH=1` before launching to swap every `bench()` call in
# every comparison script to a 100 ms warmup + 5 trials profile (instead
# of 500 ms + 100 trials). Use this to smoke-test new code against the
# CSVs from the previous full run before committing to a long benchmark.
#
# Read fresh on each call so it can be toggled in-session via
# `ENV["KF_QUICK_BENCH"] = "1"`.
# ---------------------------------------------------------------------------

quick_bench()    = parse(Bool, get(ENV, "KF_QUICK_BENCH", "false"))
quick_duration() = quick_bench() ? 0.1 : 0.5
quick_trials()   = quick_bench() ? 5   : 50

# ---------------------------------------------------------------------------
# GPU benchmarking primitives
# ---------------------------------------------------------------------------

"""
    warmup(backend, f; reset=nothing, duration=quick_duration())

Repeatedly call `f()` for `duration` seconds to warm up JIT and GPU state.
`reset()` runs before each `f()` if supplied — important for benches
where `f()` is non-idempotent (e.g. sort: without resetting, the second
iter sorts an already-sorted buffer and warms up the wrong code path).
"""
function warmup(backend, f; reset=nothing, duration=quick_duration())
    # Untimed JIT pass: the first call to `f()` triggers CUDA.jl / KA.jl
    # kernel specialization+compile for this exact (kwarg, type) tuple,
    # which can eat 100-300 ms before any real warmup happens. Without
    # this separate pass, the time-based loop's `duration` budget is
    # partly spent compiling, not warming the GPU clock. We then loop
    # for `duration` seconds of *actual* GPU activity.
    reset !== nothing && reset()
    f()
    KA.synchronize(backend)

    start = time()
    while time() - start < duration
        reset !== nothing && reset()
        f()
        KA.synchronize(backend)
    end
end

"""
    gpu_preheat(backend; duration_s=1.0, n=10_000_000)

Saturate the GPU with sustained memory-bound work to lock the clock at
its working steady-state before measurement begins. Call this ONCE at
the top of a bench script (after `include("../init.jl")`) and again
between cells if the per-cell allocation/teardown leaves a long idle
gap.

Why we need this: when we switched kernel timing from `CUDA.@profile`
(which had ~40 ms CUPTI overhead per trial → ~4 s of GPU work during
measurement) to events (~6 ms total for 10 trials), the implicit
"warmup-during-measurement" disappeared. The current `warmup()`
pattern (`f(); sync` per iter) leaves micro-gaps that power-managed
laptop GPUs interpret as idle, keeping the clock from fully ramping.
A 1 s preheat with batched-back-to-back launches puts the GPU in the
same sustained state as the measurement phase before timing starts.

Pattern matches the measurement (batches of `batch` launches between
syncs), so we get the same thermal/clock state without driving
saturated-load throttle (sustained-no-sync for 2+ s does throttle on
RTX1000 mobile — we tested).
"""
function gpu_preheat(backend; duration_s::Real = 1.0,
                              n::Int = 10_000_000,
                              batch::Int = 50)
    x = KA.allocate(backend, Float32, n)
    y = KA.allocate(backend, Float32, n)
    fill!(x, 1.0f0); fill!(y, 0.0f0)
    t0 = time()
    while time() - t0 < duration_s
        for _ in 1:batch
            y .= x .* 2.0f0
        end
        KA.synchronize(backend)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Unified single-loop timing with BATCHED inner calls.
#
# Each "trial" measures `inner` back-to-back `f()` invocations between a
# single event pair AND a single `time_ns()` window. Per-call kernel and
# wall-clock are derived by dividing by `inner`. Kernel and wall-clock
# come from the SAME batched invocation, so they share thermal state.
#
# Why batch (M3 pattern, mirrors AMDGPU.@elapsed inner=20 in matvec
# autotune)? Sync-per-trial events leave ~1-10 µs idle host gaps between
# kernels. On a 35 W laptop GPU that's enough for the clock to drop
# between every trial; medians inflate by 30-80 % at small N. Batching
# `inner=20` calls between syncs holds the GPU saturated for the
# duration of each trial — matches CUB's binary methodology and the
# self-warming effect that `CUDA.@profile` used to provide. xp study:
# xp/mapreduce_gap/v6_event_methodologies.jl.
#
# When `reset !== nothing`, `inner` drops to 1 — we can't batch ops
# whose pre-state we're resetting between calls (e.g. sort).
#
# Returns (k_samples, w_samples), each of per-call μs (already divided
# by `inner`). The public `bench()` wrapper computes median + robust MAD.
#
# Adaptive early-exit: after `min_trials` samples, stop as soon as the
# kernel CV falls below `target_cv`. We use kernel (not wall-clock)
# because it has lower variance and stabilises earlier — a wall-clock
# blip from a GC pause shouldn't end the measurement.
# ---------------------------------------------------------------------------

# Default inner batch for the non-reset path. Tuned for ~20 µs kernels;
# scales fine to ms kernels too (just runs the batch longer).
const _DEFAULT_INNER = 20

@inline function _enough_trials(k_samples::AbstractVector{<:Real},
                                min_trials::Int, target_cv::Real,
                                deadline::Float64)
    length(k_samples) < min_trials && return false
    time() > deadline && return true
    cv = std(k_samples) / max(mean(k_samples), eps())
    return cv <= target_cv
end

if _CUDA_LOADED
    @eval begin
        # Two-pass design:
        #  Pass 1: batched events (inner=20) → accurate kernel_μs.
        #          Saturates GPU, no thermal artefact, ≡ CUB's own number.
        #  Pass 2: inner=1 loop, both event-bracketed AND wall-clock-bracketed
        #          PER TRIAL → dispatch_μs = (wall − kernel_thisTrial).
        #          Same thermal state for both metrics on each sample
        #          → fair comparison even for methods whose dispatch is
        #          ~1 µs (CUDA.mapreduce) where the previous design was
        #          dominated by inter-pass thermal drift noise.
        #  total_μs reported = median(kernel from pass 1) +
        #                      median(dispatch from pass 2).
        function _bench_combined_us(f, backend; trials, reset=nothing,
                                    budget_s=10.0, min_trials=2, target_cv=0.05,
                                    wall_trials=20)
            inner = reset === nothing ? _DEFAULT_INNER : 1
            k_samples = Float64[]
            deadline = time() + budget_s
            for _ in 1:trials
                if reset !== nothing
                    reset()
                    KA.synchronize(backend)
                end
                start_ev = CUDA.CuEvent()
                stop_ev  = CUDA.CuEvent()
                CUDA.record(start_ev)
                for _ in 1:inner; f(); end
                CUDA.record(stop_ev)
                CUDA.synchronize(stop_ev)
                push!(k_samples, CUDA.elapsed(start_ev, stop_ev) * 1e6 / inner)
                _enough_trials(k_samples, min_trials, target_cv, deadline) && break
            end
            dispatch_samples = Float64[]
            for _ in 1:wall_trials
                if reset !== nothing
                    reset()
                    KA.synchronize(backend)
                end
                start_ev = CUDA.CuEvent()
                stop_ev  = CUDA.CuEvent()
                t0 = time_ns()
                CUDA.record(start_ev)
                f()
                CUDA.record(stop_ev)
                CUDA.synchronize(stop_ev)
                wall_us = (time_ns() - t0) / 1000.0
                k_this  = CUDA.elapsed(start_ev, stop_ev) * 1e6
                push!(dispatch_samples, max(0.0, wall_us - k_this))
            end
            # `w_samples` returned for the public API holds per-call TOTAL
            # = kernel_median (from batched) + dispatch (per trial). Each
            # sample carries the same kernel base so its variance reflects
            # the dispatch variance — what we want for the plot's std bar.
            k_med = median(k_samples)
            w_samples = k_med .+ dispatch_samples
            return k_samples, w_samples
        end
    end
elseif _AMD_LOADED
    @eval begin
        function _bench_combined_us(f, backend; trials, reset=nothing,
                                    budget_s=10.0, min_trials=2, target_cv=0.05,
                                    wall_trials=20)
            inner = reset === nothing ? _DEFAULT_INNER : 1
            k_samples = Float64[]
            deadline = time() + budget_s
            for _ in 1:trials
                if reset !== nothing
                    reset()
                    KA.synchronize(backend)
                end
                # AMDGPU.@elapsed returns SECONDS (like Base/CUDA.@elapsed),
                # so → µs is ×1e6 (matches the CUDA branch's `* 1e6`).
                ksec = Float64(AMDGPU.@elapsed begin
                    for _ in 1:inner; f(); end
                end)
                push!(k_samples, ksec * 1e6 / inner)
                _enough_trials(k_samples, min_trials, target_cv, deadline) && break
            end
            dispatch_samples = Float64[]
            for _ in 1:wall_trials
                if reset !== nothing
                    reset()
                    KA.synchronize(backend)
                end
                t0 = time_ns()
                ksec = Float64(AMDGPU.@elapsed f())
                wall_us = (time_ns() - t0) / 1000.0
                push!(dispatch_samples, max(0.0, wall_us - ksec * 1e6))
            end
            k_med = median(k_samples)
            w_samples = k_med .+ dispatch_samples
            return k_samples, w_samples
        end
    end
else
    @eval begin
        function _bench_combined_us(f, backend; trials, reset=nothing,
                                    budget_s=10.0, min_trials=2, target_cv=0.05)
            for _ in 1:trials
                reset !== nothing && reset()
                f()
            end
            return fill(NaN, trials), fill(NaN, trials)
        end
    end
end

"""
    bench(name, f; backend, reset=nothing,
          trials=quick_trials(), budget_s=10.0,
          min_trials=2, target_cv=0.05,
          duration=quick_duration())
        -> NamedTuple{mean_kernel_μs, std_kernel_μs, mean_total_μs, std_total_μs}

Unified single-loop benchmark of `f()`. Kernel time (via events) and host
wall-clock (via `time_ns()`) are captured from the SAME `f()` invocation
per trial, so they share thermal state — no inter-phase gap that can let
a laptop GPU's clock ramp down between measurements.

1. **Warmup** for `duration` seconds (default 500 ms; 100 ms in quick mode).
2. **Measurement** — per trial:
   - `reset()` + sync (outside timed region) if provided
   - capture `time_ns()` → record start event → `f()` → record stop
     event → sync → harvest both timings
3. Run up to `trials` (default 50; 5 in quick) and up to `budget_s`
   seconds. Adaptive early-exit: after `min_trials`, stop when kernel
   CV ≤ `target_cv` (default 5 %).

`reset` runs *outside* both the event bracket and the `time_ns()` window,
so neither kernel nor total time count the prep cost. Apple-to-apple with
the standalone CUB binary, which also brackets only the operation under
test.

Set `KF_QUICK_BENCH=1` to drop to 100 ms warmup + 5 trials.
Set `target_cv = 0` to disable the early-CV stop.
"""
function bench(name, f;
    backend,
    reset      = nothing,
    trials     = quick_trials(),
    budget_s   = quick_bench() ? 2.0 : 10.0,
    min_trials = 2,
    target_cv  = 0.05,
    duration   = quick_duration(),
)
    warmup(backend, f; reset=reset, duration=duration)
    println("=== $name ===")

    k_samples, w_samples = _bench_combined_us(f, backend;
        trials=trials, reset=reset,
        budget_s=budget_s, min_trials=min_trials, target_cv=target_cv)

    # Robust statistics — MEDIAN for central tendency, MAD × 1.4826 for
    # spread. Immune to single-trial outliers (GC pause, lazy alloc,
    # thermal blip). Field names kept (`mean_*` / `std_*`) so the CSV
    # schema and plot helpers don't change.
    robust_std(s) = 1.4826 * median(abs.(s .- median(s)))
    mean_kernel_μs = median(k_samples); std_kernel_μs = robust_std(k_samples)
    mean_total_μs  = median(w_samples); std_total_μs  = robust_std(w_samples)

    n = length(k_samples)
    println("Kernel time: $(round(mean_kernel_μs; digits=2)) ± $(round(std_kernel_μs; digits=2)) μs (n=$n)   [median]")
    println("Total time:  $(round(mean_total_μs;  digits=2)) ± $(round(std_total_μs;  digits=2)) μs (n=$n)   [median]")

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
    UInt32 => "uint32",
    UInt64 => "uint64",
    UInt8 => "uint8",
)

"""
    cub_exe(default) -> String

CUB executable path. `ENV["KF_CUB_EXE"]` overrides the build-tree
default — useful for CI or when the build output is moved.
"""
cub_exe(default::String) = get(ENV, "KF_CUB_EXE", default)

"""
    run_cub_benchmark(exe; N, iterations, warmup_ms, dtype, extra_flags="") -> JSON3 result

Run an external CUB benchmark executable and return parsed JSON.
"""
function run_cub_benchmark(exe::String;
    N::Int=100_000_000,
    iterations::Int=100,
    warmup_ms::Real=500,      # CUB binary needs its own full warmup — the
                              # Julia-side gpu_preheat does NOT transfer to
                              # the subprocess (separate CUDA context).
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
function bench_cub(exe::String, n::Int, ::Type{T}; trials::Int=100,
                   warmup_ms::Real=500, extra_flags::String="",
                   label::String="CUB") where T
    println("=== $label ===")
    results = run_cub_benchmark(exe; N=n, iterations=trials, warmup_ms=warmup_ms,
                                  dtype=T, extra_flags)

    if isempty(results)
        @warn "$label returned empty results for dtype=$T, n=$n — returning NaN"
        return (; mean_kernel_μs=NaN, std_kernel_μs=NaN, mean_total_μs=NaN, std_total_μs=NaN)
    end

    mean_kernel_μs = results[1]["mean_ms"] * 1000
    std_kernel_μs = results[1]["std_ms"] * 1000

    println("Kernel time: $(mean_kernel_μs) ± $(std_kernel_μs) μs (n=$trials)")
    return (; mean_kernel_μs, std_kernel_μs, mean_total_μs=mean_kernel_μs, std_total_μs=std_kernel_μs)
end

"""
    bench_cub_or_nan(exe, n, T; safety_factor=3.0, kw...) -> NamedTuple

Like `bench_cub` but returns NaN stats if any of:
- the executable is missing
- the subprocess fails / OOMs / hits transient `device busy`
- **free GPU memory is too low to safely run the CUB subprocess**
  (the host process holds `src`, `tmp`, `dst`, etc., and a CUB subprocess
  trying to allocate ~`safety_factor * n * sizeof(T)` on top can OOM-kill
  the parent Julia process — observed at n=1e9 Float32 on a 6 GB card).

Set `safety_factor` lower (e.g. 1.5) if you've already freed Julia-side
buffers via `GC.gc(true); CUDA.reclaim()` immediately before the call.
"""
function bench_cub_or_nan(exe::String, n::Int, ::Type{T};
                          safety_factor::Real=3.0, kw...) where T
    if isempty(exe) || !isfile(exe)
        isempty(exe) || @warn "CUB executable not found: $exe"
        return (; mean_kernel_μs=NaN, std_kernel_μs=NaN, mean_total_μs=NaN, std_total_μs=NaN)
    end
    if _CUDA_LOADED
        required = safety_factor * n * sizeof(T)
        free, _total = CUDA.Mem.info()
        if free < required
            @warn "CUB skipped (insufficient free GPU memory)" n T required free
            return (; mean_kernel_μs=NaN, std_kernel_μs=NaN, mean_total_μs=NaN, std_total_μs=NaN)
        end
    end
    try
        return bench_cub(exe, n, T; kw...)
    catch err
        @warn "CUB subprocess failed — substituting NaN" exception=(err, catch_backtrace()) n T
        return (; mean_kernel_μs=NaN, std_kernel_μs=NaN, mean_total_μs=NaN, std_total_μs=NaN)
    end
end

"""
    bench_rocprim_or_nan(exe, n, T; kw...) -> NamedTuple

AMD analogue of `bench_cub_or_nan`: runs the hipCUB/rocPRIM benchmark binary (same
CLI + JSON contract as the CUB binaries, parsed by `run_cub_benchmark`) and returns
NaN stats if the executable is missing or the subprocess fails. Reports under the
`rocPRIM` label. No GPU free-memory guard — the MI300X has 192 GB so the OOM-killer
concern that motivates `bench_cub_or_nan`'s `safety_factor` check does not apply.
"""
function bench_rocprim_or_nan(exe::String, n::Int, ::Type{T}; kw...) where T
    if isempty(exe) || !isfile(exe)
        isempty(exe) || @warn "rocPRIM executable not found: $exe"
        return (; mean_kernel_μs=NaN, std_kernel_μs=NaN, mean_total_μs=NaN, std_total_μs=NaN)
    end
    try
        return bench_cub(exe, n, T; label="rocPRIM", kw...)
    catch err
        @warn "rocPRIM subprocess failed — substituting NaN" exception=(err, catch_backtrace()) n T
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