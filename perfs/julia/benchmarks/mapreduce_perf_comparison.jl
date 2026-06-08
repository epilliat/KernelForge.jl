#=
MapReduce Performance Benchmarking Script
==========================================
Compares KernelForge.mapreduce! against CUDA.mapreduce and AcceleratedKernels.mapreduce
across different data types and configurations.
Methodology (inherits `bench_utils.jl`):
- 500 ms warmup, then two phases: 10-trial kernel timing via backend
  events (CUDA.CuEvent or AMDGPU.@elapsed — backend-agnostic), then
  100-trial host wall-clock via time_ns() + KA.synchronize.
- mapreduce closures are idempotent (`src` invariant, `dst` overwritten
  with the same result each trial), so no `reset` is needed — the
  default `reset = nothing` is correct.
- Results saved to results/<gpu_short>/mapreduce.csv.
=#
include("../init.jl")


const DEFAULT_CUB_EXE = joinpath(@__DIR__, "../../cuda_cpp/cub_nvcc/bin/cub_sum_benchmark")

# Session-level GPU preheat — see comment in sort_perf_comparison.jl.
println("→ GPU preheat …")
gpu_preheat(backend; duration_s = 1.0)
println("  done.")

# ---------------------------------------------------------------------------
# Benchmark runner — returns a vector of NamedTuples
# ---------------------------------------------------------------------------

"""
    run_mapreduce_benchmarks(n, label_T; f, init, cub_exe, cub_T)

Runs bench() for KernelForge, CUDA/generic mapreduce, AcceleratedKernels, and CUB on `src`.
`cub_T` is the dtype passed to CUB (e.g. UInt8 as proxy for UnitFloat8).
Returns a Vector of row NamedTuples for the results DataFrame.
"""
function run_mapreduce_benchmarks(src::AT{T}, label_T::String, n::Int;
  f=identity, init=zero(T),
  cub_exe::String=DEFAULT_CUB_EXE, cub_T::Type=T) where T

  rows = NamedTuple[]
  tmp_kf = KF.@allocate mapreduce1d(f, +, src)
  temp_ak = similar(src, cld(length(src), 10)) 

  dst = AT{Float32}(undef, 1)
  for (name, method, call) in [
    (has_cuda() ? "CUDA [$label_T]" : "Base [$label_T]",
    has_cuda() ? "CUDA" : "Base",
    () -> mapreduce(f, +, src)),
    ("AcceleratedKernels [$label_T]", "AK",
    () -> AcceleratedKernels.mapreduce(f, +, src; init, temp=temp_ak)),
    ("KernelForge [$label_T]", "KernelForge",
      () -> KernelForge.mapreduce!(f, +, dst,src; tmp=tmp_kf)),
  ]
    s = bench(name, call; backend)
    push!(rows, (; n, type=label_T, method,
      s.mean_kernel_μs, s.std_kernel_μs,
      s.mean_total_μs, s.std_total_μs))
  end

  if has_cuda()
    # Free Julia-side GPU buffers BEFORE launching the CUB subprocess —
    # at n=1e9 Float32 the parent holds ~4.4 GB (src + tmp_kf + temp_ak)
    # and a CUB subprocess allocating its own ~4 GB pushes total over the
    # 6 GB card limit, triggering the kernel OOM killer on the parent
    # Julia process (silent crash, no Julia-level error to catch).
    tmp_kf = nothing
    temp_ak = nothing
    dst = nothing
    GC.gc(true); CUDA.reclaim()
    # `bench_cub_or_nan` also checks free GPU mem and returns NaN if the
    # CUB subprocess would not fit (safety_factor=1.5 once we've freed).
    s = bench_cub_or_nan(cub_exe, n, cub_T; safety_factor=1.5)
    push!(rows, (; n, type=label_T, method="CUB",
      s.mean_kernel_μs, s.std_kernel_μs,
      mean_total_μs=NaN, std_total_μs=NaN))
  end

  return rows
end


# warmup
n = 100000000
src = fill!(AT{Float32}(undef, n), one(Float32))
src_u8 = fill!(AT{UInt8}(undef, n), one(UInt8))


#%%


# # Simple profiling example (without warmup here which gives slower results)

# CUDA.@profile CUDA.mapreduce(identity, +, src)
# CUDA.@profile AcceleratedKernels.mapreduce(identity, +, src; init=0.0f0)
# CUDA.@profile KernelForge.mapreduce(identity, +, src)

# src_uf8 = fill!(AT{UnitFloat8}(undef, n), one(UnitFloat8))
# u(x) = Float32(x)::Float32
# CUDA.@profile KernelForge.mapreduce(identity, +, src_uf8)
# CUDA.@profile mapreduce(u, +, src_uf8)
#%%

# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------

# n=1e9 needs ~4 GB (Float32) just for src; with KF/AK temp + CUB
# scratch it OOMs on a 6 GB card. The sweep keeps it but the script
# catches OOM per-cell (see try/catch in the loop below) so smaller
# cards still complete with NaN rows at the largest sizes.
sizes = [10^6, 10^7, 10^8, 10^9]
types = [Float32, Float64, UnitFloat8, UInt8]

all_rows = NamedTuple[]

for n in sizes, T in types
  println("\n==== n = $n, T = $T ====\n")
  gpu_preheat(backend; duration_s = 0.3)
  try
  if T === UnitFloat8
    src = fill!(AT{T}(undef, n), one(T))
    f(x) = Float32(x)
    label = "UnitFloat8→Float32"
    for (name, method, call) in [
      (has_cuda() ? "CUDA [$label]" : "Base [$label]",
        has_cuda() ? "CUDA" : "Base",
        () -> mapreduce(f, +, src)),
      ("AcceleratedKernels [$label]", "AK",
        () -> AcceleratedKernels.mapreduce(f, +, src; init=T(0))),
      ("KernelForge [$label]", "KernelForge",
        () -> KernelForge.mapreduce(f, +, src)),
    ]
      s = bench(name, call; backend)
      push!(all_rows, (; n, type=label, method,
        s.mean_kernel_μs, s.std_kernel_μs,
        s.mean_total_μs, s.std_total_μs))
    end
    if has_cuda()
      # Same OOM guard as run_mapreduce_benchmarks — free src before CUB
      # so the subprocess has room on the device.
      src = nothing
      GC.gc(true); CUDA.reclaim()
      s = bench_cub_or_nan(DEFAULT_CUB_EXE, n, UInt8; safety_factor=1.5)
      push!(all_rows, (; n, type=label, method="CUB",
        s.mean_kernel_μs, s.std_kernel_μs,
        mean_total_μs=NaN, std_total_μs=NaN))
    end
  else
    src = fill!(AT{T}(undef, n), one(T))
    append!(all_rows, run_mapreduce_benchmarks(src, string(T), n))
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
csv_path = joinpath(RESULT_DIR, "mapreduce.csv")
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

println("=== MapReduce Benchmark Results — GPU: $GPU_TAG ===")

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