#=
GEMM Performance Benchmarking Script
====================================
Compares KernelForge.gemm against cuBLAS / rocBLAS (via `A * B`) for the
matrix–matrix product, NN (non-transposed) only for now.

Design (agreed 2026-07-26):
- GEMM has THREE dims (M,N,K); we fix the ARITHMETIC WORK  F = M·N·K  (GEMM is
  compute-bound, so equal FLOPs is the fair "same work" comparison) and sweep 6
  canonical SHAPES at each F: square + 5 rectangular extremes. Base L = F^(1/3),
  aspect factor s; every dim rounded to a multiple of 64 (MMA/alignment friendly),
  so the realised F drifts slightly from the nominal — the true M·N·K is logged.
- 6 shapes (name, M, N, K), all ≈ L³:
    square      M=N=K=L
    outer       M=N=L·s,   K=L/s²        (rank-short: wide inputs, thin contraction)
    inner       M=N=L/s,   K=L·s²        (deep: long contraction, small inputs)
    A_thin      M=L·s, N=L/s², K=L·s     (N small — GEMV-adjacent)
    B_thin      M=L/s², N=L·s, K=L·s     (M small)
    tall        M=L·s², N=L/s, K=L/s     (M large, thin N,K)
- Types: Float16 (→F32 acc), Float32, Float64.  Transpose: NN only.
- Methodology mirrors matvec: `bench` gives mean_kernel_μs (device events) +
  mean_total_μs (wall). Vendor = `A * B` (cuBLAS / rocBLAS gemm). KF = default
  `gemm` (auto-resolves to the tensor-core family where supported).
- CSV → results/<GPU>/gemm.csv in the sibling benchmarks repo layout.
=#
include("../init.jl")
include("../vendor_int8.jl")     # `vendor_gemm_baseline` (cuBLAS / rocBLAS gemm|gemm_ex)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
base_sizes = [1024, 2048, 4096]     # L: square edge → F = L³ FLOP budget
aspect_s   = 8                       # rectangular stretch factor
types      = [Float16, Float32, Float64]

r64(x) = max(64, 64 * round(Int, x / 64))   # nearest multiple of 64, ≥ 64

# 6 canonical (name, M, N, K) at fixed F = L³.
function gemm_shapes(L::Int, s::Int)
    return [
        ("square", r64(L),     r64(L),      r64(L)),
        ("outer",  r64(L * s), r64(L * s),  r64(L / s^2)),
        ("inner",  r64(L / s), r64(L / s),  r64(L * s^2)),
        ("A_thin", r64(L * s), r64(L / s^2), r64(L * s)),
        ("B_thin", r64(L / s^2), r64(L * s), r64(L * s)),
        ("tall",   r64(L * s^2), r64(L / s), r64(L / s)),
    ]
end

# ---------------------------------------------------------------------------
# Benchmark runner — one (shape, type) cell
# ---------------------------------------------------------------------------
function run_gemm_benchmark(name, M, N, K, ::Type{T}) where {T}
    flops = Float64(M) * N * K
    vlabel = vendor_gemm_label(T)       # cuBLAS / rocBLAS gemm|gemm_ex
    println("\n" * "="^60)
    println("GEMM $name: M=$M N=$N K=$K  T=$T   (M·N·K = $(flops))")
    println("="^60)

    # Allocate inside a guard: the extreme rectangular cells at the largest budget
    # can exceed device memory (e.g. `outer` C = M·N). Record NaN and skip rather
    # than crash before the CSV.
    local A, B
    try
        A = fill!(AT{T}(undef, M, K), one(T))
        B = fill!(AT{T}(undef, K, N), one(T))
    catch e
        @warn "alloc failed (OOM?) — recording NaN" name M N K T
        return [(; shape = name, M, N, K, flops, type = string(T), method = m,
                 mean_kernel_μs = NaN, std_kernel_μs = NaN,
                 mean_total_μs = NaN, std_total_μs = NaN)
                for m in ("KernelForge", vlabel)]
    end

    _, vcall = vendor_gemm_baseline(T, A, B)
    rows = NamedTuple[]
    for (method, call) in [
        ("KernelForge", () -> KF.gemm(A, B)),
        (vlabel,        vcall),
    ]
        s = try
            bench("$method [$name $M×$N×$K $T]", call; backend)
        catch e
            @warn "bench failed — recording NaN" method name exception = (e, catch_backtrace())
            (mean_kernel_μs = NaN, std_kernel_μs = NaN, mean_total_μs = NaN, std_total_μs = NaN)
        end
        push!(rows, (; shape = name, M, N, K, flops, type = string(T), method,
            s.mean_kernel_μs, s.std_kernel_μs, s.mean_total_μs, s.std_total_μs))
        GC.gc(true); has_cuda() && CUDA.reclaim()
    end
    return rows
end

# ---------------------------------------------------------------------------
# Warm-up (compile both paths)
# ---------------------------------------------------------------------------
let T = Float32, A = fill!(AT{T}(undef, 256, 256), one(T)), B = fill!(AT{T}(undef, 256, 256), one(T))
    KF.gemm(A, B); vendor_gemm_baseline(T, A, B)[2](); KA.synchronize(backend)
end

# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------
all_rows = NamedTuple[]
for L in base_sizes, (name, M, N, K) in gemm_shapes(L, aspect_s), T in types
    append!(all_rows, run_gemm_benchmark(name, M, N, K, T))
end

# ---------------------------------------------------------------------------
# Save + display
# ---------------------------------------------------------------------------
df = DataFrame(all_rows)
mkpath(RESULT_DIR)
csv_path = joinpath(RESULT_DIR, "gemm.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")

df_display = transform(df,
    [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs] .=>
        (x -> round.(x; digits = 2)) .=>
            [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs])
println("=== GEMM Benchmark Results — GPU: $GPU_TAG ===")
show(df_display, allrows = true, allcols = true)
println()
