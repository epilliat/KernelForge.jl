#=
GEMM Performance Benchmarking Script
====================================
Compares KernelForge.gemm against cuBLAS / rocBLAS (via `A * B`) for the
matrix–matrix product across all four TRANSPOSE STATES (NN / NT / TN / TT).

Design (agreed 2026-07-26; transpose states added 2026-08-03):
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
- STATE ∈ {NN,NT,TN,TT}: tA/tB=:T stores the operand TRANSPOSED and passes it as an
  Adjoint (the shipped wrapper), the RowMajor staging path. M,N,K are the LOGICAL
  product dims in every state (so FLOPs match across states); the physical arrays
  are M×K/K×M (A) and K×N/N×K (B). Vendor gets the SAME Adjoint operands (`A*B` →
  cuBLAS/rocBLAS transpose). NN is the back-compat default row.
- Types: Float16 (→F32 acc), Float32, Float64.
- Methodology mirrors matvec: `bench` gives mean_kernel_μs (device events) +
  mean_total_μs (wall). Vendor = `A * B` (cuBLAS / rocBLAS gemm). KF = default
  `gemm` (auto-resolves to the tensor-core family where supported).
- CSV → results/<GPU>/gemm.csv in the sibling benchmarks repo layout, with a
  `state` column. INCREMENTAL + RESUMABLE: each (shape,state,type) cell is appended
  as it completes and a re-run skips cells already present — so a heat pause / crash
  never loses banked work. ENV filters `KF_GEMM_STATES` (e.g. "NT,TN,TT") and
  `KF_GEMM_LS` (e.g. "2048") restrict the sweep for chunked runs.
=#
include("../init.jl")
include("../vendor_int8.jl")     # `vendor_gemm_baseline` (cuBLAS / rocBLAS gemm|gemm_ex)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
base_sizes = [1024, 2048, 4096]     # L: square edge → F = L³ FLOP budget
aspect_s   = 8                       # rectangular stretch factor
types      = [Float16, Float32, Float64]
# (state, tA, tB) — NN first (back-compat default), then the three transpose states.
all_states = [(:NN, :N, :N), (:NT, :N, :T), (:TN, :T, :N), (:TT, :T, :T)]

# Optional chunking for heat-limited runs: KF_GEMM_STATES="NT,TN" / KF_GEMM_LS="2048".
gemm_states = let sel = get(ENV, "KF_GEMM_STATES", "")
    isempty(sel) ? all_states :
        filter(t -> String(t[1]) in split(sel, ','), all_states)
end
sel_Ls = let sel = get(ENV, "KF_GEMM_LS", "")
    isempty(sel) ? base_sizes : parse.(Int, split(sel, ','))
end

r64(x) = max(64, 64 * round(Int, x / 64))   # nearest multiple of 64, ≥ 64

# 6 canonical (name, M, N, K) at fixed F = L³ — LOGICAL product dims (state-independent).
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

# Build a LOGICAL r×c operand in the requested layout: :N → a plain r×c array;
# :T → a c×r array passed as its Adjoint (the shipped transpose wrapper), so the
# logical operand is still r×c but physically stored transposed (RowMajor staging).
make_operand(::Type{T}, r, c, flag) where {T} =
    flag === :N ? fill!(AT{T}(undef, r, c), one(T)) :
                  adjoint(fill!(AT{T}(undef, c, r), one(T)))

# ---------------------------------------------------------------------------
# Benchmark runner — one (shape, type) cell
# ---------------------------------------------------------------------------
function run_gemm_benchmark(name, state, tA, tB, M, N, K, ::Type{T}) where {T}
    flops = Float64(M) * N * K
    vlabel = vendor_gemm_label(T)       # cuBLAS / rocBLAS gemm|gemm_ex
    println("\n" * "="^60)
    println("GEMM $name/$state: M=$M N=$N K=$K  T=$T   (M·N·K = $(flops))")
    println("="^60)

    # Allocate inside a guard: the extreme rectangular cells at the largest budget
    # can exceed device memory (e.g. `outer` C = M·N). Record NaN and skip rather
    # than crash before the CSV.
    local A, B
    try
        A = make_operand(T, M, K, tA)   # logical M×K
        B = make_operand(T, K, N, tB)   # logical K×N
    catch e
        @warn "alloc failed (OOM?) — recording NaN" name state M N K T
        return [(; shape = name, state = string(state), M, N, K, flops, type = string(T), method = m,
                 mean_kernel_μs = NaN, std_kernel_μs = NaN,
                 mean_total_μs = NaN, std_total_μs = NaN)
                for m in ("KernelForge", vlabel)]
    end

    _, vcall = vendor_gemm_baseline(T, A, B)
    rows = NamedTuple[]
    for (method, call) in [
        # Like-for-like output precision. The vendor `A*B` (cuBLAS/rocBLAS) accumulates
        # in F32 but STORES the input eltype (F16→F16); KF's F16×F16 default returns F32
        # (higher quality, but 2× the C-write bytes), which unfairly inflated the gap on
        # memory-bound shapes (`outer`/`tall`). `outT=T` makes KF store `T` too — the
        # standard mixed-precision GEMM (F16 in, F32 accumulate, F16 out), matching the
        # vendor byte-for-byte. No-op for F32/F64. KF ALSO offers the F32-out F16 GEMM
        # (`KF.gemm(A,B)`, its default) at higher quality — not scored here as the vendor
        # has no matching F32-out F16 mode.
        ("KernelForge", () -> KF.gemm(A, B; outT = T)),
        (vlabel,        vcall),
    ]
        s = try
            bench("$method [$name $M×$N×$K $T]", call; backend)
        catch e
            @warn "bench failed — recording NaN" method name exception = (e, catch_backtrace())
            (mean_kernel_μs = NaN, std_kernel_μs = NaN, mean_total_μs = NaN, std_total_μs = NaN)
        end
        push!(rows, (; shape = name, state = string(state), M, N, K, flops, type = string(T), method,
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
# Collect — INCREMENTAL + RESUMABLE (append per cell; skip cells already banked)
# ---------------------------------------------------------------------------
mkpath(RESULT_DIR)
csv_path = joinpath(RESULT_DIR, "gemm.csv")

# Cells already in the CSV (shape,state,M,N,K,type) — skip on re-run. A pre-`state`
# (NN-only) CSV is a stale schema: ignore it and overwrite fresh. When KF_GEMM_STATES
# is set explicitly, those states are a REFRESH: their old rows are dropped (CSV
# rewritten keeping the others) and re-measured — so `KF_GEMM_STATES=TT` cleanly
# updates just the TT rows in place without duplicating them.
refresh = isempty(get(ENV, "KF_GEMM_STATES", "")) ? Set{String}() :
          Set(String(t[1]) for t in gemm_states)
done = Set{NTuple{6,String}}()
has_prev = false
if isfile(csv_path)
    cols = propertynames(CSV.File(csv_path; limit = 1))
    if :state in cols
        prev = DataFrame(CSV.File(csv_path))
        if !isempty(refresh)
            keep = prev[.!in.(string.(prev.state), Ref(refresh)), :]
            CSV.write(csv_path, keep)                 # drop the refreshed states in place
            println("Refreshing states $(collect(refresh)): dropped $(nrow(prev) - nrow(keep)) old rows")
            prev = keep
        end
        has_prev = nrow(prev) > 0
        for r in eachrow(prev)
            push!(done, (string(r.shape), string(r.state), string(r.M), string(r.N),
                         string(r.K), string(r.type)))
        end
        println("Resuming: $(length(done) ÷ 2) cells already in $csv_path")
    else
        println("Stale NN-only CSV (no `state` column) — overwriting fresh.")
    end
end

hdr_written = has_prev
for L in sel_Ls, (name, M, N, K) in gemm_shapes(L, aspect_s), (state, tA, tB) in gemm_states, T in types
    key = (name, string(state), string(M), string(N), string(K), string(T))
    key in done && continue
    rows = run_gemm_benchmark(name, state, tA, tB, M, N, K, T)
    # Append immediately so a heat pause / crash never loses a completed cell.
    CSV.write(csv_path, DataFrame(rows); append = hdr_written, writeheader = !hdr_written)
    global hdr_written = true
end
println("\nResults saved to: $csv_path\n")

# ---------------------------------------------------------------------------
# Display (read back the full CSV)
# ---------------------------------------------------------------------------
df = DataFrame(CSV.File(csv_path))
df_display = transform(df,
    [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs] .=>
        (x -> round.(x; digits = 2)) .=>
            [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs])
println("=== GEMM Benchmark Results — GPU: $GPU_TAG ===")
show(df_display, allrows = true, allcols = true)
println()
