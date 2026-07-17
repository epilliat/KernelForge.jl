#=
Sample sort (general-comparator path) benchmark
==============================================
Compares the GENERAL-COMPARATOR sort path — `KernelForge.sample_sort`, the
`algorithm=:sample` branch of `sort!` — against AcceleratedKernels' merge sort:

    1. KernelForge.sample_sort    — GPU sample sort (this package, general `lt`)
    2. AcceleratedKernels.sort!   — merge sort (takes `lt`/`by`, so the
                                    comparator/struct cells are apples-to-apples)
    3. KernelForge.sort! (radix)  — reference FLOOR on the plain-numeric cells
                                    only (radix can't take `lt`); shows what the
                                    general path gives up for generality.

The bar (maintainer's target): **beat AK on Float32/Float64**, and stay CORRECT
for every bitstype incl. user structs + arbitrary comparators. CUB/rocPRIM are
deliberately NOT benched here — they are radix, a different bar (see sort.csv).

Cells: N ∈ {1e5, 1e6, 1e7, 1e8} × {Float32, Float64}, plus two generality cells
that price what the comparator path actually costs:
  - `Float32_gt` : Float32 with `lt = (a,b) -> a > b` (descending)
  - `KV_lt`      : a 2-field struct sorted by its first field with a custom `lt`

Methodology inherits `bench_utils.jl` (warmup, kernel time via events + host
wall-clock from the SAME call, adaptive early-exit, `reset` outside both timers).
Notes specific to THIS path:
  - `sample_sort` does NOT mutate `src` (it copies into its workspace and returns
    a view), so KF needs NO per-trial reset; AK's sort IS in-place and does.
  - The workspace is pre-allocated once per cell and passed as `ws=` — otherwise
    every call allocates buf_a/buf_b/samples/slot_cache + a large block_counts.
  - CORRECTNESS GATE is `issorted(result)` AND a multiset match. A multiset check
    ALONE (`sort(dst)==sort(src)`) passes for ANY permutation — that is exactly
    what once hid the v22-v28 binsearch bug where the sort wasn't sorting.

Results → `perfs/julia/results/<GPU_TAG>/sample_sort.csv`, canonical 1D schema:
    n, type, method, mean_kernel_μs, std_kernel_μs, mean_total_μs, std_total_μs
=#
include("../init.jl")

using AcceleratedKernels

# Generality cell: a 2-field isbits struct with a custom comparator. This is the
# cell where the "correct for any struct" requirement is actually priced.
struct KV
    k::Float32
    v::Int32
end
Base.rand(::Type{KV}, n::Int) = [KV(rand(Float32), Int32(i)) for i in 1:n]

println("→ GPU preheat …")
gpu_preheat(backend; duration_s = 1.0)
println("  done.")

# `issorted` + multiset — see the correctness note in the header. Never multiset alone.
function _check_sorted(res_cpu::Vector, src_cpu::Vector, lt)
    ok_order = issorted(res_cpu; lt = lt)
    ok_perm  = sort(res_cpu; lt = lt) == sort(src_cpu; lt = lt) && length(res_cpu) == length(src_cpu)
    return ok_order && ok_perm
end

"""
    run_sample_sort_benchmarks(::Type{T}, n; lt, label, radix_ref) -> Vector{NamedTuple}

Bench KF.sample_sort vs AK.sort! for one cell. `lt === nothing` means the natural
order (both sides use their default). `radix_ref` adds the radix floor row (only
meaningful for plain numeric cells with the natural order).
"""
function run_sample_sort_benchmarks(::Type{T}, n::Int; lt = nothing,
                                    label::String = string(T),
                                    radix_ref::Bool = false) where T
    src_cpu = T <: AbstractFloat ? randn(T, n) : rand(T, n)
    src_init = AT(src_cpu)
    src_ss = copy(src_init)      # sample_sort reads this (never mutates it)
    src_ak = copy(src_init)
    temp_ak = similar(src_ak)
    ws = KF.sample_sort_workspace(T, n; backend)

    lt_kf = lt
    lt_ak = lt === nothing ? isless : lt
    cmp_check = lt === nothing ? isless : lt

    rows = NamedTuple[]

    # --- correctness FIRST (a fast wrong kernel is not a benchmark result) ---
    ok_kf = ok_ak = false
    try
        res = KF.sample_sort(src_ss; lt = lt_kf, ws)
        KA.synchronize(backend)
        ok_kf = _check_sorted(Array(res), src_cpu, cmp_check)
    catch err
        @warn "KF.sample_sort FAILED" label n err = sprint(showerror, err)
    end
    try
        copyto!(src_ak, src_init)
        AcceleratedKernels.sort!(src_ak; lt = lt_ak, temp = temp_ak)
        KA.synchronize(backend)
        ok_ak = _check_sorted(Array(src_ak), src_cpu, cmp_check)
    catch err
        @warn "AK.sort! FAILED" label n err = sprint(showerror, err)
    end
    println("  correctness: KF=$(ok_kf)  AK=$(ok_ak)")
    ok_kf || @warn "KF.sample_sort INCORRECT — timing below is meaningless" label n

    methods = Any[
        ("KernelForge.sample_sort [$label]", "KernelForge", nothing,
            () -> KF.sample_sort(src_ss; lt = lt_kf, ws)),
        ("AcceleratedKernels.sort! [$label]", "AK",
            () -> copyto!(src_ak, src_init),
            () -> AcceleratedKernels.sort!(src_ak; lt = lt_ak, temp = temp_ak)),
    ]
    if radix_ref
        src_kf = copy(src_init)
        tmp_kf = KF.@allocate sort!(src_kf)
        push!(methods, ("KernelForge.sort! radix [$label]", "KernelForge_radix",
            () -> copyto!(src_kf, src_init),
            () -> KF.sort!(src_kf; tmp = tmp_kf)))
    end

    for (name, method, reset, call) in methods
        s = try
            bench(name, call; backend, reset)
        catch err
            @warn "bench failed" name err = sprint(showerror, err)
            (; mean_kernel_μs = NaN, std_kernel_μs = NaN,
               mean_total_μs = NaN, std_total_μs = NaN)
        end
        push!(rows, (; n, type = label, method,
            s.mean_kernel_μs, s.std_kernel_μs, s.mean_total_μs, s.std_total_μs))
    end
    return rows
end

# ---------------------------------------------------------------------------
# Sweep — small N included ON PURPOSE: the general path pays ~30 launches +
# several device syncs per level, so the KF/AK crossover is expected at small N.
# ---------------------------------------------------------------------------
sizes = [10^5, 10^6, 10^7, 10^8]

all_rows = NamedTuple[]
for n in sizes
    for (T, lt, label, radix_ref) in Any[
            (Float32, nothing, "Float32", true),
            (Float64, nothing, "Float64", true),
            (Float32, (a, b) -> a > b, "Float32_gt", false),
        ]
        println("\n" * "=" ^ 60)
        println("Sample sort: n=$n, cell=$label")
        println("=" ^ 60)
        gpu_preheat(backend; duration_s = 0.3)
        try
            append!(all_rows, run_sample_sort_benchmarks(T, n; lt, label, radix_ref))
        catch err
            msg = sprint(showerror, err)
            if occursin("Out of GPU memory", msg) || occursin("hipErrorOutOfMemory", msg)
                @warn "Skipping cell — out of GPU memory" n label
                for m in ("KernelForge", "AK")
                    push!(all_rows, (; n, type = label, method = m,
                        mean_kernel_μs = NaN, std_kernel_μs = NaN,
                        mean_total_μs = NaN, std_total_μs = NaN))
                end
                GC.gc(true); has_cuda() && CUDA.reclaim()
            else
                rethrow(err)
            end
        end
    end
    # Struct cell — the generality gate. Kept separate: `randn` doesn't apply.
    println("\n" * "=" ^ 60)
    println("Sample sort: n=$n, cell=KV_lt (struct + custom lt)")
    println("=" ^ 60)
    try
        src_cpu = rand(KV, n)
        src_init = AT(src_cpu)
        src_ss = copy(src_init); src_ak = copy(src_init); temp_ak = similar(src_ak)
        ws = KF.sample_sort_workspace(KV, n; backend)
        ltkv = (a, b) -> a.k < b.k
        ok_kf = ok_ak = false
        try
            res = KF.sample_sort(src_ss; lt = ltkv, ws); KA.synchronize(backend)
            ok_kf = issorted(Array(res); lt = ltkv)
        catch err; @warn "KF struct FAILED" n err = sprint(showerror, err); end
        try
            copyto!(src_ak, src_init)
            AcceleratedKernels.sort!(src_ak; lt = ltkv, temp = temp_ak); KA.synchronize(backend)
            ok_ak = issorted(Array(src_ak); lt = ltkv)
        catch err; @warn "AK struct FAILED" n err = sprint(showerror, err); end
        println("  correctness: KF=$(ok_kf)  AK=$(ok_ak)")
        for (name, method, reset, call) in Any[
            ("KernelForge.sample_sort [KV_lt]", "KernelForge", nothing,
                () -> KF.sample_sort(src_ss; lt = ltkv, ws)),
            ("AcceleratedKernels.sort! [KV_lt]", "AK",
                () -> copyto!(src_ak, src_init),
                () -> AcceleratedKernels.sort!(src_ak; lt = ltkv, temp = temp_ak)),
        ]
            s = try
                bench(name, call; backend, reset)
            catch err
                @warn "bench failed" name err = sprint(showerror, err)
                (; mean_kernel_μs = NaN, std_kernel_μs = NaN,
                   mean_total_μs = NaN, std_total_μs = NaN)
            end
            push!(all_rows, (; n, type = "KV_lt", method,
                s.mean_kernel_μs, s.std_kernel_μs, s.mean_total_μs, s.std_total_μs))
        end
    catch err
        @warn "KV cell skipped" n err = sprint(showerror, err)
    end
end

df = DataFrame(all_rows)
csv_path = joinpath(RESULT_DIR, "sample_sort.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")
display_results(df, String(GPU_TAG), "Sample Sort (general-comparator) Benchmark Results")
