include("../../meta_helper.jl")
using Pkg
Pkg.activate("perfs/envs/benchenv/$backend_str")
Pkg.instantiate()
using Revise
include("../../architecture.jl")
include("../../bench_utils.jl")
using DataFrames
using CSV

function quick_profile_simple(total_size::Int, ::Type{Op}; arch=KF.RTX1000(), T=Float32, AT=CuArray, def=false, tuned=true, ms=500) where Op
    tuning = Op == KF.MatVec ? matvec_params[T][total_size] : vecmat_params[T][total_size]
    results = []

    function bench(backend, f!; ms=500)
        # Warmup phase
        t0 = time_ns()
        while (time_ns() - t0) < ms * 1e6
            f!()
            KernelAbstractions.synchronize(backend)
        end
        # Benchmark phase: collect individual timings
        times = Float64[]
        t0 = time_ns()
        while (time_ns() - t0) < ms * 1e6
            KernelAbstractions.synchronize(backend)
            t_start = time_ns()
            f!()
            KernelAbstractions.synchronize(backend)
            t_end = time_ns()
            push!(times, (t_end - t_start) * 1e-6)  # ms
        end
        return times
    end

    fmt_exp(x) = "10^$(round(log10(x), digits=2))"

    for ((n, p), params) in sort(collect(tuning), by=x -> x[1][1])
        src = fill!(AT{T}(undef, n, p), one(T))
        backend = get_backend(src)

        if Op == KF.MatVec
            x = fill!(AT{T}(undef, p), one(T))
            base_f! = () -> src * x
            kf_f! = () -> KernelForge.matvec(*, +, src, x; params...)
            kf_def! = () -> KernelForge.matvec(*, +, src, x)
        else
            x = fill!(AT{T}(undef, n), one(T))
            base_f! = () -> x' * src
            kf_f! = () -> KernelForge.vecmat(*, +, x, src; params...)
            kf_def! = () -> KernelForge.vecmat(*, +, x, src)
        end

        t_base_times = bench(backend, base_f!; ms)
        t_base = minimum(t_base_times)

        if tuned
            t_kf_times = bench(backend, kf_f!; ms)
            t_kf = minimum(t_kf_times)
        else
            t_kf = nothing
        end

        default_params = KF.resolve_parameters(arch, Op, src)

        if def
            t_kf_def_times = bench(backend, kf_def!; ms)
            t_kf_def = minimum(t_kf_def_times)
        else
            t_kf_def = nothing
        end

        push!(results, (n=n, p=p, params=params, default_params=default_params, t_kf=t_kf, t_base=t_base, t_kf_def=t_kf_def))
    end

    param_header = Op == KF.VecMat ? ["Ni", "Nth", "wg", "bl"] : ["Ni", "csz", "Nb", "wg"]

    time_header = String[]
    tuned && push!(time_header, "KF(ms)")
    def && push!(time_header, "Def(ms)")
    push!(time_header, "Base(ms)")
    tuned && push!(time_header, "Spdup")
    def && push!(time_header, "DefSpdup")

    header = ["n", "p", param_header..., time_header...]

    rows = map(results) do r
        if Op == KF.VecMat
            param_cols = (tuned || def) ? [
                (r.params.Nitem, r.default_params.Nitem),
                (r.params.Nthreads, r.default_params.Nthreads),
                (r.params.workgroup, r.default_params.workgroup),
                (r.params.blocks, r.default_params.blocks),
            ] : [r.params.Nitem, r.params.Nthreads, r.params.workgroup, r.params.blocks]
        else
            param_cols = (tuned || def) ? [
                (r.params.Nitem, r.default_params.Nitem),
                (r.params.chunksz, r.default_params.chunksz),
                (r.params.Nblocks, r.default_params.Nblocks),
                (r.params.workgroup, r.default_params.workgroup),
            ] : [r.params.Nitem, r.params.chunksz, r.params.Nblocks, r.params.workgroup]
        end

        time_cols = Any[]
        tuned && push!(time_cols, round(r.t_kf, digits=3))
        def && push!(time_cols, round(r.t_kf_def, digits=3))
        push!(time_cols, round(r.t_base, digits=3))
        tuned && push!(time_cols, round(r.t_base / r.t_kf, digits=2))
        def && push!(time_cols, round(r.t_base / r.t_kf_def, digits=2))

        [fmt_exp(r.n), fmt_exp(r.p), param_cols..., time_cols...]
    end

    ncols = length(header)
    spdup_col = tuned ? (ncols - (def ? 1 : 0)) : nothing
    def_spdup_col = def ? ncols : nothing

    highlighters = TextHighlighter[]

    if !isnothing(spdup_col)
        push!(highlighters,
            TextHighlighter((data, i, j) -> j == spdup_col && data[i, j] isa Number && data[i, j] > 1.0, crayon"green bold"),
            TextHighlighter((data, i, j) -> j == spdup_col && data[i, j] isa Number && data[i, j] < 1.0, crayon"red"),
        )
    end
    if !isnothing(def_spdup_col)
        push!(highlighters,
            TextHighlighter((data, i, j) -> j == def_spdup_col && data[i, j] isa Number && data[i, j] > 1.0, crayon"green"),
            TextHighlighter((data, i, j) -> j == def_spdup_col && data[i, j] isa Number && data[i, j] < 1.0, crayon"red"),
        )
    end
    if tuned || def
        push!(highlighters,
            TextHighlighter((data, i, j) -> j in 3:6 && data[i, j] isa Tuple && data[i, j][1] != data[i, j][2], crayon"yellow bold"),
        )
    end

    mat = permutedims(reduce(hcat, rows))
    pretty_table(mat;
        column_labels=header,
        highlighters=highlighters,
        style=TextTableStyle(first_line_column_label=crayon"bold cyan"),
    )
end