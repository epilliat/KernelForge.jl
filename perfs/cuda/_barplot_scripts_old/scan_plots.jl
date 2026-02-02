using Revise
using Pkg

Pkg.activate("$(@__DIR__)/../")

luma_path = abspath("$(@__DIR__)/../../../")
#Pkg.develop(path=luma_path)

ma_path = abspath("/home/emmanuel/Packages/KernelIntrinsics.jl/")
#Pkg.develop(path=ma_path)
#Pkg.instantiate()

using Luma
using Plots
using KernelAbstractions, Test, CUDA, BenchmarkTools, DataFrames
import AcceleratedKernels as AK
using Quaternions

include("helpers/extract_infos.jl")
include("helpers/illustration_tools.jl")

tmax_timed = 1
names = ["Luma Def", "Luma Opt", "CUDA.jl", "AK"]
algos = ["Scan"]
algo = "Scan"
bench = DataFrame()

#%%

#====================== Accumulate ===================#
N = Int(1e6)
for T in [Float32, Float64]
    op = +
    src = CUDA.ones(T, N)
    dst = CUDA.zeros(T, N)

    name = "Luma Def"

    start_time = time()
    while time() - start_time < 0.500
        CUDA.@sync Luma.scan!(op, dst, src)
    end
    prof = [CUDA.@profile Luma.scan!(op, dst, src) for _ in 1:100]
    dt = 0.0
    dts = Float64[]
    while dt <= 2 * tmax_timed
        timed = CUDA.@timed Luma.scan!(op, dst, src)
        u = timed.time
        dt += u
        push!(dts, u)
    end
    timed = CUDA.@timed Luma.scan!(op, dst, src)
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%%

algo = "Scan"

for T in [Float32, Float64]
    op = +
    FlagType = UInt64
    src = CUDA.ones(T, N)
    dst = CUDA.zeros(T, N)

    name = "Luma Opt"

    tmp = Luma.get_allocation(Luma.scan!, dst, src; FlagType=UInt64)
    start_time = time()
    while time() - start_time < 0.500
        CUDA.@sync Luma.scan!(op, dst, src; tmp=tmp, FlagType=UInt64)
    end

    prof = [CUDA.@profile Luma.scan!(op, dst, src; tmp=tmp, FlagType=UInt64) for _ in 1:100]
    dt = 0.0
    dts = Float64[]
    while dt <= 2 * tmax_timed
        timed = CUDA.@timed Luma.scan!(op, dst, src; tmp=tmp, FlagType=UInt64)
        u = timed.time
        dt += u
        push!(dts, u)
    end
    timed = CUDA.@timed Luma.scan!(op, dst, src; tmp=tmp, FlagType=UInt64)
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%% CUDA

algo = "Scan"

for T in [Float32, Float64]
    op = +
    src = CUDA.ones(T, N)
    dst = CUDA.zeros(T, N)

    name = "CUDA"

    start_time = time()
    while time() - start_time < 0.500
        CUDA.@sync CUDA.accumulate!(op, dst, src)
    end

    prof = [CUDA.@profile CUDA.accumulate!(op, dst, src) for _ in 1:100]
    dt = 0.0
    dts = Float64[]
    while dt <= 2 * tmax_timed
        timed = CUDA.@timed CUDA.accumulate!(op, dst, src)
        u = timed.time
        dt += u
        push!(dts, u)
    end
    timed = CUDA.@timed CUDA.accumulate!(op, dst, src)
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%% Accelerated Kernels - DecoupledLookback

algo = "Scan"

for T in [Float32, Float64]
    op = +
    src = CUDA.ones(T, N)
    dst = CUDA.zeros(T, N)

    name = "AK DL"

    try
        start_time = time()
        while time() - start_time < 0.500
            CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T), alg=AK.DecoupledLookback())
        end

        prof = [CUDA.@profile AK.accumulate!(op, dst, src; init=zero(T), alg=AK.DecoupledLookback()) for _ in 1:100]
        dt = 0.0
        dts = Float64[]
        while dt <= 2 * tmax_timed
            timed = CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T), alg=AK.DecoupledLookback())
            u = timed.time
            dt += u
            push!(dts, u)
        end
        timed = CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T), alg=AK.DecoupledLookback())
        benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
    catch e
        @warn "AK DecoupledLookback failed for $T" exception = e
    end
end

#%% Accelerated Kernels - Default

for T in [Float32, Float64]
    op = +
    src = CUDA.ones(T, N)
    dst = CUDA.zeros(T, N)

    name = "AK Def"

    start_time = time()
    while time() - start_time < 0.500
        CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T))
    end

    prof = [CUDA.@profile AK.accumulate!(op, dst, src; init=zero(T)) for _ in 1:100]
    dt = 0.0
    dts = Float64[]
    while dt <= 2 * tmax_timed
        timed = CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T))
        u = timed.time
        dt += u
        push!(dts, u)
    end
    timed = CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%%

time_cub_f32_1e8 = 4.7708 * 1000 # tmp: 407.4990 KB
time_cub_f32_1e6 = 0.0168 * 1000 # tmp: 4.74902 KB
time_cub_f64_1e8 = 9.5237 * 1000 # tmp: 1744.7490 KB
time_cub_f64_1e6 = 0.0556 * 1000 # tmp: 18.2490 KB

cub_f32 = DataFrame(
    name="Cub",
    datatype=Float32,
    algo="Scan",
    datalength=bench.datalength[1],
    kernel1=0.0,
    kernel2=0.0,
    kernel3=0.0,
    kernel4=time_cub_f32_1e6,
    kernel1_acc=0.0,
    kernel2_acc=0.0,
    kernel3_acc=0.0,
    kernel4_acc=time_cub_f32_1e6,
    kernel4_name="NVCC Benchmark",
    mean_duration_gpu=time_cub_f32_1e6,
    median_duration_pipeline=time_cub_f32_1e6,
)

cub_f64 = DataFrame(
    name="Cub",
    datatype=Float64,
    algo="Scan",
    datalength=bench.datalength[1],
    kernel1=0.0,
    kernel2=0.0,
    kernel3=0.0,
    kernel4=time_cub_f64_1e6,
    kernel1_acc=0.0,
    kernel2_acc=0.0,
    kernel3_acc=0.0,
    kernel4_acc=time_cub_f64_1e6,
    kernel4_name="NVCC Benchmark",
    mean_duration_gpu=time_cub_f64_1e6,
    median_duration_pipeline=time_cub_f64_1e6,
)
bench.kernel1_name .= missing
bench.kernel2_name .= missing

bench_with_cub = vcat(bench, cub_f32, cub_f64, cols=:union)

p1 = create_kernel_stacked_barplot(bench_with_cub,
    algo="Scan",
    kernel_colors=[:blue, :red, :green, :orange], overhead_alpha=0.7,
    names=["CUDA", "AK Def", "Luma Def", "Luma Opt", "Cub"],
    size_anotation=9)
savefig(p1, "$(@__DIR__)/../figures/scan_comparison_1e6.png")

#%%============== N = 1e8 =======================
algo = "Scan"
bench = DataFrame()
N = Int(1e8)

for T in [Float32, Float64]
    op = +
    src = CUDA.ones(T, N)
    dst = CUDA.zeros(T, N)

    name = "Luma Def"

    start_time = time()
    while time() - start_time < 0.800
        CUDA.@sync Luma.scan!(op, dst, src)
    end
    prof = [CUDA.@profile Luma.scan!(op, dst, src) for _ in 1:100]
    dt = 0.0
    dts = Float64[]
    while dt <= 2 * tmax_timed
        timed = CUDA.@timed Luma.scan!(op, dst, src)
        u = timed.time
        dt += u
        push!(dts, u)
    end
    timed = CUDA.@timed Luma.scan!(op, dst, src)
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%%

algo = "Scan"

for T in [Float32, Float64]
    op = +
    src = CUDA.ones(T, N)
    dst = CUDA.zeros(T, N)

    name = "Luma Opt"

    tmp = Luma.get_allocation(Luma.scan!, dst, src)
    start_time = time()
    while time() - start_time < 0.500
        CUDA.@sync Luma.scan!(op, dst, src; tmp=tmp)
    end

    prof = [CUDA.@profile Luma.scan!(op, dst, src; tmp=tmp) for _ in 1:100]
    dt = 0.0
    dts = Float64[]
    while dt <= 2 * tmax_timed
        timed = CUDA.@timed Luma.scan!(op, dst, src; tmp=tmp)
        u = timed.time
        dt += u
        push!(dts, u)
    end
    timed = CUDA.@timed Luma.scan!(op, dst, src; tmp=tmp)
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%% CUDA

algo = "Scan"

for T in [Float32, Float64]
    op = +
    src = CUDA.ones(T, N)
    dst = CUDA.zeros(T, N)

    name = "CUDA"

    start_time = time()
    while time() - start_time < 0.500
        CUDA.@sync CUDA.accumulate!(op, dst, src)
    end

    prof = [CUDA.@profile CUDA.accumulate!(op, dst, src) for _ in 1:100]
    dt = 0.0
    dts = Float64[]
    while dt <= 2 * tmax_timed
        timed = CUDA.@timed CUDA.accumulate!(op, dst, src)
        u = timed.time
        dt += u
        push!(dts, u)
    end
    timed = CUDA.@timed CUDA.accumulate!(op, dst, src)
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%% Accelerated Kernels - DecoupledLookback

algo = "Scan"

for T in [Float32, Float64]
    op = +
    src = CUDA.ones(T, N)
    dst = CUDA.zeros(T, N)

    name = "AK DL"

    try
        start_time = time()
        while time() - start_time < 0.500
            CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T), alg=AK.DecoupledLookback())
        end

        prof = [CUDA.@profile AK.accumulate!(op, dst, src; init=zero(T), alg=AK.DecoupledLookback()) for _ in 1:100]
        dt = 0.0
        dts = Float64[]
        while dt <= 2 * tmax_timed
            timed = CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T), alg=AK.DecoupledLookback())
            u = timed.time
            dt += u
            push!(dts, u)
        end
        timed = CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T), alg=AK.DecoupledLookback())
        benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
    catch e
        @warn "AK DecoupledLookback failed for $T" exception = e
    end
end

#%% Accelerated Kernels - Default

for T in [Float32, Float64]
    op = +
    src = CUDA.ones(T, N)
    dst = CUDA.zeros(T, N)

    name = "AK Def"

    start_time = time()
    while time() - start_time < 0.500
        CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T))
    end

    prof = [CUDA.@profile AK.accumulate!(op, dst, src; init=zero(T)) for _ in 1:100]
    dt = 0.0
    dts = Float64[]
    while dt <= 2 * tmax_timed
        timed = CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T))
        u = timed.time
        dt += u
        push!(dts, u)
    end
    timed = CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=zero(T))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%%

cub_f32 = DataFrame(
    name="Cub",
    datatype=Float32,
    algo="Scan",
    datalength=bench.datalength[1],
    kernel1=0.0,
    kernel2=0.0,
    kernel3=0.0,
    kernel4=time_cub_f32_1e8,
    kernel1_acc=0.0,
    kernel2_acc=0.0,
    kernel3_acc=0.0,
    kernel4_acc=time_cub_f32_1e8,
    kernel4_name="NVCC Benchmark",
    mean_duration_gpu=time_cub_f32_1e8,
    median_duration_pipeline=time_cub_f32_1e8,
)

cub_f64 = DataFrame(
    name="Cub",
    datatype=Float64,
    algo="Scan",
    datalength=bench.datalength[1],
    kernel1=0.0,
    kernel2=0.0,
    kernel3=0.0,
    kernel4=time_cub_f64_1e8,
    kernel1_acc=0.0,
    kernel2_acc=0.0,
    kernel3_acc=0.0,
    kernel4_acc=time_cub_f64_1e8,
    kernel4_name="NVCC Benchmark",
    mean_duration_gpu=time_cub_f64_1e8,
    median_duration_pipeline=time_cub_f64_1e8,
)
bench.kernel1_name .= missing
bench.kernel2_name .= missing

bench_with_cub = vcat(bench, cub_f32, cub_f64, cols=:union)

p2 = create_kernel_stacked_barplot(bench_with_cub,
    algo="Scan",
    kernel_colors=[:blue, :red, :green, :orange], overhead_alpha=0.7,
    names=["CUDA", "AK Def", "Luma Def", "Cub"],
    size_anotation=9,
    time_unit=:ms)
savefig(p2, "$(@__DIR__)/../figures/scan_comparison_1e8.png")

#%%
combined_plot = Plots.plot(p1, p2, layout=(1, 2), size=(1000, 500),
    margin=3Plots.mm, bottom_margin=10Plots.mm)
savefig(combined_plot, "$(@__DIR__)/../figures/combined_plot_scan.png")