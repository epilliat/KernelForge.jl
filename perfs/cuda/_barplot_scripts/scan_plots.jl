using Revise
using Pkg


Pkg.activate("$(@__DIR__())/../")

luma_path = abspath("$(@__DIR__())/../../../")
#Pkg.develop(path=luma_path)

ma_path = abspath("/home/emmanuel/Packages/MemoryAccess.jl/")
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
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0 for _ in (1:N)])

    name = "Luma Def"

    start_time = time()
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync Luma.scan!(op, dst, src)
    end
    prof = [CUDA.@profile Luma.scan!(op, dst, src) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed Luma.scan!(op, dst, src))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed Luma.scan!(op, dst, src))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%%

algo = "Scan"

for T in [Float32, Float64]
    op = +
    FlagType = UInt64
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0 for _ in (1:N)])

    name = "Luma Opt"

    start_time = time()
    tmp = Luma.get_allocation(scan!, identity, op, dst, src; FlagType=UInt64)
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync Luma.scan!(op, dst, src)
    end

    prof = [CUDA.@profile Luma.scan!(op, dst, src; tmp=tmp, FlagType=UInt64) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed Luma.scan!(op, dst, src; tmp=tmp, FlagType=UInt64))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed Luma.scan!(op, dst, src; tmp=tmp, FlagType=UInt64))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end



#%% CUDA

algo = "Scan"

for T in [Float32, Float64]
    op = +
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0 for _ in (1:N)])

    name = "CUDA"

    start_time = time()
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync CUDA.accumulate!(op, dst, src)
    end

    prof = [CUDA.@profile CUDA.accumulate!(op, dst, src) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed CUDA.accumulate!(op, dst, src))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed CUDA.accumulate!(op, dst, src))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%% Accelerated Kernels

algo = "Scan"

for T in [Float32, Float64]
    op = +
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0 for _ in (1:N)])

    name = "AK DL"

    start_time = time()
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync AK.accumulate!(op, dst, src; init=T(0), alg=AK.DecoupledLookback())
    end

    prof = [CUDA.@profile AK.accumulate!(op, dst, src; init=T(0), alg=AK.DecoupledLookback()) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=T(0), alg=AK.DecoupledLookback()))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=T(0), alg=AK.DecoupledLookback()))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end
for T in [Float32, Float64]
    op = +
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0 for _ in (1:N)])

    name = "AK Def"

    start_time = time()
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync AK.accumulate!(op, dst, src; init=T(0))
    end

    prof = [CUDA.@profile AK.accumulate!(op, dst, src; init=T(0))]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=T(0)))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=T(0)))
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
    # ... other columns
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
    # ... other columns
)
bench.kernel1_name .= missing
bench.kernel2_name .= missing

bench_with_cub = vcat(bench, cub_f32, cub_f64, cols=:union)

# Plot - it will automatically use "NVCC Benchmark" from kernel4_name
p1 = create_kernel_stacked_barplot(bench_with_cub,
    algo="Scan",
    kernel_colors=[:blue, :red, :green, :orange], overhead_alpha=0.7,
    names=["CUDA", "AK Def", "Luma Def", "Luma Opt", "Cub"],
    size_anotation=9)
savefig(p1, "$(@__DIR__())/../figures/scan_comparison_1e6.png")



#%%============== N = 1e8 =======================
algo = "Scan"
bench = DataFrame()
N = Int(1e8)
for T in [Float32, Float64]
    op = +
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0 for _ in (1:N)])

    name = "Luma Def"

    start_time = time()
    while time() - start_time < 0.800  # 500ms warm-up
        CUDA.@sync Luma.scan!(op, dst, src)
    end
    prof = [CUDA.@profile Luma.scan!(op, dst, src) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed Luma.scan!(op, dst, src))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed Luma.scan!(op, dst, src))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%%

algo = "Scan"

for T in [Float32, Float64]
    op = +
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0 for _ in (1:N)])

    name = "Luma Opt"

    start_time = time()
    tmp = Luma.get_allocation(scan!, identity, op, dst, src)
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync Luma.scan!(op, dst, src)
    end

    prof = [CUDA.@profile Luma.scan!(op, dst, src; tmp=tmp) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed Luma.scan!(op, dst, src; tmp=tmp))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed Luma.scan!(op, dst, src; tmp=tmp))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end



#%% CUDA

algo = "Scan"

for T in [Float32, Float64]
    op = +
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0 for _ in (1:N)])

    name = "CUDA"

    start_time = time()
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync CUDA.accumulate!(op, dst, src)
    end

    prof = [CUDA.@profile CUDA.accumulate!(op, dst, src) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed CUDA.accumulate!(op, dst, src))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed CUDA.accumulate!(op, dst, src))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%% Accelerated Kernels

algo = "Scan"

for T in [Float32, Float64]
    op = +
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0 for _ in (1:N)])

    name = "AK DL"

    start_time = time()
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync AK.accumulate!(op, dst, src; init=T(0), alg=AK.DecoupledLookback())
    end

    prof = [CUDA.@profile AK.accumulate!(op, dst, src; init=T(0), alg=AK.DecoupledLookback()) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=T(0), alg=AK.DecoupledLookback()))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=T(0), alg=AK.DecoupledLookback()))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end
for T in [Float32, Float64]
    op = +
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0 for _ in (1:N)])

    name = "AK Def"

    start_time = time()
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync AK.accumulate!(op, dst, src; init=T(0))
    end

    prof = [CUDA.@profile AK.accumulate!(op, dst, src; init=T(0))]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=T(0)))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed CUDA.@sync AK.accumulate!(op, dst, src; init=T(0)))
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
    kernel4=time_cub_f32_1e8,
    kernel1_acc=0.0,
    kernel2_acc=0.0,
    kernel3_acc=0.0,
    kernel4_acc=time_cub_f32_1e8,
    kernel4_name="NVCC Benchmark",
    mean_duration_gpu=time_cub_f32_1e8,
    median_duration_pipeline=time_cub_f32_1e8,
    # ... other columns
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
    # ... other columns
)
bench.kernel1_name .= missing
bench.kernel2_name .= missing

bench_with_cub = vcat(bench, cub_f32, cub_f64, cols=:union)

# Plot - it will automatically use "NVCC Benchmark" from kernel4_name
p2 = create_kernel_stacked_barplot(bench_with_cub,
    algo="Scan",
    kernel_colors=[:blue, :red, :green, :orange], overhead_alpha=0.7,
    names=["CUDA", "AK DL", "Luma Def", "Cub"],
    size_anotation=9,
    time_unit=:ms)
savefig(p2, "$(@__DIR__())/../figures/scan_comparison_1e8.png")


#%%
combined_plot = Plots.plot(p1, p2, layout=(1, 2), size=(1000, 500),
    margin=3Plots.mm, bottom_margin=10Plots.mm)
savefig(combined_plot, "$(@__DIR__())/../figures/combined_plot_scan.png")