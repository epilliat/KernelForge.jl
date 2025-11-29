using Revise
using Pkg


Pkg.activate("$(@__DIR__())/../")

luma_path = abspath("$(@__DIR__())/../../../")
#Pkg.develop(path=luma_path)

ma_path = abspath("/home/emmanuel/Packages/MemoryAccess.jl/")
#Pkg.develop(path=ma_path)
#Pkg.instantiate()

using Luma
using Luma: UnitFloat8
using Plots
using KernelAbstractions, Test, CUDA, BenchmarkTools, DataFrames
import AcceleratedKernels as AK
using Quaternions

include("helpers/extract_infos.jl")
include("helpers/illustration_tools.jl")


tmax_timed = 1
names = ["Cublas", "Luma Def", "Luma Opt", "CUDA.jl", "AK"]
algos = ["Mapreduce"]

#%%
bench = DataFrame()

algo = "Sum"
#====================== Sum ===================#
N = Int(1e6)
prof = []
for T in [Float32, Float64, UnitFloat8]
    op = +
    f(x) = x
    if T == UnitFloat8
        f(x) = Float32(x)
    end
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0])

    name = "Luma Def"

    start_time = time()
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync Luma.mapreduce!(f, op, dst, src)
    end
    prof = [CUDA.@profile Luma.mapreduce!(f, op, dst, src) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed Luma.mapreduce!(f, op, dst, src))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed Luma.mapreduce!(f, op, dst, src))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%%

algo = "Sum"
config = nothing
for T in [Float32, Float64, UnitFloat8]
    op = +
    f(x) = x
    FlagType = UInt64
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0])

    name = "Luma Opt"

    start_time = time()
    tmp = get_allocation(Luma.mapreduce1d!, f, op, dst, (src,); FlagType=FlagType, config=config)
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync Luma.mapreduce!(f, op, dst, src)
    end

    prof = [CUDA.@profile Luma.mapreduce!(f, op, dst, src; tmp=tmp, FlagType=FlagType, config=config) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed Luma.mapreduce!(f, op, dst, src; tmp=tmp, FlagType=FlagType, config=config))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed Luma.mapreduce!(f, op, dst, src; tmp=tmp, FlagType=FlagType, config=config))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%% CUDA

algo = "Sum"

for T in [Float32, Float64, UnitFloat8]
    op = +
    f(x) = x
    if T == UnitFloat8
        f(x) = Float32(x)
    end
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0])

    name = "CUDA"

    start_time = time()
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync mapreduce(f, op, src, dims=1)
    end

    prof = [CUDA.@profile mapreduce(f, op, src, dims=1) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed mapreduce(f, op, src, dims=1))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed mapreduce(f, op, src, dims=1))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%% Accelerated Kernels

algo = "Sum"

for T in [Float32, Float64, UnitFloat8]
    op = +
    f(x) = x
    if T == UnitFloat8
        f(x) = Float32(x)
    end
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0])

    name = "AK"

    start_time = time()
    #Curiously, putting dims=1 makes AK much slower
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync AK.mapreduce(f, op, src, init=T(0))
    end

    prof = [CUDA.@profile AK.mapreduce(f, op, src, init=T(0)) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed CUDA.@sync AK.mapreduce(f, op, src, init=T(0)))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed CUDA.@sync AK.mapreduce(f, op, src, init=T(0)))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%%

time_cub_u8_1e6 = 0.0070 * 1000 # tmp: 407.4990 KB
time_cub_f32_1e6 = 0.0092 * 1000 # tmp: 4.74902 KB
time_cub_f64_1e6 = 0.0269 * 1000 # tmp: 18.2490 KB

time_cub_u8_1e8 = 0.5554 * 1000 # tmp: 407.4990 KB
time_cub_f32_1e8 = 2.1917 * 1000 # tmp: 407.4990 KB
time_cub_f64_1e8 = 4.4292 * 1000 # tmp: 1744.7490 KB

cub_f32 = DataFrame(
    name="Cub",
    datatype=Float32,
    algo="Sum",
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
    algo="Sum",
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
cub_u8 = DataFrame(
    name="Cub",
    datatype=UnitFloat8,
    algo="Sum",
    datalength=bench.datalength[1],
    kernel1=0.0,
    kernel2=0.0,
    kernel3=0.0,
    kernel4=time_cub_u8_1e6,
    kernel1_acc=0.0,
    kernel2_acc=0.0,
    kernel3_acc=0.0,
    kernel4_acc=time_cub_u8_1e6,
    kernel4_name="NVCC Benchmark",
    mean_duration_gpu=time_cub_u8_1e6,
    median_duration_pipeline=time_cub_u8_1e6,
    # ... other columns
)
bench.kernel1_name .= missing
bench.kernel2_name .= missing

bench_with_cub = vcat(bench, cub_f32, cub_f64, cub_u8, cols=:union)

# Plot - it will automatically use "NVCC Benchmark" from kernel4_name
plot1 = create_kernel_stacked_barplot(bench_with_cub,
    algo="Sum",
    kernel_colors=[:blue, :red, :green, :orange], overhead_alpha=0.7,
    names=["CUDA", "AK", "Luma Def", "Luma Opt", "Cub"],
    size_anotation=9)

savefig(plot1, "$(@__DIR__())/../figures/mapreduce_comparison_1e6.png")

bench_nof64 = filter(row -> row.datatype != Float64, bench_with_cub)
plot1_nof64 = create_kernel_stacked_barplot(bench_nof64,
    algo="Sum",
    kernel_colors=[:blue, :red, :green, :orange], overhead_alpha=0.7,
    names=["CUDA", "AK", "Luma Def", "Luma Opt", "Cub"],
    size_anotation=9)
savefig(plot1_nof64, "$(@__DIR__())/../figures/mapreduce_comparison_nof64_1e6.png")

#%%
bench = DataFrame()

algo = "Sum"
#====================== Sum large N ===================#
N = Int(1e8)

prof = []
for T in [Float32, Float64, UnitFloat8]
    op = +
    f(x) = x
    if T == UnitFloat8
        f(x) = Float32(x)
    end
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0])

    name = "Luma Def"

    start_time = time()
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync Luma.mapreduce!(f, op, dst, src)
    end
    prof = [CUDA.@profile Luma.mapreduce!(f, op, dst, src) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed Luma.mapreduce!(f, op, dst, src))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed Luma.mapreduce!(f, op, dst, src))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%%

algo = "Sum"
config = nothing
for T in [Float32, Float64, UnitFloat8]
    op = +
    f(x) = x
    FlagType = UInt64
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0])

    name = "Luma Opt"

    start_time = time()
    tmp = get_allocation(Luma.mapreduce1d!, f, op, dst, (src,); FlagType=FlagType, config=config)
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync Luma.mapreduce!(f, op, dst, src)
    end

    prof = [CUDA.@profile Luma.mapreduce!(f, op, dst, src; tmp=tmp, FlagType=FlagType, config=config) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed Luma.mapreduce!(f, op, dst, src; tmp=tmp, FlagType=FlagType, config=config))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed Luma.mapreduce!(f, op, dst, src; tmp=tmp, FlagType=FlagType, config=config))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%% CUDA

algo = "Sum"

for T in [Float32, Float64, UnitFloat8]
    op = +
    f(x) = x
    if T == UnitFloat8
        f(x) = Float32(x)
    end
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0])

    name = "CUDA"

    start_time = time()
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync mapreduce(f, op, src, dims=1)
    end

    prof = [CUDA.@profile mapreduce(f, op, src, dims=1) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed mapreduce(f, op, src, dims=1))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed mapreduce(f, op, src, dims=1))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%% Accelerated Kernels

algo = "Sum"

for T in [Float32, Float64, UnitFloat8]
    op = +
    f(x) = x
    if T == UnitFloat8
        f(x) = Float32(x)
    end
    src = CuArray{T}([1 for _ in (1:N)])
    dst = CuArray{T}([0])

    name = "AK"

    start_time = time()
    #Curiously, putting dims=1 makes AK much slower
    while time() - start_time < 0.500  # 500ms warm-up
        CUDA.@sync AK.mapreduce(f, op, src, init=T(0))
    end

    prof = [CUDA.@profile AK.mapreduce(f, op, src, init=T(0)) for _ in (1:100)]
    dt = 0
    dts = []
    while dt <= 2 * tmax_timed
        timed = (CUDA.@timed CUDA.@sync AK.mapreduce(f, op, src, init=T(0)))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
        u = timed[:time]
        dt += u
        push!(dts, u)
    end
    timed = (CUDA.@timed CUDA.@sync AK.mapreduce(f, op, src, init=T(0)))
    benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#%%

time_cub_u8_1e6 = 0.0070 * 1000 # tmp: 407.4990 KB
time_cub_f32_1e6 = 0.0092 * 1000 # tmp: 4.74902 KB
time_cub_f64_1e6 = 0.0269 * 1000 # tmp: 18.2490 KB

time_cub_u8_1e8 = 0.5554 * 1000 # tmp: 407.4990 KB
time_cub_f32_1e8 = 2.1917 * 1000 # tmp: 407.4990 KB
time_cub_f64_1e8 = 4.4292 * 1000 # tmp: 1744.7490 KB

#We put UnitFloat but Cub does not support it. We make the benchmark for UInt8 only
cub_f32 = DataFrame(
    name="Cub",
    datatype=Float32,
    algo="Sum",
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

cub_u8 = DataFrame(
    name="Cub",
    datatype=UnitFloat8,
    algo="Sum",
    datalength=bench.datalength[1],
    kernel1=0.0,
    kernel2=0.0,
    kernel3=0.0,
    kernel4=time_cub_u8_1e8,
    kernel1_acc=0.0,
    kernel2_acc=0.0,
    kernel3_acc=0.0,
    kernel4_acc=time_cub_u8_1e8,
    kernel4_name="NVCC Benchmark",
    mean_duration_gpu=time_cub_u8_1e8,
    median_duration_pipeline=time_cub_u8_1e8,
    # ... other columns
)

cub_f64 = DataFrame(
    name="Cub",
    datatype=Float64,
    algo="Sum",
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

bench_with_cub = vcat(bench, cub_f32, cub_f64, cub_u8, cols=:union)
bench_no_unitfloat8 = filter(row -> row.datatype != UnitFloat8, bench_with_cub)

# Plot - it will automatically use "NVCC Benchmark" from kernel4_name

plot2 = create_kernel_stacked_barplot(bench_with_cub,
    algo="Sum",
    kernel_colors=[:blue, :red, :green, :orange], overhead_alpha=0.7,
    names=["CUDA", "AK", "Luma Def", "Luma Opt", "Cub"],
    size_anotation=9,
    time_unit=:ms)
savefig(plot2, "$(@__DIR__())/../figures/mapreduce_comparison_1e8.png")

bench_nof64 = filter(row -> row.datatype != Float64, bench_with_cub)

plot2_nof64 = create_kernel_stacked_barplot(bench_nof64,
    algo="Sum",
    kernel_colors=[:blue, :red, :green, :orange], overhead_alpha=0.7,
    names=["CUDA", "AK", "Luma Def", "Luma Opt", "Cub"],
    size_anotation=9,
    time_unit=:ms)
savefig(plot2_nof64, "$(@__DIR__())/../figures/mapreduce_comparison_nof64_1e8.png")

combined_plot = Plots.plot(plot1, plot2,
    layout=(1, 2),
    size=(1300, 500),
    left_margin=7Plots.mm,
    bottom_margin=10Plots.mm,
    right_margin=0Plots.mm
)

savefig(combined_plot, "$(@__DIR__())/../figures/mapreduce_1e6_1e8.png")

combined_plot = Plots.plot(plot1_nof64, plot2_nof64,
    layout=(1, 2),
    size=(1000, 500),
    left_margin=7Plots.mm,
    bottom_margin=10Plots.mm,
    right_margin=0Plots.mm
)

savefig(combined_plot, "$(@__DIR__())/../figures/mapreduce_nof64_1e6_1e8.png")