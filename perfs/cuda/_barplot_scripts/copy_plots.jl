using Revise
using Pkg

Pkg.activate("$(@__DIR__())/../")
include("helpers/extract_infos.jl")
include("helpers/illustration_tools.jl")

luma_path = abspath("$(@__DIR__())/../../../")
#Pkg.develop(path=luma_path)

ma_path = abspath("/home/emmanuel/Packages/MemoryAccess.jl/")
#Pkg.develop(path=ma_path)
#Pkg.instantiate()

using Luma
using KernelAbstractions, Test, CUDA, BenchmarkTools, DataFrames
import AcceleratedKernels as AK
using Quaternions

tmax_timed = 1
names = ["Cublas", "Luma", "CUDA.jl", "AK"]
algos = ["Copy"]
bench = DataFrame()

#%%

#====================== COPY ===================#

algo = "Copy"

for Nitem in [1, 4]
    for T in [Float32, UInt8]
        N = Int(100000000)
        src = CuArray{T}([1 for _ in (1:N)])
        dst = CuArray{T}([0 for _ in (1:N)])

        name = "Luma v$Nitem"

        start_time = time()
        while time() - start_time < 0.500  # 500ms warm-up
            Luma.vcopy!(dst, src, Nitem=Nitem)
        end
        prof = [CUDA.@profile Luma.vcopy!(dst, src, Nitem=Nitem) for _ in (1:100)]
        dt = 0
        dts = []
        while dt <= 2 * tmax_timed
            timed = (CUDA.@timed Luma.vcopy!(dst, src, Nitem=Nitem))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
            u = timed[:time]
            dt += u
            push!(dts, u)
        end
        timed = (CUDA.@timed Luma.vcopy!(dst, src, Nitem=Nitem))
        benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
    end
end

#%%
for Nitem in [1, 4]
    for T in [Float32, UInt8]
        N = Int(100000000)
        src = CuArray{T}([1 for _ in (1:N)])
        dst = CuArray{T}([0 for _ in (1:N)])
        name = "KA"
        @kernel inbounds = true unsafe_indices = true function copy_kernel!(dst, src)
            i = @index(Global)
            dst[i] = src[i]
        end

        function copy_ka!(dst, src)
            backend = get_backend(dst)
            copy_kernel!(backend)(dst, src; ndrange=length(dst))
            KernelAbstractions.synchronize(backend)
            return dst
        end
        start_time = time()
        while time() - start_time < 0.500  # 500ms warm-up
            CUDA.@sync copy_ka!(dst, src)
        end
        prof = [CUDA.@profile Luma.vcopy!(dst, src, Nitem=Nitem) for _ in (1:100)]
        dt = 0
        dts = []
        while dt <= 2 * tmax_timed
            timed = (CUDA.@timed copy_ka!(dst, src))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
            u = timed[:time]
            dt += u
            push!(dts, u)
        end
        timed = (CUDA.@timed copy_ka!(dst, src))
        benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
    end
end
bench[end-1:end, :]
#%%

time_cub_f32_1e8 = 4.8357 * 1000
time_cub_f32_1e6 = 0.0128 * 1000
time_cub_u8_1e8 = 1.2173 * 1000
time_cub_u8_1e6 = 0.0060 * 1000

cub_f32 = DataFrame(
    name="Cub",
    datatype=Float32,
    algo="Copy",
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
    datatype=UInt8,
    algo="Copy",
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
bench.kernel1_name .= missing
bench.kernel2_name .= missing

bench_with_cub = vcat(bench, cub_f32, cub_u8, cols=:union)

# Plot - it will automatically use "NVCC Benchmark" from kernel4_name
p1 = create_kernel_stacked_barplot(bench_with_cub,
    algo="Copy",
    kernel_colors=[:blue, :red, :green, :orange],
    overhead_alpha=0.7, names=["KA", "Luma v1", "Luma v4", "Cub"], time_unit=:ms)
plot!(ylim=(0, 7))
#savefig(plot, "$(@__DIR__())/../figures/vcopy_comparison.png")
#%%
#====================== COPY ===================#
Ns = [Int(i * 1e5) for i in (1:500)]
durations = Dict("v1" => [], "v4" => [], "Ns" => Ns)
for N in Ns
    println("computing durations for N=$N")
    for Nitem in [1, 4]
        for T in [Float32]
            src = CuArray{T}([1 for _ in (1:N)])
            dst = CuArray{T}([0 for _ in (1:N)])

            name = "Luma v$Nitem"
            #%%
            start_time = time()
            while time() - start_time < 0.500  # 500ms warm-up
                Luma.vcopy!(dst, src, Nitem=Nitem)
            end
            dt = 0
            dts = []
            while dt <= 2 * tmax_timed
                timed = (CUDA.@timed Luma.vcopy!(dst, src, Nitem=Nitem))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
                u = timed[:time]
                dt += u
                push!(dts, u)
            end
            timed = (CUDA.@timed Luma.vcopy!(dst, src, Nitem=Nitem))
            push!(durations["v$Nitem"], mean(dts))
        end
    end
end

#%%
df = DataFrame(durations)
CSV.write("$(@__DIR__())/data/memcopy_func_n.csv", df)


#%%
# Convert back to Dict if needed
df = CSV.read("$(@__DIR__())/data/memcopy_func_n.csv", DataFrame)
using Plots

plt2 = Plots.plot(df.Ns, df.v1 .* 1e6, label="Copy v1", linewidth=2,
    xscale=:log10, yscale=:log10,
    xlabel="Size (Float32)", ylabel="Duration (μs)")
Plots.plot!(df.Ns, df.v4 .* 1e6, label="Copy v4", linewidth=2, title="Memory Copy Duration vs Size (Float32)", linestyle=:dash)

vline!([12 * 1024 * 1024 / 4], label="L2 limit (read + write)", linestyle=:dashdot, color=:black) # L2 cache size = 24MB
xlabel!("Length")
ylabel!("Duration (μs)")

combined_plot = Plots.plot(p1, plt2, layout=(1, 2), size=(1000, 500),
    margin=5Plots.mm)
savefig(combined_plot, "$(@__DIR__())/../figures/combined_plot_copy.png")