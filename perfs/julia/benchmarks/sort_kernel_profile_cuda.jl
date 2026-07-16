# KF per-kernel profiling on NVIDIA via CUDA.jl's IN-PROCESS profiler. External nsys
# does NOT capture CUDA.jl kernels ("sqlite does not contain CUDA kernel data"), but
# CUDA.@profile (CUPTI in-process) does. Emits a stats CSV in the rocprofv3 kernel_stats
# layout ("Name",Calls,TotalDurationNs,AverageNs,Percentage,MinNs,MaxNs,StdDev) so
# parse_sort_kernels.jl's auto-detect reads it uniformly alongside the nsys CUB files.
#
# ENV: KF_N, KF_T (uint32|uint64|float|double), KF_OUT_STATS (output stats CSV path).
using KernelForge, KernelAbstractions, CUDA, Printf
const KF = KernelForge

const _T = Dict("uint32"=>UInt32, "uint64"=>UInt64, "float"=>Float32, "double"=>Float64)
N = parse(Int, get(ENV, "KF_N", "100000000"))
T = _T[get(ENV, "KF_T", "uint32")]
out = ENV["KF_OUT_STATS"]

h = T <: AbstractFloat ? randn(T, N) : rand(T, N)
s = CuArray(h)
tmp = KF.@allocate sort!(s)
for _ in 1:6; copyto!(s, h); KF.sort!(s; tmp = tmp); end
CUDA.synchronize()

const NPROF = 20
r = CUDA.@profile trace=true begin
    for _ in 1:NPROF; copyto!(s, h); KF.sort!(s; tmp = tmp); end
    CUDA.synchronize()
end

# Aggregate device activities by name (start/stop are in seconds).
d = r.device
agg = Dict{String,Vector{Float64}}()   # name => [calls, total_s, min_s]
for i in eachindex(d.name)
    nm = d.name[i]
    dur = d.stop[i] - d.start[i]
    a = get!(agg, nm, [0.0, 0.0, Inf])
    a[1] += 1; a[2] += dur; a[3] = min(a[3], dur)
end
tot = sum(a[2] for a in values(agg); init = 0.0)
open(out, "w") do io
    println(io, "\"Name\",\"Calls\",\"TotalDurationNs\",\"AverageNs\",\"Percentage\",\"MinNs\",\"MaxNs\",\"StdDev\"")
    for (nm, a) in agg
        totns = a[2] * 1e9
        @printf(io, "\"%s\",%d,%.1f,%.1f,%.4f,%.1f,0.0,0.0\n",
                nm, Int(a[1]), totns, totns / a[1], 100 * a[2] / tot, a[3] * 1e9)
    end
end
println(">>> wrote $out  ($(length(agg)) kernels, $(NPROF) sorts)")
