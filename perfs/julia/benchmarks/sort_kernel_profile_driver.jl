# Parametric single-config KF sort driver for PER-KERNEL profiling under an
# external profiler (rocprofv3 --kernel-trace on AMD, nsys/CUPTI on NVIDIA).
# The profiler wraps this whole process; we warm up (compile all pass kernels)
# then run a fixed number of steady-state sorts so the profiler's per-kernel
# stats have a clean MinNs (immune to the first-call compile outlier).
#
# ENV: KF_N (default 1e8), KF_T in {uint32,uint64,float,double} (default uint32).
# Backend auto-detected (ROC or CUDA) so the same driver serves both clusters.
using KernelForge, KernelAbstractions
const KF = KernelForge; const KA = KernelAbstractions

const _HAS_ROC = try @eval import AMDGPU; true catch; false end
const _HAS_CU  = _HAS_ROC ? false : (try @eval import CUDA; true catch; false end)
const backend  = _HAS_ROC ? (@eval AMDGPU.ROCBackend()) : (@eval CUDA.CUDABackend())
AT(x) = _HAS_ROC ? (@eval AMDGPU.ROCArray($x)) : (@eval CUDA.CuArray($x))

const _T = Dict("uint32"=>UInt32, "uint64"=>UInt64, "float"=>Float32, "double"=>Float64)
N = parse(Int, get(ENV, "KF_N", "100000000"))
T = _T[get(ENV, "KF_T", "uint32")]

h = T <: AbstractFloat ? randn(T, N) : rand(T, N)
src = AT(h)
tmp = KF.@allocate sort!(src)

# Warmup (compile all byte-pass kernels + histogram + scan). Multiple sorts so the
# clock is at steady state before the profiled body.
for _ in 1:6
    copyto!(src, h); KF.sort!(src; tmp = tmp)
end
KA.synchronize(backend)

println(">>> PROFILED BEGIN  T=$(T)  N=$(N)")
for _ in 1:10
    copyto!(src, h); KF.sort!(src; tmp = tmp)
end
KA.synchronize(backend)
println(">>> PROFILED END")
