using Revise
using JSON3
using KernelForge
using KernelForge: UnitFloat8
import KernelForge as KF
import KernelIntrinsics as KI
using KernelAbstractions
import KernelAbstractions as KA

has_roc() && using AMDGPU
has_cuda() && using CUDA
has_metal() && using Metal

backend = has_roc() ? ROCBackend() :
          has_cuda() ? CUDABackend() :
          has_metal() ? MetalBackend() :
          error("No supported GPU backend found")

dev = KI.device(backend)
GPU_TAG = match(r"\.(\w+)\(\)", "$(KF.detect_arch(backend, Val(KI.deviceid(dev))))")[1]
RESULT_DIR = joinpath(@__DIR__, "results", GPU_TAG)
mkpath(RESULT_DIR)
println("GPU         : $(KI.name(dev))  →  $GPU_TAG")
println("Results dir : $RESULT_DIR")


AT = has_roc() ? ROCArray :
     has_cuda() ? CuArray :
     has_metal() ? MtlArray :
     error("No supported GPU array type found")


# ---------------------------------------------------------------------------
# Save device info JSON
# ---------------------------------------------------------------------------

if @isdefined(CUDABackend)
    function save_device_info(::CUDABackend, path::String)
        dev = CUDA.device()
        attrs = Dict(
            string(a) => try
                CUDA.attribute(dev, a)
            catch
                nothing
            end
            for a in instances(CUDA.CUdevice_attribute)
        )
        info = Dict(
            "name" => CUDA.name(dev),
            "gpu_tag" => GPU_TAG,
            "compute_capability" => string(CUDA.capability(dev)),
            "total_memory_bytes" => CUDA.totalmem(dev),
            "cuda_driver_version" => string(CUDA.driver_version()),
            "cuda_runtime_version" => string(CUDA.runtime_version()),
            "attributes" => attrs,
        )
        open(path, "w") do io
            JSON3.write(io, info)
        end
        println("Device info  : $path")
    end
end

if @isdefined(ROCBackend)
    function save_device_info(backend::ROCBackend, path::String)
        dev = KI.device(backend)
        info = Dict(
            "name" => KI.name(dev),
            "gpu_tag" => GPU_TAG,
            "total_memory_bytes" => AMDGPU.HIP.properties(dev).totalGlobalMem,
            "rocm_version" => string(AMDGPU.HIP.runtime_version()),
            "wavefront_size" => AMDGPU.HIP.wavefrontsize(dev),
        )
        open(path, "w") do io
            JSON3.write(io, info)
        end
        println("Device info  : $path")
    end
end

if @isdefined(MetalBackend)
    function save_device_info(::MetalBackend, path::String)
        dev = Metal.device()
        info = Dict(
            "name" => Metal.name(dev),
            "gpu_tag" => GPU_TAG,
            "total_memory_bytes" => Metal.recommendedMaxWorkingSetSize(dev),
            "max_threads_per_threadgroup" => Metal.maxThreadsPerThreadgroup(dev),
            "unified_memory" => Metal.hasUnifiedMemory(dev),
            "macos_version" => string(Sys.os_version()),
        )
        open(path, "w") do io
            JSON3.write(io, info)
        end
        println("Device info  : $path")
    end
end
save_device_info(backend, joinpath(RESULT_DIR, "device_info.json"))

