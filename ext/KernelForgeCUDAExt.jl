module KernelForgeCUDAExt

using KernelForge
using KernelAbstractions
import KernelIntrinsics as KI
using CUDA


import KernelForge: detect_arch
import KernelIntrinsics: devices

function __init__()
    if CUDA.functional()
        for dev in devices(CUDABackend())
            detect_arch(CUDABackend(), KI.deviceid(dev))
        end
    end
end

_unsafe_free!(arr::CuArray) = CUDA.unsafe_free!(arr)

end # module KernelForgeCUDAExt