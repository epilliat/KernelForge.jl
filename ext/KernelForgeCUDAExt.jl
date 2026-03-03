module KernelForgeCUDAExt

using KernelForge
using KernelAbstractions
using CUDA


import KernelForge: detect_arch
import KernelIntrinsics: list_devices

function __init__()
    if CUDA.functional()
        for dev in list_devices(CUDABackend())
            detect_arch(Val(dev))
        end
    end
end


end # module KernelForgeCUDAExt