module KernelForgeAMDGPUExt
using KernelForge
using KernelAbstractions
using AMDGPU
import KernelForge: detect_arch
import KernelIntrinsics: list_devices

function __init__()
    if AMDGPU.functional()
        for dev in list_devices(ROCBackend())
            detect_arch(Val(dev))
        end
    end
end

end # module KernelForgeAMDGPUExt