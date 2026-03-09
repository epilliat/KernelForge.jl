module KernelForgeAMDGPUExt
using KernelForge
using KernelAbstractions
using AMDGPU
import KernelForge: detect_arch
import KernelIntrinsics: devices
import KernelIntrinsics as KI

function __init__()
    if AMDGPU.functional()
        for dev in devices(ROCBackend())
            detect_arch(ROCBackend(), KI.deviceid(dev))
        end
    end
end

end # module KernelForgeAMDGPUExt