module KernelForgeAMDGPUExt
using KernelForge
using KernelAbstractions
const KA = KernelAbstractions
using AMDGPU
import KernelForge: detect_arch, _unsafe_free!

import KernelIntrinsics: devices
import KernelIntrinsics as KI

function __init__()
    if AMDGPU.functional()
        for dev in devices(ROCBackend())
            arch = detect_arch(ROCBackend(), KI.deviceid(dev))
            # Eagerly load this device's tuning data (data/tuning/<Arch>.jl)
            # if a file exists; no-op on subsequent devices of the same arch.
            KernelForge.load_tunings_for!(nameof(typeof(arch)))
        end
    end
end

_unsafe_free!(arr::ROCArray) = AMDGPU.unsafe_free!(arr)

# Compute-Unit / Multiprocessor count for the active AMD device.
#  - `KernelForge.num_sms` (public): used by mapreduce1d/scan autotune to
#    size hardware-relative `blocks` grids.
#  - `KernelForge.Random._n_sms` (internal): sizes the persistent thread
#    pool; delegates to `num_sms`.
KernelForge.num_sms(::ROCBackend) =
    Int(AMDGPU.HIP.properties(AMDGPU.device()).multiProcessorCount)
KernelForge.Random._n_sms(be::ROCBackend) = KernelForge.num_sms(be)

# Compile-only path for the autotune script. Mirrors the prelude of KA's
# ROC functor (AMDGPU.jl `src/ROCKernels.jl:99-102`) but stops at the
# `@roc launch=false` step. AMDGPU's @roc has no `maxthreads` kwarg, so
# unlike the CUDA path we don't pass one.
#
# TODO: this branch is mirrored from the CUDA path by code review; not
# bench-tested in the session that introduced it (no AMD hardware).
function KernelForge.compile_kernel_only(
    kobj::KA.Kernel{ROCBackend}, args...; ndrange,
)
    ndrange, _, iterspace, _ = KA.launch_config(kobj, ndrange, nothing)
    ctx = KA.mkcontext(kobj, ndrange, iterspace)
    AMDGPU.@roc(launch=false, kobj.f(ctx, args...))
    return nothing
end

end # module KernelForgeAMDGPUExt