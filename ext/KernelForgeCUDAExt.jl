module KernelForgeCUDAExt

using KernelForge
using KernelAbstractions
const KA = KernelAbstractions
import KernelIntrinsics as KI
using CUDA


import KernelForge: detect_arch, _unsafe_free!
import KernelIntrinsics: devices

function __init__()
    if CUDA.functional()
        for dev in devices(CUDABackend())
            arch = detect_arch(CUDABackend(), KI.deviceid(dev))
            # Eagerly load this device's tuning data (data/tuning/<Arch>.jl)
            # if a file exists; no-op on subsequent devices of the same arch.
            KernelForge.load_tunings_for!(nameof(typeof(arch)))
        end
    end
end

_unsafe_free!(arr::CuArray) = CUDA.unsafe_free!(arr)

# Multiprocessor (SM) count for the active CUDA device. Used by the
# mapreduce1d/scan autotune to size hardware-relative `blocks` grids.
KernelForge.num_sms(::CUDABackend) =
    Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))

# Compile-only path for the autotune script. Mirrors the prelude of KA's
# CUDA functor (CUDA.jl `src/CUDAKernels.jl:115-129`) but stops at the
# `@cuda launch=false` step — populates the in-process GPUCompiler cache
# for this (kobj, arg-types, ndrange) without launching the kernel. Same
# cache key as the eventual real launch ⇒ bench loop hits the cache.
function KernelForge.compile_kernel_only(
    kobj::KA.Kernel{CUDABackend}, args...; ndrange,
)
    backend = KA.backend(kobj)
    ndrange, _, iterspace, _ = KA.launch_config(kobj, ndrange, nothing)
    ctx = KA.mkcontext(kobj, ndrange, iterspace)
    maxthreads = KA.workgroupsize(kobj) <: KA.StaticSize ?
        prod(KA.get(KA.workgroupsize(kobj))) : nothing
    CUDA.@cuda(launch=false,
               always_inline=backend.always_inline,
               maxthreads=maxthreads,
               kobj.f(ctx, args...))
    return nothing
end

end # module KernelForgeCUDAExt