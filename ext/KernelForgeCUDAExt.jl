module KernelForgeCUDAExt

using KernelForge
import KernelAbstractions as KA
using CUDA

import KernelForge: get_default_config


function get_default_config(obj::KA.Kernel{CUDABackend}, args...)
    backend = KA.backend(obj)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, 1, nothing)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    # If the kernel is statically sized we can tell the compiler about that
    if KA.workgroupsize(obj) <: KA.StaticSize
        maxthreads = prod(KA.get(KA.workgroupsize(obj)))
    else
        maxthreads = nothing
    end

    kernel = @cuda launch = false always_inline = backend.always_inline maxthreads = maxthreads obj.f(ctx, args...)
    config = CUDA.launch_configuration(kernel.fun)

    return (workgroup=config.threads, blocks=config.blocks)

end



end #end module