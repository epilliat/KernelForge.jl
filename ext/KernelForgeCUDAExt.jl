module KernelForgeCUDAExt

using KernelForge
import KernelAbstractions as KA
using CUDA

import KernelForge: asyncfill!


function asyncfill!(arr::CuArray{UInt8}, val::UInt8)
    CUDA.memset(pointer(arr), val, length(arr))
end







end #end module