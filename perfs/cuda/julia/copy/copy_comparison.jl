using Revise
using Pkg

Pkg.activate("$(@__DIR__())/../../")

using Luma
using KernelAbstractions, CUDA, BenchmarkTools
using AcceleratedKernels
using Quaternions


n = 1000000
T = Float32
src_cpu = [1 for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])
#%% LUMA


Nitem = 4 #must be set equal 1 for good performance for large n
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    Luma.copy!(dst, src, Nitem=Nitem)
end

prof = CUDA.@profile Luma.copy!(dst, src; Nitem=Nitem)

#%% CUDA
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.copy!(dst, src)
end


CUDA.@profile CUDA.copy!(dst, src)


#%% KA, default implementation in GPUArrayCore
@kernel inbounds = true unsafe_indices = true function copy_kernel!(dst, src)
    i = @index(Global)
    dst[i] = src[i]
end
function copy_ka!(dst, src)
    backend = get_backend(dst)
    copy_kernel!(backend)(dst, src; ndrange=length(dst))
    KernelAbstractions.synchronize(backend)
    return dst
end
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    copy_ka!(dst, src)
end


CUDA.@profile copy_ka!(dst, src)



