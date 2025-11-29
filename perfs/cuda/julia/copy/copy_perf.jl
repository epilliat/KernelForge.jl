using Revise
using Pkg

Pkg.activate("$(@__DIR__())/../../")

using Luma
using KernelAbstractions, CUDA, BenchmarkTools
using AcceleratedKernels
using Quaternions




#%%
n = 100000000
T = Float32
src_cpu = [1.0f0 for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])
Nitem = 1

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    Luma.vcopy!(dst, src, Nitem=Nitem)
end

prof = CUDA.@profile Luma.vcopy!(dst, src; Nitem=Nitem)

#%%
n = 1000000
T = Float32
src_cpu = [1.0f0 for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])
Nitem = 4

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    Luma.vcopy!(dst, src, Nitem=Nitem)
end

prof = CUDA.@profile Luma.vcopy!(dst, src; Nitem=Nitem)


#%%


n = 1000000
T = UInt8
src_cpu = [1.0f0 for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])
Nitem = 8 #better

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    Luma.vcopy!(dst, src, Nitem=Nitem)
end

prof = CUDA.@profile Luma.vcopy!(dst, src; Nitem=Nitem)

#%%
n = 1000000
T = UInt8
src_cpu = [1.0f0 for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])
Nitem = 4

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    Luma.vcopy!(dst, src, Nitem=Nitem)
end

prof = CUDA.@profile Luma.vcopy!(dst, src; Nitem=Nitem)







#================== setvalue! =========================#
#useful for rapid flag initialization !
using Luma
n = 1000000
T = UInt8
dst = CuArray{T}([1 for _ in (1:n)])
Nitem = 8 #better

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    Luma.setvalue!(dst, T(1), Nitem=Nitem)
end

prof = CUDA.@profile Luma.setvalue!(dst, T(1); Nitem=Nitem)

