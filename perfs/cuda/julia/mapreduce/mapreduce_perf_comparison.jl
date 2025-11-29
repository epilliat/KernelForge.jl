using Revise
using Pkg

Pkg.activate("$(@__DIR__())/../../")

using Luma
using KernelAbstractions, CUDA, BenchmarkTools
using AcceleratedKernels
using Quaternions


n = 1000000
T = Float32
FlagType = UInt8
op = +
f(x) = x
src = CuArray{T}([i for i in (1:n)])
dst = CuArray{T}([0])

start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync Luma.mapreduce!(f, op, dst, src)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src)

#%% not counting tmp memory allocation, and not initializing flags (default behavior for UInt64 flags)
n = 1000000
T = Float32
FlagType = UInt64
op = +
Nitem = 8
f(x) = x
src = CuArray{T}([i for i in (1:n)])
dst = CuArray{T}([0 for _ in (1:n)])

start_time = time()
tmp = get_allocation(Luma.mapreduce1d!, f, op, dst, (src,); FlagType=FlagType, Nitem=Nitem)

while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync Luma.mapreduce!(f, op, dst, src; tmp=tmp, FlagType=FlagType, Nitem=Nitem)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src; tmp=tmp, FlagType=FlagType, Nitem=Nitem)





#%% CUDA
start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync mapreduce(f, op, src)
end
prof = CUDA.@profile mapreduce(f, op, src)


#%% Accelerated Kernels
start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync AcceleratedKernels.mapreduce(f, op, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.mapreduce(f, op, src; init=T(0))


#%%============= Float64, larger n ============================
n = 1000000
T = Float64
FlagType = UInt8
op = +
f(x) = x
src = CuArray{T}([i for i in (1:n)])
dst = CuArray{T}([0])

#%%
start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync Luma.mapreduce!(f, op, dst, src, Nitem=1)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src, Nitem=1)



#%% CUDA
start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync mapreduce(f, op, src)
end
prof = CUDA.@profile mapreduce(f, op, src)


#%% Accelerated Kernels
start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync AcceleratedKernels.mapreduce(f, op, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.mapreduce(f, op, src; init=T(0))




#%%============================ UInt8 ================================

n = 100000000
op = +
f(x) = x
T = UInt8
src_cpu = [0x01 for _ in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync Luma.mapreduce!(f, op, dst, src, Nitem=16)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src, Nitem=16)
#%%

start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync mapreduce(f, op, src)
end
prof = CUDA.@profile mapreduce(f, op, src)

#%%
#%% Accelerated Kernels
start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync AcceleratedKernels.mapreduce(f, op, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.mapreduce(f, op, src; init=T(0))


#%% UnitFloat8

using Luma: UnitFloat8
n = 100000000
ad(x...) = +(x...)
f(x) = Float32(x)
T = UnitFloat8
src_cpu = ([rand(UnitFloat8) for _ in (1:n)])
src = CuArray{T}(src_cpu)
dst = CuArray{T}(T.([0 for _ in (1:n)]))
tmp = Luma.get_allocation(Luma.mapreduce1d!, f, op, dst, (src,); g=g, Nitem=Nitem,
    FlagType=UInt64, config=(workgroup=256, blocks=1000))
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync Luma.mapreduce!(f, ad, dst, src, Nitem=16, tmp=tmp)
end
CUDA.@profile Luma.mapreduce!(f, ad, dst, src, Nitem=16, tmp=tmp)
sum(Float32.(src_cpu)), dst[1:1], sum(src_cpu) #sum(src_cpu) gets it wrong due to overflow


#%%

start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync mapreduce(f, op, src)
end
prof = CUDA.@profile mapreduce(f, op, src)

#%%
#%% Accelerated Kernels
start_time = time()

while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync AcceleratedKernels.mapreduce(f, op, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.mapreduce(f, op, src; init=T(0))

