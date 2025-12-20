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
src = CuArray{T}([i for i in (1:n)])
dst = CuArray{T}([0 for _ in (1:n)])

start_time = time()
while time() - start_time < 0.800  # 500ms warm-up
    CUDA.@sync Luma.scan!(op, dst, src)
end
CUDA.@profile Luma.scan!(op, dst, src)

#%% not counting tmp memory allocation, and not initializing flags (default behavior for UInt64 flags)
n = 1000000
T = Float32
FlagType = UInt64
op = +
src = CuArray{T}([i for i in (1:n)])
dst = CuArray{T}([0 for _ in (1:n)])

start_time = time()
tmp = _get_allocation(Luma.scan!, identity, op, dst, src; FlagType=FlagType)
while time() - start_time < 0.800  # 500ms warm-up
    CUDA.@sync Luma.scan!(op, dst, src; tmp=tmp, FlagType=FlagType)
end
CUDA.@profile Luma.scan!(op, dst, src; tmp=tmp, FlagType=FlagType)





#%% CUDA
start_time = time()
while time() - start_time < 0.800  # 500ms warm-up
    CUDA.@sync CUDA.accumulate!(op, dst, src)
end
prof = CUDA.@profile CUDA.accumulate!(op, dst, src)


#%% Accelerated Kernels (not Decouple Lookback which is faster in this case)
start_time = time()
while time() - start_time < 0.800  # 500ms warm-up
    CUDA.@sync AcceleratedKernels.accumulate!(op, dst, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.accumulate!(op, dst, src; init=T(0))


#%%============= Float64, larger n ============================
n = 100000000
T = Float32
op = +
src = CuArray{T}([T(1) for i in (1:n)])
dst = CuArray{T}([0 for _ in (1:n)])

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync Luma.scan!(op, dst, src)
end
tmp = _get_allocation(scan!, op, dst, src)
prof = CUDA.@profile Luma.scan!(op, dst, src)

#%%
n = 100000000
T = Float64
op = +
src = CuArray{T}([i for i in (1:n)])
dst = CuArray{T}([0 for _ in (1:n)])

start_time = time()
while time() - start_time < 0.800  # 500ms warm-up
    CUDA.@sync Luma.scan!(op, dst, src)
end
prof = CUDA.@profile Luma.scan!(op, dst, src)
#%%

start_time = time()
while time() - start_time < 0.800  # 500ms warm-up
    CUDA.@sync CUDA.accumulate!(op, dst, src)
end
prof = CUDA.@profile CUDA.accumulate!(op, dst, src)

#%%
start_time = time()
while time() - start_time < 0.800  # 500ms warm-up
    CUDA.@sync AcceleratedKernels.accumulate!(op, dst, src; init=T(0), alg=AcceleratedKernels.DecoupledLookback())
end
prof = CUDA.@profile AcceleratedKernels.accumulate!(op, dst, src; init=T(0), alg=AcceleratedKernels.DecoupledLookback())







#%% ====================     Quaternions ======================

n = 100000
op = *
T = QuaternionF64
src_cpu = [QuaternionF64(x ./ sqrt(sum(x .^ 2))...) for x in eachcol(randn(4, n))]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync Luma.scan!(op, dst, src; Nitem=4)
end
CUDA.@profile Luma.scan!(op, dst, src; Nitem=4)
#%%

start_time = time()
while time() - start_time < 0.800  # 500ms warm-up
    CUDA.@sync CUDA.accumulate!(op, dst, src)
end
prof = CUDA.@profile CUDA.accumulate!(op, dst, src)

#%%
start_time = time()
while time() - start_time < 0.800  # 500ms warm-up
    CUDA.@sync AcceleratedKernels.accumulate!(op, dst, src; init=T(0), alg=AcceleratedKernels.DecoupledLookback())
end
prof = CUDA.@profile AcceleratedKernels.accumulate!(op, dst, src; init=T(0), alg=AcceleratedKernels.DecoupledLookback())


#%%============================ UInt8 ================================

n = 100000000
op = +
T = UInt8
src_cpu = [0x01 for _ in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync Luma.scan!(op, dst, src)
end
CUDA.@profile Luma.scan!(op, dst, src)
#%%

start_time = time()
while time() - start_time < 0.800  # 500ms warm-up
    CUDA.@sync CUDA.accumulate!(op, dst, src)
end
prof = CUDA.@profile CUDA.accumulate!(op, dst, src)

#%%
start_time = time()
while time() - start_time < 0.800  # 500ms warm-up
    CUDA.@sync AcceleratedKernels.accumulate!(op, dst, src; init=T(0), alg=AcceleratedKernels.DecoupledLookback())
end
prof = CUDA.@profile AcceleratedKernels.accumulate!(op, dst, src; init=T(0), alg=AcceleratedKernels.DecoupledLookback())