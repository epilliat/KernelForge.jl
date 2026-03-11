

# RTX1000

n = 10^4
p = 10^4
T = Float32
src = fill!(AT{T}(undef, n, p), one(T))
x = fill!(AT{T}(undef, p), one(T))


CUDA.@profile src * x


# ==== Reduction regime
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=1, chunksz=1, Nblocks=128, workgroup=128) # n=10^0, p=10^8
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=2, chunksz=2, Nblocks=32, workgroup=128) # n=10^1, p=10^7
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=8, Nblocks=32, workgroup=128) # n=10^2, p=10^6
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=8, Nblocks=16, workgroup=128) # n=10^3, p=10^5
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=8, Nblocks=2, workgroup=512) # n=10^4, p=10^4
# ==== Transition to copy regime ======
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=8, Nblocks=1, workgroup=512) # 10^5, 10^3
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=8, Nblocks=1, workgroup=512) # 10^6, 10^2
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=64, Nblocks=1, workgroup=256) # 10^7, 10^1
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=256, Nblocks=1, workgroup=256) # 10^8, 10^0

CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=1, chunksz=1, Nblocks=128, workgroup=128)


n = 10^6
p = 10^0
T = Float32
src = fill!(AT{T}(undef, n, p), one(T))
x = fill!(AT{T}(undef, p), one(T))


CUDA.@profile src * x
# ==== Reduction regime
prof = CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=1, chunksz=1, Nblocks=128, workgroup=128) # n=10^0, p=10^8
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=2, chunksz=2, Nblocks=32, workgroup=128) # n=10^1, p=10^5
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=8, Nblocks=32, workgroup=128) # n=10^2, p=10^4
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=8, Nblocks=16, workgroup=128) # n=10^3, p=10^3
# ==== Transition to copy regime ======
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=16, Nblocks=1, workgroup=128) # n=10^4, p=10^2
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=64, Nblocks=1, workgroup=128) # 10^5, 10^1
CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=4, chunksz=256, Nblocks=1, workgroup=128) # 10^6, 10^0

CUDA.@profile KernelForge.matvec(*, +, src, x; Nitem=1, chunksz=1, Nblocks=128, workgroup=128)
CUDA.@profile KernelForge.matvec(*, +, src, x)
isapprox(KernelForge.matvec(*, +, src, x; Nitem=16), src * x)
KernelForge.matvec(*, +, src, x; chunksz=8, Nblocks=1, workgroup=1024)
KA.synchronize(backend)
# Simple profiling example (without warmup here which gives slower results.)

@btime (src * x; KA.synchronize(backend))
@btime (KernelForge.matvec(*, +, src, x; chunksz=32, Nblocks=1, workgroup=1024); KA.synchronize(backend))
@btime (KernelForge.matvec(*, +, src, x; chunksz=16); KA.synchronize(backend))
#@btime (KernelForge.vecmat(*, +, x, src'); KA.synchronize(backend))
# CUDA.@profile src * x
# CUDA.@profile KernelForge.matvec(*, +, src, x)


#%%
using KernelForge, CUDA

# helpers
function test_matvec(n, p; kwargs...)
    A = CUDA.rand(Float32, n, p)
    x = CUDA.rand(Float32, p)
    ref = Array(A) * Array(x)
    out = Array(KF.matvec(A, x; kwargs...))
    maxerr = maximum(abs.(out .- ref))
    println("n=$n p=$p kwargs=$kwargs  maxerr=$maxerr  $(maxerr < 1e-3 ? "✅" : "❌")")
end

# Nblocks=3 (non power of 2)
test_matvec(128, 256; Nblocks=3, Nitem=2)
test_matvec(256, 512; Nblocks=4)

# Nblocks=6
test_matvec(128, 512; Nblocks=6)

# Nblocks=5, chunksz*Nblocks < warpsz  (e.g. chunksz=4, warpsz=32 → 4*5=20 < 32)
test_matvec(64, 160; Nblocks=5, chunksz=4)

# Nblocks=5, chunksz*Nblocks > warpsz but < workgroup  (edge case path)
test_matvec(64, 640; Nblocks=5, chunksz=8)

# Nblocks=7, stress the modular shuffle condition
test_matvec(128, 896; Nblocks=7, chunksz=4)

# nothing x (row reduction)
function test_matvec_nothing(n, p; kwargs...)
    A = CUDA.rand(Float32, n, p)
    ref = vec(sum(Array(A), dims=2))
    out = Array(matvec(identity, +, A, nothing; kwargs...))
    maxerr = maximum(abs.(out .- ref))
    println("n=$n p=$p kwargs=$kwargs  maxerr=$maxerr  $(maxerr < 1e-3 ? "✅" : "❌")")
end

test_matvec_nothing(128, 256; Nblocks=3)
test_matvec_nothing(64, 640; Nblocks=5, chunksz=8)
test_matvec_nothing(128, 896; Nblocks=7, chunksz=4)