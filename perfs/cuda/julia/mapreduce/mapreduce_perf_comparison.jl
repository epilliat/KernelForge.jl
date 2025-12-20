#=
MapReduce Performance Benchmarking Script
==========================================
Compares Luma.mapreduce! against CUDA.mapreduce and AcceleratedKernels.mapreduce
across different data types and configurations.

Methodology:
- 500ms warm-up phase to ensure JIT compilation and GPU initialization
- CUDA.@profile for accurate kernel timing (excludes launch overhead)
=#

using Revise
using Pkg
Pkg.activate("$(@__DIR__())/../../")

using Luma
using KernelAbstractions, CUDA, BenchmarkTools
using AcceleratedKernels
using Quaternions

#=============================================================================
  SECTION 1: Float32 Baseline (n=1M)
  ---------------------------------
  Basic mapreduce with default parameters.
  Tests the simplest case: identity map + addition reduce.
=============================================================================#

n = 1_000_000
T = Float32
op = +
f(x) = x

src = CuArray{T}(1:n)
dst = CuArray{T}([0])

# Warm-up ensures JIT compilation doesn't affect profiling
start_time = time()
while time() - start_time < 0.5
  CUDA.@sync Luma.mapreduce!(f, op, dst, src)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src)

#=============================================================================
  SECTION 2: Float32 with Pre-allocated Temporaries
  -------------------------------------------------
  - Pre-allocates tmp buffer to exclude allocation overhead from timing
  - Uses UInt64 flags (default behavior skips flag initialization)
  - Nitem=4: each thread processes 4 elements (improves ILP)
=============================================================================#

#%%
n = 1_000_000
T = Float32
FlagType = UInt64
Nitem = 4
op = +
f(x) = x

src = CuArray{T}(1:n)
dst = CuArray{T}(zeros(n))

# Pre-allocate temporaries to measure pure kernel performance
tmp = Luma.get_allocation(Luma.mapreduce1d!, (src,); blocks=100, FlagType=FlagType)

start_time = time()
while time() - start_time < 0.5
  CUDA.@sync Luma.mapreduce!(f, op, dst, src; tmp, FlagType, Nitem)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src; tmp, FlagType, Nitem)

#=============================================================================
  SECTION 3: CUDA.jl Reference (Float32)
  --------------------------------------
  Baseline comparison using CUDA.jl's built-in mapreduce.
  Note: Returns value directly (allocating), not in-place like Luma.
=============================================================================#

#%%
start_time = time()
while time() - start_time < 0.5
  CUDA.@sync mapreduce(f, op, src)
end
prof = CUDA.@profile mapreduce(f, op, src)

#=============================================================================
  SECTION 4: AcceleratedKernels Reference (Float32)
  -------------------------------------------------
  Comparison with AcceleratedKernels.jl (portable GPU library).
  Requires explicit init value for type stability.
=============================================================================#

#%%
start_time = time()
while time() - start_time < 0.5
  CUDA.@sync AcceleratedKernels.mapreduce(f, op, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.mapreduce(f, op, src; init=T(0))

#=============================================================================
  SECTION 5: Float64 (n=1M)
  -------------------------
  Double precision test.
  - Nitem=1: fewer items per thread due to larger register pressure
  - Memory bandwidth bound: 2x bytes per element vs Float32
=============================================================================#

#%%
n = 1_000_000
T = Float64
op = +
f(x) = x

src = CuArray{T}(1:n)
dst = CuArray{T}([0])

start_time = time()
while time() - start_time < 0.5
  CUDA.@sync Luma.mapreduce!(f, op, dst, src; Nitem=1)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src; Nitem=1)

#%% CUDA.jl reference (Float64)
start_time = time()
while time() - start_time < 0.5
  CUDA.@sync mapreduce(f, op, src)
end
prof = CUDA.@profile mapreduce(f, op, src)

#%% AcceleratedKernels reference (Float64)
start_time = time()
while time() - start_time < 0.5
  CUDA.@sync AcceleratedKernels.mapreduce(f, op, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.mapreduce(f, op, src; init=T(0))

#=============================================================================
  SECTION 6: UInt8 (n=1M)
  -----------------------
  Small type test (1 byte per element).
  - Nitem=8: more items per thread to compensate for small element size
  - Challenges memory coalescing efficiency
  - Tests vectorized load capabilities (can load 8 bytes = 8 UInt8 at once)
=============================================================================#

#%%
n = 1_000_000
T = UInt8
op = +
f(x) = x

src = CuArray{T}(fill(0x01, n))
dst = CuArray{T}(zeros(UInt8, n))

start_time = time()
while time() - start_time < 0.5
  CUDA.@sync Luma.mapreduce!(f, op, dst, src;)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src;)

#%% CUDA.jl reference (UInt8)
start_time = time()
while time() - start_time < 0.5
  CUDA.@sync mapreduce(f, op, src)
end
prof = CUDA.@profile mapreduce(f, op, src)

#%% AcceleratedKernels reference (UInt8)
start_time = time()
while time() - start_time < 0.5
  CUDA.@sync AcceleratedKernels.mapreduce(f, op, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.mapreduce(f, op, src; init=T(0))

#=============================================================================
  SECTION 7: UnitFloat8 Custom Type (n=1M)
  ----------------------------------------
  Tests Luma's custom 8-bit float type.
  - f(x) = Float32(x): promotes to Float32 for accumulation (avoids overflow)
  - g(x) = x: identity for output conversion
  - Manual tmp allocation with explicit config for fine control

  Note: sum(src_cpu) would overflow; Float32 accumulation is necessary.
=============================================================================#

#%%
using Luma: UnitFloat8

n = 1_000_000
T = UnitFloat8
op = +
f(x) = Float32(x)  # Promote to Float32 during reduction to avoid overflow
g(x) = x

src_cpu = [rand(UnitFloat8) for _ in 1:n]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([T(0)])

# Custom configuration: 256 threads/block, 1000 blocks
# H=Float32 because f promotes UnitFloat8 to Float32
tmp = Luma.get_allocation(Luma.mapreduce1d!, (src,); blocks=1000, H=Float32, FlagType=UInt64)

start_time = time()
while time() - start_time < 0.5
  CUDA.@sync Luma.mapreduce!(f, +, dst, src; tmp)
end
CUDA.@profile Luma.mapreduce!(f, +, dst, src; tmp)

# Verification (commented out):
# sum(Float32.(src_cpu)), dst[1:1], sum(src_cpu)  # sum(src_cpu) overflows

#%% CUDA.jl reference (UnitFloat8)
start_time = time()
while time() - start_time < 0.5
  CUDA.@sync mapreduce(f, op, src)
end
prof = CUDA.@profile mapreduce(f, op, src)

#%% AcceleratedKernels reference (UnitFloat8)
start_time = time()
while time() - start_time < 0.5
  CUDA.@sync AcceleratedKernels.mapreduce(f, op, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.mapreduce(f, op, src; init=T(0))