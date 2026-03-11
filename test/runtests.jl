# ─────────────────────────────────────────────────────────────────────────────
# Backend selection
# ─────────────────────────────────────────────────────────────────────────────
using Pkg
include("meta_helpers.jl")

TEST_BACKEND = get(ENV, "TEST_BACKEND") do
    backend_str = has_cuda() ? "cuda" : has_roc() ? "roc" : "unknown"
    @info "TEST_BACKEND not set, defaulting to $backend_str"
    backend_str
end


Pkg.activate("test/envs/$TEST_BACKEND")
#Pkg.activate("envs/$TEST_BACKEND") # when running tests
Pkg.instantiate()

using Test
using KernelAbstractions
import KernelAbstractions as KA
using KernelForge
import KernelForge as KF
using Random




if TEST_BACKEND == "cuda"
    using CUDA
    if !CUDA.functional()
        @warn "No CUDA device found — skipping tests"
        exit(0)
    end
    AT = CuArray
    backend = CUDABackend()
elseif TEST_BACKEND == "roc"
    using AMDGPU
    if !AMDGPU.functional()
        @warn "No AMDGPU device found — skipping tests"
        exit(0)
    end
    AT = ROCArray
    backend = ROCBackend()
    arch = KF.detect_arch(backend, Val(1))
else
    error("Unknown backend: $TEST_BACKEND")
end
include("helpers.jl")
#%%
include("general_routine.jl")





