using Test
using KernelAbstractions
import KernelAbstractions as KA
using KernelForge
import KernelForge as KF
using Random


const BACKEND_ARRAY_TYPES = Dict{Any,Any}(CPU() => Array)




# Default to CPU
backend = CPU()
AT = Array

if !isnothing(Base.find_package("CUDA"))
    using CUDA
    if CUDA.functional()
        @info "CUDA backend available"
        backend = CUDABackend()
        AT = CuArray
        BACKEND_ARRAY_TYPES[CUDABackend()] = CuArray
    else
        @warn "CUDA not functional"
    end
    @testset "CUDA" begin
        include("general_routine.jl")
    end
end

if !isnothing(Base.find_package("AMDGPU"))
    using AMDGPU
    if AMDGPU.functional()
        @info "ROC backend available"
        backend = ROCBackend()
        AT = ROCArray
        BACKEND_ARRAY_TYPES[ROCBackend()] = ROCArray
    else
        @warn "AMDGPU not functional"
    end
    @testset "AMDGPU" begin
        include("general_routine.jl")
    end
end







