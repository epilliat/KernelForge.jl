using Test
using KernelAbstractions
import KernelAbstractions as KA
using KernelForge
import KernelForge as KF
using Random


const BACKEND_ARRAY_TYPES = Dict{Any,Any}(CPU() => Array)

include("helpers.jl")


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





