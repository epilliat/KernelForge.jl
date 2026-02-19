using Test
using KernelAbstractions
import KernelAbstractions as KA
using KernelForge
import KernelForge as KF
using Random


const BACKEND_ARRAY_TYPES = Dict{Any,Any}(CPU() => Array)

include("helpers.jl")


try
    using CUDA
    if CUDA.functional()
        BACKEND_ARRAY_TYPES[CUDABackend()] = CuArray
        @info "CUDA backend available, running CUDA tests"
        AT = CuArray
        backend = CUDABackend()
        @testset "CUDA" begin
            include("general_routine.jl")
        end
    else
        @warn "CUDA not functional, skipping CUDA tests"
    end
catch e
    @warn "CUDA not available, skipping CUDA tests" exception = e
end





