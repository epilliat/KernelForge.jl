using Test
using CUDA
using KernelForge
import KernelForge as KF
using Random




const BACKEND_ARRAY_TYPES = Dict(
    CPU() => Array,
    CUDABackend() => CuArray,
)

@testset "CUDA" begin
    @testset "mapreduce" begin
        include("cuda/mapreduce/mapreduce1d_test.jl")
        include("cuda/mapreduce/mapreduce2d_test.jl")
        include("cuda/mapreduce/mapreduce_dims_test.jl")
        include("cuda/mapreduce/mapreduce_test.jl")
        include("cuda/mapreduce/vecmat_test.jl")
        include("cuda/mapreduce/matvec_test.jl")
    end
    @testset "copy" begin
        include("cuda/copy/copy_test.jl")
    end
    @testset "scan" begin
        include("cuda/scan/scan_test.jl")
    end
    @testset "views" begin
        include("cuda/views/views_1.jl")
    end
    @testset "argmax" begin
        include("cuda/search/argmax.jl")
    end
end
