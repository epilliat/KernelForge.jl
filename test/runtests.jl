using Pkg
Pkg.activate("dev")
using Revise

using Test
using CUDA
using Luma
using Random
import InteractiveUtils: @which

@testset "CUDA" begin
    @testset "copy" begin
        include("cuda/1D/copy_test.jl")
    end
    @testset "scan" begin
        include("cuda/1D/scan_test.jl")
    end
    @testset "mapreduce" begin
        include("cuda/1D/mapreduce_test.jl")
    end
    @testset "vecmat" begin
        include("cuda/2D/vecmat_test.jl")
    end
    @testset "matvec" begin
        include("cuda/2D/matvec_test.jl")
    end
end
