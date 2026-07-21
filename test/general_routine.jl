#KF.detect_arch(Val(device()))
#include("helpers.jl")
@testset "mapreduce" begin
    include("tests/mapreduce/mapreduce1d_test.jl")
    include("tests/mapreduce/mapreduce2d_test.jl")
    include("tests/mapreduce/mapreducedims_test.jl")
    include("tests/mapreduce/mapreduce_test.jl")
    include("tests/mapreduce/vecmat_test.jl")
    include("tests/mapreduce/matvec_test.jl")
end
@testset "copy" begin
    include("tests/copy/copy_test.jl")
end
@testset "scan" begin
    include("tests/scan/scan_test.jl")
end
@testset "sort" begin
    include("tests/sort/sort_test.jl")
    include("tests/sort/sort_keyval_test.jl")
    include("tests/sort/sortperm_test.jl")
    include("tests/sort/sample_sort_test.jl")
    include("tests/sort/sort_columns_test.jl")
end
@testset "random" begin
    include("tests/random/philox_core_test.jl")
    include("tests/random/uniforms_test.jl")
    include("tests/random/distributions_test.jl")
    include("tests/random/randperm_test.jl")
    include("tests/random/reproducibility_test.jl")
end
@testset "views" begin
    include("tests/views/views_test.jl")
end
@testset "eltype" begin
    include("tests/eltype/eltype_matrix_test.jl")
end
@testset "gemm" begin
    include("tests/gemm/gemm_test.jl")
    include("tests/gemm/gemm_mma_test.jl")
end
@testset "argmax" begin
    include("tests/search/argmax_test.jl")
end
@testset "findfirst" begin
    include("tests/search/findfirst_test.jl")
end