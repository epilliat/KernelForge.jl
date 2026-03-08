#KF.detect_arch(Val(device()))
include("helpers.jl")
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
@testset "views" begin
    include("tests/views/views_test.jl")
end
@testset "argmax" begin
    include("tests/search/argmax_test.jl")
end
@testset "findfirst" begin
    include("tests/search/findfirst_test.jl")
end