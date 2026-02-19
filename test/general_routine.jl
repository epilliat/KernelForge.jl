@testset "mapreduce" begin
    include("mapreduce/mapreduce1d_test.jl")
    include("mapreduce/mapreduce2d_test.jl")
    include("mapreduce/mapreduce_dims_test.jl")
    include("mapreduce/mapreduce_test.jl")
    include("mapreduce/vecmat_test.jl")
    include("mapreduce/matvec_test.jl")
end
@testset "copy" begin
    include("copy/copy_test.jl")
end
@testset "scan" begin
    include("scan/scan_test.jl")
end
@testset "views" begin
    include("views/views_test.jl")
end
@testset "argmax" begin
    include("search/argmax_test.jl")
end