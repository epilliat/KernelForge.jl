include("meta_helper.jl")
using Pkg
Pkg.activate("perfs/envs/benchenv/$backend_str")
Pkg.instantiate()
using Revise
include("architecture.jl")
include("bench_utils.jl")
using DataFrames
using CSV