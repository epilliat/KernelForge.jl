module KernelForge

using KernelIntrinsics
using KernelAbstractions
using KernelAbstractions: adapt
using GPUArraysCore
using Atomix: @atomicreplace, @atomic
using ArgCheck


const BitwiseTypes = Union{UInt8,UInt16,UInt32,UInt64,Int8,Int16,Int32,Int64,Float16,Float32,Float64}


include("helpers.jl")
include("algorithms.jl")

include("copy/copy_kernel.jl")
include("copy/copy.jl")

include("mapreduce/1D/mapreduce1d_kernel.jl")
include("mapreduce/1D/mapreduce1d.jl")

include("mapreduce/2D/vecmat_kernel.jl")
include("mapreduce/2D/matvec_kernel.jl")

include("mapreduce/2D/vecmat.jl")
include("mapreduce/2D/matvec.jl")

include("mapreduce/2D/mapreduce2d.jl")

include("mapreduce/ND/mapreduce_dims_kernel.jl")
include("mapreduce/ND/mapreduce_dims.jl")

include("mapreduce/mapreduce.jl")
include("mapreduce/reductions.jl")

include("scan/scan_kernel.jl")
include("scan/scan.jl")


include("search/findfirst_kernel.jl")
include("search/findfirst.jl")

include("search/argmax_kernel.jl")
include("search/argmax.jl")
include("search/highlevel.jl")


include("extras/unitfloats.jl")
include("linear_algebra/reductions.jl")


end # module KernelForge
