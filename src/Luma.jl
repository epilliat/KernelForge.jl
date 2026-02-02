module Luma

using KernelIntrinsics
using KernelAbstractions
using GPUArraysCore
using ArgCheck

const warpsz = 32 # TODO: This might change from one architecture to another

include("helpers.jl")

include("copy/copy_kernel.jl")
include("copy/copy.jl")

include("mapreduce/1D/mapreduce1d_kernel.jl")
include("mapreduce/1D/mapreduce1d.jl")

include("mapreduce/2D/vecmat_kernel.jl")
include("mapreduce/2D/matvec_kernel.jl")

include("mapreduce/2D/vecmat.jl")
include("mapreduce/2D/matvec.jl")

include("mapreduce/2D/mapreduce2d.jl")


include("mapreduce/mapreduce.jl")

include("scan/scan_kernel.jl")
include("scan/scan.jl")


include("extras/unitfloats.jl")
include("linear_algebra/reductions.jl")


end # module Luma
