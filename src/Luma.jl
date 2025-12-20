module Luma

using MemoryAccess
using KernelAbstractions
using GPUArraysCore
using ArgCheck

const warpsz = 32 # TODO: This might change from one architecture to another

export scan!, vcopy!, mapreduce!
export _get_allocation



include("helpers.jl")

include("copy/copy_kernel.jl")
include("copy/copy.jl")

include("mapreduce/1D/mapreduce1d_kernel.jl")
include("mapreduce/1D/mapreduce1d.jl")

include("mapreduce/2D/vertical/kernels/mapreduce2d_ver_splitwarp.jl")
include("mapreduce/2D/vertical/kernels/mapreduce2d_ver_splitblock.jl")
include("mapreduce/2D/vertical/kernels/mapreduce2d_ver_splitgrid.jl")
include("mapreduce/2D/vertical/mapreduce2d_ver.jl")

include("mapreduce/mapreduce.jl")

include("scan/scan_kernel.jl")
include("scan/scan.jl")


include("extras/unitfloats.jl")
include("linear_algebra/reductions.jl")


end # module Luma
