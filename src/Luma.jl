module Luma

using MemoryAccess
using KernelAbstractions
using GPUArraysCore


const warpsz = 32 # TODO: This might change from one architecture to another

export scan!, vcopy!, mapreduce!
export get_allocation



include("helpers.jl")

include("copy/copy_kernel.jl")
include("copy/copy.jl")

include("mapreduce/1D/mapreduce1d_kernel.jl")
include("mapreduce/1D/mapreduce1d.jl")
include("mapreduce/mapreduce.jl")

include("scan/scan_kernel.jl")
include("scan/scan.jl")


include("extras/unitfloats.jl")
include("linear_algebra/reductions.jl")


end # module Luma
