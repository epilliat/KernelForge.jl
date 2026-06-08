module KernelForge

using KernelIntrinsics
import KernelIntrinsics as KI

using KernelAbstractions
using KernelAbstractions: adapt
using Atomix: @atomicreplace, @atomic


#========== Helpers =============#



const BitwiseTypes = Union{UInt8,UInt16,UInt32,UInt64,Int8,Int16,Int32,Int64,Float16,Float32,Float64}


include("architectures.jl")
include("helpers.jl")
include("algorithms.jl")
include("allocations.jl")

include("tuning/loader.jl")     # process-global tuning table; populated by extensions
include("tuning/precompile.jl") # compile_kernel_only stub; implemented in backend exts

include("copy/copy_kernel.jl")
include("copy/copy.jl")

include("mapreduce/1D/mapreduce1d_kernel.jl")
include("mapreduce/1D/mapreduce1d.jl")

include("mapreduce/2D/vecmat_kernel.jl")
include("mapreduce/2D/vecmat_simple_kernel.jl")
include("mapreduce/2D/matvec_kernel.jl")

include("mapreduce/2D/vecmat.jl")
include("mapreduce/2D/matvec.jl")

include("mapreduce/2D/mapreduce2d.jl")

include("mapreduce/ND/mapreducedims_kernel.jl")
include("mapreduce/ND/mapreducedims.jl")

include("mapreduce/mapreduce.jl")
include("mapreduce/reductions.jl")

include("scan/scan_kernel.jl")
include("scan/scan.jl")


include("search/findfirst_kernel.jl")
include("search/findfirst.jl")

include("search/argmax_kernel.jl")
include("search/argmax.jl")
include("search/highlevel.jl")


include("sort/uint_map.jl")
include("sort/bucket_histogram_kernel.jl")
include("sort/scan_histogram_kernel.jl")
include("sort/onesweep_kernel.jl")
include("sort/byte_sort_kernel.jl")
include("sort/sort1d.jl")

include("extras/unitfloats.jl")
include("linear_algebra/reductions.jl")


end # module KernelForge
