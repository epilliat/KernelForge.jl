#=
architecture.jl
================
Reusable utilities for benchmark result folder setup.
Detects the current GPU and maps it to a short canonical name,
then exposes `RESULT_DIR` for CSV output.
Usage in any benchmark script:
    include(joinpath(@__DIR__, "architecture.jl"))
    # → GPU_TAG and RESULT_DIR are now defined
    # → results/<GPU_TAG>/ directory is created
=#
using CUDA
using JSON3

# ---------------------------------------------------------------------------
# GPU architecture type hierarchy
# ---------------------------------------------------------------------------

abstract type AbstractDevice end

abstract type CUDADevice <: AbstractDevice end
abstract type AMDDevice <: AbstractDevice end   # placeholder

# NVIDIA microarchitectures
abstract type Volta <: CUDADevice end
abstract type Turing <: CUDADevice end
abstract type Ampere <: CUDADevice end
abstract type Ada <: CUDADevice end
abstract type Hopper <: CUDADevice end
abstract type Blackwell <: CUDADevice end

# Volta
struct V100 <: Volta end

# Turing
struct RTX2080 <: Turing end
struct RTX2070 <: Turing end
struct T4 <: Turing end

# Ampere
struct A100 <: Ampere end
struct A10 <: Ampere end
struct A30 <: Ampere end
struct A40 <: Ampere end
struct A6000 <: Ampere end
struct RTX3090 <: Ampere end
struct RTX3080 <: Ampere end
struct RTX3070 <: Ampere end
struct RTX3060 <: Ampere end
struct RTX3000 <: Ampere end

# Hopper
struct H100 <: Hopper end
struct H200 <: Hopper end

# Blackwell
struct B100 <: Blackwell end
struct B200 <: Blackwell end

# Ada Lovelace
struct RTX4090 <: Ada end
struct RTX4080 <: Ada end
struct RTX4070 <: Ada end
struct RTX4060 <: Ada end
struct RTX5000 <: Ada end
struct RTX4500 <: Ada end
struct RTX4000 <: Ada end
struct RTX3500 <: Ada end
struct RTX2000 <: Ada end
struct RTX1000 <: Ada end
struct RTX500 <: Ada end

# Fallback
struct UnknownCUDADevice <: CUDADevice end
struct UnknownDevice <: AbstractDevice end

# ---------------------------------------------------------------------------
# GPU short-name detection — returns a Symbol
# ---------------------------------------------------------------------------

"""
    gpu_short_name(dev=CUDA.device()) -> Symbol
"""
function gpu_short_name(dev::CUDA.CuDevice=CUDA.device())
    raw = CUDA.name(dev)
    for (pattern, tag) in [
        (r"(?i)A100", :A100),
        (r"(?i)A10[^0]", :A10),
        (r"(?i)A30", :A30),
        (r"(?i)A40", :A40),
        (r"(?i)A6000", :A6000),
        (r"(?i)V100", :V100),
        (r"(?i)H200", :H200),
        (r"(?i)H100", :H100),
        (r"(?i)B200", :B200),
        (r"(?i)B100", :B100),
        (r"(?i)T4", :T4),
        (r"(?i)4090", :RTX4090),
        (r"(?i)4080", :RTX4080),
        (r"(?i)4070", :RTX4070),
        (r"(?i)4060", :RTX4060),
        (r"(?i)3090", :RTX3090),
        (r"(?i)3080", :RTX3080),
        (r"(?i)3070", :RTX3070),
        (r"(?i)3060", :RTX3060),
        (r"(?i)2080", :RTX2080),
        (r"(?i)2070", :RTX2070),
        (r"(?i)Titan", :Titan),
        (r"(?i)RTX[ _]?5000", :RTX5000),
        (r"(?i)RTX[ _]?4500", :RTX4500),
        (r"(?i)RTX[ _]?4000", :RTX4000),
        (r"(?i)RTX[ _]?3500", :RTX3500),
        (r"(?i)RTX[ _]?3000", :RTX3000),
        (r"(?i)RTX[ _]?2000", :RTX2000),
        (r"(?i)RTX[ _]?1000", :RTX1000),
        (r"(?i)RTX[ _]?500", :RTX500),
        (r"(?i)MI300", :MI300),
        (r"(?i)MI250", :MI250),
        (r"(?i)MI100", :MI100),
        (r"(?i)RX_?7900", :RX7900),
        (r"(?i)RX_?6900", :RX6900),
        (r"(?i)RX_?6800", :RX6800),
    ]
        occursin(pattern, raw) && return tag
    end
    clean = replace(raw, r"(?i)^(NVIDIA|AMD|Intel)\s+" => "")
    return Symbol(replace(first(clean, 20), " " => "_"))
end

# ---------------------------------------------------------------------------
# Val(Symbol) → architecture struct  (metaprogrammed)
# ---------------------------------------------------------------------------

const _CUDA_DEVICE_TYPES = [
    :V100,
    :T4, :RTX2080, :RTX2070,
    :A100, :A10, :A30, :A40, :A6000,
    :RTX3090, :RTX3080, :RTX3070, :RTX3060, :RTX3000,
    :H100, :H200,
    :B100, :B200,
    :RTX4090, :RTX4080, :RTX4070, :RTX4060,
    :RTX5000, :RTX4500, :RTX4000, :RTX3500, :RTX2000, :RTX1000, :RTX500,
]

# Val(CuDevice) → call gpu_short_name at specialization time, delegate to symbol dispatch
@generated function detect_device(::Val{dev}) where dev
    dev isa CUDA.CuDevice || return :(UnknownCUDADevice())
    tag = gpu_short_name(dev)
    return :(detect_device(Val($tag)))
end

for T in _CUDA_DEVICE_TYPES
    @eval detect_device(::Val{$T}) = $(T)()
end

# ---------------------------------------------------------------------------
# Exported constants
# ---------------------------------------------------------------------------

const GPU_TAG = string(gpu_short_name())
const RESULT_DIR = joinpath(@__DIR__, "results", GPU_TAG)
mkpath(RESULT_DIR)
println("GPU         : $(CUDA.name(CUDA.device()))  →  $GPU_TAG")
println("Results dir : $RESULT_DIR")

# ---------------------------------------------------------------------------
# Save device info JSON
# ---------------------------------------------------------------------------

function save_device_info(path::String)
    dev = CUDA.device()
    attrs = Dict(string(a) => CUDA.attribute(dev, a) for a in instances(CUDA.CUdevice_attribute))
    info = Dict(
        "name" => CUDA.name(dev),
        "gpu_tag" => GPU_TAG,
        "compute_capability" => string(CUDA.capability(dev)),
        "total_memory_bytes" => CUDA.totalmem(dev),
        "cuda_driver_version" => string(CUDA.driver_version()),
        "cuda_runtime_version" => string(CUDA.runtime_version()),
        "attributes" => attrs,
    )
    open(path, "w") do io
        JSON3.write(io, info)
    end
    println("Device info  : $path")
end

save_device_info(joinpath(RESULT_DIR, "device_info.json"))