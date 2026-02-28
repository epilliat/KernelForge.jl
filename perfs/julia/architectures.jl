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
# GPU short-name detection
# ---------------------------------------------------------------------------

"""
    gpu_short_name() -> String

Return a short canonical tag for the current CUDA device,
e.g. "A100", "RTX4090", "MI300". Falls back to a sanitized
16-character truncation of the raw device name.
"""
function gpu_short_name()
    raw = CUDA.name(CUDA.device())
    for (pattern, tag) in [
        (r"(?i)A100", "A100"),
        (r"(?i)A10[^0]", "A10"),
        (r"(?i)A30", "A30"),
        (r"(?i)A40", "A40"),
        (r"(?i)A6000", "A6000"),
        (r"(?i)V100", "V100"),
        (r"(?i)H100", "H100"),
        (r"(?i)H200", "H200"),
        (r"(?i)4090", "RTX4090"),
        (r"(?i)4080", "RTX4080"),
        (r"(?i)4070", "RTX4070"),
        (r"(?i)4060", "RTX4060"),
        (r"(?i)3090", "RTX3090"),
        (r"(?i)3080", "RTX3080"),
        (r"(?i)3070", "RTX3070"),
        (r"(?i)3060", "RTX3060"),
        (r"(?i)2080", "RTX2080"),
        (r"(?i)2070", "RTX2070"),
        (r"(?i)Titan", "Titan"),
        (r"(?i)MI300", "MI300"),
        (r"(?i)MI250", "MI250"),
        (r"(?i)MI100", "MI100"),
        (r"(?i)RX_?7900", "RX7900"),
        (r"(?i)RX_?6900", "RX6900"),
        (r"(?i)RX_?6800", "RX6800"),
        (r"(?i)RTX[ _]?5000", "RTX5000"),
        (r"(?i)RTX[ _]?4500", "RTX4500"),
        (r"(?i)RTX[ _]?4000", "RTX4000"),
        (r"(?i)RTX[ _]?3500", "RTX3500"),
        (r"(?i)RTX[ _]?3000", "RTX3000"),
        (r"(?i)RTX[ _]?2000", "RTX2000"),
        (r"(?i)RTX[ _]?1000", "RTX1000"),
        (r"(?i)RTX[ _]?500", "RTX500"),
    ]
        occursin(pattern, raw) && return tag
    end
    clean = replace(raw, r"(?i)^(NVIDIA|AMD|Intel)\s+" => "")
    return replace(first(clean, 20), " " => "_")
end

# ---------------------------------------------------------------------------
# Exported constants
# ---------------------------------------------------------------------------

const GPU_TAG = gpu_short_name()
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