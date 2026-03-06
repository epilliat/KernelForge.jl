# ---------------------------------------------------------------------------
# GPU arch type hierarchy
# ---------------------------------------------------------------------------
abstract type AbstractArch end
abstract type CUDAArch <: AbstractArch end
abstract type AMDArch <: AbstractArch end

# NVIDIA microarchitectures
abstract type Volta <: CUDAArch end
abstract type Turing <: CUDAArch end
abstract type Ampere <: CUDAArch end
abstract type Ada <: CUDAArch end
abstract type Hopper <: CUDAArch end
abstract type Blackwell <: CUDAArch end

# AMD GCN / RDNA / CDNA microarchitectures
abstract type GCN <: AMDArch end       # legacy GCN (gfx7xx / gfx8xx)
abstract type CDNA <: AMDArch end      # datacenter: MI100, MI200, MI300
abstract type CDNA2 <: CDNA end
abstract type CDNA3 <: CDNA end
abstract type RDNA <: AMDArch end      # consumer RDNA1: RX 5000
abstract type RDNA2 <: RDNA end        # RX 6000
abstract type RDNA3 <: RDNA end        # RX 7000
abstract type RDNA4 <: RDNA end        # RX 9000 (upcoming)

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

# CDNA1 — MI100 (gfx908)
struct MI100 <: CDNA end
# CDNA2 — MI200 series (gfx90a)
struct MI210 <: CDNA2 end
struct MI250 <: CDNA2 end
struct MI250X <: CDNA2 end
# CDNA3 — MI300 series (gfx940/941/942)
struct MI300A <: CDNA3 end   # APU variant
struct MI300X <: CDNA3 end
struct MI325X <: CDNA3 end
# RDNA1 — RX 5000 series (gfx1010–gfx1012)
struct RX5700XT <: RDNA end
struct RX5700 <: RDNA end
struct RX5600XT <: RDNA end
# RDNA2 — RX 6000 series (gfx1030–gfx1036)
struct RX6950XT <: RDNA2 end
struct RX6900XT <: RDNA2 end
struct RX6800XT <: RDNA2 end
struct RX6800 <: RDNA2 end
struct RX6750XT <: RDNA2 end
struct RX6700XT <: RDNA2 end
struct RX6650XT <: RDNA2 end
struct RX6600XT <: RDNA2 end
struct RX6600 <: RDNA2 end
# RDNA3 — RX 7000 series (gfx1100–gfx1103)
struct RX7900XTX <: RDNA3 end
struct RX7900XT <: RDNA3 end
struct RX7900GRE <: RDNA3 end
struct RX7800XT <: RDNA3 end
struct RX7700XT <: RDNA3 end
struct RX7600 <: RDNA3 end
# RDNA4 — RX 9000 series (gfx1200+, upcoming)
struct RX9070XT <: RDNA4 end
struct RX9070 <: RDNA4 end

# Fallback
struct UnknownCUDAArch <: CUDAArch end
struct UnknownAMDArch <: AMDArch end
struct UnknownArch <: AbstractArch end

function arch_tag end
function detect_arch end

# ---------------------------------------------------------------------------
# detect_arch: CuDevice / ROCmDevice / AbstractArray / Val → AbstractArch
# ---------------------------------------------------------------------------
const CUDA_ARCH_PATTERNS = Pair{Regex,Symbol}[
    r"(?i)A100"=>:A100, r"(?i)A10[^0]"=>:A10,
    r"(?i)A30\b"=>:A30, r"(?i)A40\b"=>:A40,
    r"(?i)A6000"=>:A6000, r"(?i)V100"=>:V100,
    r"(?i)H200"=>:H200, r"(?i)H100"=>:H100,
    r"(?i)B200"=>:B200, r"(?i)B100"=>:B100,
    r"(?i)\bT4\b"=>:T4,
]

const ROC_ARCH_PATTERNS = Pair{Regex,Symbol}[
    # CDNA3
    r"(?i)MI325X"=>:MI325X, r"(?i)MI300X"=>:MI300X,
    r"(?i)MI300A"=>:MI300A,
    # CDNA2
    r"(?i)MI250X"=>:MI250X, r"(?i)MI250\b"=>:MI250,
    r"(?i)MI210"=>:MI210,
    # CDNA1
    r"(?i)MI100"=>:MI100,
    # RDNA3
    r"(?i)RX_?7900_?XTX"=>:RX7900XTX, r"(?i)RX_?7900_?GRE"=>:RX7900GRE,
    r"(?i)RX_?7900_?XT\b"=>:RX7900XT,
    r"(?i)RX_?7800"=>:RX7800XT, r"(?i)RX_?7700"=>:RX7700XT,
    r"(?i)RX_?7600"=>:RX7600,
    # RDNA2
    r"(?i)RX_?6950"=>:RX6950XT, r"(?i)RX_?6900"=>:RX6900XT,
    r"(?i)RX_?6800_?XT"=>:RX6800XT, r"(?i)RX_?6800\b"=>:RX6800,
    r"(?i)RX_?6750"=>:RX6750XT, r"(?i)RX_?6700"=>:RX6700XT,
    r"(?i)RX_?6650"=>:RX6650XT, r"(?i)RX_?6600_?XT"=>:RX6600XT,
    r"(?i)RX_?6600\b"=>:RX6600,
    # RDNA1
    r"(?i)RX_?5700_?XT"=>:RX5700XT, r"(?i)RX_?5700\b"=>:RX5700,
    r"(?i)RX_?5600"=>:RX5600XT,
    # RDNA4
    r"(?i)RX_?9070_?XT"=>:RX9070XT, r"(?i)RX_?9070\b"=>:RX9070,
]

"""
    arch_tag(dev) -> Symbol

Return a canonical Symbol identifying the GPU architecture (e.g. `:A100`, `:MI300X`).
Used internally to drive `Val`-based dispatch in `detect_arch`.
"""
function arch_tag(dev)
    raw = KI.name(dev)
    # Try CUDA patterns first
    for (pattern, tag) in CUDA_ARCH_PATTERNS
        occursin(pattern, raw) && return tag
    end
    # Try ROCm patterns
    for (pattern, tag) in ROC_ARCH_PATTERNS
        occursin(pattern, raw) && return tag
    end
    # NVIDIA RTX fallback (e.g. "NVIDIA GeForce RTX 4090")
    m = match(r"(?i)RTX[ _]?(\d+)", raw)
    m !== nothing && return Symbol("RTX$(m[1])")
    return Symbol(replace(first(raw, 20), " " => "_"))
end

function detect_arch(::Val{dev}) where dev
    tag = arch_tag(dev)
    T = getproperty(KernelForge, tag)
    @eval detect_arch(::Val{$dev}) = $T()
    return Base.invokelatest(detect_arch, Val(dev))
end

detect_arch(dev) = detect_arch(Val(dev))
detect_arch(src::AbstractArray) = detect_arch(KI.device(src))


get_warpsize(::CUDAArch) = 32  # all NVIDIA
get_warpsize(::AMDArch) = 64

# AMD calls warps "wavefronts"
get_warpsize(::GCN) = 64   # all legacy GCN
get_warpsize(::CDNA) = 64   # MI100, MI200, MI300 — always 64
get_warpsize(::RDNA) = 32   # RDNA1/2/3/4 default to 32 (wave32 mode)
