# ---------------------------------------------------------------------------
# GPU arch type hierarchy
# ---------------------------------------------------------------------------

abstract type AbstractArch end

abstract type CUDAArch <: AbstractArch end
abstract type AMDArch <: AbstractArch end   # placeholder

# NVIDIA microarchitectures
abstract type Volta <: CUDAArch end
abstract type Turing <: CUDAArch end
abstract type Ampere <: CUDAArch end
abstract type Ada <: CUDAArch end
abstract type Hopper <: CUDAArch end
abstract type Blackwell <: CUDAArch end

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
struct UnknownCUDAArch <: CUDAArch end
struct UnknownArch <: AbstractArch end

function arch_tag end
function detect_arch end

# ---------------------------------------------------------------------------
# detect_arch: CuDevice / AbstractArray / Val → AbstractArch
# ---------------------------------------------------------------------------


const CUDA_ARCH_PATTERNS = Pair{Regex,Symbol}[
    r"(?i)A100"=>:A100, r"(?i)A10[^0]"=>:A10,
    r"(?i)A30\b"=>:A30, r"(?i)A40\b"=>:A40,
    r"(?i)A6000"=>:A6000, r"(?i)V100"=>:V100,
    r"(?i)H200"=>:H200, r"(?i)H100"=>:H100,
    r"(?i)B200"=>:B200, r"(?i)B100"=>:B100,
    r"(?i)\bT4\b"=>:T4, r"(?i)Titan"=>:Titan,
    r"(?i)MI300"=>:MI300, r"(?i)MI250"=>:MI250, r"(?i)MI100"=>:MI100,
    r"(?i)RX_?7900"=>:RX7900, r"(?i)RX_?6900"=>:RX6900, r"(?i)RX_?6800"=>:RX6800,
]

"""
    arch_tag(dev::CuDevice) -> Symbol

Return a canonical Symbol identifying the GPU architecture (e.g. `:A100`, `:H100`).
Used internally to drive `Val`-based dispatch in `detect_arch`.
"""
function arch_tag(dev)
    #println("abc")
    raw = KI.name(dev)
    for (pattern, tag) in CUDA_ARCH_PATTERNS
        occursin(pattern, raw) && return tag
    end
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

# trigger @eval one first time