@inline function default_nitem(::AbstractArch, ::Type{Scan1D}, n, ::Type{T}) where {T}
    return cld(16, sizeof(T))
end

@inline function default_nitem(::Ampere, ::Type{Scan1D}, n, ::Type{T}) where {T}
    return min(16, cld(64, sizeof(T)))
end

@inline function default_nitem(::RTX1000, ::Type{Scan1D}, n, ::Type{T}) where {T}
    sz = sizeof(T)
    cld(16, cld(sz, 2))
end

@inline function default_nitem(::AMDArch, ::Type{Scan1D}, n, ::Type{T}) where T # MI300X
    nitem = n <= cld(sizeof(T), 4) * 10^6 ? cld(32, cld(sizeof(T), 2)) : cld(32, cld(sizeof(T), 4))
    return nitem
end

default_workgroup(arch::AbstractArch, ::Type{Scan1D}, n, ::Type{T}) where T = default_workgroup(arch) # default fallback

@inline function default_workgroup(::AMDArch, ::Type{Scan1D}, n, ::Type{T}) where {T} # tuned for MI300X
    return 1024
end

# ============================================================================
# Public API docstrings
# ============================================================================

"""
    scan(f, op, src; kwargs...) -> GPU array
    scan(op, src; kwargs...) -> GPU array

GPU parallel prefix scan (cumulative reduction) using a decoupled lookback algorithm.

Applies `f` to each element, computes inclusive prefix scan with `op`, and
optionally applies `g` to each output element.

# Arguments
- `f`: Map function applied to each element (defaults to `identity`)
- `op`: Associative binary scan operator
- `src`: Input GPU array

# Keyword Arguments
- `g=identity`: Post-scan transformation applied to each output element
- `tmp=nothing`: Pre-allocated `KernelBuffer` (or `nothing` to allocate automatically)
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=nothing`: Workgroup size (auto-selected if nothing)
- `family=nothing`: Split-path kernel family `:blocked` or `:transposed` for wide
  aggregates (`sizeof ≥ 8`); auto-selected per arch by the autotune if nothing
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)

# Examples
```julia
# Cumulative sum
x = CUDA.rand(Float32, 10_000)
result = scan(+, x)

# Cumulative sum of squares
result = scan(x -> x^2, +, x)

# With post-scan transformation
result = scan(+, x; g = sqrt)

# With pre-allocated buffer for repeated calls
tmp = KernelForge.get_allocation(Scan1D, identity, +, x)
result = scan(+, x; tmp)
```

See also: [`KernelForge.scan!`](@ref) for the in-place version.
"""
function scan end

"""
    scan!(f, op, dst, src; kwargs...)
    scan!(op, dst, src; kwargs...)

In-place GPU parallel prefix scan using a decoupled lookback algorithm.

Applies `f` to each element, computes inclusive prefix scan with `op`, and
optionally applies `g` to each output element, writing results to `dst`.

# Arguments
- `f`: Map function applied to each element (defaults to `identity`)
- `op`: Associative binary scan operator
- `dst`: Output array for scan results
- `src`: Input GPU array

# Keyword Arguments
- `g=identity`: Post-scan transformation applied to each output element
- `tmp=nothing`: Pre-allocated `KernelBuffer` (or `nothing` to allocate automatically)
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=nothing`: Workgroup size (auto-selected if nothing)
- `family=nothing`: Split-path kernel family `:blocked` or `:transposed` for wide
  aggregates (`sizeof ≥ 8`); auto-selected per arch by the autotune if nothing
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
dst = similar(x)

# Cumulative sum
scan!(+, dst, x)

# With pre-allocated buffer for repeated calls
tmp = KernelForge.get_allocation(Scan1D, identity, +, x)
for i in 1:100
    scan!(+, dst, x; tmp)
end
```

See also: [`KernelForge.scan`](@ref) for the allocating version.
"""
function scan! end

# ============================================================================
# Buffer allocation
# ============================================================================

"""
    get_allocation(::Type{Scan1D}, f, op, src, workgroup=nothing, blocks=nothing, arch=nothing)

Allocate a `KernelBuffer` for `scan!`. Useful for repeated scans.

# Arguments
- `f`: Map function (used to infer intermediate eltype)
- `op`: Reduction operator
- `src`: Input GPU array (used to determine backend and eltype)
- `workgroup=nothing`: Workgroup size (auto-selected if nothing)
- `blocks=nothing`: Number of blocks (must match `blocks` used in `scan!`; auto-computed if nothing)
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)

# Returns
A `KernelBuffer` whose fields depend on the aggregate type `H = promote_op(f, eltype(src))`
(and, for the packed-128 case, on the architecture):
- packable `H` (primitive, `sizeof ∈ {1,2,4}`, e.g. `Float32`/`Int32`): a single
  `desc::Vector{UInt64}` packed status+value tile descriptor;
- 8-byte primitive `H` (e.g. `Float64`/`Int64`) **on CDNA3** (MI300): a single
  `desc128::Vector{UInt128}` packed status+value descriptor, read/published with one
  coherent+atomic 16-byte access (see `scan_use_packed128`);
- otherwise (e.g. `Float64` on other archs, complex/tuple/struct aggregates): `partial1`,
  `partial2` (eltype `H`) and `flag` (`UInt8`).
Either way the buffer is sized for `blocks` tiles; pass the same `tmp` to `scan!`.

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
tmp = KernelForge.get_allocation(Scan1D, identity, +, x)

dst = similar(x)
for i in 1:100
    scan!(+, dst, x; tmp)
end
```
"""
function get_allocation(
    ::Type{Scan1D},
    f::F,
    op::O,
    src::AT,
    blocks::Int;
    arch=nothing,      # keyword, NOT positional: a 6th positional would be ambiguous with the
                       # (src, workgroup, blocks) method below. MUST be the same arch `_scan_impl!`
                       # dispatches on, or the buffers here would not match the launched kernel.
) where {F<:Function,O<:Function,AT<:AbstractArray}
    T = eltype(AT)
    H = Base.promote_op(f, T)
    backend = get_backend(src)
    arch = something(arch, detect_arch(src))
    if scan_use_packed128(arch, H)
        # Packed-128 path (CDNA3 + 8-byte primitive H): one UInt128 status+value descriptor per
        # tile, read/published with a single coherent+atomic dwordx4. Depends on (arch, H) only —
        # the same predicate `_scan_impl!` uses — so the buffers always match the launched kernel.
        # Zeroed buffer = STATUS_INVALID, the valid init state.
        desc128 = KernelAbstractions.allocate(backend, UInt128, blocks)
        return KernelBuffer((; desc128))
    end
    if scan_packable(H)
        # Packed path: one UInt64 status+value descriptor per tile (see
        # scan_kernel.jl). Zeroed buffer = STATUS_INVALID, the valid init state.
        desc = KernelAbstractions.allocate(backend, UInt64, blocks)
        return KernelBuffer((; desc))
    end
    partial1 = KernelAbstractions.allocate(backend, H, blocks)
    partial2 = KernelAbstractions.allocate(backend, H, blocks)
    flag = KernelAbstractions.allocate(backend, UInt8, blocks)
    return KernelBuffer((; partial1, partial2, flag))
end

function get_allocation(
    ::Type{Scan1D},
    f::F,
    op::O,
    src::AT,
    workgroup=nothing,
    blocks=nothing,
    arch=nothing
) where {F<:Function,O<:Function,AT<:AbstractArray}
    T = eltype(AT)
    H = Base.promote_op(f, T)
    arch = something(arch, detect_arch(src))
    n = length(src)
    Nitem = default_nitem(arch, Scan1D, n, T)
    workgroup = something(workgroup, default_workgroup(arch, Scan1D, n, T))
    ndrange = cld(n, Nitem)
    blocks = something(blocks, cld(ndrange, workgroup))
    # Forward `arch`: the buffer layout depends on it (packed-128 vs packed-64 vs split), and it must
    # be the SAME arch `_scan_impl!` dispatches on — otherwise `tmp` could hold `desc128` while the
    # launched kernel expects `partial1/2/flag` (or vice versa).
    return get_allocation(Scan1D, f, op, src, blocks; arch)
end

# ============================================================================
# Allocating API
# ============================================================================

# Without map function: forward to f=identity method
function scan(
    op::O,
    src::AT;
    kwargs...
) where {O<:Function,AT<:AbstractArray}
    return scan(identity, op, src; kwargs...)
end

# Main allocating entry point
function scan(
    f::F, op::O,
    src::AT;
    g::G=identity,
    tmp::TMP=nothing,
    Nitem=nothing,
    workgroup=nothing,
    family=nothing,
    arch=nothing
) where {AT<:AbstractArray,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    T = eltype(AT)
    H = Base.promote_op(f, T)
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    dst = KernelAbstractions.allocate(backend, S, length(src))
    scan!(f, op, dst, src; g, tmp, Nitem, workgroup, family, arch)
    return dst
end

# ============================================================================
# In-place API
# ============================================================================

# Without map function: forward to f=identity method
function scan!(
    op::O,
    dst::DS,
    src::AT;
    kwargs...
) where {O<:Function,DS<:AbstractArray,AT<:AbstractArray}
    return scan!(identity, op, dst, src; kwargs...)
end

# Main in-place entry point
function scan!(
    f::F, op::O,
    dst::DS,
    src::AT;
    g::G=identity,
    tmp::TMP=nothing,
    Nitem=nothing,
    workgroup=nothing,
    family=nothing,
    arch=nothing
) where {DS<:AbstractArray,AT<:AbstractArray,F<:Function,O<:Function,G<:Function,TMP<:Union{KernelBuffer,Nothing}}
    n = length(src)
    n == 0 && return dst
    T = eltype(AT)
    H = Base.promote_op(f, T)
    backend = get_backend(src)
    arch = something(arch, detect_arch(src))
    Nitem = something(Nitem, default_nitem(arch, Scan1D, n, T))
    workgroup = something(workgroup, default_workgroup(arch, Scan1D, n, T))
    family = something(family, default_scan_family(arch, Scan1D, n, T))
    ndrange = cld(n, Nitem)
    blocks = cld(ndrange, workgroup)
    _scan_impl!(f, op, g, dst, src, Nitem, workgroup, family, ndrange, blocks, tmp, n, backend, arch)
end

# ============================================================================
# Core implementation
# ============================================================================

# Nothing dispatch: allocate buffer then forward to KernelBuffer dispatch
function _scan_impl!(
    f::F, op::O, g::G,
    dst::DS,
    src::AT,
    Nitem::Int,
    workgroup::Int,
    family::Symbol,
    ndrange::Int,
    blocks::Int,
    ::Nothing,
    n::Int,
    backend,
    arch
) where {DS<:AbstractArray,AT<:AbstractArray,F,O,G}
    tmp = get_allocation(Scan1D, f, op, src, workgroup, blocks, arch)
    _scan_impl!(f, op, g, dst, src, Nitem, workgroup, family, ndrange, blocks, tmp, n, backend, arch)
end

# Whether the packed-scan kernel should device-fence after each descriptor publish
# (prompt cross-block visibility → shorter decoupled-lookback; see scan_kernel.jl).
# Enabled on CUDA (measured +8-13% BW on A100 F32); off on AMD until validated on
# MI300X (its relaxed path is at rocPRIM parity; a cross-XCD fence may regress it).
# Correctness-neutral either way.
@inline scan_desc_fence(::CUDAArch) = true
@inline scan_desc_fence(::AMDArch) = false
@inline scan_desc_fence(::AbstractArch) = false

# Whether the SPLIT-path kernels (scan_kernel! / scan_kernel_transposed!, used for any
# non-packable H — Float64/Int64/Complex/Quaternion/composites) read & publish the
# partial1/partial2/flag descriptor with Device-scope RELAXED atomics (cache-bypassing,
# coherent) instead of the plain cached loads + `@access` Acquire/Release flag of the CUDA
# path. On AMD (gfx942, per-XCD L2 + cross-XCD Infinity Fabric) the F64 wall is the split
# lookback's DEVICE-scope acquire on the flag load AND the Device release on the flag store
# — each forces an L1 invalidate / L2 writeback (`buffer_wbl2`), capping F64 at ~half of
# F32's BW (MI300A 1e9: 24% vs 48%). Bypass makes all descriptor accesses Device-Relaxed
# (fresh from L2, no cache op), and moves ORDERING to cheap WORKGROUP-scope release/acquire
# `@fence`s whose AMD lowering appends an `s_waitcnt vmcnt(0)` DRAIN (no writeback) — exactly
# rocPRIM's `atomic_fence_release_vmem_order_only` / `_acquire_order_only` (WITHOUT_SLOW_FENCES
# gfx94x). Measured MI300A F64: 24%→~35% (Ni16, ~1.5×; 1.93×→1.31× vs rocPRIM), correct
# (stress 910 reps/0 fails). OFF on CUDA: its Acquire/Release is free on a single die AND a
# relaxed flag there is a genuine data race (measured wrong: e=1.0) — CUDA's stronger model
# needs the real acquire. Off for every other/untuned arch (safe default). REQUIRES the KI
# with the workgroup-release vmcnt(0) drain (else the two Device-Relaxed stores race).
@inline scan_desc_bypass(::AMDArch) = true
@inline scan_desc_bypass(::CUDAArch) = false
@inline scan_desc_bypass(::AbstractArch) = false

# The bypass routes partial1/partial2 loads/stores through KI atomics, which on gfx942 exist
# only up to 64 bits. So it applies ONLY to a PRIMITIVE 8-byte aggregate (Float64/Int64/
# UInt64) — exactly the split-path types the F64 win targets. Wider/composite split-path H
# (ComplexF64 16 B, QuaternionF64 32 B, tuples) has no 128-bit atomic → keep the plain path
# (a bypass atomic_load there fails GPU codegen). ≤4-byte primitives never reach here (packed).
@inline scan_bypass_eligible(::Type{H}) where {H} = isprimitivetype(H) && sizeof(H) == 8

# ── PACKED-128 descriptor: supersedes the split path for 8-byte primitives on CDNA3 ─────
# An 8-byte aggregate leaves no room for a status inside 64 bits, so the split path must publish
# {flag, partial} separately — and its lookback pays TWO serialized cross-block round-trips per step
# (read flag, vote, then a DEPENDENT read of the value). Packing {status, value} into a UInt128 and
# accessing it with ONE coherent+atomic 16-byte op (`KI.atomic_load/atomic_store!(…, Device, Relaxed)`
# → a single `global_{load,store}_dwordx4 … sc1`) delivers status+value together: the value is already
# in hand when the vote passes. One round-trip. This is rocPRIM's F64 recipe and it is what closes the
# remaining F64 gap on MI300A (measured 1e9: 1.34× → 1.195× vs rocPRIM; it also supersedes the split
# bypass above, which stays in use on non-CDNA3 AMD). Atomicity also removes the fences — a torn
# status/value pair is impossible.
#
# ARCH gate = CDNA3 (gfx94x / MI300): the KI 128-bit atomic emits `sc1`, which is the CDNA3 spelling.
# CDNA2 (gfx90a / MI250X) and CDNA1 want `glc dlc` / `glc` — until KI dispatches on that, they keep
# the split bypass path. TYPE gate = 8-byte PRIMITIVE H: `reinterpret(UInt64, v)` of a composite does
# not GPU-codegen, and the 128-bit atomic cannot carry an aggregate either (both verified on gfx942),
# so ComplexF64/tuples/structs keep the split path.
#
# Depends on (arch, H) ONLY — never on Nitem/family — so `get_allocation` and `_scan_impl!` cannot
# disagree about which buffers exist. On a packed-128 eligible config this path supersedes BOTH split
# families, so the `family` kwarg has no effect there (the transposed family is measured-dead on
# gfx942 anyway). Blocked `vload` ⇒ Nitem MUST be a power of 2 (checked at launch).
@inline scan_packed128_arch(::CDNA3) = true
@inline scan_packed128_arch(::AbstractArch) = false
@inline scan_use_packed128(arch, ::Type{H}) where {H} =
    scan_packed128_arch(arch) && scan_packable128(H)

# ── Split-path kernel FAMILY: blocked `vload` vs transposed WARP_TRANSPOSE ──────────
# For a WIDE aggregate H the blocked `vload` starves memory-level parallelism as
# items/thread grows (F64 caps ~61% of A100 peak); the transposed load (striped
# coalesced global access + shared-memory transpose, see scan_kernel_transposed!)
# stays ~73-82% at high Nitem → F64 1.28×→1.05× CUB on A100. BUT the transpose only
# pays off on a high-BW card that is MLP-starved by blocked loads: on RTX1000 (which
# saturates its narrow bus at ANY Nitem) it is a measured 7-12% REGRESSION.
#
# So the family is NOT hardcoded per arch — it is an AUTOTUNE-DISCOVERED knob, exactly
# like the matvec 2-family (`generic`/`rowthread`). `default_scan_family` returns
# `:blocked` (the safe default for every UN-tuned arch → zero regression) and the
# inline autotune (`data/tuning/<Arch>_inline.jl`) overrides it to `:transposed` only
# where that kernel actually won on that hardware, per N. The autotune benches BOTH
# families at every (Nitem, workgroup) — including blocked at small tiles — so there is
# no fit-gate blind spot: whichever wins is emitted. See [[project_scan_parity_register_lever]].
@inline default_scan_family(::AbstractArch, ::Type{Scan1D}, n, ::Type{T}) where {T} = :blocked

# Correctness gate for the transposed kernel (arch-independent): only WIDE isbits H,
# and S === H because the store transpose stages g-applied S-values in the H-typed
# shared tile (a type-changing `g` — rare for scan — must fall back to blocked).
@inline scan_transposed_applicable(::Type{H}, ::Type{S}) where {H,S} =
    isbitstype(H) && sizeof(H) >= 8 && S === H

# The transposed shared tile is WG·Nitem elements of H plus a warpsz reduction slot;
# it must fit the 48 KB static shared budget. Tiles that would overflow (very wide
# structs) fall back to the blocked split kernel (correctness-neutral).
@inline scan_transposed_fits(::Type{H}, Nitem::Int, workgroup::Int, warpsz::Int) where {H} =
    (workgroup * Nitem + warpsz) * sizeof(H) <= 49152

# Resolve the requested family to whether the transposed kernel is actually launched:
# it must be applicable (wide isbits H, S===H) AND the tile must fit, AND either the
# family says `:transposed` OR `Nitem` is non-power-of-2. The `!ispow2(Nitem)` clause is
# a HARD SAFETY: the blocked `vload`/`vstore!` kernel only compiles for pow-2 Nitem
# (non-pow-2 hits `vload_multi`, a dynamic call that fails GPU codegen), so a non-pow-2
# Nitem MUST use the transposed kernel. Non-pow-2 Nitem only ever arises for wide types
# (the blocked candidates at non-pow-2 Nitem fail to compile and are skipped in tuning),
# so this is always both safe and correct — it also defends against a mismatched
# default_nitem/default_scan_family regime boundary or a manual non-pow-2 `Nitem=` kwarg.
@inline scan_launch_transposed(family::Symbol, ::Type{H}, ::Type{S}, Nitem, workgroup, warpsz) where {H,S} =
    scan_transposed_applicable(H, S) && scan_transposed_fits(H, Nitem, workgroup, warpsz) &&
    (family === :transposed || !ispow2(Nitem))

# KernelBuffer dispatch: run the kernel
function _scan_impl!(
    f::F, op::O, g::G,
    dst::DS,
    src::AT,
    Nitem::Int,
    workgroup::Int,
    family::Symbol,
    ndrange::Int,
    blocks::Int,
    tmp::KernelBuffer,
    n::Int,
    backend,
    arch
) where {DS<:AbstractArray,AT<:AbstractArray,F,O,G}
    T = eltype(AT)
    S = eltype(DS)
    H = Base.promote_op(f, T)
    src_align = (Int(pointer(src)) ÷ sizeof(T)) % Nitem + 1
    dst_align = (Int(pointer(dst)) ÷ sizeof(S)) % Nitem + 1
    Alignment = src_align == dst_align ? src_align : -1
    warpsz = get_warpsize(arch)
    if scan_use_packed128(arch, H)
        # CDNA3 + 8-byte primitive H: the UInt128 {status,value} descriptor read/published with a
        # single coherent+atomic dwordx4 (one lookback round-trip instead of flag→value). Supersedes
        # BOTH split families, so `family` does not apply here. Same (arch,H) predicate as
        # `get_allocation`, so `tmp.arrays.desc128` is guaranteed to be the buffer that was allocated.
        # Blocked `vload` ⇒ pow-2 Nitem (a non-pow-2 would hit KI `vload_multi`, which fails codegen).
        ispow2(Nitem) || throw(ArgumentError(
            "scan: the packed-128 path (CDNA3 + 8-byte primitive eltype) uses the blocked `vload`, " *
            "which requires a power-of-2 Nitem; got Nitem=$Nitem"))
        fill!(tmp.arrays.desc128, UInt128(0))
        scan_kernel_packed128!(backend, workgroup)(
            f, op, g, dst, src, Val(Nitem),
            tmp.arrays.desc128,
            Val(Alignment), Val(warpsz);
            ndrange = ndrange,
        )
    elseif scan_launch_transposed(family, H, S, Nitem, workgroup, warpsz)
        # Wide-H fast path: striped coalesced load + shared transpose (see
        # scan_kernel_transposed!). Uses the same split (partial1/2/flag) buffers.
        fill!(tmp.arrays.flag, 0x00)
        scan_kernel_transposed!(backend, workgroup)(
            f, op, g, dst, src, Val(Nitem),
            tmp.arrays.partial1, tmp.arrays.partial2, tmp.arrays.flag,
            Val(warpsz), Val(workgroup), Val(scan_desc_bypass(arch) && scan_bypass_eligible(H));
            ndrange = ndrange,
        )
    elseif scan_packable(H)
        fill!(tmp.arrays.desc, UInt64(0))
        scan_kernel_packed!(backend, workgroup)(
            f, op, g, dst, src, Val(Nitem),
            tmp.arrays.desc,
            Val(Alignment), Val(warpsz), Val(scan_desc_fence(arch));
            ndrange = ndrange,
        )
    else
        fill!(tmp.arrays.flag, 0x00)
        scan_kernel!(backend, workgroup)(
            f, op, g, dst, src, Val(Nitem),
            tmp.arrays.partial1, tmp.arrays.partial2, tmp.arrays.flag,
            Val(Alignment), Val(warpsz), Val(scan_desc_bypass(arch) && scan_bypass_eligible(H));
            ndrange = ndrange,
        )
    end
    return dst
end