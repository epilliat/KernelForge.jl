# =============================================================================
# Per-arch matvec tuning loader.
#
# At backend-extension init time, for each detected device we call
# `load_tunings_for!(tag)`, which `include`s `data/tuning/<tag>.jl` if present
# (a tiny Julia file holding a `Dict` literal — no JSON parser at runtime, no
# extra dependency). `lookup_matvec` then provides a `(; Nitem, chunksz,
# Nblocks, workgroup)` tuple for `resolve_parameters` to use ahead of the
# heuristic `default_*` methods.
#
# Schema of `data/tuning/<Arch>.jl` (the `.json` companion is for portability;
# the `.jl` is the runtime artifact):
#
#     Dict{String,Any}(
#         "schema_version" => 1,
#         "gpu_tag"        => "RTX1000",
#         "tuned_at"       => "2026-05-27",
#         "matvec" => Dict{String,Any}(
#             "size_4" => [
#                 (n=1, p=1, Nitem=1, chunksz=1, Nblocks=1, workgroup=128),
#                 # ...
#             ],
#             # "Float32" => [...]   — optional dtype-specific override
#         ),
#     )
# =============================================================================

"""
    TUNING_TABLE

Process-global table of per-arch tunings. Outer key = arch tag (a `Symbol`
matching `nameof(typeof(arch))`, e.g. `:RTX1000`). Inner value = the Dict
read from `data/tuning/<tag>.jl`. Populated lazily by `load_tunings_for!`.
"""
const TUNING_TABLE = Ref{Dict{Symbol,Any}}(Dict{Symbol,Any}())

# Tags whose `<tag>_inline.jl` file has already been loaded (the file holds
# autotune-generated `@inline default_*` method overrides for fast ops like
# mapreduce1d/scan, where dict-based lookup at ~4 µs/call would dominate
# the kernel wall). Loaded once per Julia session.
const INLINE_LOADED = Ref{Set{Symbol}}(Set{Symbol}())

# Schema version we understand. JL files with a different
# `schema_version` are rejected (warned + skipped).
const TUNING_SCHEMA_VERSION = 1

"""
    load_tunings_for!(tag::Symbol)

Load tuning data for arch tag `tag` from two optional files in `data/tuning/`:

1. `<tag>.jl` — a `Dict{String,Any}` literal of per-cell tunings (matvec,
   vecmat). Cached into `TUNING_TABLE[][tag]` and queried at runtime by
   `lookup_matvec` / `lookup_vecmat`.
2. `<tag>_inline.jl` — `@inline default_*` method definitions for fast ops
   (mapreduce1d, scan). Evaluated once per session; subsequent
   `default_nitem(::<Arch>, ::Type{<Op>}, n, T)` calls inline to a
   regime-split formula instead of paying ~4 µs of dict lookup.

Both files are optional. Either or both may exist per arch. No-op if a
file is missing or has already been loaded.

Intended to be called from a backend extension's `__init__` once per
detected device — see `ext/KernelForgeCUDAExt.jl`.
"""
function load_tunings_for!(tag::Symbol)
    root = pkgdir(@__MODULE__)
    root === nothing && return nothing

    # 2. Inline-method file (load once per tag, before dict so methods are
    # visible even when the dict half is absent).
    if !(tag in INLINE_LOADED[])
        push!(INLINE_LOADED[], tag)
        inline_path = joinpath(root, "data", "tuning", string(tag) * "_inline.jl")
        if isfile(inline_path)
            try
                Base.include(@__MODULE__, inline_path)
            catch e
                @warn "tuning: failed to load inline $(inline_path)" exception=(e, catch_backtrace())
            end
        end
    end

    # 1. Dict-based tunings.
    haskey(TUNING_TABLE[], tag) && return TUNING_TABLE[][tag]
    path = joinpath(root, "data", "tuning", string(tag) * ".jl")
    isfile(path) || return nothing
    payload = try
        Base.include(@__MODULE__, path)
    catch e
        @warn "tuning: failed to load $(path)" exception=(e, catch_backtrace())
        return nothing
    end
    payload isa AbstractDict || (@warn "tuning: $(path) did not return a Dict"; return nothing)
    v = get(payload, "schema_version", nothing)
    if v != TUNING_SCHEMA_VERSION
        @warn "tuning: $(path) has schema_version=$v, expected $TUNING_SCHEMA_VERSION — skipping"
        return nothing
    end
    TUNING_TABLE[][tag] = payload
    return payload
end

"""
    lookup_matvec(arch, T, n, p) -> Union{NamedTuple,Nothing}

Return `(; Nitem, chunksz, Nblocks, workgroup)` if a tuning entry exists for
`(arch, T, n, p)`, else `nothing`. Looks up dtype-specific override
(`"<DType>"`) first, then size-keyed default (`"size_<sizeof(T)>"`), with
Nitem adapted by sizeof-ratio when reading the size-keyed entry.
"""
function lookup_matvec(arch::AbstractArch, ::Type{T}, n::Int, p::Int) where T
    tag = nameof(typeof(arch))
    haskey(TUNING_TABLE[], tag) || return nothing
    payload = TUNING_TABLE[][tag]
    haskey(payload, "matvec") || return nothing
    mv = payload["matvec"]
    mv isa AbstractDict || return nothing

    type_key = string(T)
    size_key = "size_" * string(sizeof(T))

    # Tier 1: dtype-specific override (e.g. "Float32"). Use Nitem as-is.
    if haskey(mv, type_key)
        cell = _nearest_cell(mv[type_key], n, p)
        cell !== nothing && return _cell_nt(cell)
    end
    # Tier 2: same-size default. Scale Nitem by ref_size / sizeof(T) (a no-op
    # in this case but keeps the code path uniform).
    if haskey(mv, size_key)
        cell = _nearest_cell(mv[size_key], n, p)
        if cell !== nothing
            return _cell_nt_scaled(cell, T, sizeof(T))
        end
    end
    # Tier 3: cross-size adaptation. If only one size has been tuned (e.g.
    # only "size_4" exists for Float32 today), fall through to it and scale
    # Nitem so the @localmem aggregate stays roughly the same byte size.
    # Pick the size whose ref_size is closest to sizeof(T) (so Float64 →
    # size_4 rather than size_2 if both existed).
    best_key, best_diff = nothing, typemax(Int)
    for k in keys(mv)
        startswith(k, "size_") || continue
        ref_size = _parse_size_key(k)
        d = abs(ref_size - sizeof(T))
        if d < best_diff
            best_key  = k
            best_diff = d
        end
    end
    if best_key !== nothing
        cell = _nearest_cell(mv[best_key], n, p)
        if cell !== nothing
            return _cell_nt_scaled(cell, T, _parse_size_key(best_key))
        end
    end
    return nothing
end

# Find the cell with the smallest log2-distance to (n,p), tie-broken to
# smaller (n,p). Cells are expected to be NamedTuples with .n and .p.
function _nearest_cell(cells, n::Int, p::Int)
    isempty(cells) && return nothing
    best = first(cells)
    log_n = log2(max(1, n))
    log_p = log2(max(1, p))
    best_d   = abs(log2(max(1, Int(best.n))) - log_n) +
               abs(log2(max(1, Int(best.p))) - log_p)
    best_sum = log2(max(1, Int(best.n))) + log2(max(1, Int(best.p)))
    for c in cells
        cn, cp = Int(c.n), Int(c.p)
        d   = abs(log2(max(1, cn)) - log_n) + abs(log2(max(1, cp)) - log_p)
        sm  = log2(max(1, cn)) + log2(max(1, cp))
        if d < best_d - 1e-9 || (abs(d - best_d) < 1e-9 && sm < best_sum)
            best = c; best_d = d; best_sum = sm
        end
    end
    return best
end

@inline _cell_nt(c) = (Nitem=Int(c.Nitem), chunksz=Int(c.chunksz),
                       Nblocks=Int(c.Nblocks), workgroup=Int(c.workgroup))

# Scale Nitem so the `@localmem NTuple{Nitem,H}` aggregate stays roughly the
# same byte size when switching from the reference dtype to T. chunksz,
# Nblocks, workgroup carry over unchanged — first-cut hypothesis; see
# verification in docs/RFC. The clamp uses the same cap autotune used.
function _cell_nt_scaled(c, ::Type{T}, ref_size::Int) where T
    sz = sizeof(T)
    Nitem_ref = Int(c.Nitem)
    # Cap matches `autotune.jl:nitems_for`: `cld(32, sz)` controls LLVM
    # unroll-driven compile RAM; the extra `min(.., 16)` guards against
    # `vload` misalignment for `sizeof(T) == 1` (Int8 at Nitem=32 trips
    # ERROR_MISALIGNED_ADDRESS — see autotune.jl:nitems_for note).
    Nitem = clamp(max(1, Nitem_ref * ref_size ÷ sz), 1, min(cld(32, sz), 16))
    return (; Nitem,
              chunksz   = Int(c.chunksz),
              Nblocks   = Int(c.Nblocks),
              workgroup = Int(c.workgroup))
end

function _parse_size_key(s::AbstractString)
    startswith(s, "size_") || error("tuning: malformed size key: $s")
    return parse(Int, @view s[6:end])
end

"""
    lookup_vecmat(arch, T, n, p) -> Union{NamedTuple,Nothing}

Return `(; Nitem, Nthreads, workgroup, blocks)` if a tuning entry exists for
`(arch, T, n, p)`, else `nothing`. Tier order mirrors `lookup_matvec`:
dtype-specific override (`"<DType>"`), then size-keyed default
(`"size_<sizeof(T)>"`), then cross-size adaptation (closest-size section,
Nitem scaled by sizeof-ratio).
"""
function lookup_vecmat(arch::AbstractArch, ::Type{T}, n::Int, p::Int) where T
    tag = nameof(typeof(arch))
    haskey(TUNING_TABLE[], tag) || return nothing
    payload = TUNING_TABLE[][tag]
    haskey(payload, "vecmat") || return nothing
    vm = payload["vecmat"]
    vm isa AbstractDict || return nothing

    type_key = string(T)
    size_key = "size_" * string(sizeof(T))

    if haskey(vm, type_key)
        cell = _nearest_cell(vm[type_key], n, p)
        cell !== nothing && return _vm_cell_nt(cell)
    end
    if haskey(vm, size_key)
        cell = _nearest_cell(vm[size_key], n, p)
        if cell !== nothing
            return _vm_cell_nt_scaled(cell, T, sizeof(T))
        end
    end
    best_key, best_diff = nothing, typemax(Int)
    for k in keys(vm)
        startswith(k, "size_") || continue
        ref_size = _parse_size_key(k)
        d = abs(ref_size - sizeof(T))
        if d < best_diff
            best_key  = k
            best_diff = d
        end
    end
    if best_key !== nothing
        cell = _nearest_cell(vm[best_key], n, p)
        if cell !== nothing
            return _vm_cell_nt_scaled(cell, T, _parse_size_key(best_key))
        end
    end
    return nothing
end

@inline _vm_cell_nt(c) = (Nitem=Int(c.Nitem), Nthreads=Int(c.Nthreads),
                          workgroup=Int(c.workgroup), blocks=Int(c.blocks))

# First-cut cross-size adaptation: scale Nitem by ref_size/sizeof(T); leave
# Nthreads/workgroup/blocks unchanged. Same cap as matvec
# (`min(cld(32, sz), 16)`) — protects byte-strided vload alignment for
# sizeof(T)==1 and bounds LLVM unroll cost.
function _vm_cell_nt_scaled(c, ::Type{T}, ref_size::Int) where T
    sz = sizeof(T)
    Nitem_ref = Int(c.Nitem)
    Nitem = clamp(max(1, Nitem_ref * ref_size ÷ sz), 1, min(cld(32, sz), 16))
    return (; Nitem,
              Nthreads  = Int(c.Nthreads),
              workgroup = Int(c.workgroup),
              blocks    = Int(c.blocks))
end
