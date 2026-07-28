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

# GEMM tunings live in their OWN table (not TUNING_TABLE) so the matvec/vecmat
# "already loaded" fast-path in `load_tunings_for!` can never skip them, and vice
# versa. Outer key = arch tag; inner value = the Dict read from `<tag>_gemm.jl`
# (a `"gemm"` → dtype → shape-regime → knobs nesting).
const GEMM_TUNING_TABLE = Ref{Dict{Symbol,Any}}(Dict{Symbol,Any}())

# ---------------------------------------------------------------------------
# Lookup memoization.
#
# `lookup_{matvec,vecmat}` and `_arch_has_{rowthread,mlp}_tuning` traverse the
# `Dict{String,Any}` payload on EVERY call. Because the payload values are
# `Any`-typed, that traversal is fully dynamic (boxing every `.n`/`.p`/`log2`)
# — measured at ~90 µs and ~44 KB allocated per call on MI300A. `matvec!`/
# `vecmat!` call these two-to-three times per invocation, so a small matvec
# whose GPU kernel is only 20-60 µs was floored at ~220 µs of pure HOST
# dispatch (the "flat ~200-370 µs whatever the shape" anomaly). The table is
# immutable after `__init__`, so the resolved answer is a pure function of the
# key — memoize it. Caches are cleared whenever `load_tunings_for!` mutates
# the table.
const _LOOKUP_MISS = :__lookup_miss__
const MATVEC_LOOKUP_CACHE = Dict{Tuple{Symbol,DataType,Int,Int},Any}()
const VECMAT_LOOKUP_CACHE = Dict{Tuple{Symbol,DataType,Int,Int},Any}()
const ARCH_ROWTHREAD_CACHE = Dict{Symbol,Bool}()
const ARCH_MLP_CACHE       = Dict{Symbol,Bool}()
const GEMM_LOOKUP_CACHE    = Dict{Tuple{Symbol,DataType,Int,Int,Int},Any}()
const GEMM_K1_LOOKUP_CACHE = Dict{Tuple{Symbol,DataType,Int,Int,Int},Any}()

@inline function _clear_lookup_caches!()
    empty!(MATVEC_LOOKUP_CACHE)
    empty!(VECMAT_LOOKUP_CACHE)
    empty!(ARCH_ROWTHREAD_CACHE)
    empty!(ARCH_MLP_CACHE)
    empty!(GEMM_LOOKUP_CACHE)
    empty!(GEMM_K1_LOOKUP_CACHE)
    return nothing
end

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

    # 1b. GEMM tunings — a separate `<tag>_gemm.jl` (a Dict with a `"gemm"` key),
    # emitted by data/tuning/gemm/autotune.jl. Loaded into GEMM_TUNING_TABLE (its own
    # table) so it is independent of the matvec/vecmat dict below.
    if !haskey(GEMM_TUNING_TABLE[], tag)
        gpath = joinpath(root, "data", "tuning", string(tag) * "_gemm.jl")
        if isfile(gpath)
            gp = try
                Base.include(@__MODULE__, gpath)
            catch e
                @warn "tuning: failed to load $(gpath)" exception = (e, catch_backtrace())
                nothing
            end
            if gp isa AbstractDict && get(gp, "schema_version", nothing) == TUNING_SCHEMA_VERSION
                GEMM_TUNING_TABLE[][tag] = gp
                empty!(GEMM_LOOKUP_CACHE)
                empty!(GEMM_K1_LOOKUP_CACHE)
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
    _clear_lookup_caches!()
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
    key = (tag, T, n, p)
    cached = get(MATVEC_LOOKUP_CACHE, key, _LOOKUP_MISS)
    cached === _LOOKUP_MISS || return cached
    val = _lookup_matvec_uncached(arch, T, n, p)
    MATVEC_LOOKUP_CACHE[key] = val
    return val
end

function _lookup_matvec_uncached(arch::AbstractArch, ::Type{T}, n::Int, p::Int) where T
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
        cell !== nothing && return _adapt_matvec_shape(_cell_nt(cell), n)
    end
    # Tier 2: same-size default. Scale Nitem by ref_size / sizeof(T) (a no-op
    # in this case but keeps the code path uniform).
    if haskey(mv, size_key)
        cell = _nearest_cell(mv[size_key], n, p)
        if cell !== nothing
            return _adapt_matvec_shape(_cell_nt_scaled(cell, T, sizeof(T)), n)
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
            return _adapt_matvec_shape(_cell_nt_scaled(cell, T, _parse_size_key(best_key)), n)
        end
    end
    return nothing
end

# Classify a runtime GEMM shape into one of the 6 tuned regimes (square, outer,
# inner, A_thin, B_thin, tall) by the most extreme dimension vs the geometric mean —
# mirrors the shapes swept by data/tuning/gemm/autotune.jl.
function _gemm_regime(M::Int, N::Int, K::Int)
    g = cbrt(float(M) * float(N) * float(K))
    lM = log(M / g); lN = log(N / g); lK = log(K / g)
    m = max(abs(lM), abs(lN), abs(lK))
    m < log(2.0) && return "square"          # all dims within 2× of the mean
    if abs(lK) == m
        return lK < 0 ? "outer" : "inner"    # K smallest → outer; largest → inner
    elseif abs(lM) == m
        return lM > 0 ? "tall" : "B_thin"    # M largest → tall; smallest → B_thin
    else
        return lN < 0 ? "A_thin" : "tall"    # N smallest → A_thin; largest ≈ tall
    end
end

"""
    lookup_gemm(arch, CT, M, N, K) -> Union{NamedTuple,Nothing}

Autotuned MMA warp/register knobs `(; WM, WN, NWM, NWN, WK, PC, PA, DB)` for the GEMM
shape regime closest to `(M,N,K)` at compute-type `CT`, or `nothing` if no
`<arch>_gemm.jl` tuning covers `CT`. Consulted by `resolve_gemm_mma_parameters` ahead
of the `default_gemm_mma_*` heuristics; a knob the caller passed explicitly still wins
(the launcher `something`-fills only the `nothing`s).
"""
function lookup_gemm(arch::AbstractArch, ::Type{CT}, M::Int, N::Int, K::Int) where {CT}
    tag = nameof(typeof(arch))
    key = (tag, CT, M, N, K)
    cached = get(GEMM_LOOKUP_CACHE, key, _LOOKUP_MISS)
    cached === _LOOKUP_MISS || return cached
    val = _lookup_gemm_uncached(arch, CT, M, N, K)
    GEMM_LOOKUP_CACHE[key] = val
    return val
end

function _lookup_gemm_uncached(arch::AbstractArch, ::Type{CT}, M, N, K) where {CT}
    tag = nameof(typeof(arch))
    haskey(GEMM_TUNING_TABLE[], tag) || return nothing
    g = get(GEMM_TUNING_TABLE[][tag], "gemm", nothing)
    g isa AbstractDict || return nothing
    tbl = get(g, string(CT), nothing)
    tbl isa AbstractDict || return nothing
    regime = _gemm_regime(Int(M), Int(N), Int(K))
    cell = get(tbl, regime, nothing)
    cell === nothing && (cell = get(tbl, "square", nothing))
    cell isa AbstractDict || return nothing
    gi(k) = Int(cell[k])
    return (; WM = gi("WM"), WN = gi("WN"), NWM = gi("NWM"), NWN = gi("NWN"),
            WK = gi("WK"), PC = gi("PC"), PA = gi("PA"), DB = Bool(cell["DB"]))
end

"""
    lookup_gemm_k1(arch, CT, M, N, K) -> Union{NamedTuple,Nothing}

Autotuned K1 (generic, CUDA-core) tile knobs `(; BM, BN, BK, TM, TN)` for the shape
regime closest to `(M,N,K)` at compute-type `CT`, or `nothing` if no `<arch>_gemm.jl`
tuning has a `gemm_k1` section covering `CT`. Consulted by `resolve_gemm_parameters`
ahead of `default_gemm_tile`; a knob the caller passed explicitly still wins. This is
the F32 (and F64/Int8 where there is no tensor path) analogue of `lookup_gemm`.
"""
function lookup_gemm_k1(arch::AbstractArch, ::Type{CT}, M::Int, N::Int, K::Int) where {CT}
    tag = nameof(typeof(arch))
    key = (tag, CT, M, N, K)
    cached = get(GEMM_K1_LOOKUP_CACHE, key, _LOOKUP_MISS)
    cached === _LOOKUP_MISS || return cached
    val = _lookup_gemm_k1_uncached(arch, CT, M, N, K)
    GEMM_K1_LOOKUP_CACHE[key] = val
    return val
end

function _lookup_gemm_k1_uncached(arch::AbstractArch, ::Type{CT}, M, N, K) where {CT}
    tag = nameof(typeof(arch))
    haskey(GEMM_TUNING_TABLE[], tag) || return nothing
    g = get(GEMM_TUNING_TABLE[][tag], "gemm_k1", nothing)
    g isa AbstractDict || return nothing
    tbl = get(g, string(CT), nothing)
    tbl isa AbstractDict || return nothing
    regime = _gemm_regime(Int(M), Int(N), Int(K))
    cell = get(tbl, regime, nothing)
    cell === nothing && (cell = get(tbl, "square", nothing))
    cell isa AbstractDict || return nothing
    gi(k) = Int(cell[k])
    return (; BM = gi("BM"), BN = gi("BN"), BK = gi("BK"), TM = gi("TM"), TN = gi("TN"))
end

# True if this arch's loaded matvec tuning contains ≥1 row-thread cell. Used by
# `_matvec_entry!` to decide whether to trust the table's kernel choices fully
# (arch re-tuned for both families) or to let the wide/square heuristic override
# stale generic-only cells (pre-rowthread tuning still on disk). Self-resolving:
# once the arch is re-tuned, its cells carry `kernel=:rowthread` and the
# heuristic goes quiet.
function _arch_has_rowthread_tuning(arch::AbstractArch)
    tag = nameof(typeof(arch))
    cached = get(ARCH_ROWTHREAD_CACHE, tag, nothing)
    cached === nothing || return cached
    val = _arch_has_rowthread_tuning_uncached(arch)
    ARCH_ROWTHREAD_CACHE[tag] = val
    return val
end

function _arch_has_rowthread_tuning_uncached(arch::AbstractArch)
    tag = nameof(typeof(arch))
    haskey(TUNING_TABLE[], tag) || return false
    payload = TUNING_TABLE[][tag]
    haskey(payload, "matvec") || return false
    mv = payload["matvec"]
    mv isa AbstractDict || return false
    for (_, cells) in pairs(mv)
        for c in cells
            _cell_kernel(c) === :rowthread && return true
        end
    end
    return false
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

# A cell records which kernel FAMILY the autotune picked for its (n,p):
#   :generic   → (Nitem, chunksz, Nblocks, workgroup)  [the shuffle-reduction kernel]
#   :rowthread → (U, ncb, workgroup)                   [the coalesced row-thread kernel]
# The `kernel` field is optional; a cell without it is a legacy generic cell.
@inline _cell_kernel(c) = hasproperty(c, :kernel) ? Symbol(c.kernel) : :generic

# Decode a tuned cell into a params NamedTuple. Always carries a `kernel` field
# so downstream dispatch (`_matvec_entry!`, `resolve_parameters`) can route.
@inline function _cell_nt(c)
    if _cell_kernel(c) === :rowthread
        return (; kernel = :rowthread, U = Int(c.U), ncb = Int(c.ncb),
                  workgroup = Int(c.workgroup))
    end
    return (; kernel = :generic, Nitem = Int(c.Nitem), chunksz = Int(c.chunksz),
              Nblocks = Int(c.Nblocks), workgroup = Int(c.workgroup))
end

# Scale Nitem so the `@localmem NTuple{Nitem,H}` aggregate stays roughly the
# same byte size when switching from the reference dtype to T. chunksz,
# Nblocks, workgroup carry over unchanged — first-cut hypothesis; see
# verification in docs/RFC. The clamp uses the same cap autotune used.
function _cell_nt_scaled(c, ::Type{T}, ref_size::Int) where T
    # Row-thread cells: U (col-unroll count) and ncb (column-split) are
    # dtype-agnostic launch knobs — carry them across sizes unchanged.
    if _cell_kernel(c) === :rowthread
        return (; kernel = :rowthread, U = Int(c.U), ncb = Int(c.ncb),
                  workgroup = Int(c.workgroup))
    end
    sz = sizeof(T)
    Nitem_ref = Int(c.Nitem)
    # Cap matches `autotune.jl:nitems_for`: `cld(32, sz)` controls LLVM
    # unroll-driven compile RAM; the extra `min(.., 16)` guards against
    # `vload` misalignment for `sizeof(T) == 1` (Int8 at Nitem=32 trips
    # ERROR_MISALIGNED_ADDRESS — see autotune.jl:nitems_for note).
    Nitem = clamp(max(1, Nitem_ref * ref_size ÷ sz), 1, min(cld(32, sz), 16))
    return (; kernel = :generic, Nitem,
              chunksz   = Int(c.chunksz),
              Nblocks   = Int(c.Nblocks),
              workgroup = Int(c.workgroup))
end

function _parse_size_key(s::AbstractString)
    startswith(s, "size_") || error("tuning: malformed size key: $s")
    return parse(Int, @view s[6:end])
end

# Batch-packing-aware `Nitem` adaptation for a tuned cell applied to a query `n`
# that differs from the cell's tuned n. `Nitem` = rows-per-thread: a cell tuned
# where n packs perfectly into Nitem (0 waste) idles ~60% of the vector lanes
# at a nearby n it does not divide (e.g. Nitem=8 at n=10 → nbatch=cld(10,8)=2,
# capacity 16 for 10 rows). Shrink Nitem (pow2) until it no longer exceeds n and
# batch-packing waste ≤ 1/8. NO-OP when Nitem already fits n well — so with the
# denser autotune n-grid (intermediate non-pow2 levels), the nearest cell is
# usually same-regime and this only trims a residual mismatch; it never touches
# tall/square shapes that sit on a matching cell. Nblocks/chunksz/workgroup are
# NOT adapted here — an earlier Nblocks p-rescale regressed single-row/mid shapes
# by starving split parallelism, and `resolve_parameters` already clamps Nblocks
# down to what p supports.
function _adapt_matvec_shape(nt, n::Int)
    # Row-thread cells have no Nitem (1 thread = 1 row); nothing to adapt.
    nt.kernel === :rowthread && return nt
    Nitem = min(Int(nt.Nitem), max(1, prevpow(2, max(1, n))))
    while Nitem > 1 && 8 * (cld(n, Nitem) * Nitem - n) > n
        Nitem >>= 1
    end
    # Alignment cap (L2a): the generic `vload` advances one column per load by
    # `n` elements, so each column's load base is `Nitem·sizeof(T)`-aligned only
    # when `Nitem | n`. A query n the tuned cell did not divide (e.g. 8∤300)
    # misaligns every column past the first → ~2× BW loss (the "vload alignment
    # cliff" — see WIDE_CORNER_N in matvec/autotune.jl). Shrink Nitem (pow2) to the
    # largest divisor of n so every column load stays aligned. No-op when Nitem|n
    # already (the common 8|n tuned case), so tall/square shapes are untouched.
    while Nitem > 1 && n % Nitem != 0
        Nitem >>= 1
    end
    return (; kernel = :generic, Nitem, chunksz = Int(nt.chunksz),
              Nblocks = Int(nt.Nblocks), workgroup = Int(nt.workgroup))
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
    key = (tag, T, n, p)
    cached = get(VECMAT_LOOKUP_CACHE, key, _LOOKUP_MISS)
    cached === _LOOKUP_MISS || return cached
    val = _lookup_vecmat_uncached(arch, T, n, p)
    VECMAT_LOOKUP_CACHE[key] = val
    return val
end

function _lookup_vecmat_uncached(arch::AbstractArch, ::Type{T}, n::Int, p::Int) where T
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

# A vecmat cell records which kernel FAMILY the autotune picked for its (n,p):
#   :generic → (Nitem, Nthreads, workgroup, blocks)  [the strided-reduction kernel]
#   :mlp     → (U, workgroup)                          [the warp-per-column MLP kernel]
# The `kernel` field is optional; a cell without it is a legacy generic cell.
@inline _vm_cell_kernel(c) = hasproperty(c, :kernel) ? Symbol(c.kernel) : :generic

@inline function _vm_cell_nt(c)
    if _vm_cell_kernel(c) === :mlp
        return (; kernel = :mlp, U = Int(c.U), workgroup = Int(c.workgroup))
    end
    return (; kernel = :generic, Nitem = Int(c.Nitem), Nthreads = Int(c.Nthreads),
              workgroup = Int(c.workgroup), blocks = Int(c.blocks))
end

# True if this arch's loaded vecmat tuning contains ≥1 mlp cell (mirrors
# `_arch_has_rowthread_tuning` for matvec — self-resolving transition gate).
function _arch_has_mlp_tuning(arch::AbstractArch)
    tag = nameof(typeof(arch))
    cached = get(ARCH_MLP_CACHE, tag, nothing)
    cached === nothing || return cached
    val = _arch_has_mlp_tuning_uncached(arch)
    ARCH_MLP_CACHE[tag] = val
    return val
end

function _arch_has_mlp_tuning_uncached(arch::AbstractArch)
    tag = nameof(typeof(arch))
    haskey(TUNING_TABLE[], tag) || return false
    payload = TUNING_TABLE[][tag]
    haskey(payload, "vecmat") || return false
    vm = payload["vecmat"]
    vm isa AbstractDict || return false
    for (_, cells) in pairs(vm)
        for c in cells
            _vm_cell_kernel(c) === :mlp && return true
        end
    end
    return false
end

# First-cut cross-size adaptation: scale Nitem by ref_size/sizeof(T); leave
# Nthreads/workgroup/blocks unchanged. Same cap as matvec
# (`min(cld(32, sz), 16)`) — protects byte-strided vload alignment for
# sizeof(T)==1 and bounds LLVM unroll cost.
function _vm_cell_nt_scaled(c, ::Type{T}, ref_size::Int) where T
    # MLP cells: U (unroll depth) and workgroup are dtype-agnostic launch knobs.
    if _vm_cell_kernel(c) === :mlp
        return (; kernel = :mlp, U = Int(c.U), workgroup = Int(c.workgroup))
    end
    sz = sizeof(T)
    Nitem_ref = Int(c.Nitem)
    Nitem = clamp(max(1, Nitem_ref * ref_size ÷ sz), 1, min(cld(32, sz), 16))
    return (; kernel = :generic, Nitem,
              Nthreads  = Int(c.Nthreads),
              workgroup = Int(c.workgroup),
              blocks    = Int(c.blocks))
end
