# ============================================================================
# Per-architecture defaults
# ============================================================================

@inline default_workgroup(arch::AbstractArch, ::Type{Sort1D}, n, ::Type{T}) where T = 256
@inline default_workgroup(::AMDArch,          ::Type{Sort1D}, n, ::Type{T}) where T = 256

# The 8-bit radix always has 256 digit buckets.
const SORT_NBUCKETS = 256

"""
    sort_workgroup(arch, n, T, decoupled) -> Int

Workgroup size a given sort path is allowed to run at.

Only the keys-only onesweep is workgroup-DECOUPLED (`has_bucket = lid <= Nbuckets`),
and it is the ONLY path `default_workgroup(Sort1D, …)` is autotuned for — it can run
at 256, 512 or 1024. The byte-sort kernel (`npasses == 1`) and the key/value onesweep
still index `bucket = lid`, which pins them to `workgroup == Nbuckets`. Handing either
of them a tuned 512 breaks them, so they take `SORT_NBUCKETS` and never consult the
tuning. `get_allocation` must apply the SAME rule, or `tmp` gets sized for a different
block count than the kernel launches with — which is an out-of-bounds write on `flag`,
not a clean error.
"""
@inline sort_workgroup(arch, n, ::Type{T}, decoupled::Bool) where T =
    decoupled ? default_workgroup(arch, Sort1D, n, T) : SORT_NBUCKETS

# Sort1D's "Nitem" is the vector load width per warp lane.
# Sweep on RTX 1000 Ada Laptop (sm_89) found (Nitem=16, Nchunks=2) the
# all-N champion. Other architectures haven't been swept yet — same
# defaults until tuning data exists.
@inline default_nitem(::AbstractArch, ::Type{Sort1D}, n, ::Type{T}) where T = 16
@inline default_nitem(::RTX1000,      ::Type{Sort1D}, n, ::Type{T}) where T = 16
@inline default_nitem(::A40,          ::Type{Sort1D}, n, ::Type{T}) where T = 16
@inline default_nitem(::AMDArch,      ::Type{Sort1D}, n, ::Type{T}) where T = 16

# Chunks per warp. We size block_size_max = workgroup * Nchunks * Nitem so
# that shared_sorted (= block_size_max * sizeof(T) bytes) fits alongside
# shared_hist (8 KB) + shared_aux (~1 KB) in the shared-memory budget.
# For sizeof(T) ≤ 4 bytes, (Nitem=16, Nchunks=2) → 8192 elements → up to 32 KB
# shared_sorted, total ≤ 41 KB.
# For sizeof(T) = 8, halve Nchunks → 4096 elements × 8 B = 32 KB, total 41 KB.
#
# These sizes date from when shared memory was static and therefore capped at
# 48 KB. The onesweep now allocates dynamically, so the real ceiling is
# `KI.max_dynamic_localmem(dev)` (99 KB Ada, 163 KB A100) and much larger tiles
# are reachable — but only where they have been MEASURED to win. The defaults
# stay at the old shape so an untuned architecture behaves exactly as before;
# the autotune raises them per (arch, N, dtype).
@inline default_nchunks(::AbstractArch, ::Type{Sort1D}, n, ::Type{T}) where T = sizeof(T) <= 4 ? 2 : 1
@inline default_nchunks(::RTX1000,      ::Type{Sort1D}, n, ::Type{T}) where T = sizeof(T) <= 4 ? 2 : 1
@inline default_nchunks(::A40,          ::Type{Sort1D}, n, ::Type{T}) where T = sizeof(T) <= 4 ? 2 : 1
@inline default_nchunks(::AMDArch,      ::Type{Sort1D}, n, ::Type{T}) where T = sizeof(T) <= 4 ? 2 : 1

# RR ("re-read"): in phase 5a, re-load each item from `src` instead of carrying
# it in a register from phase 1b. Halves the live values and stops the register
# allocator spilling at high items/thread, at the cost of a second (L1-hot) load.
# Measured on A100: wins on 4-byte keys (−8..−19%), loses on 8-byte ones
# (+3..+10%). Default `false` = the historical behaviour; the autotune flips it
# per (arch, N, dtype) where it measured a win.
@inline default_sort_rr(::AbstractArch, ::Type{Sort1D}, n, ::Type{T}) where T = false


# Decoupled-lookback descriptor access — the SAME gate scan uses
# (`scan_desc_bypass`, src/scan/scan.jl), for the same reason.
#
# The shipped lookback spins on `@access flag[b]`, which with no scope/ordering
# is a DEVICE-scope ACQUIRE. On gfx942 that L1-invalidates on EVERY spin round,
# and the walk is per-block: measured on MI300A, the lookback was 66-68% of the
# whole sort (vs 1.7% on A100 — same code, wildly different bill).
#
# Bypass routes every descriptor access through Device-scope RELAXED atomics:
# `Device` still gives cross-block COHERENCE (bypasses a stale L1), `Relaxed`
# drops the per-round acquire. The cross-block ORDERING (partials drained before
# the flag; flag read before the partials) is carried by WORKGROUP-scope
# release/acquire fences — rocPRIM's WITHOUT_SLOW_FENCES gfx94x recipe.
# MI300A 1e8: UInt32 77.0 -> 17.5 ms, UInt64 312.4 -> 59.6 ms.
#
# ⚠️ The release fence must be executed by EVERY THREAD, BEFORE the barrier.
# Scan can fence on the publishing lane alone because that lane wrote the
# partial itself; here the 256 partial stores come from wpb DIFFERENT WAVES and
# only lid==1 publishes the flag. The AMD lowering is `s_waitcnt vmcnt(0)`, and
# vmcnt is a PER-WAVE counter — wave 1 draining its own vmem says nothing about
# waves 2..wpb. Fencing on lid==1 only left UInt64 silently WRONG (UInt32 passed
# by luck, half the blocks).
#
# Gated on CDNA3 (gfx942), NOT on AMDArch: that is the only hardware the recipe
# has been measured on. CDNA2 (gfx90a, MI250X) has a different cache/coherence
# story and would silently inherit an unvalidated memory model — widen this only
# after running the suite there.
#
# OFF on CUDA and on every untuned arch: the CUDA path stays byte-identical, so
# this cannot regress A100/RTX1000. A relaxed flag is a genuine race on CUDA's
# model — it needs the real acquire.
@inline sort_desc_bypass(::CDNA3) = true
@inline sort_desc_bypass(::CUDAArch) = false
@inline sort_desc_bypass(::AbstractArch) = false


# Keyval (`sort(...; keys=k)`) defaults: shared_sorted (T) AND shared_keys
# (KT) both occupy block_size_max * size bytes, so the budget is tighter.
# Want block_size_max * (sizeof(T) + sizeof(KT)) ≤ ~32 KB after subtracting
# shared_hist (8 KB) + shared_aux (~1 KB) from 48 KB.
# At workgroup=256, Nitem=16:
#   sizeof(T)+sizeof(KT) ≤ 8  → Nchunks=1 → tile 4096 → 32 KB
#   sizeof(T)+sizeof(KT) ≤ 16 → Nitem=8, Nchunks=1 → tile 2048 → 32 KB
@inline default_nitem_keyval(::AbstractArch, n, ::Type{T}, ::Type{KT}) where {T,KT} =
    sizeof(T) + sizeof(KT) <= 8 ? 16 : 8
@inline default_nchunks_keyval(::AbstractArch, n, ::Type{T}, ::Type{KT}) where {T,KT} = 1


# ============================================================================
# Public API
# ============================================================================

"""
    sort(src::AbstractVector; by=identity, uint_map=KernelForge.uint_map, kwargs...)
    sort(src::AbstractVector; keys=k, ...) -> (sorted_src, sorted_keys)

Stable LSD radix sort on the GPU. Sorts `src` ascending by
`uint_map(by(x))`. Returns a new array with the sorted values; `src` is
not modified.

# Keyword Arguments
- `by=identity`: Field extractor. For tuples, e.g. `by=first` to sort by
  the first element. When `keys` is provided, `by` is applied to elements
  of `keys`, not `src`.
- `uint_map=KernelForge.uint_map`: Order-preserving projection of
  `by(x)` to an unsigned integer (UInt8/16/32/64). Defaults are provided
  for built-in numeric types.
- `keys=nothing`: Optional key array of the same length as `src`. When
  given, `src` and `keys` are sorted **together** using
  `uint_map(by(keys[i]))` as the sort key for `src[i]`. The function
  returns a tuple `(sorted_src, sorted_keys)`. Both arrays are newly
  allocated.
- `tmp=nothing`: Pre-allocated `KernelBuffer` (or `nothing` to allocate
  automatically).
- `Nitem=nothing`, `Nchunks=nothing`, `workgroup=nothing`,
  `arch=nothing`: tuning knobs (auto-selected if nothing).

# Examples
```julia
x = CuArray(rand(Float64, 10_000_000))
y = KernelForge.sort(x)
@assert Array(y) == sort(Array(x))

# Sort tuples by the first element.
pairs = CuArray([(rand(Float32), rand(UInt32)) for _ in 1:1_000_000])
sorted_pairs = KernelForge.sort(pairs; by=first)

# Sort an array of values according to a parallel key array.
vals = CuArray(rand(Float32, 1_000_000))
keys = CuArray(rand(UInt32, 1_000_000))
sorted_vals, sorted_keys = KernelForge.sort(vals; keys=keys)
@assert issorted(Array(sorted_keys))
```

See also: [`sort!`](@ref) for the in-place form.
"""
function sort end

"""
    sort!(dst, src; by=identity, uint_map=KernelForge.uint_map, kwargs...) -> dst
    sort!(dst, src; keys=k[, keys_dst=kd], ...) -> (dst, keys_dst)
    sort!(src; ...) -> src                                     # in-place
    sort!(src; keys=k, ...) -> (src, keys)                      # in-place + keys

In-place form of [`sort`](@ref). The two-argument form writes sorted
output into `dst`; `dst` and `src` may alias.

The single-argument form sorts `src` in place — no `dst` is allocated
and no `src → dst` copy is issued. The implementation pings between
`src` and an internal scratch buffer (in `tmp`) so that the final pass
lands the sorted result back in `src`. When `npasses` is even (UInt16,
UInt32, UInt64, Float32, Float64) the in-place form pays **zero**
additional copies vs the algorithmic minimum; when `npasses` is odd or
the byte-sort fast path applies (UInt8/Int8/Bool), one extra
`scratch ↔ src` copy is needed.

When `keys` is given, the function additionally returns the
destination key buffer; if `keys_dst` is omitted, one is allocated
automatically (or aliased to `keys` for the in-place form).

See [`sort`](@ref) for keyword arguments.
"""
function sort! end

"""
    get_allocation(::Type{Sort1D}, src; Nitem=nothing, Nchunks=nothing,
                   workgroup=nothing, arch=nothing, by=identity,
                   uint_map=KernelForge.uint_map, keys=nothing)

Allocate a `KernelBuffer` for `sort!`. Holds the global byte-histogram,
the two `partial` buffers used by decoupled lookback, the per-block
flag, and a same-size scratch ping-pong buffer.

When `keys` is given, an additional `scratch_keys` buffer of `eltype(keys)`
is included for the key-value sort path.
"""
function get_allocation(
    ::Type{Sort1D},
    src::AT;
    Nitem=nothing,
    Nchunks=nothing,
    workgroup=nothing,
    arch=nothing,
    by::F=identity,
    uint_map::M=uint_map,
    keys=nothing,
) where {AT<:AbstractArray,F,M}
    T = eltype(AT)
    n = length(src)
    arch = something(arch, detect_arch(src))

    if keys === nothing
        Nitem = something(Nitem, default_nitem(arch, Sort1D, n, T))
        Nchunks = something(Nchunks, default_nchunks(arch, Sort1D, n, T))
        K = Base.promote_op(uint_map ∘ by, T)
        K <: Unsigned || error("`uint_map ∘ by` must return a UInt8/16/32/64; got $K")
    else
        KT = eltype(keys)
        Nitem = something(Nitem, default_nitem_keyval(arch, n, T, KT))
        Nchunks = something(Nchunks, default_nchunks_keyval(arch, n, T, KT))
        K = Base.promote_op(uint_map ∘ by, KT)
        K <: Unsigned || error("`uint_map ∘ by` must return a UInt8/16/32/64; got $K")
    end
    npasses = sizeof(K)

    # Same decoupled/coupled rule as the launch sites — see `sort_workgroup`. If
    # these two disagree, `tmp` is sized for the wrong block count and the kernel
    # writes past the end of `flag`.
    workgroup = something(workgroup,
                          sort_workgroup(arch, n, T, keys === nothing && npasses > 1))

    block_size_max = workgroup * Nchunks * Nitem
    nblocks = cld(n, block_size_max)

    backend = get_backend(src)
    Nbuckets = 256
    # `partial1/partial2/flag` carry an extra `npasses` dimension so the
    # whole tile gets zeroed in ONE `fill!` per buffer before the pass
    # loop, rather than 3× per pass × npasses passes. Each pass takes a
    # 2D / 1D view `[:, :, pass]` / `[:, pass]`. Memory overhead is
    # `(npasses - 1) × old_partials_size` — small compared to scratch.
    hist     = KernelAbstractions.allocate(backend, UInt32, Nbuckets, npasses)
    # 2-stage histogram partials: one (Nbuckets, npasses) slice per count block on
    # CDNA3; a dummy (nhblk=1) elsewhere so NVIDIA pays no extra memory. `_hist_nhblk`
    # MUST match what `_sort_histograms!` launches or the count kernel writes OOB.
    hist_g   = KernelAbstractions.allocate(backend, UInt32, Nbuckets, npasses, _hist_nhblk(arch, n))
    # 2-stage combine level-1 partials (KT slices on CDNA3, dummy elsewhere).
    hist_partials = KernelAbstractions.allocate(backend, UInt32, Nbuckets, npasses, _hist_combine_kt(arch))
    partial1 = KernelAbstractions.allocate(backend, UInt32, Nbuckets, nblocks, npasses)
    partial2 = KernelAbstractions.allocate(backend, UInt32, Nbuckets, nblocks, npasses)
    flag     = KernelAbstractions.allocate(backend, UInt8,  nblocks, npasses)
    scratch  = KernelAbstractions.allocate(backend, T,      n)
    if keys === nothing
        return KernelBuffer((; hist, hist_g, hist_partials, partial1, partial2, flag, scratch))
    else
        KT = eltype(keys)
        scratch_keys = KernelAbstractions.allocate(backend, KT, n)
        return KernelBuffer((; hist, hist_g, hist_partials, partial1, partial2, flag, scratch, scratch_keys))
    end
end


# ============================================================================
# Allocating front-end
# ============================================================================

function sort(src::AT;
                algorithm::Symbol = :auto,
                by::F=identity,
                uint_map::M=uint_map,
                lt=nothing, tmax=nothing, reverse::Bool=false,
                keys=nothing,
                tmp::TMP=nothing,
                Nitem=nothing, Nchunks=nothing,
                workgroup=nothing, rr=nothing, arch=nothing) where
        {AT<:AbstractArray,F,M,TMP<:Union{KernelBuffer,SampleSortWorkspace,Nothing}}
    if keys === nothing
        dst = similar(src)
        sort!(dst, src; algorithm, by, uint_map, lt, tmax, reverse,
              tmp, Nitem, Nchunks, workgroup, rr, arch)
        return dst
    else
        length(keys) == length(src) || error("`keys` must have the same length as `src`")
        dst = similar(src)
        keys_dst = similar(keys)
        sort!(dst, src; algorithm, keys, keys_dst, by, uint_map,
              lt, tmax, reverse, tmp, Nitem, Nchunks, workgroup, rr, arch)
        return (dst, keys_dst)
    end
end


# ============================================================================
# In-place entry
# ============================================================================

function sort!(dst::AbstractArray, src::AT;
                 algorithm::Symbol = :auto,
                 by::F=identity,
                 uint_map::M=uint_map,
                 lt=nothing, tmax=nothing, reverse::Bool=false,
                 keys=nothing, keys_dst=nothing,
                 tmp::TMP=nothing,
                 Nitem=nothing, Nchunks=nothing,
                 workgroup=nothing, rr=nothing, arch=nothing) where
        {AT<:AbstractArray,F,M,TMP<:Union{KernelBuffer,SampleSortWorkspace,Nothing}}
    T = eltype(AT)
    algorithm in (:auto, :radix, :sample) ||
        error("`algorithm` must be :auto, :radix, or :sample; got $algorithm")

    if keys === nothing
        # ── algorithm selection ────────────────────────────────────────
        # Radix is the fast path when:
        #   • no custom comparator / pad sentinel,
        #   • and `uint_map ∘ by` lands in `<:Unsigned`.
        # Sample sort handles everything else (arbitrary `lt`, exotic
        # bitstypes, user `tmax`, `reverse`).
        K = Base.promote_op(uint_map ∘ by, T)
        radix_eligible = (lt === nothing) && (tmax === nothing) &&
                         !reverse && (K <: Unsigned)

        if algorithm === :sample || (algorithm === :auto && !radix_eligible)
            # Sample-sort path. `tmp` here is a `SampleSortWorkspace`
            # (preallocate with `sample_sort_workspace(T, N)`); a radix
            # `KernelBuffer` is meaningless for this path, so reject it rather
            # than silently drop it and re-allocate.
            tmp isa KernelBuffer && error(
                "the sample-sort path needs a `SampleSortWorkspace` for `tmp` " *
                "(build one with `sample_sort_workspace(eltype(src), length(src))`), " *
                "not a radix `KernelBuffer`; omit `tmp` to allocate internally.")
            sample_view = sample_sort(src; lt = lt, tmax = tmax,
                                      reverse = reverse, ws = tmp)
            copyto!(dst, sample_view)
            return dst
        end

        # Radix path (either :radix explicit or :auto-eligible). Symmetric guard:
        # a `SampleSortWorkspace` is the sample path's buffer, not the radix one.
        tmp isa SampleSortWorkspace && error(
            "the radix path needs a `KernelBuffer` for `tmp` " *
            "(build one with `get_allocation(Sort1D, src)`), not a " *
            "`SampleSortWorkspace`; omit `tmp` to allocate internally.")
        K <: Unsigned || error(
            "`uint_map ∘ by` must return a UInt8/16/32/64 for the radix " *
            "path; got $K. Use `algorithm = :sample` (or :auto with `lt=…`) " *
            "to route through the general-comparator sample sort.")
        (lt === nothing && tmax === nothing && !reverse) || error(
            "`algorithm = :radix` does not support `lt`/`tmax`/`reverse`; " *
            "use `algorithm = :sample` or `:auto`.")

        n = length(src)
        backend = get_backend(src)
        arch_ = something(arch, detect_arch(src))
        Nitem_ = something(Nitem, default_nitem(arch_, Sort1D, n, T))
        Nchunks_ = something(Nchunks, default_nchunks(arch_, Sort1D, n, T))
        rr_ = something(rr, default_sort_rr(arch_, Sort1D, n, T))
        npasses = sizeof(K)
        # Only the multi-pass onesweep is workgroup-decoupled; the byte-sort
        # kernel still indexes `bucket = lid`. See `sort_workgroup`.
        workgroup_ = something(workgroup, sort_workgroup(arch_, n, T, npasses > 1))

        if npasses == 1
            _dispatch_byte_sort!(Val(Nitem_),
                                 by, uint_map, dst, src, tmp,
                                 Nitem_, workgroup_, K, backend, arch_)
        else
            _dispatch_onesweep!(Val(Nitem_), Val(Nchunks_),
                                by, uint_map, dst, src, tmp,
                                Nitem_, Nchunks_, workgroup_, rr_, npasses, K, backend, arch_)
        end
        return dst
    else
        algorithm === :sample &&
            error("`keys=...` (keyval sort) is radix-only; `algorithm = :sample` " *
                  "is not supported with `keys`.")
        tmp isa SampleSortWorkspace && error(
            "`keys=...` (keyval sort) is radix-only; `tmp` must be a " *
            "`KernelBuffer`, not a `SampleSortWorkspace`.")
        # Key-value path: `keys` provides the sort key for each position; both
        # `src` (values) and `keys` are permuted together so that
        # `keys_dst` is sorted ascending under `uint_map(by(·))` and `dst[i]`
        # corresponds to the same source position as `keys_dst[i]`.
        KT = eltype(keys)
        K = Base.promote_op(uint_map ∘ by, KT)
        K <: Unsigned || error("`uint_map ∘ by` must return a UInt8/16/32/64; got $K")

        n = length(src)
        length(keys) == n || error("`keys` must have the same length as `src`")
        # When `keys_dst` is omitted, allocate it transparently. The function
        # returns `(dst, keys_dst)` whenever `keys` is provided so the caller
        # can recover the allocated buffer.
        keys_dst_ = keys_dst === nothing ? similar(keys) : keys_dst
        length(keys_dst_) == n || error("`keys_dst` must have the same length as `src`")

        backend = get_backend(src)
        arch_ = something(arch, detect_arch(src))
        # The key/value onesweep still indexes `bucket = lid`, so it is pinned to
        # workgroup == Nbuckets and must NOT pick up the autotuned
        # `default_workgroup` (which is tuned for the decoupled keys-only kernel).
        workgroup_ = something(workgroup, sort_workgroup(arch_, n, T, false))
        Nitem_ = something(Nitem, default_nitem_keyval(arch_, n, T, KT))
        Nchunks_ = something(Nchunks, default_nchunks_keyval(arch_, n, T, KT))
        npasses = sizeof(K)

        _dispatch_keyval!(Val(Nitem_), Val(Nchunks_),
                          by, uint_map,
                          dst, src, keys_dst_, keys, tmp,
                          Nitem_, Nchunks_, workgroup_, npasses, K, backend, arch_)
        return (dst, keys_dst_)
    end
end


# ----------------------------------------------------------------------------
# In-place entry (no dst; `src` is sorted into itself).
# ----------------------------------------------------------------------------

"""
    sort!(src; ...) -> src
    sort!(src; keys=k, ...) -> (src, keys)

In-place sort. Routes through `sort!(src, src; ...)` (with `keys_dst
=== keys` for the keyval form). The implementation detects this
aliasing and skips the redundant `src → dst` copy that the two-argument
form does when `dst` and `src` are distinct buffers.
"""
function sort!(src::AT;
                 algorithm::Symbol = :auto,
                 by::F=identity,
                 uint_map::M=uint_map,
                 lt=nothing, tmax=nothing, reverse::Bool=false,
                 keys=nothing,
                 tmp::TMP=nothing,
                 Nitem=nothing, Nchunks=nothing,
                 workgroup=nothing, rr=nothing, arch=nothing) where
        {AT<:AbstractArray,F,M,TMP<:Union{KernelBuffer,SampleSortWorkspace,Nothing}}
    if keys === nothing
        sort!(src, src; algorithm, by, uint_map, lt, tmax, reverse,
              tmp, Nitem, Nchunks, workgroup, rr, arch)
        return src
    else
        # Keys also sorted in-place: keys_dst === keys.
        sort!(src, src; algorithm, keys, keys_dst=keys, by, uint_map,
              lt, tmax, reverse, tmp, Nitem, Nchunks, workgroup, rr, arch)
        return (src, keys)
    end
end


# ----------------------------------------------------------------------------
# Val-typed dispatchers
# ----------------------------------------------------------------------------

# Fast paths: pre-compiled specs, statically dispatched, no invokelatest.
# We retrieve the kernel via `get_*_kernel(Val(...))` rather than referencing
# `byte_sort_kernel_16!` / `onesweep_kernel_16_{1,2}!` by bare name. The
# typed `Val{...}` methods (installed at package load via Core.eval) embed
# the kernel as a literal value in the method body, so the @inline expands
# to a literal-returning call with no global-binding access — sidestepping
# Julia 1.12's "access to binding in a world prior to its definition world"
# warning that bare-name references would trigger here.
@inline _dispatch_byte_sort!(::Val{16},
        by, uint_map, dst, src, tmp,
        Nitem, workgroup, ::Type{K}, backend, arch) where K =
    _sort1d_impl_byte!(get_byte_sort_kernel(Val(16)),
                       by, uint_map, dst, src, tmp,
                       Nitem, workgroup, K, backend, arch)

@inline _dispatch_onesweep!(::Val{16}, ::Val{1},
        by, uint_map, dst, src, tmp,
        Nitem, Nchunks, workgroup, rr, npasses, ::Type{K}, backend, arch) where K =
    _sort1d_impl!(get_radix_kernel(Val(16), Val(1)),
                  by, uint_map, dst, src, tmp,
                  Nitem, Nchunks, workgroup, rr, npasses, K, backend, arch)

@inline _dispatch_onesweep!(::Val{16}, ::Val{2},
        by, uint_map, dst, src, tmp,
        Nitem, Nchunks, workgroup, rr, npasses, ::Type{K}, backend, arch) where K =
    _sort1d_impl!(get_radix_kernel(Val(16), Val(2)),
                  by, uint_map, dst, src, tmp,
                  Nitem, Nchunks, workgroup, rr, npasses, K, backend, arch)

# Fallbacks: exotic specs. Factory may @eval; we then invokelatest the
# impl exactly once per fresh spec. Subsequent calls for the same spec
# hit the typed method installed by `_define_*!` (statically dispatched).
function _dispatch_byte_sort!(::Val{Nitem},
        by, uint_map, dst, src, tmp,
        Nitem_int, workgroup, ::Type{K}, backend, arch) where {Nitem,K}
    ker_fn = get_byte_sort_kernel(Val(Nitem))
    Base.invokelatest(_sort1d_impl_byte!, ker_fn,
                      by, uint_map, dst, src, tmp,
                      Nitem_int, workgroup, K, backend, arch)
end

function _dispatch_onesweep!(::Val{Nitem}, ::Val{Nchunks},
        by, uint_map, dst, src, tmp,
        Nitem_int, Nchunks_int, workgroup, rr, npasses, ::Type{K}, backend, arch) where {Nitem,Nchunks,K}
    ker_fn = get_radix_kernel(Val(Nitem), Val(Nchunks))
    Base.invokelatest(_sort1d_impl!, ker_fn,
                      by, uint_map, dst, src, tmp,
                      Nitem_int, Nchunks_int, workgroup, rr, npasses, K, backend, arch)
end


# --- Keyval dispatch ---------------------------------------------------------
# Default specs pre-compiled in keyval_onesweep_kernel.jl: (16, 1) and (8, 1).
@inline _dispatch_keyval!(::Val{16}, ::Val{1},
        by, uint_map, dst, src, dst_keys, src_keys, tmp,
        Nitem, Nchunks, workgroup, npasses, ::Type{K}, backend, arch) where K =
    _sort1d_keyval_impl!(get_keyval_kernel(Val(16), Val(1)),
                         by, uint_map, dst, src, dst_keys, src_keys, tmp,
                         Nitem, Nchunks, workgroup, npasses, K, backend, arch)

@inline _dispatch_keyval!(::Val{8}, ::Val{1},
        by, uint_map, dst, src, dst_keys, src_keys, tmp,
        Nitem, Nchunks, workgroup, npasses, ::Type{K}, backend, arch) where K =
    _sort1d_keyval_impl!(get_keyval_kernel(Val(8), Val(1)),
                         by, uint_map, dst, src, dst_keys, src_keys, tmp,
                         Nitem, Nchunks, workgroup, npasses, K, backend, arch)

function _dispatch_keyval!(::Val{Nitem}, ::Val{Nchunks},
        by, uint_map, dst, src, dst_keys, src_keys, tmp,
        Nitem_int, Nchunks_int, workgroup, npasses, ::Type{K}, backend, arch) where {Nitem,Nchunks,K}
    ker_fn = get_keyval_kernel(Val(Nitem), Val(Nchunks))
    Base.invokelatest(_sort1d_keyval_impl!, ker_fn,
                      by, uint_map, dst, src, dst_keys, src_keys, tmp,
                      Nitem_int, Nchunks_int, workgroup, npasses, K, backend, arch)
end


# ============================================================================
# Core implementation
# ============================================================================

# Stage 1+2 are shared across the byte-sort fast path and the onesweep
# multi-pass path. Mutates `hist` in place.
# Dedicated histogram launch config — the SINGLE SOURCE OF TRUTH for the number
# of partial slices `nhblk`. `get_allocation` sizes `hist_g` from this SAME nhblk,
# so if the two disagree the count kernel writes past the end of `hist_g`. The
# histogram uses FATTER blocks than the onesweep (bigger wg × Nitem) so `nhblk`
# stays small — the 2-stage combine cost scales with it (see bucket_histogram_kernel.jl).
# Gate: the 2-stage histogram is a WIN on CDNA3 (MI300A 1e8 ~halves the sort) but
# a REGRESSION on NVIDIA (A100 UInt64/Float64 1e8 +21%) — there the single-stage
# global-atomic flush is not a bottleneck (fast dedicated atomic units) and the
# 2-stage only adds a combine kernel + a fatter count kernel. So NVIDIA and every
# untuned arch keep the single-stage path, byte-identical to before.
@inline sort_hist_2stage(::CDNA3) = true
@inline sort_hist_2stage(::AbstractArch) = false

# Dedicated 2-stage histogram launch config — the SINGLE SOURCE OF TRUTH for the
# number of partial slices `nhblk`. `get_allocation` sizes `hist_g` from this SAME
# nhblk, so if the two disagree the count kernel writes past the end of `hist_g`.
# Fatter blocks than the onesweep so `nhblk` (and thus the combine cost) stay small.
@inline function _hist_launch_config(n::Int)
    hwg = 1024
    hni = 32
    nhblk = cld(cld(n, hni), hwg)
    return hwg, hni, nhblk
end
# Slice count actually allocated for `hist_g` — 1 (a dummy) unless this arch runs
# the 2-stage path, so NVIDIA pays no extra memory.
@inline _hist_nhblk(arch, n::Int) = sort_hist_2stage(arch) ? _hist_launch_config(n)[3] : 1

# The 2-stage combine is done in TWO parallel levels (see bucket_histogram_kernel.jl):
# level 1 = HIST_COMBINE_KT blocks each reduce a k-slab of `nhblk` into `hist_partials`,
# level 2 = the plain combine over HIST_COMBINE_KT (≪ nhblk) slices. A fixed KT (not
# scaled with n) keeps the buffer size and the launch n-independent; empty slabs at
# small n write 0. `hist_partials` slice count — KT on the 2-stage path, else a dummy.
const HIST_COMBINE_KT = 128
@inline _hist_combine_kt(arch) = sort_hist_2stage(arch) ? HIST_COMBINE_KT : 1

@inline function _sort_histograms!(
        by, uint_map, src, hist, hist_g, hist_partials,
        Nitem::Int, Nbuckets::Int, warpsz::Int, wpb::Int, npasses::Int,
        workgroup::Int, backend, arch)
    n = length(src)
    if sort_hist_2stage(arch)
        # Stage 1: per-block partial histograms → hist_g[:, :, block] (plain stores,
        # every entry written → no fill! of hist_g). Stage 2 = a TWO-LEVEL parallel
        # reduction of the nhblk slices (level 1 → hist_partials over KT k-slabs,
        # level 2 → hist over KT slices), both fully written → no fill!. The old
        # single-level combine launched only 4 workgroups (333 µs); this is ~30 µs.
        # See bucket_histogram_kernel.jl.
        hwg, hni, nhblk = _hist_launch_config(n)
        bucket_histogram_count_kernel!(backend, hwg, nhblk * hwg)(
            hist_g, src, by, uint_map,
            Val(hni), Val(Nbuckets), Val(warpsz), Val(npasses),
        )
        # ALTERNATIVE considered: a SINGLE-kernel "atomic-finalize" combine (KT
        # blocks each reduce a k-slab, then `@atomic hist[b,p] += slab`) needs no
        # `hist_partials` buffer and adds no named kernel — but it reintroduces a
        # `fill!(hist,0)` and cross-block atomics, and measured ~57 µs vs ~30 µs for
        # this two-level form at 1e8 UInt32 (both far below the old 333 µs). The
        # launch count is identical either way (partial+combine ≡ fill+atomic), so
        # the faster, atomic-free two-level path wins. See xp/sort/hist_combine_par.jl.
        KT = HIST_COMBINE_KT
        bucket_histogram_combine_partial_kernel!(backend, 256, KT * 256)(
            hist_partials, hist_g, Val(Nbuckets), Val(npasses), Val(nhblk), Val(KT),
        )
        bucket_histogram_combine_kernel!(backend, 256, cld(Nbuckets * npasses, 256) * 256)(
            hist, hist_partials, Val(Nbuckets), Val(npasses), Val(KT),
        )
    else
        # Single-stage (original): one block-wide histogram, atomic-flushed to hist.
        fill!(hist, UInt32(0))
        bucket_histogram_kernel!(backend, workgroup,
                                  cld(cld(n, Nitem), wpb * warpsz) * workgroup)(
            hist, src, by, uint_map,
            Val(Nitem), Val(Nbuckets), Val(warpsz), Val(npasses),
        )
    end
    scan_histogram_kernel!(backend, Nbuckets, Nbuckets * npasses)(hist, Val(Nbuckets))
    return nothing
end


# --- Byte-sort path (npasses == 1) -------------------------------------------

# Nothing-dispatch: allocate buffer then forward.
function _sort1d_impl_byte!(
    ker::KER, by::F, uint_map::M,
    dst::DS, src::AT,
    ::Nothing,
    Nitem::Int, workgroup::Int, ::Type{K}, backend, arch,
) where {KER,F,M,DS<:AbstractArray,AT<:AbstractArray,K}
    tmp = get_allocation(Sort1D, src;
                         Nitem, Nchunks=1, workgroup, arch, by, uint_map)
    _sort1d_impl_byte!(ker, by, uint_map, dst, src, tmp,
                       Nitem, workgroup, K, backend, arch)
end

function _sort1d_impl_byte!(
    ker::KER, by::F, uint_map::M,
    dst::DS, src::AT,
    tmp::KernelBuffer,
    Nitem::Int, workgroup::Int, ::Type{K}, backend, arch,
) where {KER,F,M,DS<:AbstractArray,AT<:AbstractArray,K}
    n = length(src)
    Nbuckets = 256
    warpsz = get_warpsize(arch)
    wpb = workgroup ÷ warpsz

    # This kernel indexes `bucket = lid`: unlike the keys-only onesweep it is NOT
    # workgroup-decoupled, so at workgroup != Nbuckets the per-bucket scan is simply
    # wrong. Refuse rather than corrupt.
    workgroup == Nbuckets ||
        throw(ArgumentError("the byte-sort path requires workgroup == $Nbuckets " *
                            "(got $workgroup); only the keys-only radix onesweep " *
                            "supports a larger workgroup"))

    hist = tmp.arrays.hist
    hist_g = tmp.arrays.hist_g
    hist_partials = tmp.arrays.hist_partials

    _sort_histograms!(by, uint_map, src, hist, hist_g, hist_partials,
                      Nitem, Nbuckets, warpsz, wpb, 1, workgroup, backend, arch)

    # Single-byte path: skip onesweep entirely. `hist[:, 1]` (after scan)
    # is the global exclusive prefix; the kernel mutates it into running
    # offsets via per-block atomic-add.
    byte_block_size = workgroup * Nitem
    byte_nblocks = cld(n, byte_block_size)
    global_counter = @view hist[:, 1]
    inst = ker(backend, workgroup, byte_nblocks * workgroup)
    if dst === src
        # byte_sort_kernel writes to dst at positions determined by byte
        # values (not src indices), so dst === src has a write-after-read
        # race across blocks. Use scratch as intermediate, then copy back.
        scratch = tmp.arrays.scratch
        inst(scratch, src, by, uint_map, global_counter, Val(Nbuckets), Val(warpsz))
        copyto!(src, scratch)
    else
        inst(dst, src, by, uint_map, global_counter, Val(Nbuckets), Val(warpsz))
    end
    return dst
end


# --- Onesweep path (npasses ≥ 2) ---------------------------------------------

# Nothing-dispatch: allocate buffer then forward.
function _sort1d_impl!(
    ker::KER, by::F, uint_map::M,
    dst::DS, src::AT,
    ::Nothing,
    Nitem::Int, Nchunks::Int,
    workgroup::Int, rr::Bool,
    npasses::Int, ::Type{K},
    backend, arch,
) where {KER,F,M,DS<:AbstractArray,AT<:AbstractArray,K}
    tmp = get_allocation(Sort1D, src;
                         Nitem, Nchunks, workgroup, arch, by, uint_map)
    _sort1d_impl!(ker, by, uint_map, dst, src, tmp,
                  Nitem, Nchunks, workgroup, rr, npasses, K, backend, arch)
end

function _sort1d_impl!(
    ker::KER, by::F, uint_map::M,
    dst::DS, src::AT,
    tmp::KernelBuffer,
    Nitem::Int, Nchunks::Int,
    workgroup::Int, rr::Bool,
    npasses::Int, ::Type{K},
    backend, arch,
) where {KER,F,M,DS<:AbstractArray,AT<:AbstractArray,K}
    n = length(src)
    Nbuckets = 256
    warpsz = get_warpsize(arch)
    wpb = workgroup ÷ warpsz
    block_size_max = workgroup * Nchunks * Nitem
    nblocks = cld(n, block_size_max)

    workgroup >= Nbuckets ||
        throw(ArgumentError("sort workgroup must be >= $Nbuckets (got $workgroup): " *
                            "the first $Nbuckets threads carry the per-digit buckets"))
    shmem = onesweep_shmem(eltype(src), Nbuckets, warpsz, workgroup, Nchunks, Nitem)

    hist     = tmp.arrays.hist
    partial1 = tmp.arrays.partial1
    partial2 = tmp.arrays.partial2
    flag     = tmp.arrays.flag
    scratch  = tmp.arrays.scratch
    hist_g   = tmp.arrays.hist_g
    hist_partials = tmp.arrays.hist_partials

    _sort_histograms!(by, uint_map, src, hist, hist_g, hist_partials,
                      Nitem, Nbuckets, warpsz, wpb, npasses, workgroup, backend, arch)

    # Ping-pong between dst and scratch so the final write lands in dst
    # regardless of parity. Avoid all upfront copies when possible:
    #   * dst === src + even npasses: ping (src, scratch); result back in src.
    #   * dst === src + odd  npasses: one src → scratch copy, then ping
    #     (scratch, src); result back in src.
    #   * dst !== src + even npasses: pass 1 reads src directly into scratch,
    #     then ping (dst, scratch) for passes 2..n; result in dst.
    #     This saves a full N-element src → dst memcpy (the common case for
    #     `sort` / `sort!(dst, src)` on UInt16/UInt32/UInt64/Float32/Float64).
    #   * dst !== src + odd  npasses: one src → scratch copy, ping
    #     (scratch, dst); result in dst.
    # `ker(backend, workgroup)` — NOT `ker(backend, workgroup, ndrange)`. Passing
    # the ndrange positionally bakes it in as a StaticSize type parameter, which
    # recompiles the kernel for every distinct n.
    inst = ker(backend, workgroup)
    ndrange = nblocks * workgroup
    rr_val = rr ? Val(true) : Val(false)   # keep the launch statically dispatched
    bp_val = sort_desc_bypass(arch) ? Val(true) : Val(false)

    if dst === src
        if iseven(npasses)
            a, b = src, scratch
        else
            copyto!(scratch, src)
            a, b = scratch, src
        end
    elseif iseven(npasses)
        # Pass 1 reads from src; the loop reroutes to (scratch, dst)
        # after pass 1 completes (see the trailing `if pass == 1` below).
        a, b = src, scratch
    else
        copyto!(scratch, src)
        a, b = scratch, dst
    end

    # Only `flag` needs zeroing. `partial1`/`partial2` do NOT: phase 3a writes every
    # (bucket, gid) entry of both, and only then publishes flag[gid], and the lookback
    # reads partial{1,2}[bucket, b] ONLY after it has seen flag[b] != 0 -- so no entry
    # is ever read before it is written, whatever stale contents a reused `tmp` holds.
    # Zeroing them cost two launches AND, at N=1e8, two 12.5 MB memsets of pure waste.
    fill!(flag, UInt8(0))

    for pass in 1:npasses
        p1_view = @view partial1[:, :, pass]
        p2_view = @view partial2[:, :, pass]
        fl_view = @view flag[:, pass]
        excl_col = @view hist[:, pass]
        KernelIntrinsics.launch!(
            inst, b, a, by, uint_map, Int32(8 * (pass - 1)),
            excl_col, p1_view, p2_view, fl_view,
            Val(Nbuckets), Val(warpsz), Val(wpb), rr_val, bp_val;
            ndrange, shmem)
        a, b = b, a
        # No-copy `dst !== src` + even npasses path: pass 1 reads src
        # directly into scratch, then ping-pong (scratch, dst) so the
        # final pass writes to dst. The standard swap above would
        # otherwise leave `b = src` and clobber the user's input.
        if pass == 1 && dst !== src && iseven(npasses)
            a, b = scratch, dst
        end
    end
    return dst
end


# --- Keyval onesweep path (any npasses ≥ 1) ----------------------------------

# Nothing-dispatch: allocate buffer (with scratch_keys) then forward.
function _sort1d_keyval_impl!(
    ker::KER, by::F, uint_map::M,
    dst::DS, src::AT, dst_keys::DKS, src_keys::KAT,
    ::Nothing,
    Nitem::Int, Nchunks::Int,
    workgroup::Int,
    npasses::Int, ::Type{K},
    backend, arch,
) where {KER,F,M,DS<:AbstractArray,AT<:AbstractArray,DKS<:AbstractArray,KAT<:AbstractArray,K}
    tmp = get_allocation(Sort1D, src;
                         Nitem, Nchunks, workgroup, arch, by, uint_map,
                         keys=src_keys)
    _sort1d_keyval_impl!(ker, by, uint_map,
                         dst, src, dst_keys, src_keys, tmp,
                         Nitem, Nchunks, workgroup, npasses, K, backend, arch)
end

function _sort1d_keyval_impl!(
    ker::KER, by::F, uint_map::M,
    dst::DS, src::AT, dst_keys::DKS, src_keys::KAT,
    tmp::KernelBuffer,
    Nitem::Int, Nchunks::Int,
    workgroup::Int,
    npasses::Int, ::Type{K},
    backend, arch,
) where {KER,F,M,DS<:AbstractArray,AT<:AbstractArray,DKS<:AbstractArray,KAT<:AbstractArray,K}
    n = length(src)
    Nbuckets = 256
    warpsz = get_warpsize(arch)
    wpb = workgroup ÷ warpsz

    # `bucket = lid` again — the key/value onesweep is not workgroup-decoupled.
    workgroup == Nbuckets ||
        throw(ArgumentError("the key/value sort requires workgroup == $Nbuckets " *
                            "(got $workgroup); only the keys-only radix onesweep " *
                            "supports a larger workgroup"))
    block_size_max = workgroup * Nchunks * Nitem
    nblocks = cld(n, block_size_max)

    hist         = tmp.arrays.hist
    partial1     = tmp.arrays.partial1
    partial2     = tmp.arrays.partial2
    flag         = tmp.arrays.flag
    scratch      = tmp.arrays.scratch
    scratch_keys = tmp.arrays.scratch_keys
    hist_g       = tmp.arrays.hist_g
    hist_partials = tmp.arrays.hist_partials

    # Histogram is built from the KEYS (digits come from `uint_map(by(keys))`).
    _sort_histograms!(by, uint_map, src_keys, hist, hist_g, hist_partials,
                      Nitem, Nbuckets, warpsz, wpb, npasses, workgroup, backend, arch)

    # Ping-pong between (dst, dst_keys) and (scratch, scratch_keys) so that
    # the final write lands in (dst, dst_keys) regardless of parity.
    # When dst === src and dst_keys === src_keys (in-place form), skip
    # the redundant src → src and keys → keys self-copies.
    inst = ker(backend, workgroup, nblocks * workgroup)

    in_place = (dst === src) && (dst_keys === src_keys)
    if in_place
        if iseven(npasses)
            a_v, b_v = src, scratch
            a_k, b_k = src_keys, scratch_keys
        else
            copyto!(scratch, src)
            copyto!(scratch_keys, src_keys)
            a_v, b_v = scratch, src
            a_k, b_k = scratch_keys, src_keys
        end
    elseif iseven(npasses)
        copyto!(dst, src)
        copyto!(dst_keys, src_keys)
        a_v, b_v = dst, scratch
        a_k, b_k = dst_keys, scratch_keys
    else
        copyto!(scratch, src)
        copyto!(scratch_keys, src_keys)
        a_v, b_v = scratch, dst
        a_k, b_k = scratch_keys, dst_keys
    end

    # Only `flag` needs zeroing. `partial1`/`partial2` do NOT: phase 3a writes every
    # (bucket, gid) entry of both, and only then publishes flag[gid], and the lookback
    # reads partial{1,2}[bucket, b] ONLY after it has seen flag[b] != 0 -- so no entry
    # is ever read before it is written, whatever stale contents a reused `tmp` holds.
    # Zeroing them cost two launches AND, at N=1e8, two 12.5 MB memsets of pure waste.
    fill!(flag, UInt8(0))

    for pass in 1:npasses
        p1_view = @view partial1[:, :, pass]
        p2_view = @view partial2[:, :, pass]
        fl_view = @view flag[:, pass]
        excl_col = @view hist[:, pass]
        inst(b_v, a_v, b_k, a_k, by, uint_map, Int32(8 * (pass - 1)),
             excl_col, p1_view, p2_view, fl_view,
             Val(Nbuckets), Val(warpsz), Val(wpb),
             sort_desc_bypass(arch) ? Val(true) : Val(false))
        a_v, b_v = b_v, a_v
        a_k, b_k = b_k, a_k
    end
    return dst
end
