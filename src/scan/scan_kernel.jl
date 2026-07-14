@generated function prefix_apply(op::OP, prefix::P, values::NTuple{N,T})::NTuple{N,T} where {OP,P,T,N}
    quote
        Base.Cartesian.@nexprs $N i -> v_i = @inbounds values[i]
        Base.Cartesian.@nexprs $N i -> r_i = op(prefix, v_i)
        Base.Cartesian.@ntuple $N i -> r_i
    end
end

# ============================================================================
# Packed tile-descriptor encoding (decoupled-lookback scan)
# ============================================================================
# For an aggregate type H with `isbitstype(H)` and `sizeof(H) ∈ (1,2,4)` we pack
# the tile status and the aggregate value into a single UInt64 word and publish/
# read it with ONE relaxed atomic. This halves the cross-block coherence traffic
# vs. the split (flag::UInt8 + partial1::H + partial2::H) protocol — each
# lookback step is one atomic load instead of a flag-load + dependent value-load
# — which matters most on MI300X (per-XCD L2 + cross-XCD Infinity Fabric).
# Atomicity of the 64-bit word makes status+value always mutually consistent, so
# relaxed ordering with no fences is sufficient (this matches CUB / rocPRIM); a
# single value slot also suffices (PARTIAL→PREFIX overwrites it in place),
# because the atomic read can never see a torn status/value pair.
#
# A zeroed buffer (`fill!(desc, 0)`) is the valid INVALID initial state.
const SCAN_STATUS_INVALID = UInt32(0)
const SCAN_STATUS_PARTIAL = UInt32(1)
const SCAN_STATUS_PREFIX  = UInt32(2)

# Restrict to PRIMITIVE types: `reinterpret(UInt32, v)` for a composite (tuple /
# struct) aggregate compiles on the host but is not GPU-codegen-able, so 4-byte
# composites (e.g. NTuple{2,Int16}, ComplexF16) must take the split path, which
# handles any isbits H. Primitive ≤4-byte types (Float32/Int32/UInt32/Float16/
# Int16/Int8/…) keep the packed fast path.
@inline scan_packable(::Type{H}) where {H} =
    isprimitivetype(H) && (sizeof(H) == 1 || sizeof(H) == 2 || sizeof(H) == 4)

# Bits of an H value (1/2/4 bytes) carried in the low 32 bits of the descriptor.
# @generated so only the size-correct `reinterpret` is emitted (a dead branch
# with a size-mismatched bitcast would fail to compile).
@inline @generated function _scan_to_u32(v::H) where {H}
    sz = sizeof(H)
    sz == 4 ? :(reinterpret(UInt32, v)) :
    sz == 2 ? :(UInt32(reinterpret(UInt16, v))) :
    sz == 1 ? :(UInt32(reinterpret(UInt8, v))) :
    :(throw(ArgumentError("scan: H is not packable")))
end

@inline @generated function _scan_from_u32(::Type{H}, u::UInt32) where {H}
    sz = sizeof(H)
    sz == 4 ? :(reinterpret(H, u)) :
    sz == 2 ? :(reinterpret(H, u % UInt16)) :
    sz == 1 ? :(reinterpret(H, u % UInt8)) :
    :(throw(ArgumentError("scan: H is not packable")))
end

@inline _scan_pack(s::UInt32, v::H) where {H} =
    (UInt64(s) << 32) | UInt64(_scan_to_u32(v))
@inline _scan_status(d::UInt64) = UInt32(d >> 32)
@inline _scan_value(::Type{H}, d::UInt64) where {H} = _scan_from_u32(H, d % UInt32)

# Warp-inclusive scan that hands its lane to the shuffle.
#
# `@warpreduce` cannot: its `lane` is only a GUARD index and callers are allowed to make it
# segment-local (KernelForge's own `batched_radix` runs 32-wide scans on 64-wide AMD waves and
# depends on the shuffle still moving data by PHYSICAL lane). `KI.shfl_at` is the opt-in for
# callers whose lane IS the physical one — every scan kernel below computes it as
# `(lid - 1) % warpsz + 1` with the hardware warp size, so they qualify.
#
# What it buys: on AMD the backend otherwise rederives the lane with two `mbcnt` per shuffle, and
# the kernel carries two live registers for the same value. That was a 12-VGPR spill to `scratch`
# (HBM-backed private memory) and ~3.5% on the MI300A F64 scan. On CUDA `shfl_at` forwards to the
# ordinary shuffle — same code, no change either way.
@inline function warpscan_at(op::OP, val, lane::Integer, ::Val{warpsz}) where {OP,warpsz}
    offset = 1
    while offset < warpsz
        shuffled = KI.shfl_at(KI.Up, val, offset, lane, Val(warpsz))
        val = ifelse(lane > offset, op(shuffled, val), val)
        offset <<= 1
    end
    return val
end

# ── Split-path descriptor (partial1/partial2/flag) access — `Bypass` gate (see scan_desc_bypass)
# On AMD (`Bypass` true) the decoupled-lookback descriptor is read/published through Device-scope
# RELAXED atomics (a cache-BYPASSING coherent access on gfx942 — no per-round acquire, which would
# L1-invalidate and erase the F64 win). Cross-block ORDERING (partial stores drained before the
# flag; flag read before the partial) is carried by the WORKGROUP release/acquire `@fence`s, whose
# AMD lowering appends an `s_waitcnt vmcnt(0)` drain (no L2 writeback) — rocPRIM's WITHOUT_SLOW_FENCES
# gfx94x recipe. On CUDA (`Bypass` false, byte-identical to the historical path) the flag keeps its
# Device ACQUIRE/RELEASE (a relaxed flag races on CUDA's model), the partials stay plain cached
# access, and the `@fence`s vanish (they are free on a single die). The `Bypass` const folds the
# dead branch away. `@access`/`@fence` need KI ≥ the symbol-name `scope_ordering` (bare eval broke
# package precompile) + the AMD workgroup-release vmem drain.

@kernel inbounds = true unsafe_indices = true function scan_kernel!(
    f, op, g,
    dst::AbstractVector{S},
    src::AbstractVector{T},
    ::Val{Nitem},
    partial1::AbstractVector{H},
    partial2::AbstractVector{H},
    flag::AbstractVector{UInt8},
    ::Val{Alignment},
    ::Val{warpsz},
    ::Val{Bypass}
) where {Nitem,T,H,S,Alignment,warpsz,Bypass}
    @uniform begin
        N = length(src)
        workgroup = Int(@groupsize()[1])
        nwarps = workgroup ÷ warpsz
    end

    lid = Int(@index(Local))
    gid = Int(@index(Group))
    I = (gid - 1) * workgroup + lid

    idx_base = (I - 1) * Nitem

    warp_id = cld(lid, warpsz)
    lane = (lid - 1) % warpsz + 1

    shared = @localmem H warpsz

    if idx_base + Nitem <= N
        values = f.(vload(src, I, Val(Nitem), Val(true), Val(Alignment)))
        values = tree_scan(op, values)
    else
        values = ntuple(Val(Nitem)) do i
            f(src[N]) # dummy value
        end
    end

    val = values[end]
    val = warpscan_at(op, val, lane, Val(warpsz))
    stored_val = val
    if lane == warpsz
        shared[warp_id] = val
    end

    @synchronize

    last_idx = Nitem * workgroup * gid
    if warp_id == nwarps
        val_acc = shared[lane]
        val_acc = warpscan_at(op, val_acc, lane, Val(warpsz))
        shared[lane] = val_acc
        if lane == nwarps && last_idx <= N
            # Bypass (AMD): publish the aggregate. ALL descriptor accesses (partial + flag,
            # load + store) are Device-scope RELAXED atomics — `Device` gives cross-block
            # COHERENCE (a glc/dlc load/store to the coherent point, bypassing stale L1),
            # `Relaxed` drops ORDERING. The flag is relaxed too, so it does NO synchronization
            # on its own: the partial-before-flag ORDERING comes solely from the WORKGROUP-
            # scope RELEASE @fence below (its AMD lowering appends s_waitcnt vmcnt(0), no L2 writeback) — the
            # consumer pairs it with the flag→partial control dependency + coherent reads.
            # This is rocPRIM's WITHOUT_SLOW_FENCES gfx94x recipe: coherence via bypass
            # atomics, ordering via workgroup fences. A DEVICE-scope release (on the store or a
            # fence) is model-portable but IS the wall — a per-block waitcnt that caps F64 at
            # ~24%; the workgroup fence keeps it ~35%. CUDA (Bypass=false) instead uses a real
            # Device Acquire/Release on the flag (free on one die; a relaxed flag races there).
            if Bypass
                @access Device Relaxed partial1[gid] = val_acc
                @fence Workgroup Release    # drain partial stores (s_waitcnt vmcnt(0)) before the flag
                @access Device Relaxed flag[gid] = 0x01
                @access Device Relaxed partial2[gid] = val_acc
            else
                partial1[gid] = val_acc
                @access flag[gid] = 0x01          # Device Release
                partial2[gid] = val_acc
            end
        end
    end

    prefix = val
    if gid >= 2 && warp_id == nwarps
        lookback = 0
        contains_prefix = false
        backoff = UInt32(1)
        while lookback + 1 < gid && !@shfl(Idx, contains_prefix, 1)
            idx_lookback = max(gid - lookback - lane, 1)
            # Bypass (AMD): Device-Relaxed (cache-bypass) flag load — always fresh from the
            # coherent point (`Device` scope), Relaxed so it does no ordering itself. The
            # matching ACQUIRE fence sits inside the vote-success branch (once per round, not
            # per spin), pairing with the producer's per-publish RELEASE. This closes BOTH
            # handshakes with ONE fence: flag==0x01 ⇒ acquire pairs with release(partial1→0x01)
            # ⇒ read partial1; flag==0x02 ⇒ acquire pairs with release(partial2→0x02) ⇒ read
            # partial2. The fence orders EITHER subsequent partial load after the flag load, so
            # a single acquire serves both. Both fences are workgroup-scope order-only (cheap,
            # no L1 invalidate) — rocPRIM's atomic_fence_acquire_order_only gfx94x recipe.
            if Bypass
                flg = @access Device Relaxed flag[idx_lookback]
            else
                flg = @access flag[idx_lookback]      # Device Acquire
            end
            has_aggregate = (0x01 <= flg <= 0x02)
            if @vote(All, has_aggregate, warpsz)
                if Bypass
                    @fence Workgroup Acquire    # pair the producer release; order the partial read after the flag read
                end
                has_prefix = (flg == 0x02)
                if has_prefix
                    val = Bypass ? (@access Device Relaxed partial2[idx_lookback]) : partial2[idx_lookback]
                else
                    val = Bypass ? (@access Device Relaxed partial1[idx_lookback]) : partial1[idx_lookback]
                end
                offset = 1
                contains_prefix = has_prefix
                while offset < warpsz
                    shuffled = @shfl(Down, val, offset)
                    shuffled_contains_prefix = @shfl(Down, contains_prefix, offset)
                    if !contains_prefix && lane + offset <= warpsz && gid - lookback - lane - offset >= 1
                        val = op(shuffled, val)
                        contains_prefix = contains_prefix || shuffled_contains_prefix
                    end
                    offset <<= 1
                end

                if lookback == 0
                    prefix = val
                else
                    prefix = op(val, prefix)
                end
                lookback += warpsz
                backoff = UInt32(1)
            else
                # Producers not ready: back off so the spinning warp stops
                # hammering cross-block coherence while the producers commit.
                @sleep backoff
                backoff = min(backoff << 1, UInt32(64))
            end
        end
        if lane == 1
            shared[warpsz] = prefix
        end
    end


    @synchronize

    if gid >= 2 && warp_id == nwarps && lane == 1 && last_idx <= N
        if Bypass
            pv = @access Device Relaxed partial2[gid]
            @access Device Relaxed partial2[gid] = op(prefix, pv)
            @fence Workgroup Release    # drain the inclusive-prefix store before the flag
            @access Device Relaxed flag[gid] = 0x02
        else
            partial2[gid] = op(prefix, partial2[gid])
            @access flag[gid] = 0x02          # Device Release
        end
    end

    if gid >= 2
        prefix_block = shared[warpsz]
    end


    if warp_id >= 2
        prefix_warp = shared[warp_id-1]
    end

    prefix_lane = @shfl(Idx, stored_val, max(lane - 1, 1))

    if warp_id == 1 && lane == 1 && gid >= 2
        global_prefix = prefix_block
    elseif warp_id == 1 && lane >= 2 && gid == 1
        global_prefix = prefix_lane
    elseif warp_id == 1 && lane >= 2 && gid >= 2
        global_prefix = op(prefix_block, prefix_lane)
    elseif warp_id >= 2 && lane == 1 && gid == 1
        global_prefix = prefix_warp
    elseif warp_id >= 2 && lane == 1 && gid >= 2
        global_prefix = op(prefix_block, prefix_warp)
    elseif warp_id >= 2 && lane >= 2 && gid == 1
        global_prefix = op(prefix_warp, prefix_lane)
    elseif warp_id >= 2 && lane >= 2 && gid >= 2
        global_prefix = op(prefix_block, op(prefix_warp, prefix_lane))
    end



    if idx_base + Nitem <= N
        #prefix = 0
        if (gid >= 2 || lane >= 2 || warp_id >= 2)
            values = prefix_apply(op, global_prefix, values)
        end
        vstore!(dst, I, g.(values), Val(true), Val(Alignment))
    elseif idx_base < N
        if N > Nitem
            val = op(global_prefix, f(src[idx_base+1]))
        else
            val = f(src[idx_base+1])
        end
        dst[idx_base+1] = g(val)
        for i in (2:Nitem)
            if idx_base + i <= N
                val = op(val, f(src[idx_base+i]))
                dst[idx_base+i] = g(val)
            end
        end
    end
end

# ============================================================================
# Packed-descriptor variant of scan_kernel! (decoupled lookback).
# ============================================================================
# Identical to `scan_kernel!` except the per-tile status+aggregate live in a
# single `desc::AbstractVector{UInt64}` accessed with relaxed atomics (see the
# "Packed tile-descriptor encoding" note above). Used when the aggregate type H
# is packable (`scan_packable(H)`). The aggregate type is recovered in-kernel
# from the value tuple (`H = typeof(prefix)`), so it need not appear in the
# signature.
@kernel inbounds = true unsafe_indices = true function scan_kernel_packed!(
    f, op, g,
    dst::AbstractVector{S},
    src::AbstractVector{T},
    ::Val{Nitem},
    desc::AbstractVector{UInt64},
    ::Val{Alignment},
    ::Val{warpsz},
    ::Val{DescFence}
) where {Nitem,T,S,Alignment,warpsz,DescFence}
    @uniform begin
        N = length(src)
        workgroup = Int(@groupsize()[1])
        nwarps = workgroup ÷ warpsz
    end

    lid = Int(@index(Local))
    gid = Int(@index(Group))
    I = (gid - 1) * workgroup + lid

    idx_base = (I - 1) * Nitem

    warp_id = cld(lid, warpsz)
    lane = (lid - 1) % warpsz + 1

    H = Base.promote_op(f, T)  # aggregate type, compile-time (no global load)
    shared = @localmem H warpsz

    if idx_base + Nitem <= N
        values = f.(vload(src, I, Val(Nitem), Val(true), Val(Alignment)))
        values = tree_scan(op, values)
    else
        values = ntuple(Val(Nitem)) do i
            f(src[N]) # dummy value
        end
    end

    val = values[end]
    val = warpscan_at(op, val, lane, Val(warpsz))
    stored_val = val
    if lane == warpsz
        shared[warp_id] = val
    end

    @synchronize

    last_idx = Nitem * workgroup * gid
    block_agg = val  # block aggregate, broadcast to all lanes of the last warp below
    if warp_id == nwarps
        val_acc = shared[lane]
        val_acc = warpscan_at(op, val_acc, lane, Val(warpsz))
        # block total lives at lane == nwarps; broadcast it so lane 1 can publish
        # the PREFIX descriptor without re-reading global memory.
        block_agg = @shfl(Idx, val_acc, nwarps)
        shared[lane] = val_acc
        if lane == nwarps && last_idx <= N
            KI.atomic_store!(desc, gid, _scan_pack(SCAN_STATUS_PARTIAL, block_agg),
                             KI.Device, KI.Relaxed)
            # Force the just-published descriptor to be globally visible (flush to L2)
            # so successor tiles' lookback observes it immediately instead of walking
            # past a not-yet-visible aggregate and re-spinning. Matches CUB's
            # __threadfence() on publish. The PARTIAL fence is the essential one
            # (lookback sums PARTIALs; INCLUSIVE prefixes lag). Gated to CUDA archs
            # via DescFence — measured +8-13% BW on A100 F32; left off on AMD (its
            # relaxed path is at rocPRIM parity and a cross-XCD fence may regress it)
            # until validated on MI300X. Correctness-neutral either way.
            DescFence && KI.fence(KI.Device, KI.AcqRel)
        end
    end

    prefix = val
    if gid >= 2 && warp_id == nwarps
        lookback = 0
        contains_prefix = false
        # No backoff on the packed path: 32-bit aggregates have short lookback
        # chains (producers commit quickly), so sleeping only adds latency and
        # measurably regressed F32 on MI300X (1e8 277→314 µs). Backoff is applied
        # only on the split path (scan_kernel!), where waits are longer.
        while lookback + 1 < gid && !@shfl(Idx, contains_prefix, 1)
            idx_lookback = max(gid - lookback - lane, 1)
            d = KI.atomic_load(desc, idx_lookback, KI.Device, KI.Relaxed)
            flg = _scan_status(d)
            has_aggregate = (SCAN_STATUS_PARTIAL <= flg <= SCAN_STATUS_PREFIX)
            if @vote(All, has_aggregate, warpsz)
                has_prefix = (flg == SCAN_STATUS_PREFIX)
                val = _scan_value(H, d)
                offset = 1
                contains_prefix = has_prefix
                while offset < warpsz
                    shuffled = @shfl(Down, val, offset)
                    shuffled_contains_prefix = @shfl(Down, contains_prefix, offset)
                    if !contains_prefix && lane + offset <= warpsz && gid - lookback - lane - offset >= 1
                        val = op(shuffled, val)
                        contains_prefix = contains_prefix || shuffled_contains_prefix
                    end
                    offset <<= 1
                end

                if lookback == 0
                    prefix = val
                else
                    prefix = op(val, prefix)
                end
                lookback += warpsz
            end
        end
        if lane == 1
            shared[warpsz] = prefix
        end
    end


    @synchronize

    if gid >= 2 && warp_id == nwarps && lane == 1 && last_idx <= N
        KI.atomic_store!(desc, gid, _scan_pack(SCAN_STATUS_PREFIX, op(prefix, block_agg)),
                         KI.Device, KI.Relaxed)
        DescFence && KI.fence(KI.Device, KI.AcqRel)
    end

    if gid >= 2
        prefix_block = shared[warpsz]
    end


    if warp_id >= 2
        prefix_warp = shared[warp_id-1]
    end

    prefix_lane = @shfl(Idx, stored_val, max(lane - 1, 1))

    if warp_id == 1 && lane == 1 && gid >= 2
        global_prefix = prefix_block
    elseif warp_id == 1 && lane >= 2 && gid == 1
        global_prefix = prefix_lane
    elseif warp_id == 1 && lane >= 2 && gid >= 2
        global_prefix = op(prefix_block, prefix_lane)
    elseif warp_id >= 2 && lane == 1 && gid == 1
        global_prefix = prefix_warp
    elseif warp_id >= 2 && lane == 1 && gid >= 2
        global_prefix = op(prefix_block, prefix_warp)
    elseif warp_id >= 2 && lane >= 2 && gid == 1
        global_prefix = op(prefix_warp, prefix_lane)
    elseif warp_id >= 2 && lane >= 2 && gid >= 2
        global_prefix = op(prefix_block, op(prefix_warp, prefix_lane))
    end



    if idx_base + Nitem <= N
        if (gid >= 2 || lane >= 2 || warp_id >= 2)
            values = prefix_apply(op, global_prefix, values)
        end
        vstore!(dst, I, g.(values), Val(true), Val(Alignment))
    elseif idx_base < N
        if N > Nitem
            val = op(global_prefix, f(src[idx_base+1]))
        else
            val = f(src[idx_base+1])
        end
        dst[idx_base+1] = g(val)
        for i in (2:Nitem)
            if idx_base + i <= N
                val = op(val, f(src[idx_base+i]))
                dst[idx_base+i] = g(val)
            end
        end
    end
end

# ============================================================================
# Packed-128 tile descriptor — 8-byte payloads (Float64/Int64) on CDNA3.
# ============================================================================
# Same idea as the packed UInt64 descriptor above, one size class up: an 8-byte aggregate H does not
# leave room for a status in 64 bits, so {status, value} is packed into a UInt128 and published/read
# with a SINGLE coherent+atomic 16-byte access (`KI.atomic_load/atomic_store!(…, Device, Relaxed)` →
# one `global_{load,store}_dwordx4 … sc1`, see the KernelIntrinsics AMD extension).
#
# WHY THIS MATTERS: the split path (scan_kernel!) must read the flag, vote, and only THEN issue a
# DEPENDENT load of the value — two serialized cross-block round-trips per lookback step. The atomic
# 16-byte snapshot delivers status+value together, so the value is already in hand when the vote
# passes: one round-trip. That is rocPRIM's F64 recipe, and it is what closes the F64 gap on MI300A
# (measured 1.34× → 1.195× vs rocPRIM at 1e9). Atomicity also removes the need for fences — a torn
# status/value pair is impossible — exactly as on the packed-UInt64 path.
#
# H must be an 8-byte PRIMITIVE type: `reinterpret(UInt64, v)` of a composite does not GPU-codegen
# (same restriction as `scan_packable`), and the 128-bit atomic itself cannot carry an aggregate.
# Composite/wide H keep the split path. Arch is gated to CDNA3 (`sc1`); see `scan_use_packed128`.
@inline scan_packable128(::Type{H}) where {H} = isprimitivetype(H) && sizeof(H) == 8

# low 64 bits = value bits, bits 64..95 = status. Primitive→primitive reinterpret only (GPU-safe).
@inline _scan_pack128(s::UInt32, v::H) where {H} = UInt128(reinterpret(UInt64, v)) | (UInt128(s) << 64)
@inline _scan_status128(d::UInt128) = (d >> 64) % UInt32
@inline _scan_value128(::Type{H}, d::UInt128) where {H} = reinterpret(H, d % UInt64)

@kernel inbounds = true unsafe_indices = true function scan_kernel_packed128!(
    f, op, g,
    dst::AbstractVector{S},
    src::AbstractVector{T},
    ::Val{Nitem},
    desc::AbstractVector{UInt128},
    ::Val{Alignment},
    ::Val{warpsz}
) where {Nitem,T,S,Alignment,warpsz}
    @uniform begin
        N = length(src)
        workgroup = Int(@groupsize()[1])
        nwarps = workgroup ÷ warpsz
    end

    lid = Int(@index(Local))
    gid = Int(@index(Group))
    I = (gid - 1) * workgroup + lid

    idx_base = (I - 1) * Nitem

    warp_id = cld(lid, warpsz)
    lane = (lid - 1) % warpsz + 1

    H = Base.promote_op(f, T)  # aggregate type, compile-time (no global load)
    shared = @localmem H warpsz

    if idx_base + Nitem <= N
        values = f.(vload(src, I, Val(Nitem), Val(true), Val(Alignment)))
        values = tree_scan(op, values)
    else
        values = ntuple(Val(Nitem)) do i
            f(src[N]) # dummy value
        end
    end

    val = values[end]
    val = warpscan_at(op, val, lane, Val(warpsz))
    stored_val = val
    if lane == warpsz
        shared[warp_id] = val
    end

    @synchronize

    last_idx = Nitem * workgroup * gid
    block_agg = val  # block aggregate, broadcast to all lanes of the last warp below
    if warp_id == nwarps
        val_acc = shared[lane]
        val_acc = warpscan_at(op, val_acc, lane, Val(warpsz))
        block_agg = @shfl(Idx, val_acc, nwarps)
        shared[lane] = val_acc
        if lane == nwarps && last_idx <= N
            KI.atomic_store!(desc, gid, _scan_pack128(SCAN_STATUS_PARTIAL, block_agg),
                             KI.Device, KI.Relaxed)
        end
    end

    prefix = val
    if gid >= 2 && warp_id == nwarps
        lookback = 0
        contains_prefix = false
        # No backoff: measured a wash on MI300A F64 (the warps barely spin — producers commit fast
        # enough that the all-ready vote passes first try; the cost is the coherent-load latency,
        # which rocPRIM pays too), so sleeping would only add latency. Matches the packed-64 path.
        while lookback + 1 < gid && !@shfl(Idx, contains_prefix, 1)
            idx_lookback = max(gid - lookback - lane, 1)
            # ONE coherent+atomic 16-byte load: status AND value together (no dependent 2nd load).
            d = KI.atomic_load(desc, idx_lookback, KI.Device, KI.Relaxed)
            flg = _scan_status128(d)
            has_aggregate = (SCAN_STATUS_PARTIAL <= flg <= SCAN_STATUS_PREFIX)
            if @vote(All, has_aggregate, warpsz)
                has_prefix = (flg == SCAN_STATUS_PREFIX)
                val = _scan_value128(H, d)   # already in hand — the snapshot carried it
                offset = 1
                contains_prefix = has_prefix
                while offset < warpsz
                    shuffled = @shfl(Down, val, offset)
                    shuffled_contains_prefix = @shfl(Down, contains_prefix, offset)
                    if !contains_prefix && lane + offset <= warpsz && gid - lookback - lane - offset >= 1
                        val = op(shuffled, val)
                        contains_prefix = contains_prefix || shuffled_contains_prefix
                    end
                    offset <<= 1
                end

                if lookback == 0
                    prefix = val
                else
                    prefix = op(val, prefix)
                end
                lookback += warpsz
            end
        end
        if lane == 1
            shared[warpsz] = prefix
        end
    end

    @synchronize

    if gid >= 2 && warp_id == nwarps && lane == 1 && last_idx <= N
        KI.atomic_store!(desc, gid, _scan_pack128(SCAN_STATUS_PREFIX, op(prefix, block_agg)),
                         KI.Device, KI.Relaxed)
    end

    if gid >= 2
        prefix_block = shared[warpsz]
    end
    if warp_id >= 2
        prefix_warp = shared[warp_id-1]
    end
    prefix_lane = @shfl(Idx, stored_val, max(lane - 1, 1))

    if warp_id == 1 && lane == 1 && gid >= 2
        global_prefix = prefix_block
    elseif warp_id == 1 && lane >= 2 && gid == 1
        global_prefix = prefix_lane
    elseif warp_id == 1 && lane >= 2 && gid >= 2
        global_prefix = op(prefix_block, prefix_lane)
    elseif warp_id >= 2 && lane == 1 && gid == 1
        global_prefix = prefix_warp
    elseif warp_id >= 2 && lane == 1 && gid >= 2
        global_prefix = op(prefix_block, prefix_warp)
    elseif warp_id >= 2 && lane >= 2 && gid == 1
        global_prefix = op(prefix_warp, prefix_lane)
    elseif warp_id >= 2 && lane >= 2 && gid >= 2
        global_prefix = op(prefix_block, op(prefix_warp, prefix_lane))
    end

    if idx_base + Nitem <= N
        if (gid >= 2 || lane >= 2 || warp_id >= 2)
            values = prefix_apply(op, global_prefix, values)
        end
        vstore!(dst, I, g.(values), Val(true), Val(Alignment))
    elseif idx_base < N
        if N > Nitem
            val = op(global_prefix, f(src[idx_base+1]))
        else
            val = f(src[idx_base+1])
        end
        dst[idx_base+1] = g(val)
        for i in (2:Nitem)
            if idx_base + i <= N
                val = op(val, f(src[idx_base+i]))
                dst[idx_base+i] = g(val)
            end
        end
    end
end

# ============================================================================
# Transposed (WARP_TRANSPOSE-style) variant of scan_kernel! — large-H fast path.
# ============================================================================
# For a WIDE aggregate H (`sizeof(H) ≥ 8`, e.g. Float64/ComplexF32) the blocked
# `vload` used by `scan_kernel!` starves memory-level parallelism as items/thread
# grows: each thread issues Nitem DEPENDENT contiguous loads, so the global-load
# BW ceiling collapses with Nitem (measured on A100: F64 Ni16 = 48% of peak). But
# a low Nitem forces many tiles → a long decoupled-lookback chain, so the blocked
# path is trapped at a ~61% saddle for F64.
#
# CUB's DeviceScan sidesteps this with BLOCK_LOAD_WARP_TRANSPOSE: read the tile
# STRIPED (thread `lid` reads global positions `tile_base + k·workgroup + lid` for
# k = 0…Nitem-1 — coalesced at every k, MLP-preserving at ANY Nitem), stage it in
# shared memory, then read it back BLOCKED so each thread owns a CONTIGUOUS segment
# for the local `tree_scan`; symmetric transpose on the store. This keeps the copy
# ceiling ~82% at high Nitem, enabling few tiles (cheap lookback) AND high BW
# (measured A100 F64: 61% → 73%, ≈ CUB parity).
#
# Everything between the load and the store — block scan, decoupled lookback,
# descriptor publish, tail/junk confinement — is byte-for-byte the same as
# `scan_kernel!` (the split path), so this handles ANY isbits H. Two differences
# from `scan_kernel!`:
#   • WG is a `Val` (the shared tile `WG·Nitem` must be a compile-time size);
#   • no `Alignment` — element-wise striped access has no vector-alignment cliff.
# The shared tile must fit (`(WG·Nitem + warpsz)·sizeof(H) ≤ 48 KB`); the caller
# (`scan_use_transposed`/`scan_transposed_fits`) gates this and falls back to the
# blocked split kernel for tiles that would overflow shared (very wide structs).
# NOTE: pow-2 Nitem hits shared-bank conflicts on the blocked read/write (A100 F64
# Ni16 = 50% vs Ni20 = 73%); the autotune NITEM_GRID includes non-pow-2 values so
# the tuned Nitem avoids the conflicted stride (a padded index was a wash — it fixed
# pow-2 but regressed the non-pow-2 optimum, so it is NOT used).
@kernel inbounds = true unsafe_indices = true function scan_kernel_transposed!(
    f, op, g,
    dst::AbstractVector{S},
    src::AbstractVector{T},
    ::Val{Nitem},
    partial1::AbstractVector{H},
    partial2::AbstractVector{H},
    flag::AbstractVector{UInt8},
    ::Val{warpsz},
    ::Val{WG},
    ::Val{Bypass},
) where {Nitem,T,H,S,warpsz,WG,Bypass}
    @uniform begin
        N = length(src)
        workgroup = WG
        nwarps = WG ÷ warpsz
        TILE = WG * Nitem
    end

    lid = Int(@index(Local))
    gid = Int(@index(Group))

    warp_id = cld(lid, warpsz)
    lane = (lid - 1) % warpsz + 1

    tile_base = (gid - 1) * TILE   # 0-based global offset of this tile

    tile = @localmem H TILE
    shared = @localmem H warpsz

    # ---- STRIPED global load → shared (linear tile copy, coalesced at every k) ----
    # tile[p] holds global element (tile_base + p). OOB tail slots get a valid dummy
    # f(src[N]) (no identity element is assumed); junk is confined to the highest
    # tile positions = highest-lid threads and is guarded out on publish + store.
    for k in 0:Nitem-1
        p = k * workgroup + lid          # 1..TILE
        e = tile_base + p                # 1-based global index
        tile[p] = f(src[e <= N ? e : N])
    end
    @synchronize

    # ---- read blocked-owned contiguous items → local inclusive scan ----
    tbase = (lid - 1) * Nitem
    values = ntuple(Val(Nitem)) do j
        @inbounds tile[tbase+j]
    end
    values = tree_scan(op, values)

    @synchronize   # tile fully read before it is reused for the store transpose

    # ==================== block scan + decoupled lookback (== scan_kernel!) ========
    val = values[end]
    val = warpscan_at(op, val, lane, Val(warpsz))
    stored_val = val
    if lane == warpsz
        shared[warp_id] = val
    end

    @synchronize

    last_idx = Nitem * workgroup * gid
    if warp_id == nwarps
        val_acc = shared[lane]
        val_acc = warpscan_at(op, val_acc, lane, Val(warpsz))
        shared[lane] = val_acc
        if lane == nwarps && last_idx <= N
            # Bypass (AMD): publish the aggregate. ALL descriptor accesses (partial + flag,
            # load + store) are Device-scope RELAXED atomics — `Device` gives cross-block
            # COHERENCE (a glc/dlc load/store to the coherent point, bypassing stale L1),
            # `Relaxed` drops ORDERING. The flag is relaxed too, so it does NO synchronization
            # on its own: the partial-before-flag ORDERING comes solely from the WORKGROUP-
            # scope RELEASE @fence below (its AMD lowering appends s_waitcnt vmcnt(0), no L2 writeback) — the
            # consumer pairs it with the flag→partial control dependency + coherent reads.
            # This is rocPRIM's WITHOUT_SLOW_FENCES gfx94x recipe: coherence via bypass
            # atomics, ordering via workgroup fences. A DEVICE-scope release (on the store or a
            # fence) is model-portable but IS the wall — a per-block waitcnt that caps F64 at
            # ~24%; the workgroup fence keeps it ~35%. CUDA (Bypass=false) instead uses a real
            # Device Acquire/Release on the flag (free on one die; a relaxed flag races there).
            if Bypass
                @access Device Relaxed partial1[gid] = val_acc
                @fence Workgroup Release    # drain partial stores (s_waitcnt vmcnt(0)) before the flag
                @access Device Relaxed flag[gid] = 0x01
                @access Device Relaxed partial2[gid] = val_acc
            else
                partial1[gid] = val_acc
                @access flag[gid] = 0x01          # Device Release
                partial2[gid] = val_acc
            end
        end
    end

    prefix = val
    if gid >= 2 && warp_id == nwarps
        lookback = 0
        contains_prefix = false
        backoff = UInt32(1)
        while lookback + 1 < gid && !@shfl(Idx, contains_prefix, 1)
            idx_lookback = max(gid - lookback - lane, 1)
            # Bypass (AMD): Device-Relaxed (cache-bypass) flag load — always fresh from the
            # coherent point (`Device` scope), Relaxed so it does no ordering itself. The
            # matching ACQUIRE fence sits inside the vote-success branch (once per round, not
            # per spin), pairing with the producer's per-publish RELEASE. This closes BOTH
            # handshakes with ONE fence: flag==0x01 ⇒ acquire pairs with release(partial1→0x01)
            # ⇒ read partial1; flag==0x02 ⇒ acquire pairs with release(partial2→0x02) ⇒ read
            # partial2. The fence orders EITHER subsequent partial load after the flag load, so
            # a single acquire serves both. Both fences are workgroup-scope order-only (cheap,
            # no L1 invalidate) — rocPRIM's atomic_fence_acquire_order_only gfx94x recipe.
            if Bypass
                flg = @access Device Relaxed flag[idx_lookback]
            else
                flg = @access flag[idx_lookback]      # Device Acquire
            end
            has_aggregate = (0x01 <= flg <= 0x02)
            if @vote(All, has_aggregate, warpsz)
                if Bypass
                    @fence Workgroup Acquire    # pair the producer release; order the partial read after the flag read
                end
                has_prefix = (flg == 0x02)
                if has_prefix
                    val = Bypass ? (@access Device Relaxed partial2[idx_lookback]) : partial2[idx_lookback]
                else
                    val = Bypass ? (@access Device Relaxed partial1[idx_lookback]) : partial1[idx_lookback]
                end
                offset = 1
                contains_prefix = has_prefix
                while offset < warpsz
                    shuffled = @shfl(Down, val, offset)
                    shuffled_contains_prefix = @shfl(Down, contains_prefix, offset)
                    if !contains_prefix && lane + offset <= warpsz && gid - lookback - lane - offset >= 1
                        val = op(shuffled, val)
                        contains_prefix = contains_prefix || shuffled_contains_prefix
                    end
                    offset <<= 1
                end

                if lookback == 0
                    prefix = val
                else
                    prefix = op(val, prefix)
                end
                lookback += warpsz
                backoff = UInt32(1)
            else
                @sleep backoff
                backoff = min(backoff << 1, UInt32(64))
            end
        end
        if lane == 1
            shared[warpsz] = prefix
        end
    end

    @synchronize

    if gid >= 2 && warp_id == nwarps && lane == 1 && last_idx <= N
        if Bypass
            pv = @access Device Relaxed partial2[gid]
            @access Device Relaxed partial2[gid] = op(prefix, pv)
            @fence Workgroup Release    # drain the inclusive-prefix store before the flag
            @access Device Relaxed flag[gid] = 0x02
        else
            partial2[gid] = op(prefix, partial2[gid])
            @access flag[gid] = 0x02          # Device Release
        end
    end

    if gid >= 2
        prefix_block = shared[warpsz]
    end
    if warp_id >= 2
        prefix_warp = shared[warp_id-1]
    end
    prefix_lane = @shfl(Idx, stored_val, max(lane - 1, 1))

    if warp_id == 1 && lane == 1 && gid >= 2
        global_prefix = prefix_block
    elseif warp_id == 1 && lane >= 2 && gid == 1
        global_prefix = prefix_lane
    elseif warp_id == 1 && lane >= 2 && gid >= 2
        global_prefix = op(prefix_block, prefix_lane)
    elseif warp_id >= 2 && lane == 1 && gid == 1
        global_prefix = prefix_warp
    elseif warp_id >= 2 && lane == 1 && gid >= 2
        global_prefix = op(prefix_block, prefix_warp)
    elseif warp_id >= 2 && lane >= 2 && gid == 1
        global_prefix = op(prefix_warp, prefix_lane)
    elseif warp_id >= 2 && lane >= 2 && gid >= 2
        global_prefix = op(prefix_block, op(prefix_warp, prefix_lane))
    end

    if gid >= 2 || lane >= 2 || warp_id >= 2
        values = prefix_apply(op, global_prefix, values)
    end

    # ---- store transpose: blocked results → shared → STRIPED global write --------
    @synchronize
    for j in 1:Nitem
        @inbounds tile[tbase+j] = g(values[j])
    end
    @synchronize
    for k in 0:Nitem-1
        p = k * workgroup + lid
        e = tile_base + p
        if e <= N
            dst[e] = tile[p]
        end
    end
end