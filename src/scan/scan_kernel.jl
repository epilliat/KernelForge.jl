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

@kernel inbounds = true unsafe_indices = true function scan_kernel!(
    f, op, g,
    dst::AbstractVector{S},
    src::AbstractVector{T},
    ::Val{Nitem},
    partial1::AbstractVector{H},
    partial2::AbstractVector{H},
    flag::AbstractVector{UInt8},
    ::Val{Alignment},
    ::Val{warpsz}
) where {Nitem,T,H,S,Alignment,warpsz}
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
    @warpreduce(val, op, lane, warpsz)
    stored_val = val
    if lane == warpsz
        shared[warp_id] = val
    end

    @synchronize

    last_idx = Nitem * workgroup * gid
    if warp_id == nwarps
        val_acc = shared[lane]
        @warpreduce(val_acc, op, lane, warpsz)
        shared[lane] = val_acc
        if lane == nwarps && last_idx <= N
            partial1[gid] = val_acc
            @access flag[gid] = 0x01
            partial2[gid] = val_acc
        end
    end

    prefix = val
    if gid >= 2 && warp_id == nwarps
        lookback = 0
        contains_prefix = false
        backoff = UInt32(1)
        while lookback + 1 < gid && !@shfl(Idx, contains_prefix, 1)
            idx_lookback = max(gid - lookback - lane, 1)
            @access flg = flag[idx_lookback]
            has_aggregate = (0x01 <= flg <= 0x02)
            if @vote(All, has_aggregate, warpsz)
                has_prefix = (flg == 0x02)
                if has_prefix
                    val = partial2[idx_lookback]
                else
                    val = partial1[idx_lookback]
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
        partial2[gid] = op(prefix, partial2[gid])
        @access flag[gid] = 0x02
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
    @warpreduce(val, op, lane, warpsz)
    stored_val = val
    if lane == warpsz
        shared[warp_id] = val
    end

    @synchronize

    last_idx = Nitem * workgroup * gid
    block_agg = val  # block aggregate, broadcast to all lanes of the last warp below
    if warp_id == nwarps
        val_acc = shared[lane]
        @warpreduce(val_acc, op, lane, warpsz)
        # block total lives at lane == nwarps; broadcast it so lane 1 can publish
        # the PREFIX descriptor without re-reading global memory.
        block_agg = @shfl(Idx, val_acc, nwarps)
        shared[lane] = val_acc
        if lane == nwarps && last_idx <= N
            KI.atomic_store!(desc, gid, _scan_pack(SCAN_STATUS_PARTIAL, block_agg),
                             KI.Device, KI.Relaxed)
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