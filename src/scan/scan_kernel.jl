@generated function prefix_apply(op::OP, prefix::P, values::NTuple{N,T})::NTuple{N,T} where {OP,P,T,N}
    quote
        Base.Cartesian.@nexprs $N i -> v_i = @inbounds values[i]
        Base.Cartesian.@nexprs $N i -> r_i = op(prefix, v_i)
        Base.Cartesian.@ntuple $N i -> r_i
    end
end

@kernel inbounds = true unsafe_indices = true function scan_kernel!(
    f, op,
    dst::AbstractVector{S},
    src::AbstractVector{T},
    ::Val{Nitem},
    partial1::AbstractVector{H},
    partial2::AbstractVector{H},
    flag::AbstractVector{FlagType},
    targetflag::FlagType,
    ::Type{H}
) where {Nitem,T,H,S,FlagType<:Integer}
    N = length(src)
    workgroup = Int(@groupsize()[1])
    lid = Int(@index(Local))
    gid = Int(@index(Group))
    I = (gid - 1) * workgroup + lid

    idx_base = (I - 1) * Nitem

    nwarps = workgroup ÷ warpsz
    warp_id = cld(lid, warpsz)
    lane = (lid - 1) % warpsz + 1

    shared = @localmem H warpsz

    if idx_base + Nitem <= N
        values = f.(vload(src, I, Val(Nitem)))
        values = tree_scan(op, values)
    else
        values = ntuple(Val(Nitem)) do i
            f(src[N]) # dummy value
        end
    end

    val = values[end]
    @warpreduce(val, lane, op)
    stored_val = val
    if lane == warpsz
        shared[warp_id] = val
    end

    @synchronize

    last_idx = Nitem * workgroup * gid
    if warp_id == nwarps
        val_acc = shared[lane]
        @warpreduce(val_acc, lane, op)
        shared[lane] = val_acc
        if lane == nwarps && last_idx <= N
            partial1[gid] = val_acc
            @access flag[gid] = targetflag
            partial2[gid] = val_acc
        end
    end

    prefix = val
    if gid >= 2 && warp_id == nwarps
        lookback = 0
        contains_prefix = false
        while lookback + 1 < gid && !@shfl(Idx, contains_prefix, 1, warpsz)
            idx_lookback = max(gid - lookback - lane, 1)
            @access flg = flag[idx_lookback]
            has_aggregate = (targetflag <= flg <= targetflag + FlagType(1))
            if @vote(All, has_aggregate)
                has_prefix = (flg == targetflag + FlagType(1))
                if has_prefix
                    val = partial2[idx_lookback]
                else
                    val = partial1[idx_lookback]
                end
                offset = 1
                contains_prefix = has_prefix
                while offset < warpsz
                    shuffled = @shfl(Down, val, offset, warpsz)
                    shuffled_contains_prefix = @shfl(Down, contains_prefix, offset, warpsz)
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
        partial2[gid] = op(prefix, partial2[gid])
        @access flag[gid] = targetflag + FlagType(1)
    end

    if gid >= 2
        prefix_block = shared[32]
    end


    if warp_id >= 2
        prefix_warp = shared[warp_id-1]
    end

    prefix_lane = @shfl(Idx, stored_val, max(lane - 1, 1), warpsz)

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
        vstore!(dst, I, values)
    elseif idx_base < N
        if N > Nitem
            val = op(global_prefix, f(src[idx_base+1]))
        else
            val = f(src[idx_base+1])
        end
        dst[idx_base+1] = val
        for i in (2:Nitem)
            if idx_base + i <= N
                val = op(val, f(src[idx_base+i]))
                dst[idx_base+i] = val
            end
        end
    end
end