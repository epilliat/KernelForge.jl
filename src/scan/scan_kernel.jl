
@kernel inbounds = true unsafe_indices = true function scan_kernel!(
    f, op,
    dst::AbstractVector{Outf},
    src::AbstractVector{T},
    ::Val{Nitem},
    partial1::AbstractVector{Outf},
    partial2::AbstractVector{Outf},
    flag::AbstractVector{FlagType},
    targetflag::FlagType
) where {Nitem,T,Outf,FlagType<:Integer}
    N = length(src)
    workgroup = Int(@groupsize()[1])
    lid = Int(@index(Local))
    gid = Int(@index(Group))
    I = (gid - 1) * workgroup + lid

    idx_base = (I - 1) * Nitem

    nwarps = workgroup ÷ warpsz
    warp_id = cld(lid, warpsz)
    lane = (lid - 1) % warpsz + 1

    shared = @localmem Outf 32

    if idx_base + Nitem <= N
        values = vload(src, I, Val(Nitem))
        values = accumulate(op, values)
    else
        values = ntuple(i -> idx_base + i <= N ? (f(src[idx_base+i])) : f(src[N]), Val(Nitem))
        values = accumulate(op, values)
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
    if gid >= 2 && warp_id == nwarps# && last_idx <= N
        lookback = 0
        contains_prefix = false
        cpt = 0
        while lookback + 1 < gid && !@shfl(Idx, contains_prefix, 1, warpsz) && cpt < 100000
            cpt += 1
            idx_lookback = max(gid - lookback - lane, 1)
            @access flg = flag[idx_lookback]
            has_aggregate = (targetflag <= flg <= targetflag + FlagType(1))
            if @vote(All, has_aggregate)
                has_prefix = (flg == targetflag + FlagType(1))
                #@inbounds val = has_prefix ? partial2[idx_lookback] : partial1[idx_lookback]
                cpt == 3 #&& @print("$lookback")
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
                lookback += 32
            end
        end
        if lane == 1
            shared[32] = prefix
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
        global_prefix = op(prefix_block, prefix_warp, prefix_lane)
    end
    #prefix_group = shfl_up_sync( val_acc, 1)
    #prefix_warp = shfl_up_sync( val, 1)

    #prefix = 0
    if (gid >= 2 || lane >= 2 || warp_id >= 2)
        values = op.(global_prefix, values)
    end

    if idx_base + Nitem <= N
        vstore!(dst, I, values)
    else
        for i in (1:Nitem)
            if idx_base + i <= N
                dst[idx_base+i] = values[i]
            end
        end
    end
end