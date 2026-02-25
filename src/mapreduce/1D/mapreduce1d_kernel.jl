"""
T -> f -> H -> op -> H -> g -> S
"""

@kernel inbounds = true function mapreduce1d_kernel!(
    f, op, g,
    dst::AbstractArray{S},
    srcs::NTuple{U,AbstractArray{T}},
    ::Val{Nitem},
    partial::AbstractArray{H},
    flag::AbstractArray{FlagType},
    ::Val{Alignment}
) where {U,T,H,S,FlagType<:Integer,Nitem,Alignment}

    N = length(srcs[1])
    workgroup = Int(@groupsize()[1])
    ndrange = @ndrange()[1]


    blocks = cld(ndrange, workgroup)
    lid = Int(@index(Local, Linear))
    gid = Int(@index(Group, Linear))

    I = (gid - 1) * workgroup + lid

    warp_id = cld(lid, warpsz)
    lane = (lid - 1) % warpsz + 1

    shared = @localmem H 32

    i = I

    begin
        values = broadcast_apply_across(f, srcs, i, Val(Nitem), Val(true), Val(Alignment))
        val = tree_reduce(op, values)
        i += ndrange
        while i * Nitem <= N
            values = broadcast_apply_across(f, srcs, i, Val(Nitem), Val(true), Val(Alignment))
            val = op(val, tree_reduce(op, values))

            i += ndrange
        end
        id_base = (i - 1) * Nitem + 1
        if id_base <= N
            for j in id_base:N
                val = op(val, tree_reduce(op, broadcast_apply_across(f, srcs, j, Val(1))))
            end
        end
    end

    @warpreduce(val, lane, op)
    if lane == warpsz && lid <= N || lid == N
        shared[warp_id] = val
    end
    @synchronize

    if warp_id == 1#cld(workgroup, warpsz) && warp_id <= cld(N, warpsz) || warp_id == cld(N, warpsz)
        val_acc = shared[lane]
        @warpreduce(val_acc, lane, op)
        if lane == min(cld(workgroup, warpsz), cld(N, warpsz))
            partial[gid] = val_acc
            @access flag[gid] = 0x01
        end
    end
    if gid == 1
        i = lid
        while i <= blocks
            ((@access flag[i]) == 0x01) && break
        end
        i <= blocks && (val = partial[i])
        i += workgroup

        while i <= blocks
            while true
                ((@access flag[i]) == 0x01) && break
            end
            val = op(val, partial[i])
            i += workgroup
        end
        @warpreduce(val, lane, op)
        if lane == warpsz && lid <= blocks || lid == blocks
            shared[warp_id] = val
        end
        @synchronize
        if warp_id == 1#cld(workgroup, warpsz) && warp_id <= cld(blocks, warpsz) || warp_id == cld(blocks, warpsz)
            val_acc = shared[lane]
            @warpreduce(val_acc, lane, op)
            if lane == min(cld(workgroup, warpsz), cld(blocks, warpsz))
                dst[1] = g(val_acc)
            end
        end
    end
end