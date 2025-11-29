function get_default_config end

"""
T -> f -> H -> op -> H -> g -> S
"""

@kernel function mapreduce1d_kernel!(
    f::F, op::O,
    dst::AbstractArray{S},
    srcs::NTuple{U,AbstractArray{T}},
    g::G,
    ::Val{Nitem},
    partial::AbstractArray{H},
    flag::AbstractArray{FlagType},
    targetflag::FlagType
) where {F<:Function,O<:Function,G<:Function,U,T,H,S,FlagType<:Integer,Nitem}

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
        #val = op(f.(vload(srcs[1], i, Val(Nitem)))...)
        val = op(broadcast_apply_across(f, srcs, i, Val(Nitem))...)
        srcs
        i += ndrange
        while i * Nitem <= N
            #val = op(val, f.(vload(srcs[1], i, Val(Nitem)))...)
            val = op(val, broadcast_apply_across(f, srcs, i, Val(Nitem))...)

            i += ndrange
        end
        id_base = (i - 1) * Nitem + 1
        if id_base <= N
            for j in id_base:N
                x = f(srcs[1][j])
                val = op(val, x)
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
            @access flag[gid] = targetflag
        end
    end
    if gid == 1
        i = lid
        while i <= blocks
            (@access flag[i]) == targetflag && break
        end
        i <= blocks && (val = partial[i])
        i += workgroup

        while i <= blocks
            while true
                (@access flag[i]) == targetflag && break
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