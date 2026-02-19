@kernel function argmax_kernel!(
    f, rel,
    dst::AbstractArray{<:Integer},
    srcs::NTuple{U,AbstractArray{T}},
    ::Val{Nitem},
    partial::AbstractArray{Tuple{H,Int}},
    flag::AbstractArray{UInt8},
) where {U,T,H,Nitem}

    N = length(srcs[1])
    workgroup = Int(@groupsize()[1])
    ndrange = @ndrange()[1]

    blocks = cld(ndrange, workgroup)
    lid = Int(@index(Local, Linear))
    gid = Int(@index(Group, Linear))

    I = (gid - 1) * workgroup + lid

    warp_id = cld(lid, warpsz)
    lane = (lid - 1) % warpsz + 1

    shared = @localmem Tuple{H,Int} 32

    i = I
    begin
        values = broadcast_apply_across(f, srcs, i, Val(Nitem))
        maxval = values[1]
        argm = (i - 1) * Nitem + 1
        for j in 2:Nitem
            val = values[j]
            if rel(val, maxval)
                maxval = val
                argm = (i - 1) * Nitem + j
            end
        end
        i += ndrange
        while i * Nitem <= N
            values = broadcast_apply_across(f, srcs, i, Val(Nitem))
            for j in 1:Nitem
                v = values[j]
                if rel(v, maxval)
                    maxval = v
                    argm = (i - 1) * Nitem + j
                end
            end
            i += ndrange
        end
        id_base = (i - 1) * Nitem + 1
        if id_base <= N
            for j in id_base:N
                v = broadcast_apply_across(f, srcs, j, Val(1))
                if rel(v, maxval)
                    maxval = v
                    argm = j
                end
            end
        end
    end
    # Warp reduction
    offset = 1
    while offset < warpsz
        shuffled = @shfl(Up, maxval, offset)
        argshuffled = @shfl(Up, argm, offset)
        if lane > offset && (rel(shuffled, maxval) || (shuffled == maxval && argshuffled < argm))
            maxval = shuffled
            argm = argshuffled
        end
        offset <<= 1
    end
    if lane == warpsz && lid <= N || lid == N
        shared[warp_id] = (maxval, argm)
    end
    @synchronize

    if warp_id == 1
        (maxval, argm) = shared[lane]
        offset = 1
        while offset < warpsz
            shuffled = @shfl(Up, maxval, offset)
            argshuffled = @shfl(Up, argm, offset)
            if lane > offset && (rel(shuffled, maxval) || (shuffled == maxval && argshuffled < argm))
                maxval = shuffled
                argm = argshuffled
            end
            offset <<= 1
        end
        if lane == min(cld(workgroup, warpsz), cld(N, warpsz))
            partial[gid] = (maxval, argm)
            @access flag[gid] = 0x01
        end
    end

    if gid == 1
        i = lid
        while i <= blocks
            (@access flag[i]) == 0x01 && break
        end
        i <= blocks && ((maxval, argm) = partial[i])
        i += workgroup

        while i <= blocks
            while true
                (@access flag[i]) == 0x01 && break
            end
            (val, argval) = partial[i]
            if rel(val, maxval) || (val == maxval && argval < argm)
                maxval = val
                argm = argval
            end
            i += workgroup
        end

        # Warp reduction
        offset = 1
        while offset < warpsz
            shuffled = @shfl(Up, maxval, offset)
            argshuffled = @shfl(Up, argm, offset)
            if lane > offset && (rel(shuffled, maxval) || (shuffled == maxval && argshuffled < argm))
                maxval = shuffled
                argm = argshuffled
            end
            offset <<= 1
        end
        if lane == warpsz && lid <= blocks || lid == blocks
            shared[warp_id] = (maxval, argm)
        end
        @synchronize

        if warp_id == 1
            (maxval, argm) = shared[lane]
            offset = 1
            while offset < warpsz
                shuffled = @shfl(Up, maxval, offset)
                argshuffled = @shfl(Up, argm, offset)
                if lane > offset && (rel(shuffled, maxval) || (shuffled == maxval && argshuffled < argm))
                    maxval = shuffled
                    argm = argshuffled
                end
                offset <<= 1
            end
            if lane == min(cld(workgroup, warpsz), cld(blocks, warpsz))
                dst[1] = argm
            end
        end
    end
end