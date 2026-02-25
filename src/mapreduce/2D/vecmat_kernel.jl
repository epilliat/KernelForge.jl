@kernel unsafe_indices = true inbounds = true function vecmat_kernel!(#vertical reduction, square matrix (or horizontal-rectangular)
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    x::Union{Nothing,AbstractArray},
    src::AbstractArray{T},
    ::Val{Nitem},
    ::Val{Nthreads},
    partial::Union{Nothing,AbstractArray{H}},
    flag::Union{Nothing,AbstractArray{UInt8}},
    ::Type{H}
) where {F<:Function,O<:Function,G<:Function,T,H,S,Nitem,Nthreads}
    n, p = size(src)
    workgroup = Int(@groupsize()[1])
    Nblocks = cld(Nthreads, workgroup)
    ndrange = @ndrange()[1]

    blocks = cld(ndrange, workgroup)

    lid = Int(@index(Local, Linear))
    gid = Int(@index(Group, Linear))
    wid = cld(lid, warpsz) # warp id

    I = (gid - 1) * workgroup + lid

    # which chunk ? this corresponds to global column
    chid = cld(I, Nthreads)
    # which thread in the chunk ?
    lchid = (I - 1) % Nthreads + 1

    lane = (lid - 1) % warpsz + 1
    chlane = (lid - 1) % min(Nthreads, warpsz) + 1

    local_gid = (gid - 1) % Nblocks + 1 # block index relative to the column
    id_base = (chid - 1) * n + (lchid - 1) * Nitem + 1
    if id_base + Nitem - 1 <= n * p
        values = vload(src, id_base, Val(Nitem), Val(false))
    else
        values = ntuple(i -> src[n*p], Val(Nitem)) # dummy value when chid <= p
    end

    if !isnothing(x)
        i_x = (lchid - 1) * Nitem + 1
        values_x = vload(x, i_x, Val(Nitem), Val(false)) # No bound problem if Nthreads*Nitem <=n
        values = f.(values_x, values) # careful to the order here
        i_x += Nthreads * Nitem
    else
        values = f.(values)
    end

    val = tree_reduce(op, values)
    i = id_base + Nthreads * Nitem
    while i + Nitem - 1 <= chid * n && chid <= p
        values = vload(src, i, Val(Nitem), Val(false))
        if !isnothing(x)
            values_x = vload(x, i_x, Val(Nitem), Val(false))
            values = f.(values_x, values)
            i_x += Nthreads * Nitem
        else
            values = f.(values)
        end
        val = op(val, tree_reduce(op, values))
        i += Nthreads * Nitem
    end
    # add remaining elements
    if i <= chid * n && chid <= p
        for j in (i:chid*n)
            if !isnothing(x)
                val = op(val, f(src[j], x[i_x]))
                i_x += 1
            else
                val = op(val, f(src[j]))
            end
        end
    end

    offset = 1
    while offset < min(Nthreads, warpsz)
        shuffled = @shfl(Up, val, offset, warpsz, 0xffffffff)
        if chlane > offset
            val = op(shuffled, val)
        end
        offset <<= 1
    end
    if Nthreads <= warpsz
        if chid <= p && (chlane == Nthreads && chlane <= n || chlane == n)
            dst[chid] = g(val)
        end
        Base.@goto done
    end

    # Nthreads > warpsz. Different columns correspond to different warps so warp reduction do not lead to synchronization issues above p.

    shared = @localmem H warpsz
    if lane == warpsz # Nitem * workgroup * Nblocks <= n
        shared[wid] = val
    end
    @synchronize

    chid > p && Base.@goto done # no synchronization issues below, see **


    nwarps_per_chunk = cld(Nthreads, warpsz)
    local_wid = (wid - 1) % nwarps_per_chunk + 1
    if local_wid == 1 # amounts to wid == 1 if Nthreads > workgroup (Nblocks > 1)
        val_acc = shared[lane]
        @warpreduce(val_acc, lane, op)
        val_acc = shared[min(wid - 1 + lane, warpsz)]
        offset = 1

        while offset < nwarps_per_chunk
            shuffled = @shfl(Up, val_acc, offset, warpsz, 0xffffffff)
            if chlane > offset
                val_acc = op(shuffled, val_acc)
            end
            offset <<= 1
        end
        if Nthreads <= workgroup && lane == nwarps_per_chunk
            dst[chid] = g(val_acc)
        elseif lane == cld(workgroup, warpsz)
            idx = (chid - 1) * Nblocks + local_gid
            partial[idx] = val_acc
            @access flag[idx] = 0x01
        end
    end
    Nthreads <= workgroup && Base.@goto done
    # Nblocks > 1: Several blocks per column.
    local_gid != 1 && Base.@goto done # early return for non first blocks of each column.

    i = lid
    idx = (chid - 1) * Nblocks + i
    while i <= Nblocks
        (@access flag[idx]) == 0x01 && break
    end
    i <= Nblocks && (val = partial[idx])
    i += workgroup
    while i <= Nblocks
        idx = (chid - 1) * Nblocks + i
        while true
            (@access flag[idx]) == 0x01 && break
        end
        val = op(val, partial[idx])
        i += workgroup
    end
    @warpreduce(val, lane, op)
    if lane == warpsz && lid <= Nblocks || lid == Nblocks
        shared[wid] = val
    end

    @synchronize # ** different columns correspond to different groups so no problem for chid > p.

    if wid == 1
        val_acc = shared[lane]
        @warpreduce(val_acc, lane, op)
        if lane == min(cld(workgroup, warpsz), cld(Nblocks, warpsz))
            dst[chid] = g(val_acc)
        end
    end

    Base.@label done
end