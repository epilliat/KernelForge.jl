@kernel unsafe_indices = true inbounds = true function matvec_kernel!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractArray{T},
    x,
    ::Val{chunksz}, ::Val{Nblocks},
    partial::Union{Nothing,AbstractArray{H}},
    flag::Union{Nothing,AbstractArray{FlagType}},
    targetflag::Union{Nothing,FlagType},
    ::Type{H}
) where {F<:Function,O<:Function,G<:Function,T,H,S,chunksz,Nblocks,FlagType}
    n, p = size(src)
    I = @index(Global)
    workgroup = @groupsize()[1]
    Nchunks = cld(workgroup, chunksz)

    # Shared memory for inter-warp reduction: one element per chunk
    # careful that the size is equal to sizeof(H) * chunksz * cld(workgroup,warpsz) which can be large if chunksz if large
    # This is a reason why we try to keep chunksz small in practice.
    shared = @localmem H max(chunksz * cld(workgroup, warpsz), warpsz)

    lid = @index(Local)
    gid = @index(Group)

    #chunksz is exactly the number of rows that a workgroup takes

    # which warp within the workgroup?
    wid = cld(lid, warpsz)
    lane = (lid - 1) % warpsz + 1
    # Which lane within this group?
    chlane = (lid - 1) % chunksz + 1
    # Which chunk within the group?
    chunk = cld(lid, chunksz)

    # Which set of block ?
    sid = cld(gid, Nblocks)
    # Which set of chunk for a given block?
    schid = (gid - 1) % Nblocks + 1
    first_col = (schid - 1) * Nchunks + chunk

    # Global row index
    global_row = (sid - 1) * chunksz + chlane



    # Strided reduction over columns
    col = first_col
    if global_row <= n && col <= p
        val = isnothing(x) ? f(src[global_row, col]) : f(src[global_row, col], x[col])
    else
        val = isnothing(x) ? f(src[n, col]) : f(src[n, col], x[col]) # dummy value to define val (avoid the use of zero)
    end
    col += Nchunks * Nblocks
    if global_row <= n
        while col <= p
            new_val = isnothing(x) ? f(src[global_row, col]) : f(src[global_row, col], x[col])
            val = op(val, new_val)
            col += Nchunks * Nblocks
        end
    end
    # Shuffle reduction within chunk
    offset = chunksz
    while offset < warpsz
        shuffled = @shfl(Up, val, offset, warpsz, 0xffffffff)
        if lane > offset
            val = op(val, shuffled)
        end
        offset <<= 1
    end
    if Nblocks == 1 && global_row <= n && (workgroup == warpsz || chunksz == workgroup)
        dst[global_row] = val
        Base.@goto done
    end
    if cld(lane, chunksz) == cld(warpsz, chunksz)
        idx = chunksz <= warpsz ? (wid - 1) * chunksz + chlane : (wid - 1) * warpsz + lane
        shared[idx] = val
        #gid == 1 && chlane==1 && @print("$val, $idx, $chlane, $wid, lane: $lane chunksz: $chunksz\n")
    end

    @synchronize

    #if wid % cld(warpsz, threads_per_row) == 1 
    warps_per_row = cld(workgroup, max(warpsz, chunksz))
    if wid <= cld(chunksz * warps_per_row, warpsz)
        idx = cld(lane, warps_per_row) + ((lane - 1) % warps_per_row) * chunksz + (wid - 1) * cld(warpsz, warps_per_row)
        val = shared[idx]
        offset = 1
        #gid == 1 && wid == 1 && @print("secc:  $val, $lane, $idx, $wid, idx: $idx, warps_per_row: $warps_per_row, abc: $(warps_per_row)\n")
        while offset < warps_per_row
            shuffled = @shfl(Up, val, offset, warpsz, 0xffffffff)
            if lane > offset
                val = op(val, shuffled)
            end
            offset <<= 1
        end
        if lane % warps_per_row == 0 && cld(lane, warps_per_row) <= chunksz
            idx = (sid - 1) * chunksz + (wid - 1) * cld(warpsz, warps_per_row) + cld(lane, warps_per_row)
            if idx <= n
                if Nblocks == 1
                    dst[idx] = g(val)
                else
                    partial[(schid-1)*n+idx] = val
                    @access flag[(schid-1)*n+idx] = targetflag
                end
            end
        end
    end
    Nblocks == 1 && Base.@goto done
    if gid % Nblocks == 1
        col = chunk
        if global_row <= n && col <= Nblocks
            while true
                (@access flag[(col-1)*n+global_row]) == targetflag && break
            end
            val = partial[(col-1)*n+global_row]
        end
        col += Nchunks
        if global_row <= n
            while col <= Nblocks
                while true
                    (@access flag[(col-1)*n+global_row]) == targetflag && break
                end
                @inbounds val = op(val, partial[(col-1)*n+global_row])
                col += Nchunks
            end
        end
        offset = chunksz
        #global_row == 1 && @print("$val,    $lane    $chunksz", "\n")
        #gid == 1 && global_row == 2 && @print("$val,  $(cld(lid,chunksz)),  wid: $wid, lane: $lane chunksz: $chunksz\n")
        while offset < min(chunksz * Nblocks, warpsz)
            shuffled = @shfl(Up, val, offset, warpsz, 0xffffffff)
            if lane > offset
                val = op(val, shuffled)
            end
            offset <<= 1
        end
        if chunksz * Nblocks <= warpsz
            if global_row <= n && cld(lane, chunksz) == Nblocks
                dst[global_row] = g(val)
            end
            Base.@goto done
        end
        #edge case can happen if warpsz < chunksz * Nblocks < workgroup
        idx = chunksz <= warpsz ? (wid - 1) * chunksz + chlane : (wid - 1) * warpsz + lane
        if (
            cld(lid, chunksz) <= Nblocks && cld(lane, chunksz) == cld(warpsz, chunksz)
            ||
            cld(lid, chunksz) == Nblocks #edge case
        )
            shared[idx] = val
        end
        @synchronize

        # The min is for edge case
        warps_per_row = min(cld(workgroup, max(warpsz, chunksz)), Nblocks, cld(chunksz * Nblocks, warpsz))
        if wid <= cld(chunksz * warps_per_row, warpsz)
            idx = cld(lane, warps_per_row) + ((lane - 1) % warps_per_row) * chunksz + (wid - 1) * cld(warpsz, warps_per_row)
            val = shared[idx]
            offset = 1
            idx = (sid - 1) * chunksz + (wid - 1) * cld(warpsz, warps_per_row) + cld(lane, warps_per_row)
            #idx == 7 && @print("$val $lane  $idx  $(chunksz * warps_per_row) \n")
            while offset < warps_per_row
                shuffled = @shfl(Up, val, offset, warpsz, 0xffffffff)
                #careful to edge case for reduction here: warps_per_row is not necessarily a power of two!!
                if (lane - 1) % warps_per_row + 1 > offset# || lane > warps_per_row && lane - warps_per_row > offset
                    val = op(val, shuffled)
                end
                offset <<= 1
            end

            if lane % warps_per_row == 0 && cld(lane, warps_per_row) <= chunksz
                if idx <= n
                    dst[idx] = g(val)
                end
            end
        end
    end
    Base.@label done
end
