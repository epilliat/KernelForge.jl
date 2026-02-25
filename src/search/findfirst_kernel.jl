@kernel inbounds = true unsafe_indices = true function findfirst_kernel!(dst, src, filtr::F, ::Val{Nitem}) where {F,Nitem}
    n = length(src)
    warpsz = @warpsize
    workgroup = Int(@groupsize()[1])
    ndrange = @ndrange()[1]

    lid = Int(@index(Local, Linear))
    gid = Int(@index(Group, Linear))

    I = (gid - 1) * workgroup + lid

    lane = (I - 1) % warpsz + 1

    found = false
    stop = false
    any_found = false
    best_id = n + 1
    i = I

    last_i = i + (warpsz - lane)

    while last_i * Nitem <= n
        values = vload(src, i, Val(Nitem))
        for j in 1:Nitem
            idx = (i - 1) * Nitem + j
            if !found && filtr(values[j])
                best_id = idx
                found = true
            end
        end
        first_i = i - lane + 1
        @access current_best = dst[1]
        stop = current_best < (first_i - 1) * Nitem + 1
        any_found = @vote(AnyLane, found)
        i += ndrange
        last_i += ndrange
        (any_found || stop) && break
    end
    #@print(lane, "\n")
    if !stop && !found && (i - 1) * Nitem + 1 <= n
        for j in 1:Nitem
            idx = (i - 1) * Nitem + j

            if idx <= n && filtr(src[idx])
                best_id = idx
                found = true
                break
            end
        end
    end
    # Warp-level min reduction via shuffle
    any_found = @vote(AnyLane, found)
    if !stop && any_found
        offset = 1
        while offset < warpsz
            shuffled_id = @shfl(Down, best_id, offset)
            if lane < warpsz - offset && shuffled_id < best_id
                best_id = shuffled_id
            end
            offset <<= 1
            any_found
        end
        #@print(lane, "       ", any_found, "     ", best_id, "\n")
        if lane == 1 && best_id < n + 1
            @atomic dst[1] min best_id
        end
    end
end
