@kernel unsafe_indices = true inbounds = true function vecmat_simple_kernel!(
    f::F, op::O, g::G,
    dst::AbstractMatrix{S},
    x::Union{Nothing,AbstractArray},
    src::AbstractArray{T},
    ::Val{Nitem},
    ::Val{Nthreads},
    ::Type{H},
    ::Val{warpsz}
) where {F<:Function,O<:Function,G<:Function,T,H,S,Nitem,Nthreads,warpsz}
    @uniform begin
        n, p = size(src)
        workgroup = Int(@groupsize()[1])
        Nblocks = Nthreads ÷ workgroup
        nwarps = workgroup ÷ warpsz
    end

    lid = Int(@index(Local, Linear))
    gid = Int(@index(Group, Linear))
    tid = Int(@index(Global, Linear))
    lane = (lid - 1) % warpsz + 1
    wid = (lid - 1) ÷ warpsz + 1

    shared = @localmem H warpsz

    for c in 1:p
        col_base = (c - 1) * n
        id_base = col_base + (tid - 1) * Nitem + 1

        if id_base + Nitem - 1 <= col_base + n
            values = vload(src, id_base, Val(Nitem), Val(false))
        else
            values = safe_vload(src, id_base, col_base + n, Val(Nitem))
        end

        if !isnothing(x)
            i_x = (tid - 1) * Nitem + 1
            if i_x + Nitem - 1 <= n
                values_x = vload(x, i_x, Val(Nitem), Val(false))
            else
                values_x = safe_vload(x, i_x, n, Val(Nitem))
            end
            values = f.(values_x, values)
            i_x += Nthreads * Nitem
        else
            values = f.(values)
        end
        if id_base + Nitem - 1 <= col_base + n
            val = tree_reduce(op, values)
        else
            val = values[1]
            i = 2
            for i in (2:Nitem)
                if id_base + i - 1 <= col_base + n
                    val = op(val, values[i])
                    i += 1
                end
            end
        end

        i = id_base + Nthreads * Nitem
        while i + Nitem - 1 <= col_base + n
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
        while i <= col_base + n
            if !isnothing(x)
                val = op(val, f(x[i_x], src[i]))
                i_x += 1
            else
                val = op(val, f(src[i]))
            end
            i += 1
        end

        # warp reduction
        offset = 1
        while offset < warpsz
            shuffled = @shfl(Up, val, offset)
            if lane > offset
                val = op(shuffled, val)
            end
            offset <<= 1
        end

        if lane == warpsz
            shared[wid] = val
        end
        @synchronize

        if wid == 1
            val_w = shared[lane]
            offset = 1
            while offset < nwarps
                shuffled = @shfl(Up, val_w, offset)
                if lane > offset
                    val_w = op(shuffled, val_w)
                end
                offset <<= 1
            end
            if lane == nwarps
                dst[gid, c] = val_w
            end
        end
    end
end