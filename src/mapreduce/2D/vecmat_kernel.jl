@kernel unsafe_indices = true inbounds = true function vecmat_kernel!(#vertical reduction, square matrix (or horizontal-rectangular)
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    x::Union{Nothing,AbstractArray},
    src::AbstractArray{T},
    ::Val{Nitem},
    ::Val{Nthreads},
    partial::Union{Nothing,AbstractArray{H}},
    flag::Union{Nothing,AbstractArray{UInt8}},
    ::Type{H},
    ::Val{warpsz}
) where {F<:Function,O<:Function,G<:Function,T,H,S,Nitem,Nthreads,warpsz}
    @uniform begin
        n, p = size(src)
        workgroup = Int(@groupsize()[1])
        Nblocks = cld(Nthreads, workgroup)
        ndrange = @ndrange()[1]
        blocks = cld(ndrange, workgroup)
    end

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
        values = safe_vload(src, id_base, n * p, Val(Nitem)) # dummy value when chid <= p
    end

    if !isnothing(x)
        i_x = (lchid - 1) * Nitem + 1
        if i_x + Nitem - 1 <= n
            values_x = vload(x, i_x, Val(Nitem), Val(false))
        else
            values_x = safe_vload(x, i_x, n, Val(Nitem))
        end
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
        shuffled = @shfl(Up, val, offset)
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
        @warpreduce(val_acc, op, lane)
        val_acc = shared[min(wid - 1 + lane, warpsz)]
        offset = 1

        while offset < nwarps_per_chunk
            shuffled = @shfl(Up, val_acc, offset)
            if chlane > offset
                val_acc = op(shuffled, val_acc)
            end
            offset <<= 1
        end
        if Nthreads <= workgroup && lane == nwarps_per_chunk
            dst[chid] = g(val_acc)
        end
    end
    if Nthreads > workgroup
        if local_wid == 1 && lane == cld(workgroup, warpsz)
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
    @warpreduce(val, op, lane)
    if lane == warpsz && lid <= Nblocks || lid == Nblocks
        shared[wid] = val
    end

    @synchronize # ** different columns correspond to different groups so no problem for chid > p.

    if wid == 1
        val_acc = shared[lane]
        @warpreduce(val_acc, op, lane)
        if lane == min(cld(workgroup, warpsz), cld(Nblocks, warpsz))
            dst[chid] = g(val_acc)
        end
    end

    Base.@label done
end
# ============================================================================
# MLP (medium-large-n, large-p) path — "warp-per-column" reduction.
# ============================================================================
# The generic `vecmat_kernel!` reduces a column with a SERIAL strided loop (one
# vload per iteration, `val = op(val, …)`), so only ~Nitem elements are ever in
# flight — memory-level parallelism is capped and it tops out ~55–77% of A100
# peak on the middle band while cuBLAS GEMV reaches ~85%. This kernel assigns one
# WARP per column and issues `U` INDEPENDENT strided loads per step before folding
# them into the accumulator (MLP = U), then a single warp shuffle-reduction. That
# saturates bandwidth: measured A100 matches or beats cuBLAS across the band
# (e.g. 500×2e6 0.95×, 2000×5e5 0.99×, 1000×1e5 0.96×). Requires n ≥ warpsz (every
# lane seeds a valid element); best for large p (enough columns to fill the GPU)
# and n not so huge that a single warp's serial reduction dominates (tall shapes
# keep the generic multi-block kernel). See `_vecmat_use_mlp` in vecmat.jl.

# Fold `U` independent contributions (rows i, i+warpsz, …, i+warpsz·(U-1) of column
# `c`, 0-based) into `acc` with `op`. `@generated` so U, warpsz and the x-vs-nothing
# branch resolve at compile time; the U loads are issued together → MLP.
@inline @generated function _mlp_fold(f::F, op::O, acc, x, src, i::Int, c::Int,
                                      ::Val{U}, ::Val{warpsz}) where {F,O,U,warpsz}
    load(u) = x === Nothing ?
        :(f(@inbounds src[i + $((u-1)*warpsz) + 1, c + 1])) :
        :(f(@inbounds(x[i + $((u-1)*warpsz) + 1]), @inbounds(src[i + $((u-1)*warpsz) + 1, c + 1])))
    body = Expr(:block)
    for u in 1:U
        push!(body.args, :($(Symbol(:v_, u)) = $(load(u))))
    end
    ex = :acc
    for u in 1:U
        ex = :(op($ex, $(Symbol(:v_, u))))
    end
    push!(body.args, ex)
    body
end

@kernel unsafe_indices = true inbounds = true function vecmat_mlp_kernel!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    x,
    src::AbstractArray{T},
    ::Val{U},
    n::Int, p::Int,
    ::Val{warpsz}
) where {F<:Function,O<:Function,G<:Function,T,S,U,warpsz}
    workgroup = Int(@groupsize()[1])
    lid  = Int(@index(Local, Linear))
    gid  = Int(@index(Group, Linear))
    gtid = (gid - 1) * workgroup + lid            # 1-based global thread
    lane = (lid - 1) % warpsz                     # 0-based lane in warp
    wlane = lane + 1                              # 1-based lane
    gwarp = (gtid - 1) ÷ warpsz                   # 0-based global warp
    nwarps = Int(@ndrange()[1]) ÷ warpsz          # total warps (grid-stride over cols)

    c = gwarp                                     # 0-based column, grid-strided
    while c < p
        # seed with this lane's first element (n ≥ warpsz ⇒ lane < n, always valid)
        i = lane
        acc = isnothing(x) ? f(src[i+1, c+1]) : f(x[i+1], src[i+1, c+1])
        i += warpsz
        # U-way unrolled main loop (MLP = U)
        while i + warpsz * (U - 1) < n
            acc = _mlp_fold(f, op, acc, x, src, i, c, Val(U), Val(warpsz))
            i += warpsz * U
        end
        # strided remainder
        while i < n
            v = isnothing(x) ? f(src[i+1, c+1]) : f(x[i+1], src[i+1, c+1])
            acc = op(acc, v)
            i += warpsz
        end
        # warp shuffle-reduction (up-scan; highest lane holds the result)
        offset = 1
        while offset < warpsz
            shuffled = @shfl(Up, acc, offset)
            if wlane > offset
                acc = op(shuffled, acc)
            end
            offset <<= 1
        end
        (wlane == warpsz) && (dst[c+1] = g(acc))
        c += nwarps
    end
end
