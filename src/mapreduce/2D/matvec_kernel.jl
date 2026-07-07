@kernel unsafe_indices = true function matvec_kernel!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractArray{T},
    x,
    ::Val{chunksz}, ::Val{Nblocks}, ::Val{Nitem},
    partial::Union{Nothing,AbstractArray{NTuple{Nitem,H}}},
    flag::Union{Nothing,AbstractArray{UInt8}},
    ::Type{H},
    ::Val{warpsz}
) where {F<:Function,O<:Function,G<:Function,T,H,S,chunksz,Nblocks,warpsz,Nitem}
    @uniform begin
        n, p = size(src)
        workgroup = @groupsize()[1]
        Nchunks = cld(workgroup, chunksz)
    end

    # Shared memory for inter-warp reduction: one element per chunk
    # careful that the size is equal to sizeof(H) * chunksz * cld(workgroup,warpsz) which can be large if chunksz if large
    # This is a reason why we try to keep chunksz small in practice.
    shared = @localmem NTuple{Nitem,H} max(chunksz * cld(workgroup, warpsz), warpsz)

    lid = Int(@index(Local))
    gid = Int(@index(Group))

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


    nbatch = cld(n, Nitem)
    # Strided reduction over columns
    col = first_col
    id_base = Nitem * (global_row - 1) + (col - 1) * n + 1  # 
    if id_base + Nitem - 1 <= n * p && col <= p
        values = vload(src, id_base, Val(Nitem), Val(false))
    else
        values = safe_vload(src, id_base, n * p, Val(Nitem))
        #values = ntuple(i -> src[n*p], Val(Nitem))
    end
    values = isnothing(x) ? f.(values) : f.(values, x[col])
    val = values  # now a tuple of Nitem elements


    col += Nchunks * Nblocks
    if global_row <= nbatch
        while col <= p
            id_base = Nitem * (global_row - 1) + (col - 1) * n + 1
            if id_base + Nitem - 1 <= n * p
                new_values = vload(src, id_base, Val(Nitem), Val(false))
            else
                new_values = safe_vload(src, id_base, n * p, Val(Nitem))
                #new_values = ntuple(i -> src[n*p], Val(Nitem))
            end
            new_values = isnothing(x) ? f.(new_values) : f.(new_values, x[col])
            val = op.(val, new_values)
            col += Nchunks * Nblocks
        end
    end
    # Shuffle reduction within chunk
    offset = chunksz
    while offset < warpsz
        shuffled = @shfl(Up, val, offset)
        if lane > offset
            val = op.(val, shuffled)
        end
        offset <<= 1
    end
    if Nblocks == 1 && global_row <= nbatch && (workgroup == warpsz || chunksz == workgroup)
        gval = g.(val)  # NTuple{Nitem,S}
        row_base = Nitem * (global_row - 1) + 1
        if row_base + Nitem - 1 <= n
            vstore!(dst, row_base, gval, Val(false))
        else
            # partial tail: scalar fallback
            for i in 1:Nitem
                row = row_base + i - 1
                if row <= n
                    dst[row] = gval[i]
                end
            end
        end
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
            shuffled = @shfl(Up, val, offset)
            if lane > offset
                val = op.(val, shuffled)
            end
            offset <<= 1
        end
        if lane % warps_per_row == 0 && cld(lane, warps_per_row) <= chunksz
            idx = (sid - 1) * chunksz + (wid - 1) * cld(warpsz, warps_per_row) + cld(lane, warps_per_row)
            if idx <= nbatch
                if Nblocks == 1
                    gval = g.(val)
                    row_base = Nitem * (idx - 1) + 1
                    if row_base + Nitem - 1 <= n
                        vstore!(dst, row_base, gval, Val(false))
                    else
                        for i in 1:Nitem
                            row = row_base + i - 1
                            if row <= n
                                dst[row] = gval[i]
                            end
                        end
                    end
                else
                    partial[(schid-1)*nbatch+idx] = val
                    @access flag[(schid-1)*nbatch+idx] = 0x01
                end
            end
        end
    end
    Nblocks == 1 && Base.@goto done
    if gid % Nblocks == 1
        col = chunk
        if global_row <= nbatch && col <= Nblocks
            while true
                (@access flag[(col-1)*nbatch+global_row]) == 0x01 && break
            end
            val = partial[(col-1)*nbatch+global_row]
        end
        col += Nchunks
        if global_row <= nbatch
            while col <= Nblocks
                while true
                    (@access flag[(col-1)*nbatch+global_row]) == 0x01 && break
                end
                @inbounds val = op.(val, partial[(col-1)*nbatch+global_row])
                col += Nchunks
            end
        end
        offset = chunksz
        #global_row == 1 && @print("$val,    $lane    $chunksz", "\n")
        #gid == 1 && global_row == 2 && @print("$val,  $(cld(lid,chunksz)),  wid: $wid, lane: $lane chunksz: $chunksz\n")
        while offset < min(chunksz * Nblocks, warpsz)
            shuffled = @shfl(Up, val, offset)
            if lane > offset
                val = op.(val, shuffled)
            end
            offset <<= 1
        end
        if chunksz * Nblocks <= warpsz
            if global_row <= nbatch && cld(lid, chunksz) == Nblocks
                gval = g.(val)
                row_base = Nitem * (global_row - 1) + 1
                if row_base + Nitem - 1 <= n
                    vstore!(dst, row_base, gval, Val(false))
                else
                    for i in 1:Nitem
                        row = row_base + i - 1
                        if row <= n
                            dst[row] = gval[i]
                        end
                    end
                end
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
                shuffled = @shfl(Up, val, offset)
                if (lane - 1) % warps_per_row + 1 > offset
                    val = op.(val, shuffled)
                end
                offset <<= 1
            end

            if lane % warps_per_row == 0 && cld(lane, warps_per_row) <= chunksz
                if idx <= nbatch
                    gval = g.(val)
                    row_base = Nitem * (idx - 1) + 1
                    if row_base + Nitem - 1 <= n
                        vstore!(dst, row_base, gval, Val(false))
                    else
                        for i in 1:Nitem
                            row = row_base + i - 1
                            if row <= n
                                dst[row] = gval[i]
                            end
                        end
                    end
                end
            end
        end
    end
    Base.@label done
end



# @kernel unsafe_indices = true inbounds = true function matvec_kernel!(
#     f::F, op::O, g::G,
#     dst::AbstractArray{S},
#     src::AbstractArray{T},
#     x,
#     ::Val{chunksz}, ::Val{Nblocks}, ::Val{1},
#     partial::Union{Nothing,AbstractArray{H}},
#     flag::Union{Nothing,AbstractArray{UInt8}},
#     ::Type{H},
#     ::Val{warpsz}
# ) where {F<:Function,O<:Function,G<:Function,T,H,S,chunksz,Nblocks,warpsz}
#     @print("abc\n")
#     @uniform begin
#         n, p = size(src)
#         workgroup = @groupsize()[1]
#         Nchunks = cld(workgroup, chunksz)
#     end

#     # Shared memory for inter-warp reduction: one element per chunk
#     # careful that the size is equal to sizeof(H) * chunksz * cld(workgroup,warpsz) which can be large if chunksz if large
#     # This is a reason why we try to keep chunksz small in practice.
#     shared = @localmem H max(chunksz * cld(workgroup, warpsz), warpsz)

#     lid = @index(Local)
#     gid = @index(Group)

#     #chunksz is exactly the number of rows that a workgroup takes

#     # which warp within the workgroup?
#     wid = cld(lid, warpsz)
#     lane = (lid - 1) % warpsz + 1
#     # Which lane within this group?
#     chlane = (lid - 1) % chunksz + 1
#     # Which chunk within the group?
#     chunk = cld(lid, chunksz)

#     # Which set of block ?
#     sid = cld(gid, Nblocks)
#     # Which set of chunk for a given block?
#     schid = (gid - 1) % Nblocks + 1
#     first_col = (schid - 1) * Nchunks + chunk

#     # Global row index
#     global_row = (sid - 1) * chunksz + chlane



#     # Strided reduction over columns
#     col = first_col
#     if global_row <= n && col <= p
#         val = isnothing(x) ? f(src[global_row, col]) : f(src[global_row, col], x[col])
#     else
#         val = isnothing(x) ? f(src[n, col]) : f(src[n, col], x[col]) # dummy value to define val (avoid the use of zero)
#     end
#     col += Nchunks * Nblocks
#     if global_row <= n
#         while col <= p
#             new_val = isnothing(x) ? f(src[global_row, col]) : f(src[global_row, col], x[col])
#             val = op(val, new_val)
#             col += Nchunks * Nblocks
#         end
#     end
#     # Shuffle reduction within chunk
#     offset = chunksz
#     while offset < warpsz
#         shuffled = @shfl(Up, val, offset)
#         if lane > offset
#             val = op(val, shuffled)
#         end
#         offset <<= 1
#     end
#     if Nblocks == 1 && global_row <= n && (workgroup == warpsz || chunksz == workgroup)
#         dst[global_row] = g(val)
#         Base.@goto done
#     end
#     if cld(lane, chunksz) == cld(warpsz, chunksz)
#         idx = chunksz <= warpsz ? (wid - 1) * chunksz + chlane : (wid - 1) * warpsz + lane
#         shared[idx] = val
#         #gid == 1 && chlane==1 && @print("$val, $idx, $chlane, $wid, lane: $lane chunksz: $chunksz\n")
#     end

#     @synchronize

#     #if wid % cld(warpsz, threads_per_row) == 1 
#     warps_per_row = cld(workgroup, max(warpsz, chunksz))
#     if wid <= cld(chunksz * warps_per_row, warpsz)
#         idx = cld(lane, warps_per_row) + ((lane - 1) % warps_per_row) * chunksz + (wid - 1) * cld(warpsz, warps_per_row)
#         val = shared[idx]
#         offset = 1
#         #gid == 1 && wid == 1 && @print("secc:  $val, $lane, $idx, $wid, idx: $idx, warps_per_row: $warps_per_row, abc: $(warps_per_row)\n")
#         while offset < warps_per_row
#             shuffled = @shfl(Up, val, offset)
#             if lane > offset
#                 val = op(val, shuffled)
#             end
#             offset <<= 1
#         end
#         if lane % warps_per_row == 0 && cld(lane, warps_per_row) <= chunksz
#             idx = (sid - 1) * chunksz + (wid - 1) * cld(warpsz, warps_per_row) + cld(lane, warps_per_row)
#             if idx <= n
#                 if Nblocks == 1
#                     dst[idx] = g(val)
#                 else
#                     partial[(schid-1)*n+idx] = val
#                     @access flag[(schid-1)*n+idx] = 0x01
#                 end
#             end
#         end
#     end
#     Nblocks == 1 && Base.@goto done
#     if gid % Nblocks == 1
#         col = chunk
#         if global_row <= n && col <= Nblocks
#             while true
#                 (@access flag[(col-1)*n+global_row]) == 0x01 && break
#             end
#             val = partial[(col-1)*n+global_row]
#         end
#         col += Nchunks
#         if global_row <= n
#             while col <= Nblocks
#                 while true
#                     (@access flag[(col-1)*n+global_row]) == 0x01 && break
#                 end
#                 @inbounds val = op(val, partial[(col-1)*n+global_row])
#                 col += Nchunks
#             end
#         end
#         offset = chunksz
#         #global_row == 1 && @print("$val,    $lane    $chunksz", "\n")
#         #gid == 1 && global_row == 2 && @print("$val,  $(cld(lid,chunksz)),  wid: $wid, lane: $lane chunksz: $chunksz\n")
#         while offset < min(chunksz * Nblocks, warpsz)
#             shuffled = @shfl(Up, val, offset)
#             if lane > offset
#                 val = op(val, shuffled)
#             end
#             offset <<= 1
#         end
#         if chunksz * Nblocks <= warpsz
#             if global_row <= n && cld(lane, chunksz) == Nblocks
#                 dst[global_row] = g(val)
#             end
#             Base.@goto done
#         end
#         #edge case can happen if warpsz < chunksz * Nblocks < workgroup
#         idx = chunksz <= warpsz ? (wid - 1) * chunksz + chlane : (wid - 1) * warpsz + lane
#         if (
#             cld(lid, chunksz) <= Nblocks && cld(lane, chunksz) == cld(warpsz, chunksz)
#             ||
#             cld(lid, chunksz) == Nblocks #edge case
#         )
#             shared[idx] = val
#         end
#         @synchronize

#         # The min is for edge case
#         warps_per_row = min(cld(workgroup, max(warpsz, chunksz)), Nblocks, cld(chunksz * Nblocks, warpsz))
#         if wid <= cld(chunksz * warps_per_row, warpsz)
#             idx = cld(lane, warps_per_row) + ((lane - 1) % warps_per_row) * chunksz + (wid - 1) * cld(warpsz, warps_per_row)
#             val = shared[idx]
#             offset = 1
#             idx = (sid - 1) * chunksz + (wid - 1) * cld(warpsz, warps_per_row) + cld(lane, warps_per_row)
#             #idx == 7 && @print("$val $lane  $idx  $(chunksz * warps_per_row) \n")
#             while offset < warps_per_row
#                 shuffled = @shfl(Up, val, offset)
#                 #careful to edge case for reduction here: warps_per_row is not necessarily a power of two!!
#                 if (lane - 1) % warps_per_row + 1 > offset# || lane > warps_per_row && lane - warps_per_row > offset
#                     val = op(val, shuffled)
#                 end
#                 offset <<= 1
#             end

#             if lane % warps_per_row == 0 && cld(lane, warps_per_row) <= chunksz
#                 if idx <= n
#                     dst[idx] = g(val)
#                 end
#             end
#         end
#     end
#     Base.@label done
# end

@kernel unsafe_indices = true inbounds = true function matvec_vload_kernel!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x,
    ::Val{Nitem},
    ::Type{H}
) where {F<:Function,O<:Function,G<:Function,T,H,S,Nitem}
    n, p = size(src)
    workgroup = Int(@groupsize()[1])
    lid = Int(@index(Local, Linear))
    gid = Int(@index(Group))
    I = (gid - 1) * workgroup + lid
    row_base = (I - 1) * Nitem + 1

    col_base_idx = row_base
    x_idx = 1

    # col = 1
    if col_base_idx + Nitem - 1 <= n * p
        vals = vload(src, col_base_idx, Val(Nitem), Val(false))
    else
        vals = ntuple(i -> src[n*p], Val(Nitem))
    end
    vals = isnothing(x) ? f.(vals) : f.(vals, x[x_idx])

    col_base_idx += n
    x_idx += 1

    # full columns
    while col_base_idx + Nitem - 1 <= n * p
        new_vals = vload(src, col_base_idx, Val(Nitem), Val(false))
        new_vals = isnothing(x) ? f.(new_vals) : f.(new_vals, x[x_idx])
        vals = op.(vals, new_vals)
        col_base_idx += n
        x_idx += 1
    end

    # last partial column
    if col_base_idx <= n * p
        clamped = n * p
        new_vals = ntuple(i -> src[clamped], Val(Nitem))
        new_vals = isnothing(x) ? f.(new_vals) : f.(new_vals, x[x_idx])
        vals = op.(vals, new_vals)
    end

    # write output
    if row_base + Nitem - 1 <= n
        vstore!(dst, row_base, g.(vals), Val(false))
    else
        for i in 1:Nitem
            row = Nitem * (row_base - 1) + i
            if row <= n
                dst[row] = g(vals[i])
            end
        end
    end
end

# ============================================================================
# Row-thread path — tiled over both dims (small-n..square-band, large-p).
# ============================================================================
# For wide/square matrices the generic kernel's small `chunksz` scatters a warp
# across many columns (sub-cacheline reads), capping BW at ~48% of A100 peak.
# This kernel instead maps ONE THREAD PER ROW with a register accumulator, so a
# warp reads `warpsz` CONTIGUOUS rows of one column (a fully-coalesced 128-B
# transaction) and streams down the columns. The per-thread column loop is
# U-way unrolled for memory-level parallelism.
#
# The launch grid tiles BOTH dims: grid = (row_tiles × col_blocks). A block's
# linear id `gid` decodes to `rtile = (gid-1)÷ncb` (which row-tile of height
# `wg`) and `cblock = (gid-1)%ncb` (which of `ncb` column-splits). The thread's
# global row is `r = rtile*wg + lid`; row-tiling lets `n` exceed a single block.
# Each column-block writes its partial reduction to `partial[cblock*n + r]`,
# combined across the `ncb` blocks by `matvec_rowthread_reduce!`. The earlier
# "wide" kernel is the special case ncb-only (1 row-tile). Measured A100:
# n=100,p=1e7 49%→76% peak (2.28×→1.05× cuBLAS); square/middle-band (10000² etc.)
# reaches parity or beats cuBLAS. The three tuned knobs (U, wg, ncb) are
# shape-dependent and data-driven via the autotune — see `_matvec_rowthread_impl!`
# and `_matvec_use_rowthread` in matvec.jl.

# U independent column contributions issued together (→ MLP), then folded into
# `acc` with `op`. `@generated` so the unroll count U and the x-vs-nothing
# branch are resolved at compile time (`col` is 0-based here).
@inline @generated function _rowthread_chunk(f::F, op::O, acc, src, x, col::Int, ::Val{U}, r::Int) where {F,O,U}
    load(u) = x === Nothing ? :(f(@inbounds src[r, col+$u])) :
                              :(f(@inbounds(src[r, col+$u]), @inbounds(x[col+$u])))
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

@kernel inbounds = true unsafe_indices = true function matvec_rowthread_kernel!(
    f::F, op::O,
    partial::AbstractVector{H},
    src::AbstractArray{T},
    x,
    n::Int, ::Val{U},
    ncol::Int, p::Int, wg::Int, ncb::Int
) where {F<:Function,O<:Function,T,H,U}
    gid = Int(@index(Group))
    lid = Int(@index(Local))
    rtile  = (gid - 1) ÷ ncb                    # which row-tile (height wg)
    cblock = (gid - 1) % ncb                    # which column-split (0-based)
    r      = rtile * wg + lid                   # global row (1-based)
    c0   = cblock * ncol                        # 0-based first column of this block
    cend = min(c0 + ncol, p)
    if r <= n && c0 < cend
        col = c0                               # 0-based
        acc = isnothing(x) ? f(src[r, col+1]) : f(src[r, col+1], x[col+1])
        col += 1
        while col + U <= cend
            acc = _rowthread_chunk(f, op, acc, src, x, col, Val(U), r)
            col += U
        end
        while col < cend
            v = isnothing(x) ? f(src[r, col+1]) : f(src[r, col+1], x[col+1])
            acc = op(acc, v)
            col += 1
        end
        @inbounds partial[cblock * n + r] = acc
    end
end

# Single-launch row-thread kernel for the no-column-split case (`ncbe == 1`): each
# thread owns a whole row, streams ALL p columns, applies `g`, and writes `dst[r]`
# directly — no `partial` scratch and no separate reduce launch. This is the
# tiny-work / small-p fast path (L1a): when the row-tiling alone fills the GPU
# there is no column split to reduce, so folding the reduce into the main kernel
# cuts 2 launches → 1 and drops the scratch allocation entirely.
@kernel inbounds = true unsafe_indices = true function matvec_rowthread_single_kernel!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractArray{T},
    x,
    n::Int, ::Val{U}, p::Int, wg::Int
) where {F<:Function,O<:Function,G<:Function,S,T,U}
    gid = Int(@index(Group))
    lid = Int(@index(Local))
    r   = (gid - 1) * wg + lid                   # global row (1-based)
    if r <= n
        acc = isnothing(x) ? f(src[r, 1]) : f(src[r, 1], x[1])
        col = 1                                  # 0-based cursor past the seed
        while col + U <= p
            acc = _rowthread_chunk(f, op, acc, src, x, col, Val(U), r)
            col += U
        end
        while col < p
            v = isnothing(x) ? f(src[r, col+1]) : f(src[r, col+1], x[col+1])
            acc = op(acc, v)
            col += 1
        end
        @inbounds dst[r] = g(acc)
    end
end

# Combine the `ncb` per-column-block partials for each row and apply `g`. Block
# 0 is always active (c0=0 < p), so it seeds the reduction — no op-identity.
@kernel inbounds = true unsafe_indices = true function matvec_rowthread_reduce!(
    op::O, g::G,
    dst::AbstractArray{S},
    partial::AbstractVector{H},
    n::Int, ncb::Int
) where {O<:Function,G<:Function,S,H}
    # Use the explicit `Linear` form (the codebase convention; bare `@index(Global)`
    # defaults to Cartesian). NOTE: the trailing-partial-block OOB under
    # `unsafe_indices=true` is fixed at the LAUNCH site by rounding `ndrange` up to a
    # groupsize multiple (see `_matvec_rowthread_impl!` in matvec.jl) — the `if r<=n`
    # guard then drops the extra threads.
    r = Int(@index(Global, Linear))
    if r <= n
        acc = @inbounds partial[r]
        for b in 1:ncb-1
            acc = op(acc, @inbounds partial[b * n + r])
        end
        @inbounds dst[r] = g(acc)
    end
end