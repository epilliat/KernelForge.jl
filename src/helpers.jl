@generated function apply(f, srcs::NTuple{U}, idx) where {U}
    args = [:(srcs[$u][idx]) for u in 1:U]
    return :(f($(args...)))
end

@generated function broadcast_apply_across(f, srcs::NTuple{Nsrc}, idx, ::Val{Nitem}) where {Nsrc,Nitem}
    loads = [:(vload(srcs[$k], idx, Val($Nitem))) for k in 1:Nsrc]

    # Store them in variables
    load_vars = [Symbol(:load_, k) for k in 1:Nsrc]
    assignments = [:($var = $load) for (var, load) in zip(load_vars, loads)]

    # Generate Nitem function calls
    calls = []
    for item in 1:Nitem
        # f(load_1[item], load_2[item], load_3[item], ...)
        args = [:($(load_vars[k])[$item]) for k in 1:Nsrc]
        push!(calls, :(f($(args...))))
    end

    return quote
        $(assignments...)
        tuple($(calls...))
    end
end

@generated function broadcast_apply_across(f, x, A, idx_x, idx_A, ::Val{Nitem}) where {Nitem}
    # Load from x and A
    loads = [:(vload(x, idx_x, Val($Nitem))), :(vload(A, idx_A, Val($Nitem)))]

    # Store them in variables
    load_vars = [Symbol(:load_, k) for k in 1:2]
    assignments = [:($var = $load) for (var, load) in zip(load_vars, loads)]

    # Generate Nitem function calls
    calls = []
    for item in 1:Nitem
        # f(load_1[item], load_2[item])
        args = [:($(load_vars[k])[$item]) for k in 1:2]
        push!(calls, :(f($(args...))))
    end

    return quote
        $(assignments...)
        tuple($(calls...))
    end
end

function get_partition_sizes(blocks, Types::Type...)
    return (
        let
            sz = sizeof(T)
            alignment = lcm(8, sz)
            raw_size = blocks * sz + 8 * sz
            cld(raw_size, alignment) * alignment
        end
        for T in Types
    )
end


function partition(tmp::AbstractVector{UInt8}, dim, Types...)
    sizes = get_partition_sizes(dim, Types...)
    accum_sizes = (0, accumulate(+, sizes)...)
    return (
        reinterpret(T, view(tmp, accum_sizes[i]+1:accum_sizes[i+1]))
        for (i, T) in enumerate(Types)
    )
end


@inline @generated function tree_reduce(op::OP, data::NTuple{N,T}) where {OP,T,N}
    function build_tree(indices)
        count = length(indices)
        if count == 1
            return Symbol(:v_, indices[1])
        elseif count == 2
            return :(op($(Symbol(:v_, indices[1])), $(Symbol(:v_, indices[2]))))
        else
            mid = count ÷ 2
            left = build_tree(indices[1:mid])
            right = build_tree(indices[mid+1:end])
            return :(op($left, $right))
        end
    end

    quote
        Base.Cartesian.@nexprs $N i -> v_i = @inbounds data[i]
        $(build_tree(collect(1:N)))
    end
end


@inline @generated function tree_scan(op::OP, data::NTuple{N,T}) where {OP,T,N}
    if N == 1
        return :(Base.@_inline_meta; (data[1],))
    end

    # Upsweep: build tree reductions
    # Downsweep: propagate prefix sums

    exprs = Expr[]

    # Load values
    for i in 1:N
        push!(exprs, :($(Symbol(:v_, i)) = @inbounds data[$i]))
    end

    # Upsweep phase: compute tree reductions
    # Level k combines pairs at stride 2^k
    # Store intermediate results for downsweep
    levels = ceil(Int, log2(N))

    # Track current values at each position
    # After upsweep, position 2^k will hold reduction of indices 1:2^k
    for level in 0:levels-1
        stride = 1 << level
        next_stride = stride << 1
        for i in next_stride:next_stride:N
            left = i - stride
            # u_{level}_{i} holds reduction of (i-next_stride+1):i
            push!(exprs, :($(Symbol(:u_, level, :_, i)) = op($(level == 0 ? Symbol(:v_, left) : Symbol(:u_, level - 1, :_, left)),
                $(level == 0 ? Symbol(:v_, i) : Symbol(:u_, level - 1, :_, i)))))
        end
    end

    # For prefix sum, we need to compute cumulative sums
    # Simpler approach: just do sequential accumulation (still generates efficient code)

    # Actually, let's use the straightforward approach that unrolls nicely:
    quote
        Base.Cartesian.@nexprs $N i -> v_i = @inbounds data[i]
        Base.Cartesian.@nexprs $N i -> a_i = (i == 1 ? v_1 : op(a_{i - 1}, v_i))
        Base.Cartesian.@ntuple $N i -> a_i
    end
end

@inline @generated function tuple_reduce(op::OP, data::NTuple{N,T}) where {OP,T,N}
    quote
        Base.Cartesian.@nexprs $N i -> v_i = @inbounds data[i]
        Base.Cartesian.@nexprs $N i -> a_i = (i == 1 ? v_1 : op(a_{i - 1}, v_i))
        a_N
    end
end

@inline @generated function tuple_scan(op::OP, data::NTuple{N,T}) where {OP,T,N}
    quote
        Base.Cartesian.@nexprs $N i -> v_i = @inbounds data[i]
        Base.Cartesian.@nexprs $N i -> a_i = (i == 1 ? v_1 : op(a_{i - 1}, v_i))
        Base.Cartesian.@ntuple $N i -> a_i
    end
end

# Normalize dims to a sorted tuple of Ints
@inline _normalize_dims(dims::Int, nd::Int) = (dims > 0 ? dims : nd + dims + 1,)
@inline _normalize_dims(dims::NTuple{N,Int}, nd::Int) where {N} =
    Tuple(sort!(collect(ntuple(i -> dims[i] > 0 ? dims[i] : nd + dims[i] + 1, Val(N)))))
@inline _normalize_dims(dims::AbstractVector{<:Integer}, nd::Int) =
    _normalize_dims(Tuple(dims), nd)