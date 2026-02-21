const DEFAULT_WORKGROUP = 256
const DEFAULT_BLOCKS = 100
const warpsz = 32 # TODO: This might change from one architecture to another


struct MapReduce1D end
struct MatVec end
struct VecMat end
struct Scan1D end
struct FindFirst1D end
struct Argmax1D end

@inline function default_nitem(::Type{MapReduce1D}, ::Type{T}) where {T}
    if sizeof(T) == 1
        return 8
    elseif sizeof(T) == 2
        return 4
    else
        return 1
    end
end

@inline function default_nitem(::Type{Scan1D}, ::Type{T}) where {T}
    sz = sizeof(T)
    if sz == 1
        return 16
    elseif sz == 2
        return 16
    elseif sz == 4
        return 8
    elseif sz == 8
        return 8
    else
        return 4
    end
end

@inline default_nitem(::Type{Argmax1D}, ::Type{T}) where T = default_nitem(MapReduce1D, T)
@inline default_nitem(::Type{FindFirst1D}, ::Type{T}) where T = default_nitem(MapReduce1D, T)