function maximum! end

function maximum!(rel, dst::AbstractArray{T}, src::AbstractArray{T}) where {T}
    op(x, y) = rel(x, y) ? y : x
    mapreduce1d!(identity, op, dst, src)
end

function maximum(rel::F, src::AbstractArray{T}) where {F<:Function,T}
    backend = get_backend(src)
    dst = KernelAbstractions.allocate(backend, T, 1)
    maximum!(rel, dst, src)
    Array(dst)[1]
end

maximum(src) = maximum(<, src)
minimum(src) = maximum(>, src)