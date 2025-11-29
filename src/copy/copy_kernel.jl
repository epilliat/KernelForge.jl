using KernelAbstractions

@kernel inbounds = true function copy_kernel!(dst, src, ::Val{Nitem}) where {Nitem}

    I = @index(Global, Linear)

    values = vload(src, I, Val(Nitem))#ntuple(i -> src[idx_base+i], Val(Nitem))
    vstore!(dst, I, values)

    if I == @ndrange()[1]
        idx_base = I * Nitem
        for i in (1:Nitem)
            if idx_base + i <= length(src)
                dst[idx_base+i] = src[idx_base+i]
            end
        end
    end
end

@kernel inbounds = true function setvalue_kernel!(dst::AbstractArray{T}, val::T, ::Val{Nitem}) where {Nitem,T}
    I = @index(Global, Linear)


    values = ntuple(i -> val, Val(Nitem))
    vstore!(dst, I, values)

    if I == @ndrange()[1]
        idx_base = I * Nitem
        for i in (1:Nitem)
            if idx_base + i <= length(dst)
                dst[idx_base+i] = val
            end
        end
    end
end

