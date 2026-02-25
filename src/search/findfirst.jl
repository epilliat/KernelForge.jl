@inline default_nitem(::Type{FindFirst1D}, ::Type{T}) where T = default_nitem(MapReduce1D, T)

"""
    findfirst(filtr, src; kwargs...) -> Int or CartesianIndex or nothing

GPU parallel findfirst. Returns the index of the first element in `src`
for which `filtr` returns `true`, or `nothing` if no such element exists.
For multidimensional arrays, returns a `CartesianIndex`.

# Arguments
- `filtr`: Predicate function
- `src`: Input GPU array

# Keyword Arguments
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=$(DEFAULT_WORKGROUP)`: Workgroup size
- `blocks=$(DEFAULT_BLOCKS)`: Number of blocks

# Examples
```julia
x = adapt(backend, rand(Float32, 10_000))
findfirst(>(0.99f0), x)       # returns a linear index or nothing
A = adapt(backend, rand(Float32, 100, 100))
findfirst(>(0.99f0), A)       # returns a CartesianIndex or nothing
```

See also: [`KernelForge.findlast`](@ref).
"""
function findfirst(
    filtr::F,
    src::AbstractArray{T};
    Nitem=nothing,
    workgroup::Int=DEFAULT_WORKGROUP,
    blocks::Int=DEFAULT_BLOCKS
) where {F,T}
    n = length(src)
    backend = get_backend(src)
    _Nitem = something(Nitem, default_nitem(FindFirst1D, T))
    ndrange = blocks * workgroup
    nd = min(ndrange, cld(n, _Nitem))
    gs = max(32, min(workgroup, nd))
    gs = cld(gs, 32) * 32
    dst = KernelAbstractions.allocate(backend, Int, 1)
    fill!(dst, n + 1)
    findfirst_kernel!(backend, gs, nd)(dst, src, filtr, Val(_Nitem))
    result = Array(dst)[1]
    result > n && return nothing
    if ndims(src) != 1
        result = CartesianIndices(src)[result]
    end
    return result
end

"""
    findlast(filtr, src; kwargs...) -> Int or CartesianIndex or nothing

GPU parallel findlast. Returns the index of the last element in `src`
for which `filtr` returns `true`, or `nothing` if no such element exists.
Implemented by reversing `src` and delegating to [`KernelForge.findfirst`](@ref),
so it accepts the same keyword arguments.
For multidimensional arrays, returns a `CartesianIndex`.

# Arguments
- `filtr`: Predicate function
- `src`: Input GPU array

# Keyword Arguments
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=$(DEFAULT_WORKGROUP)`: Workgroup size
- `blocks=$(DEFAULT_BLOCKS)`: Number of blocks

# Examples
```julia
x = adapt(backend, rand(Float32, 10_000))
findlast(>(0.99f0), x)        # returns a linear index or nothing
A = adapt(backend, rand(Float32, 100, 100))
findlast(>(0.99f0), A)        # returns a CartesianIndex or nothing
```

See also: [`KernelForge.findfirst`](@ref).
"""
function findlast(
    filtr::F,
    src::AbstractArray{T};
    Nitem=nothing,
    workgroup::Int=DEFAULT_WORKGROUP,
    blocks::Int=DEFAULT_BLOCKS
) where {F,T}
    n = length(src)
    rev = @view src[end:-1:1]
    result = findfirst(filtr, rev; Nitem, workgroup, blocks)
    result === nothing && return nothing
    if ndims(src) != 1
        linear_idx = n + 1 - LinearIndices(src)[result]
        return CartesianIndices(src)[linear_idx]
    else
        return n + 1 - result
    end
end