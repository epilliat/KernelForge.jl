@inline default_nitem(arch::AbstractArch, ::Type{FindFirst1D}, n, ::Type{T}) where T = 1#default_nitem(arch, MapReduce1D, n, T)

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
- `workgroup=nothing`: Workgroup size (auto-selected if nothing)
- `blocks=nothing`: Number of blocks (auto-selected if nothing)
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)

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
    workgroup=nothing,
    blocks=nothing,
    arch=nothing
) where {F,T}
    arch = something(arch, detect_arch(src))::AbstractArch
    n = length(src)
    Nitem = something(Nitem, default_nitem(arch, FindFirst1D, n, T))
    workgroup = something(workgroup, default_workgroup(arch))
    blocks = something(blocks, default_blocks(arch))
    backend = get_backend(src)
    ndrange = blocks * workgroup
    warpsz = get_warpsize(arch)

    nd = min(ndrange, cld(n, Nitem))
    gs = max(warpsz, min(workgroup, nd))
    gs = cld(gs, warpsz) * warpsz
    dst = KernelAbstractions.allocate(backend, Int, 1)
    fill!(dst, n + 1)
    findfirst_kernel!(backend, gs, nd)(dst, src, filtr, Val(Nitem), Val(warpsz))
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
- `workgroup=nothing`: Workgroup size (auto-selected if nothing)
- `blocks=nothing`: Number of blocks (auto-selected if nothing)
- `arch=nothing`: Architecture (auto-detected from `src` if nothing)

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
    workgroup=nothing,
    blocks=nothing,
    arch=nothing
) where {F,T}
    n = length(src)
    rev = @view src[end:-1:1]

    result = findfirst(filtr, rev; Nitem, workgroup, blocks, arch)
    result === nothing && return nothing
    if ndims(src) != 1
        linear_idx = n + 1 - LinearIndices(src)[result]
        return CartesianIndices(src)[linear_idx]
    else
        return n + 1 - result
    end
end