"""
    argmax(rel, src::AbstractArray)

GPU parallel search returning the `(value, index)` pair of the element that is
extremal according to the relation `rel`.

Equivalent to `argmax1d(identity, rel, src)`.

# Arguments
- `rel`: Comparison relation (e.g. `>` for maximum, `<` for minimum)
- `src`: Input GPU array

# Examples
```julia
x = CuArray([3f0, 1f0, 4f0, 1f0, 5f0])
argmax(>, x)  # returns (5f0, 5)
argmax(<, x)  # returns (1f0, 2)
```

See also: [`KernelForge.argmax1d`](@ref), [`KernelForge.argmax`](@ref), [`KernelForge.argmin`](@ref).
"""
argmax(rel, src::AbstractArray)

"""
    argmax(src::AbstractArray)

GPU parallel argmax returning the `(value, index)` pair of the maximum element.

Equivalent to `argmax(>, src)`.

# Examples
```julia
x = CuArray([3f0, 1f0, 4f0, 1f0, 5f0])
argmax(x)  # returns (5f0, 5)
```

See also: [`KernelForge.argmin`](@ref), [`KernelForge.argmax1d`](@ref).
"""
argmax(src::AbstractArray)

"""
    argmin(src::AbstractArray)

GPU parallel argmin returning the `(value, index)` pair of the minimum element.

Equivalent to `argmax(<, src)`.

# Examples
```julia
x = CuArray([3f0, 1f0, 4f0, 1f0, 5f0])
argmin(x)  # returns (1f0, 2)
```

See also: [`KernelForge.argmax`](@ref), [`KernelForge.argmax1d`](@ref).
"""
argmin(src::AbstractArray)



argmax(rel, src::AbstractArray) = argmax1d(identity, rel, src)
argmax(src) = argmax(>, src)
argmin(src) = argmax(<, src)
