"""
    vcopy!(dst::AbstractGPUVector, src::AbstractGPUVector; Nitem=4)

Copy `src` to `dst` using vectorized GPU memory access.

Performs a high-throughput copy by loading and storing `Nitem` elements per thread,
reducing memory transaction overhead compared to scalar copies.

# Arguments
- `dst`: Destination GPU vector
- `src`: Source GPU vector (must have same length as `dst`)
- `Nitem=4`: Number of elements processed per thread. Higher values improve throughput
  but require `length(src)` to be divisible by `Nitem`.

# Example
```julia
src = CUDA.rand(Float32, 1024)
dst = CUDA.zeros(Float32, 1024)
vcopy!(dst, src)
```

See also: [`Luma.setvalue!`](@ref)
"""
function vcopy!(dst::AbstractGPUVector, src::AbstractGPUVector; Nitem=4)
    backend = get_backend(src)
    ndrange = fld(length(src), Nitem)
    copy_kernel!(backend)(dst, src, Val(Nitem); ndrange=ndrange)
end

"""
    setvalue!(dst::AbstractGPUVector{T}, val::T; Nitem=4) where T

Fill `dst` with `val` using vectorized GPU memory access.

Performs a high-throughput fill by storing `Nitem` copies of `val` per thread,
reducing memory transaction overhead compared to scalar writes.

# Arguments
- `dst`: Destination GPU vector
- `val`: Value to fill (must match element type of `dst`)
- `Nitem=4`: Number of elements written per thread. Higher values improve throughput
  but require `length(dst)` to be divisible by `Nitem`.

# Example
```julia
dst = CUDA.zeros(Float32, 1024)
setvalue!(dst, 1.0f0)
```

See also: [`Luma.vcopy!`](@ref)
"""
function setvalue!(dst::AbstractGPUVector{T}, val::T; Nitem=4) where T
    backend = get_backend(dst)
    ndrange = fld(length(dst), Nitem)
    setvalue_kernel!(backend)(dst, val, Val(Nitem); ndrange=ndrange)
end