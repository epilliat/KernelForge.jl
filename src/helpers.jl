"""
    apply(f, srcs::NTuple{U}, idx) where {U}

Apply function `f` to elements at index `idx` from multiple source arrays.

This is a generated function that efficiently expands to `f(srcs[1][idx], srcs[2][idx], ..., srcs[U][idx])`
at compile time, avoiding runtime overhead.

# Arguments
- `f`: Function to apply
- `srcs`: Tuple of source arrays
- `idx`: Index to access in each array

# Returns
Result of `f(srcs[1][idx], srcs[2][idx], ...))`

# Examples
```julia
src1 = [1, 2, 3, 4]
src2 = [10, 20, 30, 40]
src3 = [100, 200, 300, 400]
srcs = (src1, src2, src3)

# Compute sum across sources at index 2
result = apply(+, srcs, 2)  # = 2 + 20 + 200 = 222
```

# Performance
Generated function avoids tuple unpacking overhead and is fully inlined at compile time.
Necessary compilation in kernels
"""
@generated function apply(f, srcs::NTuple{U}, idx) where {U}
    args = [:(srcs[$u][idx]) for u in 1:U]
    return :(f($(args...)))
end

"""
    broadcast_apply_across(f, srcs::NTuple{Nsrc}, idx, ::Val{Nitem}) where {Nsrc, Nitem}

Apply function `f` across multiple source arrays for `Nitem` consecutive positions, using vectorized loads.

This generated function loads `Nitem` consecutive elements from each source array starting at `idx`,
then applies `f` to corresponding elements across all sources, effectively transposing the operation.

# Arguments
- `f`: Function to apply (must accept `Nsrc` arguments)
- `srcs`: Tuple of `Nsrc` source arrays
- `idx`: Starting index for vectorized load
- `Val(Nitem)`: Number of consecutive elements to process

# Returns
`NTuple{Nitem}` where each element is the result of applying `f` to corresponding elements across sources.

# Examples
```julia
src1 = [1, 2, 3, 4, 5]
src2 = [10, 20, 30, 40, 50]
src3 = [100, 200, 300, 400, 500]
srcs = (src1, src2, src3)

# Process 4 consecutive elements starting at index 1
result = broadcast_apply_across(+, srcs, 1, Val(4))
# Returns: (111, 222, 333, 444)
# Computes: (1+10+100, 2+20+200, 3+30+300, 4+40+400)
```

# Generated Code Pattern
For `Nsrc=3` and `Nitem=4`, generates:
```julia
let
    load_1 = vload(srcs[1], idx, Val(4))
    load_2 = vload(srcs[2], idx, Val(4))
    load_3 = vload(srcs[3], idx, Val(4))
    tuple(
        f(load_1[1], load_2[1], load_3[1]),
        f(load_1[2], load_2[2], load_3[2]),
        f(load_1[3], load_2[3], load_3[3]),
        f(load_1[4], load_2[4], load_3[4])
    )
end
```

# Performance
- Minimizes memory accesses by using vectorized loads
- Fully unrolled at compile time for optimal GPU performance
- Requires `vload` function to be defined for the array types

# See Also
- [`apply`](@ref): For single-element access across sources
- [`vload`](@ref): For loading consecutive elements
"""
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

"""
    get_partition_sizes(blocks, Types::Type...)

Calculate aligned memory partition sizes for temporary storage arrays.

Computes the size in bytes needed for each type, rounded up to 8-byte alignment
for efficient memory access patterns on GPUs.

# Arguments
- `blocks`: Number of blocks (typically thread blocks or work units)
- `Types`: Variable number of types to allocate space for

# Returns
Tuple of partition sizes (in bytes) for each type, 8-byte aligned.

# Formula
For each type `T`: `((blocks * sizeof(T) + 8 * sizeof(T)) >> 3) << 3`
- Allocates space for `blocks` elements of type `T`
- Adds padding of `8 * sizeof(T)`
- Rounds up to nearest multiple of 8 bytes

# Examples
```julia
sizes = get_partition_sizes(100, Float32, Int64, UInt8)
# Returns sizes for 100 blocks of each type, aligned
```

# See Also
- [`partition`](@ref): Uses these sizes to partition a memory buffer
"""
function get_partition_sizes(blocks, Types::Type...)
    return (((blocks * sizeof(T) + 8 * sizeof(T)) >> 3) << 3 for T in Types)
end

"""
    partition(tmp::AbstractVector{UInt8}, blocks, Types...)

Partition a temporary buffer into typed views for multiple data types.

Takes a flat byte buffer and divides it into non-overlapping, properly aligned sections
for each requested type, using sizes computed by `get_partition_sizes`.

# Arguments
- `tmp`: Temporary buffer (typically `Vector{UInt8}` or `CuArray{UInt8}`)
- `blocks`: Number of blocks (must match value used for buffer allocation)
- `Types`: Types to partition the buffer for

# Returns
Tuple of reinterpreted views, one for each type in `Types`.

# Examples
```julia
# Allocate temporary buffer
blocks = 100
tmp = Vector{UInt8}(undef, sum(get_partition_sizes(blocks, Float32, Int64)))

# Partition into typed arrays
(floats, ints) = partition(tmp, blocks, Float32, Int64)

# Now floats and ints are views into tmp with appropriate types
floats[1] = 3.14f0
ints[1] = 42
```

# GPU Usage
```julia
# Allocate shared or global temporary storage on GPU
tmp = CuArray{UInt8}(undef, sum(get_partition_sizes(blocks, Float32, Int32)))
(flags, accumulator) = partition(tmp, blocks, UInt32, Float32)
```

# Notes
- Views are non-overlapping and properly aligned
- Zero-copy operation (no data movement)
- Thread-safe as long as different threads access different blocks

# See Also
- [`get_partition_sizes`](@ref): Computes the required sizes
"""
function partition(tmp::AbstractVector{UInt8}, blocks, Types...)
    sizes = get_partition_sizes(blocks, Types...)
    accum_sizes = (0, accumulate(+, sizes)...)
    return (
        reinterpret(T, view(tmp, accum_sizes[i]+1:accum_sizes[i+1]))
        for (i, T) in enumerate(Types)
    )
end

"""
    get_default_config(obj::KernelAbstractions.Kernel, args...)

Get default kernel launch configuration (workgroup size and number of blocks).

This is a fallback implementation that returns conservative defaults. Backend-specific
extensions should override this method to compute optimal configurations based on
hardware capabilities and kernel characteristics.

# Arguments
- `obj`: KernelAbstractions.Kernel object
- `args...`: Kernel arguments (used by backend-specific implementations)

# Returns
Named tuple with:
- `workgroup`: Number of threads per workgroup (default: 256)
- `blocks`: Number of workgroups (default: 100)

# Examples
```julia
kernel = my_kernel!(backend)
config = get_default_config(kernel, args...)
kernel(args...; workgroup=config.workgroup, blocks=config.blocks)
```

# Backend Extensions
Backend-specific extensions (e.g., LumaCUDAExt) should provide optimized implementations:
```julia
function get_default_config(obj::Kernel{CUDABackend}, args...)
    # Query device capabilities
    # Compute optimal occupancy
    # Return (workgroup=..., blocks=...)
end
```

# See Also
- [`get_default_config_cached`](@ref): Cached version avoiding recomputation
"""
function get_default_config(obj::KernelAbstractions.Kernel, args...)
    return (workgroup=256, blocks=100)
end

"""
    get_default_config_cached(obj::K, args...) where {K<:KernelAbstractions.Kernel}

Cached version of `get_default_config` that generates specialized methods for each unique
kernel type and argument signature.

Uses `@eval` to create specialized methods at runtime, so subsequent calls with the same
types return the cached configuration without recomputation.

# Arguments
- `obj`: KernelAbstractions.Kernel object
- `args...`: Kernel arguments

# Returns
Same as `get_default_config`: named tuple with `workgroup` and `blocks`.

# Examples
```julia
# First call: computes and caches
config1 = get_default_config_cached(kernel, dst, src1, src2)

# Second call with same types: uses cached method (fast)
config2 = get_default_config_cached(kernel, dst, src1, src2)
```

# Implementation Notes
- Generates specialized methods using `@eval` for type-specific caching
- First call for a given type signature: slow (computes + generates method)
- Subsequent calls: fast (directly dispatches to generated method)
- Thread-safe: Julia's method table is thread-safe

# Performance
Particularly useful for persistent kernels or frequently launched kernels where
configuration computation overhead would be noticeable.

# See Also
- [`get_default_config`](@ref): Base implementation (not cached)
"""
function get_default_config_cached(obj::K, args...) where {K<:KernelAbstractions.Kernel}
    param_types = [:(::$(typeof(arg))) for arg in args]
    config = get_default_config(obj, args...)
    @eval get_default_config_cached(::$K, $(param_types...)) = $config
    return config
end

"""
    default_value(::Type{T}) where T

Generate a default zero-initialized value for a bitstype `T`.

Creates an instance of type `T` with all bits set to zero, without requiring
a zero constructor to be defined for `T`. Useful for initializing custom types
in GPU kernels where constructors may not be available.

# Arguments
- `T`: A bitstype (primitive type or immutable struct with only bitstype fields)

# Returns
Zero-initialized instance of type `T`

# Throws
- Error if `T` is not a bitstype
- Error if `T` has zero size

# Examples
```julia
# Built-in types
default_value(Float32)    # 0.0f0
default_value(Int64)      # 0
default_value(UInt8)      # 0x00

# Custom bitstype
struct MyType
    x::Float32
    y::Int32
end

default_value(MyType)     # MyType(0.0f0, 0)
```

# GPU Usage
```julia
# Initialize shared memory in GPU kernel
@inline function my_kernel!()
    shared = @cuDynamicSharedMem(MyCustomType, 256)
    if threadIdx().x == 1
        shared[1] = default_value(MyCustomType)
    end
end
```

# Implementation
Uses `reinterpret` on a tuple of zero bytes with length equal to `sizeof(T)`.
This is a compile-time operation for constant types.

# Limitations
- Only works for bitstypes (no pointers, no heap allocations)
- Returns type with all bits zero (may not be semantically meaningful for all types)
"""
@generated function dummy_value(::Type{T}) where T
    if isbitstype(T)
        nbytes = Base.packedsize(T)
        if nbytes == 0
            return :(error("Cannot create instance of zero-sized type $T"))
        end
        bytes = ntuple(i -> zero(UInt8), Val(nbytes))
        # Create from zero bits
        u = reinterpret(T, bytes)
        return quote
            $u
        end
    else
        return :(error("Cannot create dummy value for non-bitstype $T"))
    end
end