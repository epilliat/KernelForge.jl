# ============================================================================
# Buffer type
# ============================================================================

"""
    KernelBuffer{Arrays<:NamedTuple}

Pre-allocated buffer for GPU kernels, holding typed intermediate arrays
as a `NamedTuple` for named, type-stable access.

# Examples
For `mapreduce1d!`, the buffer contains:
- `arrays.partial`: partial reduction values (eltype = output of `f`)
- `arrays.flag`: synchronization flags (eltype = `UInt8`)
"""
struct KernelBuffer{Arrays<:NamedTuple}
    arrays::Arrays
end

macro allocate(expr)
    @assert expr.head == :call "expected a function call"
    func = expr.args[1]
    args = expr.args[2:end]
    tag = get(KERNEL_TAGS, func, nothing)
    tag === nothing && error("Unknown ForgeAlgorithm: $func")
    return :(get_allocation($(tag), $(esc.(args)...)))
end