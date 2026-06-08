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
    tag = get(KERNEL_TAGS, func, nothing)
    tag === nothing && error("Unknown ForgeAlgorithm: $func")
    rest = expr.args[2:end]

    # In Julia's :call AST, a `; kwarg=value` block is represented as
    # `Expr(:parameters, ...)` and lives at args[2] (right after the function
    # name). When present, it must remain at args[2] of the rewritten call —
    # otherwise the macro emits `get_allocation(Tag, Expr(:parameters, ...), src)`
    # which Julia parses as a malformed positional argument.
    new_args = Any[GlobalRef(@__MODULE__, :get_allocation)]
    if !isempty(rest) && isa(rest[1], Expr) && rest[1].head == :parameters
        push!(new_args, rest[1])              # kwargs block stays at position 2
        push!(new_args, tag)                  # tag is the first positional arg
        append!(new_args, rest[2:end])
    else
        push!(new_args, tag)
        append!(new_args, rest)
    end
    return esc(Expr(:call, new_args...))
end

function _unsafe_free! end

function _unsafe_free!(tmp::KernelBuffer)
    for arr in tmp.arrays
        _unsafe_free!(arr)
    end
end