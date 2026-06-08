default_workgroup(::AbstractArch) = 256
default_blocks(::AbstractArch) = 128
default_blocks(::AMDArch) = 256 # tuned for MI300X

abstract type ForgeAlgorithm end

const _FORGE_ALGORITHM_TYPES = DataType[]

macro forge_algorithm(name)
    quote
        struct $(esc(name)) <: ForgeAlgorithm end
        push!(_FORGE_ALGORITHM_TYPES, $(esc(name)))
    end
end

@forge_algorithm MapReduce
@forge_algorithm MapReduce1D
@forge_algorithm MapReduce2D
@forge_algorithm MapReduceDims
@forge_algorithm MatVec
@forge_algorithm VecMat
@forge_algorithm Scan
@forge_algorithm Scan1D
@forge_algorithm FindFirst
@forge_algorithm FindFirst1D
@forge_algorithm Argmax
@forge_algorithm Argmax1D
@forge_algorithm Sort1D
@forge_algorithm SortColumns
@forge_algorithm Sortperm

function _type_to_symbol(T::DataType)
    return Symbol(lowercase(string(nameof(T))))
end

const KERNEL_TAGS = Dict{Symbol,DataType}(
    sym => T
    for T in _FORGE_ALGORITHM_TYPES
    for sym in (_type_to_symbol(T), Symbol(_type_to_symbol(T), :!))
)
# Public function-name aliases — the function for Sort1D is named `sort`,
# so `@allocate sort(...)` looks up `:sort` and resolves to the Sort1D tag.
KERNEL_TAGS[:sort]  = Sort1D
KERNEL_TAGS[:sort!] = Sort1D
KERNEL_TAGS[:sort_columns]  = SortColumns
KERNEL_TAGS[:sort_columns!] = SortColumns
KERNEL_TAGS[:batched_radix_sort_columns!] = SortColumns


