const DEFAULT_WORKGROUP = 256
const DEFAULT_BLOCKS = 100
const warpsz = 32 # TODO: This might change from one architecture to another


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

function _type_to_symbol(T::DataType)
    return Symbol(lowercase(string(nameof(T))))
end

const KERNEL_TAGS = Dict{Symbol,DataType}(
    sym => T
    for T in _FORGE_ALGORITHM_TYPES
    for sym in (_type_to_symbol(T), Symbol(_type_to_symbol(T), :!))
)


