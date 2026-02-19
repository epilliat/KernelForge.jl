@kernel function findfirst_kernel!(dst, src, x, ::Val{Nitem}) where {Nitem}
    I = @index(Global)
    n = length(src)
    warpsz = @warpsize
    lane = (I - 1) % warpsz + 1
    ndrange = @ndrange()[1]
    found = false
    best_id = n + 1
    i = I

    while (i - 1) * Nitem + 1 <= n
        values = vload(src, i, Val(Nitem))
        for j in 1:Nitem
            idx = (i - 1) * Nitem + j
            if !found && idx <= n && values[j] == x
                best_id = idx
                found = true
            end
        end
        # Check every iteration
        first_i = i - lane + 1
        @access current_best = dst[1]
        current_best < (first_i - 1) * Nitem + 1 && Base.@goto done
        any_found = @vote(AnyLane, found)
        any_found && break
        i += ndrange
    end

    # Warp-level min reduction via shuffle
    offset = 1
    while offset < warpsz
        shuffled_id = @shfl(Up, best_id, offset)
        if lane > offset && shuffled_id < best_id
            best_id = shuffled_id
        end
        offset <<= 1
    end
    # Last lane holds the warp minimum
    if lane == warpsz && best_id < n + 1
        Atomix.@atomic dst[1] min best_id
    end
end

# Test
n = 101
src = CUDA.rand(Float64, n)
CUDA.@allowscalar x = src[31]
Nitem = 4
ndrange = 100 * 256
dst = CuArray([n + 1])
CUDA.@sync findfirst_kernel!(CUDABackend(), 256, ndrange)(dst, src, x, Val(Nitem))
CUDA.@allowscalar dst[1]  # should be 100000



using Test

n_values = [
    1, 2, 3, 4,           # tiny edge cases
    7, 8, 9,                  # around Nitem boundaries
    31, 32, 33,               # around warp size
    255, 256, 257,            # around group size
    1023, 1024, 1025,         # around ndrange
    10_000, 100_000,          # medium
    1_000_000, 10_000_000,    # large
]

Nitem = 4
groupsize = 256
ndrange = 100 * groupsize


n = 2

src = CUDA.rand(Float64, n)

# Test finding element at various positions
positions = filter(<=(n), [1, 2, n ÷ 2, n - 1, n])
unique!(positions)

pos = 1
x = CUDA.@allowscalar src[pos]
dst = CuArray([n + 1])
nd = min(ndrange, cld(n, Nitem))  # don't launch more threads than needed

CUDA.@sync findfirst_kernel!(CUDABackend(), min(groupsize, nd), nd)(dst, src, x, Val(Nitem))
result = CUDA.@allowscalar dst[1]

# Result should be <= pos (there might be duplicates earlier)
@test result <= pos
# Result should point to the correct value
@test CUDA.@allowscalar src[result] == x


# Test element not found
dst = CuArray([n + 1])
x = -1.0  # won't appear in CUDA.rand output
nd = min(ndrange, cld(n, Nitem))
CUDA.@sync findfirst_kernel!(CUDABackend(), min(groupsize, nd), nd)(dst, src, x, Val(Nitem))
result = CUDA.@allowscalar dst[1]
@test result == n + 1
