using Luma
using CUDA


src = CuArray{Float64}(rand(Float32, 10^6))
dst = similar(src)

#copy with vectorized loads and stores:
vcopy!(dst, src, Nitem=4)

isapprox(dst, src)