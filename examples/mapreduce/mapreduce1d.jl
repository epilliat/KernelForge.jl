using Luma
using CUDA

src = CuArray{Float64}(rand(Float32, 10^6))
dst = similar([0.])


f(x) = x^2
op(x) = +(x...)
Luma.mapreduce!(f, op, dst, src)
isapprox(Array(dst), [mapreduce(f, op, Array(src))])



#%%
using Luma: UnitFloat8
n = 1000000
f(x::UnitFloat8) = Float32(x)
src = CuArray{UnitFloat8}([rand(UnitFloat8) for _ in (1:n)])
dst = CuArray{UnitFloat8}([0])

Luma.mapreduce!(f, +, dst, src)
dst

# dst is in (-1,1) range because of overflow of UnitFloat8, BUT since we are reducing Float32 values, the result has correct sign:

sign(Float32(CUDA.@allowscalar dst[1])) == sign(mapreduce(f, +, Array(Float32.(src))))