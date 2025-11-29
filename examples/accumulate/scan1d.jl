using Luma
using CUDA

src = CuArray{Float64}(rand(Float32, 10^6))
dst = similar(src)


op(x, y) = x + y
op(x...) = op(x[1], op(x[2:end]...))
Luma.scan!(op, dst, src)
# test if it corresponds to Base.accumulate:
isapprox(Array(dst), accumulate(+, Array(src)))



#%%
using Quaternions
n = 1000000
op(x::QuaternionF64...) = *(x...)
T = QuaternionF64
src_cpu = [QuaternionF64(x ./ sqrt(sum(x .^ 2))...) for x in eachcol(randn(4, n))]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])


Luma.scan!(op, dst, src)
isapprox(Array(dst), accumulate(op, src_cpu)) #works with non commutative structures, without passing neutral or even init!