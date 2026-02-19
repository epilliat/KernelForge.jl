n = 100000007
T = Float32
src_cpu = [rand() for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])

Nitem = 1
KernelForge.vcopy!(dst, src; Nitem=Nitem)

@test all(dst .== src)

#%%n = 1000000
T = UInt8

n = 100000001
src_cpu = [0x03 for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])

Nitem = 4


KernelForge.vcopy!(dst, src; Nitem=Nitem)

@test all(dst .== src)

#%%
struct U
    x::UInt8
    y::UInt8
    z::UInt8
end
T = U


n = 100005
src_cpu = [T(1, 1, 1) for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([T(0, 0, 0) for _ in (1:n)])

Nitem = 2
KernelForge.vcopy!(dst, src; Nitem=Nitem)

@test all(dst .== src)


#%%

n = 100003
T = UInt8
dst = CuArray{T}([0xff for _ in (1:n)])

Nitem = 4
KernelForge.setvalue!(dst, 0x00; Nitem=Nitem)

@test all(dst .== 0x00)