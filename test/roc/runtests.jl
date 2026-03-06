using Pkg
Pkg.activate("test/roc")
using Revise

Pkg.develop(path=".")
Pkg.develop(path="../KernelIntrinsics.jl")
Pkg.add(["AMDGPU"])

using KernelForge
import KernelForge as KF
using KernelIntrinsics
import KernelIntrinsics as KI
using AMDGPU
using BenchmarkTools

src = ROCArray{Float64}(undef, 50)
KI.device(src)
arch = KF.detect_arch(src)

@btime AMDGPU.device_id(dev)
@btime dev = KI.device()
KI.deviceid(src)
typeof(dev)
isbitstype(dev)

KF.get_warpsize(arch)


AT = ROCArray
T = Float64
n = 1000
src = AT(T.([i for i in 1:n]))
dst = AT([T(0)])

# Warm up
KF.mapreduce1d!(identity, +, dst, src)