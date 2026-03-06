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
KF.detect_arch(src)

@btime AMDGPU.device_id(dev)
@btime dev = KI.device()
KI.deviceid(src)
typeof(dev)
isbitstype(dev)