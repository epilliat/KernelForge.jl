include("../quick_profile_simple.jl")

#%%
vecmat_params = Dict(
    Float32 => Dict(
        10^8 => Dict(
            (10^0, 10^8) => (Nitem=1, Nthreads=1, workgroup=256, blocks=128),
            (10^1, 10^7) => (Nitem=8, Nthreads=1, workgroup=256, blocks=128),
            (10^2, 10^6) => (Nitem=8, Nthreads=2, workgroup=256, blocks=128),
            (10^3, 10^5) => (Nitem=8, Nthreads=16, workgroup=256, blocks=128),
            (10^4, 10^4) => (Nitem=8, Nthreads=128, workgroup=256, blocks=128),
            (10^5, 10^3) => (Nitem=8, Nthreads=256, workgroup=256, blocks=128),
            (10^6, 10^2) => (Nitem=8, Nthreads=512, workgroup=256, blocks=128),
            (10^7, 10^1) => (Nitem=8, Nthreads=32 * 256, workgroup=256, blocks=128),
            (10^8, 10^0) => (Nitem=8, Nthreads=256 * 256, workgroup=256, blocks=128),
        ),
        10^7 => Dict(
            (10^0, 10^7) => (Nitem=1, Nthreads=1, workgroup=256, blocks=128),
            (10^1, 10^6) => (Nitem=8, Nthreads=1, workgroup=256, blocks=128),
            (10^2, 10^5) => (Nitem=8, Nthreads=2, workgroup=256, blocks=128),
            (10^3, 10^4) => (Nitem=16, Nthreads=16, workgroup=256, blocks=128),
            (10^4, 10^3) => (Nitem=16, Nthreads=128, workgroup=256, blocks=128),
            (10^5, 10^2) => (Nitem=16, Nthreads=4 * 256, workgroup=256, blocks=128),
            (10^6, 10^1) => (Nitem=16, Nthreads=32 * 256, workgroup=256, blocks=128),
            (10^7, 10^0) => (Nitem=16, Nthreads=256 * 256, workgroup=256, blocks=128),
        ),
        10^6 => Dict(
            (10^0, 10^6) => (Nitem=1, Nthreads=1, workgroup=256, blocks=128),
            (10^1, 10^5) => (Nitem=8, Nthreads=1, workgroup=256, blocks=128),
            (10^2, 10^4) => (Nitem=8, Nthreads=2, workgroup=256, blocks=128),
            (10^3, 10^3) => (Nitem=8, Nthreads=16, workgroup=256, blocks=128),
            (10^4, 10^2) => (Nitem=8, Nthreads=128, workgroup=256, blocks=128),
            (10^5, 10^1) => (Nitem=8, Nthreads=1024, workgroup=256, blocks=128),
            (10^6, 10^0) => (Nitem=8, Nthreads=40 * 256, workgroup=256, blocks=128),
        ),
    ),
    Float64 => Dict(
        10^8 => Dict(
            (10^0, 10^8) => (Nitem=1, Nthreads=1, workgroup=256, blocks=128),
            (10^1, 10^7) => (Nitem=8, Nthreads=1, workgroup=256, blocks=128),
            (10^2, 10^6) => (Nitem=8, Nthreads=2, workgroup=256, blocks=128),
            (10^3, 10^5) => (Nitem=8, Nthreads=16, workgroup=256, blocks=128),
            (10^4, 10^4) => (Nitem=8, Nthreads=128, workgroup=256, blocks=128),
            (10^5, 10^3) => (Nitem=8, Nthreads=128, workgroup=256, blocks=128),
            (10^6, 10^2) => (Nitem=8, Nthreads=128, workgroup=256, blocks=128),
            #(10^7, 10^1) => (Nitem=8, Nthreads=1024, workgroup=256, blocks=128),
            #(10^8, 10^0) => (Nitem=8, Nthreads=40 * 256, workgroup=256, blocks=128),
        ),
        10^7 => Dict(
            (10^0, 10^7) => (Nitem=1, Nthreads=1, workgroup=256, blocks=128),
            (10^1, 10^6) => (Nitem=1, Nthreads=1, workgroup=256, blocks=128),
            (10^2, 10^5) => (Nitem=4, Nthreads=4, workgroup=256, blocks=128),
            (10^3, 10^4) => (Nitem=4, Nthreads=32, workgroup=256, blocks=128),
            (10^4, 10^3) => (Nitem=4, Nthreads=256, workgroup=256, blocks=128),
            (10^5, 10^2) => (Nitem=4, Nthreads=256, workgroup=256, blocks=128),
            (10^6, 10^1) => (Nitem=4, Nthreads=3072, workgroup=256, blocks=128),
            (10^7, 10^0) => (Nitem=4, Nthreads=32768, workgroup=256, blocks=128),
        ),
        10^6 => Dict(
            (10^0, 10^6) => (Nitem=1, Nthreads=128, workgroup=128, blocks=1),
            (10^1, 10^5) => (Nitem=1, Nthreads=128, workgroup=128, blocks=1),
            (10^2, 10^4) => (Nitem=1, Nthreads=128, workgroup=128, blocks=1),
            (10^3, 10^3) => (Nitem=1, Nthreads=128, workgroup=128, blocks=1),
            (10^4, 10^2) => (Nitem=1, Nthreads=128, workgroup=128, blocks=1),
            (10^5, 10^1) => (Nitem=1, Nthreads=128, workgroup=128, blocks=1),
            (10^6, 10^0) => (Nitem=1, Nthreads=128, workgroup=128, blocks=1),
        ),
    ),
)


src = ROCArray{Float32}(undef, 10^5, 10^3)

arch = KF.MI300X()
KF.resolve_parameters(arch, KF.VecMat, src)
quick_profile_simple_roc(10^7, KF.VecMat, T=Float32, ms=10, def=true, tuned=true, AT=ROCArray, arch=arch)