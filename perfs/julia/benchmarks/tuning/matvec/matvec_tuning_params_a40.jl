include("../quick_profile.jl")

#%%
matvec_params = Dict(
    Float32 => Dict(
        10^8 => Dict(
            (10^0, 10^8) => (Nitem=1, chunksz=1, Nblocks=256 * 10, workgroup=256),
            #(7, 10^7) => (Nitem=2, chunksz=2, Nblocks=64, workgroup=128),
            (10^1, 10^7) => (Nitem=2, chunksz=4, Nblocks=128, workgroup=256),
            (10^2, 10^6) => (Nitem=4, chunksz=4, Nblocks=32, workgroup=256),
            (10^3, 10^5) => (Nitem=4, chunksz=8, Nblocks=8, workgroup=256),
            (10^4, 10^4) => (Nitem=2, chunksz=32, Nblocks=2, workgroup=256),
            (10^5, 10^3) => (Nitem=2, chunksz=32, Nblocks=1, workgroup=256),
            (10^6, 10^2) => (Nitem=2, chunksz=64, Nblocks=1, workgroup=256),
            (10^7, 10^1) => (Nitem=2, chunksz=128, Nblocks=1, workgroup=256),
            (10^8, 10^0) => (Nitem=2, chunksz=256, Nblocks=1, workgroup=256),
        ),
        10^7 => Dict(
            (10^0, 10^7) => (Nitem=1, chunksz=1, Nblocks=2048, workgroup=256),
            (10^1, 10^6) => (Nitem=2, chunksz=4, Nblocks=128, workgroup=256),
            (10^2, 10^5) => (Nitem=2, chunksz=8, Nblocks=32, workgroup=256),
            (10^3, 10^4) => (Nitem=2, chunksz=16, Nblocks=8, workgroup=256),
            (10^4, 10^3) => (Nitem=2, chunksz=32, Nblocks=2, workgroup=256),
            (10^5, 10^2) => (Nitem=2, chunksz=64, Nblocks=1, workgroup=256),
            (10^6, 10^1) => (Nitem=2, chunksz=128, Nblocks=1, workgroup=256),
            (10^7, 10^0) => (Nitem=2, chunksz=256, Nblocks=1, workgroup=256),
        ),
        10^6 => Dict(
            (10^0, 10^6) => (Nitem=1, chunksz=1, Nblocks=2048, workgroup=256),
            (10^1, 10^5) => (Nitem=2, chunksz=4, Nblocks=128, workgroup=256),
            (10^2, 10^4) => (Nitem=2, chunksz=8, Nblocks=32, workgroup=256),
            (10^3, 10^3) => (Nitem=2, chunksz=16, Nblocks=8, workgroup=256),
            (10^4, 10^2) => (Nitem=2, chunksz=64, Nblocks=2, workgroup=256),
            (10^5, 10^1) => (Nitem=2, chunksz=128, Nblocks=1, workgroup=256),
            (10^6, 10^0) => (Nitem=2, chunksz=256, Nblocks=1, workgroup=256),
        ),
    ),
    Float64 => Dict(
        10^7 => Dict(
            (10^0, 10^7) => (Nitem=1, chunksz=1, Nblocks=128, workgroup=128),
            (10^1, 10^6) => (Nitem=2, chunksz=4, Nblocks=64, workgroup=128),
            (10^2, 10^5) => (Nitem=2, chunksz=8, Nblocks=16, workgroup=256),
            (10^3, 10^4) => (Nitem=2, chunksz=8, Nblocks=8, workgroup=256),
            (10^4, 10^3) => (Nitem=2, chunksz=16, Nblocks=2, workgroup=512),
            (10^5, 10^2) => (Nitem=2, chunksz=64, Nblocks=1, workgroup=512),
            (10^6, 10^1) => (Nitem=2, chunksz=64, Nblocks=1, workgroup=256),
            (10^7, 10^0) => (Nitem=2, chunksz=256, Nblocks=1, workgroup=256),
        ),
        10^6 => Dict(
            (10^0, 10^6) => (Nitem=1, chunksz=1, Nblocks=2048, workgroup=256),
            (10^1, 10^5) => (Nitem=2, chunksz=8, Nblocks=128, workgroup=256),
            (10^2, 10^4) => (Nitem=2, chunksz=16, Nblocks=32, workgroup=256),
            (10^3, 10^3) => (Nitem=2, chunksz=32, Nblocks=8, workgroup=256),
            (10^4, 10^2) => (Nitem=2, chunksz=64, Nblocks=2, workgroup=256),
            (10^5, 10^1) => (Nitem=2, chunksz=128, Nblocks=1, workgroup=256),
            (10^6, 10^0) => (Nitem=2, chunksz=256, Nblocks=1, workgroup=256),
        ),
    ),
)

arch = KF.A40()
quick_profile(10^6, KF.MatVec, T=Float32, arch=arch, ms=0.01, def=true, tuned=true)