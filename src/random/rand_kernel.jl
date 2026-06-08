# Generic Philox-driven sampling kernel.
#
# Parameterised on two compile-time `Val`s:
#   SPP = samples_per_philox(d)   — 4 (Float32 1u/sample), 2 (Normal{Float32},
#                                    Uniform{Float64}), or 1 (Float64 2u/sample).
#   NPC = Philox calls per thread — 1 (round-2 default), 2, 4, ...
#                                    Selected by default_nitem at the wrapper.
#
# Each thread does NPC Philox calls and emits SPP*NPC samples in a single
# `vstore!`. Total samples-per-thread = Nitem = SPP × NPC.
#
# Pattern follows src/scan/scan_kernel.jl: the @kernel takes `Val{...}`
# parameters directly, and a small `@generated` helper handles the
# compile-time tuple flattening (analogous to `prefix_apply` /
# `tree_scan`). No `@eval` factory needed.

using KernelAbstractions
using KernelIntrinsics: vstore!

# Compile-time helper: NPC philox_block calls + sample_block, flattened
# into one NTuple{SPP*NPC, T}. Avoids the closure of `ntuple(Val(NPC)) do c`
# which the GPU compiler doesn't always fully inline. Body is fully
# unrolled at @generated time → scalar SSA on the GPU.
@generated function _gather_samples(dist, seed::UInt64, base_blk::UInt64,
                                    ::Val{SPP}, ::Val{NPC}) where {SPP, NPC}
    blk_syms = [Symbol("blk_$c") for c in 1:NPC]
    blk_assigns = [:($(blk_syms[c]) = sample_block(dist,
                       philox_block(seed, base_blk + UInt64($(c - 1)))))
                   for c in 1:NPC]
    elems = [:($(blk_syms[c])[$j]) for c in 1:NPC for j in 1:SPP]
    quote
        $(blk_assigns...)
        $(Expr(:tuple, elems...))
    end
end

@kernel inbounds = true function _rand_kernel!(
    out::AbstractVector, dist, seed::UInt64,
    ::Val{SPP}, ::Val{NPC}, Ntotal::Int
) where {SPP, NPC}
    tid      = @index(Global, Linear)
    base_blk = (UInt64(tid) - UInt64(1)) * UInt64(NPC)
    base_idx = (Int(tid) - 1) * SPP * NPC

    if base_idx + SPP * NPC <= Ntotal
        vstore!(out, tid, _gather_samples(dist, seed, base_blk, Val(SPP), Val(NPC)))
    else
        # Tail: per-element via sample_at. Hit at most once per launch.
        for j in 1:(SPP * NPC)
            idx = base_idx + j
            if idx <= Ntotal
                out[idx] = sample_at(dist, seed, idx)
            end
        end
    end
end

# Persistent-thread kernel: launch a fixed small number of threads
# (= ndrange = n_sms × blocks_per_sm × workgroup), each looping over
# its share of output blocks. Wins at small N where the standard
# kernel launches more blocks than the GPU can run concurrently.
#
# Block index in `out` for thread tid, iter j is `tid + j × stride`
# (1-based). Philox counter is `block_idx - 1` — same mapping as the
# standard kernel → byte-identical output for any (seed, N).
# Verified by the "algo invariance" testset.
@kernel inbounds = true function _rand_persistent_kernel!(
    out::AbstractVector, dist, seed::UInt64, ::Val{SPP}, Ntotal::Int
) where {SPP}
    tid    = @index(Global, Linear)
    stride = prod(@ndrange())

    j = 0
    while true
        block_idx = Int(tid) + j * stride
        base_idx  = (block_idx - 1) * SPP
        base_idx >= Ntotal && break

        u4 = philox_block(seed, UInt64(block_idx - 1))
        if base_idx + SPP <= Ntotal
            vstore!(out, block_idx, sample_block(dist, u4))
        else
            for k in 1:SPP
                idx = base_idx + k
                if idx <= Ntotal
                    out[idx] = sample_at(dist, seed, idx)
                end
            end
        end
        j += 1
    end
end
