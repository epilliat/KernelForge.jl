# ─────────────────────────────────────────────────────────────────────────────
# Backend selection
# ─────────────────────────────────────────────────────────────────────────────
using Pkg
include("meta_helpers.jl")

TEST_BACKEND = get(ENV, "TEST_BACKEND") do
    backend_str = has_cuda() ? "cuda" : has_roc() ? "roc" : "unknown"
    @info "TEST_BACKEND not set, defaulting to $backend_str"
    backend_str
end


Pkg.activate(joinpath(@__DIR__, "envs", TEST_BACKEND))
Pkg.instantiate()

using Test
using KernelAbstractions
import KernelAbstractions as KA
using KernelForge
import KernelForge as KF
using Random




if TEST_BACKEND == "cuda"
    using CUDA
    if !CUDA.functional()
        @warn "No CUDA device found — skipping tests"
        exit(0)
    end
    AT = CuArray
    backend = CUDABackend()
elseif TEST_BACKEND == "roc"
    using AMDGPU
    if !AMDGPU.functional()
        @warn "No AMDGPU device found — skipping tests"
        exit(0)
    end
    AT = ROCArray
    backend = ROCBackend()
    arch = KF.detect_arch(backend, Val(1))
else
    error("Unknown backend: $TEST_BACKEND")
end
include("helpers.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Test target dispatch.
#
# - No ARGS                    → full suite (general_routine.jl), backwards-compat.
# - ARGS = ["<dir>", ...]      → run all tests/<dir>/*_test.jl
# - ARGS = ["<name>", ...]     → run a specific tests/*/<name>_test.jl file
#
# Examples:
#   julia --project=test/envs/cuda test/runtests.jl sort
#   Pkg.test("KernelForge"; test_args=["matvec"])
#   Pkg.test("KernelForge"; test_args=["sort", "scan"])
# ─────────────────────────────────────────────────────────────────────────────

function _run_test_target(name::AbstractString)
    dir = joinpath(@__DIR__, "tests", name)
    if isdir(dir)
        @testset "$name" begin
            for f in sort(readdir(dir; join=true))
                endswith(f, "_test.jl") && include(f)
            end
        end
        return
    end
    for d in readdir(joinpath(@__DIR__, "tests"); join=true)
        isdir(d) || continue
        f = joinpath(d, "$(name)_test.jl")
        if isfile(f)
            @testset "$name" begin include(f) end
            return
        end
    end
    error("Unknown test target `$name`: no tests/$name/ directory and no " *
          "tests/*/$(name)_test.jl. Try one of: " *
          string(filter(isdir, readdir(joinpath(@__DIR__, "tests"); join=false))))
end

if isempty(ARGS)
    include("general_routine.jl")
else
    @testset "selected ($(join(ARGS, ", ")))" begin
        for name in ARGS
            _run_test_target(name)
        end
    end
end





