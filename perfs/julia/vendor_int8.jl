# =============================================================================
# perfs/julia/vendor_int8.jl
# =============================================================================
# Vendor baselines for the matvec/vecmat perf comparison, dispatched by dtype.
#
# rocBLAS/cuBLAS have NO integer GEMV, so for 1-byte types (Int8/UInt8) we
# express the gemv as a single-column (N=1) GEMM via `rocblas_gemm_ex`
# (int8 inputs → int32 accumulate) — the only vendor int8 path. Float types
# keep the native gemv (`A*x` / `x'*A` → rocBLAS s/d gemv or cuBLAS).
#
# AMD-only for the int8 path; on CUDA/other the float branch is always taken
# (there is no int8-gemv comparison target there in this harness).
# =============================================================================

if @isdefined(AMDGPU)
    const _RB       = AMDGPU.rocBLAS
    const _RB_NONE  = _RB.rocblas_operation_none
    const _RB_TRANS = _RB.rocblas_operation_transpose
    const _RB_I8    = _RB.rocblas_datatype_i8_r
    const _RB_I32   = _RB.rocblas_datatype_i32_r
    const _RB_ALGO  = _RB.rocblas_gemm_algo_standard

    # Same-byte-size reinterpret so UInt8 arrays feed the i8_r path unchanged
    # (values are irrelevant to a memory-bound timing comparison).
    _as_i8(A) = eltype(A) === Int8 ? A : reinterpret(Int8, A)

    # matvec gemv: C(n×1) = A(n×p) · x(p)    (transA = none, m=n, N=1, k=p)
    function _gemmex_matvec!(C::AMDGPU.ROCArray{Int32}, A, x)
        Ai = _as_i8(A); xi = _as_i8(x); m, k = size(A)
        al = Ref(Int32(1)); bt = Ref(Int32(0)); h = _RB.handle()
        _RB.rocblas_gemm_ex_64(h, _RB_NONE, _RB_NONE, Int64(m), Int64(1), Int64(k),
            al, pointer(Ai), _RB_I8, Int64(m), pointer(xi), _RB_I8, Int64(k),
            bt, pointer(C), _RB_I32, Int64(m), pointer(C), _RB_I32, Int64(m),
            _RB_I32, _RB_ALGO, Int64(0), UInt32(0))
        return nothing
    end

    # vecmat gevm: C(p×1) = Aᵀ(p×n) · x(n)   (transA = transpose, m=p, N=1, k=n)
    function _gemmex_vecmat!(C::AMDGPU.ROCArray{Int32}, A, x)
        Ai = _as_i8(A); xi = _as_i8(x); n, p = size(A)
        al = Ref(Int32(1)); bt = Ref(Int32(0)); h = _RB.handle()
        _RB.rocblas_gemm_ex_64(h, _RB_TRANS, _RB_NONE, Int64(p), Int64(1), Int64(n),
            al, pointer(Ai), _RB_I8, Int64(n), pointer(xi), _RB_I8, Int64(n),
            bt, pointer(C), _RB_I32, Int64(p), pointer(C), _RB_I32, Int64(p),
            _RB_I32, _RB_ALGO, Int64(0), UInt32(0))
        return nothing
    end
end

# (method_label, baseline_call) for the vendor matvec/vecmat baseline. 1-byte
# types on AMD → gemm_ex; everything else → native gemv (the harness's prior
# behavior, label unchanged so existing CSVs/plots stay consistent).
_float_vendor_label() = has_cuda() ? "cuBLAS" : "LinearAlgebra"

function vendor_matvec_baseline(::Type{T}, A, x) where T
    if sizeof(T) == 1 && has_roc()
        C = AMDGPU.zeros(Int32, size(A, 1), 1)
        return ("rocBLAS gemm_ex", () -> _gemmex_matvec!(C, A, x))
    end
    return (_float_vendor_label(), () -> A * x)
end

function vendor_vecmat_baseline(::Type{T}, A, x) where T
    if sizeof(T) == 1 && has_roc()
        C = AMDGPU.zeros(Int32, size(A, 2), 1)
        return ("rocBLAS gemm_ex", () -> _gemmex_vecmat!(C, A, x))
    end
    return (_float_vendor_label(), () -> x' * A)
end

# Type-safe RHS vector: a 1:L ramp overflows UInt8 (InexactError at L>255);
# for 1-byte types use ones (values don't affect memory-bound timing).
_bench_vec(::Type{T}, L::Int) where T =
    sizeof(T) == 1 ? fill!(AT{T}(undef, L), one(T)) : AT{T}(1:L)
