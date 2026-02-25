using JSON3
function run_kokkos_scan_benchmark(exe::String;
    N::Int=100_000_000,
    iterations::Int=100,
    warmup_ms::Real=500,
    dtype::Type=Float32
)
    dtype_str = if dtype == Float32
        "float"
    elseif dtype == Float64
        "double"
    elseif dtype == Int32
        "int"
    elseif dtype == UInt64
        "uint64"
    else
        error("Unsupported dtype: $dtype. Use Float32, Float64, Int32, or UInt64.")
    end

    exe_path = abspath(expanduser(exe))
    cmd = `$exe_path -n $N -i $iterations -w $warmup_ms -t $dtype_str -j`
    return JSON3.read(read(cmd, String))
end

kokkos_exe::String = "$(@__DIR__)/../../cuda_cpp/kokkos/build/bench_kokkos_scan"
run_kokkos_scan_benchmark(kokkos_exe, dtype=Float64)