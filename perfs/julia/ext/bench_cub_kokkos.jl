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
    result = JSON3.read(read(cmd, String))

    # Save to results folder next to current file
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    filename = "kokkos_scan_$(dtype_str)_N$(N)_iter$(iterations).json"
    open(joinpath(results_dir, filename), "w") do io
        JSON3.pretty(io, result)
    end

    return result
end

kokkos_exe::String = "$(@__DIR__)/../../cuda_cpp/kokkos/build/bench_kokkos_scan"
run_kokkos_scan_benchmark(kokkos_exe, dtype=Float64, N=10^9)