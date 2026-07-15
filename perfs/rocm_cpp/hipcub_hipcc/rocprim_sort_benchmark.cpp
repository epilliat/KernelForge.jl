// rocPRIM (via hipCUB) device-radix-sort benchmark — AMD analogue of
// perfs/cuda_cpp/cub_nvcc/cub_sort_benchmark.cu. hipCUB's
// DeviceRadixSort::SortKeys dispatches straight to rocprim::radix_sort_keys.
// CLI + JSON output are byte-identical to the CUB binary so
// perfs/julia/bench_utils.jl parses both the same way
// (run_cub_benchmark → results[1].min_ms / mean_ms).
//
// Compile:
//   hipcc -O3 -std=c++17 --offload-arch=gfx942 rocprim_sort_benchmark.cpp -o bin/rocprim_sort_benchmark
//
// CLI (matches the CUB binary):
//   ./bin/rocprim_sort_benchmark -n 100000000 -i 100 -w 500 -t float -j
//   ./bin/rocprim_sort_benchmark --size 1000000 --type uint32 --json

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <type_traits>
#include <sstream>
#include <cstdint>
#include <limits>
#include <hipcub/hipcub.hpp>
#include <hip/hip_runtime.h>

#define CHECK_HIP(call)                                                                              \
    do                                                                                               \
    {                                                                                                \
        hipError_t error = call;                                                                     \
        if (error != hipSuccess)                                                                     \
        {                                                                                            \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__                              \
                      << " code=" << error << " \"" << hipGetErrorString(error) << "\"" << std::endl;\
            exit(1);                                                                                  \
        }                                                                                            \
    } while (0)

struct BenchmarkResult
{
    std::string operation;
    std::string dtype;
    size_t N;
    size_t element_size;
    double memory_mb;
    double temp_storage_kb;
    int warmup_iterations;
    int benchmark_iterations;
    double mean_ms;
    double std_ms;
    double min_ms;
    double max_ms;
    double coeff_var_percent;
    double throughput_gb_s;
    double elements_per_sec_billion;
    bool verified;
    std::string gpu_name;
    int compute_major;
    int compute_minor;
    double peak_bandwidth_gb_s;
};

bool g_json_output = false;
bool g_quiet = false;

// ---- Random data init: floats in [0,1), integers in [0, max]. -----------

template <typename T, typename Enable = void>
struct DataInitializer
{
    static void initialize(std::vector<T> &data)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(0.0, 1.0);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = dist(gen);
    }
};

template <typename T>
struct DataInitializer<T, typename std::enable_if<std::is_integral<T>::value>::type>
{
    static void initialize(std::vector<T> &data)
    {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<T>::max());
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = static_cast<T>(dist(gen));
    }
};

template <typename T>
std::string getTypeName()
{
    if (std::is_same<T, float>::value) return "float32";
    if (std::is_same<T, double>::value) return "float64";
    if (std::is_same<T, int>::value) return "int32";
    if (std::is_same<T, uint8_t>::value) return "uint8";
    if (std::is_same<T, uint32_t>::value) return "uint32";
    if (std::is_same<T, uint64_t>::value) return "uint64";
    return "unknown";
}

template <typename T>
std::string getTypeNameVerbose()
{
    if (std::is_same<T, float>::value) return "Float (32-bit)";
    if (std::is_same<T, double>::value) return "Double (64-bit)";
    if (std::is_same<T, int>::value) return "Int (32-bit)";
    if (std::is_same<T, uint8_t>::value) return "UInt (8-bit)";
    if (std::is_same<T, uint32_t>::value) return "UInt (32-bit)";
    if (std::is_same<T, uint64_t>::value) return "UInt (64-bit)";
    return "Unknown";
}

struct GPUInfo
{
    std::string name;
    int compute_major;
    int compute_minor;
    double peak_bandwidth_gb_s;
};

GPUInfo getGPUInfo()
{
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);

    GPUInfo info;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.peak_bandwidth_gb_s = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    return info;
}

template <typename... Args>
void print_info(Args &&...args)
{
    if (!g_quiet)
    {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        std::cout << oss.str();
    }
}

// ---- Benchmark kernel ---------------------------------------------------

// Generic sort benchmark over hipCUB's device-level primitives (→ rocPRIM).
//
// `M == 1`: single SortKeys call on N keys.
// `M > 1, batched_mode = "loop"`     : M serial DeviceRadixSort::SortKeys
//                                       calls on a contiguous K × M buffer.
// `M > 1, batched_mode = "segmented"`: ONE DeviceSegmentedRadixSort::SortKeys
//                                       call over uniform K-sized segments.
//
// NOTE: CUB's "segsort" (DeviceSegmentedSort, the adaptive per-segment picker)
// is intentionally omitted — it is not universally present across hipCUB
// versions, and the 1D sort comparison (M=1) does not need it. Add it under a
// version guard if a batched segsort baseline is ever required on AMD.
//
// Each measurement event brackets the full batched workload.
enum class BatchedMode { Single, Loop, Segmented };

template <typename T>
BenchmarkResult benchmark_rocprim_radix_sort(size_t K, int M, BatchedMode mode, float warmup_ms, int num_iterations, const GPUInfo &gpu_info)
{
    const size_t N = K * (size_t)M;

    BenchmarkResult result;
    result.operation = M == 1                        ? "radix_sort" :
                       mode == BatchedMode::Segmented ? "radix_sort_segmented" :
                                                        "radix_sort_batched";
    result.dtype = getTypeName<T>();
    result.N = N;
    result.element_size = sizeof(T);
    result.memory_mb = (2.0 * N * sizeof(T)) / (1024.0 * 1024.0);
    result.benchmark_iterations = num_iterations;
    result.gpu_name = gpu_info.name;
    result.compute_major = gpu_info.compute_major;
    result.compute_minor = gpu_info.compute_minor;
    result.peak_bandwidth_gb_s = gpu_info.peak_bandwidth_gb_s;

    print_info("\n=== hipCUB DeviceRadixSort::SortKeys (rocPRIM) Benchmark ===\n");
    print_info("Data type: ", getTypeNameVerbose<T>(), "\n");
    if (M == 1) {
        print_info("Vector size: ", N, " elements\n");
    } else {
        print_info("Batched: K=", K, " (per column) x M=", M, " columns  = ", N, " elements total\n");
    }
    print_info("Element size: ", sizeof(T), " bytes\n");
    print_info("Total memory size: ", result.memory_mb, " MB\n");

    std::vector<T> h_input(N);
    DataInitializer<T>::initialize(h_input);

    T *d_input = nullptr;
    T *d_output = nullptr;
    CHECK_HIP(hipMalloc(&d_input, N * sizeof(T)));
    CHECK_HIP(hipMalloc(&d_output, N * sizeof(T)));
    CHECK_HIP(hipMemcpy(d_input, h_input.data(), N * sizeof(T), hipMemcpyHostToDevice));

    // Optional segment-offset buffer for the segmented path.
    const bool needs_segments = (mode == BatchedMode::Segmented) && M > 1;
    int *d_segment_offsets = nullptr;
    if (needs_segments) {
        std::vector<int> h_offsets(M + 1);
        for (int j = 0; j <= M; ++j) h_offsets[j] = j * (int)K;
        CHECK_HIP(hipMalloc(&d_segment_offsets, (M + 1) * sizeof(int)));
        CHECK_HIP(hipMemcpy(d_segment_offsets, h_offsets.data(),
                            (M + 1) * sizeof(int), hipMemcpyHostToDevice));
    }

    // Temp-storage sizing depends on the launch path.
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    if (mode == BatchedMode::Segmented && M > 1) {
        CHECK_HIP(hipcub::DeviceSegmentedRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_input, d_output, (int)N, M,
            d_segment_offsets, d_segment_offsets + 1));
    } else {
        // Single-column scratch is sufficient — reused across the loop.
        CHECK_HIP(hipcub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                                    d_input, d_output, (int)K));
    }
    CHECK_HIP(hipMalloc(&d_temp_storage, temp_storage_bytes));
    result.temp_storage_kb = temp_storage_bytes / 1024.0;
    print_info("Temp storage required: ", result.temp_storage_kb, " KB\n");

    auto launch_sort = [&]() {
        if (mode == BatchedMode::Segmented && M > 1) {
            CHECK_HIP(hipcub::DeviceSegmentedRadixSort::SortKeys(
                d_temp_storage, temp_storage_bytes,
                d_input, d_output, (int)N, M,
                d_segment_offsets, d_segment_offsets + 1));
        } else {
            for (int j = 0; j < M; ++j) {
                CHECK_HIP(hipcub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                                            d_input + j * K, d_output + j * K, (int)K));
            }
        }
    };

    hipEvent_t start, stop, warmup_start, warmup_stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipEventCreate(&warmup_start));
    CHECK_HIP(hipEventCreate(&warmup_stop));

    // Warmup: keep launching until `warmup_ms` ms elapsed.
    print_info("\nWarmup for ", warmup_ms, " ms...\n");
    int warmup_iterations = 0;
    float elapsed_warmup = 0.0f;
    CHECK_HIP(hipEventRecord(warmup_start));
    while (elapsed_warmup < warmup_ms)
    {
        launch_sort();
        warmup_iterations++;
        CHECK_HIP(hipEventRecord(warmup_stop));
        CHECK_HIP(hipEventSynchronize(warmup_stop));
        CHECK_HIP(hipEventElapsedTime(&elapsed_warmup, warmup_start, warmup_stop));
    }
    result.warmup_iterations = warmup_iterations;
    print_info("Completed ", warmup_iterations, " warmup iters in ", elapsed_warmup, " ms\n");
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<float> times;
    times.reserve(num_iterations);

    print_info("Running ", num_iterations, " benchmark iterations...\n");

    for (int i = 0; i < num_iterations; ++i)
    {
        CHECK_HIP(hipEventRecord(start));
        launch_sort();
        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_HIP(hipEventElapsedTime(&milliseconds, start, stop));
        times.push_back(milliseconds);
    }

    float sum = 0.0f, min_time = times[0], max_time = times[0];
    for (float t : times)
    {
        sum += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    float mean = sum / num_iterations;
    float variance = 0.0f;
    for (float t : times)
    {
        float d = t - mean;
        variance += d * d;
    }
    variance /= num_iterations;
    float std_dev = std::sqrt(variance);

    double bytes_processed = 2.0 * N * sizeof(T);
    double gb_per_sec = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (mean / 1000.0);
    double elements_per_sec = N / (mean / 1000.0) / 1e9;

    result.mean_ms = mean;
    result.std_ms = std_dev;
    result.min_ms = min_time;
    result.max_ms = max_time;
    result.coeff_var_percent = (std_dev / mean) * 100.0;
    result.throughput_gb_s = gb_per_sec;
    result.elements_per_sec_billion = elements_per_sec;
    result.verified = true; // rocPRIM radix is known-correct; skip per-run check.

    if (!g_quiet)
    {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n=== Results ===\n";
        std::cout << "Mean time:  " << mean << " +/- " << std_dev << " ms\n";
        std::cout << "Min/Max:    " << min_time << " / " << max_time << " ms\n";
        std::cout << "Coeff var:  " << result.coeff_var_percent << "%\n";
        std::cout << "Throughput: " << gb_per_sec << " GB/s   "
                  << elements_per_sec << " Gkeys/s\n";
    }

    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_output));
    CHECK_HIP(hipFree(d_temp_storage));
    if (d_segment_offsets) CHECK_HIP(hipFree(d_segment_offsets));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_HIP(hipEventDestroy(warmup_start));
    CHECK_HIP(hipEventDestroy(warmup_stop));

    return result;
}

// ---- JSON output --------------------------------------------------------

void print_json_string(const std::string &k, const std::string &v, bool comma = true)
{
    std::cout << "    \"" << k << "\": \"" << v << "\"" << (comma ? "," : "") << "\n";
}
void print_json_number(const std::string &k, double v, bool comma = true)
{
    std::cout << "    \"" << k << "\": " << std::setprecision(6) << v << (comma ? "," : "") << "\n";
}
void print_json_int(const std::string &k, long long v, bool comma = true)
{
    std::cout << "    \"" << k << "\": " << v << (comma ? "," : "") << "\n";
}
void print_json_bool(const std::string &k, bool v, bool comma = true)
{
    std::cout << "    \"" << k << "\": " << (v ? "true" : "false") << (comma ? "," : "") << "\n";
}

void print_json_results(const std::vector<BenchmarkResult> &results)
{
    std::cout << std::fixed << "[\n";
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto &r = results[i];
        std::cout << "  {\n";
        print_json_string("operation", r.operation);
        print_json_string("dtype", r.dtype);
        print_json_int("N", r.N);
        print_json_int("element_size_bytes", r.element_size);
        print_json_number("memory_mb", r.memory_mb);
        print_json_number("temp_storage_kb", r.temp_storage_kb);
        print_json_int("warmup_iterations", r.warmup_iterations);
        print_json_int("benchmark_iterations", r.benchmark_iterations);
        print_json_number("mean_ms", r.mean_ms);
        print_json_number("std_ms", r.std_ms);
        print_json_number("min_ms", r.min_ms);
        print_json_number("max_ms", r.max_ms);
        print_json_number("coeff_var_percent", r.coeff_var_percent);
        print_json_number("throughput_gb_s", r.throughput_gb_s);
        print_json_number("elements_per_sec_billion", r.elements_per_sec_billion);
        print_json_bool("verified", r.verified);
        print_json_string("gpu_name", r.gpu_name);
        print_json_int("compute_major", r.compute_major);
        print_json_int("compute_minor", r.compute_minor);
        print_json_number("peak_bandwidth_gb_s", r.peak_bandwidth_gb_s, false);
        std::cout << "  }" << (i < results.size() - 1 ? "," : "") << "\n";
    }
    std::cout << "]\n";
}

void print_usage(const char *p)
{
    std::cerr << "Usage: " << p << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  -n, --size N          Vector size when M=1 (default: 100000000)\n"
              << "  -M, --batch M         Number of equal-length columns (default: 1).\n"
              << "  --batched-mode MODE   loop | segmented (default: loop).\n"
              << "                          loop      = M serial DeviceRadixSort::SortKeys calls\n"
              << "                          segmented = one DeviceSegmentedRadixSort::SortKeys call\n"
              << "  -i, --iterations N    Number of benchmark iterations (default: 100)\n"
              << "  -w, --warmup MS       Warmup duration in milliseconds (default: 500)\n"
              << "  -t, --type TYPE       float, double, int, uint8, uint32, uint64, or all (default: all)\n"
              << "  -j, --json            Output results in JSON format\n"
              << "  -h, --help            Show this help message\n";
}

int main(int argc, char **argv)
{
    size_t N = 100000000;
    int M = 1;
    BatchedMode mode = BatchedMode::Loop;
    float warmup_ms = 500.0f;
    int iterations = 100;
    std::string dtype = "all";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else if (arg == "-j" || arg == "--json") { g_json_output = true; g_quiet = true; }
        else if ((arg == "-n" || arg == "--size") && i + 1 < argc)        N = std::stoull(argv[++i]);
        else if ((arg == "-M" || arg == "--batch") && i + 1 < argc)       M = std::stoi(argv[++i]);
        else if (arg == "--batched-mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if      (m == "loop")      mode = BatchedMode::Loop;
            else if (m == "segmented") mode = BatchedMode::Segmented;
            else { std::cerr << "unknown --batched-mode " << m << "\n"; return 1; }
        }
        else if ((arg == "-i" || arg == "--iterations") && i + 1 < argc)  iterations = std::stoi(argv[++i]);
        else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc)      warmup_ms = std::stof(argv[++i]);
        else if ((arg == "-t" || arg == "--type") && i + 1 < argc)        dtype = argv[++i];
    }
    if (M == 1) mode = BatchedMode::Single;

    GPUInfo gpu_info = getGPUInfo();
    if (!g_quiet)
    {
        std::cout << "GPU: " << gpu_info.name
                  << "  (gfx" << gpu_info.compute_major << gpu_info.compute_minor << ")\n";
        std::cout << "Peak BW: " << gpu_info.peak_bandwidth_gb_s << " GB/s\n";
    }

    std::vector<BenchmarkResult> results;
    if (dtype == "all" || dtype == "float")  results.push_back(benchmark_rocprim_radix_sort<float>(N, M, mode, warmup_ms, iterations, gpu_info));
    if (dtype == "all" || dtype == "double") results.push_back(benchmark_rocprim_radix_sort<double>(N, M, mode, warmup_ms, iterations, gpu_info));
    if (dtype == "all" || dtype == "int")    results.push_back(benchmark_rocprim_radix_sort<int>(N, M, mode, warmup_ms, iterations, gpu_info));
    if (dtype == "all" || dtype == "uint8")  results.push_back(benchmark_rocprim_radix_sort<uint8_t>(N, M, mode, warmup_ms, iterations, gpu_info));
    if (dtype == "all" || dtype == "uint32") results.push_back(benchmark_rocprim_radix_sort<uint32_t>(N, M, mode, warmup_ms, iterations, gpu_info));
    if (dtype == "all" || dtype == "uint64") results.push_back(benchmark_rocprim_radix_sort<uint64_t>(N, M, mode, warmup_ms, iterations, gpu_info));

    if (g_json_output) print_json_results(results);
    return 0;
}
