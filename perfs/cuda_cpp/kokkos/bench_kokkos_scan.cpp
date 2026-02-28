/**
 * Kokkos Inclusive Scan Benchmark with JSON Output
 * Uses Kokkos::parallel_scan built-in function
 * Compatible with CUB benchmark JSON format for Julia integration
 */

#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <string>
#include <type_traits>

#define CUDA_CHECK(call)                                                 \
    do                                                                   \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                          \
        {                                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(1);                                                     \
        }                                                                \
    } while (0)

// Global flags
bool g_json_output = false;
bool g_quiet = false;

// Struct to hold benchmark results (matching CUB format)
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
    std::string gpu_name;
    int compute_major;
    int compute_minor;
    double peak_bandwidth_gb_s;
    bool verified;
};

// GPU info struct
struct GPUInfo
{
    std::string name;
    int compute_major;
    int compute_minor;
    double peak_bandwidth_gb_s;
};

// Get GPU info
GPUInfo getGPUInfo()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    GPUInfo info;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    // Peak bandwidth = memory_clock (Hz) * bus_width (bits) * 2 (DDR) / 8 (bits to bytes) / 1e9 (to GB)
    info.peak_bandwidth_gb_s = (prop.memoryClockRate * 1000.0) * (prop.memoryBusWidth / 8.0) * 2.0 / 1e9;

    return info;
}

// Helper to get type name as string
template <typename T>
std::string getTypeName()
{
    if (std::is_same<T, float>::value)
        return "float32";
    if (std::is_same<T, double>::value)
        return "float64";
    if (std::is_same<T, int>::value)
        return "int32";
    if (std::is_same<T, uint64_t>::value)
        return "uint64";
    if (std::is_same<T, uint8_t>::value)
        return "uint8";
    return "unknown";
}

// Escape string for JSON
std::string escapeJson(const std::string &s)
{
    std::string result;
    for (char c : s)
    {
        switch (c)
        {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            result += c;
        }
    }
    return result;
}

// Print results as JSON
void print_json_results(const std::vector<BenchmarkResult> &results)
{
    std::cout << "[\n";
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto &r = results[i];
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  {\n"
                  << "    \"operation\": \"" << escapeJson(r.operation) << "\",\n"
                  << "    \"dtype\": \"" << escapeJson(r.dtype) << "\",\n"
                  << "    \"N\": " << r.N << ",\n"
                  << "    \"element_size\": " << r.element_size << ",\n"
                  << "    \"memory_mb\": " << r.memory_mb << ",\n"
                  << "    \"temp_storage_kb\": " << r.temp_storage_kb << ",\n"
                  << "    \"warmup_iterations\": " << r.warmup_iterations << ",\n"
                  << "    \"benchmark_iterations\": " << r.benchmark_iterations << ",\n"
                  << "    \"mean_ms\": " << r.mean_ms << ",\n"
                  << "    \"std_ms\": " << r.std_ms << ",\n"
                  << "    \"min_ms\": " << r.min_ms << ",\n"
                  << "    \"max_ms\": " << r.max_ms << ",\n"
                  << "    \"coeff_var_percent\": " << r.coeff_var_percent << ",\n"
                  << "    \"throughput_gb_s\": " << r.throughput_gb_s << ",\n"
                  << "    \"elements_per_sec_billion\": " << r.elements_per_sec_billion << ",\n"
                  << "    \"gpu_name\": \"" << escapeJson(r.gpu_name) << "\",\n"
                  << "    \"compute_major\": " << r.compute_major << ",\n"
                  << "    \"compute_minor\": " << r.compute_minor << ",\n"
                  << "    \"peak_bandwidth_gb_s\": " << r.peak_bandwidth_gb_s << ",\n"
                  << "    \"verified\": " << (r.verified ? "true" : "false") << "\n"
                  << "  }" << (i < results.size() - 1 ? "," : "") << "\n";
    }
    std::cout << "]\n";
}

template <typename T>
BenchmarkResult benchmark_kokkos_scan(size_t N, double warmup_ms, int num_iterations, const GPUInfo &gpu_info)
{
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = ExecutionSpace::memory_space;
    using ViewType = Kokkos::View<T *, MemorySpace>;

    BenchmarkResult result;
    result.operation = "inclusive_scan";
    result.dtype = getTypeName<T>();
    result.N = N;
    result.element_size = sizeof(T);
    result.memory_mb = (N * sizeof(T)) / (1024.0 * 1024.0);
    result.temp_storage_kb = 0.0; // Kokkos manages this internally
    result.gpu_name = gpu_info.name;
    result.compute_major = gpu_info.compute_major;
    result.compute_minor = gpu_info.compute_minor;
    result.peak_bandwidth_gb_s = gpu_info.peak_bandwidth_gb_s;
    result.benchmark_iterations = num_iterations;

    if (!g_quiet)
    {
        std::cout << "Vector size: " << N << " elements" << std::endl;
        std::cout << "Data type: " << result.dtype << " (" << sizeof(T) << " bytes)" << std::endl;
        std::cout << "Memory size: " << result.memory_mb << " MB" << std::endl;
    }

    // Allocate Kokkos views
    ViewType d_input("input", N);
    ViewType d_output("output", N);

    // Initialize with random data
    {
        auto h_input = Kokkos::create_mirror_view(d_input);
        std::mt19937 gen(42);

        if constexpr (std::is_floating_point<T>::value)
        {
            std::uniform_real_distribution<T> dist(0.0, 1.0);
            for (size_t i = 0; i < N; ++i)
            {
                h_input(i) = dist(gen);
            }
        }
        else
        {
            std::uniform_int_distribution<int> dist(0, 100);
            for (size_t i = 0; i < N; ++i)
            {
                h_input(i) = static_cast<T>(dist(gen));
            }
        }
        Kokkos::deep_copy(d_input, h_input);
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup phase - run for specified duration
    if (!g_quiet)
    {
        std::cout << "Warming up for " << warmup_ms << " ms..." << std::endl;
    }

    auto warmup_start = std::chrono::high_resolution_clock::now();
    int warmup_count = 0;
    while (true)
    {
        Kokkos::parallel_scan(
            "inclusive_scan",
            Kokkos::RangePolicy<ExecutionSpace>(0, N),
            KOKKOS_LAMBDA(const size_t j, T &update, const bool final) {
                update += d_input(j);
                if (final)
                    d_output(j) = update;
            });
        Kokkos::fence();
        warmup_count++;

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - warmup_start).count();
        if (elapsed_ms >= warmup_ms)
            break;
    }
    result.warmup_iterations = warmup_count;

    if (!g_quiet)
    {
        std::cout << "Warmup complete (" << warmup_count << " iterations)" << std::endl;
        std::cout << "Running " << num_iterations << " benchmark iterations..." << std::endl;
    }

    // Benchmark iterations
    std::vector<float> times(num_iterations);

    for (int i = 0; i < num_iterations; ++i)
    {
        CUDA_CHECK(cudaEventRecord(start));

        Kokkos::parallel_scan(
            "inclusive_scan",
            Kokkos::RangePolicy<ExecutionSpace>(0, N),
            KOKKOS_LAMBDA(const size_t j, T &update, const bool final) {
                update += d_input(j);
                if (final)
                    d_output(j) = update;
            });

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[i] = ms;
    }

    // Calculate statistics
    double sum = 0.0, sum_sq = 0.0;
    double min_time = times[0], max_time = times[0];

    for (float t : times)
    {
        sum += t;
        sum_sq += t * t;
        min_time = std::min(min_time, (double)t);
        max_time = std::max(max_time, (double)t);
    }

    result.mean_ms = sum / num_iterations;
    result.std_ms = std::sqrt((sum_sq / num_iterations) - (result.mean_ms * result.mean_ms));
    result.min_ms = min_time;
    result.max_ms = max_time;
    result.coeff_var_percent = (result.std_ms / result.mean_ms) * 100.0;

    // Calculate throughput (read N elements, write N elements)
    double bytes_processed = 2 * N * sizeof(T);
    result.throughput_gb_s = (bytes_processed / (result.mean_ms / 1000.0)) / 1e9;
    result.elements_per_sec_billion = (N / (result.mean_ms / 1000.0)) / 1e9;

    // Verify result
    result.verified = true;
    {
        auto h_input = Kokkos::create_mirror_view(d_input);
        auto h_output = Kokkos::create_mirror_view(d_output);
        Kokkos::deep_copy(h_input, d_input);
        Kokkos::deep_copy(h_output, d_output);

        T expected = T(0);
        for (size_t i = 0; i < std::min(N, size_t(1000)); ++i)
        {
            expected += h_input(i);
            if constexpr (std::is_floating_point<T>::value)
            {
                if (std::abs(h_output(i) - expected) > 1e-3 * std::abs(expected) + 1e-6)
                {
                    result.verified = false;
                    break;
                }
            }
            else
            {
                if (h_output(i) != expected)
                {
                    result.verified = false;
                    break;
                }
            }
        }
    }

    if (!g_quiet)
    {
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Mean time: " << std::fixed << std::setprecision(4) << result.mean_ms << " ms" << std::endl;
        std::cout << "  Std dev:   " << result.std_ms << " ms" << std::endl;
        std::cout << "  Min time:  " << result.min_ms << " ms" << std::endl;
        std::cout << "  Max time:  " << result.max_ms << " ms" << std::endl;
        std::cout << "  CV:        " << std::setprecision(2) << result.coeff_var_percent << "%" << std::endl;
        std::cout << "\nPerformance:" << std::endl;
        std::cout << "  Throughput:    " << std::setprecision(2) << result.throughput_gb_s << " GB/s" << std::endl;
        std::cout << "  Elements/sec:  " << result.elements_per_sec_billion << " billion" << std::endl;
        std::cout << "  % of peak BW:  " << (result.throughput_gb_s / gpu_info.peak_bandwidth_gb_s * 100.0) << "%" << std::endl;
        std::cout << "\nVerification: " << (result.verified ? "PASSED" : "FAILED") << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return result;
}

void print_help()
{
    std::cout << "Kokkos Inclusive Scan Benchmark\n\n"
              << "Usage: kokkos_scan_benchmark [options]\n\n"
              << "Options:\n"
              << "  -n, --size N         Vector size (default: 100000000)\n"
              << "  -i, --iterations N   Number of benchmark iterations (default: 100)\n"
              << "  -w, --warmup MS      Warmup duration in milliseconds (default: 500)\n"
              << "  -t, --type TYPE      Data type: float, double, int, uint64, or all (default: all)\n"
              << "  -j, --json           Output results as JSON\n"
              << "  -q, --quiet          Suppress non-JSON output\n"
              << "  -h, --help           Show this help message\n\n"
              << "Examples:\n"
              << "  kokkos_scan_benchmark -n 1000000 -j\n"
              << "  kokkos_scan_benchmark --size 100000000 --iterations 200 --json\n"
              << "  kokkos_scan_benchmark -n 10000000 -t float -j\n";
}

int main(int argc, char *argv[])
{
    // Default parameters
    size_t N = 100000000;
    int iterations = 100;
    double warmup_ms = 500.0;
    std::string dtype = "all";

    // Parse arguments (before Kokkos::initialize to handle --help)
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            print_help();
            return 0;
        }
        else if (arg == "-j" || arg == "--json")
        {
            g_json_output = true;
            g_quiet = true;
        }
        else if (arg == "-q" || arg == "--quiet")
        {
            g_quiet = true;
        }
        else if ((arg == "-n" || arg == "--size") && i + 1 < argc)
        {
            N = std::stoull(argv[++i]);
        }
        else if ((arg == "-i" || arg == "--iterations") && i + 1 < argc)
        {
            iterations = std::stoi(argv[++i]);
        }
        else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc)
        {
            warmup_ms = std::stod(argv[++i]);
        }
        else if ((arg == "-t" || arg == "--type") && i + 1 < argc)
        {
            dtype = argv[++i];
        }
    }

    Kokkos::initialize(argc, argv);
    {
        // Get GPU info
        GPUInfo gpu_info = getGPUInfo();

        if (!g_quiet)
        {
            std::cout << "=== Kokkos Inclusive Scan Benchmark ===" << std::endl;
            std::cout << "GPU: " << gpu_info.name << std::endl;
            std::cout << "Compute Capability: " << gpu_info.compute_major << "." << gpu_info.compute_minor << std::endl;
            std::cout << "Peak Memory Bandwidth: " << gpu_info.peak_bandwidth_gb_s << " GB/s" << std::endl;
            std::cout << "\nKokkos configuration:" << std::endl;
            Kokkos::print_configuration(std::cout);
        }

        std::vector<BenchmarkResult> results;

        if (dtype == "all" || dtype == "float")
        {
            if (!g_quiet)
                std::cout << "\n### Float (32-bit) ###" << std::endl;
            results.push_back(benchmark_kokkos_scan<float>(N, warmup_ms, iterations, gpu_info));
        }

        if (dtype == "all" || dtype == "double")
        {
            if (!g_quiet)
                std::cout << "\n### Double (64-bit) ###" << std::endl;
            results.push_back(benchmark_kokkos_scan<double>(N, warmup_ms, iterations, gpu_info));
        }

        if (dtype == "all" || dtype == "int")
        {
            if (!g_quiet)
                std::cout << "\n### Int (32-bit) ###" << std::endl;
            results.push_back(benchmark_kokkos_scan<int>(N, warmup_ms, iterations, gpu_info));
        }

        if (dtype == "all" || dtype == "uint64")
        {
            if (!g_quiet)
                std::cout << "\n### UInt64 (64-bit) ###" << std::endl;
            results.push_back(benchmark_kokkos_scan<uint64_t>(N, warmup_ms, iterations, gpu_info));
        }

        // Output JSON if requested
        if (g_json_output)
        {
            print_json_results(results);
        }
    }
    Kokkos::finalize();
    return 0;
}

// Compilation command (adjust paths as needed):
// nvcc -x cu -std=c++17 -O3 -arch=sm_70 \
//   -I${KOKKOS_PATH}/include \
//   -L${KOKKOS_PATH}/lib -lkokkoscore \
//   kokkos_scan_benchmark.cpp -o kokkos_scan_benchmark
//
// Or with CMake/Makefile that links Kokkos properly
//
// Usage:
// ./kokkos_scan_benchmark --help
// ./kokkos_scan_benchmark -n 1000000 -j
// ./kokkos_scan_benchmark --size 100000000 --iterations 200 --json
// ./kokkos_scan_benchmark -n 10000000 -t float -j