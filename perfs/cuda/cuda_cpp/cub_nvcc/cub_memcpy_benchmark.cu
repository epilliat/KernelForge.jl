#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <type_traits>
#include <chrono>
#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t error = call;                                                                      \
        if (error != cudaSuccess)                                                                      \
        {                                                                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__                               \
                      << " code=" << error << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
            exit(1);                                                                                   \
        }                                                                                              \
    } while (0)

// Struct to hold benchmark results
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

// Global flags
bool g_json_output = false;
bool g_quiet = false;

// Helper to get GPU info
GPUInfo getGPUInfo()
{
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    GPUInfo info;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    // Peak bandwidth = memory_clock (Hz) * bus_width (bits) * 2 (DDR) / 8 (bits to bytes) / 1e9 (to GB)
    info.peak_bandwidth_gb_s = (prop.memoryClockRate * 1000.0) * (prop.memoryBusWidth / 8.0) * 2.0 / 1e9;

    return info;
}

// Helper template for data initialization - generic version for floating point
template <typename T, typename Enable = void>
struct DataInitializer
{
    static void initialize(std::vector<T> &data)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(0.0, 1.0);
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = dist(gen);
        }
    }
};

// Specialization for integer types (excluding uint8_t which needs special handling)
template <typename T>
struct DataInitializer<T, typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, uint8_t>::value>::type>
{
    static void initialize(std::vector<T> &data)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<T> dist(0, 100);
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = dist(gen);
        }
    }
};

// Specialization for uint8_t
template <>
struct DataInitializer<uint8_t, void>
{
    static void initialize(std::vector<uint8_t> &data)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, 255);
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = static_cast<uint8_t>(dist(gen));
        }
    }
};

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
BenchmarkResult benchmark_cub_memcpy(size_t N, double warmup_ms, int num_iterations, const GPUInfo &gpu_info)
{
    BenchmarkResult result;
    result.operation = "memcpy";
    result.dtype = getTypeName<T>();
    result.N = N;
    result.element_size = sizeof(T);
    result.memory_mb = (N * sizeof(T)) / (1024.0 * 1024.0);
    result.temp_storage_kb = 0.0; // No temp storage for memcpy
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
        std::cout << "Total bandwidth (R+W): " << (2 * result.memory_mb) << " MB" << std::endl;
    }

    // Allocate host memory
    std::vector<T> h_input(N);
    std::vector<T> h_output(N);

    // Initialize with random data
    DataInitializer<T>::initialize(h_input);

    // Allocate device memory
    T *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(T)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup phase - run for specified duration
    if (!g_quiet)
    {
        std::cout << "Warming up for " << warmup_ms << " ms..." << std::endl;
    }

    auto warmup_start = std::chrono::high_resolution_clock::now();
    int warmup_count = 0;
    while (true)
    {
        CHECK_CUDA(cudaMemcpy(d_output, d_input, N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
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
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(d_output, d_input, N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
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

    // Calculate throughput (read + write = 2 * N elements)
    double bytes_processed = 2.0 * N * sizeof(T);
    result.throughput_gb_s = (bytes_processed / (result.mean_ms / 1000.0)) / 1e9;
    result.elements_per_sec_billion = (N / (result.mean_ms / 1000.0)) / 1e9;

    // Verify result (check first and last elements)
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, N * sizeof(T), cudaMemcpyDeviceToHost));

    result.verified = true;
    size_t check_count = std::min(size_t(100), N);
    for (size_t i = 0; i < check_count; ++i)
    {
        if (h_output[i] != h_input[i])
        {
            result.verified = false;
            break;
        }
    }
    // Also check last few elements
    for (size_t i = N - std::min(size_t(100), N); i < N; ++i)
    {
        if (h_output[i] != h_input[i])
        {
            result.verified = false;
            break;
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

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return result;
}

void print_help()
{
    std::cout << "CUB Memory Copy Benchmark\n\n"
              << "Usage: cub_memcpy_benchmark [options]\n\n"
              << "Options:\n"
              << "  -n, --size N         Vector size (default: 100000000)\n"
              << "  -i, --iterations N   Number of benchmark iterations (default: 100)\n"
              << "  -w, --warmup MS      Warmup duration in milliseconds (default: 500)\n"
              << "  -t, --type TYPE      Data type: float, double, int, uint64, uint8, or all (default: all)\n"
              << "  -j, --json           Output results as JSON\n"
              << "  -q, --quiet          Suppress non-JSON output\n"
              << "  -h, --help           Show this help message\n\n"
              << "Examples:\n"
              << "  cub_memcpy_benchmark -n 1000000 -j\n"
              << "  cub_memcpy_benchmark --size 100000000 --iterations 200 --json\n"
              << "  cub_memcpy_benchmark -n 10000000 -t float -j\n"
              << "  cub_memcpy_benchmark -n 10000000 -t uint8 -j\n";
}

int main(int argc, char *argv[])
{
    // Default parameters
    size_t N = 100000000;
    int iterations = 100;
    double warmup_ms = 500.0;
    std::string dtype = "all";

    // Parse arguments
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
        else if (arg[0] != '-' && i == 1)
        {
            // Legacy positional argument for N
            N = std::stoull(arg);
        }
        else if (arg[0] != '-' && i == 2)
        {
            // Legacy positional argument for iterations
            iterations = std::stoi(arg);
        }
        else if (arg[0] != '-' && i == 3)
        {
            // Legacy positional argument for warmup_ms
            warmup_ms = std::stod(arg);
        }
    }

    // Get GPU info
    GPUInfo gpu_info = getGPUInfo();

    if (!g_quiet)
    {
        std::cout << "=== CUB Memory Copy Benchmark ===" << std::endl;
        std::cout << "GPU: " << gpu_info.name << std::endl;
        std::cout << "Compute Capability: " << gpu_info.compute_major << "." << gpu_info.compute_minor << std::endl;
        std::cout << "Peak Memory Bandwidth: " << gpu_info.peak_bandwidth_gb_s << " GB/s" << std::endl;
    }

    std::vector<BenchmarkResult> results;

    if (dtype == "all" || dtype == "uint8")
    {
        if (!g_quiet)
            std::cout << "\n### UInt8 (8-bit) ###" << std::endl;
        results.push_back(benchmark_cub_memcpy<uint8_t>(N, warmup_ms, iterations, gpu_info));
    }

    if (dtype == "all" || dtype == "float")
    {
        if (!g_quiet)
            std::cout << "\n### Float (32-bit) ###" << std::endl;
        results.push_back(benchmark_cub_memcpy<float>(N, warmup_ms, iterations, gpu_info));
    }

    if (dtype == "all" || dtype == "double")
    {
        if (!g_quiet)
            std::cout << "\n### Double (64-bit) ###" << std::endl;
        results.push_back(benchmark_cub_memcpy<double>(N, warmup_ms, iterations, gpu_info));
    }

    if (dtype == "all" || dtype == "int")
    {
        if (!g_quiet)
            std::cout << "\n### Int (32-bit) ###" << std::endl;
        results.push_back(benchmark_cub_memcpy<int>(N, warmup_ms, iterations, gpu_info));
    }

    if (dtype == "all" || dtype == "uint64")
    {
        if (!g_quiet)
            std::cout << "\n### UInt64 (64-bit) ###" << std::endl;
        results.push_back(benchmark_cub_memcpy<uint64_t>(N, warmup_ms, iterations, gpu_info));
    }

    // Output JSON if requested
    if (g_json_output)
    {
        print_json_results(results);
    }

    return 0;
}

// Compilation command:
// nvcc -O3 -std=c++17 -arch=sm_70 cub_memcpy_benchmark.cu -o bin/cub_memcpy_benchmark
//
// Usage (new style with flags):
// ./cub_memcpy_benchmark --help
// ./cub_memcpy_benchmark -n 1000000 -j
// ./cub_memcpy_benchmark --size 100000000 --iterations 200 --json
// ./cub_memcpy_benchmark -n 10000000 -t float -j
// ./cub_memcpy_benchmark -n 10000000 -t uint8 -j
//
// Usage (legacy positional):
// ./cub_memcpy_benchmark [N] [iterations] [warmup_ms]