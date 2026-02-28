#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <type_traits>
#include <sstream>
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
    bool verified;
    std::string gpu_name;
    int compute_major;
    int compute_minor;
    double peak_bandwidth_gb_s;
};

// Global flag for JSON output
bool g_json_output = false;
bool g_quiet = false; // Suppress non-JSON output when in JSON mode

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

// Specialization for integer types
template <typename T>
struct DataInitializer<T, typename std::enable_if<std::is_integral<T>::value && sizeof(T) <= 8>::type>
{
    static void initialize(std::vector<T> &data)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<typename std::conditional<sizeof(T) == 1, int, T>::type> dist(0, 100);
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = static_cast<T>(dist(gen));
        }
    }
};

// Get type name for display
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
    return "unknown";
}

template <typename T>
std::string getTypeNameVerbose()
{
    if (std::is_same<T, float>::value)
        return "Float (32-bit)";
    if (std::is_same<T, double>::value)
        return "Double (64-bit)";
    if (std::is_same<T, int>::value)
        return "Int (32-bit)";
    if (std::is_same<T, uint64_t>::value)
        return "UInt64 (64-bit)";
    return "Unknown";
}

// Helper function to compute absolute difference
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
compute_abs_diff(T a, T b)
{
    return std::abs(a - b);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type
compute_abs_diff(T a, T b)
{
    return std::abs(a - b);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value, T>::type
compute_abs_diff(T a, T b)
{
    return (a > b) ? (a - b) : (b - a);
}

// Get GPU info
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
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    GPUInfo info;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.peak_bandwidth_gb_s = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    return info;
}

// Print functions that respect quiet mode
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

void print_info_line()
{
    if (!g_quiet)
        std::cout << std::endl;
}

template <typename T>
BenchmarkResult benchmark_cub_inclusive_scan(size_t N, float warmup_ms, int num_iterations, const GPUInfo &gpu_info)
{
    BenchmarkResult result;
    result.operation = "inclusive_scan";
    result.dtype = getTypeName<T>();
    result.N = N;
    result.element_size = sizeof(T);
    result.memory_mb = (2.0 * N * sizeof(T)) / (1024.0 * 1024.0);
    result.benchmark_iterations = num_iterations;
    result.gpu_name = gpu_info.name;
    result.compute_major = gpu_info.compute_major;
    result.compute_minor = gpu_info.compute_minor;
    result.peak_bandwidth_gb_s = gpu_info.peak_bandwidth_gb_s;

    print_info("\n=== CUB Inclusive Scan (Cumulative Sum) Benchmark ===\n");
    print_info("Data type: ", getTypeNameVerbose<T>(), "\n");
    print_info("Vector size: ", N, " elements\n");
    print_info("Element size: ", sizeof(T), " bytes\n");
    print_info("Memory size (input): ", (N * sizeof(T)) / (1024.0 * 1024.0), " MB\n");
    print_info("Memory size (output): ", (N * sizeof(T)) / (1024.0 * 1024.0), " MB\n");
    print_info("Total memory size: ", result.memory_mb, " MB\n");

    // Allocate host memory
    std::vector<T> h_input(N);
    std::vector<T> h_output(N);

    // Initialize with random data
    DataInitializer<T>::initialize(h_input);

    // Allocate device memory
    T *d_input = nullptr;
    T *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(T)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    // Determine temporary storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_input, d_output, N));
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    result.temp_storage_kb = temp_storage_bytes / 1024.0;
    print_info("Temp storage required: ", result.temp_storage_kb, " KB\n");

    // Create CUDA events for timing
    cudaEvent_t start, stop, warmup_start, warmup_stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&warmup_start));
    CHECK_CUDA(cudaEventCreate(&warmup_stop));

    // Warmup runs
    print_info("\nPerforming warmup for ", warmup_ms, " ms...\n");
    int warmup_iterations = 0;
    float elapsed_warmup = 0.0f;

    CHECK_CUDA(cudaEventRecord(warmup_start));
    while (elapsed_warmup < warmup_ms)
    {
        CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                                 d_input, d_output, N));
        warmup_iterations++;

        CHECK_CUDA(cudaEventRecord(warmup_stop));
        CHECK_CUDA(cudaEventSynchronize(warmup_stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_warmup, warmup_start, warmup_stop));
    }
    result.warmup_iterations = warmup_iterations;
    print_info("Completed ", warmup_iterations, " warmup iterations in ", elapsed_warmup, " ms\n");
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark runs
    std::vector<float> times;
    times.reserve(num_iterations);

    print_info("Performing ", num_iterations, " benchmark iterations...\n");

    for (int i = 0; i < num_iterations; ++i)
    {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                                 d_input, d_output, N));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        times.push_back(milliseconds);
    }

    // Calculate statistics
    float sum = 0.0f;
    float min_time = times[0];
    float max_time = times[0];

    for (float time : times)
    {
        sum += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }

    float mean = sum / num_iterations;

    float variance = 0.0f;
    for (float time : times)
    {
        float diff = time - mean;
        variance += diff * diff;
    }
    variance /= num_iterations;
    float std_dev = std::sqrt(variance);

    // Calculate throughput
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

    // Verify result
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, N * sizeof(T), cudaMemcpyDeviceToHost));

    result.verified = true;
    if (N <= 1000000)
    {
        std::vector<T> expected(N);
        expected[0] = h_input[0];
        for (size_t i = 1; i < N; ++i)
        {
            expected[i] = expected[i - 1] + h_input[i];
        }

        size_t check_count = std::min(size_t(10), N / 2);
        T tolerance = std::is_floating_point<T>::value ? T(1e-3) : T(0);

        for (size_t i = 0; i < check_count; ++i)
        {
            T diff = compute_abs_diff(h_output[i], expected[i]);
            if (diff > tolerance)
            {
                result.verified = false;
                print_info("Verification failed at index ", i, ": expected ",
                           expected[i], ", got ", h_output[i], "\n");
                break;
            }
        }
        for (size_t i = N - check_count; i < N && result.verified; ++i)
        {
            T diff = compute_abs_diff(h_output[i], expected[i]);
            if (diff > tolerance)
            {
                result.verified = false;
                print_info("Verification failed at index ", i, ": expected ",
                           expected[i], ", got ", h_output[i], "\n");
                break;
            }
        }

        if (result.verified)
        {
            print_info("\nVerification: PASSED\n");
        }
    }
    else
    {
        print_info("\nVerification: Skipped (array too large)\n");
    }

    // Print results
    if (!g_quiet)
    {
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Mean time:          " << mean << " ± " << std_dev << " ms" << std::endl;
        std::cout << "Min time:           " << min_time << " ms" << std::endl;
        std::cout << "Max time:           " << max_time << " ms" << std::endl;
        std::cout << "Coefficient of var: " << result.coeff_var_percent << "%" << std::endl;
        std::cout << "\n=== Performance ===" << std::endl;
        std::cout << "Throughput:         " << gb_per_sec << " GB/s" << std::endl;
        std::cout << "Elements/sec:       " << elements_per_sec << " billion elements/s" << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp_storage));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(warmup_start));
    CHECK_CUDA(cudaEventDestroy(warmup_stop));

    return result;
}

template <typename T>
BenchmarkResult benchmark_cub_exclusive_scan(size_t N, float warmup_ms, int num_iterations, const GPUInfo &gpu_info)
{
    BenchmarkResult result;
    result.operation = "exclusive_scan";
    result.dtype = getTypeName<T>();
    result.N = N;
    result.element_size = sizeof(T);
    result.memory_mb = (2.0 * N * sizeof(T)) / (1024.0 * 1024.0);
    result.benchmark_iterations = num_iterations;
    result.gpu_name = gpu_info.name;
    result.compute_major = gpu_info.compute_major;
    result.compute_minor = gpu_info.compute_minor;
    result.peak_bandwidth_gb_s = gpu_info.peak_bandwidth_gb_s;

    print_info("\n=== CUB Exclusive Scan Benchmark ===\n");
    print_info("Data type: ", getTypeNameVerbose<T>(), "\n");
    print_info("Vector size: ", N, " elements\n");
    print_info("Element size: ", sizeof(T), " bytes\n");
    print_info("Memory size (input): ", (N * sizeof(T)) / (1024.0 * 1024.0), " MB\n");
    print_info("Memory size (output): ", (N * sizeof(T)) / (1024.0 * 1024.0), " MB\n");

    // Allocate host memory
    std::vector<T> h_input(N);
    std::vector<T> h_output(N);

    DataInitializer<T>::initialize(h_input);

    T *d_input = nullptr;
    T *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CHECK_CUDA(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_input, d_output, N));
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    result.temp_storage_kb = temp_storage_bytes / 1024.0;
    print_info("Temp storage required: ", result.temp_storage_kb, " KB\n");

    cudaEvent_t start, stop, warmup_start, warmup_stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&warmup_start));
    CHECK_CUDA(cudaEventCreate(&warmup_stop));

    print_info("\nPerforming warmup for ", warmup_ms, " ms...\n");
    int warmup_iterations = 0;
    float elapsed_warmup = 0.0f;

    CHECK_CUDA(cudaEventRecord(warmup_start));
    while (elapsed_warmup < warmup_ms)
    {
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                                 d_input, d_output, N));
        warmup_iterations++;

        CHECK_CUDA(cudaEventRecord(warmup_stop));
        CHECK_CUDA(cudaEventSynchronize(warmup_stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_warmup, warmup_start, warmup_stop));
    }
    result.warmup_iterations = warmup_iterations;
    print_info("Completed ", warmup_iterations, " warmup iterations in ", elapsed_warmup, " ms\n");
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(num_iterations);

    print_info("Performing ", num_iterations, " benchmark iterations...\n");

    for (int i = 0; i < num_iterations; ++i)
    {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                                 d_input, d_output, N));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        times.push_back(milliseconds);
    }

    float sum = 0.0f;
    float min_time = times[0];
    float max_time = times[0];

    for (float time : times)
    {
        sum += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }

    float mean = sum / num_iterations;

    float variance = 0.0f;
    for (float time : times)
    {
        float diff = time - mean;
        variance += diff * diff;
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
    result.verified = true; // No verification for exclusive scan in this version

    if (!g_quiet)
    {
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Mean time:          " << mean << " ± " << std_dev << " ms" << std::endl;
        std::cout << "Min time:           " << min_time << " ms" << std::endl;
        std::cout << "Max time:           " << max_time << " ms" << std::endl;
        std::cout << "Coefficient of var: " << result.coeff_var_percent << "%" << std::endl;
        std::cout << "\n=== Performance ===" << std::endl;
        std::cout << "Throughput:         " << gb_per_sec << " GB/s" << std::endl;
        std::cout << "Elements/sec:       " << elements_per_sec << " billion elements/s" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp_storage));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(warmup_start));
    CHECK_CUDA(cudaEventDestroy(warmup_stop));

    return result;
}

// JSON output functions
void print_json_string(const std::string &key, const std::string &value, bool comma = true)
{
    std::cout << "    \"" << key << "\": \"" << value << "\"" << (comma ? "," : "") << "\n";
}

void print_json_number(const std::string &key, double value, bool comma = true)
{
    std::cout << "    \"" << key << "\": " << std::setprecision(6) << value << (comma ? "," : "") << "\n";
}

void print_json_int(const std::string &key, long long value, bool comma = true)
{
    std::cout << "    \"" << key << "\": " << value << (comma ? "," : "") << "\n";
}

void print_json_bool(const std::string &key, bool value, bool comma = true)
{
    std::cout << "    \"" << key << "\": " << (value ? "true" : "false") << (comma ? "," : "") << "\n";
}

void print_json_results(const std::vector<BenchmarkResult> &results)
{
    std::cout << std::fixed;
    std::cout << "[\n";
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

void print_usage(const char *program_name)
{
    std::cerr << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  -n, --size N          Vector size (default: 100000000)\n"
              << "  -i, --iterations N    Number of benchmark iterations (default: 100)\n"
              << "  -w, --warmup MS       Warmup duration in milliseconds (default: 500)\n"
              << "  -m, --mode MODE       Scan mode: inclusive, exclusive, or both (default: inclusive)\n"
              << "  -t, --type TYPE       Data type: float, double, int, uint64, or all (default: all)\n"
              << "  -j, --json            Output results in JSON format\n"
              << "  -h, --help            Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " -n 1000000 -j\n"
              << "  " << program_name << " --size 100000000 --iterations 200 --json\n"
              << "  " << program_name << " -n 10000000 -m both -t float -j\n";
}

int main(int argc, char **argv)
{
    // Default parameters
    size_t N = 100000000;
    float warmup_ms = 500.0f;
    int iterations = 100;
    std::string mode = "inclusive";
    std::string dtype = "all";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "-j" || arg == "--json")
        {
            g_json_output = true;
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
            warmup_ms = std::stof(argv[++i]);
        }
        else if ((arg == "-m" || arg == "--mode") && i + 1 < argc)
        {
            mode = argv[++i];
        }
        else if ((arg == "-t" || arg == "--type") && i + 1 < argc)
        {
            dtype = argv[++i];
        }
        else
        {
            // Legacy positional argument support
            if (i == 1 && arg[0] != '-')
            {
                N = std::stoull(arg);
            }
            else if (i == 2 && arg[0] != '-')
            {
                iterations = std::stoi(arg);
            }
            else if (i == 3 && arg[0] != '-')
            {
                warmup_ms = std::stof(arg);
            }
            else if (i == 4 && arg[0] != '-')
            {
                mode = arg;
                if (mode == "ex")
                    mode = "exclusive";
            }
            else if (i == 5 && arg[0] != '-')
            {
                dtype = arg;
            }
        }
    }

    // Get GPU info
    GPUInfo gpu_info = getGPUInfo();

    if (!g_quiet)
    {
        std::cout << "GPU: " << gpu_info.name << std::endl;
        std::cout << "Compute Capability: " << gpu_info.compute_major << "." << gpu_info.compute_minor << std::endl;
        std::cout << "Peak Memory Bandwidth: " << gpu_info.peak_bandwidth_gb_s << " GB/s" << std::endl;
    }

    std::vector<BenchmarkResult> results;

    bool run_inclusive = (mode == "inclusive" || mode == "both");
    bool run_exclusive = (mode == "exclusive" || mode == "both");

    // Run inclusive scan benchmarks
    if (run_inclusive)
    {
        if (dtype == "all" || dtype == "float")
        {
            if (!g_quiet)
                std::cout << "\n### Float (32-bit) - Inclusive Scan ###" << std::endl;
            results.push_back(benchmark_cub_inclusive_scan<float>(N, warmup_ms, iterations, gpu_info));
        }

        if (dtype == "all" || dtype == "double")
        {
            if (!g_quiet)
                std::cout << "\n### Double (64-bit) - Inclusive Scan ###" << std::endl;
            results.push_back(benchmark_cub_inclusive_scan<double>(N, warmup_ms, iterations, gpu_info));
        }

        if (dtype == "all" || dtype == "int")
        {
            if (!g_quiet)
                std::cout << "\n### Int (32-bit) - Inclusive Scan ###" << std::endl;
            results.push_back(benchmark_cub_inclusive_scan<int>(N, warmup_ms, iterations, gpu_info));
        }

        if (dtype == "all" || dtype == "uint64")
        {
            if (!g_quiet)
                std::cout << "\n### UInt64 (64-bit) - Inclusive Scan ###" << std::endl;
            results.push_back(benchmark_cub_inclusive_scan<uint64_t>(N, warmup_ms, iterations, gpu_info));
        }
    }

    // Run exclusive scan benchmarks
    if (run_exclusive)
    {
        if (dtype == "all" || dtype == "float")
        {
            if (!g_quiet)
                std::cout << "\n### Float (32-bit) - Exclusive Scan ###" << std::endl;
            results.push_back(benchmark_cub_exclusive_scan<float>(N, warmup_ms, iterations, gpu_info));
        }

        if (dtype == "all" || dtype == "double")
        {
            if (!g_quiet)
                std::cout << "\n### Double (64-bit) - Exclusive Scan ###" << std::endl;
            results.push_back(benchmark_cub_exclusive_scan<double>(N, warmup_ms, iterations, gpu_info));
        }

        if (dtype == "all" || dtype == "int")
        {
            if (!g_quiet)
                std::cout << "\n### Int (32-bit) - Exclusive Scan ###" << std::endl;
            results.push_back(benchmark_cub_exclusive_scan<int>(N, warmup_ms, iterations, gpu_info));
        }

        if (dtype == "all" || dtype == "uint64")
        {
            if (!g_quiet)
                std::cout << "\n### UInt64 (64-bit) - Exclusive Scan ###" << std::endl;
            results.push_back(benchmark_cub_exclusive_scan<uint64_t>(N, warmup_ms, iterations, gpu_info));
        }
    }

    // Output JSON if requested
    if (g_json_output)
    {
        print_json_results(results);
    }

    return 0;
}

// Compilation command:
// nvcc -O3 -std=c++17 -arch=sm_70 cub_scan_benchmark.cu -o bin/cub_scan_benchmark
//
// Usage (new style with flags):
// ./cub_scan_benchmark --help
// ./cub_scan_benchmark -n 1000000 -j
// ./cub_scan_benchmark --size 100000000 --iterations 200 --json
// ./cub_scan_benchmark -n 10000000 -m both -t float -j
//
// Usage (legacy positional):
// ./cub_scan_benchmark [N] [iterations] [warmup_ms] [mode] [dtype]