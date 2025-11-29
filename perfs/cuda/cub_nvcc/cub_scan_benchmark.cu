#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <type_traits>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

// uint128_t support removed due to compatibility issues

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

// Specialization for integer types (but not uint128_t)
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
        return "Float (32-bit)";
    if (std::is_same<T, double>::value)
        return "Double (64-bit)";
    if (std::is_same<T, int>::value)
        return "Int (32-bit)";
    if (std::is_same<T, uint64_t>::value)
        return "UInt64 (64-bit)";
    return "Unknown";
}

// Helper function to compute absolute difference that works for both signed and unsigned types
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

template <typename T>
void benchmark_cub_inclusive_scan(size_t N, float warmup_ms = 500.0f, int num_iterations = 100)
{
    std::cout << "\n=== CUB Inclusive Scan (Cumulative Sum) Benchmark ===" << std::endl;
    std::cout << "Data type: " << getTypeName<T>() << std::endl;
    std::cout << "Vector size: " << N << " elements" << std::endl;
    std::cout << "Element size: " << sizeof(T) << " bytes" << std::endl;
    std::cout << "Memory size (input): " << (N * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Memory size (output): " << (N * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Total memory size: " << (2 * N * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;

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

    std::cout << "Temp storage required: " << temp_storage_bytes / 1024.0 << " KB" << std::endl;

    // Create CUDA events for timing
    cudaEvent_t start, stop, warmup_start, warmup_stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&warmup_start));
    CHECK_CUDA(cudaEventCreate(&warmup_stop));

    // Warmup runs for specified duration
    std::cout << "\nPerforming warmup for " << warmup_ms << " ms..." << std::endl;
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
    std::cout << "Completed " << warmup_iterations << " warmup iterations in " << elapsed_warmup << " ms" << std::endl;
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark runs
    std::vector<float> times;
    times.reserve(num_iterations);

    std::cout << "Performing " << num_iterations << " benchmark iterations..." << std::endl;

    for (int i = 0; i < num_iterations; ++i)
    {
        // Record start event
        CHECK_CUDA(cudaEventRecord(start));

        // Execute CUB inclusive scan
        CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                                 d_input, d_output, N));

        // Record stop event
        CHECK_CUDA(cudaEventRecord(stop));
        // Wait for completion
        CHECK_CUDA(cudaEventSynchronize(stop));

        // Calculate elapsed time
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

    // Calculate standard deviation
    float variance = 0.0f;
    for (float time : times)
    {
        float diff = time - mean;
        variance += diff * diff;
    }
    variance /= num_iterations;
    float std_dev = std::sqrt(variance);

    // Calculate throughput
    double bytes_processed = 2 * N * sizeof(T); // input + output (both full arrays)
    double gb_per_sec = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (mean / 1000.0);
    double elements_per_sec = N / (mean / 1000.0) / 1e9; // in billions

    // Verify result (optional - check first few and last few elements)
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, N * sizeof(T), cudaMemcpyDeviceToHost));

    // Compute expected values for verification (skip for large arrays)
    bool verify = true;
    if (N <= 1000000) // Skip verification for large arrays
    {
        std::vector<T> expected(N);
        expected[0] = h_input[0];
        for (size_t i = 1; i < N; ++i)
        {
            expected[i] = expected[i - 1] + h_input[i];
        }

        // Check first 10 and last 10 elements (or all if N < 20)
        size_t check_count = std::min(size_t(10), N / 2);
        T tolerance = std::is_floating_point<T>::value ? T(1e-3) : T(0);

        for (size_t i = 0; i < check_count; ++i)
        {
            T diff = compute_abs_diff(h_output[i], expected[i]);
            if (diff > tolerance)
            {
                verify = false;
                std::cout << "Verification failed at index " << i << ": expected "
                          << expected[i] << ", got " << h_output[i] << std::endl;
                break;
            }
        }
        for (size_t i = N - check_count; i < N && verify; ++i)
        {
            T diff = compute_abs_diff(h_output[i], expected[i]);
            if (diff > tolerance)
            {
                verify = false;
                std::cout << "Verification failed at index " << i << ": expected "
                          << expected[i] << ", got " << h_output[i] << std::endl;
                break;
            }
        }

        if (verify)
        {
            std::cout << "\nVerification: PASSED" << std::endl;
        }
    }
    else
    {
        std::cout << "\nVerification: Skipped (array too large)" << std::endl;
    }

    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mean time:          " << mean << " ± " << std_dev << " ms" << std::endl;
    std::cout << "Min time:           " << min_time << " ms" << std::endl;
    std::cout << "Max time:           " << max_time << " ms" << std::endl;
    std::cout << "Coefficient of var: " << (std_dev / mean) * 100 << "%" << std::endl;
    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "Throughput:         " << gb_per_sec << " GB/s" << std::endl;
    std::cout << "Elements/sec:       " << elements_per_sec << " billion elements/s" << std::endl;

    // Show sample output (first 10 and last 10 elements)
    if (N <= 100)
    {
        std::cout << "\n=== Sample Output ===" << std::endl;
        std::cout << "Input:  ";
        for (size_t i = 0; i < std::min(size_t(10), N); ++i)
            std::cout << h_input[i] << " ";
        if (N > 10)
            std::cout << "...";
        std::cout << std::endl;

        std::cout << "Output: ";
        for (size_t i = 0; i < std::min(size_t(10), N); ++i)
            std::cout << h_output[i] << " ";
        if (N > 10)
            std::cout << "...";
        std::cout << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp_storage));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(warmup_start));
    CHECK_CUDA(cudaEventDestroy(warmup_stop));
}

// Additional benchmark for exclusive scan
template <typename T>
void benchmark_cub_exclusive_scan(size_t N, float warmup_ms = 500.0f, int num_iterations = 100)
{
    std::cout << "\n=== CUB Exclusive Scan Benchmark ===" << std::endl;
    std::cout << "Data type: " << getTypeName<T>() << std::endl;
    std::cout << "Vector size: " << N << " elements" << std::endl;
    std::cout << "Element size: " << sizeof(T) << " bytes" << std::endl;
    std::cout << "Memory size (input): " << (N * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Memory size (output): " << (N * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;

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

    CHECK_CUDA(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_input, d_output, N));
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    std::cout << "Temp storage required: " << temp_storage_bytes / 1024.0 << " KB" << std::endl;

    // Create CUDA events for timing
    cudaEvent_t start, stop, warmup_start, warmup_stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&warmup_start));
    CHECK_CUDA(cudaEventCreate(&warmup_stop));

    // Warmup runs for specified duration
    std::cout << "\nPerforming warmup for " << warmup_ms << " ms..." << std::endl;
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
    std::cout << "Completed " << warmup_iterations << " warmup iterations in " << elapsed_warmup << " ms" << std::endl;
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark runs
    std::vector<float> times;
    times.reserve(num_iterations);

    std::cout << "Performing " << num_iterations << " benchmark iterations..." << std::endl;

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

    // Calculate standard deviation
    float variance = 0.0f;
    for (float time : times)
    {
        float diff = time - mean;
        variance += diff * diff;
    }
    variance /= num_iterations;
    float std_dev = std::sqrt(variance);

    // Calculate throughput
    double bytes_processed = 2 * N * sizeof(T);
    double gb_per_sec = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (mean / 1000.0);
    double elements_per_sec = N / (mean / 1000.0) / 1e9;

    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mean time:          " << mean << " ± " << std_dev << " ms" << std::endl;
    std::cout << "Min time:           " << min_time << " ms" << std::endl;
    std::cout << "Max time:           " << max_time << " ms" << std::endl;
    std::cout << "Coefficient of var: " << (std_dev / mean) * 100 << "%" << std::endl;
    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "Throughput:         " << gb_per_sec << " GB/s" << std::endl;
    std::cout << "Elements/sec:       " << elements_per_sec << " billion elements/s" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp_storage));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(warmup_start));
    CHECK_CUDA(cudaEventDestroy(warmup_stop));
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    size_t N = 100000000;     // Default: 100 million elements
    float warmup_ms = 500.0f; // Default: 500ms warmup
    int iterations = 100;
    bool exclusive = false;
    std::string dtype = "all"; // Default: run all types

    if (argc > 1)
    {
        N = std::stoull(argv[1]);
    }
    if (argc > 2)
    {
        iterations = std::stoi(argv[2]);
    }
    if (argc > 3)
    {
        warmup_ms = std::stof(argv[3]);
    }
    if (argc > 4)
    {
        std::string mode(argv[4]);
        if (mode == "exclusive" || mode == "ex")
            exclusive = true;
    }
    if (argc > 5)
    {
        dtype = argv[5];
    }

    // Print GPU info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Peak Memory Bandwidth: " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;

    if (!exclusive)
    {
        // Run inclusive scan benchmarks for different data types
        if (dtype == "all" || dtype == "float")
        {
            std::cout << "\n### Float (32-bit) - Inclusive Scan ###" << std::endl;
            benchmark_cub_inclusive_scan<float>(N, warmup_ms, iterations);
        }

        if (dtype == "all" || dtype == "double")
        {
            std::cout << "\n### Double (64-bit) - Inclusive Scan ###" << std::endl;
            benchmark_cub_inclusive_scan<double>(N, warmup_ms, iterations);
        }

        if (dtype == "all" || dtype == "int")
        {
            std::cout << "\n### Int (32-bit) - Inclusive Scan ###" << std::endl;
            benchmark_cub_inclusive_scan<int>(N, warmup_ms, iterations);
        }

        if (dtype == "all" || dtype == "uint64")
        {
            std::cout << "\n### UInt64 (64-bit) - Inclusive Scan ###" << std::endl;
            benchmark_cub_inclusive_scan<uint64_t>(N, warmup_ms, iterations);
        }
    }
    else
    {
        // Run exclusive scan benchmarks
        if (dtype == "all" || dtype == "float")
        {
            std::cout << "\n### Float (32-bit) - Exclusive Scan ###" << std::endl;
            benchmark_cub_exclusive_scan<float>(N, warmup_ms, iterations);
        }

        if (dtype == "all" || dtype == "double")
        {
            std::cout << "\n### Double (64-bit) - Exclusive Scan ###" << std::endl;
            benchmark_cub_exclusive_scan<double>(N, warmup_ms, iterations);
        }

        if (dtype == "all" || dtype == "int")
        {
            std::cout << "\n### Int (32-bit) - Exclusive Scan ###" << std::endl;
            benchmark_cub_exclusive_scan<int>(N, warmup_ms, iterations);
        }

        if (dtype == "all" || dtype == "uint64")
        {
            std::cout << "\n### UInt64 (64-bit) - Exclusive Scan ###" << std::endl;
            benchmark_cub_exclusive_scan<uint64_t>(N, warmup_ms, iterations);
        }
    }

    return 0;
}

// Compilation command:
// nvcc -O3 -std=c++14 -arch=sm_70 cub_scan_benchmark.cu -o cub_scan_benchmark
//
// Usage:
// ./cub_scan_benchmark [N] [iterations] [warmup_ms] [mode] [dtype]
//
// Examples:
// ./cub_scan_benchmark 1000000                    # 1 million elements, all types
// ./cub_scan_benchmark 100000000 200 1000         # 100M elements, 200 iterations, 1000ms warmup
// ./cub_scan_benchmark 100000000 100 500 inclusive uint64 # 100M elements, inclusive scan, uint64 only
// ./cub_scan_benchmark 100000000 100 500 exclusive all    # Exclusive scan, all types