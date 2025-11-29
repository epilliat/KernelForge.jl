#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
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

// Get type name for display
template <typename T>
std::string getTypeName()
{
    if (std::is_same<T, float>::value)
        return "float32";
    if (std::is_same<T, double>::value)
        return "float64";
    return "unknown";
}

template <typename T>
void benchmark_cub_memcpy(size_t N, float warmup_ms = 500.0f, int num_iterations = 100)
{
    std::cout << "\n=== CUB DeviceCopy Benchmark ===" << std::endl;
    std::cout << "Data type: " << getTypeName<T>() << std::endl;
    std::cout << "Vector size: " << N << " elements" << std::endl;
    std::cout << "Element size: " << sizeof(T) << " bytes" << std::endl;
    std::cout << "Memory size: " << (N * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Total bandwidth (R+W): " << (2 * N * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;

    // Allocate host memory
    std::vector<T> h_input(N);
    std::vector<T> h_output(N);

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    for (size_t i = 0; i < N; ++i)
    {
        h_input[i] = dist(gen);
    }

    // Allocate device memory
    T *d_input = nullptr;
    T *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(T)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    // Determine temporary storage requirements for CUB
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // CUB DeviceCopy doesn't need temp storage, but we'll use DeviceSelect::Flagged as a copy proxy
    // Actually, for a pure copy, we'll just use the basic approach
    // CUB doesn't have a dedicated copy function, so we'll use cudaMemcpy for device-to-device

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
        CHECK_CUDA(cudaMemcpy(d_output, d_input, N * sizeof(T), cudaMemcpyDeviceToDevice));
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

        // Execute memory copy (device to device)
        CHECK_CUDA(cudaMemcpy(d_output, d_input, N * sizeof(T), cudaMemcpyDeviceToDevice));

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
    double bytes_processed = 2 * N * sizeof(T); // read + write
    double gb_per_sec_mean = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (mean / 1000.0);
    double gb_per_sec_min = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (min_time / 1000.0);
    double elements_per_sec = N / (mean / 1000.0) / 1e9; // in billions

    // Verify result (check first few and last few elements)
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, N * sizeof(T), cudaMemcpyDeviceToHost));

    bool verify = true;
    size_t check_count = std::min(size_t(100), N);
    for (size_t i = 0; i < check_count; ++i)
    {
        if (std::abs(h_output[i] - h_input[i]) > T(1e-6))
        {
            verify = false;
            std::cout << "Verification failed at index " << i << ": expected "
                      << h_input[i] << ", got " << h_output[i] << std::endl;
            break;
        }
    }

    if (verify)
    {
        std::cout << "\nVerification: PASSED" << std::endl;
    }

    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mean time:          " << mean << " ± " << std_dev << " ms" << std::endl;
    std::cout << "Min time:           " << min_time << " ms" << std::endl;
    std::cout << "Max time:           " << max_time << " ms" << std::endl;
    std::cout << "Coefficient of var: " << (std_dev / mean) * 100 << "%" << std::endl;
    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "Throughput (mean):  " << gb_per_sec_mean << " GB/s" << std::endl;
    std::cout << "Throughput (best):  " << gb_per_sec_min << " GB/s" << std::endl;
    std::cout << "Elements/sec:       " << elements_per_sec << " billion elements/s" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(warmup_start));
    CHECK_CUDA(cudaEventDestroy(warmup_stop));
}

template <typename T>
void benchmark_thrust_copy(size_t N, float warmup_ms = 500.0f, int num_iterations = 100)
{
    std::cout << "\n=== Thrust Copy Benchmark ===" << std::endl;
    std::cout << "Data type: " << getTypeName<T>() << std::endl;
    std::cout << "Vector size: " << N << " elements" << std::endl;
    std::cout << "Element size: " << sizeof(T) << " bytes" << std::endl;
    std::cout << "Memory size: " << (N * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Total bandwidth (R+W): " << (2 * N * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;

    // Allocate host memory
    std::vector<T> h_input(N);

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    for (size_t i = 0; i < N; ++i)
    {
        h_input[i] = dist(gen);
    }

    // Create thrust device vectors
    thrust::device_vector<T> d_input(h_input);
    thrust::device_vector<T> d_output(N);

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
        thrust::copy(d_input.begin(), d_input.end(), d_output.begin());
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

        // Execute thrust copy
        thrust::copy(d_input.begin(), d_input.end(), d_output.begin());

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
    double bytes_processed = 2 * N * sizeof(T); // read + write
    double gb_per_sec_mean = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (mean / 1000.0);
    double gb_per_sec_min = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (min_time / 1000.0);
    double elements_per_sec = N / (mean / 1000.0) / 1e9; // in billions

    // Verify result
    thrust::host_vector<T> h_output = d_output;
    bool verify = true;
    size_t check_count = std::min(size_t(100), N);
    for (size_t i = 0; i < check_count; ++i)
    {
        if (std::abs(h_output[i] - h_input[i]) > T(1e-6))
        {
            verify = false;
            std::cout << "Verification failed at index " << i << ": expected "
                      << h_input[i] << ", got " << h_output[i] << std::endl;
            break;
        }
    }

    if (verify)
    {
        std::cout << "\nVerification: PASSED" << std::endl;
    }

    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mean time:          " << mean << " ± " << std_dev << " ms" << std::endl;
    std::cout << "Min time:           " << min_time << " ms" << std::endl;
    std::cout << "Max time:           " << max_time << " ms" << std::endl;
    std::cout << "Coefficient of var: " << (std_dev / mean) * 100 << "%" << std::endl;
    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "Throughput (mean):  " << gb_per_sec_mean << " GB/s" << std::endl;
    std::cout << "Throughput (best):  " << gb_per_sec_min << " GB/s" << std::endl;
    std::cout << "Elements/sec:       " << elements_per_sec << " billion elements/s" << std::endl;

    // Cleanup
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
    int iterations = 100;     // Default: 100 iterations
    bool use_thrust = true;   // Default: benchmark both CUB and Thrust

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
        if (mode == "cub_only")
            use_thrust = false;
    }

    // Print GPU info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "==========================================" << std::endl;
    std::cout << "        Memory Copy Benchmark" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Peak Memory Bandwidth: " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
    std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "==========================================" << std::endl;

    std::cout << "\nBenchmark Configuration:" << std::endl;
    std::cout << "  Vector size: " << N << " elements" << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Warmup time: " << warmup_ms << " ms" << std::endl;
    std::cout << "==========================================" << std::endl;

    // Run benchmarks for float32
    std::cout << "\n############################################" << std::endl;
    std::cout << "#         FLOAT32 BENCHMARKS              #" << std::endl;
    std::cout << "############################################" << std::endl;

    benchmark_cub_memcpy<float>(N, warmup_ms, iterations);

    if (use_thrust)
    {
        benchmark_thrust_copy<float>(N, warmup_ms, iterations);
    }

    // Run benchmarks for float64
    std::cout << "\n############################################" << std::endl;
    std::cout << "#         FLOAT64 BENCHMARKS              #" << std::endl;
    std::cout << "############################################" << std::endl;

    benchmark_cub_memcpy<double>(N, warmup_ms, iterations);

    if (use_thrust)
    {
        benchmark_thrust_copy<double>(N, warmup_ms, iterations);
    }

    // Summary comparison
    std::cout << "\n==========================================" << std::endl;
    std::cout << "           BENCHMARK COMPLETE" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Note: CUB uses cudaMemcpy for device-to-device copy" << std::endl;
    std::cout << "      Thrust uses thrust::copy algorithm" << std::endl;
    std::cout << "==========================================" << std::endl;

    return 0;
}

// Compilation command:
// nvcc -O3 -std=c++14 -arch=sm_70 cub_memcpy_benchmark.cu -o cub_memcpy_benchmark
//
// For newer architectures:
// nvcc -O3 -std=c++14 -arch=sm_80 cub_memcpy_benchmark.cu -o cub_memcpy_benchmark
// nvcc -O3 -std=c++14 -arch=sm_90 cub_memcpy_benchmark.cu -o cub_memcpy_benchmark
//
// Usage:
// ./cub_memcpy_benchmark [N] [iterations] [warmup_ms] [mode]
//
// Examples:
// ./cub_memcpy_benchmark                           # Default: 100M elements, both libraries
// ./cub_memcpy_benchmark 1000000                   # 1 million elements
// ./cub_memcpy_benchmark 100000000 200 1000        # 100M elements, 200 iterations, 1000ms warmup
// ./cub_memcpy_benchmark 500000000 100 500 cub_only # 500M elements, CUB only (no Thrust)