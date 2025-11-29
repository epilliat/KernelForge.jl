#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <type_traits>
#include <chrono>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
// profiling: sudo ncu --launch-count 10     --metrics gpu__time_duration.sum     --csv     ./cub_profile
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

// Specialization for integer types (excluding uint8_t)
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

// Specialization for uint8_t (uniform_int_distribution requires at least 16-bit types)
template <>
struct DataInitializer<uint8_t>
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

template <typename T>
void benchmark_cub_sum(size_t N, double warmup_ms = 500.0, int num_iterations = 100)
{
    std::cout << "\n=== CUB Sum Reduction Benchmark ===" << std::endl;
    std::cout << "Vector size: " << N << " elements" << std::endl;
    std::cout << "Data type: " << sizeof(T) << " bytes" << std::endl;
    std::cout << "Memory size: " << (N * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;

    // Allocate host memory
    std::vector<T> h_input(N);
    T h_output;

    // Initialize with random data
    DataInitializer<T>::initialize(h_input);

    // Allocate device memory
    T *d_input = nullptr;
    T *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(T)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    // Determine temporary storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, N));

    // Allocate temporary storage
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    std::cout << "Temp storage required: " << temp_storage_bytes / 1024.0 << " KB" << std::endl;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Time-based warmup (500ms)
    std::cout << "\nPerforming " << warmup_ms << " ms warmup..." << std::endl;
    auto warmup_start = std::chrono::high_resolution_clock::now();
    int warmup_iterations = 0;

    while (true)
    {
        CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, N));
        CHECK_CUDA(cudaDeviceSynchronize());
        warmup_iterations++;

        auto warmup_current = std::chrono::high_resolution_clock::now();
        auto warmup_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  warmup_current - warmup_start)
                                  .count();

        if (warmup_elapsed >= warmup_ms)
            break;
    }

    std::cout << "Completed " << warmup_iterations << " warmup iterations" << std::endl;

    // Benchmark runs
    std::vector<float> times;
    times.reserve(num_iterations);

    std::cout << "Performing " << num_iterations << " benchmark iterations..." << std::endl;

    for (int i = 0; i < num_iterations; ++i)
    {
        // Record start event
        CHECK_CUDA(cudaEventRecord(start));

        // Execute CUB sum reduction
        CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, N));

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
    double bytes_processed = N * sizeof(T) + sizeof(T); // input + output
    double gb_per_sec = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (mean / 1000.0);
    double elements_per_sec = N / (mean / 1000.0) / 1e9; // in billions

    // Verify result (optional)
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost));

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
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    size_t N = 100000000; // Default: 100 million elements
    double warmup_ms = 500.0;
    int iterations = 100;

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
        warmup_ms = std::stod(argv[3]);
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

    // Run benchmarks for different data types
    std::cout << "\n### UInt8 (8-bit) ###" << std::endl;
    benchmark_cub_sum<uint8_t>(N, warmup_ms, iterations);

    std::cout << "\n### Float (32-bit) ###" << std::endl;
    benchmark_cub_sum<float>(N, warmup_ms, iterations);

    std::cout << "\n### Double (64-bit) ###" << std::endl;
    benchmark_cub_sum<double>(N, warmup_ms, iterations);

    std::cout << "\n### Int (32-bit) ###" << std::endl;
    benchmark_cub_sum<int>(N, warmup_ms, iterations);

    return 0;
}

// Compilation command:
// nvcc -O3 -std=c++14 -arch=sm_70 cub_sum_benchmark.cu -o cub_sum_benchmark
//
// Usage:
// ./cub_sum_benchmark [N] [iterations] [warmup_ms]
//
// Examples:
// ./cub_sum_benchmark 1000000              # 1 million elements
// ./cub_sum_benchmark 100000000 200 1000   # 100M elements, 200 iterations, 1000ms warmup