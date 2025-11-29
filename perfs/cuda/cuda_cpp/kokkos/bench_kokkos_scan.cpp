/**
 * Kokkos Inclusive Scan Benchmark with CUDA Events
 * Uses Kokkos::parallel_scan built-in function
 * N = 1e8, Float32
 */

#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

constexpr size_t N = 100'000'000;  // 1e8
constexpr int WARMUP_RUNS = 5;
constexpr int BENCHMARK_RUNS = 20;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

struct BenchmarkResult {
    double min_ms;
    double max_ms;
    double mean_ms;
    double stddev_ms;
    double bandwidth_gb_s;
};

BenchmarkResult compute_stats(const std::vector<float>& times_ms, size_t bytes_transferred) {
    BenchmarkResult result;
    result.min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    result.max_ms = *std::max_element(times_ms.begin(), times_ms.end());
    
    double sum = 0.0;
    for (auto t : times_ms) sum += t;
    result.mean_ms = sum / times_ms.size();
    
    double sq_sum = 0.0;
    for (auto t : times_ms) sq_sum += (t - result.mean_ms) * (t - result.mean_ms);
    result.stddev_ms = std::sqrt(sq_sum / times_ms.size());
    
    result.bandwidth_gb_s = (bytes_transferred / 1e9) / (result.min_ms / 1000.0);
    
    return result;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        std::cout << "========================================================" << std::endl;
        std::cout << "    Kokkos Inclusive Scan Benchmark" << std::endl;
        std::cout << "========================================================" << std::endl;
        std::cout << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  N = " << N << " elements (" << (N * sizeof(float) / 1e9) << " GB)" << std::endl;
        std::cout << "  Type: Float32" << std::endl;
        std::cout << "  Warmup runs: " << WARMUP_RUNS << std::endl;
        std::cout << "  Benchmark runs: " << BENCHMARK_RUNS << std::endl;
        std::cout << std::endl;

        // Print GPU info
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB" << std::endl;
        std::cout << "  Memory bandwidth: " << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6) << " GB/s (theoretical)" << std::endl;
        std::cout << std::endl;

        std::cout << "Kokkos configuration:" << std::endl;
        Kokkos::print_configuration(std::cout);
        std::cout << std::endl;

        using ExecutionSpace = Kokkos::DefaultExecutionSpace;
        using MemorySpace = ExecutionSpace::memory_space;
        using ViewType = Kokkos::View<float*, MemorySpace>;

        ViewType d_input("input", N);
        ViewType d_output("output", N);

        // Initialize with random data
        std::cout << "Generating input data..." << std::endl;
        {
            auto h_input = Kokkos::create_mirror_view(d_input);
            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (size_t i = 0; i < N; ++i) {
                h_input(i) = dist(gen);
            }
            Kokkos::deep_copy(d_input, h_input);
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        std::cout << "Warming up..." << std::endl;
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            Kokkos::parallel_scan(
                "inclusive_scan",
                Kokkos::RangePolicy<ExecutionSpace>(0, N),
                KOKKOS_LAMBDA(const size_t j, float& update, const bool final) {
                    update += d_input(j);
                    if (final) d_output(j) = update;
                }
            );
            Kokkos::fence();
        }

        // Benchmark
        std::cout << "Benchmarking..." << std::endl;
        std::vector<float> times_ms;
        times_ms.reserve(BENCHMARK_RUNS);

        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            
            Kokkos::parallel_scan(
                "inclusive_scan",
                Kokkos::RangePolicy<ExecutionSpace>(0, N),
                KOKKOS_LAMBDA(const size_t j, float& update, const bool final) {
                    update += d_input(j);
                    if (final) d_output(j) = update;
                }
            );
            
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            times_ms.push_back(elapsed_ms);
        }

        // Verify
        bool verified = true;
        {
            auto h_input = Kokkos::create_mirror_view(d_input);
            auto h_output = Kokkos::create_mirror_view(d_output);
            Kokkos::deep_copy(h_input, d_input);
            Kokkos::deep_copy(h_output, d_output);
            
            float expected = 0.0f;
            for (size_t i = 0; i < std::min(N, size_t(1000)); ++i) {
                expected += h_input(i);
                if (std::abs(h_output(i) - expected) > 1e-3f * std::abs(expected) + 1e-6f) {
                    std::cerr << "Verification failed at index " << i << std::endl;
                    verified = false;
                    break;
                }
            }
        }

        size_t bytes_transferred = 2 * N * sizeof(float);
        auto stats = compute_stats(times_ms, bytes_transferred);

        std::cout << std::endl;
        std::cout << "Results:" << std::endl;
        std::cout << "  Verification: " << (verified ? "PASS" : "FAIL") << std::endl;
        std::cout << "  Min time:     " << std::fixed << std::setprecision(3) << stats.min_ms << " ms" << std::endl;
        std::cout << "  Max time:     " << stats.max_ms << " ms" << std::endl;
        std::cout << "  Mean time:    " << stats.mean_ms << " ms" << std::endl;
        std::cout << "  Stddev:       " << stats.stddev_ms << " ms" << std::endl;
        std::cout << "  Bandwidth:    " << std::setprecision(1) << stats.bandwidth_gb_s << " GB/s" << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    Kokkos::finalize();
    return 0;
}
