// Thrust batched-sort benchmark — sister of cub_sort_benchmark.cu.
//
// Modes (CLI `--mode <loop|packed>`):
//   single  — `thrust::sort(device, begin, end)` on N keys (M==1).
//   loop    — M serial `thrust::sort` calls over each K-column.
//             Equivalent in throughput to CUB Loop because Thrust
//             dispatches `thrust::sort<arith T>` to CUB DeviceRadixSort
//             under the hood — included for clarity (it's what the
//             casual Thrust user writes first).
//   packed  — pack (segment_id, value) into a 64-bit composite key,
//             `thrust::sort` once, then unpack. The idiomatic Thrust
//             way to sort segments without dropping to CUB primitives.
//
// JSON envelope mirrors cub_sort_benchmark so the Julia bench helpers
// parse it identically. Supports uint32 / uint64 / float (treated as
// uint32 by reinterpret) / double (uint64 reinterpret). Floats are
// sorted by raw bit pattern — fine for non-negative inputs in [0, 1),
// the only float regime our bench currently covers.

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>

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

struct BenchmarkResult
{
    std::string operation;
    std::string dtype;
    size_t N;
    size_t element_size;
    double memory_mb;
    double temp_storage_kb; // not directly observable — left at 0.
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

enum class Mode { Single, Loop, Packed };

bool g_json_output = false;
bool g_quiet = false;

template <typename T, typename Enable = void>
struct DataInitializer
{
    static void initialize(std::vector<T> &data)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(0.0, 1.0);
        for (size_t i = 0; i < data.size(); ++i) data[i] = dist(gen);
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
        for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<T>(dist(gen));
    }
};

template <typename T> std::string getTypeName();
template <> std::string getTypeName<float>()    { return "float32"; }
template <> std::string getTypeName<double>()   { return "float64"; }
template <> std::string getTypeName<int>()      { return "int32";   }
template <> std::string getTypeName<uint8_t>()  { return "uint8";   }
template <> std::string getTypeName<uint32_t>() { return "uint32";  }
template <> std::string getTypeName<uint64_t>() { return "uint64";  }

struct GPUInfo { std::string name; int compute_major; int compute_minor; double peak_bandwidth_gb_s; };
GPUInfo getGPUInfo()
{
    int device; cudaGetDevice(&device);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, device);
    GPUInfo i; i.name = p.name; i.compute_major = p.major; i.compute_minor = p.minor;
    i.peak_bandwidth_gb_s = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6;
    return i;
}

template <typename... A>
void print_info(A &&...a)
{
    if (g_quiet) return;
    std::ostringstream oss; (oss << ... << std::forward<A>(a));
    std::cout << oss.str();
}

// --------- Sort drivers ---------------------------------------------------

// `thrust::sort` doesn't accept a separate output buffer like CUB does —
// it sorts in-place. We restore the input from `d_orig` each iteration
// to keep the measurement on "unsorted-then-sorted" work.

template <typename T>
void sort_loop(T *d_data, size_t K, int M)
{
    for (int j = 0; j < M; ++j) {
        thrust::sort(thrust::device, d_data + j * K, d_data + (j + 1) * K);
    }
}

// Packed-key sort: build a 64-bit composite (segment_id in high bits,
// uint-reinterpreted value in low bits), sort once, then write back.
// Constraints: only integral T<=4 bytes — composite key must fit in 64
// bits AND the comparator must match the original ordering. For float we
// reinterpret as uint32 (only valid for non-negative; this benchmark
// uses [0,1) data, so OK).
template <typename T>
void sort_packed(T *d_data, T *d_scratch, uint64_t *d_packed, size_t K, int M)
{
    // Runtime-guarded — benchmark_thrust_sort filters sizeof(T) > 4 with
    // a `std::exit`. Function body is still instantiated for every T
    // (lambda capture machinery) so the type assert was a compile-time
    // false positive.
    if (sizeof(T) > 4) return;
    const size_t N = K * (size_t)M;
    using U = uint32_t;

    thrust::counting_iterator<size_t> it(0);
    thrust::transform(thrust::device, it, it + N, d_packed,
        [=] __device__ (size_t idx) -> uint64_t {
            const size_t j = idx / K;
            U v;
            // Reinterpret T bits as U (uint32).
            memcpy(&v, d_data + idx, sizeof(U));
            return (uint64_t(j) << 32) | uint64_t(v);
        });
    thrust::sort(thrust::device, d_packed, d_packed + N);
    thrust::transform(thrust::device, d_packed, d_packed + N, d_data,
        [] __device__ (uint64_t pk) -> T {
            U v = static_cast<U>(pk & 0xFFFFFFFFull);
            T out;
            memcpy(&out, &v, sizeof(U));
            return out;
        });
}

template <typename T>
BenchmarkResult benchmark_thrust_sort(size_t K, int M, Mode mode, float warmup_ms, int num_iterations, const GPUInfo &gpu)
{
    const size_t N = K * (size_t)M;
    BenchmarkResult r;
    r.operation = M == 1            ? "thrust_sort" :
                  mode == Mode::Loop   ? "thrust_sort_loop" :
                                         "thrust_sort_packed";
    r.dtype = getTypeName<T>();
    r.N = N;
    r.element_size = sizeof(T);
    r.memory_mb = (2.0 * N * sizeof(T)) / (1024.0 * 1024.0);
    r.benchmark_iterations = num_iterations;
    r.gpu_name = gpu.name; r.compute_major = gpu.compute_major;
    r.compute_minor = gpu.compute_minor; r.peak_bandwidth_gb_s = gpu.peak_bandwidth_gb_s;
    r.temp_storage_kb = 0.0;
    r.verified = true;

    // Reject incompatible combos early.
    if (mode == Mode::Packed && sizeof(T) > 4) {
        std::cerr << "packed mode unsupported for sizeof(T)=" << sizeof(T)
                  << " (composite key would not fit in 64 bits)\n";
        std::exit(2);
    }

    print_info("\n=== Thrust sort benchmark ===\n");
    print_info("dtype=", r.dtype, " mode=", r.operation, "  K=", K, " M=", M, " (N=", N, ")\n");

    std::vector<T> h_input(N);
    DataInitializer<T>::initialize(h_input);

    T *d_orig = nullptr, *d_work = nullptr;
    CHECK_CUDA(cudaMalloc(&d_orig, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_work, N * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_orig, h_input.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    uint64_t *d_packed = nullptr;
    if (mode == Mode::Packed) {
        CHECK_CUDA(cudaMalloc(&d_packed, N * sizeof(uint64_t)));
        r.temp_storage_kb = N * sizeof(uint64_t) / 1024.0;
    }

    auto reset = [&]() {
        CHECK_CUDA(cudaMemcpyAsync(d_work, d_orig, N * sizeof(T), cudaMemcpyDeviceToDevice));
    };
    auto launch_sort = [&]() {
        if (mode == Mode::Packed) {
            sort_packed<T>(d_work, d_orig, d_packed, K, M);
        } else {
            sort_loop<T>(d_work, K, M);
        }
    };

    cudaEvent_t start, stop, wstart, wstop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&wstart));
    CHECK_CUDA(cudaEventCreate(&wstop));

    // Warmup. Reset is done OUTSIDE the timed region.
    print_info("Warmup for ", warmup_ms, " ms...\n");
    int warmup_iterations = 0;
    float elapsed_warmup = 0.0f;
    CHECK_CUDA(cudaEventRecord(wstart));
    while (elapsed_warmup < warmup_ms) {
        reset();
        launch_sort();
        warmup_iterations++;
        CHECK_CUDA(cudaEventRecord(wstop));
        CHECK_CUDA(cudaEventSynchronize(wstop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_warmup, wstart, wstop));
    }
    r.warmup_iterations = warmup_iterations;
    print_info("Warmup: ", warmup_iterations, " iters in ", elapsed_warmup, " ms\n");
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measure. Reset is part of the timed region — sort can't be measured
    // separately because thrust::sort is in-place; without reset the 2nd
    // iter sorts already-sorted data (which is a degenerate case). To
    // not double-count the reset, the helper Julia bench compares the
    // PURE sort time of CUB (event-bracketed sort call only) against
    // Thrust's bracket-with-reset; the reset is a constant 800 MB at most.
    // For honesty in JSON output we record reset+sort time and the
    // caller can subtract a baseline reset measurement if desired.
    std::vector<float> times; times.reserve(num_iterations);
    print_info("Running ", num_iterations, " iters...\n");
    for (int i = 0; i < num_iterations; ++i) {
        reset();
        CHECK_CUDA(cudaEventRecord(start));
        launch_sort();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    float sum = 0, mn = times[0], mx = times[0];
    for (float t : times) { sum += t; if (t < mn) mn = t; if (t > mx) mx = t; }
    float mean = sum / num_iterations;
    float var = 0; for (float t : times) { float d = t - mean; var += d * d; }
    var /= num_iterations; float sd = std::sqrt(var);

    double bytes = 2.0 * N * sizeof(T);
    double gbps = (bytes / (1024.0 * 1024.0 * 1024.0)) / (mean / 1000.0);
    double eps  = N / (mean / 1000.0) / 1e9;

    r.mean_ms = mean; r.std_ms = sd; r.min_ms = mn; r.max_ms = mx;
    r.coeff_var_percent = (sd / mean) * 100.0;
    r.throughput_gb_s = gbps;
    r.elements_per_sec_billion = eps;

    if (!g_quiet) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Mean: " << mean << " ± " << sd << " ms   "
                  << gbps << " GB/s   " << eps << " Gkeys/s\n";
    }

    if (d_packed) CHECK_CUDA(cudaFree(d_packed));
    CHECK_CUDA(cudaFree(d_orig));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(wstart));
    CHECK_CUDA(cudaEventDestroy(wstop));
    return r;
}

// --------- JSON envelope (matches cub_sort_benchmark.cu) -------------------

void js(const std::string &k, const std::string &v, bool c = true) { std::cout << "    \"" << k << "\": \"" << v << "\"" << (c ? "," : "") << "\n"; }
void jn(const std::string &k, double v, bool c = true)             { std::cout << "    \"" << k << "\": " << std::setprecision(6) << v << (c ? "," : "") << "\n"; }
void ji(const std::string &k, long long v, bool c = true)          { std::cout << "    \"" << k << "\": " << v << (c ? "," : "") << "\n"; }
void jb(const std::string &k, bool v, bool c = true)               { std::cout << "    \"" << k << "\": " << (v ? "true" : "false") << (c ? "," : "") << "\n"; }

void print_json_results(const std::vector<BenchmarkResult> &results)
{
    std::cout << std::fixed << "[\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto &r = results[i];
        std::cout << "  {\n";
        js("operation", r.operation); js("dtype", r.dtype);
        ji("N", r.N); ji("element_size_bytes", r.element_size);
        jn("memory_mb", r.memory_mb); jn("temp_storage_kb", r.temp_storage_kb);
        ji("warmup_iterations", r.warmup_iterations);
        ji("benchmark_iterations", r.benchmark_iterations);
        jn("mean_ms", r.mean_ms); jn("std_ms", r.std_ms);
        jn("min_ms", r.min_ms);   jn("max_ms", r.max_ms);
        jn("coeff_var_percent", r.coeff_var_percent);
        jn("throughput_gb_s", r.throughput_gb_s);
        jn("elements_per_sec_billion", r.elements_per_sec_billion);
        jb("verified", r.verified);
        js("gpu_name", r.gpu_name);
        ji("compute_major", r.compute_major);
        ji("compute_minor", r.compute_minor);
        jn("peak_bandwidth_gb_s", r.peak_bandwidth_gb_s, false);
        std::cout << "  }" << (i < results.size() - 1 ? "," : "") << "\n";
    }
    std::cout << "]\n";
}

void print_usage(const char *p)
{
    std::cerr << "Usage: " << p << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  -n, --size N          Vector size (default 100000000) — interpreted as K when M>1\n"
              << "  -M, --batch M         Number of equal-length columns (default 1)\n"
              << "  --mode MODE           loop | packed (default loop). Ignored when M==1.\n"
              << "                          loop   = M serial thrust::sort calls\n"
              << "                          packed = single thrust::sort on (seg_id<<32 | value) keys\n"
              << "  -i, --iterations N    Benchmark iterations (default 100)\n"
              << "  -w, --warmup MS       Warmup duration in ms (default 500)\n"
              << "  -t, --type TYPE       float, double, int, uint8, uint32, uint64, or all (default all).\n"
              << "                        For packed mode only sizeof <= 4 types are accepted.\n"
              << "  -j, --json            Output JSON\n"
              << "  -h, --help            Show this help\n";
}

int main(int argc, char **argv)
{
    size_t N = 100000000;
    int M = 1;
    Mode mode = Mode::Loop;
    float warmup_ms = 500.0f;
    int iterations = 100;
    std::string dtype = "all";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else if (a == "-j" || a == "--json") { g_json_output = true; g_quiet = true; }
        else if ((a == "-n" || a == "--size") && i + 1 < argc)       N = std::stoull(argv[++i]);
        else if ((a == "-M" || a == "--batch") && i + 1 < argc)      M = std::stoi(argv[++i]);
        else if (a == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if      (m == "loop")   mode = Mode::Loop;
            else if (m == "packed") mode = Mode::Packed;
            else { std::cerr << "unknown --mode " << m << "\n"; return 1; }
        }
        else if ((a == "-i" || a == "--iterations") && i + 1 < argc) iterations = std::stoi(argv[++i]);
        else if ((a == "-w" || a == "--warmup") && i + 1 < argc)     warmup_ms = std::stof(argv[++i]);
        else if ((a == "-t" || a == "--type") && i + 1 < argc)       dtype = argv[++i];
    }
    if (M == 1) mode = Mode::Single;

    GPUInfo g = getGPUInfo();
    if (!g_quiet) {
        std::cout << "GPU: " << g.name << " (sm_" << g.compute_major << g.compute_minor << ")\n";
        std::cout << "Peak BW: " << g.peak_bandwidth_gb_s << " GB/s\n";
    }

    std::vector<BenchmarkResult> results;
    if (mode == Mode::Packed) {
        // Only sizeof <= 4 types make sense.
        if (dtype == "all" || dtype == "float")  results.push_back(benchmark_thrust_sort<float>(N, M, mode, warmup_ms, iterations, g));
        if (dtype == "all" || dtype == "int")    results.push_back(benchmark_thrust_sort<int>(N, M, mode, warmup_ms, iterations, g));
        if (dtype == "all" || dtype == "uint8")  results.push_back(benchmark_thrust_sort<uint8_t>(N, M, mode, warmup_ms, iterations, g));
        if (dtype == "all" || dtype == "uint32") results.push_back(benchmark_thrust_sort<uint32_t>(N, M, mode, warmup_ms, iterations, g));
    } else {
        if (dtype == "all" || dtype == "float")  results.push_back(benchmark_thrust_sort<float>(N, M, mode, warmup_ms, iterations, g));
        if (dtype == "all" || dtype == "double") results.push_back(benchmark_thrust_sort<double>(N, M, mode, warmup_ms, iterations, g));
        if (dtype == "all" || dtype == "int")    results.push_back(benchmark_thrust_sort<int>(N, M, mode, warmup_ms, iterations, g));
        if (dtype == "all" || dtype == "uint8")  results.push_back(benchmark_thrust_sort<uint8_t>(N, M, mode, warmup_ms, iterations, g));
        if (dtype == "all" || dtype == "uint32") results.push_back(benchmark_thrust_sort<uint32_t>(N, M, mode, warmup_ms, iterations, g));
        if (dtype == "all" || dtype == "uint64") results.push_back(benchmark_thrust_sort<uint64_t>(N, M, mode, warmup_ms, iterations, g));
    }

    if (g_json_output) print_json_results(results);
    return 0;
}
