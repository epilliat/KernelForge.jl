cmake_minimum_required(VERSION 3.18)
project(ScanBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
find_package(Kokkos REQUIRED)

set_source_files_properties(bench_kokkos_scan.cpp PROPERTIES LANGUAGE CUDA)
add_executable(bench_kokkos_scan bench_kokkos_scan.cpp)
target_link_libraries(bench_kokkos_scan Kokkos::kokkos)