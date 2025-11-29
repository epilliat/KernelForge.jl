cmake_minimum_required(VERSION 3.18)
project(ScanBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(Kokkos REQUIRED)

add_executable(bench_kokkos_scan bench_kokkos_scan.cpp)
target_link_libraries(bench_kokkos_scan Kokkos::kokkos)
