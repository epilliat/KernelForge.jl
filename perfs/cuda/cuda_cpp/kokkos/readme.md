rm -rf build && mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DCMAKE_C_COMPILER=gcc-12 \
    -DKokkos_ROOT=/usr/local/kokkos \
    -DCMAKE_CUDA_ARCHITECTURES=89

make
./bench_kokkos_scan