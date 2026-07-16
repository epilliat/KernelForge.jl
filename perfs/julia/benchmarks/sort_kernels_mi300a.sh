#!/bin/bash
#SBATCH --account=cad17552
#SBATCH --constraint=MI300
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH -o /home/epilliat/kf/sortkernels_%j.out
export JULIA_DEPOT_PATH=/home/epilliat/kf/depot JULIA_AMDGPU_DISABLE_ARTIFACTS=1 ROCM_PATH=/opt/rocm-6.4.3
export PATH=/opt/rocm-6.4.3/bin:$PATH
export LD_LIBRARY_PATH=/home/epilliat/kf/julia-1.12.6/lib/julia:$LD_LIBRARY_PATH
cd /home/epilliat/kf/KernelForge.jl
JL=/home/epilliat/kf/julia-1.12.6/bin/julia
PROJ=/home/epilliat/kf/env
DRIVER=perfs/julia/benchmarks/sort_kernel_profile_driver.jl
RP=/home/epilliat/kf/KernelForge.jl/perfs/rocm_cpp/hipcub_hipcc/bin/rocprim_sort_benchmark
STATS=/home/epilliat/kf/ksort_stats
rm -rf $STATS; mkdir -p $STATS

declare -A JLTYPE=( [uint32]=UInt32 [uint64]=UInt64 [float]=Float32 [double]=Float64 )
NS="1000000 10000000 100000000"
TS="uint32 uint64 float double"

grab () {  # $1=outdir  $2=dest-basename
  f=$(find "$1" -name "*_kernel_stats.csv" 2>/dev/null | head -1)
  [ -n "$f" ] && cp "$f" "$STATS/$2" && echo "  -> $2" || echo "  !! no kernel_stats in $1"
}

for N in $NS; do
  for T in $TS; do
    JT=${JLTYPE[$T]}
    echo "#### KF  $JT  $N ####"
    d=/home/epilliat/kf/ks_kf_${T}_${N}; rm -rf $d
    KF_N=$N KF_T=$T srun rocprofv3 --kernel-trace --stats --output-format csv -d $d -- \
        $JL --project=$PROJ -t 8 $DRIVER 2>&1 | grep -E ">>>|ERROR" | head -3
    grab $d "KernelForge__${JT}__${N}__stats.csv"

    echo "#### rocPRIM  $JT  $N ####"
    d=/home/epilliat/kf/ks_rp_${T}_${N}; rm -rf $d
    srun rocprofv3 --kernel-trace --stats --output-format csv -d $d -- \
        $RP -n $N -t $T -i 20 -w 100 2>&1 | grep -E "Mean time|ERROR" | head -2
    grab $d "rocPRIM__${JT}__${N}__stats.csv"
  done
done

echo "==== PARSE -> sort_kernels.csv ===="
KF_STATS_DIR=$STATS KF_OUT=/home/epilliat/kf/sort_kernels_MI300A.csv \
    $JL --project=$PROJ perfs/julia/benchmarks/parse_sort_kernels.jl 2>&1 | tail -3
echo "==== RESULT ===="
cat /home/epilliat/kf/sort_kernels_MI300A.csv
echo "=== DONE ==="
