#!/bin/bash
#SBATCH --job-name=sortkfA100
#SBATCH --account=ubc@a100
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=00:35:00
#SBATCH --output=/lustre/fswork/projects/rech/ubc/utk53wr/logs/sortkfA100_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ubc/utk53wr/logs/sortkfA100_%j.out
source /lustre/fswork/projects/rech/ubc/utk53wr/kf_env.sh
cd $KF
DRIVER=perfs/julia/benchmarks/sort_kernel_profile_cuda.jl
STATS=$WORK/ksort_stats_a100          # keep the nsys CUB stats already here
declare -A JLTYPE=( [uint32]=UInt32 [uint64]=UInt64 [float]=Float32 [double]=Float64 )
NS="1000000 10000000 100000000"
TS="uint32 uint64 float double"

for N in $NS; do
  for T in $TS; do
    JT=${JLTYPE[$T]}
    echo "#### KF(CUDA.@profile)  $JT  $N ####"
    KF_N=$N KF_T=$T KF_OUT_STATS=$STATS/KernelForge__${JT}__${N}__stats.csv \
        $JULIA --project=$KF/test/envs/cuda $DRIVER 2>&1 | grep -E ">>>|ERROR" | head -3
  done
done

echo "==== PARSE (KF CUDA.@profile + CUB nsys) -> sort_kernels_A100.csv ===="
KF_STATS_DIR=$STATS KF_OUT=$WORK/sort_kernels_A100.csv \
    $JULIA --project=$KF/test/envs/cuda perfs/julia/benchmarks/parse_sort_kernels.jl 2>&1 | tail -3
echo "==== KF rows ===="
grep KernelForge $WORK/sort_kernels_A100.csv | head -14
echo "=== DONE ==="
