#!/bin/bash
#SBATCH --job-name=sortkA100
#SBATCH --account=ubc@a100
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=00:55:00
#SBATCH --output=/lustre/fswork/projects/rech/ubc/utk53wr/logs/sortkA100_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ubc/utk53wr/logs/sortkA100_%j.out
source /lustre/fswork/projects/rech/ubc/utk53wr/kf_env.sh
# `module` is NOT a command in a plain sbatch shell (only in a LOGIN shell) — a bare
# `module load` here fails with "commande introuvable", nsys never lands on PATH, the
# profiler silently no-ops and the parse writes a header-only CSV (seen 2026-07-17,
# job 1960288). Source the modulecmd init first, and fail loudly if nsys is missing.
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module load arch/a100 cuda/12.6.3 || echo "WARNING: module load failed"
command -v nsys >/dev/null || { echo "FATAL: nsys not on PATH — profiling would silently produce nothing"; exit 1; }
cd $KF
DRIVER=perfs/julia/benchmarks/sort_kernel_profile_driver.jl
CUB=$KF/perfs/cuda_cpp/cub_nvcc/bin/cub_sort_benchmark
STATS=$WORK/ksort_stats_a100
REP=$WORK/ksort_rep
rm -rf $STATS $REP; mkdir -p $STATS $REP

declare -A JLTYPE=( [uint32]=UInt32 [uint64]=UInt64 [float]=Float32 [double]=Float64 )
NS="1000000 10000000 100000000"
TS="uint32 uint64 float double"

profile () {  # $1=repbase  $2=dest-stats-basename  ; rest = command
  local rep=$1; shift; local dest=$1; shift
  nsys profile -t cuda -o $rep --force-overwrite true "$@" > /dev/null 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export true $rep.nsys-rep 2>/dev/null > "$STATS/$dest"
}

for N in $NS; do
  for T in $TS; do
    JT=${JLTYPE[$T]}
    echo "#### KF  $JT  $N ####"
    KF_N=$N KF_T=$T profile $REP/kf_${T}_${N} "KernelForge__${JT}__${N}__stats.csv" \
        $JULIA --project=$KF/test/envs/cuda $DRIVER
    echo "#### CUB  $JT  $N ####"
    profile $REP/cub_${T}_${N} "CUB__${JT}__${N}__stats.csv" \
        $CUB -n $N -t $T -i 20 -w 100
  done
done

echo "==== PARSE -> sort_kernels_A100.csv ===="
KF_STATS_DIR=$STATS KF_OUT=$WORK/sort_kernels_A100.csv KF_PROFILER=nsys \
    $JULIA --project=$KF/test/envs/cuda perfs/julia/benchmarks/parse_sort_kernels.jl 2>&1 | tail -3
echo "==== RESULT ===="
cat $WORK/sort_kernels_A100.csv 2>/dev/null
echo "=== DONE ==="
