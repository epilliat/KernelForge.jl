#!/bin/bash
#SBATCH --job-name=sortaggA100
#SBATCH --account=ubc@a100
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fswork/projects/rech/ubc/utk53wr/logs/sortaggA100_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ubc/utk53wr/logs/sortaggA100_%j.out
source /lustre/fswork/projects/rech/ubc/utk53wr/kf_env.sh
# Aggregate sort.csv (KF/Base/AK/CUB, kernel + total) to a local results tree,
# pulled back and deposited in the sibling repo afterwards.
export KF_RESULTS_ROOT=$WORK/results
cd $KF
$JULIA --project=$KF/perfs/envs/benchenv/cuda perfs/julia/benchmarks/sort_perf_comparison.jl 2>&1 | grep -E ">>>|Kernel time|Total|===|CUB|KernelForge|Results saved|ERROR" | tail -60
echo "=== CSV ==="
cat $WORK/results/A100/sort.csv 2>/dev/null
echo "=== DONE ==="
