#!/bin/bash
#SBATCH --job-name=samplesortA100
#SBATCH --account=ubc@a100
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:50:00
#SBATCH --output=/lustre/fswork/projects/rech/ubc/utk53wr/logs/samplesortA100_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ubc/utk53wr/logs/samplesortA100_%j.out
source /lustre/fswork/projects/rech/ubc/utk53wr/kf_env.sh
export KF_RESULTS_ROOT=$WORK/results
cd $KF
$JULIA --project=$KF/perfs/envs/benchenv/cuda perfs/julia/benchmarks/sample_sort_perf_comparison.jl 2>&1 | grep -E ">>>|correctness|Sample sort:|Results saved|ERROR|Warning" | tail -60
echo "=== CSV ==="
cat $WORK/results/A100/sample_sort.csv 2>/dev/null
echo "=== DONE ==="
