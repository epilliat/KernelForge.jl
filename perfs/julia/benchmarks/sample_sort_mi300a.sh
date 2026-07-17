#!/bin/bash
#SBATCH --account=cad17552
#SBATCH --constraint=MI300
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=00:50:00
#SBATCH -o /home/epilliat/kf/samplesort_%j.out
export JULIA_DEPOT_PATH=/home/epilliat/kf/depot JULIA_AMDGPU_DISABLE_ARTIFACTS=1 ROCM_PATH=/opt/rocm-6.4.3
export PATH=/opt/rocm-6.4.3/bin:$PATH
export LD_LIBRARY_PATH=/home/epilliat/kf/julia-1.12.6/lib/julia:$LD_LIBRARY_PATH
export KF_RESULTS_ROOT=/home/epilliat/kf/results
cd /home/epilliat/kf/KernelForge.jl
JL=/home/epilliat/kf/julia-1.12.6/bin/julia
srun $JL --project=perfs/envs/benchenv/roc perfs/julia/benchmarks/sample_sort_perf_comparison.jl 2>&1 | grep -E ">>>|correctness|===|Sample sort:|Results saved|ERROR|WARN|Warning" | tail -80
echo "=== CSV ==="
cat /home/epilliat/kf/results/MI300A/sample_sort.csv 2>/dev/null
echo "=== DONE ==="
