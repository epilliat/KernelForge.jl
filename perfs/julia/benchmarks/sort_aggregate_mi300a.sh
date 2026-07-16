#!/bin/bash
#SBATCH --account=cad17552
#SBATCH --constraint=MI300
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH -o /home/epilliat/kf/sortagg_%j.out
export JULIA_DEPOT_PATH=/home/epilliat/kf/depot JULIA_AMDGPU_DISABLE_ARTIFACTS=1 ROCM_PATH=/opt/rocm-6.4.3
export PATH=/opt/rocm-6.4.3/bin:$PATH
export LD_LIBRARY_PATH=/home/epilliat/kf/julia-1.12.6/lib/julia:$LD_LIBRARY_PATH
# Write the aggregate sort.csv (KF/Base/AK/rocPRIM, kernel + total) to a local
# results tree; pulled back and deposited in the sibling repo afterwards.
export KF_RESULTS_ROOT=/home/epilliat/kf/results
cd /home/epilliat/kf/KernelForge.jl
JL=/home/epilliat/kf/julia-1.12.6/bin/julia
srun $JL --project=perfs/envs/benchenv/roc perfs/julia/benchmarks/sort_perf_comparison.jl 2>&1 | grep -E ">>>|Kernel time|Total|===|rocPRIM|KernelForge|Results saved|ERROR" | tail -60
echo "=== CSV ==="
cat /home/epilliat/kf/results/MI300A/sort.csv 2>/dev/null
echo "=== DONE ==="
