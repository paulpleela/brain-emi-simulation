#!/bin/bash
#SBATCH --job-name=gprmax_gpu_submit
#SBATCH --output=logs/sim_gpu_submit_%j.out
#SBATCH --error=logs/sim_gpu_submit_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# GPU orchestration entrypoint: scenario-parallel mode.
#
# This submitter creates:
# 1) One GPU array job where each task handles exactly one scenario end-to-end:
#    generate inputs -> run 16 TX on GPU -> build .s16p -> cleanup
# 2) One dependent CPU job that builds FD tensors + normalization stats for the selected range.
#
# Defaults: scenarios 1..1000, up to 8 scenarios running concurrently.
#
# Examples:
#   sbatch run_simulation_gpu.sh
#   sbatch --export=ALL,START_SCENARIO=1,END_SCENARIO=200,MAX_CONCURRENT_SCENARIOS=16 run_simulation_gpu.sh

set -euo pipefail

START_SCENARIO="${START_SCENARIO:-1}"
END_SCENARIO="${END_SCENARIO:-1000}"
MAX_CONCURRENT_SCENARIOS="${MAX_CONCURRENT_SCENARIOS:-8}"
GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-8}"
GPU_TIME_LIMIT="${GPU_TIME_LIMIT:-24:00:00}"
DELETE_OUT="${DELETE_OUT:-1}"
DELETE_IN="${DELETE_IN:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-gprmax}"
GPRMAX_MODULE="${GPRMAX_MODULE:-gprMax}"

if [[ "$START_SCENARIO" -gt "$END_SCENARIO" ]]; then
  echo "ERROR: START_SCENARIO must be <= END_SCENARIO"
  exit 1
fi

if [[ "$MAX_CONCURRENT_SCENARIOS" -lt 1 ]]; then
  echo "ERROR: MAX_CONCURRENT_SCENARIOS must be >= 1"
  exit 1
fi

mkdir -p logs

echo "========================================"
echo "GPU Scenario-Parallel Submitter"
echo "Scenario range: ${START_SCENARIO}..${END_SCENARIO}"
echo "Max concurrent scenarios: ${MAX_CONCURRENT_SCENARIOS}"
echo "GPU CPUs per task: ${GPU_CPUS_PER_TASK}"
echo "GPU time limit: ${GPU_TIME_LIMIT}"
echo "DELETE_IN=${DELETE_IN} DELETE_OUT=${DELETE_OUT}"
echo "Node: ${SLURM_NODELIST:-submit}"
echo "Start time: $(date)"
echo "========================================"

scenario_job_id=$(sbatch --parsable \
  --partition=a100 \
  --time="${GPU_TIME_LIMIT}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${GPU_CPUS_PER_TASK}" \
  --gres=gpu:1 \
  --array="${START_SCENARIO}-${END_SCENARIO}%${MAX_CONCURRENT_SCENARIOS}" \
  --job-name="gprmax_scn_gpu" \
  --output="logs/scenario_gpu_%A_%a.out" \
  --error="logs/scenario_gpu_%A_%a.err" \
  --export="ALL,USE_GPU=1,DELETE_IN=${DELETE_IN},DELETE_OUT=${DELETE_OUT},RUN_BUILD_FD=0,RUN_FIT_STATS=0,CONDA_ENV_NAME=${CONDA_ENV_NAME},GPRMAX_MODULE=${GPRMAX_MODULE}" \
  --wrap='sid=${SLURM_ARRAY_TASK_ID}; START_SCENARIO=${sid} END_SCENARIO=${sid} bash run_simulation_core.sh')

post_job_id=$(sbatch --parsable \
  --partition=cpu \
  --time=12:00:00 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --dependency="afterok:${scenario_job_id}" \
  --job-name="gprmax_fd_post" \
  --output="logs/gpu_post_%j.out" \
  --error="logs/gpu_post_%j.err" \
  --export="ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME}" \
  --wrap="set -euo pipefail; cd '${PWD}'; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV_NAME}'; python build_fd_tensors.py --range ${START_SCENARIO} ${END_SCENARIO} --fit-stats")

echo "Submitted scenario array job: ${scenario_job_id}"
echo "Submitted dependent FD post job: ${post_job_id}"
echo "Monitor with: squeue -j ${scenario_job_id},${post_job_id}"
