#!/bin/bash
#SBATCH --job-name=gprmax_boundary_submit
#SBATCH --output=logs/boundary_submit_%j.out
#SBATCH --error=logs/boundary_submit_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# GPU scenario-parallel submitter for the additional valid in-head boundary
# scenarios generated in dataset_metadata_boundary_valid.csv.
#
# Default:
#   scenarios 1001..1050 from dataset_metadata_boundary_valid.csv
#
# Usage:
#   mkdir -p logs
#   sbatch run_boundary_scenarios_gpu.sh
#
# Optional overrides:
#   sbatch --export=ALL,START_SCENARIO=1001,END_SCENARIO=1100 run_boundary_scenarios_gpu.sh
#   sbatch --export=ALL,MAX_CONCURRENT_SCENARIOS=12,GPU_TIME_LIMIT=24:00:00 run_boundary_scenarios_gpu.sh
#
# By default this script does NOT refit normalization statistics. It builds new
# FD tensors using the existing fd_tensors/normalization_freq_full.npz from the
# original train split. Set REFIT_STATS=1 only if you intentionally want stats
# fit from this boundary-only metadata file.

set -euo pipefail

METADATA_FILE="${METADATA_FILE:-dataset_metadata_boundary_valid.csv}"
START_SCENARIO="${START_SCENARIO:-1001}"
END_SCENARIO="${END_SCENARIO:-1050}"
SCENARIO_OFFSET="${SCENARIO_OFFSET:-1000}"
MAX_CONCURRENT_SCENARIOS="${MAX_CONCURRENT_SCENARIOS:-8}"
GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-8}"
GPU_TIME_LIMIT="${GPU_TIME_LIMIT:-24:00:00}"
DELETE_OUT="${DELETE_OUT:-1}"
DELETE_IN="${DELETE_IN:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-gprmax}"
GPRMAX_MODULE="${GPRMAX_MODULE:-gprMax}"
REFIT_STATS="${REFIT_STATS:-0}"

if [[ ! -f "$METADATA_FILE" ]]; then
  echo "ERROR: metadata file not found: ${METADATA_FILE}"
  echo "Generate it first with: python generate_boundary_metadata.py"
  exit 1
fi

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
echo "Valid Boundary Scenario GPU Submitter"
echo "Metadata file: ${METADATA_FILE}"
echo "Scenario range: ${START_SCENARIO}..${END_SCENARIO}"
echo "Array task IDs: 1..$((END_SCENARIO - START_SCENARIO + 1)) mapped with SCENARIO_OFFSET=${SCENARIO_OFFSET}"
echo "Max concurrent scenarios: ${MAX_CONCURRENT_SCENARIOS}"
echo "GPU CPUs per task: ${GPU_CPUS_PER_TASK}"
echo "GPU time limit: ${GPU_TIME_LIMIT}"
echo "DELETE_IN=${DELETE_IN} DELETE_OUT=${DELETE_OUT}"
echo "REFIT_STATS=${REFIT_STATS}"
echo "Node: ${SLURM_NODELIST:-submit}"
echo "Start time: $(date)"
echo "========================================"

ARRAY_COUNT=$((END_SCENARIO - START_SCENARIO + 1))
if [[ "$ARRAY_COUNT" -lt 1 ]]; then
  echo "ERROR: scenario range is empty"
  exit 1
fi

scenario_job_id=$(sbatch --parsable \
  --partition=a100 \
  --time="${GPU_TIME_LIMIT}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${GPU_CPUS_PER_TASK}" \
  --gres=gpu:1 \
  --array="1-${ARRAY_COUNT}%${MAX_CONCURRENT_SCENARIOS}" \
  --job-name="gprmax_boundary_gpu" \
  --output="logs/boundary_gpu_%A_%a.out" \
  --error="logs/boundary_gpu_%A_%a.err" \
  --export="ALL,USE_GPU=1,DELETE_IN=${DELETE_IN},DELETE_OUT=${DELETE_OUT},RUN_BUILD_FD=0,RUN_FIT_STATS=0,METADATA_FILE=${METADATA_FILE},CONDA_ENV_NAME=${CONDA_ENV_NAME},GPRMAX_MODULE=${GPRMAX_MODULE},SCENARIO_OFFSET=${SCENARIO_OFFSET}" \
  --wrap='sid=$((SCENARIO_OFFSET + SLURM_ARRAY_TASK_ID)); START_SCENARIO=${sid} END_SCENARIO=${sid} bash run_simulation_core.sh')

if [[ "$REFIT_STATS" == "1" ]]; then
  post_cmd="python build_fd_tensors.py --metadata '${METADATA_FILE}' --range ${START_SCENARIO} ${END_SCENARIO} --fit-stats"
else
  post_cmd="if [[ ! -f fd_tensors/normalization_freq_full.npz ]]; then echo 'ERROR: missing fd_tensors/normalization_freq_full.npz. Run the original training-set extraction first, or submit with REFIT_STATS=1.'; exit 1; fi; python build_fd_tensors.py --metadata '${METADATA_FILE}' --range ${START_SCENARIO} ${END_SCENARIO}"
fi

post_job_id=$(sbatch --parsable \
  --partition=cpu \
  --time=12:00:00 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --dependency="afterok:${scenario_job_id}" \
  --job-name="boundary_fd_post" \
  --output="logs/boundary_post_%j.out" \
  --error="logs/boundary_post_%j.err" \
  --export="ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME}" \
  --wrap="set -euo pipefail; cd '${PWD}'; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV_NAME}'; ${post_cmd}")

echo "Submitted boundary scenario array job: ${scenario_job_id}"
echo "Submitted dependent FD post job: ${post_job_id}"
echo "Monitor with: squeue -j ${scenario_job_id},${post_job_id}"
