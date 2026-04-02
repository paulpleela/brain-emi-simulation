#!/bin/bash

# Friendly wrapper for scenario runs.
# Supports: single scenario, range, or all scenarios.
#
# Examples:
#   ./run_scenarios.sh --scenario 1
#   ./run_scenarios.sh --range 1 20
#   ./run_scenarios.sh --all
#   ./run_scenarios.sh --all
#   ./run_scenarios.sh --all --cpu
#   ./run_scenarios.sh --range 1 20 --local
#
# Default behavior submits to SLURM via sbatch (GPU + TX-parallel by default).

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_scenarios.sh --scenario N [options]
  ./run_scenarios.sh --range START END [options]
  ./run_scenarios.sh --all [options]

Selection (required, choose one):
  --scenario N        Run exactly one scenario
  --range START END   Run a sequential scenario range (inclusive)
  --all               Run all scenarios from dataset_metadata.csv

Options:
  --gpu               Use GPU mode (default)
  --cpu               Use CPU mode
  --tx-sequential     In GPU mode, disable TX-parallel and use single-GPU sequential mode
  --tx-concurrency N  In GPU mode, max concurrent TX tasks (optional cap)
  --tx-cpus N         In GPU mode, CPUs per TX task (default: 2)
  --local             Run directly on current machine (no sbatch)
  --keep-in           Keep generated .in files (default deletes)
  --keep-out          Keep generated .out files (default deletes)
  --keep-slurm-logs   Keep per-scenario SLURM tx/finalize logs (default deletes on success)
  -h, --help          Show this help

Notes:
  - Default GPU mode: each scenario runs 16 TX jobs in parallel (one GPU per TX),
    while scenarios remain sequential.
  - GPU mode uses a queued CPU finalize job dependent on TX completion.
  - CPU mode or --tx-sequential uses run_simulation_core.sh.
EOF
}

is_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

get_total_scenarios() {
  local metadata="dataset_metadata.csv"
  if [[ -f "$metadata" ]]; then
    local lines
    lines=$(wc -l < "$metadata")
    if is_int "$lines" && [[ "$lines" -ge 2 ]]; then
      echo $((lines - 1))
      return
    fi
  fi
  echo 1000
}

MODE=""
START_SCENARIO=""
END_SCENARIO=""
USE_GPU=1
GPU_TX_PARALLEL=1
TX_CONCURRENCY=""
TX_CPUS=2
RUN_LOCAL=0
DELETE_IN=1
DELETE_OUT=1
DELETE_SLURM_LOGS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario)
      [[ $# -ge 2 ]] || { echo "ERROR: --scenario requires a value"; usage; exit 1; }
      MODE="scenario"
      START_SCENARIO="$2"
      END_SCENARIO="$2"
      shift 2
      ;;
    --range)
      [[ $# -ge 3 ]] || { echo "ERROR: --range requires START and END"; usage; exit 1; }
      MODE="range"
      START_SCENARIO="$2"
      END_SCENARIO="$3"
      shift 3
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --gpu)
      USE_GPU=1
      shift
      ;;
    --cpu)
      USE_GPU=0
      shift
      ;;
    --tx-sequential)
      GPU_TX_PARALLEL=0
      shift
      ;;
    --tx-concurrency)
      [[ $# -ge 2 ]] || { echo "ERROR: --tx-concurrency requires a value"; usage; exit 1; }
      TX_CONCURRENCY="$2"
      shift 2
      ;;
    --tx-cpus)
      [[ $# -ge 2 ]] || { echo "ERROR: --tx-cpus requires a value"; usage; exit 1; }
      TX_CPUS="$2"
      shift 2
      ;;
    --local)
      RUN_LOCAL=1
      shift
      ;;
    --keep-in)
      DELETE_IN=0
      shift
      ;;
    --keep-out)
      DELETE_OUT=0
      shift
      ;;
    --keep-slurm-logs)
      DELETE_SLURM_LOGS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "ERROR: one of --scenario, --range, or --all is required"
  usage
  exit 1
fi

if [[ "$MODE" == "all" ]]; then
  total=$(get_total_scenarios)
  START_SCENARIO=1
  END_SCENARIO="$total"
fi

if ! is_int "$START_SCENARIO" || ! is_int "$END_SCENARIO"; then
  echo "ERROR: scenario values must be non-negative integers"
  exit 1
fi

if [[ "$START_SCENARIO" -lt 1 || "$END_SCENARIO" -lt 1 ]]; then
  echo "ERROR: scenario values must be >= 1"
  exit 1
fi

if [[ "$START_SCENARIO" -gt "$END_SCENARIO" ]]; then
  echo "ERROR: START_SCENARIO must be <= END_SCENARIO"
  exit 1
fi

if ! is_int "$TX_CONCURRENCY" || [[ "$TX_CONCURRENCY" -lt 1 ]]; then
  if [[ -n "$TX_CONCURRENCY" ]]; then
    echo "ERROR: --tx-concurrency must be an integer >= 1"
    exit 1
  fi
fi

if ! is_int "$TX_CPUS" || [[ "$TX_CPUS" -lt 1 ]]; then
  echo "ERROR: --tx-cpus must be an integer >= 1"
  exit 1
fi

if [[ ! -f "run_simulation_core.sh" ]]; then
  echo "ERROR: run_simulation_core.sh not found. Run from repository root."
  exit 1
fi

echo "Mode: ${MODE}"
echo "Range: ${START_SCENARIO}..${END_SCENARIO}"
echo "GPU: ${USE_GPU}"
echo "GPU TX parallel: ${GPU_TX_PARALLEL}"
if [[ -n "$TX_CONCURRENCY" ]]; then
  echo "TX concurrency cap: ${TX_CONCURRENCY}"
else
  echo "TX concurrency cap: scheduler-managed"
fi
echo "TX cpus-per-task: ${TX_CPUS}"
echo "Delete .in: ${DELETE_IN}"
echo "Delete .out: ${DELETE_OUT}"
echo "Delete SLURM tx/finalize logs on success: ${DELETE_SLURM_LOGS}"

FIRST_SCENARIO="${FIRST_SCENARIO:-$START_SCENARIO}"

submit_gpu_parallel_pipeline() {
  local sid="$START_SCENARIO"
  local sid_pad
  sid_pad=$(printf "%03d" "$sid")

  mkdir -p logs

  # Generate the first scenario inputs immediately (no queued prep job).
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate gprmax
  python generate_dataset.py --scenario "$sid"

  local tx_cmd
  tx_cmd=$(cat <<EOF
set -euo pipefail
cd "${PWD}"
  source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate gprmax
  module load cuda/12.2 || true
  PYTHON="\$(command -v python)"
TX=\$(printf "%02d" \${SLURM_ARRAY_TASK_ID})
  "\$PYTHON" -m gprMax brain_inputs/scenario_${sid_pad}_tx\${TX}.in -n 1 -gpu
EOF
)

  local array_spec="1-16"
  if [[ -n "${TX_CONCURRENCY}" ]]; then
    array_spec="1-16%${TX_CONCURRENCY}"
  fi

  local tx_job
  tx_job=$(sbatch --parsable \
    --partition=a100 --gres=gpu:1 --cpus-per-task=${TX_CPUS} \
    --array=${array_spec} \
    --job-name="tx_s${sid_pad}" \
    --output="logs/tx_s${sid_pad}_%A_%a.out" \
    --error="logs/tx_s${sid_pad}_%A_%a.err" \
    --wrap "$tx_cmd")

  local finalize_cmd
  finalize_cmd=$(cat <<EOF
set -euo pipefail
cd "${PWD}"
  source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate gprmax
  PYTHON="\$(command -v python)"

TX_JOB_ID="${tx_job}"
TX_STATE=\$(sacct -n -X -j "\$TX_JOB_ID" --format=State 2>/dev/null | head -n 1 | awk '{print \$1}')
if [[ -z "\$TX_STATE" ]]; then
  TX_STATE=\$(scontrol show job "\$TX_JOB_ID" 2>/dev/null | awk -F= '/JobState=/{print \$2; exit}' | awk '{print \$1}')
fi
if [[ "\$TX_STATE" != "COMPLETED" ]]; then
  echo "ERROR: TX array job \$TX_JOB_ID finished with state '\$TX_STATE' (expected COMPLETED)."
  echo "       Check TX logs before rerunning."
  exit 1
fi

"\$PYTHON" build_s16p.py --scenario ${sid} --no-delete
if [[ ! -f "sparams/scenario_${sid_pad}.s16p" ]]; then
  echo "ERROR: missing sparams/scenario_${sid_pad}.s16p after build_s16p"
  exit 1
fi

if [[ "${DELETE_IN}" == "1" ]]; then
  rm -f brain_inputs/scenario_${sid_pad}_tx*.in
fi
if [[ "${DELETE_OUT}" == "1" ]]; then
  rm -f brain_inputs/scenario_${sid_pad}_tx*.out
fi

if [[ "${DELETE_SLURM_LOGS}" == "1" ]]; then
  rm -f logs/tx_s${sid_pad}_${tx_job}_*.out logs/tx_s${sid_pad}_${tx_job}_*.err
  rm -f "logs/final_s${sid_pad}_\${SLURM_JOB_ID}.out" "logs/final_s${sid_pad}_\${SLURM_JOB_ID}.err"
fi

NEXT_SCENARIO=\$(( ${sid} + 1 ))
if [[ "\$NEXT_SCENARIO" -le "${END_SCENARIO}" ]]; then
  NEXT_ARGS="--range \"\$NEXT_SCENARIO\" \"${END_SCENARIO}\" --gpu --tx-cpus ${TX_CPUS}"
  if [[ -n "${TX_CONCURRENCY}" ]]; then
    NEXT_ARGS="\$NEXT_ARGS --tx-concurrency ${TX_CONCURRENCY}"
  fi
  if [[ "${DELETE_IN}" == "0" ]]; then
    NEXT_ARGS="\$NEXT_ARGS --keep-in"
  fi
  if [[ "${DELETE_OUT}" == "0" ]]; then
    NEXT_ARGS="\$NEXT_ARGS --keep-out"
  fi
  if [[ "${DELETE_SLURM_LOGS}" == "0" ]]; then
    NEXT_ARGS="\$NEXT_ARGS --keep-slurm-logs"
  fi
  eval "FIRST_SCENARIO=${FIRST_SCENARIO} bash run_scenarios.sh \$NEXT_ARGS"
else
  "\$PYTHON" build_fd_tensors.py --range ${FIRST_SCENARIO} ${END_SCENARIO} --fit-stats
  if [[ ! -f "fd_tensors/scenario_${sid_pad}_fd.npz" ]]; then
    echo "ERROR: missing fd_tensors/scenario_${sid_pad}_fd.npz after build_fd_tensors"
    exit 1
  fi
  if [[ ! -f "fd_tensors/normalization_freq_full.npz" ]]; then
    echo "ERROR: missing fd_tensors/normalization_freq_full.npz after build_fd_tensors"
    exit 1
  fi
fi
EOF
)

  local final_job
  final_job=$(sbatch --parsable \
    --partition=cpu --cpus-per-task=1 \
    --job-name="final_s${sid_pad}" \
    --output="logs/final_s${sid_pad}_%j.out" \
    --error="logs/final_s${sid_pad}_%j.err" \
    --dependency="afterany:${tx_job}" \
    --wrap "$finalize_cmd")

  echo "Submitted rolling GPU chain for scenario ${sid_pad}."
  echo "TX array job: ${tx_job}"
  echo "Finalize job: ${final_job}"
  echo "Next scenario is submitted by finalize job after successful extraction."
}

if [[ "$RUN_LOCAL" == "1" ]]; then
  if [[ "$USE_GPU" == "1" && "$GPU_TX_PARALLEL" == "1" ]]; then
    echo "NOTE: --local cannot fan out to 16 GPUs via SLURM arrays; falling back to sequential core runner."
  fi
  echo "Launching locally..."
  START_SCENARIO="$START_SCENARIO" \
  END_SCENARIO="$END_SCENARIO" \
  USE_GPU="$USE_GPU" \
  DELETE_IN="$DELETE_IN" \
  DELETE_OUT="$DELETE_OUT" \
  bash run_simulation_core.sh
else
  if [[ "$USE_GPU" == "1" && "$GPU_TX_PARALLEL" == "1" ]]; then
    submit_gpu_parallel_pipeline
  else
    export_vars="ALL,START_SCENARIO=${START_SCENARIO},END_SCENARIO=${END_SCENARIO},USE_GPU=${USE_GPU},DELETE_IN=${DELETE_IN},DELETE_OUT=${DELETE_OUT}"
    mkdir -p logs
    if [[ "$USE_GPU" == "1" ]]; then
      echo "Submitting GPU sequential SLURM job (--tx-sequential)..."
      sbatch --partition=a100 --gres=gpu:1 --export="$export_vars" run_simulation_core.sh
    else
      echo "Submitting CPU SLURM job..."
      sbatch --export="$export_vars" run_simulation_core.sh
    fi
  fi
fi
