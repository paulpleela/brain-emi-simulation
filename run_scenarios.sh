#!/bin/bash

# Friendly wrapper for sequential scenario runs.
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
# Default behavior submits to SLURM via sbatch (GPU by default).

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
  --local             Run directly on current machine (no sbatch)
  --keep-in           Keep generated .in files (default deletes)
  --keep-out          Keep generated .out files (default deletes)
  -h, --help          Show this help

Notes:
  - Sequential execution is performed by run_simulation_core.sh.
  - In SLURM mode, this wrapper submits one job with START_SCENARIO/END_SCENARIO.
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
RUN_LOCAL=0
DELETE_IN=1
DELETE_OUT=1

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

if [[ ! -f "run_simulation_core.sh" ]]; then
  echo "ERROR: run_simulation_core.sh not found. Run from repository root."
  exit 1
fi

echo "Mode: ${MODE}"
echo "Range: ${START_SCENARIO}..${END_SCENARIO}"
echo "GPU: ${USE_GPU}"
echo "Delete .in: ${DELETE_IN}"
echo "Delete .out: ${DELETE_OUT}"

if [[ "$RUN_LOCAL" == "1" ]]; then
  echo "Launching locally..."
  START_SCENARIO="$START_SCENARIO" \
  END_SCENARIO="$END_SCENARIO" \
  USE_GPU="$USE_GPU" \
  DELETE_IN="$DELETE_IN" \
  DELETE_OUT="$DELETE_OUT" \
  bash run_simulation_core.sh
else
  export_vars="ALL,START_SCENARIO=${START_SCENARIO},END_SCENARIO=${END_SCENARIO},USE_GPU=${USE_GPU},DELETE_IN=${DELETE_IN},DELETE_OUT=${DELETE_OUT}"

  mkdir -p logs

  if [[ "$USE_GPU" == "1" ]]; then
    echo "Submitting GPU SLURM job..."
    sbatch --partition=a100 --gres=gpu:1 --export="$export_vars" run_simulation_core.sh
  else
    echo "Submitting CPU SLURM job..."
    sbatch --export="$export_vars" run_simulation_core.sh
  fi
fi
