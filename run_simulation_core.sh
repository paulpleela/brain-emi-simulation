#!/bin/bash
#SBATCH --job-name=gprmax_seq
#SBATCH --output=logs/seq_%j.out
#SBATCH --error=logs/seq_%j.err
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Low-storage sequential pipeline:
#   1) Generate ONE scenario's 16 input files from metadata
#   2) Run all 16 TX simulations for that scenario
#   3) Extract .s16p
#   4) Delete that scenario's .in/.out files
#   5) Move to next scenario
#   6) Build each scenario's frequency-domain tensor immediately after extraction
#   7) After range completes, refresh train-only normalization stats
#
# Default mode processes scenario 1 only.
#
# Usage examples:
#   mkdir -p logs
#   sbatch run_simulation_core.sh
#
#   # Process scenario range 1..20 on CPU
#   sbatch --export=ALL,START_SCENARIO=1,END_SCENARIO=20 run_simulation_core.sh
#
#   # GPU mode
#   sbatch --partition=a100 --gres=gpu:1 \
#          --export=ALL,START_SCENARIO=1,END_SCENARIO=20,USE_GPU=1 \
#          run_simulation_core.sh
#
# Tunables via environment variables:
#   START_SCENARIO (default 1)
#   END_SCENARIO   (default START_SCENARIO)
#   USE_GPU        (0/1, default 1)
#   DELETE_OUT     (0/1, default 1)
#   DELETE_IN      (0/1, default 1)

set -euo pipefail

START_SCENARIO="${START_SCENARIO:-1}"
END_SCENARIO="${END_SCENARIO:-$START_SCENARIO}"
USE_GPU="${USE_GPU:-1}"
DELETE_OUT="${DELETE_OUT:-1}"
DELETE_IN="${DELETE_IN:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-gprmax}"
METADATA_FILE="${METADATA_FILE:-dataset_metadata_v3.csv}"
SKIP_FINAL_STATS="${SKIP_FINAL_STATS:-0}"

if [[ "$START_SCENARIO" -gt "$END_SCENARIO" ]]; then
  echo "ERROR: START_SCENARIO must be <= END_SCENARIO"
  exit 1
fi

echo "========================================"
echo "gprMax Sequential Scenario Runner"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Scenario range: ${START_SCENARIO}..${END_SCENARIO}"
echo "USE_GPU=${USE_GPU} DELETE_IN=${DELETE_IN} DELETE_OUT=${DELETE_OUT}"
echo "METADATA_FILE=${METADATA_FILE}"
echo "Start time: $(date)"
echo "========================================"

if [[ ! -f "${METADATA_FILE}" ]]; then
  echo "ERROR: metadata file not found: ${METADATA_FILE}"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
if ! python - <<'PY' >/dev/null 2>&1
import numpy
PY
then
  echo "ERROR: numpy is missing in conda env '${CONDA_ENV_NAME}'."
  echo "Install it with: conda activate ${CONDA_ENV_NAME} && conda install -y numpy"
  exit 1
fi
if [[ -d "${PWD}/gprMax/gprMax" ]]; then
  export PYTHONPATH="${PWD}/gprMax:${PYTHONPATH:-}"
fi
PYTHON="$(command -v python)"
GPRMAX_MODULE="${GPRMAX_MODULE:-gprMax}"

if [[ "$USE_GPU" == "1" ]]; then
  module load cuda/12.2 || true
fi

mkdir -p brain_inputs sparams fd_tensors logs

for sid in $(seq "$START_SCENARIO" "$END_SCENARIO"); do
  sid_pad=$(printf "%03d" "$sid")
  echo ""
  echo "--- Scenario ${sid_pad} ---"

  # 1) Generate exactly this scenario from metadata
  "$PYTHON" generate_dataset.py --scenario "$sid" --metadata "${METADATA_FILE}"

  # 2) Run all 16 TX simulations
  for tx in $(seq -w 1 16); do
    input_file="brain_inputs/scenario_${sid_pad}_tx${tx}.in"
    if [[ ! -f "$input_file" ]]; then
      echo "ERROR: missing input file: $input_file"
      exit 1
    fi

    echo "Running ${input_file}"
    if [[ "$USE_GPU" == "1" ]]; then
      "$PYTHON" -m "$GPRMAX_MODULE" "$input_file" -n 1 -gpu
    else
      export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
      "$PYTHON" -m "$GPRMAX_MODULE" "$input_file" -n 1
    fi
  done

  # 3) Extract S-parameters
  "$PYTHON" build_s16p.py --scenario "$sid" --no-delete

  # 4) Build this scenario's FD tensor immediately.
  "$PYTHON" build_fd_tensors.py --scenario "$sid" --metadata "${METADATA_FILE}"

  # 5) Delete intermediate files for this scenario
  if [[ "$DELETE_IN" == "1" ]]; then
    rm -f "brain_inputs/scenario_${sid_pad}_tx"*.in
  fi

  if [[ "$DELETE_OUT" == "1" ]]; then
    rm -f "brain_inputs/scenario_${sid_pad}_tx"*.out
  fi

  echo "Scenario ${sid_pad} complete"
done

# Refresh train-only normalization stats after the range finishes.
if [[ "$SKIP_FINAL_STATS" != "1" ]]; then
  "$PYTHON" build_fd_tensors.py --fit-stats --fit-only --metadata "${METADATA_FILE}"
fi

echo ""
echo "========================================"
echo "All scenarios complete"
echo "End time: $(date)"
echo "Outputs: sparams/scenario_XXX.s16p"
echo "         fd_tensors/scenario_XXX_fd.npz"
echo "         fd_tensors/normalization_freq_full.npz"
echo "========================================"
