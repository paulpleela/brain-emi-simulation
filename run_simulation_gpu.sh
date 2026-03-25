#!/bin/bash
#SBATCH --job-name=gprmax_seq_gpu
#SBATCH --output=logs/sim_seq_gpu_%j.out
#SBATCH --error=logs/sim_seq_gpu_%j.err
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Default simulation entrypoint (GPU): low-storage sequential mode.
#
# Processes one scenario at a time:
#   generate inputs -> run 16 TX sims on GPU -> extract s16p+npz -> delete intermediates
#
# Defaults to full dataset (1..1000).
# Override range at submit time with START_SCENARIO / END_SCENARIO.
#
# Examples:
#   sbatch run_simulation_gpu.sh
#   sbatch --export=ALL,START_SCENARIO=1,END_SCENARIO=20 run_simulation_gpu.sh

set -euo pipefail

export START_SCENARIO="${START_SCENARIO:-1}"
export END_SCENARIO="${END_SCENARIO:-1000}"
export USE_GPU=1
export DELETE_OUT="${DELETE_OUT:-1}"
export DELETE_IN="${DELETE_IN:-1}"

bash run_simulation_core.sh
