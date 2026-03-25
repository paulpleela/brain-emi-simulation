#!/bin/bash
#SBATCH --job-name=gprmax_seq_cpu
#SBATCH --output=logs/sim_seq_cpu_%j.out
#SBATCH --error=logs/sim_seq_cpu_%j.err
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Default simulation entrypoint (CPU): low-storage sequential mode.
#
# Processes one scenario at a time:
#   generate inputs -> run 16 TX sims -> extract s16p+npz -> delete intermediates
#
# Defaults to full dataset (1..1000).
# Override range at submit time with START_SCENARIO / END_SCENARIO.
#
# Examples:
#   sbatch run_simulation.sh
#   sbatch --export=ALL,START_SCENARIO=1,END_SCENARIO=20 run_simulation.sh

set -euo pipefail

export START_SCENARIO="${START_SCENARIO:-1}"
export END_SCENARIO="${END_SCENARIO:-1000}"
export USE_GPU=0
export DELETE_OUT="${DELETE_OUT:-1}"
export DELETE_IN="${DELETE_IN:-1}"

bash run_simulation_core.sh
