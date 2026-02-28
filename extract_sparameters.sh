#!/bin/bash
#SBATCH --job-name=extract_sparams
#SBATCH --output=logs/extract_%a.out
#SBATCH --error=logs/extract_%a.err
#SBATCH --array=1-300%50
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
# FALLBACK ONLY: S-parameter extraction is normally triggered automatically
# inside run_simulation.sh as each scenario's 16 jobs complete.
#
# Use this script only to re-extract or catch any scenarios that were missed:
#   sbatch extract_sparameters.sh

echo "========================================"
echo "S-Parameter Extraction"
echo "Scenario: $SLURM_ARRAY_TASK_ID / 300"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Activate conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gprmax
PYTHON="$HOME/miniconda3/envs/gprmax/bin/python"

mkdir -p logs sparams

# Each array task handles one scenario
SCENARIO=$SLURM_ARRAY_TASK_ID

echo "Extracting scenario $SCENARIO..."
$PYTHON extract_sparameters.py --scenario $SCENARIO

echo ""
echo "End time: $(date)"
echo "========================================"
