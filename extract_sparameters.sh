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
# Extracts S-parameters for all 300 scenarios and saves as .s16p
# Run AFTER all simulation batches are complete:
#
#   JOB1=$(sbatch --array=1-1000%16    run_simulation.sh | awk '{print $4}')
#   JOB2=$(sbatch --array=1001-2000%16 run_simulation.sh | awk '{print $4}')
#   JOB3=$(sbatch --array=2001-3000%16 run_simulation.sh | awk '{print $4}')
#   JOB4=$(sbatch --array=3001-4000%16 run_simulation.sh | awk '{print $4}')
#   JOB5=$(sbatch --array=4001-4800%16 run_simulation.sh | awk '{print $4}')
#   sbatch --dependency=afterok:$JOB1:$JOB2:$JOB3:$JOB4:$JOB5 extract_sparameters.sh
#
# Or run manually after all jobs complete:
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
