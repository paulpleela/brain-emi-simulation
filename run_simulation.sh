#!/bin/bash
#SBATCH --job-name=gprmax_brain_cpu
#SBATCH --output=logs/sim_%a.out
#SBATCH --error=logs/sim_%a.err
#SBATCH --array=1-4800%16
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# CPU-based HPC job array script for gprMax brain imaging on Rangpur
# 300 scenarios x 16 transmit antennas = 4800 jobs
# Expected runtime: 45-60 min per job
# Runs max 16 parallel jobs at once
# Uses #transmission_line for accurate S-parameter extraction
#
# IMPORTANT: Adjust --array parameter based on batch:
#   Batch 1 (healthy): --array=1-160%16     (10 scenarios × 16 files)
#   Batch 2 (hemorrhage): --array=1-512%16  (32 scenarios × 16 files)
#   Batch 3 (rotation): --array=1-160%16    (10 scenarios × 16 files)
#
# Usage:
#   1. Generate inputs: python generate_inputs.py
#   2. Upload brain_inputs/ to HPC
#   3. Create logs/ directory: mkdir -p logs
#   4. Edit --array parameter above for current batch
#   5. Submit: sbatch run_simulation.sh
#   6. Monitor: squeue -u $USER
#   7. Download .out files for S-parameter extraction

echo "========================================"
echo "gprMax Brain EMI Simulation"
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"
echo ""

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
    # If conda is available in PATH, activate directly
    conda activate gprmax || true
else
    # Fallback: source common conda profile locations then activate
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate gprmax || true
fi

# Find input file based on array task ID
INPUT_DIR="brain_inputs"

# Get list of all input files sorted
INPUT_FILES=($(ls $INPUT_DIR/scenario_*_tx*.in | sort))

# Select the file for this task (array index starts at 1, subtract 1 for 0-based array)
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
INPUT_FILE="${INPUT_FILES[$TASK_INDEX]}"

echo "Task $SLURM_ARRAY_TASK_ID processing: $INPUT_FILE"

# Check input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

echo ""
echo "Running gprMax on CPU with $SLURM_CPUS_PER_TASK threads..."
echo ""

# Run gprMax on CPU using multiple threads
python -m gprMax "$INPUT_FILE" -n $SLURM_CPUS_PER_TASK

# Check if simulation completed successfully
OUTPUT_FILE="${INPUT_FILE%.in}.out"
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✓ Simulation completed successfully"
    echo "Output file: $OUTPUT_FILE"
    echo "File size: $(du -h $OUTPUT_FILE | cut -f1)"
else
    echo ""
    echo "✗ ERROR: Output file not created"
    exit 1
fi

echo ""
echo "End time: $(date)"
echo "========================================"
