#!/bin/bash
#SBATCH --job-name=gprmax_brain_cpu
#SBATCH --output=logs/monopole_%a.out
#SBATCH --error=logs/monopole_%a.err
#SBATCH --array=1-16
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# CPU-based HPC job array script for gprMax brain imaging on Rangpur
# Expected runtime: 45-60 min per job
# This runs 16 simulations in parallel (one per transmit antenna)
# Uses #transmission_line for accurate S-parameter extraction
#
# Rangpur HPC info:
#
# Usage:
#   1. Upload brain_monopole_realistic/ to HPC
#   2. Create logs/ directory: mkdir -p logs
#   3. Submit: sbatch run_monopole_brain_array.sh
#   4. Monitor: squeue -u $USER
#   5. After completion, download all .out files for S-parameter extraction

echo "========================================"
echo "gprMax CPU Brain Imaging (0-2 GHz)"
echo "Array job ID: $SLURM_ARRAY_TASK_ID / 16"
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

# Input and output directories - Updated for realistic model
INPUT_DIR="brain_monopole_realistic"
INPUT_FILE="${INPUT_DIR}/brain_realistic_tx$(printf "%02d" $SLURM_ARRAY_TASK_ID).in"

echo "Input file: $INPUT_FILE"

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
