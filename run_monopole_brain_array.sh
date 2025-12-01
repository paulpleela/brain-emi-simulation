#!/bin/bash
#SBATCH --job-name=gprmax_brain_monopole
#SBATCH --output=logs/monopole_%a.out
#SBATCH --error=logs/monopole_%a.err
#SBATCH --array=1-16
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# HPC job array script for gprMax brain monopole simulations on Rangpur
# Configured for a100-grind partition (max 5 days, GPU enabled)
# This runs 16 simulations in parallel (one per transmit antenna)
#
# Rangpur HPC info:
#   - a100-grind: GPU partition for long jobs (up to 5 days, limited nodes)
#   - a100: GPU partition (1hr max before preemption, use --requeue)
#   - a100-test: GPU dev/test partition (20 min max)
#   - cpu/vcpu: CPU-only partitions
#
# Usage:
#   1. Upload brain_monopole_realistic/ to HPC
#   2. Create logs/ directory: mkdir -p logs
#   3. Submit: sbatch run_monopole_brain_array.sh
#   4. Monitor: squeue -u $USER
#   5. After completion, download all .out files for S-parameter extraction

echo "========================================"
echo "gprMax Realistic Brain Imaging (0-2 GHz)"
echo "Array job ID: $SLURM_ARRAY_TASK_ID / 16"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Load required modules (adjust for your HPC environment)
# module load python/3.11
# module load hdf5
# (No CUDA / GPU module required for CPU-only runs)

# Activate conda environment (if using conda)
# Prefer the user's existing environment name `gprmax`.
# Use a robust activation that works whether conda is initialized or not.
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
OUTPUT_DIR="brain_monopole_realistic"

echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"

# Check input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# Run gprMax (CPU-only)
echo ""
echo "Running gprMax in CPU-only mode using $SLURM_CPUS_PER_TASK threads..."
echo ""

# Run gprMax on CPU using the number of CPU threads allocated by Slurm
python -m gprMax "$INPUT_FILE" -n "$SLURM_CPUS_PER_TASK"

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
