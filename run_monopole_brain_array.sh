#!/bin/bash
#SBATCH --job-name=gprmax_brain_monopole
#SBATCH --output=logs/monopole_%a.out
#SBATCH --error=logs/monopole_%a.err
#SBATCH --array=1-16
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# HPC job array script for gprMax brain monopole simulations
# This runs 16 simulations in parallel (one per transmit antenna)
#
# Usage:
#   1. Upload brain_monopole_simulations/ to HPC
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
# module load openmpi

# Activate conda environment (if using conda)
# source activate gprmax

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

# Run gprMax
echo ""
echo "Running gprMax..."
echo ""

# For CPU-only execution
gprmax "$INPUT_FILE" -n 8

# For GPU execution (if available, uncomment and adjust):
# gprmax "$INPUT_FILE" -gpu 0

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
