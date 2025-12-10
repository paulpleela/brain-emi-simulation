#!/bin/bash
#SBATCH --job-name=gprmax_brain_gpu
#SBATCH --output=logs/monopole_%a.out
#SBATCH --error=logs/monopole_%a.err
#SBATCH --array=1-16
#SBATCH --partition=a100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# GPU-accelerated HPC job array script for gprMax brain imaging on Rangpur A100
# Expected runtime: 3-5 min per job (vs 45 min on CPU)
# This runs 16 simulations in parallel (one per transmit antenna)
#
# Rangpur HPC info:
#   - a100: GPU partition with NVIDIA A100 40GB GPUs (typical speedup: 15-25x vs CPU)
#   - a100-test: GPU dev/test partition (20 min max, use for testing)
#   - cpu/vcpu: CPU-only partitions (fallback if GPU unavailable)
#
# Usage:
#   1. Install pycuda: conda activate gprmax; conda install -c conda-forge pycuda
#   2. Upload brain_monopole_realistic/ to HPC
#   3. Create logs/ directory: mkdir -p logs
#   4. Submit: sbatch run_monopole_brain_array.sh
#   5. Monitor: squeue -u $USER
#   6. After completion, download all .out files for S-parameter extraction

echo "========================================"
echo "gprMax GPU Brain Imaging (0-2 GHz)"
echo "Array job ID: $SLURM_ARRAY_TASK_ID / 16"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Load CUDA module - try available versions on Rangpur
echo "Loading CUDA module..."
if module load cuda/12.2 2>/dev/null; then
    echo "Loaded CUDA 12.2"
elif module load cuda/11.4 2>/dev/null; then
    echo "Loaded CUDA 11.4"
elif module load cuda/11.1 2>/dev/null; then
    echo "Loaded CUDA 11.1"
else
    echo "WARNING: Could not load CUDA module, trying manual PATH setup..."
    # Manually add CUDA 11.4 to PATH (most compatible)
    export PATH="/usr/local/cuda-11.4/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"
    echo "Manually added CUDA 11.4 to PATH"
fi

# Verify nvcc is available
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found: $(which nvcc)"
    nvcc --version | head -n 1
else
    echo "✗ ERROR: nvcc still not found in PATH"
    echo "PATH: $PATH"
    exit 1
fi
echo ""
    fi
fi
echo ""

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

echo "Input file: $INPUT_FILE"

# Check input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# Verify GPU availability
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "Warning: nvidia-smi not available"
echo ""

# Run gprMax on GPU
echo "Running gprMax on GPU 0..."
echo ""

# Run gprMax using GPU (use -gpu 0 for first GPU)
python -m gprMax "$INPUT_FILE" -gpu 0

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
