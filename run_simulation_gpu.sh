#!/bin/bash
#SBATCH --job-name=gprmax_brain_gpu
#SBATCH --output=logs/sim_gpu_%a.out
#SBATCH --error=logs/sim_gpu_%a.err
#SBATCH --array=1-1000%32
#SBATCH --partition=a100
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# GPU-accelerated HPC job array script for gprMax brain imaging on Rangpur
#
# Partition:  a100        (production — full A100 GPU nodes)
# For testing first: srun -p a100-test --gres=shard:1 --pty bash
# 300 scenarios x 16 transmit antennas = 4800 jobs total
# At 32 parallel GPU slots: ~4800/32 * 4 min ≈ 10 hours total
#
# Submit in 5 batches of 1000 (Rangpur caps arrays at 1000):
#   Batch 1: --array=1-1000%32
#   Batch 2: --array=1001-2000%32
#   Batch 3: --array=2001-3000%32
#   Batch 4: --array=3001-4000%32
#   Batch 5: --array=4001-4800%32
#
# Usage:
#   conda activate gprmax
#   python generate_dataset.py
#   mkdir -p logs
#   sbatch run_simulation_gpu.sh                        # batch 1 (1-1000)
#   sbatch --array=1001-2000%32 run_simulation_gpu.sh   # batch 2
#   sbatch --array=2001-3000%32 run_simulation_gpu.sh   # batch 3
#   sbatch --array=3001-4000%32 run_simulation_gpu.sh   # batch 4
#   sbatch --array=4001-4800%32 run_simulation_gpu.sh   # batch 5
#
# Requires: pycuda installed in gprmax conda environment
#   pip install pycuda
# Check available GPU nodes on Rangpur:
#   sinfo -p gpu

echo "========================================"
echo "gprMax Brain EMI Simulation (GPU)"
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================"
echo ""

# Load the CUDA module so nvcc is available for PyCUDA
module load cuda/12.2

# Activate conda environment (robust method for SLURM non-interactive shells)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gprmax

# Use the conda env Python directly as a fallback
PYTHON="$HOME/miniconda3/envs/gprmax/bin/python"

echo "Python executable: $PYTHON"
$PYTHON - << 'PY'
import gprMax
import gprMax.gprMax as gm
print('gprMax package path:', gprMax.__file__)
print('gprMax solver module path:', gm.__file__)
PY

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
echo "Running gprMax on GPU 0..."
echo ""

# Run gprMax with GPU acceleration
# -gpu flag enables CUDA solver; default device ID 0 (single GPU per job)
$PYTHON -m gprMax "$INPUT_FILE" -n 1 -gpu

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
echo "NOTE: Automatic S-parameter extraction is disabled in GPU script."
echo "Run extract_sparameters.py manually after confirming all 16 TX outputs exist for a scenario."

echo ""
echo "End time: $(date)"
echo "========================================"
