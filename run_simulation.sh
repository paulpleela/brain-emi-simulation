#!/bin/bash
#SBATCH --job-name=gprmax_brain
#SBATCH --output=logs/sim_%a.out
#SBATCH --error=logs/sim_%a.err
#SBATCH --array=1-1000%16
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# CPU-based HPC job array script for gprMax brain imaging on Rangpur
# 300 scenarios x 16 transmit antennas = 4800 jobs total
# Submit in 5 batches of 1000 (most clusters cap arrays at 1000):
#   Batch 1: --array=1-1000%16
#   Batch 2: --array=1001-2000%16
#   Batch 3: --array=2001-3000%16
#   Batch 4: --array=3001-4000%16
#   Batch 5: --array=4001-4800%16
# Expected runtime: 45-60 min per job
#
# Usage:
#   conda activate gprmax
#   python generate_dataset.py
#   mkdir -p logs
#   sbatch run_simulation.sh                        # batch 1 (1-1000)
#   sbatch --array=1001-2000%16 run_simulation.sh   # batch 2
#   sbatch --array=2001-3000%16 run_simulation.sh   # batch 3
#   sbatch --array=3001-4000%16 run_simulation.sh   # batch 4
#   sbatch --array=4001-4800%16 run_simulation.sh   # batch 5

echo "========================================"
echo "gprMax Brain EMI Simulation"
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"
echo ""

# Activate conda environment (robust method for SLURM non-interactive shells)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gprmax

# Use the conda env Python directly as a fallback
PYTHON="$HOME/miniconda3/envs/gprmax/bin/python"

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

# Set OpenMP threads to match allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run gprMax (-n 1 = single model run; threads controlled by OMP_NUM_THREADS)
$PYTHON -m gprMax "$INPUT_FILE" -n 1

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

# ── Per-scenario S-parameter extraction ──────────────────────────────────────
# Parse scenario number directly from filename (e.g. brain_inputs/scenario_042_tx07.in → 042)
BASENAME=$(basename "$INPUT_FILE")                         # scenario_042_tx07.in
SCENARIO_PAD=$(echo "$BASENAME" | cut -d_ -f2)            # 042  (zero-padded)
SCENARIO_NUM=$(echo "$SCENARIO_PAD" | sed 's/^0*//')      # 42   (plain integer)

echo ""
echo "Checking if all 16 .out files ready for scenario ${SCENARIO_PAD}..."

# Check if ALL 16 .out files for this scenario now exist
ALL_DONE=true
MISSING_COUNT=0
for tx in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
    EXPECTED="${INPUT_DIR}/scenario_${SCENARIO_PAD}_tx${tx}.out"
    if [ ! -f "$EXPECTED" ]; then
        ALL_DONE=false
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

if [ "$ALL_DONE" = true ]; then
    echo "All 16 .out files present — extracting S-parameters for scenario ${SCENARIO_PAD}..."
    mkdir -p sparams

    # Lockfile prevents two near-simultaneous jobs both running extraction
    LOCKFILE="/tmp/extract_scenario_${SCENARIO_PAD}.lock"
    (
        flock -n 200 || { echo "Another job already extracting scenario ${SCENARIO_PAD}, skipping."; exit 0; }
        $PYTHON extract_sparameters.py --scenario $SCENARIO_NUM --no-delete
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "✗ ERROR: extract_sparameters.py failed with exit code $EXIT_CODE"
        else
            echo "✓ S-parameter extraction complete: sparams/scenario_${SCENARIO_PAD}.s16p"
        fi
    ) 200>"$LOCKFILE"
else
    echo "Scenario ${SCENARIO_PAD}: ${MISSING_COUNT}/16 .out files still missing — skipping extraction for now."
fi

echo ""
echo "End time: $(date)"
echo "========================================"
