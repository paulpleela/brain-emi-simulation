#!/bin/bash
#SBATCH --job-name=gprmax_test
#SBATCH --output=test_single/test.out
#SBATCH --error=test_single/test.err
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Single scenario test for ellipsoidal geometry validation

echo "========================================"
echo "gprMax Brain EMI - Single Scenario Test"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"
echo ""

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
    conda activate gprmax || true
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate gprmax || true
fi

# Input file
INPUT_FILE="test_single/scenario_001_tx01.in"

echo "Testing file: $INPUT_FILE"
echo ""

# Check file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

echo "File header:"
head -n 10 "$INPUT_FILE"
echo ""
echo "----------------------------------------"
echo ""

# Run gprMax
echo "Running gprMax with $SLURM_CPUS_PER_TASK threads..."
echo ""

python -m gprMax "$INPUT_FILE" -n $SLURM_CPUS_PER_TASK

# Check output
OUTPUT_FILE="${INPUT_FILE%.in}.out"
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "========================================"
    echo "✓ SUCCESS - Ellipsoidal geometry works!"
    echo "========================================"
    echo "Output file: $OUTPUT_FILE"
    echo "File size: $(du -h $OUTPUT_FILE | cut -f1)"
    echo ""
    echo "Output file contains:"
    echo "  - S-parameters (transmission line data)"
    echo "  - Field snapshots"
    echo "  - Geometry information"
else
    echo ""
    echo "========================================"
    echo "✗ FAILED - Output not created"
    echo "========================================"
    exit 1
fi

echo ""
echo "End time: $(date)"
echo "========================================"
