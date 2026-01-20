# Dataset Expansion Complete ✅

## What Was Done

Successfully expanded from **1 scenario** to **300 scenarios** for deep learning training!

### Generated Files

1. **`generate_dataset.py`** (NEW)
   - Full dataset generator
   - 50 healthy baselines (no lesion)
   - 250 hemorrhage cases (5 sizes × 50 positions)
   - Systematic 3D position grid across brain volume
   
2. **`generate_metadata.py`** (NEW)
   - Creates CSV with scenario information
   - Includes lesion parameters (size, position)
   - Assigns train/val/test splits (70/15/15%)
   
3. **`dataset_metadata.csv`** (NEW)
   - 300 rows (one per scenario)
   - Columns: scenario_id, has_lesion, lesion_size_mm, lesion_x/y/z, split
   - Use this for ML training to match scenarios with S-parameters

4. **`brain_inputs/`** (UPDATED)
   - 4800 input files total
   - Format: `scenario_XXX_txYY.in`
   - Each scenario has 16 files (one per transmit antenna)

5. **`run_simulation.sh`** (UPDATED)
   - Changed `--array=1-160%16` → `--array=1-4800%16`
   - Now processes all 300 scenarios

6. **`README.md`** (UPDATED)
   - Added dataset overview section
   - Lists all files and their purposes

---

## Dataset Structure

### Scenario Breakdown

| Scenario Range | Type | Description |
|----------------|------|-------------|
| 001-050 | Healthy | No hemorrhage (baselines) |
| 051-100 | Hemorrhage | 5mm radius, 50 positions |
| 101-150 | Hemorrhage | 10mm radius, 50 positions |
| 151-200 | Hemorrhage | 15mm radius, 50 positions |
| 201-250 | Hemorrhage | 20mm radius, 50 positions |
| 251-300 | Hemorrhage | 25mm radius, 50 positions |

### Hemorrhage Position Grid

Positions systematically cover the brain volume:
- Radial extent: ±5cm from center (avoids skull)
- Vertical extent: ±8cm from center
- Avoids CSF ventricles (x > 1.5cm from center)
- 50 positions per lesion size

### Dataset Splits

From `dataset_metadata.csv`:
- **Train**: 210 scenarios (70%)
  - 35 healthy + 175 hemorrhage
- **Val**: 44 scenarios (15%)
  - 7 healthy + 37 hemorrhage
- **Test**: 46 scenarios (15%)
  - 8 healthy + 38 hemorrhage

---

## Next Steps

### 1. Test Single Scenario First (RECOMMENDED)

Before running all 300 scenarios, test one to verify ellipsoidal geometry works on HPC:

```bash
# Create a test directory
mkdir test_run
cp brain_inputs/scenario_001_tx01.in test_run/

# Upload to HPC
scp -r test_run/ run_simulation.sh s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/

# SSH to HPC
ssh s4910027@login0.hpc.griffith.edu.au

# Run single job
cd ~/brain-emi-simulation
mkdir -p logs
python -m gprMax test_run/scenario_001_tx01.in -n 8

# Check output
ls -lh test_run/*.out
```

**Expected output**: `scenario_001_tx01.out` (~50-100 MB)

### 2. Run Small Batch (10 scenarios)

If single test works, try 10 scenarios (160 jobs):

```bash
# Modify run_simulation.sh line 5:
#SBATCH --array=1-160%16

# Upload all files
scp -r brain_inputs/ run_simulation.sh s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/

# Submit job
cd ~/brain-emi-simulation
sbatch run_simulation.sh

# Monitor
watch -n 10 'squeue -u $USER'
```

**Expected runtime**: ~1 hour (16 parallel jobs × 10 iterations)

### 3. Run Full Dataset (300 scenarios)

Once small batch succeeds, run everything:

```bash
# run_simulation.sh already configured for:
#SBATCH --array=1-4800%16

# Just submit (files already uploaded)
sbatch run_simulation.sh

# Monitor progress
watch -n 60 'squeue -u $USER'
sacct --format=JobID,JobName,State,Elapsed
```

**Expected runtime**: ~300 hours total (16 parallel → ~19 hours wall time)

### 4. Extract S-Parameters

After simulations complete, create extraction script (TODO):

```python
# extract_sparams.py (needs to be created)
# - Read .out files (HDF5 format)
# - Extract transmission line data
# - Compute 16×16 S-matrix for each scenario
# - Save as .s16p or .npz files
```

---

## File Sizes

- **Input files**: 4800 files × ~10 KB = ~48 MB
- **Output files** (estimated): 4800 files × ~80 MB = ~380 GB
- **S-parameter files** (after extraction): 300 files × ~5 KB = ~1.5 MB

**Tip**: Process outputs in batches to avoid storage issues. Delete .out files after extracting S-parameters.

---

## Verification Checklist

✅ Dataset generated (4800 files)
✅ Metadata CSV created (300 scenarios)
✅ SLURM script updated (--array=1-4800%16)
✅ README updated with dataset info
✅ Files use TRUE ellipsoidal geometry
✅ CSF ventricles included in all scenarios
✅ Train/val/test splits assigned

⏳ Test single scenario on HPC
⏳ Run small batch (10 scenarios)
⏳ Run full dataset (300 scenarios)
⏳ Create S-parameter extraction script
⏳ Extract S-matrices from .out files
⏳ Train ML model

---

## Important Notes

### Geometry Confirmation

All 4800 input files use **TRUE ellipsoidal geometry**:
- Semi-axes: a=9.5cm, b=7.5cm, c=11.5cm
- Voxelized with 4mm resolution
- NOT spherical approximation ✅

Verify by checking any file:
```bash
head -n 20 brain_inputs/scenario_001_tx01.in
```

Should see `## Ellipsoidal head geometry` and `#python:` blocks with `in_ellipsoid()` function.

### CSF Ventricles

All scenarios include left + right lateral ventricles:
- Size: 2×1×4cm each
- Material: εr=80, σ=2.0 S/m
- Position: ±7.5mm from head center

### Hemorrhage Positions

Positions avoid:
- Skull (>5cm from center)
- CSF ventricles (|x| > 1.5cm)

Healthy scenarios (001-050) have no hemorrhage.

### HPC Resource Usage

Per job:
- Time: 45-60 min
- CPUs: 8 cores
- Memory: ~4-8 GB

Full dataset:
- Total CPU-hours: 2400-3200 hours
- With 16 parallel: ~19 hours wall time
- Storage: ~380 GB for .out files

---

## Questions?

Check:
1. **README.md** - Overview and head model details
2. **dataset_metadata.csv** - Scenario parameters
3. **generate_dataset.py** - How scenarios were created
4. **WORKFLOW.md** - Step-by-step HPC guide (now outdated for 50 scenarios)

---

**Status**: Ready for HPC testing! 🚀

Start with single-scenario test, then scale up to full 300 scenarios.
