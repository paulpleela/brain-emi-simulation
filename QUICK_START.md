# Quick Start Guide - Realistic Brain Imaging

## Current Status ✓

All files have been updated for the realistic model:
- ✅ Frequency range: 0-2 GHz (optimized for medical imaging)
- ✅ Coupling medium: 5mm tissue-equivalent layer
- ✅ Fixed antenna positions touching head surface
- ✅ Realistic monopole antennas (37.5mm @ 2 GHz)
- ✅ 16 input files generated in `brain_monopole_realistic/`
- ✅ All scripts updated to use realistic model
- ✅ Obsolete files removed
- ✅ Visualization tested and working

## Files Ready to Use

### For HPC Execution
```
brain_monopole_realistic/          (16 input files)
run_monopole_brain_array.sh        (SLURM job script)
extract_port_sparameters.py        (S-parameter extraction)
```

### For Visualization & Analysis
```
visualize_head.py                  (Model diagram - TESTED ✓)
visualize_s16p.py                  (S-parameter plots)
plot_sij.py                        (Individual S-parameters)
generate_all_sij.py                (Batch S-parameter plots)
```

### Documentation
```
REALISTIC_MODEL_README.md          (Comprehensive guide)
PROJECT_FILES.md                   (File structure)
QUICK_START.md                     (This file)
```

## Next Steps

### Option 1: Test Locally (Optional)
```powershell
# Run one simulation to verify everything works
gprmax brain_monopole_realistic/brain_realistic_tx01.in -n 8

# This will take 45-90 minutes
# Output: brain_monopole_realistic/brain_realistic_tx01.out
```

### Option 2: Go Straight to HPC (Recommended)

#### Step 1: Prepare HPC
```bash
# On HPC, create working directory
mkdir -p ~/brain_imaging
cd ~/brain_imaging
mkdir -p logs
```

#### Step 2: Upload Files
Upload these to HPC:
- `brain_monopole_realistic/` (entire directory)
- `run_monopole_brain_array.sh`
- `extract_port_sparameters.py`

#### Step 3: Customize HPC Script
Edit `run_monopole_brain_array.sh` on HPC:
```bash
# Uncomment and modify these lines for your system:
# module load anaconda3
# conda activate gprmax-env

# OR
# module load python/3.11
# module load hdf5/1.12
```

#### Step 4: Submit Job
```bash
sbatch run_monopole_brain_array.sh
```

#### Step 5: Monitor
```bash
# Check job status
squeue -u $USER

# Check logs (while running)
tail -f logs/brain_realistic_tx01.out

# Check all jobs
ls -lh logs/
```

#### Step 6: Verify Completion
```bash
# Check output files exist
ls -lh brain_monopole_realistic/*.out

# Should see 16 .out files, each ~50-100 MB
# Total size: ~1-2 GB
```

#### Step 7: Extract S-Parameters
```bash
# On HPC or after downloading files locally
python extract_port_sparameters.py

# Output: brain_realistic_calibrated.s16p
```

#### Step 8: Download Results
Download from HPC:
- `brain_realistic_calibrated.s16p`
- All `.out` files (if you want to do further analysis)

#### Step 9: Visualize (Locally)
```powershell
# On your local machine
python visualize_s16p.py brain_realistic_calibrated.s16p
python generate_all_sij.py
```

## Expected Runtime

| Task | Time | Notes |
|------|------|-------|
| Single simulation | 45-90 min | 8 CPU cores |
| 16 simulations (serial) | 12-24 hours | Not recommended |
| 16 simulations (HPC array) | 1.5-3 hours | **Recommended** |
| S-parameter extraction | 2-5 min | Fast |
| Visualization | 1-2 min | Fast |

## Troubleshooting

### Simulation crashes
- Check memory: increase `--mem=` in SLURM script
- Reduce domain or increase dx_dy_dz

### No output files
- Check logs in `logs/` directory
- Verify gprMax is installed on HPC
- Check module loads in script

### S-parameter extraction fails
- Ensure all 16 .out files exist
- Check file paths in script
- Verify HDF5 library installed

### Poor S-parameter quality
- Increase time window in input files
- Check antenna positions (must touch PEC ground plane)
- Verify material properties

## Key Model Parameters

```
Domain:          600×600×600 mm (27M cells)
Grid:            2 mm isotropic
Time window:     15 ns
Frequency:       0-2 GHz
Antennas:        16 monopoles (37.5 mm)
Ground planes:   75×75 mm PEC
Coupling:        5 mm, εr=32, σ=0.58 S/m
Head radius:     ~95-110 mm (multi-layer)
Lesion:          2 blood spheres at offset position
```

## Getting Help

1. Check `REALISTIC_MODEL_README.md` for detailed explanations
2. Check `PROJECT_FILES.md` for file structure
3. Review HPC logs in `logs/` directory
4. Verify gprMax installation: `gprmax --version`

## Output Files You'll Get

After complete workflow:
```
brain_realistic_calibrated.s16p    (16×16 S-matrix vs freq)
figures/realistic_head_diagram.png (Model visualization)
sij_plots/s1j_all_sources.png      (S-parameter plots)
... (more visualization files)
```

## What Changed from Original Model

- ❌ Removed: Hertzian dipoles (unrealistic)
- ❌ Removed: 0-130 GHz range (too broad)
- ❌ Removed: No coupling medium
- ❌ Removed: Floating antennas

- ✅ Added: Realistic monopoles with ground planes
- ✅ Added: 0-2 GHz optimization
- ✅ Added: 5mm coupling medium
- ✅ Added: Antennas touching head
- ✅ Added: Proper 50Ω transmission line ports

---

**You're ready to run simulations on HPC!**

Upload the files and submit the job. Results should be ready in ~2 hours.

Good luck with your thesis! 🎓
