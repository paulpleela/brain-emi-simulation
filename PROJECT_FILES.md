# Project File Structure

## Current Working Files (Realistic Model - 0-2 GHz)

### 📁 Input Generation
- **`generate_realistic_brain_inputs.py`** - Main script to generate 16 simulation input files
  - Creates files in `brain_monopole_realistic/` directory
  - Generates: `brain_realistic_tx01.in` through `brain_realistic_tx16.in`
  - Features: coupling medium, realistic head dimensions, 0-2 GHz optimization

### 📁 Simulation Files
- **`brain_monopole_realistic/`** - Directory containing all simulation input files
  - 16 input files (`.in` format)
  - After HPC run: will contain 16 output files (`.out` format in HDF5)

### 📁 S-Parameter Extraction
- **`extract_port_sparameters.py`** - Extracts S-parameters from transmission line port data
  - Reads V(t) and I(t) from `.out` files
  - Computes proper 50Ω S-parameters
  - Outputs: `brain_realistic_calibrated.s16p`

### 📁 HPC Workflow
- **`run_monopole_brain_array.sh`** - SLURM job array script
  - Submits 16 parallel jobs to HPC
  - Each job runs one simulation
  - Estimated runtime: 1.5-3 hours total (parallel)

### 📁 Visualization
- **`visualize_head.py`** - Creates 2D cross-section diagram
  - Shows coupling medium layer
  - Displays antenna positions with ground planes
  - Saves to: `figures/realistic_head_diagram.png`

- **`visualize_s16p.py`** - Visualizes S-parameter results
  - Heatmaps of S-matrix
  - Frequency response plots
  - Polar pattern diagrams

- **`plot_sij.py`** - Plots specific S-parameter curves
  - Flexible port selection
  - Magnitude and phase plots

- **`generate_all_sij.py`** - Batch generates all S1j plots
  - Creates s1j_all_sources.png through s16j_all_sources.png

### 📁 Analysis & Utilities
- **`analyze_simulation_parameters.py`** - Analyzes mesh resolution, PML, etc.
- **`lesion_receiver_distances.py`** - Computes distances from lesion to antennas
- **`verify_lesion_position.py`** - Detailed position analysis

### 📁 Calibration (Optional)
- **`monopole_calibration.in`** - Single antenna in free space for calibration
- **`antenna_design_monopole.py`** - Shows monopole design parameters

### 📁 Documentation
- **`REALISTIC_MODEL_README.md`** - Main documentation
  - Model description
  - Simulation parameters
  - HPC workflow
  - Troubleshooting guide
  
- **`PROJECT_FILES.md`** - This file (file structure overview)

### 📁 Configuration
- **`.gitignore`** - Excludes outputs and temporary files from version control

## Output Directories

### `brain_monopole_realistic/`
- Input files: `brain_realistic_tx01.in` through `tx16.in` (✓ created)
- Output files: `brain_realistic_tx01.out` through `tx16.out` (after HPC run)

### `figures/`
- `realistic_head_diagram.png` - Model visualization (✓ created)
- Future: S-parameter heatmaps, frequency plots, etc.

### `sij_plots/` (created after S-parameter extraction)
- S-parameter visualization images
- One plot per receiver port

## Obsolete Files (Removed)

The following files/directories were removed as they're no longer needed:

- ❌ `brain_monopole_simulations/` - Old monopole model (0.5m domain, no coupling medium)
- ❌ `s16p_simulations/` - Old Hertzian dipole model (not realistic)
- ❌ `ANTENNA_UPGRADE_GUIDE.md` - Merged into REALISTIC_MODEL_README.md
- ❌ `MONOPOLE_HPC_README.md` - Merged into REALISTIC_MODEL_README.md
- ❌ `generate_monopole_brain_inputs.py` - Replaced by generate_realistic_brain_inputs.py
- ❌ `brain_hemorrhage.s16p` - Old S-parameters from Hertzian dipole model
- ❌ Old PNG files (s1j_all_sources.png, etc.) - Will be regenerated with new data

## Workflow Summary

### 1. Generate Input Files (✓ Complete)
```bash
python generate_realistic_brain_inputs.py
```

### 2. Visualize Model (✓ Complete)
```bash
python visualize_head.py
```

### 3. Run Simulations on HPC (Next Step)
```bash
# Upload to HPC:
# - brain_monopole_realistic/
# - run_monopole_brain_array.sh
# - extract_port_sparameters.py

# On HPC:
mkdir -p logs
sbatch run_monopole_brain_array.sh

# Monitor:
squeue -u $USER
```

### 4. Extract S-Parameters (After HPC)
```bash
# Download all .out files from HPC

# Run extraction:
python extract_port_sparameters.py
```

### 5. Visualize Results
```bash
python visualize_s16p.py brain_realistic_calibrated.s16p
python generate_all_sij.py
```

## Model Parameters Summary

| Parameter | Value |
|-----------|-------|
| **Frequency Range** | 0-2 GHz |
| **Domain Size** | 600×600×600 mm |
| **Grid Resolution** | 2 mm (isotropic) |
| **Total Cells** | 27 million |
| **Time Window** | 15 ns |
| **Monopole Length** | 37.5 mm (λ/4 @ 2 GHz) |
| **Ground Plane** | 75×75 mm PEC |
| **Coupling Medium** | 5 mm, εᵣ=32, σ=0.58 S/m |
| **Head Model** | Multi-layer spheres (avg radius ~95mm) |
| **Number of Antennas** | 16 (equatorial ring) |
| **Port Impedance** | 50Ω (transmission line) |

## Key Improvements from Original Model

1. ✅ **Frequency optimization**: 0-2 GHz (medical imaging range)
2. ✅ **Coupling medium**: 5mm tissue-equivalent layer
3. ✅ **Realistic antennas**: Quarter-wave monopoles with ground planes
4. ✅ **Proper ports**: 50Ω transmission line ports (not Hertzian dipoles)
5. ✅ **Fixed positions**: Antennas touching head surface
6. ✅ **Realistic dimensions**: Based on human head ellipsoid geometry

## Next Actions

- [ ] Test single simulation locally (optional)
- [ ] Upload to HPC
- [ ] Run array job on HPC
- [ ] Download results
- [ ] Extract S-parameters
- [ ] Analyze hemorrhage detection capability

---
*Last updated: 2025-12-01*
