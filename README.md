# Brain EMI Simulation for Stroke Detection

Electromagnetic imaging simulation for brain hemorrhage detection using gprMax FDTD solver. Generates S-parameter data for deep learning model training.

## Current Status

✅ **TRUE ellipsoidal head geometry** (NOT spherical approximation)
✅ **CSF ventricles included** (left + right lateral ventricles, εr=80)
✅ **CPU-based method** (#transmission_line for accurate S-parameters)
✅ **16-antenna monopole array** (37.5mm λ/4 monopoles, circular arrangement)
✅ **300-scenario dataset generated** (50 healthy + 250 hemorrhage)

---

## Dataset Overview

**Full ML training dataset**: 300 scenarios
- **50 healthy baselines** - no hemorrhage
- **250 hemorrhage cases** - 5 sizes × 50 positions
  - Sizes: 5mm, 10mm, 15mm, 20mm, 25mm radius
  - Positions: Systematic 3D grid across brain volume

**Total files**: 4800 input files (300 scenarios × 16 transmit antennas)
- Scenarios 001-050: Healthy baselines
- Scenarios 051-300: Hemorrhage variations

**Dataset splits** (see `dataset_metadata.csv`):
- Train: 210 scenarios (70%)
- Val: 44 scenarios (15%)
- Test: 46 scenarios (15%)

**Expected HPC runtime**: ~300 hours total (16 parallel jobs → ~19 hours wall time)

---

## Files

- **`generate_dataset.py`** - Full dataset generator (300 scenarios)
- **`generate_inputs.py`** - Single-scenario generator (for testing)
- **`generate_metadata.py`** - Creates ML metadata CSV
- **`dataset_metadata.csv`** - Scenario info, lesion params, train/val/test splits
- **`run_simulation.sh`** - SLURM job array script (configured for 4800 jobs)
- **`brain_inputs/`** - Generated `.in` files (4800 files)
- **`gprMax/`** - gprMax FDTD solver source

---

## Quick Start

### 1. Setup

```bash
conda env create -f gprMax/conda_env.yml
conda activate gprmax
cd gprMax
pip install -e .
```

### 2. Generate Inputs

```bash
python generate_inputs.py
# Creates brain_inputs/brain_tx01.in through brain_tx16.in
```

### 3. Run on HPC

```bash
# Upload
scp -r . s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/

# SSH and submit
ssh s4910027@login0.hpc.griffith.edu.au
cd ~/brain-emi-simulation
mkdir -p logs
sbatch run_simulation.sh

# Monitor (~4 hours runtime)
watch -n 10 'squeue -u $USER'
```

### 4. Extract S-Parameters

After simulations complete:
1. Download `.out` files from `brain_inputs/`
2. Create extraction script to read `tls/tl*` groups from HDF5
3. Compute S-matrix and save as `.s16p` Touchstone format

---

## Current Simulation Parameters

| Parameter | Value |
|-----------|-------|
| **Domain** | 600×600×600mm |
| **Resolution** | 2mm (λ/10 at 2 GHz) |
| **Frequency** | 0-2 GHz |
| **Waveform** | Gaussian pulse (1 GHz center) |
| **Time window** | 15ns |
| **Antennas** | 16 monopoles (37.5mm, λ/4) |
| **Array** | Circular (80mm radius) |

### Head Model (TRUE Ellipsoidal)

**Geometry**: Voxelized ellipsoidal layers (4mm resolution)
- **Semi-axes**: a=9.5cm (front-back), b=7.5cm (side-side), c=11.5cm (top-bottom)
- **Layers** (outside to inside):
  - Coupling medium: εr=32, σ=0.6 S/m, 5mm thick
  - Scalp/skull: εr=12, σ=0.2 S/m, 10mm thick
  - Gray matter: εr=52, σ=0.9 S/m, 3mm thick
  - White matter: εr=38, σ=0.5 S/m (core)
- **CSF ventricles**: εr=80, σ=2.0 S/m
  - Left lateral ventricle: 2×1×4cm ellipsoid
  - Right lateral ventricle: 2×1×4cm ellipsoid
  - Separation: 1.5cm center-to-center
- **Hemorrhage**: εr=61, σ=1.54 S/m, variable size/position

---

## Next Steps for Thesis

### Immediate: Expand Dataset Generation

Current generator creates single scenario. For DL training, need to expand to generate variations:

**Recommended approach (300 simulations)**:

1. **Healthy baselines** (50 simulations)
   - No hemorrhage
   - Small variations: ±2mm head position, ±2° rotation

2. **Hemorrhage dataset** (250 simulations)
   - 5 sizes: 5mm, 10mm, 15mm, 20mm, 25mm radius
   - 50 positions: Systematic grid covering brain volume
     - Frontal lobe (12 positions)
     - Temporal lobe (12 positions)
     - Parietal lobe (12 positions)
     - Occipital lobe (10 positions)
     - Deep brain (4 positions)
   - 5 sizes × 50 positions = 250 scenarios

**Implementation**:
- Modify `generate_inputs.py` to loop over lesion sizes and positions
- Use systematic position grid instead of hand-picked locations
- Add random jitter (±2mm) to positions for robustness

**Timeline**: 
- Setup: 1-2 days to modify generator
- HPC runtime: ~250 hours (10-12 days with 16 parallel jobs)
- S-parameter extraction: 2-3 days
- Total: ~2 weeks

### Alternative: Start with 100 simulations pilot

If time-constrained:
- 20 healthy
- 80 hemorrhage (4 sizes × 20 positions)
- Use this to validate ML pipeline
- Expand later if needed

---

## References

- gprMax: http://www.gprmax.com
- Frequency: 0-2 GHz (optimal for brain penetration/resolution)
- Gabriel et al. (1996): Brain tissue dielectric properties

---

## Setup

### Install Dependencies

```bash
conda env create -f gprMax/conda_env.yml
conda activate gprmax
cd gprMax
pip install -e .
```

### Generate Input Files

```bash
python generate_inputs.py
```

Creates 16 files in `brain_inputs/` directory.

---

## Running Simulations

### Local Testing

```bash
python -m gprMax brain_inputs/brain_tx01.in -n 8
```

### HPC (Rangpur)

```bash
# Upload to HPC
scp -r . s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/

# SSH and submit
ssh s4910027@login0.hpc.griffith.edu.au
cd ~/brain-emi-simulation
mkdir -p logs
sbatch run_simulation.sh

# Monitor
squeue -u $USER
watch -n 10 'squeue -u $USER'

# Check logs
tail -f logs/monopole_1.out
```

**Runtime**: ~45-60 min per job, ~4 hours total (16 jobs in parallel)

---

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| **Domain** | 600×600×600mm |
| **Resolution** | 2mm (λ/10 at 2 GHz) |
| **Frequency** | 0-2 GHz |
| **Waveform** | Gaussian pulse (1 GHz center) |
| **Time window** | 15ns |
| **Antennas** | 16 monopoles (37.5mm, λ/4 at 2 GHz) |
| **Array** | Circular (80mm radius) |

### Brain Model

- **Scalp/skull**: εr=12, σ=0.2 S/m
- **Gray matter**: εr=52, σ=0.9 S/m  
- **White matter**: εr=38, σ=0.5 S/m
- **Coupling medium**: εr=78, σ=0.05 S/m (water-based)
- **Hemorrhage**: 15mm radius sphere, εr=58, σ=1.3 S/m (blood clot)

---

## Output

Each simulation produces a `.out` HDF5 file in `brain_inputs/`:
- `tls/` group: Transmission line voltage/current data for S-parameter extraction
- 256 S-parameters per frequency point (16×16 antenna matrix)

---

## Next Steps

1. **Extract S-parameters**: Create script to read `tls/tl*/Vinc`, `Vref`, `Iinc`, `Iref` from HDF5 and compute S-matrix → `.s16p` Touchstone format
2. **Generate dataset**: Modify `generate_inputs.py` to create 100s-1000s of simulations:
   - Healthy baselines (no hemorrhage)
   - Different positions (frontal, temporal, parietal, occipital)
   - Different sizes (5-20mm radius)
3. **Train ML model**: Use S-parameter frequency responses for detection/localization

---

## Improving Realism

### Current Limitations

The current simulations have several simplifications that limit their applicability to real-world measurements:

**Major Issues:**
1. **Spherical Approximation**: Using concentric spheres instead of true ellipsoid geometry
   - Real heads are asymmetric
   - Affects wave propagation and localization accuracy
   
2. **Missing CSF Ventricles**: No cerebrospinal fluid cavities
   - CSF (εr≈80) is a major electromagnetic feature
   - Creates strong reflections that don't appear in current model
   
3. **No Anatomical Variation**: All simulations use identical head dimensions
   - Real adults: ±15-20% variation in head size
   - Age, sex, body mass affect tissue properties
   
4. **Homogeneous Tissue Layers**: Uniform gray/white matter
   - Real brains have heterogeneous distributions
   - Blood vessels, skull sutures missing
   
5. **Simplified Hemorrhage**: Perfect spheres at fixed location
   - Real hemorrhages are irregular shapes
   - No surrounding edema (swelling)

**Expected Gap**: Simulation vs. real measurements will show **20-40% error** in S-parameter magnitudes without these improvements.

### Minimum Requirements for ML Deployment

To train a model that works on real patients:

✅ **Critical (must have):**
- Add CSF ventricles (lateral ventricles, ~40mm wide each)
- Use true ellipsoidal or voxelized geometry
- Generate ≥100 healthy baselines with varied anatomy
- Generate ≥200 stroke cases with varied size (5-30mm) and position
- Add ±15% variation in head dimensions

⚠️ **Important (should have):**
- Tissue property variations (±10%)
- Antenna positioning errors (±3mm)
- Skull thickness variations (5-10mm by location)

### Recommended Improvements

**Priority 1 - Add CSF Ventricles** (biggest impact):
```python
# Add to generate_inputs.py
ventricle_eps_r = 80  # CSF
ventricle_sigma = 2.0
# Add two ellipsoidal cavities for lateral ventricles
```

**Priority 2 - Anatomical Variations**:
```python
# Randomize head dimensions
head_scale = np.random.uniform(0.85, 1.15)
# Randomize tissue properties
eps_variation = np.random.uniform(0.9, 1.1)
```

**Priority 3 - Dataset Generation**:
- 100+ healthy brains (varied anatomy, no lesion)
- 200+ stroke cases (varied location, size, type)
- Different hemorrhage types (acute, chronic, different blood states)

Without these improvements, a model trained on current simulations will likely **fail on real clinical data**.

---

## References

- gprMax: http://www.gprmax.com
- Frequency band: 0-2 GHz (optimal brain tissue penetration/resolution)
