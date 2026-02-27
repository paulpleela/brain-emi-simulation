# Brain EMI Simulation for Stroke Detection

Electromagnetic imaging simulation for brain hemorrhage detection using gprMax FDTD solver. Generates S-parameter data for deep learning model training.

---

## Project Status

✅ **TRUE ellipsoidal head geometry** (voxelized, NOT spherical)  
✅ **CSF ventricles included** (left + right lateral ventricles, εr=80)  
✅ **300-scenario dataset** (50 healthy + 250 hemorrhage)  
✅ **16-antenna monopole array** (circular arrangement, 37.5mm λ/4 monopoles)  
✅ **HPC-ready** (SLURM script configured for Rangpur)

---

## Dataset Overview

**Total**: 300 scenarios for ML training
- **50 healthy baselines** - No hemorrhage
- **250 hemorrhage cases** - 5 sizes × 50 positions
  - Sizes: 5mm, 10mm, 15mm, 20mm, 25mm radius
  - Positions: Systematic 3D grid across brain volume

**Files**: 4800 input files (300 scenarios × 16 transmit antennas each)

**Dataset splits** (in `dataset_metadata.csv`):
- Train: 210 scenarios (70%)
- Validation: 44 scenarios (15%)
- Test: 46 scenarios (15%)

---

## Files

| File | Purpose |
|------|---------|
| `generate_dataset.py` | Generate all 300 scenarios |
| `generate_inputs.py` | Generate single scenario (for testing) |
| `generate_metadata.py` | Create ML metadata CSV |
| `dataset_metadata.csv` | Scenario parameters and train/val/test splits |
| `run_simulation.sh` | SLURM job script for HPC |
| `brain_inputs/` | Directory with 4800 input files |
| `HPC_GUIDE.md` | Complete HPC deployment instructions |
| `gprMax/` | gprMax FDTD solver source code |

---

## Quick Start

### Generate Dataset

```bash
# Generate all 300 scenarios (4800 files)
python generate_dataset.py

# Generate metadata CSV
python generate_metadata.py
```

### Run on HPC

See **`HPC_GUIDE.md`** for complete instructions.

**Summary**:
1. Push code: `git push origin main`
2. SSH to Rangpur: `ssh s4910027@login0.hpc.griffith.edu.au`
3. Pull code: `cd ~/brain-emi-simulation && git pull`
4. Setup (first time): `conda env create -f gprMax/conda_env.yml`
5. Test: `python -m gprMax brain_inputs/scenario_001_tx01.in -n 8`
6. Run all: `sbatch run_simulation.sh`

**Runtime**: ~19 hours (300 scenarios, 16 parallel jobs)

---

## Simulation Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Domain** | 600×600×600mm | Cubic simulation space |
| **Resolution** | 2mm | λ/10 at 2 GHz |
| **Frequency** | 0-2 GHz | Optimal for brain imaging |
| **Time window** | 15ns | Full wave propagation |
| **Waveform** | Gaussian 1 GHz | Center frequency |
| **Antennas** | 16 monopoles | 37.5mm (λ/4) length |
| **Array** | Circular | Single ring at head center |

---

## Head Model (TRUE Ellipsoidal Geometry)

**Geometry**: Voxelized ellipsoidal layers (4mm resolution)

**Semi-axes**: 
- a = 9.5 cm (front-back, anterior-posterior)
- b = 7.5 cm (side-side, left-right)
- c = 11.5 cm (top-bottom, superior-inferior)

**Layers** (outside → inside):
1. **Coupling medium**: εr=32, σ=0.585 S/m, 5mm thick
2. **Scalp/skull**: εr=12, σ=0.2 S/m, 10mm thick
3. **Gray matter**: εr=52, σ=0.97 S/m, 3mm thick
4. **White matter**: εr=38, σ=0.57 S/m (core)

**CSF ventricles** (critical feature):
- Material: εr=80, σ=2.0 S/m
- Left lateral ventricle: 2×1×4cm ellipsoid
- Right lateral ventricle: 2×1×4cm ellipsoid
- Separation: 1.5cm center-to-center

**Hemorrhage** (variable):
- Material: εr=61, σ=1.54 S/m (blood)
- Sizes: 5-25mm radius
- Positions: 50 locations across brain volume

---

## Antenna Configuration

**Type**: Quarter-wave monopoles (37.5mm for λ/4 at 2 GHz)

**Array**: 16 antennas in circular arrangement
- Single ring at head center (z = 0.25m)
- Even angular spacing (22.5° between antennas)
- Positioned just outside coupling medium

**Excitation**: One transmitter at a time
- Each scenario: 16 simulations (one per transmit antenna)
- All antennas act as receivers (transmission lines)
- Enables full 16×16 S-parameter matrix extraction

**Impedance**: 50Ω transmission lines

---

## Output Files

Each simulation produces `.out` file (HDF5 format):
- **Location**: `brain_inputs/scenario_XXX_txYY.out`
- **Size**: ~50-100 MB per file
- **Contents**: 
  - `tls/` group: Transmission line voltage/current data
  - Incident/reflected voltage and current
  - Time-domain waveforms

**Total dataset**: 4800 output files (~380 GB)

---

## Next Steps

### 1. Run HPC Simulations
- Follow `HPC_GUIDE.md`
- Expected: ~19 hours for full dataset

### 2. Extract S-Parameters
Create script to:
- Read HDF5 `.out` files
- Extract transmission line data from `tls/tl*/` groups
- Compute 16×16 S-matrix per scenario
- Save as `.s16p` Touchstone or `.npz` format

### 3. Train ML Model
- Use `dataset_metadata.csv` for train/val/test splits
- Input: S-parameter frequency responses (16×16 matrix)
- Outputs:
  - Classification: Hemorrhage present? (binary)
  - Localization: Position (x, y, z)
  - Sizing: Radius (mm)

---

## Dataset Generation

### Hemorrhage Position Strategy

Positions systematically cover brain volume:
- **Radial extent**: ±5cm from center (stay in tissue)
- **Vertical extent**: ±8cm from center
- **Avoids**: CSF ventricles (|x| > 1.5cm from center)
- **Distribution**: 3D grid with 50 sampling points

### Scenario Naming

Format: `scenario_XXX_txYY.in`
- `XXX`: Scenario ID (001-300)
- `YY`: Transmit antenna (01-16)

**Scenario ranges**:
- 001-050: Healthy (no hemorrhage)
- 051-100: 5mm hemorrhage, 50 positions
- 101-150: 10mm hemorrhage, 50 positions
- 151-200: 15mm hemorrhage, 50 positions
- 201-250: 20mm hemorrhage, 50 positions
- 251-300: 25mm hemorrhage, 50 positions

---

## Key Features

### Why This Is Realistic

✅ **Ellipsoidal geometry** (not spherical)
- Average adult head dimensions
- Proper aspect ratios
- Voxelized implementation

✅ **CSF ventricles** (critical EM feature)
- Highest dielectric constant in brain (εr=80)
- Creates strong reflections
- Affects wave propagation patterns

✅ **Systematic variations**
- 50 hemorrhage positions
- 5 hemorrhage sizes
- 50 healthy baselines

✅ **Clinical relevance**
- Hemorrhage sizes: 5-25mm (clinical range)
- Frequencies: 0-2 GHz (brain penetration)
- Realistic tissue properties (Gabriel 1996)

---

## References

- **gprMax**: http://www.gprmax.com
- **Tissue properties**: Gabriel et al. (1996) - Dielectric properties of biological tissues
- **Frequency selection**: 0-2 GHz optimal for brain imaging (penetration vs resolution)
- **CSF properties**: Akhtari et al. (2006) - Conductivities of brain tissues

---

## Installation

### Prerequisites
- Python 3.6+
- Conda package manager
- Git

### Setup
```bash
# Clone repository
git clone https://github.com/paulpleela/brain-emi-simulation.git
cd brain-emi-simulation

# Create conda environment
conda env create -f gprMax/conda_env.yml

# Activate environment
conda activate gprmax

# Install gprMax
cd gprMax
python setup.py install
cd ..
```

### Verify Installation
```bash
python -c "import gprMax; print('gprMax installed!')"
```

---

## Contact

For questions about this simulation setup, see GitHub repository:
https://github.com/paulpleela/brain-emi-simulation
