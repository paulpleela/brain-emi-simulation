# Brain EMI Simulation for Stroke Detection# Brain EMI Simulation for Stroke Detection# Brain EMI Simulation for Stroke Detection



Electromagnetic imaging simulation for brain hemorrhage detection using gprMax FDTD solver. Generates S-parameter data for deep learning model training.



---Electromagnetic imaging simulation for brain hemorrhage detection using gprMax FDTD solver. Generates S-parameter data for deep learning model training.Electromagnetic imaging simulation for brain hemorrhage detection using gprMax FDTD solver. Generates S-parameter data for deep learning model training.



## Project Status



✅ TRUE ellipsoidal head geometry (voxelized, not spherical)  ------

✅ CSF ventricles included (left + right lateral ventricles, εr=80)  

✅ 16-antenna wire dipole array (z-directed, 114 mm, resonant at 1.25 GHz)  

✅ S-parameter extractor (`extract_sparameters.py`, Z0=73 Ω, 0.5–2 GHz)  

✅ S-parameter visualiser (`visualise_s16p.py`, return loss + transmission)  ## Project Status## Project Status

✅ HPC-ready (SLURM script configured for Rangpur)  

✅ Confirmed working: 90 frequency points, 0.517–2.000 GHz, R 73  

🔲 Expanded dataset (1,360 scenarios) — see `DATASET_PLAN.md`

✅ **TRUE ellipsoidal head geometry** (voxelized, not spherical)  ✅ **TRUE ellipsoidal head geometry** (voxelized, NOT spherical)  

---

✅ **CSF ventricles included** (left + right lateral ventricles, εr=80)  ✅ **CSF ventricles included** (left + right lateral ventricles, εr=80)  

## Repository Structure

✅ **300-scenario dataset** (50 healthy + 250 hemorrhage)  ✅ **300-scenario dataset** (50 healthy + 250 hemorrhage)  

| File / Folder | Purpose |

|---|---|✅ **16-antenna wire dipole array** (z-directed, 114 mm, resonant at 1.25 GHz)  ✅ **16-antenna monopole array** (circular arrangement, 37.5mm λ/4 monopoles)  

| `generate_dataset.py` | Generate all scenario `.in` files |

| `generate_inputs.py` | Generate 16 smoke-test files (`brain_tx*.in`) |✅ **S-parameter extractor** (`extract_sparameters.py`, Z₀=73 Ω, 0.5–2 GHz)  ✅ **HPC-ready** (SLURM script configured for Rangpur)

| `extract_sparameters.py` | HDF5 `.out` → 16×16 S-matrix → `.s16p` |

| `visualise_s16p.py` | Plot S-parameters from `.s16p` |✅ **S-parameter visualiser** (`visualise_s16p.py`, return loss + transmission)  

| `validate_s16p.py` | Validate Touchstone file structure |

| `generate_metadata.py` | Create ML metadata CSV |✅ **HPC-ready** (SLURM script configured for Rangpur)  ---

| `plot_setup_diagram.py` | Generate `setup_diagram.png` annotated figure |

| `dataset_metadata.csv` | Scenario parameters and train/val/test splits |✅ **Confirmed working**: 90 frequency points, 0.517–2.000 GHz, R 73

| `run_simulation.sh` | SLURM job script for HPC |

| `brain_inputs/` | Generated `.in` input files (not committed) |## Dataset Overview

| `sparams/` | Extracted `.s16p` files and visualisation PNGs |

| `DATASET_PLAN.md` | Full specification for the 1,360-scenario dataset |---

| `HPC_GUIDE.md` | Complete HPC deployment instructions |

| `gprMax/` | gprMax FDTD solver source code |**Total**: 300 scenarios for ML training


## Quick Start



### Planned (1,360 scenarios — see `DATASET_PLAN.md`)  YZ-plane cross-section  (one dipole, side view)



| Group | Scenarios | Notes |  ────────────────────────────────────────────────### Generate Dataset

|---|---|---|

| Healthy | 135 | 27 anatomy configs × 5 tilt angles |

| Hemorrhage | 1,225 | 5 sizes × 35 locations × 7 anatomy configs |

| **Total** | **1,360** | 21,760 `.in` files |   z = 0.308 m ----  top of upper PEC arm```bash



Split: 80% train / 10% val / 10% test → **1,088 training scenarios**                      |  PEC wire# Generate all 300 scenarios (4800 files)



---                      |  (56 mm upper arm, 28 cells)python generate_dataset.py



## Simulation Parameters   z = 0.252 m ----  top of feed gap   (cz + gap)



| Parameter | Value | Notes |                   [==TL==]  <-- #transmission_line: z  73 ohm  waveform# Generate metadata CSV

|---|---|---|

| Domain | 600×600×600 mm | Cubic |                      |      feed gap = 2 mm = 1 cellpython generate_metadata.py

| Grid resolution | 2 mm | λ/10 at ~1.5 GHz |

| Time window | 60 ns | Δf = 16.7 MHz → ~90 frequency points |   z = 0.250 m ----  feed point  cz  (TL placed here)```

| Waveform | Gaussian, centre 1.25 GHz | Covers 0.5–2 GHz band |

| Antennas | 16 z-directed wire dipoles | Physical PEC arms |                      |  free_space edge overwrites PEC in gap zone

| Dipole arm length | 56 mm (28 cells) | Resonance at 1.250 GHz in free space |

| Feed gap | 2 mm (1 cell) | Carved with `free_space` override |                      |  PEC wire### Run on HPC

| Total dipole length | 114 mm | z: 0.194–0.308 m |

| TL impedance | 73 Ω | Half-wave dipole impedance |                      |  (56 mm lower arm, 28 cells)

| Confirmed frequency points | 90 | 0.517–2.000 GHz, scenario_001 |

   z = 0.194 m ----  bottom of lower PEC armSee **`HPC_GUIDE.md`** for complete instructions.

---



## Head Model

   Total dipole: 0.194 -> 0.308 m = 114 mm**Summary**:

**Geometry**: Voxelized ellipsoidal layers at 2 mm resolution

   ┌─────────────────────────────────────────────────────────────┐1. Push code: `git push origin main`

**Semi-axes** (average adult):

- a = 9.5 cm (front-back)   │  #edge: cx cy 0.194  cx cy 0.308  pec          <- full wire  │2. SSH to Rangpur

- b = 7.5 cm (left-right)

- c = 11.5 cm (top-bottom)   │  #edge: cx cy 0.250  cx cy 0.252  free_space   <- carve gap  │3. Pull code: `cd ~/brain-emi-simulation && git pull`



**Tissue layers** (outside → inside):   │  #transmission_line: z cx cy 0.250  73  waveform            │4. Setup (first time): `conda env create -f gprMax/conda_env.yml`



| Layer | εr | σ (S/m) | Thickness |   └─────────────────────────────────────────────────────────────┘5. Test: `python -m gprMax brain_inputs/scenario_001_tx01.in -n 8`

|---|---|---|---|

| Coupling medium (glycerol/water) | 36 | 0.3 | 5 mm |6. Run all: `sbatch run_simulation.sh`

| Scalp + skull | 12 | 0.2 | 10 mm |

| Gray matter (cortex) | 52 | 0.97 | 3 mm |

| White matter (core) | 38 | 0.57 | fills interior |

| CSF ventricles | 80 | 2.0 | 2×1×4 cm ellipsoids |  Antenna ring (equatorial plane, z = 0.25 m)**Runtime**: ~19 hours (300 scenarios, 16 parallel jobs)



**Hemorrhage** (variable): εr=61, σ=1.54 S/m (blood), spherical, 5–25 mm radius  ─────────────────────────────────────────────



------



## Antenna Configuration   head_center = (0.25, 0.25)



**Type**: Physical half-wave wire dipole using gprMax `#edge` + `#transmission_line`   r_ring = (a + scalp_skull) + coupling_thickness + 2*cell## Simulation Parameters



```          = (0.095 + 0.010) + 0.005 + 0.004 ~ 0.114 m

#edge: cx cy (cz-arm)  cx cy (cz+arm+gap)  pec         <- full PEC wire

#edge: cx cy cz         cx cy (cz+gap)      free_space  <- carve feed gap| Parameter | Value | Notes |

#transmission_line: z cx cy cz  73  waveform            <- TL at gap

```   Antenna i:  angle = i * 22.5 deg  (16 antennas, 360/16)|-----------|-------|-------|



- 16 antennas evenly at 22.5° intervals, equatorial ring (z = 0.25 m)               cx    = 0.25 + r_ring * cos(angle)| **Domain** | 600×600×600mm | Cubic simulation space |

- All z-directed — no axis-alignment ambiguity

- TX: 1 active `#transmission_line` (Gaussian waveform)               cy    = 0.25 + r_ring * sin(angle)| **Resolution** | 2mm | λ/10 at 2 GHz |

- RX: 15 passive `#transmission_line` (`rx_null`) — records V/I for S-parameters

- Enables full 16×16 S-parameter matrix per scenario               cz    = 0.25| **Frequency** | 0-2 GHz | Optimal for brain imaging |



**Reference**: gprMax official example `antenna_wire_dipole_fs.in`  | **Time window** | 15ns | Full wave propagation |

(150 mm → 950 MHz; our 114 mm → 1,250 MHz by scaling)

   All 16 dipoles are z-directed (vertical).| **Waveform** | Gaussian 1 GHz | Center frequency |

---

   TX: 1 active transmission line with Gaussian pulse| **Antennas** | 16 monopoles | 37.5mm (λ/4) length |

## Quick Start

   RX: 15 passive transmission lines with rx_null| **Array** | Circular | Single ring at head center |

### Generate and test locally



```bash

# Generate 16 smoke-test input files---

python generate_inputs.py

  Frequency / time domain

# Generate full dataset (300 scenarios, 4800 files)

python generate_dataset.py  ──────────────────────────## Head Model (TRUE Ellipsoidal Geometry)



# Generate metadata CSV

python generate_metadata.py

```   #waveform: gaussian  1  1.25e9  tx_pulse**Geometry**: Voxelized ellipsoidal layers (4mm resolution)



### Run on HPC   #time_window: 60e-9



See `HPC_GUIDE.md` for full instructions. Summary:**Semi-axes**: 



```bash   Delta_f = 1 / 60ns = 16.7 MHz- a = 9.5 cm (front-back, anterior-posterior)

git push origin main

# SSH to Rangpur, then:   Frequency points in 0.5-2 GHz: ~90 points- b = 7.5 cm (side-side, left-right)

git pull

rm brain_inputs/scenario_*.in && python generate_dataset.py   Expected resonance: 0.475 * c0 / 0.114 = 1.250 GHz  (free space)  [confirmed]- c = 11.5 cm (top-bottom, superior-inferior)

sbatch run_simulation.sh

``````



> **Important**: Always regenerate `.in` files on the HPC after any code change.**Layers** (outside → inside):

> The `.in` files are not committed to git.

---1. **Coupling medium**: εr=32, σ=0.585 S/m, 5mm thick

### Extract and visualise S-parameters

2. **Scalp/skull**: εr=12, σ=0.2 S/m, 10mm thick

```bash

python extract_sparameters.py --scenario 1## Dataset Overview3. **Gray matter**: εr=52, σ=0.97 S/m, 3mm thick

python visualise_s16p.py sparams/scenario_001.s16p

```4. **White matter**: εr=38, σ=0.57 S/m (core)



---**Total**: 300 scenarios for ML training



## Changelog- **50 healthy baselines** — no hemorrhage**CSF ventricles** (critical feature):



| Commit | Change |- **250 hemorrhage cases** — 5 sizes × 50 positions- Material: εr=80, σ=2.0 S/m

|---|---|

| initial | Hertzian dipole (point source), no physical resonance |  - Sizes: 5 mm, 10 mm, 15 mm, 20 mm, 25 mm radius- Left lateral ventricle: 2×1×4cm ellipsoid

| `15fe072` | Physical wire dipole: correct PEC arm + free_space gap syntax |

| `60b4b1b` | Extractor Z0 50→73 Ω; F_MIN 0→0.5 GHz; visualiser added |  - Positions: Systematic 3D grid across brain volume- Right lateral ventricle: 2×1×4cm ellipsoid

| `7d66243` | Dipole arm 10→56 mm (resonance fix); time window 15→60 ns (resolution fix) |

- Separation: 1.5cm center-to-center

### Root causes of previous simulation failures

**Files**: 4800 input files (300 scenarios × 16 transmit antennas each)

**1. Stale `.in` files on HPC** — `generate_dataset.py` was updated locally but old monopole-design files were still running on the HPC. Fix: delete and regenerate on HPC.

**Hemorrhage** (variable):

**2. Arms too short (10 mm → resonance at 6.5 GHz)** — far above the 2 GHz band; S11 showed no dip. Fix: 10 mm → 56 mm arms → 114 mm total → resonance at 1.250 GHz.

**Dataset splits** (in `dataset_metadata.csv`):- Material: εr=61, σ=1.54 S/m (blood)

**3. Time window too short (15 ns → 23 frequency points)** — too coarse to resolve resonance features. Fix: 15 ns → 60 ns → 90 frequency points.

- Train: 210 scenarios (70%)- Sizes: 5-25mm radius

---

- Validation: 44 scenarios (15%)- Positions: 50 locations across brain volume

## References

- Test: 46 scenarios (15%)

- **gprMax**: http://www.gprmax.com

- **Wire dipole example**: `antenna_wire_dipole_fs.in` (150 mm → 950 MHz in free space)---

- **Tissue properties**: Gabriel et al. (1996) — Dielectric properties of biological tissues

- **Coupling medium**: Meaney et al. — Glycerol/water mixtures for microwave brain imaging---

- **CSF properties**: Akhtari et al. (2006) — Conductivities of brain tissues

## Antenna Configuration

---

## Files

## Installation

**Type**: Quarter-wave monopoles (37.5mm for λ/4 at 2 GHz)

```bash

git clone https://github.com/paulpleela/brain-emi-simulation.git| File | Purpose |

cd brain-emi-simulation

conda env create -f gprMax/conda_env.yml|------|---------|**Array**: 16 antennas in circular arrangement

conda activate gprmax

cd gprMax && python setup.py install && cd ..| `generate_dataset.py` | Generate all 300 × 16 = 4800 `.in` files |- Single ring at head center (z = 0.25m)

python -c "import gprMax; print('gprMax installed!')"

```| `generate_inputs.py` | Generate 16 smoke-test files (`brain_tx*.in`) |- Even angular spacing (22.5° between antennas)


| `extract_sparameters.py` | HDF5 `.out` → 16×16 S-matrix → `.s16p` |- Positioned just outside coupling medium

| `visualise_s16p.py` | Plot S-parameters from `.s16p` |

| `validate_s16p.py` | Validate Touchstone file structure |**Excitation**: One transmitter at a time

| `generate_metadata.py` | Create ML metadata CSV |- Each scenario: 16 simulations (one per transmit antenna)

| `dataset_metadata.csv` | Scenario parameters and train/val/test splits |- All antennas act as receivers (transmission lines)

| `run_simulation.sh` | SLURM job script for HPC |- Enables full 16×16 S-parameter matrix extraction

| `brain_inputs/` | Directory with 4800 `.in` input files |

| `sparams/` | Extracted `.s16p` and visualisation PNGs |**Impedance**: 50Ω transmission lines

| `HPC_GUIDE.md` | Complete HPC deployment instructions |

| `gprMax/` | gprMax FDTD solver source code |---



---## Output Files



## Quick StartEach simulation produces `.out` file (HDF5 format):

- **Location**: `brain_inputs/scenario_XXX_txYY.out`

### Generate Dataset- **Size**: ~50-100 MB per file

- **Contents**: 

```bash  - `tls/` group: Transmission line voltage/current data

# Generate all 300 scenarios (4800 files)  - Incident/reflected voltage and current

python generate_dataset.py  - Time-domain waveforms



# Generate metadata CSV**Total dataset**: 4800 output files (~380 GB)

python generate_metadata.py

```---



### Run on HPC## Next Steps



See **`HPC_GUIDE.md`** for complete instructions.### 1. Run HPC Simulations

- Follow `HPC_GUIDE.md`

**Summary**:- Expected: ~19 hours for full dataset

1. Push code: `git push origin main`

2. SSH to Rangpur### 2. Extract S-Parameters

3. Pull code: `cd ~/brain-emi-simulation && git pull`Create script to:

4. Regenerate inputs: `rm brain_inputs/scenario_*.in && python generate_dataset.py`- Read HDF5 `.out` files

5. Setup (first time): `conda env create -f gprMax/conda_env.yml`- Extract transmission line data from `tls/tl*/` groups

6. Test: `python -m gprMax brain_inputs/scenario_001_tx01.in -n 8`- Compute 16×16 S-matrix per scenario

7. Run all: `sbatch run_simulation.sh`- Save as `.s16p` Touchstone or `.npz` format



> **Important**: Always regenerate `.in` files on the HPC after any code change.  ### 3. Train ML Model

> The `.in` files are not committed to git — they must be generated fresh each time.- Use `dataset_metadata.csv` for train/val/test splits

- Input: S-parameter frequency responses (16×16 matrix)

**Runtime**: ~19 hours (300 scenarios, 16 parallel jobs)- Outputs:

  - Classification: Hemorrhage present? (binary)

### Extract and Visualise S-Parameters  - Localization: Position (x, y, z)

  - Sizing: Radius (mm)

```bash

# Extract S-parameters for one scenario---

python extract_sparameters.py --scenario 1

## Dataset Generation

# Visualise

python visualise_s16p.py sparams/scenario_001.s16p### Hemorrhage Position Strategy

```

Positions systematically cover brain volume:

---- **Radial extent**: ±5cm from center (stay in tissue)

- **Vertical extent**: ±8cm from center

## Simulation Parameters- **Avoids**: CSF ventricles (|x| > 1.5cm from center)

- **Distribution**: 3D grid with 50 sampling points

| Parameter | Value | Notes |

|-----------|-------|-------|### Scenario Naming

| **Domain** | 600×600×600 mm | Cubic simulation space |

| **Resolution** | 2 mm | λ/10 at ~1.5 GHz |Format: `scenario_XXX_txYY.in`

| **Time window** | 60 ns | Δf = 16.7 MHz → ~90 pts in 0.5–2 GHz |- `XXX`: Scenario ID (001-300)

| **Waveform** | Gaussian, centre 1.25 GHz | Covers full 0.5–2 GHz band |- `YY`: Transmit antenna (01-16)

| **Antennas** | 16 z-directed wire dipoles | Physical PEC arms, not point sources |

| **Dipole arm length** | 56 mm each | 28 cells @ 2mm grid |**Scenario ranges**:

| **Feed gap** | 2 mm (1 cell) | Carved with `free_space` override |- 001-050: Healthy (no hemorrhage)

| **Total dipole length** | 114 mm | z_bot = 0.194 m, z_top = 0.308 m |- 051-100: 5mm hemorrhage, 50 positions

| **TL impedance** | 73 Ω | Half-wave dipole impedance in free space |- 101-150: 10mm hemorrhage, 50 positions

| **Resonance** | 1.250 GHz | 0.475 × c₀ / 0.114 m (confirmed in sim) |- 151-200: 15mm hemorrhage, 50 positions

| **Array ring** | z = 0.25 m, equatorial | 22.5° spacing, r ≈ 0.114 m |- 201-250: 20mm hemorrhage, 50 positions

| **Confirmed freq. points** | 90 | scenario_001.s16p, 0.517–2.000 GHz |- 251-300: 25mm hemorrhage, 50 positions



------



## Head Model (TRUE Ellipsoidal Geometry)## Key Features



**Geometry**: Voxelized ellipsoidal layers (2 mm grid)### Why This Is Realistic



**Semi-axes**:✅ **Ellipsoidal geometry** (not spherical)

- a = 9.5 cm (front-back, anterior-posterior)- Average adult head dimensions

- b = 7.5 cm (side-side, left-right)- Proper aspect ratios

- c = 11.5 cm (top-bottom, superior-inferior)- Voxelized implementation



**Layers** (outside → inside):✅ **CSF ventricles** (critical EM feature)

- Highest dielectric constant in brain (εr=80)

| Layer | Material | εr | σ (S/m) | Thickness |- Creates strong reflections

|-------|----------|----|---------|-----------|- Affects wave propagation patterns

| Coupling medium | glycerol/water | 36 | 0.3 | 5 mm |

| Scalp + skull | combined | 12 | 0.2 | 10 mm |✅ **Systematic variations**

| Gray matter | cortex | 52 | 0.97 | 3 mm |- 50 hemorrhage positions

| White matter | interior | 38 | 0.57 | fills core |- 5 hemorrhage sizes

| CSF (ventricles) | cerebrospinal fluid | 80 | 2.0 | 2×1×4 cm ellipsoids |- 50 healthy baselines



**CSF ventricles** (critical EM feature):✅ **Clinical relevance**

- Left and right lateral ventricles, each 2×1×4 cm ellipsoid- Hemorrhage sizes: 5-25mm (clinical range)

- Separation: 1.5 cm centre-to-centre- Frequencies: 0-2 GHz (brain penetration)

- εr=80 — highest permittivity in the model; creates strong reflections- Realistic tissue properties (Gabriel 1996)



**Hemorrhage** (variable, only in hemorrhage scenarios):---

- Material: blood — εr=61, σ=1.54 S/m

- Sizes: 5–25 mm radius## References

- Positions: 50 locations across brain volume

- **gprMax**: http://www.gprmax.com

---- **Tissue properties**: Gabriel et al. (1996) - Dielectric properties of biological tissues

- **Frequency selection**: 0-2 GHz optimal for brain imaging (penetration vs resolution)

## Antenna Configuration- **CSF properties**: Akhtari et al. (2006) - Conductivities of brain tissues



**Type**: Physical half-wave wire dipole (gprMax `#edge` + `#transmission_line`)---



**gprMax syntax per antenna** (inside `#python:` block):## Installation

```python

# Full PEC wire (lower arm bottom to upper arm top, including gap)### Prerequisites

#edge: cx cy (cz-arm)  cx cy (cz+arm+gap)  pec- Python 3.6+

# Carve feed gap (overwrite PEC with free_space at feed point)- Conda package manager

#edge: cx cy cz         cx cy (cz+gap)      free_space- Git

# TL placed at gap start — records V and I for S-parameter extraction

#transmission_line: z cx cy cz  73  waveform### Setup

``````bash

# Clone repository

**Dimensions**:git clone https://github.com/paulpleela/brain-emi-simulation.git

| Quantity | Value | Cells (@2 mm) |cd brain-emi-simulation

|----------|-------|--------------|

| Arm length | 56 mm | 28 cells |# Create conda environment

| Feed gap | 2 mm | 1 cell |conda env create -f gprMax/conda_env.yml

| Total dipole | 114 mm | 57 cells |

| Expected resonance | 1.250 GHz | 0.475×c₀/0.114 m |# Activate environment

conda activate gprmax

**Reference**: gprMax official example `antenna_wire_dipole_fs.in`  

(150 mm total → 950 MHz in free space; our 114 mm → 1250 MHz, scaling: 950 × 150/114 ≈ 1250 MHz ✓)# Install gprMax

cd gprMax

**Array layout**:python setup.py install

- 16 antennas evenly spaced at 22.5° intervals in the equatorial plane (z = 0.25 m)cd ..

- All z-directed — no axis-alignment ambiguity```

- TX: 1 active `#transmission_line` (Gaussian waveform)

- RX: 15 passive `#transmission_line` (`rx_null` — records V/I, no excitation)### Verify Installation

- Enables full 16×16 S-parameter matrix per scenario```bash

python -c "import gprMax; print('gprMax installed!')"

---```



## S-Parameter Extraction---



Script: `extract_sparameters.py`## Contact



| Setting | Value |For questions about this simulation setup, see GitHub repository:

|---------|-------|https://github.com/paulpleela/brain-emi-simulation

| Reference impedance Z₀ | 73 Ω |
| Frequency range | 0.5–2.0 GHz |
| Output format | Touchstone `.s16p` |
| Header normalization | `R 73` |

**Confirmed output** (scenario_001):
- 90 frequency points
- Range: 0.517–2.000 GHz
- All 16×16 ports present

---

## Changelog

| Commit | Change |
|--------|--------|
| initial | Hertzian dipole (point source), no physical resonance |
| `15fe072` | Physical wire dipole: correct PEC arm + free_space gap syntax |
| `60b4b1b` | Extractor: Z₀ 50→73 Ω; F_MIN 0→0.5 GHz; visualiser added |
| `7d66243` | **Dipole arm 10→56 mm** (resonance fix); **time window 15→60 ns** (resolution fix) |

### Root causes of previous failures

**1. Stale `.in` files on HPC**  
`generate_dataset.py` was updated locally but the HPC was still running old files with a monopole design (`#cylinder`, 50 Ω, x/y-polarised). S11≈1 for all ports — antennas were not matched at all.  
Fix: `rm brain_inputs/scenario_*.in && python generate_dataset.py` on HPC.

**2. Dipole arms too short (10 mm → resonance at 6.5 GHz)**  
Total dipole = 22 mm → resonance = 0.475 × c₀ / 0.022 = **6.5 GHz** — far above the 2 GHz band. S11 showed monotonic decrease, no dip in band.  
Fix: `dipole_arm_len: 0.010 → 0.056 m` → total 114 mm → resonance **1.250 GHz** ✓

**3. Time window too short (15 ns → only 23 frequency points)**  
Δf = 1/15 ns = 67 MHz → only 23 points in 0.5–2 GHz. Any narrow resonance feature would be missed.  
Fix: `time_window: 15e-9 → 60e-9` → Δf = 16.7 MHz → **90 points** ✓ (matches gprMax's own dipole example)

---

## Key Physics Notes

### Why wire dipoles (not monopoles or Hertzian dipoles)

| Antenna type | Problem |
|---|---|
| Hertzian dipole (`#hertzian_dipole`) | Point source — no physical impedance, no resonance behaviour |
| Monopole over ground plane (`#cylinder` + `#box`) | Requires ground plane; polarisation ambiguity for off-axis positions |
| **Wire dipole (`#edge` + `#transmission_line`)** | Physical arms, defined impedance, correct resonance, z-directed ✓ |

### Coupling medium purpose

The 5 mm glycerol/water shell (εr=36, σ=0.3 S/m) improves impedance matching between the free-space antennas and the high-permittivity head tissues. It is **not** an absorber — the PML walls handle domain termination. Low σ (0.3 S/m) keeps signal attenuation low before the wave reaches the head.

### Frequency band

| Consideration | Value |
|---|---|
| Upper limit | 2 GHz — λ/10 = 15 mm > 2 mm cell ✓ |
| Lower limit | ~0.5 GHz — Gaussian waveform at 1.25 GHz |
| Resonance | 1.25 GHz — dipole resonance in free space |
| Penetration | ~1–3 dB/cm attenuation in tissue at 1–2 GHz |

---

## Dataset Generation

### Hemorrhage Position Strategy

Positions systematically cover brain volume:
- **Radial extent**: ±5 cm from centre (stay in tissue)
- **Vertical extent**: ±8 cm from centre
- **Avoids**: CSF ventricles (|x| > 1.5 cm from centre)
- **Distribution**: 3D grid with 50 sampling points per size

### Scenario Naming

Format: `scenario_XXX_txYY.in`
- `XXX`: Scenario ID (001–300)
- `YY`: Transmit antenna (01–16)

**Scenario ranges**:
- 001–050: Healthy (no hemorrhage)
- 051–100: 5 mm hemorrhage, 50 positions
- 101–150: 10 mm hemorrhage, 50 positions
- 151–200: 15 mm hemorrhage, 50 positions
- 201–250: 20 mm hemorrhage, 50 positions
- 251–300: 25 mm hemorrhage, 50 positions

---

## References

- **gprMax**: http://www.gprmax.com  
- **Wire dipole example**: `antenna_wire_dipole_fs.in` (150 mm → 950 MHz in free space)  
- **Tissue properties**: Gabriel et al. (1996) — Dielectric properties of biological tissues  
- **Coupling medium**: Meaney et al. — Glycerol/water mixtures for microwave brain imaging  
- **CSF properties**: Akhtari et al. (2006) — Conductivities of brain tissues  

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

https://github.com/paulpleela/brain-emi-simulation
