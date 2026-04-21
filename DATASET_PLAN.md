# Dataset Plan (Current) - 1000 Scenarios

This document reflects the active dataset design implemented by:
- [generate_metadata.py](generate_metadata.py)
- [generate_dataset.py](generate_dataset.py)
- [run_simulation_core.sh](run_simulation_core.sh)

## Scope

- Total scenarios: 1000
- Scenario IDs: 1..1000 (three-digit file names, e.g. scenario_001)
- Ports per scenario: 16 TX files (`tx01`..`tx16`)
- Output targets:
  - Frequency domain: `sparams/scenario_XXX.s16p`
  - Frequency-domain ML tensor: `fd_tensors/scenario_XXX_fd.npz`
  - Global normalization stats: `fd_tensors/normalization_freq_full.npz`

## Split Policy

- Split sizes: train=700, val=150, test=150
- Assignment method: deterministic shuffle of scenario IDs with fixed seed (not contiguous ID ranges)
- Scenario 1 is pinned to train as the explicit base case

## Class Composition

- No anomaly: 300 samples
- Anomaly: 700 samples
- Per split target composition: 30% healthy, 70% anomaly

### No-Anomaly Groups (300 total)

- N1_baseline / N2_property_variation / N3_noise_variation
- Deterministic distribution with near-even counts
- Scenario 1 is forced to N1 baseline for controlled comparisons

### Anomaly Group (700 total)

- Group label: `A_anomaly`
- Size buckets (lesion radius):
  - small: 230 (5-10 mm)
  - medium: 240 (10-20 mm)
  - large: 230 (20-30 mm)
- Regions: left, right, deep, boundary
- Shapes: sphere and ellipsoid (region-local 70/30 split target)
- Noise schedule: deterministic 50% low, 30% medium, 20% high within split/label subgroups

## Primary Analysis Variables

- `head_scale` in [0.9, 1.1]
- `head_rotation_deg` in [-15, 15] (z-axis rotation)
- `noise_level` in {none, low, medium, high}

Coverage policy:
- `low/medium/high` are required in every split.
- `none` is reserved for the single base case only.
- Generation enforces that no analysis variable is isolated to a single split.

Independence policy:
- `head_scale`, `head_rotation_deg`, and `noise_level` are sampled independently of lesion presence, with matched coverage across split/label subgroups to reduce spurious correlations.

## Base Case

- Exactly one base case is encoded with `is_base_case=1`.
- Current base case is scenario 1 and is fixed to:
  - healthy (`has_lesion=0`)
  - `head_scale=1.0`
  - `head_rotation_deg=0.0`
  - `noise_level=none`
- Additional near-nominal cases can appear naturally (values close to nominal) and are useful for statistical comparisons around the baseline.

## Metadata Fields (Current CSV)

`dataset_metadata.csv` columns:

- `scenario_id`
- `is_base_case`
- `has_lesion`
- `lesion_size_mm`
- `lesion_x`
- `lesion_y`
- `lesion_z`
- `epsilon_variation`
- `sigma_variation`
- `head_scale`
- `head_rotation_deg`
- `noise_level`
- `split`
- `group`
- `size_bucket`
- `region`
- `shape`
- `epsilon_anomaly_variation`
- `sigma_anomaly_variation`
- `background_epsilon_variation`
- `background_sigma_variation`

## Parameter Ranges and Mathematical Usage

This section links metadata values to their implemented numeric ranges and where
they appear in the simulation/tensor math.

### Global Geometry and Grid Constants

- Cell size: `CELL = 0.002 m`
- Head center: `(0.25, 0.25, 0.25) m`
- Base head semi-axes: `a=0.095 m`, `b=0.075 m`, `c=0.115 m`
- Layer thicknesses: scalp/skull `0.010 m`, gray-matter shell `0.003 m`
- Coupling thickness: `0.020 m`

### Metadata Value Ranges

| Parameter | Value Range / Allowed Values | Units | Notes |
|---|---|---|---|
| `head_scale` | `[0.9, 1.1]` | scale factor | Base case fixed to `1.0` |
| `head_rotation_deg` | `[-15, 15]` | degrees | z-axis rotation; base case fixed to `0.0` |
| `noise_level` | `{none, low, medium, high}` | category | `none` reserved for base case |
| `lesion_size_mm` | healthy: `0`; anomaly: small `[5,10]`, medium `[10,20]`, large `[20,30]` | mm | Lesion radius (not diameter) |
| `lesion_x` | healthy: `0`; anomaly by region: left `[-0.050,-0.015]`, right `[0.015,0.050]`, deep `[-0.020,0.020]`, boundary via elliptical near-skull band (`theta in [0,2pi)`, `frac in [0.85,0.95]`) | m | Local x offset before scale/rotation |
| `lesion_y` | healthy: `0`; anomaly by region: left/right `[-0.030,0.030]`, deep `[-0.020,0.020]`, boundary via elliptical near-skull band (`theta in [0,2pi)`, `frac in [0.85,0.95]`) | m | Local y offset before scale/rotation |
| `lesion_z` | healthy: `0`; anomaly by region: left/right `[-0.070,0.070]`, deep `[-0.050,0.050]`, boundary `[-0.060,0.060]` | m | Local z offset before scale/rotation |
| `epsilon_variation` | N1 `[-2,2]`, N2 `[-10,10]`, N3 `[-5,5]`, anomaly uses background draw `[-10,10]` | % | Background dielectric variation |
| `sigma_variation` | N1 `[-2,2]`, N2 `[-10,10]`, N3 `[-5,5]`, anomaly uses background draw `[-10,10]` | % | Background conductivity variation |
| `background_epsilon_variation` | healthy mirrors `epsilon_variation`; anomaly `[-10,10]` | % | Explicit background dielectric field |
| `background_sigma_variation` | healthy mirrors `sigma_variation`; anomaly `[-10,10]` | % | Explicit background conductivity field |
| `epsilon_anomaly_variation` | healthy: `0`; anomaly `[-15,15]` | % | Lesion dielectric variation |
| `sigma_anomaly_variation` | healthy: `0`; anomaly `[-15,15]` | % | Lesion conductivity variation |

- `head_scale`: `[0.9, 1.1]` (base case fixed to `1.0`)
- `head_rotation_deg`: `[-15, 15]` degrees about z-axis (base case fixed to `0.0`)
- `noise_level`: `{none, low, medium, high}` where `none` is base case only
- `lesion_size_mm` (anomaly only):
  - interpreted as radius (not diameter)
  - small: `[5, 10]`
  - medium: `[10, 20]`
  - large: `[20, 30]`
- `lesion_x, lesion_y, lesion_z` in meters (anomaly only):
  - left: `x in [-0.050, -0.015]`, `y in [-0.030, 0.030]`, `z in [-0.070, 0.070]`
  - right: `x in [0.015, 0.050]`, `y in [-0.030, 0.030]`, `z in [-0.070, 0.070]`
  - deep: `x in [-0.020, 0.020]`, `y in [-0.020, 0.020]`, `z in [-0.050, 0.050]`
  - boundary: `theta in [0, 2pi)`, `frac in [0.85, 0.95]`, `z in [-0.060, 0.060]`, with
    - `r_edge(theta) = 1 / sqrt((cos(theta)^2/0.095^2) + (sin(theta)^2/0.075^2))`
    - `r = frac * r_edge(theta)`
    - `x=r*cos(theta)`, `y=r*sin(theta)`
- Background property variation (%):
  - healthy N1 baseline: `epsilon_variation, sigma_variation in [-2, 2]`
  - healthy N2 property variation: `epsilon_variation, sigma_variation in [-10, 10]`
  - healthy N3 noise variation: `epsilon_variation, sigma_variation in [-5, 5]`
  - anomaly background: `background_epsilon_variation, background_sigma_variation in [-10, 10]`
- Lesion property variation (%), anomaly only:
  - `epsilon_anomaly_variation, sigma_anomaly_variation in [-15, 15]`

### How Values Are Used Mathematically

1. Material property scaling (percent to absolute)
   - `scaled_value = base_value * (1 + pct/100)`
   - Applied to dielectric constant (`eps`) and conductivity (`sig`) in:
     - coupling/scalp/gray/white/csf using background variation
     - blood (lesion material) using anomaly variation

2. Head pose transform
   - Convert `theta = deg2rad(head_rotation_deg)`
   - World-to-head local transform used during voxelization:
     - `dx = x - cx`, `dy = y - cy`, `dz = z - cz`
     - `xr = cos(theta)*dx + sin(theta)*dy`
     - `yr = -sin(theta)*dx + cos(theta)*dy`
     - Local coordinates entering ellipsoid tests:
       - `x_local = xr / head_scale`
       - `y_local = yr / head_scale`
       - `z_local = dz / head_scale`
   - This makes anatomy simultaneously scaled and rotated in a physically consistent way.

3. Lesion placement transform
   - Metadata lesion offsets are first scaled:
     - `x_s = lesion_x * head_scale`, `y_s = lesion_y * head_scale`, `z_s = lesion_z * head_scale`
   - Then rotated about z-axis:
     - `x_r = cos(theta)*x_s - sin(theta)*y_s`
     - `y_r = sin(theta)*x_s + cos(theta)*y_s`
     - `z_r = z_s`
   - Final lesion center:
     - `(x, y, z) = head_center + (x_r, y_r, z_r)`
  - `lesion_size_mm` is stored as radius.
  - Sphere lesion radius uses `lesion_size_mm/1000` (meters).
  - Ellipsoid lesion axes are `(ra, rb, rc) = (r, 0.8r, 1.2r)`.

4. Antenna ring positioning
   - Ring radii are fixed for all scenarios:
     - `r_x = 0.124 m`
     - `r_y = 0.104 m`
   - Antenna `i` angle: `phi_i = 2*pi*i/16`
   - Position:
     - `x_i = head_cx + r_x*cos(phi_i)`
     - `y_i = head_cy + r_y*sin(phi_i)`

5. Frequency-domain channel construction
   - For each frequency bin `f`, S-parameters are read as complex values:
     - `S_ij(f) = mag_ij(f) * exp(j * deg2rad(angle_ij(f)))`
   - Full matrix flattened to 256 channels (`16x16`), then split to real/imag:
     - output tensor shape: `(512, F)` where channel pairs are `(Re(S_ij), Im(S_ij))`

6. Measurement-noise injection in tensor space
   - Noise scales by metadata `noise_level`:
     - `none: 0.0`, `low: 0.001`, `medium: 0.005`, `high: 0.01`
   - Per channel, sigma is proportional to that channel's empirical std:
     - `sigma_ch = max(std_ch * noise_scale, 1e-8)`
   - Additive noise:
     - `Re' = Re + N(0, sigma_re)`
     - `Im' = Im + N(0, sigma_im)`

7. Dataset normalization (fit on train split only)
   - For each channel-frequency entry:
     - `x_norm = (x - mean_train) / max(std_train, 1e-8)`
   - The same train-derived `mean/std` is applied to train, val, and test.

## Generation and Simulation Flow

1. Generate metadata source-of-truth:
    - `python generate_metadata.py`
2. Generate one or more scenario input files from metadata:
    - `python generate_dataset.py --scenario N`
    - `python generate_dataset.py --range A B`
3. Run sequential low-storage pipeline (default behavior):
    - [run_simulation.sh](run_simulation.sh) for CPU
    - [run_simulation_gpu.sh](run_simulation_gpu.sh) for GPU
4. Per scenario, pipeline does:
    - generate `.in` files
    - run 16 simulations
    - build `.s16p`
    - build frequency-domain full-S tensor `.npz`
    - delete intermediates (`.in`/`.out`) by default

## ML Input Format

Current NPZ output from [build_fd_tensors.py](build_fd_tensors.py):

- Key `signal`: shape `(512, F)`
- Key `channels`: channel names
- Channels are full Sij real/imag pairs:
  - `S11_real`, `S11_imag`, ..., `S1616_real`, `S1616_imag`

Noise injection for ML tensors:
- Applied after `.s16p -> tensor` conversion (not during simulation)
- Gaussian perturbation is added independently to real and imaginary channels
- Per-channel noise standard deviation is proportional to that channel's signal standard deviation
- Scale factors: `none`: 0.0, `low`: 0.001, `medium`: 0.005, `high`: 0.01

## Notes

- The previous 1360-scenario anatomy/tilt expansion plan is not active.
- Current implementation is metadata-driven and deterministic.
- Use [README.md](README.md) as the operational runbook.
