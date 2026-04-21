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
- Size buckets:
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
