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

- Train: 1..700 (700 samples)
- Validation: 701..850 (150 samples)
- Test: 851..1000 (150 samples)

Split is deterministic from `scenario_id`.

## Class Composition

- No anomaly: 300 samples
- Anomaly: 700 samples

### No-Anomaly Groups (300 total)

- N1_baseline: 100
- N2_property_variation: 100
- N3_measurement_variation: 100

### Anomaly Group (700 total)

- Group label: `A_anomaly`
- Size buckets:
  - small: 230 (5-10 mm)
  - medium: 240 (10-20 mm)
  - large: 230 (20-30 mm)
- Regions: left, right, deep, boundary
- Shapes: sphere and ellipsoid (region-local 70/30 split target)
- Noise schedule: 50% low, 30% medium, 20% high (deterministic shuffle)

## Metadata Fields (Current CSV)

`dataset_metadata.csv` columns:

- `scenario_id`
- `has_lesion`
- `lesion_size_mm`
- `lesion_x`
- `lesion_y`
- `lesion_z`
- `epsilon_variation`
- `sigma_variation`
- `antenna_offset`
- `coupling_thickness`
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
    - build time-domain `.npz`
    - delete intermediates (`.in`/`.out`) by default

## ML Input Format

Current NPZ output from [build_fd_tensors.py](build_fd_tensors.py):

- Key `signal`: shape `(32, T)`
- Key `channels`: channel names
- Channels are Sii-only, real/imag pairs:
  - `S11_real`, `S11_imag`, ..., `S1616_real`, `S1616_imag`

## Notes

- The previous 1360-scenario anatomy/tilt expansion plan is not active.
- Current implementation is metadata-driven and deterministic.
- Use [README.md](README.md) as the operational runbook.
