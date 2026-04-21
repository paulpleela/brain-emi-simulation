# Brain EMI Simulation for Stroke Detection

Electromagnetic brain-imaging simulation pipeline using gprMax.

This repository supports a full workflow:
1. Generate stratified scenario metadata.
2. Generate gprMax input files from metadata.
3. Run low-storage sequential simulations on HPC.
4. Extract Touchstone S-parameters.
5. Convert S-parameters to frequency-domain full-S tensors for deep learning.

## Current Workflow

### 1) Generate metadata (source of truth)

```bash
conda run -n brain-emi-simulation python generate_metadata.py
```

This writes [dataset_metadata.csv](dataset_metadata.csv) with:
- `scenario_id` (1..1000)
- deterministic shuffled split assignment (train=700, val=150, test=150)
- label field (`has_lesion`) with balanced split composition (30% healthy, 70% anomaly per split)
- lesion geometry and tissue variation fields
- primary analysis variables:
  - `head_scale` in [0.9, 1.1]
  - `head_rotation_deg` in [-15, 15]
  - `noise_level` in {none, low, medium, high}
- base-case marker `is_base_case` (scenario 1 is fixed baseline)

Base case policy:
- Scenario 1 is the controlled baseline: healthy, `head_scale=1.0`, `head_rotation_deg=0.0`, `noise_level=none`.
- All other scenarios use low/medium/high noise schedules for robustness analysis.
- Healthy groups are `N1_baseline`, `N2_property_variation`, and `N3_noise_variation`.

### 2) Generate simulation inputs from metadata (optional/manual)

```bash
conda run -n brain-emi-simulation python generate_dataset.py
```

This creates 16 `.in` files per scenario in [brain_inputs](brain_inputs):
- `scenario_001_tx01.in` ... `scenario_001_tx16.in`
- ... up to scenario 1000

You can generate subsets:

```bash
conda run -n brain-emi-simulation python generate_dataset.py --scenario 1
conda run -n brain-emi-simulation python generate_dataset.py --range 1 20
```

Important:
- The default SLURM runners already call [generate_dataset.py](generate_dataset.py) per scenario.
- You only need manual generation when debugging or inspecting `.in` files.

## HPC Execution

### Single-scenario smoke test (one TX file)

Use [test_single/run_simulation.sh](test_single/run_simulation.sh):

```bash
mkdir -p logs
sbatch test_single/run_simulation.sh
```

This runs only `brain_inputs/scenario_001_tx01.in`.

### Full CPU run (all scenarios)

Use [run_simulation.sh](run_simulation.sh). This is now the default low-storage mode.

```bash
mkdir -p logs
sbatch run_simulation.sh
```

Notes:
- It processes scenarios sequentially (default `1..1000`).
- For each scenario: generate `.in` -> run 16 TX -> extract `.s16p` + `.npz` -> delete `.in/.out`.
- Override scenario range:

```bash
sbatch --export=ALL,START_SCENARIO=1,END_SCENARIO=20 run_simulation.sh
```

### Full GPU run (recommended for speed)

Use [run_simulation_gpu.sh](run_simulation_gpu.sh):

```bash
mkdir -p logs
sbatch run_simulation_gpu.sh
```

Notes:
- The script is configured for `a100` with `--gres=gpu:1`.
- It uses the same sequential low-storage pipeline as CPU, but runs gprMax with `-gpu`.
- Override scenario range:

```bash
sbatch --export=ALL,START_SCENARIO=1,END_SCENARIO=20 run_simulation_gpu.sh
```

### Sequential core script

[run_simulation_core.sh](run_simulation_core.sh) is the underlying
engine used by both [run_simulation.sh](run_simulation.sh) and
[run_simulation_gpu.sh](run_simulation_gpu.sh). It processes one scenario at a time:
1. generate inputs for one scenario,
2. run all 16 TX jobs,
3. extract `.s16p` and `.npz`,
4. delete that scenario's `.in`/`.out`,
5. continue to next scenario.

Single scenario (scenario 1):

```bash
mkdir -p logs
sbatch run_simulation_core.sh
```

Scenario range on CPU (example 1..20):

```bash
sbatch --export=ALL,START_SCENARIO=1,END_SCENARIO=20 run_simulation_core.sh
```

All scenarios on CPU (1..1000):

```bash
sbatch --export=ALL,START_SCENARIO=1,END_SCENARIO=1000 run_simulation_core.sh
```

GPU sequential mode:

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --export=ALL,START_SCENARIO=1,END_SCENARIO=1000,USE_GPU=1 \
  run_simulation_core.sh
```

Toggles:
- keep `.out` files: add `DELETE_OUT=0`
- keep `.in` files: add `DELETE_IN=0`

## S-Parameter and Frequency-Domain Extraction

### From `.out` to `.s16p`

```bash
conda run -n brain-emi-simulation python build_s16p.py --scenario 1
conda run -n brain-emi-simulation python build_s16p.py --range 1 100
conda run -n brain-emi-simulation python build_s16p.py --all
```

Output folder: [sparams](sparams)

### From `.s16p` to training tensors (frequency full-S only)

```bash
conda run -n brain-emi-simulation python build_fd_tensors.py --scenario 1
conda run -n brain-emi-simulation python build_fd_tensors.py --range 1 100
conda run -n brain-emi-simulation python build_fd_tensors.py --all

# Fit train-only normalization stats and generate tensors for all selected scenarios
conda run -n brain-emi-simulation python build_fd_tensors.py --all --fit-stats
```

Output folder: [fd_tensors](fd_tensors)

ML tensor format (fixed):
- NPZ keys: `signal`, `channels`
- Full S-matrix channels only (`S11..S1616`, real/imag pairs)
- `signal` shape: `(512, F)` for 16-port setup
- Measurement noise is injected after `.s16p -> tensor` conversion using metadata `noise_level`:
  - Gaussian perturbation is added independently to real/imag channels
  - per-channel std is scaled from that channel's signal std
  - scale factors: `none=0.0`, `low=0.001`, `medium=0.005`, `high=0.01`
- Normalization: train-split fit only, reused for train/val/test
- Global stats file: `fd_tensors/normalization_freq_full.npz`

### End-to-end wrapper

Run both extraction stages in one command:

```bash
conda run -n brain-emi-simulation python run_extraction_pipeline.py --scenario 1 --keep-out
```

Also supports `--range` and `--all`.

## Visualisation

### Frequency-domain (`.s16p`)

```bash
conda run -n brain-emi-simulation python visualise_s16p.py sparams/scenario_001.s16p
```

## Key Scripts

- [generate_metadata.py](generate_metadata.py): stratified deterministic metadata generator
- [generate_dataset.py](generate_dataset.py): metadata-driven `.in` generator
- [run_simulation.sh](run_simulation.sh): default CPU SLURM sequential pipeline
- [run_simulation_gpu.sh](run_simulation_gpu.sh): default GPU SLURM sequential pipeline
- [run_simulation_core.sh](run_simulation_core.sh): shared sequential engine
- [build_s16p.py](build_s16p.py): `.out -> .s16p`
- [build_fd_tensors.py](build_fd_tensors.py): `.s16p -> frequency-domain full-S tensors with train-fit normalization`
- [run_extraction_pipeline.py](run_extraction_pipeline.py): combined extraction wrapper
- [visualise_s16p.py](visualise_s16p.py): S-parameter plotting

## Recommended Run Order on HPC

```bash
# 1) Pull latest code

git pull

# 2) Regenerate metadata
conda run -n brain-emi-simulation python generate_metadata.py

# 3) Launch simulations (inputs are generated per scenario automatically)
mkdir -p logs
sbatch run_simulation.sh
# or: sbatch run_simulation_gpu.sh
```

## Cleanup Notes

Legacy files removed from the active workflow:
- `generate_inputs.py`
- `test_single_job.sh`

Use [test_single/run_simulation.sh](test_single/run_simulation.sh) for single-file smoke tests.
