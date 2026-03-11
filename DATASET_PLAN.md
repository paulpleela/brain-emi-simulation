# Dataset Expansion Plan — 1010 Scenarios

**Goal**: Expand from 300 → ≥1000 scenarios for viable deep learning training.  
**Strategy**: Vary two clinically realistic axes (anatomy, head tilt) independently of the hemorrhage dimension, so the model must generalise across patients and scanner positioning.

---

## Design Principles

1. **Healthy = anatomy × tilt**: Every combination of head anatomy variant and head tilt angle forms one healthy scenario. The model learns what "normal" looks like across the population.

2. **Hemorrhage = anatomy × (size × location)**: Each hemorrhage case uses a representative anatomy variant crossed with every combination of lesion size and location. Tilt is kept at 0° for hemorrhage scenarios — the location/size variation is the signal; adding tilt on top would expand the dataset too much without adding proportional ML value.

3. **Shape is fixed (sphere)**: Haemorrhages are always spherical. This keeps the geometry simple and ensures size is the only shape parameter the model must learn.

4. **Class balance**: Healthy (13%) vs Hemorrhage (87%) is intentionally unbalanced toward hemorrhage — there are many more (size, location) combinations than anatomy×tilt combinations. Training should use class-weighted loss or oversampling.

---

## Dimension 1 — Head Anatomy Variants

Captures inter-subject variability. Three independent parameters, each with a small set of clinically realistic values:

### 1a. Scalp + Skull Thickness
Population range from MRI: 6–15 mm.

| Label | Value | Notes |
|-------|-------|-------|
| `thin` | 8 mm | Child / thin adult |
| `normal` | 10 mm | Average adult (current baseline) |
| `thick` | 13 mm | Older adult, prominent bone |

### 1b. Cortical Gray Matter Thickness
Population range from MRI: 1.5–4.5 mm.

| Label | Value | Notes |
|-------|-------|-------|
| `thin` | 2 mm | Elderly / atrophy |
| `normal` | 3 mm | Average adult (current baseline) |
| `thick` | 4 mm | Young adult |

### 1c. Brain Size (Semi-axes)
Scales all three semi-axes together.

| Label | a (front-back) | b (left-right) | c (top-bottom) | Notes |
|-------|---------------|----------------|----------------|-------|
| `small` | 90 mm | 70 mm | 110 mm | Small head, ~5th percentile |
| `average` | 95 mm | 75 mm | 115 mm | Current baseline |
| `large` | 100 mm | 80 mm | 120 mm | Large head, ~95th percentile |

**Full factorial**: 3 × 3 × 3 = **27 anatomy configurations**

---

## Dimension 2 — Head Tilt

Captures realistic patient positioning in a scanner. The antenna ring remains fixed; the head ellipsoid is rotated about the head centre before voxelisation.

Tilt is implemented as a 3D rotation of the coordinate system used for ellipsoid testing — no physical domain change needed.

| ID | Roll (°) | Pitch (°) | Description |
|----|---------|----------|-------------|
| `T0` | 0 | 0 | Upright (canonical) |
| `T1` | +8 | 0 | Roll left |
| `T2` | −8 | 0 | Roll right |
| `T3` | 0 | +8 | Pitch forward (chin down) |
| `T4` | 0 | −8 | Pitch backward (chin up) |

**5 tilt configurations**

**Implementation note**: Rotation matrix `R(roll, pitch)` applied to each voxel coordinate `(x−cx, y−cy, z−cz)` before the ellipsoid membership test. The coupling medium, scalp/skull and brain layers all rotate together; the hemorrhage position (stored as head-relative) also rotates.

---

## Healthy Scenarios

| Axis | Count |
|------|-------|
| Anatomy configurations | 27 |
| Tilt configurations | 5 |
| **Total healthy** | **135** |

Scenario IDs: `0001–0135`  
Naming: `scenario_{id:04d}_tx{n:02d}.in`

Each healthy scenario encodes: `(scalp_mm, gray_mm, head_size, roll_deg, pitch_deg)`

---

## Hemorrhage Scenarios

### Lesion Size
5 spherical radii covering the clinical range for intracerebral haemorrhage:

| Label | Radius | Volume | Clinical equivalent |
|-------|--------|--------|---------------------|
| `S1` | 5 mm | 0.5 mL | Microbleed / petechial |
| `S2` | 10 mm | 4.2 mL | Small ICH |
| `S3` | 15 mm | 14 mL | Moderate ICH |
| `S4` | 20 mm | 34 mL | Large ICH |
| `S5` | 25 mm | 65 mL | Massive ICH |

### Lesion Location
35 systematic positions covering the clinically relevant brain regions. Coordinates are head-relative offsets from head centre (metres):

| # | Region | x (m) | y (m) | z (m) | Notes |
|---|--------|-------|-------|-------|-------|
| L01 | Right frontal | +0.040 | 0 | +0.040 | Frontal lobe |
| L02 | Left frontal | −0.040 | 0 | +0.040 | |
| L03 | Right parietal | +0.040 | 0 | 0 | Equatorial |
| L04 | Left parietal | −0.040 | 0 | 0 | |
| L05 | Right temporal | 0 | +0.040 | 0 | |
| L06 | Left temporal | 0 | −0.040 | 0 | |
| L07 | Right occipital | +0.030 | 0 | −0.040 | |
| L08 | Left occipital | −0.030 | 0 | −0.040 | |
| L09 | Superior right | +0.025 | 0 | +0.060 | Near vertex |
| L10 | Superior left | −0.025 | 0 | +0.060 | |
| L11 | Inferior right | +0.030 | 0 | −0.055 | Near base |
| L12 | Inferior left | −0.030 | 0 | −0.055 | |
| L13 | Right deep | +0.020 | +0.015 | +0.010 | Basal ganglia region |
| L14 | Left deep | −0.020 | −0.015 | +0.010 | |
| L15 | Posterior right | +0.030 | +0.030 | −0.020 | |
| L16 | Posterior left | −0.030 | −0.030 | −0.020 | |
| L17 | Anterior right | +0.035 | −0.020 | +0.030 | |
| L18 | Anterior left | −0.035 | +0.020 | +0.030 | |
| L19 | Central right | +0.020 | 0 | +0.020 | Periventricular |
| L20 | Central left | −0.020 | 0 | +0.020 | |
| L21 | Superior central | 0 | +0.020 | +0.050 | |
| L22 | Right mid-lateral | +0.040 | +0.020 | +0.010 | |
| L23 | Left mid-lateral | −0.040 | −0.020 | +0.010 | |
| L24 | Right posterior-superior | +0.025 | +0.025 | +0.040 | |
| L25 | Left posterior-superior | −0.025 | −0.025 | +0.040 | |
| L26 | Right frontal deep | +0.030 | −0.015 | +0.050 | |
| L27 | Left frontal deep | −0.030 | +0.015 | +0.050 | |
| L28 | Right inferior temporal | +0.020 | +0.045 | −0.020 | |
| L29 | Left inferior temporal | −0.020 | −0.045 | −0.020 | |
| L30 | Right superior parietal | +0.035 | +0.010 | +0.045 | |
| L31 | Left superior parietal | −0.035 | −0.010 | +0.045 | |
| L32 | Posterior central | 0 | −0.030 | −0.030 | Occipital |
| L33 | Anterior central | 0 | +0.030 | +0.030 | Prefrontal |
| L34 | Right subcortical | +0.015 | +0.030 | −0.010 | Near thalamus |
| L35 | Left subcortical | −0.015 | −0.030 | −0.010 | |

> All positions are head-relative offsets (metres) from head_center = (0.25, 0.25, 0.25).  
> Each position must be verified to lie inside the white/gray matter boundary at the given anatomy variant before voxelisation; a safety check in `generate_dataset.py` will flag any that fall outside.

### Anatomy Configurations for Hemorrhage
7 representative anatomy configs (not full 27-factorial — keeps hemorrhage count tractable):

| ID | Scalp (mm) | Gray (mm) | Head size | Notes |
|----|-----------|----------|-----------|-------|
| `A1` | 8 | 2 | small | Thin child-like |
| `A2` | 8 | 3 | average | Thin scalp, normal brain |
| `A3` | 10 | 2 | average | Normal scalp, thin cortex |
| `A4` | 10 | 3 | average | **Canonical** (current baseline) |
| `A5` | 10 | 4 | average | Normal scalp, thick cortex |
| `A6` | 13 | 3 | average | Thick scalp, normal |
| `A7` | 13 | 3 | large | Thick scalp, large head |

Tilt: always `T0` (0°, upright) for hemorrhage scenarios.

### Hemorrhage Count

| Factor | Count |
|--------|-------|
| Lesion sizes | 5 |
| Lesion locations | 35 |
| Anatomy configs | 7 |
| **Total hemorrhage** | **1,225** |

Scenario IDs: `0136–1360`

---

## Grand Total

| Group | Scenarios | .in files |
|-------|-----------|-----------|
| Healthy (27 anatomy × 5 tilt) | 135 | 2,160 |
| Hemorrhage (5 sizes × 35 locations × 7 anatomy) | 1,225 | 19,600 |
| **Total** | **1,360** | **21,760** |

### Dataset Splits (stratified by class)

| Split | Healthy | Hemorrhage | Total |
|-------|---------|------------|-------|
| Train (80%) | 108 | 980 | 1,088 |
| Val (10%) | 14 | 122 | 136 |
| Test (10%) | 13 | 123 | 136 |

---

## Metadata Schema

`dataset_metadata.csv` — one row per scenario:

| Column | Type | Description |
|--------|------|-------------|
| `scenario_id` | int | 1–1010 |
| `has_lesion` | 0/1 | Healthy=0, Hemorrhage=1 |
| `scalp_skull_mm` | float | Scalp+skull thickness (mm) |
| `gray_mm` | float | Cortical gray matter thickness (mm) |
| `head_size` | str | small / average / large |
| `a_mm` | float | Brain semi-axis a (mm) |
| `b_mm` | float | Brain semi-axis b (mm) |
| `c_mm` | float | Brain semi-axis c (mm) |
| `roll_deg` | float | Head roll angle (deg) |
| `pitch_deg` | float | Head pitch angle (deg) |
| `lesion_size_mm` | float | Hemorrhage radius (mm), 0 if healthy |
| `lesion_x` | float | Head-relative x offset (m), 0 if healthy |
| `lesion_y` | float | Head-relative y offset (m) |
| `lesion_z` | float | Head-relative z offset (m) |
| `lesion_region` | str | Anatomical label (e.g. "right_frontal"), "" if healthy |
| `split` | str | train / val / test |

---

## ML Task Definitions

The richer dataset enables training for multiple tasks simultaneously:

| Task | Output | Label columns |
|------|--------|---------------|
| Detection (binary) | Is hemorrhage present? | `has_lesion` |
| Localisation (regression) | Where is the lesion? | `lesion_x, lesion_y, lesion_z` |
| Sizing (regression) | How large is the lesion? | `lesion_size_mm` |
| Anatomy robustness | Does model hold across anatomy? | `scalp_skull_mm, gray_mm, head_size` |
| Tilt robustness | Does model hold across tilts? | `roll_deg, pitch_deg` |

---

| Storage and Runtime Estimates

| Quantity | Estimate |
|----------|----------|
| Raw `.out` files | 21,760 × ~75 MB = ~1.6 TB |
| Extracted `.s16p` files | 1,360 × ~0.2 MB = ~270 MB |
| Time on HPC (16 parallel slots, 5 min/sim) | ~113 hours (~4.7 days) |

> If runtime is a constraint, run healthy scenarios first (2,160 files = ~11 h), verify S-parameters, then proceed with hemorrhage.

---

## Implementation Changes Required

### `generate_dataset.py`

1. **Add anatomy config table** — list of 27 dicts for healthy, 7 dicts for hemorrhage
2. **Add tilt logic** — rotate voxel coordinates by (roll, pitch) before ellipsoid test  
   ```python
   def apply_tilt(x, y, z, cx, cy, cz, roll_deg, pitch_deg):
       """Rotate (x,y,z) relative to head centre by roll then pitch."""
       dx, dy, dz = x-cx, y-cy, z-cz
       # Roll: rotate around z-axis
       r = math.radians(roll_deg)
       dx2 = dx*math.cos(r) - dy*math.sin(r)
       dy2 = dx*math.sin(r) + dy*math.cos(r)
       dz2 = dz
       # Pitch: rotate around y-axis
       p = math.radians(pitch_deg)
       dx3 = dx2*math.cos(p) + dz2*math.sin(p)
       dy3 = dy2
       dz3 = -dx2*math.sin(p) + dz2*math.cos(p)
       return cx+dx3, cy+dy3, cz+dz3
   ```
3. **Update scenario counter** — 4-digit IDs (`{id:04d}`) to accommodate 1360 scenarios
4. **Pass anatomy params into `write_scenario`** — `scalp_mm, gray_mm, head_size, roll, pitch`
5. **Update `write_scenario`** to write the correct material parameters and tilt-rotated geometry

### `generate_metadata.py`

1. **Extend CSV schema** — add new columns: `scalp_skull_mm, gray_mm, head_size, a_mm, b_mm, c_mm, roll_deg, pitch_deg, lesion_region`
2. **Parse new header comments** in `.in` files that encode the anatomy+tilt config
3. **Stratified split** — stratify by `(has_lesion, head_size)` not just `has_lesion`

### `run_simulation.sh`

Update SLURM array line:
```bash
#SBATCH --array=1-21760%16
```

---

## Scenario ID Assignment

```
0001 – 0135  :  Healthy    (27 anatomy × 5 tilt, in order: A0T0, A0T1, ..., A26T4)
0136 – 1360  :  Hemorrhage (7 anatomy × 5 sizes × 35 locations, in order: A1S1L01, ...)
```

Anatomy configs are ordered: scalp_vals × gray_vals × size_vals  
(outer loop = scalp, middle = gray, inner = head_size)

---

## Comparison with Current Dataset

| Parameter | Current (300) | New (1,360) |
|-----------|--------------|-------------|
| Total scenarios | 300 | 1,360 |
| Healthy | 50 | 135 |
| Hemorrhage | 250 | 1,225 |
| Anatomy variation | Fixed | 27 configs |
| Head tilt | None | 5 angles |
| Lesion locations | 50 | 35 (more systematic) |
| Lesion sizes | 5 | 5 (unchanged) |
| Metadata columns | 7 | 16 |
| .in files | 4,800 | 21,760 |
| Storage (raw) | ~360 GB | ~1.6 TB |
| HPC runtime | ~19 h | ~113 h |
| Train / Val / Test split | 70/15/15 | 80/10/10 |
| Training scenarios | 210 | 1,088 |
