# Dataset Metadata Table

This file summarizes the columns in `dataset_metadata.csv`, what each parameter controls, and the range of values currently generated in the full 1000-scenario dataset.

## Dataset overview

- Total scenarios: 1000
- Split: train = 700, val = 150, test = 150
- No-lesion samples: 300
- Lesion samples: 700
- Lesion size range: 0.0 to 29.8732 mm
- The dataset is intentionally imbalanced toward lesion-present cases to improve representation of anomaly variability.
- Evaluation should therefore prioritize balanced accuracy and F1-score over raw accuracy alone.
- Head scale, head rotation, and noise level are treated as controlled variability factors for robustness and generalisation analysis.

## Numeric fields

| Column | Meaning | Current range / values | Notes |
|---|---|---:|---|
| `scenario_id` | Unique scenario index | 1 to 1000 | Deterministic ordering |
| `has_lesion` | Binary target label | 0 or 1 | 0 = no lesion, 1 = lesion |
| `lesion_size_mm` | Lesion diameter / size parameter | 0.0 to 29.8732 | Zero for all no-lesion cases |
| `lesion_x` | Lesion x-offset from head center (m) | -0.054363 to 0.05312 | Only non-zero for lesion cases |
| `lesion_y` | Lesion y-offset from head center (m) | -0.054352 to 0.054861 | Only non-zero for lesion cases |
| `lesion_z` | Lesion z-offset from head center (m) | -0.079218 to 0.079773 | Only non-zero for lesion cases |
| `epsilon_variation` | Background dielectric variation (%) | -9.9602 to 9.9922 | Applied to background/coupling tissues |
| `sigma_variation` | Background conductivity variation (%) | -9.9708 to 9.9902 | Applied to background/coupling tissues |
| `antenna_offset` | Antenna ring radial offset (cells) | Legacy values (deprecated) | Retained for backward compatibility; no longer used |
| `coupling_thickness` | Coupling medium thickness (m) | 0.01 to 0.03 | Used in input generation and diagrams |
| `head_scale` | Uniform anatomy scaling factor | 0.9 to 1.1 (new metadata) | Scales head structures and lesion offsets |
| `head_rotation_deg` | In-plane head rotation (deg) | -15 to 15 (new metadata) | Rotates head and lesion about center |
| `epsilon_anomaly_variation` | Lesion dielectric variation (%) | -14.9863 to 14.9455 | Lesion-only variability |
| `sigma_anomaly_variation` | Lesion conductivity variation (%) | -14.9910 to 14.8630 | Lesion-only variability |
| `background_epsilon_variation` | Background dielectric variation (%) | -9.9602 to 9.9922 | Same values as `epsilon_variation` |
| `background_sigma_variation` | Background conductivity variation (%) | -9.9708 to 9.9902 | Same values as `sigma_variation` |

## Categorical fields

| Column | Meaning | Current values / counts | Notes |
|---|---|---:|---|
| `split` | Dataset split | train=700, val=150, test=150 | Deterministic split assignment |
| `group` | Scenario family | `A_anomaly`=700, `N1_baseline`=100, `N2_property_variation`=100, `N3_measurement_variation`=100 | Baseline groups are no-lesion |
| `size_bucket` | Lesion size class | `none`=300, `small`=230, `medium`=240, `large`=230 | `none` means no lesion |
| `region` | Lesion location region | `none`=300, `left`=176, `right`=176, `deep`=174, `boundary`=174 | `none` means no lesion |
| `shape` | Lesion geometry | `none`=300, `sphere`=492, `ellipsoid`=208 | `none` means no lesion |
| `noise_level` | Noise class | `low`=550, `medium`=310, `high`=140 | Noise is part of the metadata design |

## How the fields are used

- `has_lesion`, `lesion_size_mm`, `lesion_x`, `lesion_y`, `lesion_z`, `shape`, `region`
  - control whether a scenario contains a hemorrhage and where it is placed.
  - in addition to classification (`has_lesion`), lesion position variables (`lesion_x`, `lesion_y`, `lesion_z`) enable supervised localisation tasks.

- `epsilon_variation`, `sigma_variation`, `background_epsilon_variation`, `background_sigma_variation`
  - control background tissue property perturbations.
  - `background_*` fields are explicit duplicates for clarity and may be consolidated in future revisions.

- `epsilon_anomaly_variation`, `sigma_anomaly_variation`
  - control lesion material perturbations.

- `antenna_offset`, `coupling_thickness`
  - `coupling_thickness` controls setup geometry.
  - `antenna_offset` is retained for backward compatibility but no longer used in geometry generation or analysis.

- `head_scale`, `head_rotation_deg`
  - control anatomical size and orientation variability.
  - these variables are sampled independently of lesion presence to avoid spurious correlations.

- `split`
  - controls which scenarios should be used for train, validation, and test.

- `group`, `size_bucket`, `region`, `shape`, `noise_level`
  - provide stratification and scenario-family labels for analysis.

## Practical reading guide

- The first 300 scenarios are no-lesion baseline and variation-only cases.
- The remaining 700 scenarios are lesion cases with stratified size, region, and shape.
- Train/val/test are already encoded in `split`, but for the proof-of-concept work we initially used only the generated train-heavy subsets.
- For a final benchmark, you should train on `train` and evaluate on `val` / `test` only.

## Mathematical interpretation

This section states how each field is used in the generated simulation or in downstream processing.

| Field | Mathematical role |
|---|---|
| `scenario_id` | Index only. It does not enter the physics directly; it seeds deterministic generation and identifies the sample. |
| `has_lesion` | Binary target label \(y \in \{0,1\}\). In the classifier this is the supervision signal. |
| `lesion_size_mm` | Converted to meters as \(r = \text{lesion\_size\_mm} / 1000\). For a sphere this is the radius. For an ellipsoid, the code uses \(r_a = r\), \(r_b = 0.8r\), \(r_c = 1.2r\). |
| `lesion_x`, `lesion_y`, `lesion_z` | Offsets from head center. The lesion center is \((x_c, y_c, z_c) = (x_0 + \Delta x, y_0 + \Delta y, z_0 + \Delta z)\). |
| `epsilon_variation` | Background dielectric scaling factor. If base permittivity is \(\varepsilon_0\), then \(\varepsilon = \varepsilon_0 (1 + v_\varepsilon/100)\). |
| `sigma_variation` | Background conductivity scaling factor. If base conductivity is \(\sigma_0\), then \(\sigma = \sigma_0 (1 + v_\sigma/100)\). |
| `background_epsilon_variation` | Same mathematical role as `epsilon_variation`; stored explicitly so the background perturbation is easy to read. |
| `background_sigma_variation` | Same mathematical role as `sigma_variation`; stored explicitly so the background perturbation is easy to read. |
| `epsilon_anomaly_variation` | Lesion permittivity scaling. If lesion base permittivity is \(\varepsilon_{\text{blood}}\), then \(\varepsilon_{\text{lesion}} = \varepsilon_{\text{blood}} (1 + v_a/100)\). |
| `sigma_anomaly_variation` | Lesion conductivity scaling. If lesion base conductivity is \(\sigma_{\text{blood}}\), then \(\sigma_{\text{lesion}} = \sigma_{\text{blood}} (1 + v_a/100)\). |
| `antenna_offset` | Backward-compatibility field. It is retained in metadata but ignored in geometry generation and analysis in the updated pipeline. |
| `coupling_thickness` | Thickness of the coupling medium. It expands the outer ellipsoid semi-axes by adding \(t_{\text{coupling}}\) to the head boundary in the geometry generator and diagram. |
| `head_scale` | Uniform scaling \(s\): semi-axes become \((a', b', c') = (sa, sb, sc)\). Lesion positions are sampled/applied relative to the scaled anatomy to preserve anatomical consistency. |
| `head_rotation_deg` | In-plane rotation \(\theta\): \(x' = x\cos\theta - y\sin\theta\), \(y' = x\sin\theta + y\cos\theta\). Rotation is applied to the head and lesion while antennas remain fixed. |
| `noise_level` | Active tensor-noise control after S-parameter conversion. Gaussian noise is applied to real/imag S-parameter components with std proportional to per-channel signal standard deviation, scaled by factors 0.001 (low), 0.005 (medium), 0.01 (high). |
| `split` | Dataset partition label. It determines whether the sample belongs to train, validation, or test. |
| `group` | Scenario family label. It does not change physics by itself; it indicates which generation branch produced the sample. |
| `size_bucket` | Lesion-size stratification label. It selects the size interval used when sampling lesion size. |
| `region` | Lesion-location stratification label. It selects the allowed spatial range for \((\Delta x, \Delta y, \Delta z)\). |
| `shape` | Geometry choice. `sphere` produces a sphere; `ellipsoid` changes the lesion to an axis-scaled ellipsoid. |

### In the simulation pipeline

- Tissue-property perturbations are applied multiplicatively: \(p' = p(1 + v/100)\).
- Geometry is built from ellipsoids and spheres on a 2 mm grid.
- The antenna ring is placed at a nominal fixed radius; antenna offset is no longer used as a variation source.
- Head scale and head rotation are applied to anatomical structures and lesion placement; the antenna system remains fixed as the measurement frame.
- The extracted tensors store the full S-matrix as real and imaginary channels, so the CNN sees frequency-dependent complex response rather than summary statistics.

### In the proof-of-concept CNN

- The CNN input is a tensor \(X \in \mathbb{R}^{512 \times F}\) after downsampling to \(F=64\) by default.
- Train-only normalization is applied per channel and per frequency: \(\hat{X} = (X - \mu) / \sigma\).
- The classifier outputs a logit \(z\), then a lesion probability \(p = \sigma(z) = 1 / (1 + e^{-z})\).
- The default decision threshold is \(p \ge 0.5\) for lesion prediction.
