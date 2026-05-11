# Experiment 3: Coarse Lesion Localisation

## Objective

Implement a multi-label neural network for **coarse localisation** of brain lesions using a 4×4 grid discretization of the head cross-section. This experiment determines which regions (cells) of the head contain the simulated lesion, rather than binary classification (present/absent).

## Problem Formulation

### Input
- **Normalized magnitude-phase data**: Shape `(B, 90, 16, 16, 3)` where:
  - B = batch size (700 train, 150 val)
  - 90 = frequency points (1–90 GHz)
  - 16×16 = antenna port matrix
  - 3 = channels (magnitude, cos(phase), sin(phase))

### Output
- **4×4 multi-label grid** (16 cells)
- Each cell is a binary target: 1 if lesion overlaps with cell, 0 otherwise
- Supports partial overlap (circle-rectangle intersection geometry)

### Geometry
- Head ellipse: major axis `a=0.095 m`, minor axis `b=0.075 m`
- Each cell is a rectangular region in the normalized head coordinate space
- Lesion is a circle with center (x_lesion, y_lesion) and radius r_lesion (from metadata)
- Intersection computed using circle-rectangle collision detection

## Architecture

### Model: FlattenMLPGridClassifier
```
Input (B, 90, 16, 16, 3) → Flatten → (B, 69120)
  ↓
Dense(69120 → 256) + ReLU + Dropout(0.2)
  ↓
Dense(256 → 128) + ReLU + Dropout(0.2)
  ↓
Dense(128 → 64) + ReLU + Dropout(0.2)
  ↓
Dense(64 → 16 logits)  [one per grid cell]
  ↓
Sigmoid for prediction (threshold=0.5)
```

**Parameters**: 17,737,168

### Training Configuration
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Loss**: BCEWithLogitsLoss with per-cell positive weighting
  - `pos_weight[i] = neg_count[i] / pos_count[i]` for each cell
  - Addresses class imbalance in sparse multi-label regime
- **Regularization**:
  - Dropout (0.2 on hidden layers)
  - Gradient clipping (max_norm=1.0)
  - Weight decay (1e-4)
- **Epochs**: 20
- **Early stopping**: Tracking by validation IoU

## Dataset Statistics

| Metric | Train | Val |
|--------|-------|-----|
| Total samples | 700 | 150 |
| Lesion-positive samples | 490 (70%) | 105 (70%) |
| Active cells (≥1 lesion overlap) | 1,352 | 297 |
| Avg cells per positive sample | 2.76 | 2.83 |

## Results

### Validation Metrics

| Metric | Value |
|--------|-------|
| Best IoU (Jaccard) | 0.1272 (epoch 18) |
| Macro F1 | 0.1444 |
| Exact match accuracy | 0.0000 |
| Mean IoU | 0.1272 |

### Interpretation
- **Low IoU (~13%)**: Multi-label localization from flattened input is significantly harder than binary classification (Exp 2: AUC=0.7084)
- **Zero exact match**: No samples perfectly predict all 16 cells; model struggles with spatial structure
- **Macro F1 > IoU**: Per-cell recall is higher than intersection-over-union, suggesting conservative predictions
- **Baseline established**: These results represent a lower bound for future improvements (spatial convolutions, attention, coarser grids)

### Per-Epoch Performance
| Epoch | Train Loss | Val Loss | Train IoU | Val IoU |
|-------|-----------|----------|-----------|---------|
| 1 | 1.7240 | 1.2525 | 0.1016 | 0.0876 |
| 5 | 1.2419 | 1.2275 | 0.1132 | 0.1088 |
| 10 | 1.2284 | 1.2267 | 0.1178 | 0.1098 |
| 15 | 1.2272 | 1.2265 | 0.1179 | 0.1088 |
| 18 | (best) | (best) | - | **0.1272** |
| 20 | 1.2206 | 1.2261 | 0.1139 | 0.1080 |

## Outputs

### 1. Training Curves (`experiment_3_localisation_training_curves.png`)
- **Panel 1**: BCEWithLogitsLoss (train & val) vs. epoch
- **Panel 2**: Jaccard IoU (train & val) vs. epoch
- Highlights best validation epoch (18) with vertical line

### 2. Grid Statistics (`experiment_3_localisation_grid_statistics.png`)
- **2×2 confusion grids** (test set):
  - **TP (True Positives)**: Correctly predicted active cells
  - **FP (False Positives)**: Predicted active but ground truth empty
  - **FN (False Negatives)**: Missed lesion-containing cells
  - **TN (True Negatives)**: Correctly predicted inactive cells
- Cell-wise counts annotated; cells colored by frequency

### 3. Example Predictions (`localisation_examples.png`)
- **9 example samples** (3 rows):
  - Row 1: Correctly predicted cases (high IoU)
  - Row 2: Boundary predictions (moderate IoU, edge cases)
  - Row 3: Failure cases (low IoU, systematic errors)
- **3 columns per row**:
  - **Column 1**: Ground truth 4×4 binary grid
  - **Column 2**: Predicted probability heatmap (continuous 0–1)
  - **Column 3**: Thresholded predictions (binary, threshold=0.5)
- White gridlines overlaid for cell boundaries

### 4. Results JSON (`experiment_3_localisation_results.json`)
Complete metadata including:
- Model architecture and parameter counts
- Dataset split sizes and statistics
- Per-cell and aggregate metrics
- Per-sample IoU scores
- Confusion matrix (TP/FP/FN/TN counts)
- Training history (per-epoch losses, metrics)

### 5. Saved Model (`experiment_3_localisation_best_model.pt`)
Best-performing weights (epoch 18) saved for inference or fine-tuning

## Scripts

### `build_localisation_dataset.py`
Preprocesses the binary classification archive into multi-label 4×4 grid targets:
- Loads normalized mag_phase tensors
- Maps lesion coordinates (from metadata) to grid cell labels
- Validates binary order alignment vs. source archive
- Outputs: `dataset_mag_phase_localisation.npz`

### `train_localisation_mlp.py`
Trains the FlattenMLPGridClassifier:
- Computes per-cell positive weights from training set
- Runs 20-epoch training with early stopping tracking
- Computes per-cell and aggregate metrics
- Generates all visualizations and results JSON

### `run_experiment_3.py`
Orchestrator that chains both scripts in sequence

## How to Run

```bash
cd /Users/paul/Documents/Projects/brain-emi-simulation
python experiments/experiment_3_localisation/run_experiment_3.py
```

Or run scripts individually:
```bash
python experiments/experiment_3_localisation/scripts/build_localisation_dataset.py
python experiments/experiment_3_localisation/scripts/train_localisation_mlp.py
```

## Key Findings

1. **Multi-label localization is fundamentally harder** than binary classification for this dataset:
   - Binary (Exp 2): AUC ~0.71
   - Localization (Exp 3): IoU ~0.13
   - ~5× performance gap suggests spatial structure is lost in flattened encoding

2. **Class imbalance is severe**:
   - Average 2.76 active cells per lesion across 16 possible cells
   - Many cells have <100 active samples in training set
   - Per-cell weighting partially mitigates but cannot fully compensate

3. **Exact match is intractable** (0% accuracy):
   - Perfect prediction of all 16 cells requires learning precise spatial boundaries
   - Task may benefit from:
     - Structured prediction (CRF or structured SVM)
     - Spatial attention mechanisms
     - Coarser grid (3×3) to reduce cardinality
     - Finer input encoding (spatial convolutions vs. flattening)

## Future Work

1. **Architectural improvements**:
   - Replace flattening with 2D convolutions on port matrix
   - Add spatial attention over grid cells
   - Try structured output layers (e.g., graph neural networks)

2. **Grid granularity**:
   - Evaluate 3×3 (9 cells) and 5×5 (25 cells) grids
   - Trade-off between spatial precision and label cardinality

3. **Lesion size stratification**:
   - Analyze performance separately for small/medium/large lesions
   - Different architectures may suit different lesion sizes

4. **Data augmentation**:
   - Frequency dropout, noise injection
   - Synthetic lesion position jittering

## References

- **Grid mapping**: Elliptical head geometry (HEAD_A=0.095m, HEAD_B=0.075m) with circle-rectangle intersection
- **Metrics**: Per-cell accuracy, macro F1, Jaccard IoU (samples-averaged), exact match accuracy
- **Loss function**: BCEWithLogitsLoss(pos_weight=cell-specific)
- **Dataset**: 700 train / 150 val samples from normalized magnitude-phase tensors (90 frequencies, 16×16 port matrix, 3 channels)
