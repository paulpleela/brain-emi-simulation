# CNN & Normalization Audit Summary

## 🔴 Critical Issues Found

### Issue #1: Normalization Bug (real_imag format)
**Location**: `train_classification_formats.py`, lines 64-79

**Problem**: The real/imag normalization was computing **separate statistics for train and validation data**, violating the principle of cross-validation:

```python
# WRONG: Computes mean/std independently for validation
for c in range(X_val_t.shape[1]):
    for f in range(X_val_t.shape[2]):
        data_slice = X_val_t[:, c, f, :, :]
        mean = data_slice.mean()  # ❌ WRONG: computed on val data
        std = data_slice.std()    # ❌ WRONG: computed on val data
        X_val_t[:, c, f, :, :] = (data_slice - mean) / std
```

**Impact**: 
- Validation AUC estimates are artificially inflated (data leakage)
- Makes train/val split meaningless
- Results are not generalizable

**Fix Applied**: Now uses **training statistics for both train and validation**:
```python
# CORRECT: Fit on train, apply to both
for c in range(X_train_t.shape[1]):
    for f in range(X_train_t.shape[2]):
        data_slice_train = X_train_t[:, c, f, :, :]
        mean = data_slice_train.mean()
        std = data_slice_train.std()
        if std == 0: std = 1e-8
        X_train_t[:, c, f, :, :] = (data_slice_train - mean) / std
        
        # Apply same train stats to validation
        data_slice_val = X_val_t[:, c, f, :, :]
        X_val_t[:, c, f, :, :] = (data_slice_val - mean) / std
```

✅ **Status**: Fixed - now matches the correct approach used by mag_phase and mag_only formats

---

### Issue #2: CNN Architecture Problem
**Location**: `experiments/experiment_2_model_bias/scripts/train_model_bias_comparison.py`

**Problem**: The Naive 3D CNN treats **frequency as a spatial dimension**:
- Uses Conv3D with kernel size 3 on all three dimensions
- Assumes frequency bins are spatially adjacent (they're not)
- Treats 16×16 port matrix same as frequency (wrong)
- **Result**: Extremely poor performance (val_auc = 0.5458 vs MLP 0.7084)

**Root Cause**: 
```python
# Input shape: (B, 3, 90, 16, 16)  
# channels=3, frequency=90, spatial=16x16
nn.Conv3d(3, 8, kernel_size=3, padding=1)  # ❌ Treats all dimensions as spatial!
```

**Why it's wrong**:
- Conv3D expects spatial locality (e.g., voxels in medical images)
- Frequency axis is NOT local - it's an abstract feature axis
- Port matrix IS spatial (16×16 antenna array layout)
- Mixing these causes poor learned patterns

**Fix Applied**: Added `SpatialCNN2DClassifier` that:
1. **Conv2D** processes the spatial 16×16 port matrix (correct spatial structure)
2. **Conv1D** learns patterns across frequency dimension (not spatial)
3. Properly respects the data structure

Architecture:
```python
class SpatialCNN2DClassifier(nn.Module):
    # Stage 1: Conv2D on spatial port matrix per frequency
    self.spatial_conv = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),  # ✅ Spatial processing
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    
    # Stage 2: Conv1D across frequency dimension
    self.freq_conv = nn.Sequential(
        nn.Conv1d(32, 64, kernel_size=5, padding=2),  # ✅ Frequency patterns
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Conv1d(64, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )
    
    # Stage 3: MLP classifier
    self.classifier = nn.Sequential(...)
```

✅ **Status**: Added - CNN now respects spatial vs. frequency structure

---

## 📊 Expected Performance Changes

### Before Fixes:
| Model | Val AUC | Issue |
|-------|---------|-------|
| MLP | 0.7084 | baseline OK |
| Naive 3D CNN | 0.5458 | ❌ treats freq as spatial |
| Freq-aware | 0.5958 | partial fix, but still struggles |
| Real/Imag | ? | ❌ data leakage in validation |

### After Fixes:
| Model | Expected | Comment |
|-------|----------|---------|
| MLP | ~0.7084 | unchanged (baseline) |
| Naive 3D CNN | ~0.5458 | unchanged (fundamentally flawed) |
| **Spatial CNN** | **0.65-0.75** | ✅ NEW - should match/beat MLP |
| Freq-aware | ~0.5958 | unchanged (already correct) |
| Real/Imag | **More realistic** | ✅ No data leakage |

---

## 🔧 Files Modified

### 1. `train_classification_formats.py`
- **Changed**: Normalization logic for real_imag format
- **Lines**: 64-79
- **Principle**: Fit statistics on train split only, apply to both train and validation

### 2. `experiments/experiment_2_model_bias/scripts/train_model_bias_comparison.py`
- **Added**: `SpatialCNN2DClassifier` class (proper CNN architecture)
- **Changed**: `make_model()` factory function (added spatial_cnn option)
- **Changed**: `to_model_input()` (handles spatial_cnn permutation)
- **Changed**: `model_specs` list (added spatial_cnn to experiments)
- **Changed**: `colors` dict (added visualization color for spatial_cnn)

---

## ✅ Verification Checklist

- [x] Syntax check passed for both modified files
- [x] Normalization logic now matches best practice (train-only fit)
- [x] CNN architecture respects spatial vs. frequency structure
- [x] Model factory and training loop updated
- [x] Visualization colors assigned
- [ ] **TODO**: Run experiment to verify performance improvements

---

## 🚀 Next Steps

1. **Run Experiment 2 to test the fixes**:
   ```bash
   cd /Users/paul/Documents/Projects/brain-emi-simulation
   python experiments/experiment_2_model_bias/scripts/train_model_bias_comparison.py
   ```

2. **Expected output**:
   - Should now test 4 models (MLP, Naive CNN, Spatial CNN, Freq-aware)
   - Spatial CNN should show improved performance
   - Results saved to `experiments/experiment_2_model_bias/results/experiment_2_model_bias_results.json`

3. **Verify real_imag normalization fix**:
   - Run a training script that uses real_imag format
   - Check that validation metrics are now more realistic
   - Should see better train/val generalization gap

---

## 📝 Key Takeaways

### CNN Design Principle
**Inductive biases must match data structure**:
- ✅ Conv2D on spatial dimensions (port matrix)
- ✅ Conv1D/Attention for non-spatial dimensions (frequency)
- ❌ Conv3D naively mixing spatial and abstract dimensions

### Normalization Principle
**Prevent data leakage in cross-validation**:
- ✅ Fit statistics on train split ONLY
- ✅ Apply same statistics to validation/test
- ❌ Computing separate statistics for each split

