# Dataset Generation Workflow

## Quick Reference

**Total Dataset**: ~50 simulations across 3 batches
- Batch 1: Healthy baselines (10 scenarios)
- Batch 2: Hemorrhage variations (32 scenarios)  
- Batch 3: Rotation study (10 scenarios)

---

## Step-by-Step Workflow

### BATCH 1: Healthy Baselines (Start Here)

**What**: 10 healthy brain simulations (no hemorrhage)

**Steps**:

1. **Generate inputs**:
```bash
# Edit generate_inputs.py line 31:
BATCH = "healthy_baseline"

python generate_inputs.py
```

2. **Check output**:
```bash
ls brain_inputs/
# Should see: scenario_001_tx01.in through scenario_010_tx16.in (160 files)
```

3. **Upload to HPC**:
```bash
scp -r brain_inputs/ run_simulation.sh s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/
```

4. **Run on HPC**:
```bash
ssh s4910027@login0.hpc.griffith.edu.au
cd ~/brain-emi-simulation
mkdir -p logs
sbatch run_simulation.sh
squeue -u $USER
```

5. **Wait for completion** (~4 hours):
```bash
watch -n 10 'squeue -u $USER'
```

6. **Download results**:
```bash
# On local machine
scp -r s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/brain_inputs/*.out ./results/batch1_healthy/
```

7. **Extract S-parameters** (create script - see below):
```bash
python extract_sparameters.py --input results/batch1_healthy/ --output results/
# Creates: scenario_001.s16p through scenario_010.s16p
```

8. **Clean up for next batch**:
```bash
# On HPC
rm brain_inputs/*.in
rm brain_inputs/*.out
```

---

### BATCH 2: Hemorrhage Variations (Main Dataset)

**What**: 32 scenarios = 4 sizes × 8 positions

**Steps**:

1. **Generate inputs**:
```bash
# Edit generate_inputs.py line 31:
BATCH = "hemorrhage_main"

python generate_inputs.py
# Creates: scenario_101_tx01.in through scenario_132_tx16.in (512 files)
```

2. **Upload and run** (same as Batch 1 steps 3-7)

3. **Extract S-parameters**:
```bash
python extract_sparameters.py --input results/batch2_hemorrhage/ --output results/
# Creates: scenario_101.s16p through scenario_132.s16p
```

---

### BATCH 3: Rotation Study

**What**: 10 scenarios testing rotation sensitivity

**Steps**:

1. **Generate inputs**:
```bash
# Edit generate_inputs.py line 31:
BATCH = "rotation_study"

python generate_inputs.py
# Creates: scenario_201_tx01.in through scenario_210_tx16.in (160 files)
```

2. **Upload and run** (same as previous batches)

3. **Extract S-parameters**:
```bash
python extract_sparameters.py --input results/batch3_rotation/ --output results/
# Creates: scenario_201.s16p through scenario_210.s16p
```

---

## Final Dataset Structure

After all batches complete:

```
results/
├── scenario_001.s16p  # Healthy baseline 1
├── scenario_002.s16p  # Healthy baseline 2
├── ...
├── scenario_010.s16p  # Healthy baseline 10
├── scenario_101.s16p  # Hemorrhage: 5mm, left frontal
├── scenario_102.s16p  # Hemorrhage: 5mm, right frontal
├── ...
├── scenario_132.s16p  # Hemorrhage: 20mm, right-posterior deep
├── scenario_201.s16p  # Rotation: 0°, with lesion
├── scenario_202.s16p  # Rotation: 5°, with lesion
├── ...
└── scenario_210.s16p  # Rotation: -10°, without lesion
```

---

## ML Training Organization

### Create metadata file:

```python
# metadata.csv
scenario_id,has_lesion,lesion_size_mm,lesion_x,lesion_y,lesion_z,rotation_deg,dataset
001,0,0,0,0,0,0,train
002,0,0,0,0,0,0,train
...
010,0,0,0,0,0,0,test
101,1,5,-0.03,0.00,0.01,0,train
102,1,5,0.03,0.00,0.01,0,train
...
132,1,20,0.02,0.02,0.00,0,test
201,1,15,-0.02,0.00,0.01,0,rotation
...
210,0,0,0,0,0,-10,rotation
```

### Python training code:

```python
import pandas as pd
import numpy as np

# Load metadata
meta = pd.read_csv('metadata.csv')

# Load S-parameters
def load_s16p(filename):
    # Read Touchstone file
    # Returns: frequencies, S_matrix (shape: [n_freq, 16, 16])
    pass

# Training set: scenarios 001-008, 101-128
train_meta = meta[meta['dataset'] == 'train']
X_train = [load_s16p(f'scenario_{id:03d}.s16p') for id in train_meta['scenario_id']]
y_train = train_meta[['has_lesion', 'lesion_x', 'lesion_y', 'lesion_z']].values

# Test set: scenarios 009-010, 129-132
test_meta = meta[meta['dataset'] == 'test']
X_test = [load_s16p(f'scenario_{id:03d}.s16p') for id in test_meta['scenario_id']]
y_test = test_meta[['has_lesion', 'lesion_x', 'lesion_y', 'lesion_z']].values

# Train model
model.fit(X_train, y_train)
```

---

## Scenario ID Reference

### Healthy Baselines (001-010)
```
001-010: Healthy brain (no hemorrhage), 0° rotation
```

### Hemorrhage Main Dataset (101-132)
```
4 sizes: 5mm, 10mm, 15mm, 20mm
8 positions:
  - Left frontal, Right frontal
  - Left occipital, Right occipital  
  - Anterior, Posterior
  - Left-anterior deep, Right-posterior deep

Numbering:
101-104: 5mm × 4 positions (frontal L/R, occipital L/R)
105-108: 5mm × 4 positions (anterior, posterior, deep L/R)
109-112: 10mm × 4 positions (frontal L/R, occipital L/R)
...
129-132: 20mm × 4 positions (deep positions)
```

### Rotation Study (201-210)
```
201: 15mm lesion, 0° rotation, with lesion
202: 15mm lesion, 5° rotation, with lesion
203: 15mm lesion, -5° rotation, with lesion
204: 15mm lesion, 10° rotation, with lesion
205: 15mm lesion, -10° rotation, with lesion
206: Healthy, 0° rotation
207: Healthy, 5° rotation
208: Healthy, -5° rotation
209: Healthy, 10° rotation
210: Healthy, -10° rotation
```

---

## Timeline Estimate

- Batch 1 (10 scenarios × 16 files): ~4 hours HPC time
- Batch 2 (32 scenarios × 16 files): ~13 hours HPC time
- Batch 3 (10 scenarios × 16 files): ~4 hours HPC time

**Total HPC time**: ~21 hours (can run batches on different days)

---

## Troubleshooting

**Q: How do I know which batch is currently generated?**
```bash
ls brain_inputs/scenario_*.in | head -5
# If you see scenario_001_* → healthy_baseline
# If you see scenario_101_* → hemorrhage_main
# If you see scenario_201_* → rotation_study
```

**Q: Can I change batch parameters?**

Yes! Edit `BATCH_CONFIGS` in `generate_inputs.py`:
- Add more positions to `lesion_positions`
- Add more sizes to `lesion_sizes`
- Add more rotations to `rotation_angles`

**Q: How do I resume if HPC job fails?**

Check which scenarios completed:
```bash
ls brain_inputs/*.out | grep scenario_
```

Modify `run_simulation.sh` to skip completed scenarios or regenerate only failed ones.
