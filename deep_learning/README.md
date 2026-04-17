# Deep Learning Proof of Concept

This folder contains isolated training utilities so the root workspace stays clean.

## What was checked

For the currently copied 100 samples (`scenario_001` to `scenario_100`):
- `sparams/`: 100 files found
- `fd_tensors/`: 100 files found
- IDs align between both folders
- But metadata shows these 100 are all:
  - split = `train`
  - `has_lesion = 0`

This means the data is structurally fine, but **not suitable for supervised lesion classification** yet.

## Included scripts

- `check_subset.py`
  - Reports ID coverage, split distribution, and label distribution for the copied subset.

- `train_poc.py`
  - `autoencoder` mode (default): trains an unsupervised proof-of-concept model on the 100 normal samples.
    - Uses PyTorch if available.
    - Falls back to a NumPy autoencoder if PyTorch is not installed.
  - `classifier` mode: binary lesion classifier (requires both classes).
    - Uses PyTorch if available.
    - Falls back to a NumPy logistic classifier if PyTorch is not installed.

- `predict.py`
  - Loads a trained classifier checkpoint and prints readable per-scenario output.
  - Includes lesion probability, predicted class, and (if metadata provided) true label and correctness.
  - Prints summary metrics with confusion counts.

- `train_cnn_poc.py`
  - Trains a lightweight frequency-aware 1D CNN on raw tensors.
  - Uses signal shape `(512, F)` directly, downsampled to `50..100` points (default `64`).
  - Applies train-only per-channel, per-frequency normalization.
  - Saves best checkpoint by validation balanced accuracy.

- `predict_cnn.py`
  - Loads `classifier_cnn_poc.pt` and prints readable per-scenario output + summary metrics.

## Setup

Use your existing conda env if available. Otherwise:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r deep_learning/requirements.txt
```

Install PyTorch (recommended):

```bash
conda install -n brain-emi-simulation pytorch torchvision torchaudio -c pytorch
```

## Run checks

```bash
python deep_learning/check_subset.py \
  --fd-dir fd_tensors \
  --metadata dataset_metadata.csv
```

## Train proof-of-concept (works with current 100)

```bash
python deep_learning/train_poc.py \
  --mode autoencoder \
  --fd-dir fd_tensors \
  --metadata dataset_metadata.csv \
  --require-torch \
  --max-samples 100 \
  --epochs 30 \
  --batch-size 16
```

Outputs:
- `deep_learning/outputs/autoencoder_poc.pt`
  - or `deep_learning/outputs/autoencoder_poc_numpy.npz` when using NumPy fallback

## Supervised classifier (after stratified copy)

When you copy a stratified 100-sample subset with both lesion/non-lesion and val/test coverage:

```bash
python deep_learning/train_poc.py \
  --mode classifier \
  --fd-dir fd_tensors \
  --metadata dataset_metadata.csv \
  --require-torch \
  --max-samples 100 \
  --epochs 40 \
  --batch-size 16
```

Outputs:
- `deep_learning/outputs/classifier_poc.pt`
  - or `deep_learning/outputs/classifier_poc_numpy.npz` when using NumPy fallback

## Readable prediction output

```bash
python deep_learning/predict.py \
  --checkpoint deep_learning/outputs/classifier_poc.pt \
  --fd-dir fd_tensors \
  --metadata dataset_metadata.csv \
  --max-samples 150 \
  --show 40
```

Show only errors/mismatches:

```bash
python deep_learning/predict.py \
  --checkpoint deep_learning/outputs/classifier_poc.pt \
  --fd-dir fd_tensors \
  --metadata dataset_metadata.csv \
  --max-samples 150 \
  --only-errors \
  --show 200
```

## Frequency-aware 1D CNN (new)

Train a small CNN on `(512, 64)` frequency-aware inputs:

```bash
python deep_learning/train_cnn_poc.py \
  --fd-dir fd_tensors \
  --metadata dataset_metadata.csv \
  --max-samples 200 \
  --freq-points 64 \
  --epochs 40 \
  --batch-size 16 \
  --lr 1e-3 \
  --out-dir deep_learning/outputs
```

Readable report from CNN checkpoint:

```bash
python deep_learning/predict_cnn.py \
  --checkpoint deep_learning/outputs/classifier_cnn_poc.pt \
  --fd-dir fd_tensors \
  --metadata dataset_metadata.csv \
  --max-samples 200 \
  --show 50
```

## Notes

- Input tensors are expected in current project format: `signal` key with shape `(512, F)`.
- Features are reduced to channel-wise stats (mean/std over frequency) for a lightweight POC.
- This is intentionally compact and fast to verify end-to-end ML workflow first.
