# Experiment 2: Model Inductive Bias Study

**Status:** Draft | **Focus:** How should structured S-parameter data be modelled?

## Question

After Experiment 1 fixed the representation to `mag_phase`, this experiment asks:

**What model inductive biases are appropriate for S-parameter data?**

We are not trying to find the absolute best architecture. The goal is to compare the assumptions each model makes about the data.

## Models

1. **MLP baseline**
   - Minimal structure assumptions
   - Treats all features globally

2. **Naive CNN**
   - Assumes frequency and port dimensions are spatially local
   - Useful as a contrast to show why that bias may be wrong

3. **Port-matrix 2D CNN**
   - Keeps the 16x16 port matrix spatial
   - Folds frequency and representation channels into the channel axis
   - Tests a fairer CNN prior for S-parameter tensors

## Data

This experiment uses the raw `mag_phase` representation from:

`datasets/dataset_mag_phase.npz`

The archive provides `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, and `y_test`.
Normalization is fit on the train split only and applied to val/test inside the training script.

## Run

```bash
cd experiments/experiment_2_model_bias
python run_experiment_2.py
```

## Outputs

- `results/experiment_2_model_bias_results.png`
- `results/experiment_2_model_bias_results.json`

## Interpretation

The frequency-aware model is the most interesting comparison because it encodes a meaningful inductive bias for EM data:

- frequencies are sequential measurements
- ports are structured physical channels
- not every axis should be treated as generic image space
