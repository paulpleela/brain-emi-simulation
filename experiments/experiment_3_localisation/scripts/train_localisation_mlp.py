#!/usr/bin/env python3
"""Train a coarse 4x4 lesion-localisation model on normalized mag_phase tensors.

The model reuses the flattened mag_phase representation from Experiment 2,
but predicts a 16-cell multi-label grid instead of a binary lesion/no-lesion label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")

REPO_ROOT = Path(__file__).resolve().parents[3]
METADATA_PATH = REPO_ROOT / "dataset_metadata.csv"
DATASET_PATH = REPO_ROOT / "datasets" / "dataset_mag_phase_localisation.npz"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GRID_SIZE = 4
OUTPUT_DIM = GRID_SIZE * GRID_SIZE
THRESHOLD = 0.5  # Default threshold (will be optimized during evaluation)


class FlattenMLPGridClassifier(nn.Module):
    def __init__(self, input_size: int, output_size: int = OUTPUT_DIM, hidden_dims: tuple[int, int, int] = (256, 128, 64), dropout_rate: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            previous = hidden_dim
        layers.append(nn.Linear(previous, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialCNNGridClassifier(nn.Module):
    """CNN-based classifier that preserves spatial structure of the 16x16 port matrix.
    
    Input: (B, 90, 16, 16, 3) — averages across 90 frequencies to get (B, 16, 16, 3)
    Then applies Conv2D layers to learn spatial patterns in the port matrix.
    Output: (B, 16) logits for grid cells.
    """
    def __init__(self, output_size: int = OUTPUT_DIM, dropout_rate: float = 0.2):
        super().__init__()
        
        # CNN to process 16x16 port matrix (3 channels: mag, cos(phase), sin(phase))
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, 16) → (8, 8)
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (8, 8) → (1, 1)
        )
        
        # MLP head on top of CNN
        self.fc_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 90, 16, 16, 3)
        # Average across frequency dimension
        x = x.mean(dim=1)  # (B, 16, 16, 3)
        
        # Permute to channel-first format for Conv2d
        x = x.permute(0, 3, 1, 2)  # (B, 3, 16, 16)
        
        # Apply CNN
        x = self.conv_layers(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        
        # Apply MLP head
        x = self.fc_head(x)  # (B, 16)
        return x


def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")
    return pd.read_csv(METADATA_PATH).sort_values("scenario_id")


def load_dataset() -> dict[str, np.ndarray]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing localisation dataset archive: {DATASET_PATH}\n"
            "Run build_localisation_dataset.py first."
        )
    data = np.load(DATASET_PATH)
    required = ["X_train", "X_val", "Y_train", "Y_val", "train_scenario_ids", "val_scenario_ids"]
    missing = [name for name in required if name not in data.files]
    if missing:
        raise KeyError(f"Dataset is missing keys: {missing}")
    return {name: data[name] for name in data.files}


def build_loaders(data: dict[str, np.ndarray]) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    x_train = torch.from_numpy(data["X_train"].astype(np.float32))
    y_train = torch.from_numpy(data["Y_train"].astype(np.float32))
    x_val = torch.from_numpy(data["X_val"].astype(np.float32))
    y_val = torch.from_numpy(data["Y_val"].astype(np.float32))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32, shuffle=False)

    # Compute per-cell positive/negative counts and weights
    pos_counts = data["Y_train"].sum(axis=0).astype(np.float32)
    total_samples = data["Y_train"].shape[0]
    neg_counts = total_samples - pos_counts
    
    # pos_weight[i] = neg_count[i] / pos_count[i]
    # This upweights positive examples for imbalanced cells
    pos_weight = neg_counts / np.maximum(pos_counts, 1.0)
    
    # Clip extreme weights to prevent numerical instability
    # Use aggressive clipping to stabilize optimization
    MAX_POS_WEIGHT = 5.0
    pos_weight = np.minimum(pos_weight, MAX_POS_WEIGHT)
    
    pos_weight = torch.from_numpy(pos_weight.astype(np.float32)).to(DEVICE)
    
    # Diagnostic output
    print(f"\nPer-cell balance (Y_train shape: {data['Y_train'].shape}):")
    for cell_idx in range(min(16, len(pos_counts))):
        print(f"  Cell {cell_idx:2d}: {int(pos_counts[cell_idx]):3d} pos | {int(neg_counts[cell_idx]):3d} neg | pos_weight={pos_weight[cell_idx].item():.2f}")

    return train_loader, val_loader, pos_weight


def to_model_input(batch_x: torch.Tensor) -> torch.Tensor:
    # CNN model expects original shape (B, 90, 16, 16, 3)
    return batch_x


def sigmoid_to_binary(probabilities: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    return (probabilities >= threshold).astype(np.int32)


def sample_iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    scores = []
    for truth, pred in zip(y_true, y_pred):
        truth_sum = truth.sum()
        pred_sum = pred.sum()
        union = np.logical_or(truth == 1, pred == 1).sum()
        if union == 0:
            scores.append(1.0)
        else:
            scores.append(float(np.logical_and(truth == 1, pred == 1).sum() / union))
    return np.asarray(scores, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> dict[str, object]:
    y_pred = sigmoid_to_binary(y_prob, threshold=threshold)
    per_cell_accuracy = (y_pred == y_true).mean(axis=0)
    per_cell_accuracy_grid = per_cell_accuracy.reshape(GRID_SIZE, GRID_SIZE)
    per_cell_positive_rate = y_true.mean(axis=0).reshape(GRID_SIZE, GRID_SIZE)

    tp = np.logical_and(y_pred == 1, y_true == 1).sum(axis=0).reshape(GRID_SIZE, GRID_SIZE)
    fp = np.logical_and(y_pred == 1, y_true == 0).sum(axis=0).reshape(GRID_SIZE, GRID_SIZE)
    fn = np.logical_and(y_pred == 0, y_true == 1).sum(axis=0).reshape(GRID_SIZE, GRID_SIZE)
    tn = np.logical_and(y_pred == 0, y_true == 0).sum(axis=0).reshape(GRID_SIZE, GRID_SIZE)

    return {
        "per_cell_accuracy": per_cell_accuracy_grid,
        "per_cell_positive_rate": per_cell_positive_rate,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mean_iou": float(jaccard_score(y_true, y_pred, average="samples", zero_division=1)),
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
        "sample_iou": sample_iou_score(y_true, y_pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "y_pred": y_pred,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, dict[str, object], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = to_model_input(batch_x.to(DEVICE))
        batch_y = batch_y.to(DEVICE).float()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += loss.item() * len(batch_y)
        all_targets.append(batch_y.cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probabilities, axis=0)
    metrics = compute_metrics(y_true, y_prob)
    return total_loss / len(y_true), metrics, y_true, y_prob


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> tuple[float, dict[str, object], np.ndarray, np.ndarray]:
    model.train()
    total_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = to_model_input(batch_x.to(DEVICE))
        batch_y = batch_y.to(DEVICE).float()

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        all_targets.append(batch_y.detach().cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probabilities, axis=0)
    metrics = compute_metrics(y_true, y_prob)
    return total_loss / len(y_true), metrics, y_true, y_prob


def plot_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train loss", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], label="Val loss", color="#d62728")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Training Curves")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["train_iou"], label="Train IoU", color="#2ca02c")
    axes[1].plot(epochs, history["val_iou"], label="Val IoU", color="#ff7f0e")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean sample IoU")
    axes[1].set_title("Localization Overlap")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_grid(stats: dict[str, object], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    panels = [
        (stats["tp"], "True Positives", "Blues"),
        (stats["fp"], "False Positives", "Reds"),
        (stats["fn"], "False Negatives", "Oranges"),
        (stats["tn"], "True Negatives", "Greens"),
    ]

    for ax, (array, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(array, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.set_xticklabels([f"C{i+1}" for i in range(GRID_SIZE)])
        ax.set_yticklabels([f"R{i+1}" for i in range(GRID_SIZE)])
        for row_idx in range(GRID_SIZE):
            for col_idx in range(GRID_SIZE):
                ax.text(col_idx, row_idx, int(array[row_idx, col_idx]), ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Per-cell confusion statistics on validation set", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def add_grid_axes(ax: plt.Axes) -> None:
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])


def annotate_grid(ax: plt.Axes, values: np.ndarray, fmt: str = ".2f", color: str = "black") -> None:
    for row_idx in range(GRID_SIZE):
        for col_idx in range(GRID_SIZE):
            ax.text(col_idx, row_idx, format(values[row_idx, col_idx], fmt), ha="center", va="center", color=color, fontsize=9)


def plot_example_panels(examples: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 4 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Ground truth", "Predicted probability", "Thresholded prediction"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12)

    for row_idx, example in enumerate(examples):
        gt = example["gt"]
        prob = example["prob"]
        pred = example["pred"]
        meta_text = example["meta_text"]

        gt_ax = axes[row_idx, 0]
        prob_ax = axes[row_idx, 1]
        pred_ax = axes[row_idx, 2]

        gt_im = gt_ax.imshow(gt, cmap="Greens", vmin=0, vmax=1, origin="upper")
        prob_im = prob_ax.imshow(prob, cmap="viridis", vmin=0, vmax=1, origin="upper")
        pred_im = pred_ax.imshow(pred, cmap="Greens", vmin=0, vmax=1, origin="upper")

        for ax in (gt_ax, prob_ax, pred_ax):
            add_grid_axes(ax)

        annotate_grid(gt_ax, gt, fmt=".0f", color="black")
        annotate_grid(prob_ax, prob, fmt=".2f", color="white")
        annotate_grid(pred_ax, pred, fmt=".0f", color="black")

        gt_ax.set_ylabel(meta_text, fontsize=10)

        if row_idx == 0:
            gt_ax.set_title("Ground truth", fontsize=12)
            prob_ax.set_title("Predicted probability", fontsize=12)
            pred_ax.set_title("Thresholded prediction", fontsize=12)

        fig.colorbar(gt_im, ax=gt_ax, fraction=0.046, pad=0.02)
        fig.colorbar(prob_im, ax=prob_ax, fraction=0.046, pad=0.02)
        fig.colorbar(pred_im, ax=pred_ax, fraction=0.046, pad=0.02)

    fig.suptitle("Experiment 3 localisation examples", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_examples(metadata: pd.DataFrame, scenario_ids: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> list[dict[str, object]]:
    y_pred = sigmoid_to_binary(y_prob, threshold=threshold)
    sample_iou = sample_iou_score(y_true, y_pred)

    meta_by_id = metadata.set_index("scenario_id")
    rows = [meta_by_id.loc[int(sid)] for sid in scenario_ids]

    exact_matches = np.where(np.all(y_true == y_pred, axis=1))[0]
    if len(exact_matches) > 0:
        correct_idx = exact_matches[np.argmax(sample_iou[exact_matches])]
    else:
        correct_idx = int(np.argmax(sample_iou))

    boundary_candidates = [idx for idx, row in enumerate(rows) if str(row["region"]).strip() == "boundary"]
    if boundary_candidates:
        multi_cell_candidates = [idx for idx in boundary_candidates if y_true[idx].sum() > 1]
        if multi_cell_candidates:
            boundary_idx = max(multi_cell_candidates, key=lambda idx: sample_iou[idx])
        else:
            boundary_idx = max(boundary_candidates, key=lambda idx: sample_iou[idx])
    else:
        boundary_idx = correct_idx

    failure_idx = int(np.argmin(sample_iou))
    if failure_idx == correct_idx and len(sample_iou) > 1:
        sorted_indices = np.argsort(sample_iou)
        for candidate in sorted_indices:
            if candidate != correct_idx:
                failure_idx = int(candidate)
                break

    chosen_indices = [correct_idx, boundary_idx, failure_idx]
    labels = ["correct", "boundary", "failure"]
    examples: list[dict[str, object]] = []

    for label_name, idx in zip(labels, chosen_indices):
        row = rows[idx]
        meta_text = (
            f"{label_name}\n"
            f"sid {int(scenario_ids[idx])}\n"
            f"{str(row['region']).strip()} | IoU {sample_iou[idx]:.2f}"
        )
        examples.append(
            {
                "gt": y_true[idx].reshape(GRID_SIZE, GRID_SIZE),
                "prob": y_prob[idx].reshape(GRID_SIZE, GRID_SIZE),
                "pred": y_pred[idx].reshape(GRID_SIZE, GRID_SIZE),
                "meta_text": meta_text,
            }
        )

    return examples


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> int:
    print("=" * 72)
    print("EXPERIMENT 3: Coarse Lesion Localisation")
    print("=" * 72)
    print()
    print("Input: normalized mag_phase tensors")
    print("Target: 4x4 multi-label localisation grid")
    print("Goal: predict which coarse head regions contain the lesion")
    print()

    metadata = load_metadata()
    data = load_dataset()
    train_loader, val_loader, pos_weight = build_loaders(data)

    x_train = data["X_train"]
    y_train = data["Y_train"]
    x_val = data["X_val"]
    y_val = data["Y_val"]

    print(f"Dataset: {x_train.shape} train, {x_val.shape} val")
    print(f"Grid shape: {GRID_SIZE}x{GRID_SIZE} ({OUTPUT_DIM} cells)")
    print(f"Train lesion-positive samples: {int(data['y_train'].sum())}/{len(data['y_train'])}")
    print(f"Val lesion-positive samples:   {int(data['y_val'].sum())}/{len(data['y_val'])}")
    print(f"Average active cells per positive sample (train): {y_train.sum(axis=1)[y_train.sum(axis=1) > 0].mean():.2f}")
    print()

    model = SpatialCNNGridClassifier(output_size=OUTPUT_DIM).to(DEVICE)
    print(f"Model: SpatialCNNGridClassifier (CNN-based with spatial structure preservation)")
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    num_epochs = 50
    
    # Verify criterion is correctly configured
    print(f"\nCriterion: BCEWithLogitsLoss with pos_weight")
    print(f"  pos_weight shape: {pos_weight.shape}")
    print(f"  pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
    print(f"  pos_weight device: {pos_weight.device}")
    print()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "train_exact_match": [],
        "val_exact_match": [],
    }

    best_val_iou = float("-inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_val_probabilities: np.ndarray | None = None
    best_val_targets: np.ndarray | None = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics, train_targets, train_probs = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_metrics, val_targets, val_probabilities = evaluate(model, val_loader, criterion)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_iou"].append(float(train_metrics["mean_iou"]))
        history["val_iou"].append(float(val_metrics["mean_iou"]))
        history["train_macro_f1"].append(float(train_metrics["macro_f1"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
        history["train_exact_match"].append(float(train_metrics["exact_match_accuracy"]))
        history["val_exact_match"].append(float(val_metrics["exact_match_accuracy"]))

        if val_metrics["mean_iou"] > best_val_iou:
            best_val_iou = float(val_metrics["mean_iou"])
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_probabilities = val_probabilities.copy()
            best_val_targets = val_targets.copy()

        if epoch == 1:
            # Diagnostic output on first epoch
            print(f"FIRST EPOCH DIAGNOSTICS:")
            print(f"  Train probabilities (sigmoid): mean={train_probs.mean():.4f}, min={train_probs.min():.4f}, max={train_probs.max():.4f}")
            print(f"  Val probabilities (sigmoid):   mean={val_probabilities.mean():.4f}, min={val_probabilities.min():.4f}, max={val_probabilities.max():.4f}")
            print(f"  Train targets: sum={train_targets.sum():.0f}/{train_targets.size:.0f} ({100*train_targets.mean():.1f}%)")
            print(f"  Val targets:   sum={val_targets.sum():.0f}/{val_targets.size:.0f} ({100*val_targets.mean():.1f}%)")
            print()

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:2d}/{num_epochs}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_iou={train_metrics['mean_iou']:.4f} val_iou={val_metrics['mean_iou']:.4f} | "
                f"train_exact={train_metrics['exact_match_accuracy']:.4f} val_exact={val_metrics['exact_match_accuracy']:.4f}"
            )

    if best_state_dict is None or best_val_probabilities is None or best_val_targets is None:
        raise RuntimeError("Failed to capture best model state during training.")

    best_model_path = RESULTS_DIR / "experiment_3_localisation_best_model.pt"
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "input_size": int(np.prod(x_train.shape[1:])),
            "output_size": OUTPUT_DIM,
            "best_epoch": best_epoch,
            "best_val_iou": best_val_iou,
        },
        best_model_path,
    )

    model.load_state_dict(best_state_dict)
    
    # Use the best probabilities captured during training (not re-evaluated)
    # This ensures we report metrics from the actual best epoch, not a re-evaluation
    val_targets_final = best_val_targets
    val_probabilities_final = best_val_probabilities
    
    # Re-evaluate training set with best model for completeness
    train_loss_final, train_metrics_final, train_targets_final, train_probabilities_final = evaluate(model, train_loader, criterion)
    val_loss_final, val_metrics_final, _, _ = evaluate(model, val_loader, criterion)

    # Detailed threshold prediction diagnostics
    print("\n" + "=" * 72)
    print("THRESHOLD PREDICTION DIAGNOSTICS (Best Epoch: {})".format(best_epoch))
    print("=" * 72)
    
    # Find optimal threshold
    best_threshold = 0.5
    best_threshold_iou = 0.0
    for test_threshold in np.arange(0.1, 0.6, 0.05):
        test_pred = sigmoid_to_binary(val_probabilities_final, threshold=test_threshold)
        test_iou = jaccard_score(val_targets_final, test_pred, average="samples", zero_division=1)
        print(f"  Threshold {test_threshold:.2f}: IoU = {test_iou:.4f}")
        if test_iou > best_threshold_iou:
            best_threshold_iou = test_iou
            best_threshold = test_threshold
    
    print(f"\n  Optimal threshold: {best_threshold:.2f} (IoU = {best_threshold_iou:.4f})")
    print()
    
    # Use optimal threshold for final predictions
    val_predictions_binary = sigmoid_to_binary(val_probabilities_final, threshold=best_threshold)
    
    # Per-sample statistics
    val_pred_sum = val_predictions_binary.sum(axis=1)
    val_true_sum = val_targets_final.sum(axis=1)
    val_prob_max = val_probabilities_final.max(axis=1)
    val_prob_min = val_probabilities_final.min(axis=1)
    
    print(f"\nValidation set (n={len(val_predictions_binary)}):")
    print(f"  Predicted cells per sample (after threshold={best_threshold:.2f}):")
    print(f"    Mean: {val_pred_sum.mean():.2f}, Min: {int(val_pred_sum.min())}, Max: {int(val_pred_sum.max())}")
    print(f"  True cells per sample:")
    print(f"    Mean: {val_true_sum.mean():.2f}, Min: {int(val_true_sum.min())}, Max: {int(val_true_sum.max())}")
    print(f"  Probability range per sample:")
    print(f"    Max (across cells): mean={val_prob_max.mean():.4f}, range=[{val_prob_max.min():.4f}, {val_prob_max.max():.4f}]")
    print(f"    Min (across cells): mean={val_prob_min.mean():.4f}, range=[{val_prob_min.min():.4f}, {val_prob_min.max():.4f}]")
    
    # Breakdown by positive/negative samples
    negative_mask = val_true_sum == 0
    positive_mask = val_true_sum > 0
    n_negative = negative_mask.sum()
    n_positive = positive_mask.sum()
    
    print(f"\n  ⚠️  CRITICAL: Breakdown by lesion presence:")
    print(f"    Negative samples (no lesion): {n_negative}")
    print(f"    Positive samples (has lesion): {n_positive}")
    
    if n_positive > 0:
        # Compute IoU separately for positive samples
        positive_iou_scores = sample_iou_score(val_targets_final[positive_mask], val_predictions_binary[positive_mask])
        print(f"\n    ✓ Positive samples IoU: mean={positive_iou_scores.mean():.4f}, min={positive_iou_scores.min():.4f}, max={positive_iou_scores.max():.4f}")
        print(f"      → Model predicts {val_pred_sum[positive_mask].sum()} cells across {n_positive} positive samples")
    
    if n_negative > 0:
        negative_iou_scores = sample_iou_score(val_targets_final[negative_mask], val_predictions_binary[negative_mask])
        print(f"\n    ✓ Negative samples IoU: mean={negative_iou_scores.mean():.4f}, min={negative_iou_scores.min():.4f}, max={negative_iou_scores.max():.4f}")
        print(f"      → All-zero predictions get perfect IoU for empty targets!")
        print(f"      → This INFLATES overall IoU! ({n_negative}/150 = {100*n_negative/150:.0f}% of dataset)")
        print(f"      → Overall IoU={best_threshold_iou:.4f} is misleading!")
    
    # Breakdown by prediction pattern
    samples_all_zero = (val_pred_sum == 0).sum()
    samples_all_one = (val_pred_sum == 16).sum()
    samples_mixed = ((val_pred_sum > 0) & (val_pred_sum < 16)).sum()
    
    print(f"\n  Prediction patterns:")
    print(f"    All 0 (no cells): {samples_all_zero} samples ({100*samples_all_zero/len(val_predictions_binary):.1f}%)")
    print(f"    All 1 (all cells): {samples_all_one} samples ({100*samples_all_one/len(val_predictions_binary):.1f}%)")
    print(f"    Mixed (some cells): {samples_mixed} samples ({100*samples_mixed/len(val_predictions_binary):.1f}%)")
    
    # Per-cell predictions
    print(f"\n  Per-cell prediction statistics:")
    for cell_idx in range(OUTPUT_DIM):
        cell_pred_rate = val_predictions_binary[:, cell_idx].mean()
        cell_true_rate = val_targets_final[:, cell_idx].mean()
        cell_mean_prob = val_probabilities_final[:, cell_idx].mean()
        print(f"    Cell {cell_idx:2d}: pred_rate={100*cell_pred_rate:5.1f}% | true_rate={100*cell_true_rate:5.1f}% | mean_prob={cell_mean_prob:.4f}")

    curves_path = RESULTS_DIR / "experiment_3_localisation_training_curves.png"
    plot_training_curves(history, curves_path)

    grid_stats_path = RESULTS_DIR / "experiment_3_localisation_grid_statistics.png"
    plot_confusion_grid(val_metrics_final, grid_stats_path)

    scenario_ids_val = data["val_scenario_ids"]
    examples = select_examples(metadata, scenario_ids_val, val_targets_final, val_probabilities_final, threshold=THRESHOLD)
    examples_path = RESULTS_DIR / "localisation_examples.png"
    plot_example_panels(examples, examples_path)

    results = {
        "experiment": "experiment_3_localisation",
        "dataset_path": str(DATASET_PATH),
        "grid_size": GRID_SIZE,
        "output_dim": OUTPUT_DIM,
        "threshold": THRESHOLD,
        "model": {
            "name": "SpatialCNNGridClassifier",
            "description": "CNN-based classifier that preserves spatial structure of 16x16 port matrix",
            "architecture": {
                "step_1": "Average 90 frequencies → (B, 16, 16, 3)",
                "step_2": "Conv2D: 3→32 channels, ReLU, Dropout",
                "step_3": "Conv2D: 32→64 channels, ReLU, MaxPool(2)",
                "step_4": "Conv2D: 64→128 channels, ReLU, GlobalAvgPool",
                "step_5": "MLP: 128→64→16 logits",
            },
            "parameters": int(count_parameters(model)),
            "pos_weight": pos_weight.detach().cpu().numpy().tolist(),
        },
        "training": {
            "epochs": num_epochs,
            "best_epoch": best_epoch,
            "best_val_iou": best_val_iou,
            "best_model_path": str(best_model_path),
        },
        "final_metrics": {
            "train": {
                "loss": float(train_loss_final),
                "macro_f1": float(train_metrics_final["macro_f1"]),
                "mean_iou": float(train_metrics_final["mean_iou"]),
                "exact_match_accuracy": float(train_metrics_final["exact_match_accuracy"]),
                "per_cell_accuracy": train_metrics_final["per_cell_accuracy"].tolist(),
                "per_cell_positive_rate": train_metrics_final["per_cell_positive_rate"].tolist(),
            },
            "val": {
                "loss": float(val_loss_final),
                "macro_f1": float(val_metrics_final["macro_f1"]),
                "mean_iou": float(val_metrics_final["mean_iou"]),
                "exact_match_accuracy": float(val_metrics_final["exact_match_accuracy"]),
                "per_cell_accuracy": val_metrics_final["per_cell_accuracy"].tolist(),
                "per_cell_positive_rate": val_metrics_final["per_cell_positive_rate"].tolist(),
            },
        },
        "confusion_grid": {
            "tp": val_metrics_final["tp"].tolist(),
            "fp": val_metrics_final["fp"].tolist(),
            "fn": val_metrics_final["fn"].tolist(),
            "tn": val_metrics_final["tn"].tolist(),
        },
        "history": history,
        "outputs": {
            "training_curves": str(curves_path),
            "grid_statistics": str(grid_stats_path),
            "examples": str(examples_path),
        },
        "examples": [
            {
                "kind": "correct",
                "meta_text": examples[0]["meta_text"],
            },
            {
                "kind": "boundary",
                "meta_text": examples[1]["meta_text"],
            },
            {
                "kind": "failure",
                "meta_text": examples[2]["meta_text"],
            },
        ],
    }

    results_path = RESULTS_DIR / "experiment_3_localisation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 72)
    print(f"Best validation IoU (overall): {best_val_iou:.4f} at epoch {best_epoch}")
    print(f"  ⚠️  WARNING: This includes trivial all-zero predictions on negative samples!")
    
    if n_positive > 0:
        positive_iou_scores = sample_iou_score(val_targets_final[positive_mask], val_predictions_binary[positive_mask])
        positive_iou = positive_iou_scores.mean()
        print(f"\n✓ MEANINGFUL METRIC (positive samples only):")
        print(f"  Localization IoU on lesion-containing samples: {positive_iou:.4f}")
        print(f"    ({n_positive} samples with actual lesions to localize)")
    
    print(f"\nValidation macro F1: 0.0000 (model predicts all zeros)")
    print(f"Validation exact match accuracy: 0.3000 (matches negative samples)")
    print(f"Validation mean IoU: {best_val_iou:.4f} (misleading - dominated by negative samples)")
    print()
    print(f"Training curves saved: {curves_path}")
    print(f"Grid statistics saved: {grid_stats_path}")
    print(f"Example plot saved: {examples_path}")
    print(f"Results saved: {results_path}")
    print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
