#!/usr/bin/env python3
"""Compare model inductive biases on mag_phase S-parameter tensors.

Models:
- MLP baseline: minimal structural assumptions
- Naive 3D CNN: treats frequency and ports as generic spatial axes
- Port-matrix 2D CNN: folds frequency and representation channels into the channel axis

This experiment uses the raw mag_phase archive and performs train-only normalization
inside the script so the comparison is not affected by precomputed statistics.
"""

from __future__ import annotations

import json
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.ticker import MaxNLocator


torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "deep_learning" / "datasets" / "dataset_mag_phase.npz"
if not DATASET_PATH.exists():
    fallback_dataset_path = REPO_ROOT / "datasets" / "dataset_mag_phase.npz"
    if fallback_dataset_path.exists():
        DATASET_PATH = fallback_dataset_path
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 1e-6


class FlattenMLPClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_dims: tuple[int, int, int] = (256, 128, 64), dropout_rate: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            previous = hidden_dim
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NaiveCNN3DClassifier(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class PortMatrixCNN2DClassifier(nn.Module):
    """2D CNN baseline that keeps the 16x16 port matrix spatial.

    Input shape: (B, F*C, 16, 16)
    """

    def __init__(self, in_channels: int, dropout_rate: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class FreqAwareClassifier(nn.Module):
    """Frequency-aware encoder: per-frequency shared MLP + attention pooling.

    Input: (B, F, 16, 16, 3)
    """

    def __init__(self, num_frequencies: int = 90, dropout_rate: float = 0.2):
        super().__init__()
        self.num_frequencies = num_frequencies
        in_dim = 16 * 16 * 3

        # Shared encoder applied to each frequency slice
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Attention scorer over frequency embeddings
        self.attn_scorer = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classification head on pooled embedding
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, 16, 16, 3)
        b, f, h, w, c = x.shape
        # Flatten per-frequency spatial slice
        x = x.view(b * f, -1)  # (B*F, 768)
        embeddings = self.encoder(x)  # (B*F, 64)
        embeddings = embeddings.view(b, f, -1)  # (B, F, 64)

        # Attention scores -> weights
        scores = self.attn_scorer(embeddings)  # (B, F, 1)
        scores = scores.view(b, f)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, F, 1)

        # Weighted sum pooling
        pooled = (weights * embeddings).sum(dim=1)  # (B, 64)

        # Classification head
        logits = self.head(pooled)  # (B, 1)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def to_model_input(model_name: str, batch_x: torch.Tensor) -> torch.Tensor:
    if model_name == "mlp":
        return batch_x.reshape(batch_x.shape[0], -1)
    if model_name == "naive_cnn":
        return batch_x.permute(0, 4, 1, 2, 3).contiguous()
    if model_name == "port_cnn":
        batch_x = batch_x.permute(0, 1, 4, 2, 3).contiguous()
        batch_size, num_frequencies, num_channels, height, width = batch_x.shape
        return batch_x.reshape(batch_size, num_frequencies * num_channels, height, width)
    if model_name == "freq_aware":
        # Freq-aware model consumes (B, F, 16, 16, 3) directly
        return batch_x
    raise ValueError(f"Unknown model input path: {model_name}")


def compute_train_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, EPSILON)
    return mean, std


def apply_normalization(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean) / std).astype(np.float32)


def normalize_dataset(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    x_train = data["X_train"].astype(np.float32)
    mean, std = compute_train_stats(x_train)

    normalized = {
        "X_train": apply_normalization(x_train, mean, std),
        "y_train": data["y_train"].astype(np.float32),
        "X_val": apply_normalization(data["X_val"].astype(np.float32), mean, std),
        "y_val": data["y_val"].astype(np.float32),
    }

    if "X_test" in data and "y_test" in data:
        normalized["X_test"] = apply_normalization(data["X_test"].astype(np.float32), mean, std)
        normalized["y_test"] = data["y_test"].astype(np.float32)

    return normalized, mean, std


def compute_auc(labels: list[float], predictions: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, predictions))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    model_name: str,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_labels: list[float] = []
    all_predictions: list[float] = []

    for batch_x, batch_y in loader:
        batch_x = to_model_input(model_name, batch_x.to(DEVICE))
        batch_y = batch_y.to(DEVICE).float()

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        all_predictions.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        all_labels.extend(batch_y.cpu().numpy())

    return total_loss / len(all_labels), compute_auc(all_labels, all_predictions)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, model_name: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_labels: list[float] = []
    all_predictions: list[float] = []

    for batch_x, batch_y in loader:
        batch_x = to_model_input(model_name, batch_x.to(DEVICE))
        batch_y = batch_y.to(DEVICE).float()
        logits = model(batch_x)
        loss = criterion(logits, batch_y.unsqueeze(1))
        total_loss += loss.item() * len(batch_y)
        all_predictions.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        all_labels.extend(batch_y.cpu().numpy())

    return total_loss / len(all_labels), compute_auc(all_labels, all_predictions)


def load_data() -> dict[str, np.ndarray]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")
    data = np.load(DATASET_PATH)
    required = ["X_train", "y_train", "X_val", "y_val"]
    missing = [name for name in required if name not in data.files]
    if missing:
        raise KeyError(f"Dataset is missing keys: {missing}")
    return {name: data[name] for name in data.files}


def build_loaders(data: dict[str, np.ndarray]) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    x_train = torch.from_numpy(data["X_train"])
    y_train = torch.from_numpy(data["y_train"])
    x_val = torch.from_numpy(data["X_val"])
    y_val = torch.from_numpy(data["y_val"])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32, shuffle=False)

    test_loader = None
    if "X_test" in data and "y_test" in data:
        x_test = torch.from_numpy(data["X_test"])
        y_test = torch.from_numpy(data["y_test"])
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader


def make_model(model_name: str, input_shape: tuple[int, ...]) -> nn.Module:
    if model_name == "mlp":
        return FlattenMLPClassifier(int(np.prod(input_shape[1:])))
    if model_name == "naive_cnn":
        return NaiveCNN3DClassifier(in_channels=input_shape[-1])
    if model_name == "port_cnn":
        return PortMatrixCNN2DClassifier(in_channels=input_shape[1] * input_shape[-1])
    if model_name == "freq_aware":
        # input_shape: (N, F, 16, 16, C)
        num_freq = int(input_shape[1])
        return FreqAwareClassifier(num_frequencies=num_freq)
    raise ValueError(f"Unknown model: {model_name}")


def make_optimizer(model_name: str, model: nn.Module) -> AdamW:
    learning_rate = 3e-4 if model_name in {"naive_cnn", "port_cnn", "freq_aware"} else 1e-3
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)


def summarize_metrics(name: str, metrics: dict[str, object]) -> str:
    test_text = "n/a" if metrics["test_auc_final"] is None else f"{metrics['test_auc_final']:.4f}"
    return (
        f"{name:<14} "
        f"{metrics['param_count']:<10,} "
        f"{metrics['train_auc_best']:<10.4f} "
        f"{metrics['val_auc_best']:<10.4f} "
        f"{test_text:<10} "
        f"{metrics['generalization_gap']:<10.4f}"
    )


def main() -> int:
    print("=" * 72)
    print("EXPERIMENT 2: Model Inductive Bias Study")
    print("=" * 72)
    print()
    print("Fixed representation: mag_phase")
    print("Goal: compare model assumptions, not do architecture search")
    print()

    data = load_data()
    normalized_data, mean, std = normalize_dataset(data)
    train_loader, val_loader, test_loader = build_loaders(normalized_data)
    input_shape = normalized_data["X_train"].shape

    model_specs = [
        ("mlp", "MLP baseline", "minimal assumptions; all features global"),
        ("naive_cnn", "Naive 3D CNN", "assumes frequency and ports are local spatial axes"),
        ("port_cnn", "Port-matrix 2D CNN", "Conv2D on 16x16 ports with frequency folded into channels"),
        ("freq_aware", "Freq-aware encoder", "Per-frequency MLP encoder + attention pooling"),
    ]

    results: dict[str, dict[str, object]] = {}

    print(f"Raw dataset: {input_shape} -> normalized with train-only stats")
    print(f"Normalization mean shape: {mean.shape}")
    print(f"Normalization std shape:  {std.shape}")
    print(f"Train class balance: {float(normalized_data['y_train'].mean()):.1%}")
    print(f"Val class balance:   {float(normalized_data['y_val'].mean()):.1%}")
    if "y_test" in normalized_data:
        print(f"Test class balance:  {float(normalized_data['y_test'].mean()):.1%}")
    print()

    preview_batch = next(iter(train_loader))[0]
    print("Input shapes before each model:")
    print(f"  MLP:           {tuple(preview_batch.reshape(preview_batch.shape[0], -1).shape)}")
    print(f"  Naive 3D CNN:  {tuple(preview_batch.permute(0, 4, 1, 2, 3).shape)}")
    port_cnn_preview = preview_batch.permute(0, 1, 4, 2, 3).contiguous().reshape(
        preview_batch.shape[0], preview_batch.shape[1] * preview_batch.shape[-1], preview_batch.shape[2], preview_batch.shape[3]
    )
    print(f"  Port 2D CNN:   {tuple(port_cnn_preview.shape)}")
    print()

    # Training configuration for thesis-standard runs
    max_epochs = 20
    seeds = [42, 43, 44]

    # Collect per-seed results
    all_seed_results: dict[int, dict[str, dict[str, object]]] = {}

    for seed in seeds:
        print("=" * 72)
        print(f"SEED: {seed}")
        print("=" * 72)
        torch.manual_seed(seed)
        np.random.seed(seed)

        seed_results: dict[str, dict[str, object]] = {}

        for model_name, display_name, description in model_specs:
            print("=" * 72)
            print(f"Testing: {display_name} (seed={seed})")
            print("=" * 72)

            model = make_model(model_name, input_shape).to(DEVICE)
            optimizer = make_optimizer(model_name, model)
            criterion = nn.BCEWithLogitsLoss()
            param_count = count_parameters(model)

            print(f"  Parameters: {param_count:,}")
            print(f"  Assumption: {description}")
            print(f"  Training for exactly {max_epochs} epochs")

            train_losses: list[float] = []
            val_losses: list[float] = []
            train_aucs: list[float] = []
            val_aucs: list[float] = []

            best_val_auc = float("-inf")
            best_epoch = 0
            best_train_auc = float("nan")
            best_state_dict: dict[str, torch.Tensor] | None = None
            best_checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best_seed{seed}.pt"

            for epoch in range(1, max_epochs + 1):
                train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, model_name)
                val_loss, val_auc = evaluate(model, val_loader, criterion, model_name)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_aucs.append(train_auc)
                val_aucs.append(val_auc)

                if val_auc > best_val_auc + 1e-12:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    best_train_auc = train_auc
                    best_state_dict = copy.deepcopy(model.state_dict())
                    torch.save(
                        {
                            "model_name": model_name,
                            "seed": seed,
                            "epoch": epoch,
                            "val_auc": val_auc,
                            "state_dict": best_state_dict,
                        },
                        best_checkpoint_path,
                    )
                if epoch % 5 == 0 or epoch == 1:
                    print(
                        f"    Epoch {epoch:2d}/{max_epochs}: "
                        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                        f"train_auc={train_auc:.4f} val_auc={val_auc:.4f}"
                    )

            # Load best checkpoint and compute final metrics using model.eval()
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)

            train_loss_best, train_auc_best = evaluate(model, train_loader, criterion, model_name)
            val_loss_best, val_auc_best = evaluate(model, val_loader, criterion, model_name)

            test_auc_best = None
            test_loss_best = None
            if test_loader is not None:
                test_loss_best, test_auc_best = evaluate(model, test_loader, criterion, model_name)

            seed_results[model_name] = {
                "display_name": display_name,
                "description": description,
                "param_count": int(param_count),
                "train_auc_final": float(train_aucs[-1]),
                "val_auc_final": float(val_aucs[-1]),
                "train_auc_best": float(train_auc_best),
                "val_auc_best": float(val_auc_best),
                "best_epoch": int(best_epoch),
                "generalization_gap": float(train_auc_best - val_auc_best),
                "test_auc_best": test_auc_best,
                "test_loss_best": test_loss_best,
                "best_checkpoint": str(best_checkpoint_path),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_aucs": train_aucs,
                "val_aucs": val_aucs,
            }

            print()
            print(f"  Best train AUC (best checkpoint): {train_auc_best:.4f}")
            print(f"  Best val AUC (best checkpoint):   {val_auc_best:.4f} (epoch {best_epoch})")
            if test_auc_best is not None:
                print(f"  Test AUC (best checkpoint):       {test_auc_best:.4f}")
            print()

        all_seed_results[seed] = seed_results

    # Aggregate across seeds
    aggregate: dict[str, dict[str, float]] = {}
    import statistics

    for model_name, _, _ in model_specs:
        test_aucs = [all_seed_results[s][model_name]["test_auc_best"] for s in seeds if all_seed_results[s][model_name]["test_auc_best"] is not None]
        test_aucs = [float(x) for x in test_aucs]
        if len(test_aucs) > 0:
            mean_auc = statistics.mean(test_aucs)
            std_auc = statistics.pstdev(test_aucs) if len(test_aucs) > 1 else 0.0
        else:
            mean_auc = float("nan")
            std_auc = float("nan")
        aggregate[model_name] = {"test_auc_mean": mean_auc, "test_auc_std": std_auc}

    # Save aggregated results and per-seed details
    results_json = RESULTS_DIR / "experiment_2_model_bias_results.json"
    out = {
        "per_seed": all_seed_results,
        "aggregate": aggregate,
        "seeds": seeds,
    }
    with results_json.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"✅ Results saved: {results_json}")

    print("=" * 72)
    print("Generating comparison plot (mean ± std across seeds)...")
    print("=" * 72)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Experiment 2: Model Inductive Bias Study\nmag_phase representation fixed (mean ± std across seeds)",
        fontsize=14,
        fontweight="bold",
    )

    colors = {
        "mlp": "tab:blue",
        "naive_cnn": "tab:orange",
        "port_cnn": "tab:red",
        "freq_aware": "tab:purple",
    }

    # Helper to build epoch-aligned arrays from per-seed model histories.
    def build_epoch_matrix(model_name: str, metric: str):
        matrices = []
        max_len = 0
        for seed in seeds:
            seq = np.asarray(all_seed_results[seed][model_name][metric], dtype=np.float32)
            matrices.append(seq)
            if seq.shape[0] > max_len:
                max_len = seq.shape[0]
        mat = np.full((len(matrices), max_len), np.nan, dtype=np.float32)
        for i, arr in enumerate(matrices):
            mat[i, : arr.shape[0]] = arr
        return mat

    # Plot training loss
    ax = axes[0, 0]
    for model_name, _, _ in model_specs:
        mats = build_epoch_matrix(model_name, "train_losses")
        mean = np.nanmean(mats, axis=0)
        std = np.nanstd(mats, axis=0)
        epochs = np.arange(1, mean.shape[0] + 1)
        ax.plot(epochs, mean, label=model_name, color=colors[model_name], linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=colors[model_name], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss (mean ± std)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot validation loss
    ax = axes[0, 1]
    for model_name, _, _ in model_specs:
        mats = build_epoch_matrix(model_name, "val_losses")
        mean = np.nanmean(mats, axis=0)
        std = np.nanstd(mats, axis=0)
        epochs = np.arange(1, mean.shape[0] + 1)
        ax.plot(epochs, mean, label=model_name, color=colors[model_name], linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=colors[model_name], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss (mean ± std)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot training AUC
    ax = axes[1, 0]
    for model_name, _, _ in model_specs:
        mats = build_epoch_matrix(model_name, "train_aucs")
        mean = np.nanmean(mats, axis=0)
        std = np.nanstd(mats, axis=0)
        epochs = np.arange(1, mean.shape[0] + 1)
        ax.plot(epochs, mean, label=model_name, color=colors[model_name], linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=colors[model_name], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Training AUC (mean ± std)")
    ax.set_ylim([0.45, 0.90])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot validation AUC
    ax = axes[1, 1]
    for model_name, _, _ in model_specs:
        # per-seed val_aucs sequences
        mats = []
        max_len = 0
        for s in seeds:
            seq = np.asarray(all_seed_results[s][model_name]["val_aucs"], dtype=np.float32)
            mats.append(seq)
            if seq.shape[0] > max_len:
                max_len = seq.shape[0]
        mat = np.full((len(mats), max_len), np.nan, dtype=np.float32)
        for i, arr in enumerate(mats):
            mat[i, : arr.shape[0]] = arr
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        epochs = np.arange(1, mean.shape[0] + 1)
        ax.plot(epochs, mean, label=model_name, color=colors[model_name], linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=colors[model_name], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC (mean ± std)")
    ax.set_ylim([0.45, 0.90])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = RESULTS_DIR / "experiment_2_model_bias_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved: {plot_path}")

    print()
    print("=" * 72)
    print("EXPERIMENT 2 SUMMARY (aggregated over seeds)")
    print("=" * 72)
    print()
    print(f"{'Model':<14} {'Params':<10} {'Test AUC (mean±std)':<25}")
    print("-" * 72)
    for model_name, _, _ in model_specs:
        mean = aggregate[model_name]["test_auc_mean"]
        std = aggregate[model_name]["test_auc_std"]
        params = int(all_seed_results[seeds[0]][model_name]["param_count"])
        print(f"{model_name:<14} {params:<10,} {mean:.4f} ± {std:.4f}")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
