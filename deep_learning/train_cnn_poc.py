from __future__ import annotations

import argparse
import csv
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
EPS = 1e-8


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_fd_ids(fd_dir: str) -> List[int]:
    ids: List[int] = []
    pattern = re.compile(r"scenario_(\d{3})_fd\.npz$")
    for name in os.listdir(fd_dir):
        m = pattern.match(name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_metadata(metadata_path: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid_raw = (row.get("scenario_id") or "").strip()
            if not sid_raw:
                continue
            try:
                sid = int(sid_raw)
            except ValueError:
                continue
            out[sid] = row
    return out


def downsample_frequency(signal: np.ndarray, target_points: int) -> np.ndarray:
    # signal shape: (C, F)
    c, f = signal.shape
    if f == target_points:
        return signal.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, f, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_points, dtype=np.float32)
    out = np.empty((c, target_points), dtype=np.float32)
    for i in range(c):
        out[i] = np.interp(x_new, x_old, signal[i].astype(np.float32)).astype(np.float32)
    return out


def load_fd_signal(fd_path: str, target_freq_points: int) -> np.ndarray:
    data = np.load(fd_path)
    if "signal" not in data:
        raise ValueError(f"Missing 'signal' key in {fd_path}")

    signal = data["signal"].astype(np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Expected signal shape (C,F), got {signal.shape} in {fd_path}")
    if signal.shape[0] != 512:
        raise ValueError(f"Expected 512 channels, got {signal.shape[0]} in {fd_path}")

    return downsample_frequency(signal, target_freq_points)


@dataclass
class Sample:
    sid: int
    x: np.ndarray  # shape (512, F_target)
    y: int


class CNNDataset(Dataset):
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs = torch.from_numpy(xs).float()
        self.ys = torch.from_numpy(ys).float()

    def __len__(self) -> int:
        return self.xs.shape[0]

    def __getitem__(self, idx: int):
        return self.xs[idx], self.ys[idx]


class SmallFreqCNN(nn.Module):
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).squeeze(-1)
        return self.classifier(h).squeeze(1)


def stratified_split(samples: List[Sample], train_ratio: float = 0.8) -> Tuple[List[Sample], List[Sample]]:
    pos = [s for s in samples if s.y == 1]
    neg = [s for s in samples if s.y == 0]

    if not pos or not neg:
        raise ValueError("Need both classes for classifier training")

    random.shuffle(pos)
    random.shuffle(neg)

    n_pos_train = max(1, int(len(pos) * train_ratio))
    n_neg_train = max(1, int(len(neg) * train_ratio))

    train = pos[:n_pos_train] + neg[:n_neg_train]
    val = pos[n_pos_train:] + neg[n_neg_train:]

    random.shuffle(train)
    random.shuffle(val)

    if not val:
        raise ValueError("Validation split is empty. Increase samples.")

    return train, val


def fit_norm(train_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Per-channel, per-frequency normalization fit on train only.
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def calc_metrics(logits: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    y_pred = (probs >= threshold).float()

    tp = float(((y_pred == 1) & (y_true == 1)).sum().item())
    tn = float(((y_pred == 0) & (y_true == 0)).sum().item())
    fp = float(((y_pred == 1) & (y_true == 0)).sum().item())
    fn = float(((y_pred == 0) & (y_true == 1)).sum().item())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    bal_acc = 0.5 * ((tp / max(tp + fn, 1.0)) + (tn / max(tn + fp, 1.0)))

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "bal_acc": bal_acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_samples(fd_dir: str, metadata_path: str, max_samples: int | None, freq_points: int) -> List[Sample]:
    ids = list_fd_ids(fd_dir)
    if max_samples is not None:
        ids = ids[:max_samples]

    meta = load_metadata(metadata_path)
    samples: List[Sample] = []
    for sid in ids:
        row = meta.get(sid)
        if row is None:
            continue
        y = int((row.get("has_lesion") or "0").strip() or "0")
        fd_path = os.path.join(fd_dir, f"scenario_{sid:03d}_fd.npz")
        x = load_fd_signal(fd_path, target_freq_points=freq_points)
        samples.append(Sample(sid=sid, x=x, y=y))

    if not samples:
        raise ValueError("No usable samples were loaded")

    return samples


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)

    if args.freq_points < 50 or args.freq_points > 100:
        raise ValueError("--freq-points should be within 50..100 for this lightweight setup")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"Using device: {device}")

    samples = load_samples(args.fd_dir, args.metadata, args.max_samples, args.freq_points)
    n_pos = sum(s.y for s in samples)
    n_neg = len(samples) - n_pos
    print(f"Loaded {len(samples)} samples | class0={n_neg} class1={n_pos} | freq_points={args.freq_points}")

    train_s, val_s = stratified_split(samples, train_ratio=args.train_ratio)
    print(f"Train size={len(train_s)} | Val size={len(val_s)}")

    x_train = np.stack([s.x for s in train_s], axis=0).astype(np.float32)
    y_train = np.array([s.y for s in train_s], dtype=np.float32)
    x_val = np.stack([s.x for s in val_s], axis=0).astype(np.float32)
    y_val = np.array([s.y for s in val_s], dtype=np.float32)

    mean, std = fit_norm(x_train)
    x_train = apply_norm(x_train, mean, std)
    x_val = apply_norm(x_val, mean, std)

    train_dl = DataLoader(CNNDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(CNNDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = SmallFreqCNN(in_channels=512).to(device)

    n_pos_train = max(1.0, float(y_train.sum()))
    n_neg_train = max(1.0, float(len(y_train) - y_train.sum()))
    pos_weight = torch.tensor([n_neg_train / n_pos_train], dtype=torch.float32, device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_bal_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * xb.size(0)
            train_n += xb.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        all_logits: List[torch.Tensor] = []
        all_y: List[torch.Tensor] = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)

                val_loss_sum += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)
                all_logits.append(logits.detach().cpu())
                all_y.append(yb.detach().cpu())

        val_logits = torch.cat(all_logits, dim=0)
        val_y = torch.cat(all_y, dim=0)
        metrics = calc_metrics(val_logits, val_y, threshold=args.threshold)

        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)

        if metrics["bal_acc"] > best_bal_acc:
            best_bal_acc = metrics["bal_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[CNN] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_acc={metrics['acc']:.3f} val_bal_acc={metrics['bal_acc']:.3f} "
            f"val_f1={metrics['f1']:.3f}"
        )

    if best_state is None:
        raise RuntimeError("Training finished without a best model state")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "classifier_cnn_poc.pt")
    torch.save(
        {
            "model_state": best_state,
            "feature_mean": mean,
            "feature_std": std,
            "freq_points": args.freq_points,
            "threshold": args.threshold,
            "train_ids": [s.sid for s in train_s],
            "val_ids": [s.sid for s in val_s],
            "best_val_bal_acc": best_bal_acc,
            "model_name": "SmallFreqCNN",
        },
        out_path,
    )
    print(f"Saved CNN checkpoint: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight frequency-aware 1D CNN on fd_tensors")
    parser.add_argument("--fd-dir", default="fd_tensors")
    parser.add_argument("--metadata", default="dataset_metadata.csv")
    parser.add_argument("--max-samples", type=int, default=150)
    parser.add_argument("--freq-points", type=int, default=64, help="Target frequency points (recommended 50..100)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", default="deep_learning/outputs")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
