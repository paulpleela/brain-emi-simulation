from __future__ import annotations

import argparse
import csv
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    TORCH_AVAILABLE = False

SEED = 42


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
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


def load_fd_feature(fd_path: str) -> np.ndarray:
    data = np.load(fd_path)
    if "signal" not in data:
        raise ValueError(f"Missing 'signal' key: {fd_path}")

    signal = data["signal"].astype(np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal (C,F), got shape {signal.shape} in {fd_path}")

    # Lightweight features for quick POC: per-channel mean and std over frequency.
    mean_feat = signal.mean(axis=1)
    std_feat = signal.std(axis=1)
    feat = np.concatenate([mean_feat, std_feat], axis=0)
    return feat.astype(np.float32)


@dataclass
class Sample:
    sid: int
    x: np.ndarray
    y: int


class FeatureDataset(Dataset):
    def __init__(self, xs: np.ndarray, ys: np.ndarray | None = None):
        self.xs = torch.from_numpy(xs).float()
        self.ys = None if ys is None else torch.from_numpy(ys).float()

    def __len__(self) -> int:
        return self.xs.shape[0]

    def __getitem__(self, idx: int):
        if self.ys is None:
            return self.xs[idx]
        return self.xs[idx], self.ys[idx]


if TORCH_AVAILABLE:
    class Autoencoder(nn.Module):
        def __init__(self, in_dim: int, latent_dim: int = 64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, in_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.encoder(x)
            return self.decoder(z)


    class BinaryMLP(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(1)


def prepare_samples(fd_dir: str, metadata_path: str, max_samples: int | None) -> List[Sample]:
    ids = list_fd_ids(fd_dir)
    if max_samples is not None:
        ids = ids[:max_samples]

    metadata = load_metadata(metadata_path)
    samples: List[Sample] = []
    for sid in ids:
        row = metadata.get(sid)
        if row is None:
            continue
        y = int((row.get("has_lesion") or "0").strip() or "0")
        fd_path = os.path.join(fd_dir, f"scenario_{sid:03d}_fd.npz")
        x = load_fd_feature(fd_path)
        samples.append(Sample(sid=sid, x=x, y=y))

    if not samples:
        raise ValueError("No usable samples found. Check fd tensors and metadata.")
    return samples


def zscore_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def train_autoencoder(samples: List[Sample], epochs: int, batch_size: int, lr: float, out_path: str, device: str) -> None:
    x = np.stack([s.x for s in samples], axis=0).astype(np.float32)
    mean, std = zscore_fit(x)
    x_norm = zscore_apply(x, mean, std)

    ds = FeatureDataset(x_norm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = Autoencoder(in_dim=x_norm.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        n = 0
        for xb in dl:
            xb = xb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        print(f"[AE] epoch {epoch:03d}/{epochs} loss={loss_sum / max(n, 1):.6f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "input_dim": x_norm.shape[1],
            "sample_ids": [s.sid for s in samples],
        },
        out_path,
    )
    print(f"Saved autoencoder checkpoint: {out_path}")


def train_autoencoder_numpy(samples: List[Sample], epochs: int, batch_size: int, lr: float, out_path: str) -> None:
    x = np.stack([s.x for s in samples], axis=0).astype(np.float32)
    mean, std = zscore_fit(x)
    x_norm = zscore_apply(x, mean, std)

    n, in_dim = x_norm.shape
    latent_dim = min(64, max(8, in_dim // 8))
    rng = np.random.default_rng(SEED)

    w1 = rng.normal(0.0, 0.02, size=(in_dim, latent_dim)).astype(np.float32)
    b1 = np.zeros((latent_dim,), dtype=np.float32)
    w2 = rng.normal(0.0, 0.02, size=(latent_dim, in_dim)).astype(np.float32)
    b2 = np.zeros((in_dim,), dtype=np.float32)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        x_shuf = x_norm[perm]
        loss_sum = 0.0
        seen = 0

        for i in range(0, n, batch_size):
            xb = x_shuf[i : i + batch_size]
            m = xb.shape[0]

            z_pre = xb @ w1 + b1
            z = np.tanh(z_pre)
            y = z @ w2 + b2

            diff = y - xb
            loss = float(np.mean(diff * diff))

            # dL/dy for MSE mean over all elements.
            dy = (2.0 / (m * in_dim)) * diff

            grad_w2 = z.T @ dy
            grad_b2 = dy.sum(axis=0)
            dz = dy @ w2.T
            dz_pre = dz * (1.0 - z * z)
            grad_w1 = xb.T @ dz_pre
            grad_b1 = dz_pre.sum(axis=0)

            w1 -= lr * grad_w1.astype(np.float32)
            b1 -= lr * grad_b1.astype(np.float32)
            w2 -= lr * grad_w2.astype(np.float32)
            b2 -= lr * grad_b2.astype(np.float32)

            loss_sum += loss * m
            seen += m

        print(f"[AE-NP] epoch {epoch:03d}/{epochs} loss={loss_sum / max(seen, 1):.6f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        feature_mean=mean,
        feature_std=std,
        sample_ids=np.array([s.sid for s in samples], dtype=np.int32),
    )
    print(f"Saved NumPy autoencoder checkpoint: {out_path}")


def split_classifier(samples: List[Sample], train_ratio: float = 0.8) -> Tuple[List[Sample], List[Sample]]:
    pos = [s for s in samples if s.y == 1]
    neg = [s for s in samples if s.y == 0]

    if not pos or not neg:
        raise ValueError("Classifier mode requires both classes in the selected subset")

    random.shuffle(pos)
    random.shuffle(neg)

    n_pos_train = max(1, int(len(pos) * train_ratio))
    n_neg_train = max(1, int(len(neg) * train_ratio))

    train = pos[:n_pos_train] + neg[:n_neg_train]
    valid = pos[n_pos_train:] + neg[n_neg_train:]

    random.shuffle(train)
    random.shuffle(valid)

    if not valid:
        raise ValueError("Validation split became empty. Increase sample count.")

    return train, valid


def train_classifier(samples: List[Sample], epochs: int, batch_size: int, lr: float, out_path: str, device: str) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Classifier mode requires PyTorch. Install torch in your environment first.")

    train_s, val_s = split_classifier(samples)

    x_train = np.stack([s.x for s in train_s], axis=0).astype(np.float32)
    y_train = np.array([s.y for s in train_s], dtype=np.float32)
    x_val = np.stack([s.x for s in val_s], axis=0).astype(np.float32)
    y_val = np.array([s.y for s in val_s], dtype=np.float32)

    mean, std = zscore_fit(x_train)
    x_train = zscore_apply(x_train, mean, std)
    x_val = zscore_apply(x_val, mean, std)

    train_dl = DataLoader(FeatureDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(FeatureDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    model = BinaryMLP(in_dim=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
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
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                val_correct += int((preds == yb).sum().item())
                val_loss_sum += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)

        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)
        print(f"[CLS] epoch {epoch:03d}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.3f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "input_dim": x_train.shape[1],
            "train_ids": [s.sid for s in train_s],
            "val_ids": [s.sid for s in val_s],
        },
        out_path,
    )
    print(f"Saved classifier checkpoint: {out_path}")


def train_classifier_numpy(samples: List[Sample], epochs: int, batch_size: int, lr: float, out_path: str) -> None:
    train_s, val_s = split_classifier(samples)

    x_train = np.stack([s.x for s in train_s], axis=0).astype(np.float32)
    y_train = np.array([s.y for s in train_s], dtype=np.float32)
    x_val = np.stack([s.x for s in val_s], axis=0).astype(np.float32)
    y_val = np.array([s.y for s in val_s], dtype=np.float32)

    mean, std = zscore_fit(x_train)
    x_train = zscore_apply(x_train, mean, std)
    x_val = zscore_apply(x_val, mean, std)

    n, d = x_train.shape
    rng = np.random.default_rng(SEED)
    w = rng.normal(0.0, 0.01, size=(d,)).astype(np.float32)
    b = np.float32(0.0)

    n_pos = max(1.0, float(y_train.sum()))
    n_neg = max(1.0, float((1.0 - y_train).sum()))
    # Simple inverse-frequency weighting for imbalance.
    pos_w = np.float32(0.5 * n / n_pos)
    neg_w = np.float32(0.5 * n / n_neg)

    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        x_shuf = x_train[perm]
        y_shuf = y_train[perm]

        train_loss_sum = 0.0
        seen = 0

        for i in range(0, n, batch_size):
            xb = x_shuf[i : i + batch_size]
            yb = y_shuf[i : i + batch_size]
            m = xb.shape[0]

            logits = xb @ w + b
            probs = sigmoid(logits)

            weights = np.where(yb > 0.5, pos_w, neg_w).astype(np.float32)
            eps = 1e-7
            loss_vec = -weights * (yb * np.log(probs + eps) + (1.0 - yb) * np.log(1.0 - probs + eps))
            loss = float(np.mean(loss_vec))

            grad_logits = (weights * (probs - yb)) / max(m, 1)
            grad_w = xb.T @ grad_logits
            grad_b = grad_logits.sum()

            w -= lr * grad_w.astype(np.float32)
            b -= np.float32(lr * float(grad_b))

            train_loss_sum += loss * m
            seen += m

        # Validation metrics.
        val_logits = x_val @ w + b
        val_probs = sigmoid(val_logits)
        val_preds = (val_probs >= 0.5).astype(np.float32)
        val_acc = float((val_preds == y_val).mean())

        tp = float(((val_preds == 1.0) & (y_val == 1.0)).sum())
        tn = float(((val_preds == 0.0) & (y_val == 0.0)).sum())
        fp = float(((val_preds == 1.0) & (y_val == 0.0)).sum())
        fn = float(((val_preds == 0.0) & (y_val == 1.0)).sum())
        tpr = tp / max(tp + fn, 1.0)
        tnr = tn / max(tn + fp, 1.0)
        bal_acc = 0.5 * (tpr + tnr)

        print(
            f"[CLS-NP] epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss_sum / max(seen, 1):.6f} "
            f"val_acc={val_acc:.3f} val_bal_acc={bal_acc:.3f}"
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        w=w,
        b=np.array([b], dtype=np.float32),
        feature_mean=mean,
        feature_std=std,
        train_ids=np.array([s.sid for s in train_s], dtype=np.int32),
        val_ids=np.array([s.sid for s in val_s], dtype=np.int32),
    )
    print(f"Saved NumPy classifier checkpoint: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="POC training on copied fd_tensors subset")
    parser.add_argument("--mode", choices=["autoencoder", "classifier"], default="autoencoder")
    parser.add_argument("--fd-dir", default="fd_tensors")
    parser.add_argument("--metadata", default="dataset_metadata.csv")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="deep_learning/outputs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--require-torch", action="store_true", help="Fail fast if PyTorch is unavailable")
    args = parser.parse_args()

    seed_all(args.seed)
    if args.require_torch and not TORCH_AVAILABLE:
        raise RuntimeError("--require-torch was set but PyTorch is not installed.")

    device = "cpu"
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    samples = prepare_samples(args.fd_dir, args.metadata, args.max_samples)
    labels = [s.y for s in samples]
    print(f"Loaded {len(samples)} samples | class0={sum(1 for y in labels if y == 0)} class1={sum(1 for y in labels if y == 1)}")

    if args.mode == "autoencoder":
        if TORCH_AVAILABLE:
            out_path = os.path.join(args.out_dir, "autoencoder_poc.pt")
            train_autoencoder(samples, args.epochs, args.batch_size, args.lr, out_path, device)
        else:
            out_path = os.path.join(args.out_dir, "autoencoder_poc_numpy.npz")
            train_autoencoder_numpy(samples, args.epochs, args.batch_size, args.lr, out_path)
    else:
        if TORCH_AVAILABLE:
            out_path = os.path.join(args.out_dir, "classifier_poc.pt")
            train_classifier(samples, args.epochs, args.batch_size, args.lr, out_path, device)
        else:
            out_path = os.path.join(args.out_dir, "classifier_poc_numpy.npz")
            train_classifier_numpy(samples, args.epochs, args.batch_size, args.lr, out_path)


if __name__ == "__main__":
    main()
