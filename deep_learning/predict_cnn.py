from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deep_learning.train_cnn_poc import SmallFreqCNN, load_fd_signal


def parse_scenario_id(name: str) -> int | None:
    m = re.match(r"scenario_(\d{3})_fd\.npz$", name)
    if not m:
        return None
    return int(m.group(1))


def list_fd_paths(fd_dir: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for name in os.listdir(fd_dir):
        sid = parse_scenario_id(name)
        if sid is None:
            continue
        out.append((sid, os.path.join(fd_dir, name)))
    out.sort(key=lambda x: x[0])
    return out


def load_labels(metadata_path: str) -> Dict[int, int]:
    labels: Dict[int, int] = {}
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
            labels[sid] = int((row.get("has_lesion") or "0").strip() or "0")
    return labels


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    bal_acc = 0.5 * ((tp / max(tp + fn, 1.0)) + (tn / max(tn + fp, 1.0)))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Readable predictions from CNN checkpoint")
    parser.add_argument("--checkpoint", default="deep_learning/outputs/classifier_cnn_poc.pt")
    parser.add_argument("--fd-dir", default="fd_tensors")
    parser.add_argument("--metadata", default="dataset_metadata.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--show", type=int, default=25)
    parser.add_argument("--only-errors", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.fd_dir):
        raise FileNotFoundError(f"FD dir not found: {args.fd_dir}")

    labels: Dict[int, int] = {}
    has_labels = os.path.isfile(args.metadata)
    if has_labels:
        labels = load_labels(args.metadata)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {"model_state", "feature_mean", "feature_std", "freq_points"}
    missing = required - set(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")

    threshold = float(ckpt.get("threshold", 0.5)) if args.threshold is None else args.threshold

    model = SmallFreqCNN(in_channels=512)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = ckpt["feature_mean"].astype(np.float32)
    std = ckpt["feature_std"].astype(np.float32)
    freq_points = int(ckpt["freq_points"])

    fd_paths = list_fd_paths(args.fd_dir)
    if args.max_samples is not None:
        fd_paths = fd_paths[: args.max_samples]

    rows: List[Tuple[int, float, int, int | None]] = []
    with torch.no_grad():
        for sid, path in fd_paths:
            x = load_fd_signal(path, target_freq_points=freq_points).astype(np.float32)
            x = ((x - mean) / np.maximum(std, 1e-8)).astype(np.float32)
            x_t = torch.from_numpy(x[None, ...])
            prob = float(torch.sigmoid(model(x_t)).item())
            pred = 1 if prob >= threshold else 0
            y = labels.get(sid) if has_labels else None
            rows.append((sid, prob, pred, y))

    print("CNN Prediction report")
    print("=" * 80)
    print(f"checkpoint: {args.checkpoint}")
    print(f"samples scored: {len(rows)}")
    print(f"threshold: {threshold:.2f}")

    pred_pos = sum(1 for _, _, p, _ in rows if p == 1)
    pred_neg = len(rows) - pred_pos
    print(f"predicted positives: {pred_pos} | predicted negatives: {pred_neg}")

    if has_labels and rows:
        scored = [(sid, prob, pred, y) for sid, prob, pred, y in rows if y is not None]
        if scored:
            y_true = np.array([int(y) for _, _, _, y in scored], dtype=np.int32)
            y_pred = np.array([pred for _, _, pred, _ in scored], dtype=np.int32)
            m = compute_metrics(y_true, y_pred)
            print("metrics:")
            print(
                f"  accuracy={m['accuracy']:.3f} precision={m['precision']:.3f} "
                f"recall={m['recall']:.3f} f1={m['f1']:.3f} bal_acc={m['balanced_accuracy']:.3f}"
            )
            print(f"  confusion: TP={int(m['tp'])} TN={int(m['tn'])} FP={int(m['fp'])} FN={int(m['fn'])}")

    print("-" * 80)
    print("scenario_id  prob_lesion  pred", end="")
    if has_labels:
        print("  true  ok")
    else:
        print()

    shown = 0
    for sid, prob, pred, y in rows:
        if args.only_errors and has_labels and y is not None and pred == y:
            continue

        if has_labels and y is not None:
            ok = "yes" if pred == y else "no"
            print(f"{sid:10d}  {prob:11.4f}  {pred:4d}  {y:4d}  {ok:>2s}")
        else:
            print(f"{sid:10d}  {prob:11.4f}  {pred:4d}")

        shown += 1
        if shown >= args.show:
            break

    if shown == 0:
        print("(no rows to display for current filters)")


if __name__ == "__main__":
    main()
