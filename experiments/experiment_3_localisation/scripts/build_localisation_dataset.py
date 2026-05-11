#!/usr/bin/env python3
"""Build a coarse 4x4 lesion-localisation dataset from the normalized mag_phase archive.

This script preserves the existing train/val split and sample ordering used by the
classification archive, then adds multi-label 4x4 grid targets derived from the
metadata lesion position and size.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
METADATA_PATH = REPO_ROOT / "dataset_metadata.csv"
SOURCE_DATASET_PATH = REPO_ROOT / "datasets" / "dataset_mag_phase_normalized.npz"
OUTPUT_DATASET_PATH = REPO_ROOT / "datasets" / "dataset_mag_phase_localisation.npz"
SPARAMS_GLOB = str(REPO_ROOT / "sparams" / "scenario_*.s16p")

GRID_SIZE = 4
HEAD_A = 0.095
HEAD_B = 0.075


def circle_intersects_rect(cx: float, cy: float, radius: float, x0: float, x1: float, y0: float, y1: float) -> bool:
    closest_x = min(max(cx, x0), x1)
    closest_y = min(max(cy, y0), y1)
    dx = closest_x - cx
    dy = closest_y - cy
    return (dx * dx) + (dy * dy) <= radius * radius


def build_grid_label(row: pd.Series) -> np.ndarray:
    label = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    if int(row["has_lesion"]) == 0:
        return label.reshape(-1)

    head_scale = float(row["head_scale"]) if float(row["head_scale"]) > 0 else 1.0
    center_x = float(row["lesion_x"]) * head_scale
    center_y = float(row["lesion_y"]) * head_scale
    radius = float(row["lesion_size_mm"]) / 1000.0

    x_edges = np.linspace(-HEAD_A * head_scale, HEAD_A * head_scale, GRID_SIZE + 1)
    y_edges = np.linspace(HEAD_B * head_scale, -HEAD_B * head_scale, GRID_SIZE + 1)

    for row_idx in range(GRID_SIZE):
        y_top = float(y_edges[row_idx])
        y_bottom = float(y_edges[row_idx + 1])
        y0 = min(y_top, y_bottom)
        y1 = max(y_top, y_bottom)
        for col_idx in range(GRID_SIZE):
            x_left = float(x_edges[col_idx])
            x_right = float(x_edges[col_idx + 1])
            x0 = min(x_left, x_right)
            x1 = max(x_left, x_right)
            if circle_intersects_rect(center_x, center_y, radius, x0, x1, y0, y1):
                label[row_idx, col_idx] = 1.0

    return label.reshape(-1)


def load_ordered_samples(metadata: pd.DataFrame) -> list[pd.Series]:
    meta_by_id = metadata.set_index("scenario_id")
    ordered_samples: list[pd.Series] = []
    for filepath in glob.glob(SPARAMS_GLOB):
        basename = os.path.basename(filepath)
        try:
            scenario_id = int(basename.split("_")[1].split(".")[0])
        except ValueError:
            continue
        if scenario_id not in meta_by_id.index:
            continue
        row = meta_by_id.loc[scenario_id].copy()
        row["scenario_id"] = scenario_id
        if str(row["split"]).strip() in {"train", "val"}:
            ordered_samples.append(row)
    return ordered_samples


def validate_binary_order(source_dataset: dict[str, np.ndarray], ordered_samples: Iterable[pd.Series]) -> None:
    ordered_samples = list(ordered_samples)
    train_labels = np.array([int(row["has_lesion"]) for row in ordered_samples if str(row["split"]).strip() == "train"], dtype=np.int64)
    val_labels = np.array([int(row["has_lesion"]) for row in ordered_samples if str(row["split"]).strip() == "val"], dtype=np.int64)

    if not np.array_equal(train_labels, source_dataset["y_train"]):
        raise RuntimeError("Train split order does not match the source classification archive; aborting localisation label build.")
    if not np.array_equal(val_labels, source_dataset["y_val"]):
        raise RuntimeError("Val split order does not match the source classification archive; aborting localisation label build.")


def main() -> int:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")
    if not SOURCE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing normalized dataset archive: {SOURCE_DATASET_PATH}")

    metadata = pd.read_csv(METADATA_PATH).sort_values("scenario_id")
    source_dataset = np.load(SOURCE_DATASET_PATH)
    ordered_samples = load_ordered_samples(metadata)
    validate_binary_order(source_dataset, ordered_samples)

    x_train = source_dataset["X_train"]
    x_val = source_dataset["X_val"]

    y_train = []
    y_val = []
    train_scenario_ids = []
    val_scenario_ids = []

    for row in ordered_samples:
        label = build_grid_label(row)
        if str(row["split"]).strip() == "train":
            y_train.append(label)
            train_scenario_ids.append(int(row["scenario_id"]))
        elif str(row["split"]).strip() == "val":
            y_val.append(label)
            val_scenario_ids.append(int(row["scenario_id"]))

    y_train_array = np.asarray(y_train, dtype=np.float32)
    y_val_array = np.asarray(y_val, dtype=np.float32)

    if y_train_array.shape[0] != x_train.shape[0]:
        raise RuntimeError(f"Train localisation labels mismatch: {y_train_array.shape[0]} vs {x_train.shape[0]}")
    if y_val_array.shape[0] != x_val.shape[0]:
        raise RuntimeError(f"Val localisation labels mismatch: {y_val_array.shape[0]} vs {x_val.shape[0]}")

    os.makedirs(OUTPUT_DATASET_PATH.parent, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DATASET_PATH,
        X_train=x_train,
        X_val=x_val,
        Y_train=y_train_array,
        Y_val=y_val_array,
        y_train=source_dataset["y_train"],
        y_val=source_dataset["y_val"],
        train_scenario_ids=np.asarray(train_scenario_ids, dtype=np.int32),
        val_scenario_ids=np.asarray(val_scenario_ids, dtype=np.int32),
        grid_size=np.array([GRID_SIZE], dtype=np.int32),
        head_a=np.array([HEAD_A], dtype=np.float32),
        head_b=np.array([HEAD_B], dtype=np.float32),
    )

    train_active_cells = int(y_train_array.sum())
    val_active_cells = int(y_val_array.sum())
    print("=" * 80)
    print("EXPERIMENT 3 LOCALISATION DATASET BUILDER")
    print("=" * 80)
    print(f"Source archive: {SOURCE_DATASET_PATH}")
    print(f"Output archive: {OUTPUT_DATASET_PATH}")
    print(f"Train samples: {x_train.shape[0]}")
    print(f"Val samples:   {x_val.shape[0]}")
    print(f"Train active cells: {train_active_cells}")
    print(f"Val active cells:   {val_active_cells}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
