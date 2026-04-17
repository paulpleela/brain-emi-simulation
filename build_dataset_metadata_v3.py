"""
Build dataset_metadata_v3.csv for robust subset analysis.

Goals:
- Keep one large dataset (same scenarios) for flexible subset experiments
- Ensure both classes appear in train/val/test
- Ensure head_scale, head_rotation_deg, and noise_level are well represented
- Include explicit nominal baseline anchors for performance-vs-variation studies

Output columns remain backward-compatible with generation scripts; one extra analysis
column is added:
- analysis_profile: "baseline_nominal" or "varied"
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_SRC = Path("dataset_metadata.csv")
DEFAULT_DST = Path("dataset_metadata_v3.csv")
SEED = 20260417

SPLIT_TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}

# Analysis-friendly controlled levels.
SCALE_LEVELS = [0.90, 0.95, 1.00, 1.05, 1.10]
ROT_LEVELS = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
NOISE_LEVELS = ["low", "medium", "high"]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def assign_split_by_label(rows: List[Dict[str, str]]) -> None:
    rng = random.Random(SEED)
    by_label: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        y = int((row.get("has_lesion") or "0").strip() or "0")
        by_label[y].append(row)

    for y, group in by_label.items():
        group_sorted = sorted(group, key=lambda r: int(r["scenario_id"]))
        rng.shuffle(group_sorted)
        n = len(group_sorted)
        n_train = int(round(SPLIT_TARGET["train"] * n))
        n_val = int(round(SPLIT_TARGET["val"] * n))
        n_test = n - n_train - n_val

        # Guard against tiny rounding edge cases.
        if n_test < 0:
            n_test = 0
        if n_train + n_val + n_test != n:
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            n_test = n - n_train - n_val

        for i, row in enumerate(group_sorted):
            if i < n_train:
                row["split"] = "train"
            elif i < n_train + n_val:
                row["split"] = "val"
            else:
                row["split"] = "test"


def assign_variability(rows: List[Dict[str, str]]) -> None:
    rng = random.Random(SEED + 1)
    by_group: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        split = (row.get("split") or "train").strip().lower()
        y = int((row.get("has_lesion") or "0").strip() or "0")
        by_group[(split, y)].append(row)

    for key, group in by_group.items():
        group_sorted = sorted(group, key=lambda r: int(r["scenario_id"]))
        rng.shuffle(group_sorted)

        for i, row in enumerate(group_sorted):
            row["head_scale"] = f"{SCALE_LEVELS[i % len(SCALE_LEVELS)]:.4f}"
            row["head_rotation_deg"] = f"{ROT_LEVELS[i % len(ROT_LEVELS)]:.4f}"
            row["noise_level"] = NOISE_LEVELS[i % len(NOISE_LEVELS)]
            row["analysis_profile"] = "varied"

        # Explicit nominal anchors per split+label for clean baseline comparisons.
        n_anchor = max(5, int(round(0.10 * len(group_sorted))))
        for row in group_sorted[:n_anchor]:
            row["head_scale"] = "1.0000"
            row["head_rotation_deg"] = "0.0000"
            row["noise_level"] = "low"
            row["analysis_profile"] = "baseline_nominal"


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows to write")

    # Keep source column order and append analysis_profile.
    fieldnames = list(rows[0].keys())
    if "analysis_profile" not in fieldnames:
        fieldnames.append("analysis_profile")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, str]]) -> None:
    split_label_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    profile_counts: Dict[str, int] = defaultdict(int)
    noise_counts: Dict[str, int] = defaultdict(int)

    hs_vals = []
    hr_vals = []
    for row in rows:
        split = row.get("split", "").strip().lower()
        y = int((row.get("has_lesion") or "0").strip() or "0")
        split_label_counts[(split, y)] += 1
        profile_counts[row.get("analysis_profile", "")] += 1
        noise_counts[row.get("noise_level", "")] += 1
        hs_vals.append(float(row.get("head_scale", "1.0")))
        hr_vals.append(float(row.get("head_rotation_deg", "0.0")))

    print("rows:", len(rows))
    print("split+label:")
    for split in ("train", "val", "test"):
        for y in (0, 1):
            print(f"  {split} y={y}: {split_label_counts[(split, y)]}")
    print("analysis_profile:", dict(profile_counts))
    print("noise_level:", dict(noise_counts))
    print("head_scale range:", min(hs_vals), max(hs_vals))
    print("head_rotation_deg range:", min(hr_vals), max(hr_vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build analysis-ready metadata v3")
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="Source metadata CSV")
    parser.add_argument("--out", default=str(DEFAULT_DST), help="Output metadata CSV")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.out)
    if not src.is_file():
        raise FileNotFoundError(f"Source metadata not found: {src}")

    rows = read_rows(src)
    assign_split_by_label(rows)
    assign_variability(rows)
    write_rows(dst, rows)
    print(f"Wrote: {dst}")
    summarize(rows)


if __name__ == "__main__":
    main()
