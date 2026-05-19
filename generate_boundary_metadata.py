#!/usr/bin/env python3
"""Generate extra valid in-head boundary lesion metadata.

This creates a second metadata CSV for additional boundary-focused scenarios
without modifying the original `dataset_metadata.csv`.

Default output:
  dataset_metadata_boundary_valid.csv

Default scenario IDs:
  1001..1050

The generator uses rejection sampling and only accepts lesions whose sampled
surface remains inside the ellipsoidal head.  It also records the same geometry
validity columns used by the localisation filtering workflow.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


OUTPUT_FILE = Path("dataset_metadata_boundary_valid.csv")
SCENARIO_ID_START = 1001
TOTAL_SAMPLES = 50

HEAD_A = 0.095
HEAD_B = 0.075
HEAD_C = 0.115

FIELDNAMES = [
    "scenario_id",
    "is_base_case",
    "has_lesion",
    "lesion_size_mm",
    "lesion_x",
    "lesion_y",
    "lesion_z",
    "epsilon_variation",
    "sigma_variation",
    "head_scale",
    "head_rotation_deg",
    "noise_level",
    "split",
    "group",
    "size_bucket",
    "region",
    "shape",
    "epsilon_anomaly_variation",
    "sigma_anomaly_variation",
    "background_epsilon_variation",
    "background_sigma_variation",
    "lesion_center_inside_head",
    "lesion_fully_inside_head",
    "lesion_head_max_ellipsoid_value",
    "lesion_outside_head_fraction_sampled",
    "exclude_from_localisation",
    "exclusion_reason",
]

SIZE_BUCKETS = [
    ("small", 5.0, 10.0),
    ("medium", 10.0, 20.0),
    ("large", 20.0, 30.0),
]
SHAPE_PROBS = {"sphere": 0.70, "ellipsoid": 0.30}
NOISE_LEVELS = ["low", "medium", "high"]
NOISE_PROBS = [0.50, 0.30, 0.20]
SPLITS = ["train", "val", "test"]
SPLIT_PROBS = [0.70, 0.15, 0.15]


def sample_unit_sphere(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    points = rng.normal(size=(num_samples, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    extrema = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            point = np.zeros(3)
            point[axis] = sign
            extrema.append(point)
    return np.vstack([points, np.asarray(extrema)])


def ellipsoid_value(points: np.ndarray) -> np.ndarray:
    return (points[:, 0] / HEAD_A) ** 2 + (points[:, 1] / HEAD_B) ** 2 + (points[:, 2] / HEAD_C) ** 2


def local_to_world(points: np.ndarray, head_scale: float, head_rotation_deg: float) -> np.ndarray:
    theta = np.deg2rad(head_rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    scaled = points * head_scale
    x = cos_t * scaled[:, 0] - sin_t * scaled[:, 1]
    y = sin_t * scaled[:, 0] + cos_t * scaled[:, 1]
    z = scaled[:, 2]
    return np.stack([x, y, z], axis=1)


def world_to_local(points: np.ndarray, head_scale: float, head_rotation_deg: float) -> np.ndarray:
    theta = np.deg2rad(head_rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = cos_t * points[:, 0] + sin_t * points[:, 1]
    y = -sin_t * points[:, 0] + cos_t * points[:, 1]
    z = points[:, 2]
    return np.stack([x / head_scale, y / head_scale, z / head_scale], axis=1)


def lesion_surface_local(
    centre_local: np.ndarray,
    size_mm: float,
    shape: str,
    head_scale: float,
    head_rotation_deg: float,
    unit_sphere: np.ndarray,
) -> np.ndarray:
    centre_world = local_to_world(centre_local[None, :], head_scale, head_rotation_deg)[0]
    radius = size_mm / 1000.0
    if shape == "ellipsoid":
        axes = np.asarray([radius, radius * 0.8, radius * 1.2], dtype=np.float64)
    else:
        axes = np.asarray([radius, radius, radius], dtype=np.float64)
    surface_world = centre_world[None, :] + unit_sphere * axes[None, :]
    return world_to_local(surface_world, head_scale, head_rotation_deg)


def validate_lesion(
    centre_local: np.ndarray,
    size_mm: float,
    shape: str,
    head_scale: float,
    head_rotation_deg: float,
    unit_sphere: np.ndarray,
) -> tuple[bool, bool, float, float]:
    centre_value = float(ellipsoid_value(centre_local[None, :])[0])
    surface = lesion_surface_local(centre_local, size_mm, shape, head_scale, head_rotation_deg, unit_sphere)
    values = ellipsoid_value(surface)
    outside = values > 1.0 + 1e-9
    return centre_value <= 1.0 + 1e-9, not bool(outside.any()), float(values.max()), float(outside.mean())


def make_balanced_labels(total: int, labels: list[str], probs: list[float], rng: np.random.Generator) -> list[str]:
    raw = np.asarray(probs, dtype=np.float64)
    raw = raw / raw.sum()
    counts = np.floor(raw * total).astype(int)
    while counts.sum() < total:
        counts[np.argmax(raw * total - counts)] += 1
    out: list[str] = []
    for label, count in zip(labels, counts):
        out.extend([label] * int(count))
    rng.shuffle(out)
    return out


def make_size_plan(total: int, rng: np.random.Generator) -> list[tuple[str, float, float]]:
    counts = [total // 3] * 3
    for i in range(total - sum(counts)):
        counts[i] += 1
    plan: list[tuple[str, float, float]] = []
    for bucket, count in zip(SIZE_BUCKETS, counts):
        plan.extend([bucket] * count)
    rng.shuffle(plan)
    return plan


def sample_valid_boundary_position(
    rng: np.random.Generator,
    size_mm: float,
    shape: str,
    head_scale: float,
    head_rotation_deg: float,
    unit_sphere: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return a boundary-like lesion centre whose volume remains inside head.

    We accept samples whose lesion surface reaches close to the head boundary
    while no sampled surface point exits the ellipsoid.
    """

    for _ in range(20000):
        theta = rng.uniform(0.0, 2.0 * math.pi)
        edge_r = 1.0 / math.sqrt((math.cos(theta) ** 2) / (HEAD_A**2) + (math.sin(theta) ** 2) / (HEAD_B**2))

        # Large lesions need more inward centres, but their surface can still
        # be boundary-adjacent.  Rejection below enforces that condition.
        frac = rng.uniform(0.42, 0.90)
        x = frac * edge_r * math.cos(theta)
        y = frac * edge_r * math.sin(theta)
        z = rng.uniform(-0.045, 0.045)
        centre = np.asarray([x, y, z], dtype=np.float64)

        centre_inside, fully_inside, max_value, outside_fraction = validate_lesion(
            centre,
            size_mm,
            shape,
            head_scale,
            head_rotation_deg,
            unit_sphere,
        )
        centre_value = float(ellipsoid_value(centre[None, :])[0])
        if centre_inside and fully_inside and 0.88 <= max_value <= 0.985 and centre_value >= 0.25:
            return centre, max_value, outside_fraction

    raise RuntimeError("Could not sample a valid in-head boundary lesion after 20000 attempts")


def generate_rows(start_id: int, total: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    unit_sphere = sample_unit_sphere(8000, rng)

    split_labels = make_balanced_labels(total, SPLITS, SPLIT_PROBS, rng)
    noise_labels = make_balanced_labels(total, NOISE_LEVELS, NOISE_PROBS, rng)
    size_plan = make_size_plan(total, rng)
    shape_labels = make_balanced_labels(total, list(SHAPE_PROBS), list(SHAPE_PROBS.values()), rng)

    # Stratified head nuisance variation, shuffled independently.
    head_scales = np.linspace(0.9, 1.1, total) + rng.uniform(-0.0004, 0.0004, total)
    rotations = np.linspace(-15.0, 15.0, total) + rng.uniform(-0.05, 0.05, total)
    rng.shuffle(head_scales)
    rng.shuffle(rotations)

    rows: list[dict[str, object]] = []
    for idx in range(total):
        sid = start_id + idx
        size_bucket, size_min, size_max = size_plan[idx]
        size_mm = float(rng.uniform(size_min, size_max))
        shape = shape_labels[idx]
        head_scale = float(np.clip(head_scales[idx], 0.9, 1.1))
        head_rotation_deg = float(np.clip(rotations[idx], -15.0, 15.0))

        centre, max_value, outside_fraction = sample_valid_boundary_position(
            rng,
            size_mm,
            shape,
            head_scale,
            head_rotation_deg,
            unit_sphere,
        )

        bg_eps_var = float(rng.uniform(-10.0, 10.0))
        bg_sig_var = float(rng.uniform(-10.0, 10.0))
        eps_anom_var = float(rng.uniform(-15.0, 15.0))
        sig_anom_var = float(rng.uniform(-15.0, 15.0))

        rows.append(
            {
                "scenario_id": sid,
                "is_base_case": 0,
                "has_lesion": 1,
                "lesion_size_mm": round(size_mm, 4),
                "lesion_x": round(float(centre[0]), 6),
                "lesion_y": round(float(centre[1]), 6),
                "lesion_z": round(float(centre[2]), 6),
                "epsilon_variation": round(bg_eps_var, 4),
                "sigma_variation": round(bg_sig_var, 4),
                "head_scale": round(head_scale, 4),
                "head_rotation_deg": round(head_rotation_deg, 4),
                "noise_level": noise_labels[idx],
                "split": split_labels[idx],
                "group": "A_boundary_valid_extra",
                "size_bucket": size_bucket,
                "region": "boundary",
                "shape": shape,
                "epsilon_anomaly_variation": round(eps_anom_var, 4),
                "sigma_anomaly_variation": round(sig_anom_var, 4),
                "background_epsilon_variation": round(bg_eps_var, 4),
                "background_sigma_variation": round(bg_sig_var, 4),
                "lesion_center_inside_head": True,
                "lesion_fully_inside_head": True,
                "lesion_head_max_ellipsoid_value": round(max_value, 6),
                "lesion_outside_head_fraction_sampled": round(outside_fraction, 8),
                "exclude_from_localisation": False,
                "exclusion_reason": "",
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--start-id", type=int, default=SCENARIO_ID_START)
    parser.add_argument("--count", type=int, default=TOTAL_SAMPLES)
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()

    rows = generate_rows(args.start_id, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {split: sum(1 for row in rows if row["split"] == split) for split in SPLITS}
    size_counts = {bucket: sum(1 for row in rows if row["size_bucket"] == bucket) for bucket, _, _ in SIZE_BUCKETS}
    shape_counts = {shape: sum(1 for row in rows if row["shape"] == shape) for shape in SHAPE_PROBS}
    max_values = np.asarray([float(row["lesion_head_max_ellipsoid_value"]) for row in rows])

    print(f"Generated valid boundary metadata: {args.output}")
    print(f"Scenario IDs: {args.start_id}..{args.start_id + args.count - 1}")
    print(f"Rows: {len(rows)}")
    print(f"Splits: {split_counts}")
    print(f"Size buckets: {size_counts}")
    print(f"Shapes: {shape_counts}")
    print(f"Surface ellipsoid max value range: {max_values.min():.4f}..{max_values.max():.4f}")
    print("All rows are lesion-positive boundary cases with sampled lesion volumes inside the head.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
