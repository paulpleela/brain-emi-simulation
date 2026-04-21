"""
Generate stratified metadata CSV for EMI deep-learning dataset.

Design goals:
- Total samples: 1000 (scenario_id: 1..1000)
- Label balance preserved globally (300 healthy, 700 anomaly)
- Split assignment by deterministic shuffle (not contiguous scenario ID ranges)
- Split balance for labels (30% healthy, 70% anomaly in each split)
- Analysis variables are independently distributed in each split/label subgroup:
  - head_scale in [0.9, 1.1]
  - head_rotation_deg in [-15, 15]
  - noise_level in 50/30/20 (low/medium/high)
- Explicit base case at scenario_id=1 for controlled comparisons

Output:
- dataset_metadata.csv
"""

import csv
import math
import numpy as np

OUTPUT_FILE = "dataset_metadata.csv"
TOTAL_SAMPLES = 1000
SCENARIO_ID_START = 1

SPLIT_COUNTS = {
    "train": 700,
    "val": 150,
    "test": 150,
}

HEALTHY_COUNTS = {
    "train": 210,
    "val": 45,
    "test": 45,
}

REGIONS = ["left", "right", "deep", "boundary"]
HEALTHY_GROUPS = ["N1_baseline", "N2_property_variation", "N3_noise_variation"]
SIZE_BUCKETS = [
    ("small", 230, 5.0, 10.0),
    ("medium", 240, 10.0, 20.0),
    ("large", 230, 20.0, 30.0),
]

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
]


def distribute_evenly(total, n_groups):
    base = total // n_groups
    rem = total % n_groups
    counts = [base] * n_groups
    for i in range(rem):
        counts[i] += 1
    return counts


def make_noise_schedule(total, seed):
    """50% low, 30% medium, 20% high with deterministic shuffle."""
    n_low = int(total * 0.50)
    n_medium = int(total * 0.30)
    n_high = total - n_low - n_medium

    labels = (["low"] * n_low) + (["medium"] * n_medium) + (["high"] * n_high)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)
    return labels


def make_stratified_uniform(total, low, high, seed):
    """Deterministic near-uniform coverage across [low, high]."""
    if total <= 0:
        return []

    rng = np.random.RandomState(seed)
    u = (np.arange(total) + rng.uniform(0.0, 1.0, size=total)) / total
    vals = low + (high - low) * u
    rng.shuffle(vals)
    return [round(float(v), 4) for v in vals]


def random_position_by_region(region):
    """
    Return lesion position (x, y, z) in meters relative to head center.
    Values are constrained to realistic offsets for each region.
    """
    if region == "left":
        x = np.random.uniform(-0.050, -0.015)
        y = np.random.uniform(-0.030, 0.030)
        z = np.random.uniform(-0.070, 0.070)
    elif region == "right":
        x = np.random.uniform(0.015, 0.050)
        y = np.random.uniform(-0.030, 0.030)
        z = np.random.uniform(-0.070, 0.070)
    elif region == "deep":
        x = np.random.uniform(-0.012, 0.012)
        y = np.random.uniform(-0.012, 0.012)
        z = np.random.uniform(-0.050, 0.050)
    else:  # boundary
        theta = np.random.uniform(0, 2 * math.pi)
        r = np.random.uniform(0.040, 0.055)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = np.random.uniform(-0.080, 0.080)

    return float(round(x, 6)), float(round(y, 6)), float(round(z, 6))


def make_no_anomaly_sample(scenario_id, split, group_name, noise_level, head_scale, head_rotation_deg, is_base_case=False):
    np.random.seed(scenario_id)

    if is_base_case:
        eps_var = 0.0
        sig_var = 0.0
        noise_level = "none"
        head_scale = 1.0
        head_rotation_deg = 0.0
    elif group_name == "N1_baseline":
        eps_var = float(np.random.uniform(-2.0, 2.0))
        sig_var = float(np.random.uniform(-2.0, 2.0))
    elif group_name == "N2_property_variation":
        eps_var = float(np.random.uniform(-10.0, 10.0))
        sig_var = float(np.random.uniform(-10.0, 10.0))
    else:  # N3_noise_variation
        eps_var = float(np.random.uniform(-5.0, 5.0))
        sig_var = float(np.random.uniform(-5.0, 5.0))

    return {
        "scenario_id": scenario_id,
        "is_base_case": 1 if is_base_case else 0,
        "has_lesion": 0,
        "lesion_size_mm": 0.0,
        "lesion_x": 0.0,
        "lesion_y": 0.0,
        "lesion_z": 0.0,
        "epsilon_variation": round(eps_var, 4),
        "sigma_variation": round(sig_var, 4),
        "head_scale": round(float(head_scale), 4),
        "head_rotation_deg": round(float(head_rotation_deg), 4),
        "noise_level": noise_level,
        "split": split,
        "group": group_name,
        "size_bucket": "none",
        "region": "none",
        "shape": "none",
        "epsilon_anomaly_variation": 0.0,
        "sigma_anomaly_variation": 0.0,
        "background_epsilon_variation": round(eps_var, 4),
        "background_sigma_variation": round(sig_var, 4),
    }


def build_anomaly_plan():
    """
    Build deterministic anomaly bucket plan:
    - size buckets with requested counts
    - each split approximately evenly across 4 regions
    - shape in each region bucket: 70% sphere, 30% ellipsoid
    """
    items = []

    for size_idx, (size_name, size_count, size_min, size_max) in enumerate(SIZE_BUCKETS):
        region_counts = distribute_evenly(size_count, len(REGIONS))

        for region_idx, (region, region_count) in enumerate(zip(REGIONS, region_counts)):
            n_sphere = int(round(region_count * 0.70))
            n_ellipsoid = region_count - n_sphere
            shapes = (["sphere"] * n_sphere) + (["ellipsoid"] * n_ellipsoid)

            seed = 1000 + size_idx * 100 + region_idx * 10 + region_count
            rng = np.random.RandomState(seed)
            rng.shuffle(shapes)

            for shape in shapes:
                items.append(
                    {
                        "size_bucket": size_name,
                        "size_min": size_min,
                        "size_max": size_max,
                        "region": region,
                        "shape": shape,
                    }
                )

    if len(items) != 700:
        raise RuntimeError(f"Anomaly plan size mismatch: {len(items)} != 700")

    rng = np.random.RandomState(12345)
    rng.shuffle(items)
    return items


def make_anomaly_sample(scenario_id, split, plan_item, noise_level, head_scale, head_rotation_deg):
    np.random.seed(scenario_id)

    size_mm = float(np.random.uniform(plan_item["size_min"], plan_item["size_max"]))
    lesion_x, lesion_y, lesion_z = random_position_by_region(plan_item["region"])

    eps_anom_var = float(np.random.uniform(-15.0, 15.0))
    sig_anom_var = float(np.random.uniform(-15.0, 15.0))

    bg_eps_var = float(np.random.uniform(-10.0, 10.0))
    bg_sig_var = float(np.random.uniform(-10.0, 10.0))

    return {
        "scenario_id": scenario_id,
        "is_base_case": 0,
        "has_lesion": 1,
        "lesion_size_mm": round(size_mm, 4),
        "lesion_x": lesion_x,
        "lesion_y": lesion_y,
        "lesion_z": lesion_z,
        "epsilon_variation": round(bg_eps_var, 4),
        "sigma_variation": round(bg_sig_var, 4),
        "head_scale": round(float(head_scale), 4),
        "head_rotation_deg": round(float(head_rotation_deg), 4),
        "noise_level": noise_level,
        "split": split,
        "group": "A_anomaly",
        "size_bucket": plan_item["size_bucket"],
        "region": plan_item["region"],
        "shape": plan_item["shape"],
        "epsilon_anomaly_variation": round(eps_anom_var, 4),
        "sigma_anomaly_variation": round(sig_anom_var, 4),
        "background_epsilon_variation": round(bg_eps_var, 4),
        "background_sigma_variation": round(bg_sig_var, 4),
    }


def validate_analysis_coverage(rows):
    expected_splits = ["train", "val", "test"]
    required_noise = {"low", "medium", "high"}
    allowed_noise = {"none", "low", "medium", "high"}

    for split in expected_splits:
        subset = [r for r in rows if r["split"] == split]
        if not subset:
            raise RuntimeError(f"Missing split: {split}")

        noise_values = {str(r["noise_level"]).strip().lower() for r in subset}
        if not noise_values.issubset(allowed_noise):
            raise RuntimeError(
                f"Unexpected noise_level values in split={split}: {sorted(noise_values)}"
            )
        if not required_noise.issubset(noise_values):
            raise RuntimeError(
                f"noise_level not fully represented in split={split}: "
                f"required {sorted(required_noise)}, got {sorted(noise_values)}"
            )

        scales = np.array([float(r["head_scale"]) for r in subset], dtype=np.float64)
        rots = np.array([float(r["head_rotation_deg"]) for r in subset], dtype=np.float64)

        if np.any((scales < 0.9) | (scales > 1.1)):
            raise RuntimeError(f"head_scale out of range in split={split}")
        if np.any((rots < -15.0) | (rots > 15.0)):
            raise RuntimeError(f"head_rotation_deg out of range in split={split}")

        if float(scales.max() - scales.min()) <= 0.0:
            raise RuntimeError(f"head_scale has no variation in split={split}")
        if float(rots.max() - rots.min()) <= 0.0:
            raise RuntimeError(f"head_rotation_deg has no variation in split={split}")

    base_rows = [r for r in rows if int(r["is_base_case"]) == 1]
    if len(base_rows) != 1:
        raise RuntimeError(f"Expected exactly one base case, found {len(base_rows)}")
    base = base_rows[0]
    if str(base["noise_level"]).strip().lower() != "none":
        raise RuntimeError("Base case must use noise_level=none")

    none_rows = [r for r in rows if str(r["noise_level"]).strip().lower() == "none"]
    if len(none_rows) != 1:
        raise RuntimeError(f"Expected exactly one noise_level=none row, found {len(none_rows)}")
    if int(none_rows[0]["is_base_case"]) != 1:
        raise RuntimeError("noise_level=none is reserved for base case only")

    all_scales = np.array([float(r["head_scale"]) for r in rows], dtype=np.float64)
    all_rots = np.array([float(r["head_rotation_deg"]) for r in rows], dtype=np.float64)
    all_noise = {str(r["noise_level"]).strip().lower() for r in rows}

    if not all_noise.issubset(allowed_noise):
        raise RuntimeError("Unexpected noise_level values found globally")
    if not required_noise.issubset(all_noise):
        raise RuntimeError("noise_level coverage is incomplete globally for low/medium/high")
    if float(all_scales.max() - all_scales.min()) <= 0.0:
        raise RuntimeError("head_scale has no global variation")
    if float(all_rots.max() - all_rots.min()) <= 0.0:
        raise RuntimeError("head_rotation_deg has no global variation")


def main():
    rows = []

    all_ids = list(range(SCENARIO_ID_START, SCENARIO_ID_START + TOTAL_SAMPLES))
    rng = np.random.RandomState(20260421)
    base_case_id = 1

    split_to_ids = {"train": [], "val": [], "test": []}
    shuffled = [sid for sid in all_ids if sid != base_case_id]
    rng.shuffle(shuffled)

    split_to_ids["train"] = [base_case_id] + shuffled[: SPLIT_COUNTS["train"] - 1]
    cursor = SPLIT_COUNTS["train"] - 1
    split_to_ids["val"] = shuffled[cursor : cursor + SPLIT_COUNTS["val"]]
    cursor += SPLIT_COUNTS["val"]
    split_to_ids["test"] = shuffled[cursor : cursor + SPLIT_COUNTS["test"]]

    if any(len(split_to_ids[k]) != SPLIT_COUNTS[k] for k in SPLIT_COUNTS):
        raise RuntimeError("Split assignment size mismatch")

    healthy_ids_by_split = {}

    for split, ids in split_to_ids.items():
        target = HEALTHY_COUNTS[split]
        picked = []
        available = ids.copy()

        if split == "train" and base_case_id in available:
            picked.append(base_case_id)
            available.remove(base_case_id)

        need = target - len(picked)
        if need > 0:
            sampled = rng.choice(np.array(available, dtype=int), size=need, replace=False).tolist()
            picked.extend(sampled)

        healthy_ids_by_split[split] = set(picked)

    healthy_group_labels_by_split = {}
    for idx, split in enumerate(["train", "val", "test"]):
        n_healthy = HEALTHY_COUNTS[split]
        counts = distribute_evenly(n_healthy, len(HEALTHY_GROUPS))
        labels = []
        for group_name, count in zip(HEALTHY_GROUPS, counts):
            labels.extend([group_name] * count)
        rg = np.random.RandomState(3000 + idx)
        rg.shuffle(labels)
        healthy_group_labels_by_split[split] = labels

    anomaly_plan = build_anomaly_plan()
    anomaly_cursor = 0

    for split_idx, split in enumerate(["train", "val", "test"]):
        ids = split_to_ids[split]
        healthy_ids = sorted(healthy_ids_by_split[split])
        anomaly_ids = sorted(set(ids) - healthy_ids_by_split[split])

        n_healthy = len(healthy_ids)
        n_anomaly = len(anomaly_ids)

        healthy_noise = make_noise_schedule(n_healthy, seed=4000 + split_idx)
        healthy_scale = make_stratified_uniform(n_healthy, 0.9, 1.1, seed=5000 + split_idx)
        healthy_rot = make_stratified_uniform(n_healthy, -15.0, 15.0, seed=6000 + split_idx)

        for i, sid in enumerate(healthy_ids):
            is_base = sid == base_case_id
            group_name = "N1_baseline" if is_base else healthy_group_labels_by_split[split][i]
            rows.append(
                make_no_anomaly_sample(
                    scenario_id=sid,
                    split=split,
                    group_name=group_name,
                    noise_level=healthy_noise[i],
                    head_scale=healthy_scale[i],
                    head_rotation_deg=healthy_rot[i],
                    is_base_case=is_base,
                )
            )

        anomaly_items = anomaly_plan[anomaly_cursor : anomaly_cursor + n_anomaly]
        anomaly_cursor += n_anomaly

        anomaly_noise = make_noise_schedule(n_anomaly, seed=7000 + split_idx)
        anomaly_scale = make_stratified_uniform(n_anomaly, 0.9, 1.1, seed=8000 + split_idx)
        anomaly_rot = make_stratified_uniform(n_anomaly, -15.0, 15.0, seed=9000 + split_idx)

        for i, sid in enumerate(anomaly_ids):
            rows.append(
                make_anomaly_sample(
                    scenario_id=sid,
                    split=split,
                    plan_item=anomaly_items[i],
                    noise_level=anomaly_noise[i],
                    head_scale=anomaly_scale[i],
                    head_rotation_deg=anomaly_rot[i],
                )
            )

    if len(rows) != TOTAL_SAMPLES:
        raise RuntimeError(f"Total rows mismatch: {len(rows)} != {TOTAL_SAMPLES}")

    rows.sort(key=lambda r: int(r["scenario_id"]))
    validate_analysis_coverage(rows)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_healthy = sum(1 for r in rows if r["has_lesion"] == 0)
    n_anomaly = sum(1 for r in rows if r["has_lesion"] == 1)
    split_counts = {
        "train": sum(1 for r in rows if r["split"] == "train"),
        "val": sum(1 for r in rows if r["split"] == "val"),
        "test": sum(1 for r in rows if r["split"] == "test"),
    }

    print("Generated analysis-ready metadata")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Total: {len(rows)}")
    print(f"  No anomaly: {n_healthy}")
    print(f"  Anomaly: {n_anomaly}")
    print(f"  Split: train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}")
    print("  Base case: scenario_id=1 (is_base_case=1)")


if __name__ == "__main__":
    main()
