"""
Generate stratified metadata CSV for EMI deep-learning dataset.

Design summary:
- Total samples: 1000 (scenario_id: 0..999)
- No anomaly: 300 (3 groups x 100)
- Anomaly: 700 (size/location/shape/property/measurement/noise stratified)
- Deterministic generation with per-sample seeding

Output:
- dataset_metadata.csv
"""

import csv
import math
import numpy as np

OUTPUT_FILE = "dataset_metadata.csv"
TOTAL_SAMPLES = 1000
NO_ANOMALY_SAMPLES = 300
ANOMALY_SAMPLES = 700
SCENARIO_ID_START = 1

REGIONS = ["left", "right", "deep", "boundary"]
SIZE_BUCKETS = [
    ("small", 230, 5.0, 10.0),
    ("medium", 240, 10.0, 20.0),
    ("large", 230, 20.0, 30.0),
]

FIELDNAMES = [
    "scenario_id",
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


def assign_split(scenario_id):
    if 1 <= scenario_id <= 700:
        return "train"
    if 701 <= scenario_id <= 850:
        return "val"
    return "test"


def distribute_evenly(total, n_groups):
    base = total // n_groups
    rem = total % n_groups
    counts = [base] * n_groups
    for i in range(rem):
        counts[i] += 1
    return counts


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


def make_no_anomaly_sample(scenario_id, group_name):
    np.random.seed(scenario_id)

    if group_name == "N1_baseline":
        eps_var = float(np.random.uniform(-2.0, 2.0))
        sig_var = float(np.random.uniform(-2.0, 2.0))
        noise = "low"
    elif group_name == "N2_property_variation":
        eps_var = float(np.random.uniform(-10.0, 10.0))
        sig_var = float(np.random.uniform(-10.0, 10.0))
        noise = "low"
    else:  # N3_measurement_variation
        eps_var = float(np.random.uniform(-5.0, 5.0))
        sig_var = float(np.random.uniform(-5.0, 5.0))
        noise = "medium"

    head_scale = float(np.random.uniform(0.9, 1.1))
    head_rotation_deg = float(np.random.uniform(-15.0, 15.0))

    return {
        "scenario_id": scenario_id,
        "has_lesion": 0,
        "lesion_size_mm": 0.0,
        "lesion_x": 0.0,
        "lesion_y": 0.0,
        "lesion_z": 0.0,
        "epsilon_variation": round(eps_var, 4),
        "sigma_variation": round(sig_var, 4),
        "head_scale": round(head_scale, 4),
        "head_rotation_deg": round(head_rotation_deg, 4),
        "noise_level": noise,
        "split": assign_split(scenario_id),
        "group": group_name,
        "size_bucket": "none",
        "region": "none",
        "shape": "none",
        "epsilon_anomaly_variation": 0.0,
        "sigma_anomaly_variation": 0.0,
        "background_epsilon_variation": round(eps_var, 4),
        "background_sigma_variation": round(sig_var, 4),
    }


def make_noise_schedule_for_anomaly(total):
    """50% low, 30% medium, 20% high with deterministic shuffle."""
    n_low = int(total * 0.50)
    n_medium = int(total * 0.30)
    n_high = total - n_low - n_medium

    labels = (["low"] * n_low) + (["medium"] * n_medium) + (["high"] * n_high)
    rng = np.random.RandomState(999)
    rng.shuffle(labels)
    return labels


def build_anomaly_plan():
    """
    Build deterministic anomaly bucket plan:
    - size buckets with requested counts
    - each split approximately evenly across 4 regions
    - shape in each region bucket: 70% sphere, 30% ellipsoid
    """
    items = []

    for size_name, size_count, size_min, size_max in SIZE_BUCKETS:
        region_counts = distribute_evenly(size_count, len(REGIONS))

        for region, region_count in zip(REGIONS, region_counts):
            n_sphere = int(round(region_count * 0.70))
            n_ellipsoid = region_count - n_sphere
            shapes = (["sphere"] * n_sphere) + (["ellipsoid"] * n_ellipsoid)

            # Deterministic local shuffle per bucket
            seed = abs(hash((size_name, region, region_count))) % (2**32 - 1)
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

    if len(items) != ANOMALY_SAMPLES:
        raise RuntimeError(f"Anomaly plan size mismatch: {len(items)} != {ANOMALY_SAMPLES}")

    # Global deterministic shuffle to avoid structured ordering by bucket.
    rng = np.random.RandomState(12345)
    rng.shuffle(items)
    return items


def make_anomaly_sample(scenario_id, anomaly_index, plan_item, noise_level):
    np.random.seed(scenario_id)

    size_mm = float(np.random.uniform(plan_item["size_min"], plan_item["size_max"]))
    lesion_x, lesion_y, lesion_z = random_position_by_region(plan_item["region"])

    # Lesion property variation: base +/-15%
    eps_anom_var = float(np.random.uniform(-15.0, 15.0))
    sig_anom_var = float(np.random.uniform(-15.0, 15.0))

    # Background variation: +/-10%
    bg_eps_var = float(np.random.uniform(-10.0, 10.0))
    bg_sig_var = float(np.random.uniform(-10.0, 10.0))

    head_scale = float(np.random.uniform(0.9, 1.1))
    head_rotation_deg = float(np.random.uniform(-15.0, 15.0))

    return {
        "scenario_id": scenario_id,
        "has_lesion": 1,
        "lesion_size_mm": round(size_mm, 4),
        "lesion_x": lesion_x,
        "lesion_y": lesion_y,
        "lesion_z": lesion_z,
        "epsilon_variation": round(bg_eps_var, 4),
        "sigma_variation": round(bg_sig_var, 4),
        "head_scale": round(head_scale, 4),
        "head_rotation_deg": round(head_rotation_deg, 4),
        "noise_level": noise_level,
        "split": assign_split(scenario_id),
        "group": "A_anomaly",
        "size_bucket": plan_item["size_bucket"],
        "region": plan_item["region"],
        "shape": plan_item["shape"],
        "epsilon_anomaly_variation": round(eps_anom_var, 4),
        "sigma_anomaly_variation": round(sig_anom_var, 4),
        "background_epsilon_variation": round(bg_eps_var, 4),
        "background_sigma_variation": round(bg_sig_var, 4),
    }


def main():
    rows = []

    # 300 no-anomaly samples: 3 groups x 100
    no_anomaly_groups = (
        ["N1_baseline"] * 100
        + ["N2_property_variation"] * 100
        + ["N3_measurement_variation"] * 100
    )

    for idx, sid in enumerate(range(SCENARIO_ID_START, SCENARIO_ID_START + NO_ANOMALY_SAMPLES)):
        rows.append(make_no_anomaly_sample(sid, no_anomaly_groups[idx]))

    # 700 anomaly samples with stratified plan + noise schedule
    plan = build_anomaly_plan()
    noise_schedule = make_noise_schedule_for_anomaly(ANOMALY_SAMPLES)

    for i in range(ANOMALY_SAMPLES):
        sid = SCENARIO_ID_START + NO_ANOMALY_SAMPLES + i
        rows.append(make_anomaly_sample(sid, i, plan[i], noise_schedule[i]))

    if len(rows) != TOTAL_SAMPLES:
        raise RuntimeError(f"Total rows mismatch: {len(rows)} != {TOTAL_SAMPLES}")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # Console summary
    n_healthy = sum(1 for r in rows if r["has_lesion"] == 0)
    n_anomaly = sum(1 for r in rows if r["has_lesion"] == 1)
    split_counts = {
        "train": sum(1 for r in rows if r["split"] == "train"),
        "val": sum(1 for r in rows if r["split"] == "val"),
        "test": sum(1 for r in rows if r["split"] == "test"),
    }

    print("Generated stratified metadata")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Total: {len(rows)}")
    print(f"  No anomaly: {n_healthy}")
    print(f"  Anomaly: {n_anomaly}")
    print(f"  Split: train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}")


if __name__ == "__main__":
    main()
