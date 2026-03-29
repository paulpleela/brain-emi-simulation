"""
Convert Touchstone .s16p data into ML tensors.

Supports both time-domain and frequency-domain representations, and either
diagonal-only (Sii) or full-matrix (Sij) channel layouts.

Input:
    sparams/scenario_XXX.s16p

Output:
    sparams_time/scenario_XXX_td.npz (time mode)
    sparams_time/scenario_XXX_fd.npz (freq mode)
        - signal: shape (C, F_or_T)
        - channels: channel names in deterministic order

Usage:
    python build_time_dataset.py --all
    python build_time_dataset.py --scenario 1
    python build_time_dataset.py --range 1 300
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_INPUT_DIR = "sparams"
DEFAULT_TIME_OUTPUT_DIR = "sparams_time"
DEFAULT_METADATA_FILE = "dataset_metadata.csv"
DEFAULT_N_PORTS = 16
EPSILON = 1e-8

DEFAULT_METADATA_FIELDS = [
    "scenario_id",
    "has_lesion",
    "condition_label",
    "lesion_size_mm",
    "lesion_x",
    "lesion_y",
    "lesion_z",
    "epsilon_variation",
    "sigma_variation",
    "antenna_offset",
    "coupling_thickness",
    "noise_level",
    "split",
    "group",
    "size_bucket",
    "region",
    "shape",
    "label_stratify_key",
]


def parse_scenario_id(path: str) -> int:
    name = os.path.basename(path)
    m = re.match(r"scenario_(\d+)\.s\d+p$", name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse scenario ID from filename: {name}")
    return int(m.group(1))


def load_s16p(filepath: str, n_ports: int = DEFAULT_N_PORTS) -> Tuple[np.ndarray, np.ndarray]:
    tokens: List[str] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue
            tokens.extend(line.split())

    vals_per_freq = 1 + n_ports * n_ports * 2
    if len(tokens) < vals_per_freq:
        raise ValueError(f"Insufficient data in {filepath}")
    if len(tokens) % vals_per_freq != 0:
        raise ValueError(f"Malformed touchstone data in {filepath}")

    n_freq = len(tokens) // vals_per_freq
    freqs_ghz = np.zeros(n_freq, dtype=np.float64)
    S = np.zeros((n_ports, n_ports, n_freq), dtype=np.complex128)

    for fi in range(n_freq):
        base = fi * vals_per_freq
        freqs_ghz[fi] = float(tokens[base])
        k = base + 1
        for i in range(n_ports):
            for j in range(n_ports):
                mag = float(tokens[k])
                ang_deg = float(tokens[k + 1])
                S[i, j, fi] = mag * np.exp(1j * np.deg2rad(ang_deg))
                k += 2

    if n_freq > 2:
        spacing = np.diff(freqs_ghz)
        if not np.allclose(spacing, spacing[0], rtol=1e-5, atol=1e-10):
            raise ValueError(f"Frequency samples are not uniformly spaced in {filepath}")

    return freqs_ghz, S


def extract_sii(S: np.ndarray) -> np.ndarray:
    n_ports = S.shape[0]
    return np.stack([S[i, i, :] for i in range(n_ports)], axis=0)


def extract_sij(S: np.ndarray) -> np.ndarray:
    n_ports, _, n_freq = S.shape
    return S.reshape(n_ports * n_ports, n_freq)


def complex_to_real_imag(channels_complex: np.ndarray) -> np.ndarray:
    n_channels, n_points = channels_complex.shape
    out = np.zeros((2 * n_channels, n_points), dtype=np.float32)
    out[0::2, :] = np.real(channels_complex).astype(np.float32)
    out[1::2, :] = np.imag(channels_complex).astype(np.float32)
    return out


def convert_to_time_domain(channels_complex: np.ndarray) -> np.ndarray:
    channels_td = np.fft.ifft(channels_complex, axis=1)
    return complex_to_real_imag(channels_td)


def convert_to_frequency_domain(channels_complex: np.ndarray) -> np.ndarray:
    return complex_to_real_imag(channels_complex)


def normalise_maxabs(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    return signal


def normalise_sample_2d(signal: np.ndarray) -> np.ndarray:
    mean = np.mean(signal)
    std = np.std(signal)
    if std < EPSILON:
        return signal - mean
    return (signal - mean) / std


def build_channel_names(n_ports: int) -> List[str]:
    names: List[str] = []
    for i in range(1, n_ports + 1):
        sii = f"S{i}{i}"
        names.append(f"{sii}_real")
        names.append(f"{sii}_imag")
    return names


def build_channel_names_full(n_ports: int) -> List[str]:
    names: List[str] = []
    for i in range(1, n_ports + 1):
        for j in range(1, n_ports + 1):
            sij = f"S{i}{j}"
            names.append(f"{sij}_real")
            names.append(f"{sij}_imag")
    return names


class RunningStats2D:
    def __init__(self) -> None:
        self.count: int = 0
        self.mean: Optional[np.ndarray] = None
        self.m2: Optional[np.ndarray] = None

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float64)
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.m2 = np.zeros_like(x)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.m2 is None or self.count == 0:
            raise ValueError("No samples collected for dataset-2d normalization")
        if self.count == 1:
            std = np.ones_like(self.mean, dtype=np.float64)
        else:
            var = self.m2 / (self.count - 1)
            std = np.sqrt(np.maximum(var, EPSILON))
        return self.mean.astype(np.float32), std.astype(np.float32)


def build_complex_channels(S: np.ndarray, matrix: str) -> np.ndarray:
    if matrix == "diag":
        return extract_sii(S)
    if matrix == "full":
        return extract_sij(S)
    raise ValueError(f"Unsupported matrix mode: {matrix}")


def transform_channels(channels_complex: np.ndarray, domain: str) -> np.ndarray:
    if domain == "time":
        return convert_to_time_domain(channels_complex)
    if domain == "freq":
        return convert_to_frequency_domain(channels_complex)
    raise ValueError(f"Unsupported domain: {domain}")


def build_channels(n_ports: int, matrix: str) -> List[str]:
    if matrix == "diag":
        return build_channel_names(n_ports)
    if matrix == "full":
        return build_channel_names_full(n_ports)
    raise ValueError(f"Unsupported matrix mode: {matrix}")


def apply_normalization(
    signal: np.ndarray,
    norm: str,
    dataset_mean: Optional[np.ndarray] = None,
    dataset_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    if norm == "none":
        return signal
    if norm == "maxabs":
        return normalise_maxabs(signal)
    if norm == "sample-2d":
        return normalise_sample_2d(signal)
    if norm == "dataset-2d":
        if dataset_mean is None or dataset_std is None:
            raise ValueError("dataset-2d normalization requires dataset mean/std")
        return (signal - dataset_mean) / np.maximum(dataset_std, EPSILON)
    raise ValueError(f"Unsupported normalization mode: {norm}")


def save_npz(out_path: str, signal: np.ndarray, channels: List[str]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, signal=signal, channels=np.array(channels, dtype="U32"))


def save_norm_stats(out_dir: str, mean: np.ndarray, std: np.ndarray, channels: List[str], domain: str, matrix: str) -> None:
    out_path = os.path.join(out_dir, f"normalization_{domain}_{matrix}.npz")
    np.savez_compressed(
        out_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        channels=np.array(channels, dtype="U32"),
    )


def assign_split_legacy(scenario_id: int) -> str:
    if 0 <= scenario_id <= 699:
        return "train"
    if 700 <= scenario_id <= 849:
        return "val"
    if 850 <= scenario_id <= 999:
        return "test"
    return "train"


def load_existing_metadata(path: str) -> Tuple[Dict[int, Dict[str, str]], List[str]]:
    rows: Dict[int, Dict[str, str]] = {}
    fieldnames: List[str] = []
    if not os.path.isfile(path):
        return rows, fieldnames
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            fieldnames = list(reader.fieldnames)
        for row in reader:
            sid_raw = row.get("scenario_id", "").strip()
            if not sid_raw:
                continue
            try:
                sid = int(sid_raw)
            except ValueError:
                continue
            rows[sid] = dict(row)
    return rows, fieldnames


def lesion_size_bucket(size_mm: float) -> str:
    if size_mm <= 0:
        return "none"
    if size_mm < 7:
        return "small"
    if size_mm < 13:
        return "medium"
    return "large"


def enrich_row_labels(row: Dict[str, str]) -> Dict[str, str]:
    out = dict(row)

    has_lesion = out.get("has_lesion", "0").strip()
    out["condition_label"] = "hemorrhage" if has_lesion == "1" else "no_hemorrhage"

    size_bucket = out.get("size_bucket", "").strip()
    if not size_bucket:
        try:
            lesion_size = float(out.get("lesion_size_mm", "0") or 0.0)
        except ValueError:
            lesion_size = 0.0
        out["size_bucket"] = lesion_size_bucket(lesion_size)

    if not out.get("region", "").strip():
        out["region"] = "none" if out["condition_label"] == "no_hemorrhage" else "unknown"
    if not out.get("shape", "").strip():
        out["shape"] = "none" if out["condition_label"] == "no_hemorrhage" else "unknown"

    return out


def build_stratify_key(row: Dict[str, str], stratify_cols: List[str]) -> str:
    parts: List[str] = []
    for col in stratify_cols:
        parts.append((row.get(col, "unknown") or "unknown").strip().lower())
    return "|".join(parts)


def stable_sort_key(scenario_id: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{scenario_id}".encode("ascii")).hexdigest()


def allocate_counts(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = max(1, n_test)

    total = n_train + n_val + n_test
    while total > n:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        total = n_train + n_val + n_test

    while total < n:
        n_train += 1
        total += 1

    return n_train, n_val, n_test


def assign_splits_stratified(
    rows_by_id: Dict[int, Dict[str, str]],
    stratify_cols: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    strata: Dict[str, List[int]] = {}
    for sid, row in rows_by_id.items():
        key = build_stratify_key(row, stratify_cols)
        row["label_stratify_key"] = key
        strata.setdefault(key, []).append(sid)

    for key, sids in strata.items():
        ordered = sorted(sids, key=lambda sid: stable_sort_key(sid, seed))
        n_train, n_val, n_test = allocate_counts(len(ordered), train_ratio, val_ratio)

        train_ids = ordered[:n_train]
        val_ids = ordered[n_train : n_train + n_val]
        test_ids = ordered[n_train + n_val : n_train + n_val + n_test]

        for sid in train_ids:
            rows_by_id[sid]["split"] = "train"
        for sid in val_ids:
            rows_by_id[sid]["split"] = "val"
        for sid in test_ids:
            rows_by_id[sid]["split"] = "test"


def assign_splits_legacy(rows_by_id: Dict[int, Dict[str, str]]) -> None:
    for sid, row in rows_by_id.items():
        row["split"] = assign_split_legacy(sid)


def merge_metadata_row(
    scenario_id: int,
    existing_row: Dict[str, str] | None,
    epsilon_variation: float,
    sigma_variation: float,
    antenna_offset: float,
    coupling_thickness: float,
    noise_level: float,
) -> Dict[str, str]:
    row = dict(existing_row) if existing_row else {}
    row["scenario_id"] = str(scenario_id)
    row["has_lesion"] = row.get("has_lesion", "0")
    row["lesion_size_mm"] = row.get("lesion_size_mm", "0")
    row["lesion_x"] = row.get("lesion_x", "0")
    row["lesion_y"] = row.get("lesion_y", "0")
    row["lesion_z"] = row.get("lesion_z", "0")
    row["epsilon_variation"] = str(epsilon_variation)
    row["sigma_variation"] = str(sigma_variation)
    row["antenna_offset"] = str(antenna_offset)
    row["coupling_thickness"] = str(coupling_thickness)
    row["noise_level"] = str(noise_level)
    row.setdefault("split", "")
    return enrich_row_labels(row)


def save_metadata(path: str, rows_by_id: Dict[int, Dict[str, str]], existing_fieldnames: List[str]) -> None:
    merged_fields: List[str] = []
    for k in DEFAULT_METADATA_FIELDS + existing_fieldnames:
        if k and k not in merged_fields:
            merged_fields.append(k)
    for row in rows_by_id.values():
        for k in row.keys():
            if k not in merged_fields:
                merged_fields.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        for sid in sorted(rows_by_id.keys()):
            row = rows_by_id[sid]
            writer.writerow({k: row.get(k, "") for k in merged_fields})


def should_use_for_stats(
    sid: int,
    rows_by_id: Dict[int, Dict[str, str]],
    norm_fit_split: str,
) -> bool:
    if norm_fit_split == "all":
        return True
    split = (rows_by_id.get(sid, {}).get("split", "") or "").strip().lower()
    return split == norm_fit_split


def select_files(input_dir: str, scenario: int | None, process_all: bool, range_vals: Tuple[int, int] | None) -> List[str]:
    all_files = sorted(glob.glob(os.path.join(input_dir, "scenario_*.s*p")))
    selected: List[str] = []
    for path in all_files:
        try:
            sid = parse_scenario_id(path)
        except ValueError:
            continue
        if scenario is not None and sid == scenario:
            selected.append(path)
        elif process_all:
            selected.append(path)
        elif range_vals is not None and range_vals[0] <= sid <= range_vals[1]:
            selected.append(path)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .s16p into ML tensors with configurable domain/channel layout")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=int)
    group.add_argument("--all", action="store_true")
    group.add_argument("--range", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_TIME_OUTPUT_DIR)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--n-ports", type=int, default=DEFAULT_N_PORTS)
    parser.add_argument("--domain", choices=["time", "freq"], default="time")
    parser.add_argument("--matrix", choices=["diag", "full"], default="diag")
    parser.add_argument("--norm", choices=["none", "maxabs", "sample-2d", "dataset-2d"], default="maxabs")
    parser.add_argument("--split-mode", choices=["legacy-id", "stratified"], default="stratified")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--stratify-cols",
        type=str,
        default="condition_label,size_bucket,region,shape,noise_level",
        help="Comma-separated metadata columns used to preserve label diversity across splits",
    )
    parser.add_argument(
        "--norm-fit-split",
        choices=["train", "val", "test", "all"],
        default="train",
        help="Which split to use for fitting dataset-2d normalization statistics",
    )
    parser.add_argument("--epsilon-variation", type=float, default=0.0)
    parser.add_argument("--sigma-variation", type=float, default=0.0)
    parser.add_argument("--antenna-offset", type=float, default=0.0)
    parser.add_argument("--coupling-thickness", type=float, default=0.0)
    parser.add_argument("--noise-level", type=float, default=0.0)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Invalid split ratios: require 0 < train_ratio and train_ratio + val_ratio < 1")

    files = select_files(args.input_dir, args.scenario, args.all, tuple(args.range) if args.range else None)
    if not files:
        print("No matching .s16p files found")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    rows, existing_fieldnames = load_existing_metadata(args.metadata)

    # Ensure selected scenarios exist in metadata and update augmentation/noise values.
    for path in files:
        sid = parse_scenario_id(path)
        rows[sid] = merge_metadata_row(
            sid,
            rows.get(sid),
            args.epsilon_variation,
            args.sigma_variation,
            args.antenna_offset,
            args.coupling_thickness,
            args.noise_level,
        )

    # Ensure label fields are present on all rows before split assignment.
    for sid in list(rows.keys()):
        rows[sid] = enrich_row_labels(rows[sid])

    if args.split_mode == "legacy-id":
        assign_splits_legacy(rows)
    else:
        stratify_cols = [c.strip() for c in args.stratify_cols.split(",") if c.strip()]
        if not stratify_cols:
            raise ValueError("--stratify-cols must include at least one column")
        assign_splits_stratified(rows, stratify_cols, args.train_ratio, args.val_ratio, args.split_seed)

    channels = build_channels(args.n_ports, args.matrix)
    dataset_mean: Optional[np.ndarray] = None
    dataset_std: Optional[np.ndarray] = None

    if args.norm == "dataset-2d":
        stats = RunningStats2D()
        fit_count = 0
        for path in files:
            try:
                sid = parse_scenario_id(path)
                if not should_use_for_stats(sid, rows, args.norm_fit_split):
                    continue
                _, S = load_s16p(path, n_ports=args.n_ports)
                channels_complex = build_complex_channels(S, args.matrix)
                signal = transform_channels(channels_complex, args.domain)
                stats.update(signal)
                fit_count += 1
            except Exception as exc:
                print(f"  WARN: skipped stats update for {os.path.basename(path)} -> {exc}")

        if fit_count == 0:
            print(f"  WARN: no samples found in split '{args.norm_fit_split}' for stats; falling back to all selected files")
            for path in files:
                try:
                    _, S = load_s16p(path, n_ports=args.n_ports)
                    channels_complex = build_complex_channels(S, args.matrix)
                    signal = transform_channels(channels_complex, args.domain)
                    stats.update(signal)
                except Exception as exc:
                    print(f"  WARN: skipped fallback stats update for {os.path.basename(path)} -> {exc}")

        dataset_mean, dataset_std = stats.finalize()
        save_norm_stats(args.output_dir, dataset_mean, dataset_std, channels, args.domain, args.matrix)
        print(
            f"Saved dataset-2d normalization stats: "
            f"{os.path.join(args.output_dir, f'normalization_{args.domain}_{args.matrix}.npz')}"
        )
        print(f"Normalization fit split: {args.norm_fit_split}")

    ok = 0
    fail = 0
    print(f"Found {len(files)} scenario file(s) to process")

    for path in files:
        try:
            sid = parse_scenario_id(path)
            _, S = load_s16p(path, n_ports=args.n_ports)
            channels_complex = build_complex_channels(S, args.matrix)
            signal = transform_channels(channels_complex, args.domain)
            signal = apply_normalization(signal, args.norm, dataset_mean, dataset_std)
            out_suffix = "td" if args.domain == "time" else "fd"
            out_path = os.path.join(args.output_dir, f"scenario_{sid:03d}_{out_suffix}.npz")
            save_npz(out_path, signal, channels)
            ok += 1
            print(f"  Saved: {out_path} | signal shape={signal.shape}")
        except Exception as exc:
            fail += 1
            print(f"  ERROR: {os.path.basename(path)} -> {exc}")

    save_metadata(args.metadata, rows, existing_fieldnames)

    split_counts = {"train": 0, "val": 0, "test": 0}
    for row in rows.values():
        split = (row.get("split", "") or "").strip().lower()
        if split in split_counts:
            split_counts[split] += 1

    print("\n" + "=" * 60)
    print(f"Done: {ok} succeeded, {fail} failed")
    print(f"Representation: domain={args.domain}, matrix={args.matrix}, norm={args.norm}")
    print(f"Split mode: {args.split_mode} | counts: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")
    print(f"Tensor files in: {args.output_dir}/")
    print(f"Metadata: {args.metadata}")
    print("=" * 60)


if __name__ == "__main__":
    main()
