"""
Build frequency-domain training tensors from Touchstone .s16p files.

Active mode is fixed to:
- full S-matrix (Sij for all 16x16 ports)
- frequency-domain features (real/imag channels)

Outputs:
- fd_tensors/scenario_XXX_fd.npz           (signal shape: 512 x F)
- fd_tensors/normalization_freq_full.npz   (mean/std shape: 512 x F)

Normalization workflow:
1) Fit stats on TRAIN split only (from dataset_metadata.csv)
2) Apply same stats to every processed scenario (train/val/test)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_INPUT_DIR = "sparams"
DEFAULT_OUTPUT_DIR = "fd_tensors"
DEFAULT_METADATA_FILE = "dataset_metadata.csv"
DEFAULT_STATS_FILE = "normalization_freq_full.npz"
N_PORTS = 16
EPSILON = 1e-8


def parse_scenario_id(path: str) -> int:
    name = os.path.basename(path)
    match = re.match(r"scenario_(\d+)\.s\d+p$", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse scenario ID from filename: {name}")
    return int(match.group(1))


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


def load_s16p(filepath: str, n_ports: int = N_PORTS) -> Tuple[np.ndarray, np.ndarray]:
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

    return freqs_ghz, S


def build_s16p(scenario_id: int, input_dir: str = DEFAULT_INPUT_DIR) -> str:
    """Return expected .s16p path for one scenario.

    Extraction itself is handled by build_s16p.py in the simulation pipeline.
    """
    return os.path.join(input_dir, f"scenario_{scenario_id:03d}.s16p")


def extract_full_smatrix(S: np.ndarray) -> np.ndarray:
    """Flatten S(i,j,f) to channels x frequency in deterministic row-major order."""
    n_ports, _, n_freq = S.shape
    if n_ports != N_PORTS:
        raise ValueError(f"Expected {N_PORTS} ports, got {n_ports}")
    return S.reshape(n_ports * n_ports, n_freq)


def build_frequency_tensor(full_smatrix: np.ndarray) -> np.ndarray:
    """Convert complex Sij channels to real/imag tensor with shape (512, F)."""
    n_channels, n_freq = full_smatrix.shape
    out = np.zeros((2 * n_channels, n_freq), dtype=np.float32)
    out[0::2, :] = np.real(full_smatrix).astype(np.float32)
    out[1::2, :] = np.imag(full_smatrix).astype(np.float32)
    return out


def build_channel_names() -> List[str]:
    names: List[str] = []
    for i in range(1, N_PORTS + 1):
        for j in range(1, N_PORTS + 1):
            sij = f"S{i}{j}"
            names.append(f"{sij}_real")
            names.append(f"{sij}_imag")
    return names


class RunningStats2D:
    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, x: np.ndarray) -> None:
        x64 = x.astype(np.float64)
        if self.mean is None:
            self.mean = np.zeros_like(x64)
            self.m2 = np.zeros_like(x64)

        self.count += 1
        delta = x64 - self.mean
        self.mean += delta / self.count
        delta2 = x64 - self.mean
        self.m2 += delta * delta2

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.m2 is None or self.count == 0:
            raise ValueError("No samples collected for normalization")

        if self.count == 1:
            std = np.ones_like(self.mean, dtype=np.float64)
        else:
            var = self.m2 / (self.count - 1)
            std = np.sqrt(np.maximum(var, EPSILON))

        return self.mean.astype(np.float32), std.astype(np.float32)


def read_split_map(metadata_path: str) -> Dict[int, str]:
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    split_map: Dict[int, str] = {}
    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "scenario_id" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError("Metadata must include 'scenario_id' and 'split' columns")

        for row in reader:
            sid_raw = (row.get("scenario_id") or "").strip()
            if not sid_raw:
                continue
            try:
                sid = int(sid_raw)
            except ValueError:
                continue
            split_map[sid] = (row.get("split") or "").strip().lower()

    return split_map


def fit_normalization_stats(train_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    stats = RunningStats2D()
    for path in train_paths:
        _, S = load_s16p(path, n_ports=N_PORTS)
        full = extract_full_smatrix(S)
        signal = build_frequency_tensor(full)
        stats.update(signal)
    return stats.finalize()


def apply_normalization(signal: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((signal - mean) / np.maximum(std, EPSILON)).astype(np.float32)


def save_npz(out_path: str, signal: np.ndarray, channels: List[str]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, signal=signal.astype(np.float32), channels=np.array(channels, dtype="U32"))


def save_normalization_stats(stats_path: str, mean: np.ndarray, std: np.ndarray, channels: List[str]) -> None:
    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    np.savez_compressed(
        stats_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        channels=np.array(channels, dtype="U32"),
    )


def load_normalization_stats(stats_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"Normalization stats file not found: {stats_path}")

    data = np.load(stats_path, allow_pickle=True)
    if "mean" not in data or "std" not in data or "channels" not in data:
        raise ValueError(f"Stats file missing required keys: {stats_path}")

    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    channels = [str(c) for c in data["channels"].tolist()]
    return mean, std, channels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frequency-domain full-S training tensors from .s16p files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=int)
    group.add_argument("--all", action="store_true")
    group.add_argument("--range", type=int, nargs=2, metavar=("START", "END"))

    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--fit-stats", action="store_true", help="Fit train-only normalization stats before applying")
    parser.add_argument("--fit-only", action="store_true", help="Fit and save stats, then exit without writing scenario tensors")
    args = parser.parse_args()

    files = select_files(args.input_dir, args.scenario, args.all, tuple(args.range) if args.range else None)
    if not files:
        print("No matching .s16p files found")
        return

    channels = build_channel_names()
    stats_path = os.path.join(args.output_dir, DEFAULT_STATS_FILE)

    if args.fit_stats or not os.path.isfile(stats_path):
        split_map = read_split_map(args.metadata)
        train_paths: List[str] = []
        for sid, split in split_map.items():
            if split != "train":
                continue
            path = build_s16p(sid, input_dir=args.input_dir)
            if os.path.isfile(path):
                train_paths.append(path)

        if not train_paths:
            raise ValueError("No train .s16p files found to fit normalization stats")

        mean, std = fit_normalization_stats(train_paths)
        save_normalization_stats(stats_path, mean, std, channels)
        print(f"Saved train-only normalization stats: {stats_path} | train files: {len(train_paths)}")

    if args.fit_only:
        return

    mean, std, stats_channels = load_normalization_stats(stats_path)
    if stats_channels != channels:
        raise ValueError("Channel ordering in stats file does not match expected frequency full-S ordering")

    ok = 0
    fail = 0
    print(f"Processing {len(files)} scenario file(s)")

    for path in files:
        try:
            sid = parse_scenario_id(path)
            _, S = load_s16p(path, n_ports=N_PORTS)
            full = extract_full_smatrix(S)
            signal = build_frequency_tensor(full)
            signal = apply_normalization(signal, mean, std)

            if signal.shape[0] != 2 * N_PORTS * N_PORTS:
                raise ValueError(f"Unexpected tensor channel count: {signal.shape}")

            out_path = os.path.join(args.output_dir, f"scenario_{sid:03d}_fd.npz")
            save_npz(out_path, signal, channels)
            ok += 1
            print(f"  Saved: {out_path} | shape={signal.shape}")
        except Exception as exc:
            fail += 1
            print(f"  ERROR: {os.path.basename(path)} -> {exc}")

    print("\n" + "=" * 60)
    print(f"Done: {ok} succeeded, {fail} failed")
    print(f"Outputs: {args.output_dir}/scenario_XXX_fd.npz")
    print(f"Stats: {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
