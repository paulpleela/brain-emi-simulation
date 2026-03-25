"""
Convert Touchstone .s16p frequency-domain data to time-domain tensors for ML.

Input:
  sparams/scenario_XXX.s16p

Output:
  sparams_time/scenario_XXX_td.npz
    - signal: shape (32, T)
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
import os
import re
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_INPUT_DIR = "sparams"
DEFAULT_TIME_OUTPUT_DIR = "sparams_time"
DEFAULT_METADATA_FILE = "dataset_metadata.csv"
DEFAULT_N_PORTS = 16

METADATA_FIELDS = [
    "scenario_id",
    "has_lesion",
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


def convert_to_time_domain(sii: np.ndarray) -> np.ndarray:
    n_ports, _ = sii.shape
    channels: List[np.ndarray] = []
    for i in range(n_ports):
        s_t = np.fft.ifft(sii[i, :])
        channels.append(np.real(s_t))
        channels.append(np.imag(s_t))
    return np.stack(channels, axis=0).astype(np.float32)


def normalise_signal(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    return signal


def build_channel_names(n_ports: int) -> List[str]:
    names: List[str] = []
    for i in range(1, n_ports + 1):
        sii = f"S{i}{i}"
        names.append(f"{sii}_real")
        names.append(f"{sii}_imag")
    return names


def save_npz(out_path: str, signal: np.ndarray, channels: List[str]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, signal=signal, channels=np.array(channels, dtype="U32"))


def assign_split(scenario_id: int) -> str:
    if 0 <= scenario_id <= 699:
        return "train"
    if 700 <= scenario_id <= 849:
        return "val"
    if 850 <= scenario_id <= 999:
        return "test"
    return "train"


def load_existing_metadata(path: str) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    if not os.path.isfile(path):
        return rows
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid_raw = row.get("scenario_id", "").strip()
            if not sid_raw:
                continue
            try:
                sid = int(sid_raw)
            except ValueError:
                continue
            rows[sid] = dict(row)
    return rows


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
    row["split"] = assign_split(scenario_id)
    return row


def save_metadata(path: str, rows_by_id: Dict[int, Dict[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for sid in sorted(rows_by_id.keys()):
            row = rows_by_id[sid]
            writer.writerow({k: row.get(k, "") for k in METADATA_FIELDS})


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
    parser = argparse.ArgumentParser(description="Convert .s16p to normalized time-domain ML tensors (Sii only)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=int)
    group.add_argument("--all", action="store_true")
    group.add_argument("--range", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_TIME_OUTPUT_DIR)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--n-ports", type=int, default=DEFAULT_N_PORTS)
    parser.add_argument("--epsilon-variation", type=float, default=0.0)
    parser.add_argument("--sigma-variation", type=float, default=0.0)
    parser.add_argument("--antenna-offset", type=float, default=0.0)
    parser.add_argument("--coupling-thickness", type=float, default=0.0)
    parser.add_argument("--noise-level", type=float, default=0.0)
    args = parser.parse_args()

    files = select_files(args.input_dir, args.scenario, args.all, tuple(args.range) if args.range else None)
    if not files:
        print("No matching .s16p files found")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_existing_metadata(args.metadata)

    ok = 0
    fail = 0
    print(f"Found {len(files)} scenario file(s) to process")

    for path in files:
        try:
            sid = parse_scenario_id(path)
            _, S = load_s16p(path, n_ports=args.n_ports)
            sii = extract_sii(S)
            signal = normalise_signal(convert_to_time_domain(sii))
            channels = build_channel_names(args.n_ports)
            out_path = os.path.join(args.output_dir, f"scenario_{sid:03d}_td.npz")
            save_npz(out_path, signal, channels)
            rows[sid] = merge_metadata_row(
                sid,
                rows.get(sid),
                args.epsilon_variation,
                args.sigma_variation,
                args.antenna_offset,
                args.coupling_thickness,
                args.noise_level,
            )
            ok += 1
            print(f"  Saved: {out_path} | signal shape={signal.shape}")
        except Exception as exc:
            fail += 1
            print(f"  ERROR: {os.path.basename(path)} -> {exc}")

    save_metadata(args.metadata, rows)

    print("\n" + "=" * 60)
    print(f"Done: {ok} succeeded, {fail} failed")
    print(f"Time-domain files in: {args.output_dir}/")
    print(f"Metadata: {args.metadata}")
    print("=" * 60)


if __name__ == "__main__":
    main()
