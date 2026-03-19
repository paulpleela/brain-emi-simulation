"""Compare S11 dip uniformity across ports for one or more Touchstone files.

Usage examples:
    python compare_s11_uniformity.py sparams/scenario_001.s16p
    python compare_s11_uniformity.py sparams/scenario_001_cpu.s16p sparams/scenario_001_gpu.s16p
    python compare_s11_uniformity.py sparams/scenario_001.s16p --fmin 0.8 --fmax 1.8
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import skrf as rf


def summarize_file(path: str, fmin_ghz: float, fmax_ghz: float) -> Dict[str, float]:
    ntwk = rf.Network(path)
    freqs_ghz = ntwk.f / 1e9
    s = ntwk.s

    n_ports = s.shape[1]
    if s.shape[1] != s.shape[2]:
        raise ValueError(f"S-parameter matrix is not square in {path}")

    mask = (freqs_ghz >= fmin_ghz) & (freqs_ghz <= fmax_ghz)
    if not np.any(mask):
        raise ValueError(
            f"No frequency points in requested band [{fmin_ghz}, {fmax_ghz}] GHz for {path}"
        )

    freqs_band = freqs_ghz[mask]

    dip_depths_db: List[float] = []
    dip_freqs_ghz: List[float] = []

    for i in range(n_ports):
        s11_db = 20.0 * np.log10(np.abs(s[:, i, i]) + 1e-12)
        s11_band = s11_db[mask]
        dip_idx = int(np.argmin(s11_band))
        dip_depths_db.append(float(s11_band[dip_idx]))
        dip_freqs_ghz.append(float(freqs_band[dip_idx]))

    depths = np.array(dip_depths_db, dtype=float)
    freqs = np.array(dip_freqs_ghz, dtype=float)

    return {
        "file": path,
        "n_ports": n_ports,
        "band_start": float(freqs_band[0]),
        "band_end": float(freqs_band[-1]),
        "mean_dip_db": float(np.mean(depths)),
        "std_dip_db": float(np.std(depths)),
        "min_dip_db": float(np.min(depths)),
        "max_dip_db": float(np.max(depths)),
        "mean_dip_freq_ghz": float(np.mean(freqs)),
        "std_dip_freq_mhz": float(np.std(freqs) * 1000.0),
    }


def print_summary_table(results: List[Dict[str, float]]) -> None:
    print("\nS11 Uniformity Summary")
    print("=" * 100)
    header = (
        f"{'file':42} {'ports':>5} {'mean dip':>10} {'dip std':>9} "
        f"{'dip min':>9} {'dip max':>9} {'mean f0':>10} {'f0 std':>9}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        name = os.path.basename(r["file"])
        print(
            f"{name:42} "
            f"{r['n_ports']:5d} "
            f"{r['mean_dip_db']:10.3f} "
            f"{r['std_dip_db']:9.3f} "
            f"{r['min_dip_db']:9.3f} "
            f"{r['max_dip_db']:9.3f} "
            f"{r['mean_dip_freq_ghz']:10.4f} "
            f"{r['std_dip_freq_mhz']:9.2f}"
        )

    print("=" * 100)
    print("Notes:")
    print("- More negative dip values mean deeper resonance (stronger matching at dip).")
    print("- Smaller dip std means dip depth is more uniform across ports.")
    print("- Smaller f0 std means resonance frequency is more uniform across ports.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare S11 dip uniformity across one or more Touchstone files."
    )
    parser.add_argument("files", nargs="+", help="Input Touchstone files (.sNp)")
    parser.add_argument(
        "--fmin",
        type=float,
        default=0.5,
        help="Lower frequency bound in GHz for dip search (default: 0.5)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=2.0,
        help="Upper frequency bound in GHz for dip search (default: 2.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fmax <= args.fmin:
        raise ValueError("--fmax must be greater than --fmin")

    missing = [f for f in args.files if not os.path.isfile(f)]
    if missing:
        missing_list = "\n".join(missing)
        raise FileNotFoundError(f"Missing files:\n{missing_list}")

    results = [summarize_file(f, args.fmin, args.fmax) for f in args.files]
    print_summary_table(results)


if __name__ == "__main__":
    main()
