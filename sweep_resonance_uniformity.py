"""Automated sweep runner for Scenario 001 S11 uniformity on HPC.

This script automates the full loop for each configuration:
1) regenerate test inputs with env overrides
2) run Scenario 001 (TX01..TX16) on GPU via SLURM
3) wait for jobs to finish
4) verify TL Vtotal is non-zero for all active TX ports
5) extract scenario_001.s16p
6) score port-to-port resonance uniformity

Run on HPC login node (where sbatch/squeue are available):
    python sweep_resonance_uniformity.py

Custom sweep examples:
    python sweep_resonance_uniformity.py --arms-mm 52 56 60 --offset-cells 1 0 -0.5 --thickness-mm 5 10 20
    python sweep_resonance_uniformity.py --concurrency 16 --poll-sec 20
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np

from compare_s11_uniformity import summarize_file


@dataclass
class SweepConfig:
    arm_mm: float
    gap_mm: float
    thickness_mm: float
    offset_cells: float
    tl_ohms: float

    def tag(self) -> str:
        def fmt(x: float) -> str:
            return str(x).replace("-", "m").replace(".", "p")

        return (
            f"arm{fmt(self.arm_mm)}_"
            f"gap{fmt(self.gap_mm)}_"
            f"thk{fmt(self.thickness_mm)}_"
            f"off{fmt(self.offset_cells)}_"
            f"z{fmt(self.tl_ohms)}"
        )


def run_cmd(cmd: List[str], env: dict | None = None, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def submit_gpu_array(concurrency: int) -> str:
    out = run_cmd(["sbatch", f"--array=1-16%{concurrency}", "run_simulation_gpu.sh"])
    m = re.search(r"Submitted batch job\s+(\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse SLURM job ID from sbatch output: {out}")
    return m.group(1)


def wait_for_job(job_id: str, poll_sec: int) -> None:
    while True:
        # Empty output means no longer in queue (completed/failed/cancelled)
        out = run_cmd(["squeue", "-h", "-j", job_id, "-o", "%A"])
        if not out:
            return
        time.sleep(poll_sec)


def clean_scenario_outputs(workspace: Path) -> None:
    for p in workspace.glob("brain_inputs/scenario_001_tx*.out"):
        p.unlink(missing_ok=True)
    (workspace / "sparams/scenario_001.s16p").unlink(missing_ok=True)


def verify_all_nonzero(workspace: Path) -> Tuple[bool, List[float]]:
    vals: List[float] = []
    ok = True
    for tx in range(1, 17):
        fn = workspace / f"brain_inputs/scenario_001_tx{tx:02d}.out"
        if not fn.exists():
            ok = False
            vals.append(0.0)
            continue

        with h5py.File(fn, "r") as h:
            tl = f"tl{tx}"
            if "tls" not in h or tl not in h["tls"]:
                ok = False
                vals.append(0.0)
                continue

            vmax = float(np.max(np.abs(h["tls"][tl]["Vtotal"][:])))
            vals.append(vmax)
            if vmax == 0.0:
                ok = False

    return ok, vals


def build_env(base_env: dict, cfg: SweepConfig) -> dict:
    env = dict(base_env)
    env["DIPOLE_ARM_LEN_M"] = str(cfg.arm_mm / 1000.0)
    env["DIPOLE_GAP_M"] = str(cfg.gap_mm / 1000.0)
    env["DIPOLE_TL_OHMS"] = str(cfg.tl_ohms)
    env["COUPLING_THICKNESS_M"] = str(cfg.thickness_mm / 1000.0)
    env["ANTENNA_OFFSET_CELLS"] = str(cfg.offset_cells)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated Scenario 001 resonance-uniformity sweep on HPC.")
    parser.add_argument("--arms-mm", type=float, nargs="+", default=[56.0], help="Dipole arm lengths in mm")
    parser.add_argument("--gap-mm", type=float, nargs="+", default=[2.0], help="Feed gap in mm")
    parser.add_argument("--thickness-mm", type=float, nargs="+", default=[5.0, 10.0, 20.0], help="Coupling thickness in mm")
    parser.add_argument("--offset-cells", type=float, nargs="+", default=[1.0, 0.0, -0.5], help="Antenna radial offset in cells")
    parser.add_argument("--tl-ohms", type=float, nargs="+", default=[73.0], help="Transmission-line resistance in ohms")
    parser.add_argument("--concurrency", type=int, default=8, help="SLURM array concurrency for TX01..TX16")
    parser.add_argument("--poll-sec", type=int, default=20, help="Polling interval while waiting for SLURM jobs")
    parser.add_argument("--fmin", type=float, default=0.5, help="Lower GHz for dip search")
    parser.add_argument("--fmax", type=float, default=2.0, help="Upper GHz for dip search")
    parser.add_argument("--out-csv", default="sparams/sweep_results.csv", help="CSV output path")
    args = parser.parse_args()

    workspace = Path.cwd()
    if not (workspace / "run_simulation_gpu.sh").exists():
        raise RuntimeError("Run this script from the project root (missing run_simulation_gpu.sh)")

    combos = [
        SweepConfig(arm, gap, thk, off, z0)
        for arm, gap, thk, off, z0 in itertools.product(
            args.arms_mm, args.gap_mm, args.thickness_mm, args.offset_cells, args.tl_ohms
        )
    ]

    print(f"Total configurations: {len(combos)}")
    print("Starting sweep...\n")

    rows = []
    base_env = os.environ.copy()

    for idx, cfg in enumerate(combos, start=1):
        print(f"[{idx}/{len(combos)}] {cfg}")
        env = build_env(base_env, cfg)

        clean_scenario_outputs(workspace)

        run_cmd([sys.executable, "generate_dataset.py", "--test"], env=env, cwd=workspace)

        job_id = submit_gpu_array(args.concurrency)
        print(f"  Submitted SLURM job: {job_id}")
        wait_for_job(job_id, args.poll_sec)
        print("  Job finished")

        nonzero_ok, vmax_list = verify_all_nonzero(workspace)
        if not nonzero_ok:
            print("  WARNING: Some TX Vtotal maxima are zero or missing")

        run_cmd([sys.executable, "extract_sparameters.py", "--scenario", "1", "--no-delete"], cwd=workspace)

        src = workspace / "sparams/scenario_001.s16p"
        if not src.exists():
            raise RuntimeError("Expected sparams/scenario_001.s16p was not created")

        tagged = workspace / f"sparams/scenario_001_{cfg.tag()}.s16p"
        tagged.write_bytes(src.read_bytes())

        stats = summarize_file(str(tagged), args.fmin, args.fmax)
        score = stats["std_dip_db"] + 0.01 * stats["std_dip_freq_mhz"]

        row = {
            "tag": cfg.tag(),
            "arm_mm": cfg.arm_mm,
            "gap_mm": cfg.gap_mm,
            "thickness_mm": cfg.thickness_mm,
            "offset_cells": cfg.offset_cells,
            "tl_ohms": cfg.tl_ohms,
            "all_nonzero": nonzero_ok,
            "vtotal_min": float(min(vmax_list)) if vmax_list else 0.0,
            "vtotal_max": float(max(vmax_list)) if vmax_list else 0.0,
            "mean_dip_db": stats["mean_dip_db"],
            "std_dip_db": stats["std_dip_db"],
            "min_dip_db": stats["min_dip_db"],
            "max_dip_db": stats["max_dip_db"],
            "mean_dip_freq_ghz": stats["mean_dip_freq_ghz"],
            "std_dip_freq_mhz": stats["std_dip_freq_mhz"],
            "score": score,
            "file": str(tagged),
        }
        rows.append(row)

        print(
            f"  mean dip={row['mean_dip_db']:.3f} dB, dip std={row['std_dip_db']:.3f} dB, "
            f"f0 std={row['std_dip_freq_mhz']:.1f} MHz, score={row['score']:.3f}\n"
        )

    rows.sort(key=lambda r: (not r["all_nonzero"], r["score"]))

    out_csv = workspace / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("Sweep complete.")
    print(f"CSV: {out_csv}")

    print("\nTop 5 configurations:")
    for i, r in enumerate(rows[:5], start=1):
        print(
            f"{i}. {r['tag']} | nonzero={r['all_nonzero']} | "
            f"dip std={r['std_dip_db']:.3f} dB | "
            f"f0 std={r['std_dip_freq_mhz']:.1f} MHz | score={r['score']:.3f}"
        )


if __name__ == "__main__":
    main()
