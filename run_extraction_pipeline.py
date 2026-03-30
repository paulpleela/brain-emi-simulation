"""
Run the full extraction pipeline:
  1) .out -> .s16p      via build_s16p.py
    2) .s16p -> .npz FD   via build_time_dataset.py

Usage examples:
  python run_extraction_pipeline.py --scenario 1
  python run_extraction_pipeline.py --range 1 300
  python run_extraction_pipeline.py --all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


def build_selector_args(args: argparse.Namespace) -> List[str]:
    if args.scenario is not None:
        return ["--scenario", str(args.scenario)]
    if args.range is not None:
        return ["--range", str(args.range[0]), str(args.range[1])]
    return ["--all"]


def run_cmd(cmd: List[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run .out -> .s16p -> frequency-domain tensor pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=int, help="Process one scenario")
    group.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), help="Process scenario range")
    group.add_argument("--all", action="store_true", help="Process all scenarios")

    parser.add_argument("--keep-out", action="store_true", help="Keep .out files after .s16p extraction")
    parser.add_argument("--input-dir", default="sparams", help="Input .s16p folder for FD stage")
    parser.add_argument("--output-dir", default="fd_tensors", help="Output folder for FD .npz files")
    parser.add_argument("--metadata", default="dataset_metadata.csv", help="Metadata CSV path")
    parser.add_argument("--fit-stats", action="store_true", help="Fit train-only normalization stats before applying")

    args = parser.parse_args()

    selector = build_selector_args(args)

    # Stage 1: .out -> .s16p
    cmd_stage1 = [sys.executable, "build_s16p.py", *selector]
    if args.keep_out:
        cmd_stage1.append("--no-delete")
    run_cmd(cmd_stage1)

    # Stage 2: .s16p -> frequency-domain npz
    cmd_stage2 = [
        sys.executable,
        "build_time_dataset.py",
        *selector,
        "--input-dir",
        args.input_dir,
        "--output-dir",
        args.output_dir,
        "--metadata",
        args.metadata,
    ]
    if args.fit_stats:
        cmd_stage2.append("--fit-stats")
    run_cmd(cmd_stage2)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
