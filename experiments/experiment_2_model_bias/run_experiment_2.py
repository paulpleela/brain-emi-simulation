#!/usr/bin/env python3
"""Run Experiment 2: model inductive bias comparison on mag_phase data."""

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"


def main() -> int:
    script = SCRIPT_DIR / "train_model_bias_comparison.py"
    result = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
