#!/usr/bin/env python3
"""Run Experiment 3: coarse lesion localisation on the normalized mag_phase dataset."""

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"


def run_script(script_name: str) -> int:
    script_path = SCRIPT_DIR / script_name
    result = subprocess.run([sys.executable, str(script_path)], cwd=REPO_ROOT)
    return result.returncode


def main() -> int:
    exit_code = run_script("build_localisation_dataset.py")
    if exit_code != 0:
        return exit_code
    return run_script("train_localisation_mlp.py")


if __name__ == "__main__":
    raise SystemExit(main())
