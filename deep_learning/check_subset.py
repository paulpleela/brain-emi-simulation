from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from typing import Dict, List


def list_fd_ids(fd_dir: str) -> List[int]:
    ids: List[int] = []
    pattern = re.compile(r"scenario_(\d{3})_fd\.npz$")
    for name in os.listdir(fd_dir):
        m = pattern.match(name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_metadata(metadata_path: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid_raw = (row.get("scenario_id") or "").strip()
            if not sid_raw:
                continue
            try:
                sid = int(sid_raw)
            except ValueError:
                continue
            out[sid] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check copied FD subset against metadata")
    parser.add_argument("--fd-dir", default="fd_tensors")
    parser.add_argument("--metadata", default="dataset_metadata.csv")
    args = parser.parse_args()

    if not os.path.isdir(args.fd_dir):
        raise FileNotFoundError(f"FD tensor directory not found: {args.fd_dir}")
    if not os.path.isfile(args.metadata):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    ids = list_fd_ids(args.fd_dir)
    if not ids:
        raise ValueError("No scenario_XXX_fd.npz files found")

    meta = load_metadata(args.metadata)
    missing_meta = [sid for sid in ids if sid not in meta]

    split_counts = Counter()
    label_counts = Counter()
    for sid in ids:
        row = meta.get(sid)
        if not row:
            continue
        split_counts[(row.get("split") or "").strip().lower()] += 1
        label_counts[int((row.get("has_lesion") or "0").strip() or "0")] += 1

    print("Subset integrity report")
    print("-" * 60)
    print(f"count: {len(ids)}")
    print(f"scenario range: {ids[0]}..{ids[-1]}")
    print(f"first 10 ids: {ids[:10]}")
    print(f"missing metadata rows: {len(missing_meta)}")
    if missing_meta:
        print(f"missing metadata sample: {missing_meta[:10]}")

    print("split counts:", dict(split_counts))
    print("has_lesion counts:", dict(label_counts))

    if len(label_counts) < 2:
        print("WARNING: only one class present. Supervised classification is not possible on this subset.")


if __name__ == "__main__":
    main()
