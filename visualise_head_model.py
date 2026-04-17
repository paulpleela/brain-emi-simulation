"""
Render a scenario-specific diagram of the brain-EMI head model setup.

This script reads one row from dataset_metadata.csv and draws a top-down
equatorial (XY) view of the simulation geometry for that scenario:

- head layers (coupling, scalp/skull, gray matter, white matter)
- ventricles
- lesion location and size
- 16 antenna positions using the scenario's antenna_offset and
  coupling_thickness

Usage:
    python visualise_head_model.py --scenario 317
    python visualise_head_model.py --scenario 317 --metadata dataset_metadata.csv --out scenario_317_setup.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


CELL = 0.002  # 2 mm grid
HEAD_CENTER = (0.25, 0.25, 0.25)
HEAD_SEMI_AXES = {"a": 0.095, "b": 0.075, "c": 0.115}
SCALP_SKULL_THICKNESS = 0.010
GRAY_MATTER_THICKNESS = 0.003
N_ANTENNAS = 16
DIPOLE_ARM_LEN = 0.056
DIPOLE_GAP = 0.002


def parse_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    raw = str(row.get(key, "")).strip()
    if raw == "":
        return float(default)
    return float(raw)


def parse_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    raw = str(row.get(key, "")).strip()
    if raw == "":
        return int(default)
    return int(raw)


def load_metadata_row(metadata_path: str, scenario_id: int) -> Dict[str, str]:
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
            if sid == scenario_id:
                return row
    raise FileNotFoundError(f"Scenario {scenario_id:03d} not found in {metadata_path}")


def m2cm(v: np.ndarray | List[float] | float) -> np.ndarray:
    return np.asarray(v) * 100.0


def ellipse(ax, cx, cy, a, b, **kwargs):
    theta = np.linspace(0, 2 * np.pi, 400)
    x = cx + a * np.cos(theta)
    y = cy + b * np.sin(theta)
    ax.fill(*m2cm([x, y]), **kwargs)
    ax.plot(*m2cm([x, y]), color=kwargs.get("edgecolor", "k"), lw=0.8)


def draw_material_layers(ax, coupling_thickness: float):
    layers = [
        ("coupling", HEAD_SEMI_AXES["a"] + SCALP_SKULL_THICKNESS + coupling_thickness,
         HEAD_SEMI_AXES["b"] + SCALP_SKULL_THICKNESS + coupling_thickness,
         "#d4e9ff", "#5599cc", 2),
        ("scalp_skull", HEAD_SEMI_AXES["a"] + SCALP_SKULL_THICKNESS,
         HEAD_SEMI_AXES["b"] + SCALP_SKULL_THICKNESS,
         "#f5c28a", "#b07030", 3),
        ("gray", HEAD_SEMI_AXES["a"], HEAD_SEMI_AXES["b"],
         "#c8a0c8", "#804080", 4),
        ("white", HEAD_SEMI_AXES["a"] - GRAY_MATTER_THICKNESS,
         HEAD_SEMI_AXES["b"] - GRAY_MATTER_THICKNESS,
         "#e8e8f8", "#6060a0", 5),
    ]

    for _, a, b, fc, ec, zo in layers:
        ellipse(ax, HEAD_CENTER[0], HEAD_CENTER[1], a, b, facecolor=fc, edgecolor=ec, zorder=zo)


def draw_ventricles(ax):
    vent_a, vent_b = 0.020, 0.010
    vent_sep = 0.015
    vent_left_cx = HEAD_CENTER[0] - vent_sep / 2
    vent_right_cx = HEAD_CENTER[0] + vent_sep / 2
    theta = np.linspace(0, 2 * np.pi, 200)
    for vx in [vent_left_cx, vent_right_cx]:
        x = vx + vent_a * np.cos(theta)
        y = HEAD_CENTER[1] + vent_b * np.sin(theta)
        ax.fill(*m2cm([x, y]), color="#80d0ff", zorder=6)
        ax.plot(*m2cm([x, y]), color="#0060b0", lw=0.8, zorder=6)


def lesion_patch(row: Dict[str, str]):
    has_lesion = parse_int(row, "has_lesion", 0) == 1
    if not has_lesion:
        return None, None, None

    lesion_size_m = parse_float(row, "lesion_size_mm", 0.0) / 1000.0
    lx = HEAD_CENTER[0] + parse_float(row, "lesion_x", 0.0)
    ly = HEAD_CENTER[1] + parse_float(row, "lesion_y", 0.0)
    lz = HEAD_CENTER[2] + parse_float(row, "lesion_z", 0.0)
    shape = str(row.get("shape", "sphere")).strip().lower()

    if shape == "ellipsoid":
        ra = lesion_size_m
        rb = lesion_size_m * 0.8
        return (lx, ly), (ra, rb), lz

    return (lx, ly), (lesion_size_m, lesion_size_m), lz


def draw_lesion(ax, row: Dict[str, str]):
    center, radii, lz = lesion_patch(row)
    if center is None:
        return

    shape = str(row.get("shape", "sphere")).strip().lower()
    color = "#cc2222"
    if shape == "ellipsoid":
        lesion = mpatches.Ellipse(m2cm(center), m2cm(radii[0]) * 2, m2cm(radii[1]) * 2,
                                  facecolor=color, edgecolor="#990000", alpha=0.95, zorder=7)
        ax.add_patch(lesion)
    else:
        lesion = plt.Circle(m2cm(center), m2cm(radii[0]), color=color, zorder=7)
        ax.add_patch(lesion)

    label_x = m2cm(center[0] + 0.020)
    label_y = m2cm(center[1] + 0.018)
    ax.annotate(
        f"Lesion\n{parse_float(row, 'lesion_size_mm', 0.0):.1f} mm, {shape}\nz={((lz - HEAD_CENTER[2]) * 100):.1f} cm",
        xy=m2cm(center), xycoords="data",
        xytext=(label_x, label_y), textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#880000", lw=1.0),
        fontsize=8, color="#660000", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cc8888", alpha=0.9),
        zorder=12,
    )


def compute_antennas(row: Dict[str, str]):
    antenna_offset_cells = parse_float(row, "antenna_offset", -0.50)
    coupling_thickness = parse_float(row, "coupling_thickness", 0.020)
    antennas = []
    for i in range(N_ANTENNAS):
        angle = 2 * math.pi * i / N_ANTENNAS
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        r_x = HEAD_SEMI_AXES["a"] + SCALP_SKULL_THICKNESS + coupling_thickness + antenna_offset_cells * CELL
        r_y = HEAD_SEMI_AXES["b"] + SCALP_SKULL_THICKNESS + coupling_thickness + antenna_offset_cells * CELL
        cx = round((HEAD_CENTER[0] + r_x * cos_a) / CELL) * CELL
        cy = round((HEAD_CENTER[1] + r_y * sin_a) / CELL) * CELL
        antennas.append({"cx": cx, "cy": cy, "angle": angle, "idx": i + 1})
    return antennas, coupling_thickness, antenna_offset_cells


def draw_antennas(ax, antennas):
    arm_cm = 2.0
    gap_cm = DIPOLE_GAP * 100

    for ant in antennas:
        cx_cm, cy_cm = m2cm(ant["cx"]), m2cm(ant["cy"])
        ax.annotate(
            "", xy=(cx_cm, cy_cm + arm_cm), xytext=(cx_cm, cy_cm + gap_cm / 2),
            arrowprops=dict(arrowstyle="->", color="#0044cc", lw=1.5), zorder=9
        )
        ax.annotate(
            "", xy=(cx_cm, cy_cm - arm_cm), xytext=(cx_cm, cy_cm - gap_cm / 2),
            arrowprops=dict(arrowstyle="->", color="#0044cc", lw=1.5), zorder=9
        )
        ax.plot(cx_cm, cy_cm, "o", color="#0044cc", ms=4, zorder=10, markeredgecolor="#002288", markeredgewidth=0.5)

        angle = ant["angle"]
        lx = HEAD_CENTER[0] + (HEAD_SEMI_AXES["a"] + SCALP_SKULL_THICKNESS + 0.028) * math.cos(angle)
        ly = HEAD_CENTER[1] + (HEAD_SEMI_AXES["b"] + SCALP_SKULL_THICKNESS + 0.028) * math.sin(angle)
        ax.text(m2cm(lx), m2cm(ly), str(ant["idx"]), ha="center", va="center", fontsize=6.5,
                color="#222222", fontweight="bold", zorder=11)


def make_diagram(row: Dict[str, str], scenario_id: int, out_path: str, show: bool = False):
    antennas, coupling_thickness, antenna_offset_cells = compute_antennas(row)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    draw_material_layers(ax, coupling_thickness)
    draw_ventricles(ax)
    draw_lesion(ax, row)
    draw_antennas(ax, antennas)

    ax.plot(*m2cm([HEAD_CENTER[0], HEAD_CENTER[1]]), "+", color="black", ms=8, mew=1.2, zorder=12)

    legend_patches = [
        mpatches.Patch(fc="#d4e9ff", ec="#5599cc", label=f"Coupling medium (thickness={coupling_thickness*1000:.1f} mm)"),
        mpatches.Patch(fc="#f5c28a", ec="#b07030", label="Scalp + Skull (10 mm)"),
        mpatches.Patch(fc="#c8a0c8", ec="#804080", label="Gray matter"),
        mpatches.Patch(fc="#e8e8f8", ec="#6060a0", label="White matter"),
        mpatches.Patch(fc="#80d0ff", ec="#0060b0", label="CSF ventricles"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#0044cc", markersize=6, label="Dipole feed gap"),
        plt.Line2D([0], [0], color="#0044cc", lw=1.5, label="Dipole arms ±z"),
        mpatches.Patch(fc="#cc2222", ec="#cc2222", label="Lesion / blood"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7.5, framealpha=0.92,
              edgecolor="#aaaaaa", title="Materials & elements", title_fontsize=8)

    all_x = [m2cm(a["cx"]) for a in antennas]
    all_y = [m2cm(a["cy"]) for a in antennas]
    margin = 6
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_xlabel("x (cm) — anterior ↔ posterior", fontsize=10)
    ax.set_ylabel("y (cm) — left ↔ right", fontsize=10)

    split = str(row.get("split", "unknown")).strip()
    group = str(row.get("group", "unknown")).strip()
    size_bucket = str(row.get("size_bucket", "none")).strip()
    region = str(row.get("region", "none")).strip()
    shape = str(row.get("shape", "none")).strip()
    has_lesion = parse_int(row, "has_lesion", 0)
    lesion_mm = parse_float(row, "lesion_size_mm", 0.0)
    antenna_offset_mm = antenna_offset_cells * CELL * 1000.0

    ax.set_title(
        f"Brain-EMI setup diagram — scenario {scenario_id:03d} | split={split} | group={group}\n"
        f"has_lesion={has_lesion} | size={lesion_mm:.1f} mm | region={region} | shape={shape} | "
        f"antenna_offset={antenna_offset_cells:.2f} cells ({antenna_offset_mm:+.1f} mm) | "
        f"coupling_thickness={coupling_thickness*1000:.1f} mm | size_bucket={size_bucket}",
        fontsize=10, pad=10,
    )
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise the head-model setup for any scenario")
    parser.add_argument("--scenario", type=int, required=True, help="Scenario ID from dataset_metadata.csv")
    parser.add_argument("--metadata", default="dataset_metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--out", default=None, help="Output image path (default: scenario_XXX_setup.png)")
    parser.add_argument("--show", action="store_true", help="Display the plot after saving")
    args = parser.parse_args()

    row = load_metadata_row(args.metadata, args.scenario)
    out_path = args.out or f"scenario_{args.scenario:03d}_setup.png"
    make_diagram(row, args.scenario, out_path, show=args.show)


if __name__ == "__main__":
    main()