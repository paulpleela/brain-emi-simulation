"""
Render a scenario-specific head setup diagram from dataset metadata.

This script visualizes the XY (equatorial) setup used to generate gprMax inputs:
- Rotated/scaled head tissue layers
- Lesion location and size (if present)
- 16-antenna ring feed positions

Usage examples:
  python visualize_head_setup.py --scenario 1
  python visualize_head_setup.py --scenario 237 --output setup_237.png --show
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np


DEFAULT_METADATA = "dataset_metadata.csv"

# Must match generate_dataset.py
CELL = 0.002
HEAD_CENTER = np.array([0.25, 0.25, 0.25], dtype=np.float64)
HEAD_SEMI_AXES = {"a": 0.095, "b": 0.075, "c": 0.115}
SCALP_SKULL_THICKNESS = 0.010
GRAY_MATTER_THICKNESS = 0.003
COUPLING_THICKNESS = 0.020
N_ANTENNAS = 16
FIXED_ANTENNA_RADIUS_X = 0.124
FIXED_ANTENNA_RADIUS_Y = 0.104


def parse_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return int(default)
    return int(raw)


def parse_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return float(default)
    return float(raw)


def load_scenario_row(metadata_path: str, scenario_id: int) -> Dict[str, str]:
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = parse_int(row, "scenario_id", -1)
            if sid == scenario_id:
                return row

    raise ValueError(f"Scenario {scenario_id} not found in {metadata_path}")


def lesion_world_from_metadata(row: Dict[str, str]) -> Tuple[float, float, float, float]:
    head_scale = parse_float(row, "head_scale", 1.0)
    if head_scale <= 0:
        head_scale = 1.0

    theta_deg = parse_float(row, "head_rotation_deg", 0.0)
    theta = np.deg2rad(theta_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    local_x = parse_float(row, "lesion_x", 0.0) * head_scale
    local_y = parse_float(row, "lesion_y", 0.0) * head_scale
    local_z = parse_float(row, "lesion_z", 0.0) * head_scale

    world_x = cos_t * local_x - sin_t * local_y
    world_y = sin_t * local_x + cos_t * local_y
    world_z = local_z

    lx = float(HEAD_CENTER[0] + world_x)
    ly = float(HEAD_CENTER[1] + world_y)
    lz = float(HEAD_CENTER[2] + world_z)
    radius_m = parse_float(row, "lesion_size_mm", 0.0) / 1000.0
    return lx, ly, lz, radius_m


def antenna_positions() -> list[Tuple[float, float, int]]:
    out: list[Tuple[float, float, int]] = []
    for i in range(N_ANTENNAS):
        angle = 2.0 * math.pi * i / N_ANTENNAS
        cx = round((HEAD_CENTER[0] + FIXED_ANTENNA_RADIUS_X * math.cos(angle)) / CELL) * CELL
        cy = round((HEAD_CENTER[1] + FIXED_ANTENNA_RADIUS_Y * math.sin(angle)) / CELL) * CELL
        out.append((cx, cy, i + 1))
    return out


def m_to_cm(v: float) -> float:
    return float(v * 100.0)


def add_layer(ax, a: float, b: float, head_scale: float, theta_deg: float, face: str, edge: str, zorder: int, label: str) -> None:
    e = Ellipse(
        xy=(m_to_cm(HEAD_CENTER[0]), m_to_cm(HEAD_CENTER[1])),
        width=m_to_cm(2.0 * a * head_scale),
        height=m_to_cm(2.0 * b * head_scale),
        angle=theta_deg,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
        alpha=0.95,
        zorder=zorder,
        label=label,
    )
    ax.add_patch(e)


def render_scenario(row: Dict[str, str], output_path: str | None, show: bool) -> None:
    scenario_id = parse_int(row, "scenario_id")
    has_lesion = parse_int(row, "has_lesion", 0) == 1
    shape = (row.get("shape") or "none").strip().lower()
    group = (row.get("group") or "").strip()
    split = (row.get("split") or "").strip()
    noise_level = (row.get("noise_level") or "").strip()

    head_scale = parse_float(row, "head_scale", 1.0)
    if head_scale <= 0:
        head_scale = 1.0
    theta_deg = parse_float(row, "head_rotation_deg", 0.0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    # Draw outside -> inside so inner layers remain visible.
    add_layer(
        ax,
        HEAD_SEMI_AXES["a"] + SCALP_SKULL_THICKNESS + COUPLING_THICKNESS,
        HEAD_SEMI_AXES["b"] + SCALP_SKULL_THICKNESS + COUPLING_THICKNESS,
        1.0,
        0.0,
        face="#d7ecff",
        edge="#5b96c8",
        zorder=1,
        label="Coupling (fixed orientation)",
    )
    add_layer(
        ax,
        HEAD_SEMI_AXES["a"] + SCALP_SKULL_THICKNESS,
        HEAD_SEMI_AXES["b"] + SCALP_SKULL_THICKNESS,
        head_scale,
        theta_deg,
        face="#f8d3a4",
        edge="#b0783d",
        zorder=2,
        label="Scalp/Skull",
    )
    add_layer(
        ax,
        HEAD_SEMI_AXES["a"],
        HEAD_SEMI_AXES["b"],
        head_scale,
        theta_deg,
        face="#ceb8de",
        edge="#7a5a98",
        zorder=3,
        label="Gray Matter",
    )
    add_layer(
        ax,
        HEAD_SEMI_AXES["a"] - GRAY_MATTER_THICKNESS,
        HEAD_SEMI_AXES["b"] - GRAY_MATTER_THICKNESS,
        head_scale,
        theta_deg,
        face="#eef0f9",
        edge="#6e7794",
        zorder=4,
        label="White Matter",
    )

    lesion_info = "none"
    if has_lesion:
        lx, ly, _lz, lesion_r = lesion_world_from_metadata(row)
        if shape == "ellipsoid":
            lesion_patch = Ellipse(
                (m_to_cm(lx), m_to_cm(ly)),
                width=m_to_cm(2.0 * lesion_r),
                height=m_to_cm(2.0 * lesion_r * 0.8),
                angle=0.0,
                facecolor="#c92a2a",
                edgecolor="#7b1a1a",
                linewidth=1.1,
                alpha=0.9,
                zorder=8,
                label="Lesion (ellipsoid XY projection)",
            )
            ax.add_patch(lesion_patch)
        else:
            lesion_patch = Circle(
                (m_to_cm(lx), m_to_cm(ly)),
                radius=m_to_cm(lesion_r),
                facecolor="#c92a2a",
                edgecolor="#7b1a1a",
                linewidth=1.1,
                alpha=0.9,
                zorder=8,
                label="Lesion (sphere)",
            )
            ax.add_patch(lesion_patch)
        lesion_info = f"{shape}, radius={parse_float(row, 'lesion_size_mm', 0.0):.2f} mm"

    for ax_m, ay_m, idx in antenna_positions():
        ax_cm = m_to_cm(ax_m)
        ay_cm = m_to_cm(ay_m)
        ax.plot(ax_cm, ay_cm, marker="o", markersize=4, color="#174a9c", zorder=10)

        # Move index labels radially outward for readability.
        dx = ax_m - HEAD_CENTER[0]
        dy = ay_m - HEAD_CENTER[1]
        norm = math.hypot(dx, dy)
        ux = dx / norm if norm > 0 else 0.0
        uy = dy / norm if norm > 0 else 0.0
        ax.text(
            ax_cm + 0.8 * ux,
            ay_cm + 0.8 * uy,
            str(idx),
            fontsize=7,
            ha="center",
            va="center",
            color="#1f1f1f",
            zorder=11,
        )

    ax.plot(
        m_to_cm(HEAD_CENTER[0]),
        m_to_cm(HEAD_CENTER[1]),
        marker="+",
        markersize=10,
        markeredgewidth=1.4,
        color="black",
        zorder=12,
    )

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.grid(True, alpha=0.25, linewidth=0.6)

    ax.set_title(
        (
            f"Scenario {scenario_id:03d} setup (XY cross-section)\n"
            f"group={group} | split={split} | noise={noise_level} | "
            f"head_scale={head_scale:.3f} | head_rotation_deg={theta_deg:.2f} | lesion={lesion_info}"
        ),
        fontsize=10,
        pad=12,
    )

    # Expand bounds around antenna ring.
    ant_xy = np.array([(m_to_cm(x), m_to_cm(y)) for x, y, _ in antenna_positions()], dtype=np.float64)
    margin_cm = 4.0
    ax.set_xlim(float(np.min(ant_xy[:, 0]) - margin_cm), float(np.max(ant_xy[:, 0]) + margin_cm))
    ax.set_ylim(float(np.min(ant_xy[:, 1]) - margin_cm), float(np.max(ant_xy[:, 1]) + margin_cm))

    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved setup diagram: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def default_output_for(scenario_id: int) -> str:
    return os.path.join("setup_diagrams", f"scenario_{scenario_id:03d}_setup.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize head setup for one scenario from metadata")
    parser.add_argument("--scenario", type=int, required=True, help="Scenario ID to render")
    parser.add_argument("--metadata", default=DEFAULT_METADATA, help="Path to dataset metadata CSV")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Also show the figure window")
    parser.add_argument("--open-only", action="store_true", help="Open the figure without saving a PNG")
    args = parser.parse_args()

    row = load_scenario_row(args.metadata, args.scenario)
    output_path = None if args.open_only else (args.output if args.output else default_output_for(args.scenario))
    render_scenario(row, output_path, args.show)


if __name__ == "__main__":
    main()
