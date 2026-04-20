"""
Generate gprMax scenario input files from dataset metadata.

Each metadata row (scenario) produces 16 input files:
  brain_inputs/scenario_XXX_tx01.in ... scenario_XXX_tx16.in

This script makes simulation generation metadata-driven, so the physics setup
(lesion presence/size/position, property variation, head scale, head rotation)
is read per scenario from dataset_metadata.csv. Coupling thickness is fixed at
0.02 m and antenna offset is fixed at -0.5 cells.
"""

import argparse
import csv
import os
import numpy as np


OUTPUT_DIR = "brain_inputs"
DEFAULT_METADATA = "dataset_metadata.csv"
N_ANTENNAS = 16
CELL = 0.002

HEAD_SEMI_AXES = {'a': 0.095, 'b': 0.075, 'c': 0.115}
HEAD_CENTER = np.array([0.25, 0.25, 0.25])
SCALP_SKULL_THICKNESS = 0.010
GRAY_MATTER_THICKNESS = 0.003

BASE_MATERIALS = {
    "coupling": {"eps": 36.0, "sig": 0.3},
    "scalp_skull": {"eps": 12.0, "sig": 0.2},
    "gray": {"eps": 52.0, "sig": 0.97},
    "white": {"eps": 38.0, "sig": 0.57},
    "csf": {"eps": 80.0, "sig": 2.0},
    "blood": {"eps": 61.0, "sig": 1.54},
}

DEFAULT_COUPLING_THICKNESS = 0.020
DEFAULT_ANTENNA_OFFSET_CELLS = -0.50

DIPOLE_ARM_LEN = float(os.getenv("DIPOLE_ARM_LEN_M", "0.056"))
DIPOLE_GAP = float(os.getenv("DIPOLE_GAP_M", "0.002"))
DIPOLE_TL_OHMS = float(os.getenv("DIPOLE_TL_OHMS", "73"))


def pct_scale(value, pct):
    return value * (1.0 + pct / 100.0)


def parse_float(row, key, default=0.0):
    raw = str(row.get(key, "")).strip()
    if raw == "":
        return float(default)
    return float(raw)


def parse_int(row, key, default=0):
    raw = str(row.get(key, "")).strip()
    if raw == "":
        return int(default)
    return int(raw)


def load_metadata(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = parse_int(row, "scenario_id", -1)
            if sid <= 0:
                continue
            rows.append(row)
    rows.sort(key=lambda r: int(r["scenario_id"]))
    return rows


def filter_rows(rows, scenario=None, range_vals=None):
    if scenario is not None:
        return [r for r in rows if int(r["scenario_id"]) == scenario]
    if range_vals is not None:
        lo, hi = range_vals
        return [r for r in rows if lo <= int(r["scenario_id"]) <= hi]
    return rows


def build_materials(row):
    eps_var_bg = parse_float(row, "epsilon_variation", 0.0)
    sig_var_bg = parse_float(row, "sigma_variation", 0.0)
    eps_var_anom = parse_float(row, "epsilon_anomaly_variation", 0.0)
    sig_var_anom = parse_float(row, "sigma_anomaly_variation", 0.0)

    coupling_eps = pct_scale(BASE_MATERIALS["coupling"]["eps"], eps_var_bg)
    coupling_sig = pct_scale(BASE_MATERIALS["coupling"]["sig"], sig_var_bg)

    scalp_eps = pct_scale(BASE_MATERIALS["scalp_skull"]["eps"], eps_var_bg)
    scalp_sig = pct_scale(BASE_MATERIALS["scalp_skull"]["sig"], sig_var_bg)

    gray_eps = pct_scale(BASE_MATERIALS["gray"]["eps"], eps_var_bg)
    gray_sig = pct_scale(BASE_MATERIALS["gray"]["sig"], sig_var_bg)

    white_eps = pct_scale(BASE_MATERIALS["white"]["eps"], eps_var_bg)
    white_sig = pct_scale(BASE_MATERIALS["white"]["sig"], sig_var_bg)

    csf_eps = pct_scale(BASE_MATERIALS["csf"]["eps"], eps_var_bg)
    csf_sig = pct_scale(BASE_MATERIALS["csf"]["sig"], sig_var_bg)

    blood_eps = pct_scale(BASE_MATERIALS["blood"]["eps"], eps_var_anom)
    blood_sig = pct_scale(BASE_MATERIALS["blood"]["sig"], sig_var_anom)

    return {
        "coupling": (coupling_eps, coupling_sig),
        "scalp": (scalp_eps, scalp_sig),
        "gray": (gray_eps, gray_sig),
        "white": (white_eps, white_sig),
        "csf": (csf_eps, csf_sig),
        "blood": (blood_eps, blood_sig),
    }


def write_lesion(f, row):
    has_lesion = parse_int(row, "has_lesion", 0) == 1
    if not has_lesion:
        return

    lesion_size_m = parse_float(row, "lesion_size_mm", 0.0) / 1000.0
    head_scale = parse_float(row, "head_scale", 1.0)
    if head_scale <= 0:
        head_scale = 1.0

    head_rotation_deg = parse_float(row, "head_rotation_deg", 0.0)
    theta = np.deg2rad(head_rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Scale lesion offsets with the head, then rotate around z-axis.
    local_x = parse_float(row, "lesion_x", 0.0) * head_scale
    local_y = parse_float(row, "lesion_y", 0.0) * head_scale
    local_z = parse_float(row, "lesion_z", 0.0) * head_scale

    world_x = cos_t * local_x - sin_t * local_y
    world_y = sin_t * local_x + cos_t * local_y
    world_z = local_z

    lx = HEAD_CENTER[0] + world_x
    ly = HEAD_CENTER[1] + world_y
    lz = HEAD_CENTER[2] + world_z
    shape = str(row.get("shape", "sphere")).strip().lower()

    if shape == "ellipsoid":
        ra = lesion_size_m
        rb = lesion_size_m * 0.8
        rc = lesion_size_m * 1.2
        f.write("## Hemorrhage (ellipsoid)\n#python:\n")
        f.write(f"cx, cy, cz = {lx:.6f}, {ly:.6f}, {lz:.6f}\n")
        f.write(f"ra, rb, rc = {ra:.6f}, {rb:.6f}, {rc:.6f}\n")
        f.write("geo_res = 0.004\n")
        f.write("import numpy as np\n")
        f.write("for x in np.arange(cx-ra, cx+ra, geo_res):\n")
        f.write("    for y in np.arange(cy-rb, cy+rb, geo_res):\n")
        f.write("        for z in np.arange(cz-rc, cz+rc, geo_res):\n")
        f.write("            dx=((x-cx)/ra)**2\n")
        f.write("            dy=((y-cy)/rb)**2\n")
        f.write("            dz=((z-cz)/rc)**2\n")
        f.write("            if dx+dy+dz <= 1.0:\n")
        f.write("                print(f'#box: {x} {y} {z} {x+geo_res} {y+geo_res} {z+geo_res} blood')\n")
        f.write("#end_python:\n\n")
    else:
        f.write(f"## Hemorrhage\n#sphere: {lx:.6f} {ly:.6f} {lz:.6f} {lesion_size_m:.6f} blood\n\n")


def write_scenario(row, output_dir):
    scenario_id = parse_int(row, "scenario_id")
    has_lesion = parse_int(row, "has_lesion", 0) == 1
    noise_level = str(row.get("noise_level", "low")).strip() or "low"
    group_name = str(row.get("group", "unknown")).strip() or "unknown"

    coupling_thickness = DEFAULT_COUPLING_THICKNESS
    antenna_offset_cells = DEFAULT_ANTENNA_OFFSET_CELLS

    head_scale = parse_float(row, "head_scale", 1.0)
    if head_scale <= 0:
        head_scale = 1.0
    head_rotation_deg = parse_float(row, "head_rotation_deg", 0.0)

    materials = build_materials(row)

    for src_idx in range(N_ANTENNAS):
        src_num = src_idx + 1
        filename = os.path.join(output_dir, f"scenario_{scenario_id:03d}_tx{src_num:02d}.in")

        with open(filename, "w") as f:
            f.write(f"## Scenario {scenario_id:03d} - Transmit {src_num}/16\n")
            f.write(f"## group={group_name} noise={noise_level}\n")
            if has_lesion:
                f.write(
                    "## Hemorrhage: "
                    f"{parse_float(row, 'lesion_size_mm', 0.0):.2f}mm at "
                    f"({parse_float(row, 'lesion_x', 0.0):.4f}, {parse_float(row, 'lesion_y', 0.0):.4f}, {parse_float(row, 'lesion_z', 0.0):.4f})\n"
                )
            else:
                f.write("## Healthy baseline\n")
            f.write(f"#title: scenario_{scenario_id:03d}_tx{src_num:02d}\n\n")

            f.write("#domain: 0.6 0.6 0.6\n")
            f.write("#dx_dy_dz: 0.002 0.002 0.002\n")
            f.write("#time_window: 60e-9\n\n")

            f.write("## Materials\n")
            f.write(f"#material: {materials['coupling'][0]:.6f} {materials['coupling'][1]:.6f} 1 0 coupling_medium\n")
            f.write(f"#material: {materials['scalp'][0]:.6f} {materials['scalp'][1]:.6f} 1 0 scalp_skull\n")
            f.write(f"#material: {materials['gray'][0]:.6f} {materials['gray'][1]:.6f} 1 0 gray_matter\n")
            f.write(f"#material: {materials['white'][0]:.6f} {materials['white'][1]:.6f} 1 0 white_matter\n")
            f.write(f"#material: {materials['csf'][0]:.6f} {materials['csf'][1]:.6f} 1 0 csf\n")
            f.write(f"#material: {materials['blood'][0]:.6f} {materials['blood'][1]:.6f} 1 0 blood\n\n")

            f.write("## Ellipsoidal head geometry\n#python:\n")
            f.write("import numpy as np\n")
            f.write(f"head_center = np.array([{HEAD_CENTER[0]}, {HEAD_CENTER[1]}, {HEAD_CENTER[2]}])\n")
            f.write(f"a, b, c = {HEAD_SEMI_AXES['a']}, {HEAD_SEMI_AXES['b']}, {HEAD_SEMI_AXES['c']}\n")
            f.write(f"scalp_thickness = {SCALP_SKULL_THICKNESS}\n")
            f.write(f"gray_thickness = {GRAY_MATTER_THICKNESS}\n")
            f.write(f"coupling_thickness = {coupling_thickness}\n")
            f.write(f"head_scale = {head_scale:.6f}\n")
            f.write(f"theta_deg = {head_rotation_deg:.6f}\n")
            f.write("geo_res = 0.004\n\n")
            f.write("theta = np.deg2rad(theta_deg)\n")
            f.write("cos_t = np.cos(theta)\n")
            f.write("sin_t = np.sin(theta)\n\n")
            f.write("def to_head_frame(x, y, z):\n")
            f.write("    dx = x - head_center[0]\n")
            f.write("    dy = y - head_center[1]\n")
            f.write("    dz = z - head_center[2]\n")
            f.write("    xr = cos_t * dx + sin_t * dy\n")
            f.write("    yr = -sin_t * dx + cos_t * dy\n")
            f.write("    return xr / head_scale, yr / head_scale, dz / head_scale\n\n")
            f.write("def in_ellipsoid_local(x, y, z, a, b, c, cx=0.0, cy=0.0, cz=0.0):\n")
            f.write("    dx, dy, dz = (x-cx)/a, (y-cy)/b, (z-cz)/c\n")
            f.write("    return (dx*dx + dy*dy + dz*dz) <= 1.0\n\n")
            f.write("outer_a = (a + scalp_thickness + coupling_thickness) * head_scale\n")
            f.write("outer_b = (b + scalp_thickness + coupling_thickness) * head_scale\n")
            f.write("outer_c = (c + scalp_thickness + coupling_thickness) * head_scale\n")
            f.write("r_xy = max(outer_a, outer_b) + 0.01\n")
            f.write("r_z = outer_c + 0.01\n")
            f.write("x_min = head_center[0] - r_xy\n")
            f.write("x_max = head_center[0] + r_xy\n")
            f.write("y_min = head_center[1] - r_xy\n")
            f.write("y_max = head_center[1] + r_xy\n")
            f.write("z_min = head_center[2] - r_z\n")
            f.write("z_max = head_center[2] + r_z\n\n")
            f.write("for x in np.arange(x_min, x_max, geo_res):\n")
            f.write("    for y in np.arange(y_min, y_max, geo_res):\n")
            f.write("        for z in np.arange(z_min, z_max, geo_res):\n")
            f.write("            xh, yh, zh = to_head_frame(x, y, z)\n")
            f.write("            if in_ellipsoid_local(xh, yh, zh, a+scalp_thickness+coupling_thickness, b+scalp_thickness+coupling_thickness, c+scalp_thickness+coupling_thickness):\n")
            f.write("                material = 'coupling_medium'\n")
            f.write("                if in_ellipsoid_local(xh, yh, zh, a+scalp_thickness, b+scalp_thickness, c+scalp_thickness):\n")
            f.write("                    material = 'scalp_skull'\n")
            f.write("                    if in_ellipsoid_local(xh, yh, zh, a, b, c):\n")
            f.write("                        material = 'gray_matter'\n")
            f.write("                        if in_ellipsoid_local(xh, yh, zh, a-gray_thickness, b-gray_thickness, c-gray_thickness):\n")
            f.write("                            material = 'white_matter'\n")
            f.write("                print(f'#box: {x} {y} {z} {x+geo_res} {y+geo_res} {z+geo_res} {material}')\n")
            f.write("#end_python:\n\n")

            f.write("## CSF Ventricles\n#python:\n")
            f.write("vent_a, vent_b, vent_c = 0.020, 0.010, 0.040\n")
            f.write("vent_left_local = np.array([-0.0075, 0.0, 0.0])\n")
            f.write("vent_right_local = np.array([0.0075, 0.0, 0.0])\n")
            f.write("for x in np.arange(head_center[0]-0.035*head_scale, head_center[0]+0.035*head_scale, geo_res):\n")
            f.write("    for y in np.arange(head_center[1]-0.015*head_scale, head_center[1]+0.015*head_scale, geo_res):\n")
            f.write("        for z in np.arange(head_center[2]-0.045*head_scale, head_center[2]+0.045*head_scale, geo_res):\n")
            f.write("            xh, yh, zh = to_head_frame(x, y, z)\n")
            f.write("            if in_ellipsoid_local(xh, yh, zh, vent_a, vent_b, vent_c, vent_left_local[0], vent_left_local[1], vent_left_local[2]) or in_ellipsoid_local(xh, yh, zh, vent_a, vent_b, vent_c, vent_right_local[0], vent_right_local[1], vent_right_local[2]):\n")
            f.write("                print(f'#box: {x} {y} {z} {x+geo_res} {y+geo_res} {z+geo_res} csf')\n")
            f.write("#end_python:\n\n")

            write_lesion(f, row)

            f.write("## Waveforms\n")
            f.write("#waveform: gaussian 1 1.25e9 tx_pulse\n")
            # Keep receiver excitation effectively zero while ensuring TL channels are retained.
            f.write("#waveform: gaussian 1e-12 1.25e9 rx_null\n\n")

            f.write("## Antenna array (16 z-directed wire dipoles at equatorial ring)\n")
            f.write("#python:\n")
            f.write("import math\n")
            f.write(f"cell               = {CELL}\n")
            f.write(f"n_antennas         = {N_ANTENNAS}\n")
            f.write(f"head_center        = ({HEAD_CENTER[0]}, {HEAD_CENTER[1]}, {HEAD_CENTER[2]})\n")
            f.write(f"a                  = {HEAD_SEMI_AXES['a']}\n")
            f.write(f"b                  = {HEAD_SEMI_AXES['b']}\n")
            f.write(f"scalp_thickness    = {SCALP_SKULL_THICKNESS}\n")
            f.write(f"coupling_thickness = {coupling_thickness}\n")
            f.write(f"antenna_offset_cells = {antenna_offset_cells}\n")
            f.write(f"arm                = {DIPOLE_ARM_LEN}\n")
            f.write(f"gap                = {DIPOLE_GAP}\n")
            f.write("antennas = []\n")
            f.write("for i in range(n_antennas):\n")
            f.write("    angle = 2 * math.pi * i / n_antennas\n")
            f.write("    cos_a = math.cos(angle)\n")
            f.write("    sin_a = math.sin(angle)\n")
            f.write("    r_x = a + scalp_thickness + coupling_thickness + antenna_offset_cells * cell\n")
            f.write("    r_y = b + scalp_thickness + coupling_thickness + antenna_offset_cells * cell\n")
            f.write("    cx = round((head_center[0] + r_x * cos_a) / cell) * cell\n")
            f.write("    cy = round((head_center[1] + r_y * sin_a) / cell) * cell\n")
            f.write("    cz = head_center[2]\n")
            f.write("    antennas.append((cx, cy, cz))\n")
            f.write("#end_python:\n\n")

            for ant_idx in range(N_ANTENNAS):
                f.write(f"## Antenna {ant_idx+1}\n#python:\n")
                f.write(f"cx, cy, cz = antennas[{ant_idx}]\n")
                f.write("print(f'#edge: {cx} {cy} {round(cz-arm,6)} {cx} {cy} {round(cz+arm+gap,6)} pec')\n")
                f.write("print(f'#edge: {cx} {cy} {round(cz,6)} {cx} {cy} {round(cz+gap,6)} free_space')\n")
                if ant_idx == src_idx:
                    f.write(f"print(f'#transmission_line: z {{cx}} {{cy}} {{cz}} {DIPOLE_TL_OHMS} tx_pulse')\n")
                else:
                    f.write(f"print(f'#transmission_line: z {{cx}} {{cy}} {{cz}} {DIPOLE_TL_OHMS} rx_null')\n")
                f.write("#end_python:\n\n")


def main():
    parser = argparse.ArgumentParser(description="Generate gprMax inputs from dataset metadata")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scenario", type=int, help="Generate one scenario by ID")
    group.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), help="Generate scenarios START..END")
    parser.add_argument("--metadata", default=DEFAULT_METADATA, help="Metadata CSV path")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for .in files")
    args = parser.parse_args()

    rows = load_metadata(args.metadata)
    rows = filter_rows(rows, scenario=args.scenario, range_vals=tuple(args.range) if args.range else None)

    if not rows:
        print("No matching metadata rows found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("METADATA-DRIVEN GPRMAX INPUT GENERATOR")
    print("=" * 80)
    print(f"Metadata rows to generate: {len(rows)}")
    print(f"Output directory: {args.output_dir}")

    for i, row in enumerate(rows, start=1):
        sid = int(row["scenario_id"])
        write_scenario(row, args.output_dir)
        if i % 10 == 0 or i == len(rows):
            print(f"  Generated scenarios: {i}/{len(rows)} (latest: {sid:03d})")

    total_files = len(rows) * N_ANTENNAS
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Generated {total_files} input files for {len(rows)} scenarios.")


if __name__ == "__main__":
    main()
