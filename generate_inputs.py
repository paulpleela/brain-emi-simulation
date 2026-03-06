"""
Generate brain imaging input files with:
- TRUE ellipsoidal head geometry (NOT spherical approximation)
- CSF ventricles (left and right lateral ventricles)
- Coupling medium between antennas and head
- 0.5-2 GHz bandwidth (centre 1.25 GHz)
- 16 z-directed wire dipole antennas in circular array

Each antenna is a physical half-wave wire dipole:
  - Two PEC arms along ±z, each 10 mm long (#edge commands)
  - 2 mm feed gap at the equatorial plane (z = head_centre_z)
  - #transmission_line: z at the gap, 73 Ω, records V/I for S-params
  - Total dipole length = 22 mm  (2 × 10 mm arms + 2 mm gap)
  - Resonance in coupling medium (εᵣ=36): λ/4 @ 1.25 GHz ≈ 10 mm ✓

All 16 dipoles are z-directed — no axis-alignment ambiguity.
Feed points sit at the outer surface of the coupling layer in the XY plane.

Coupling medium: low-loss glycerol/water-like medium (εᵣ≈36, σ≈0.3 S/m).
High εᵣ improves impedance matching to tissue; low σ avoids absorbing the
signal before it reaches the head.
"""

import os
import math
import numpy as np

# Output directory
output_dir = "brain_inputs"
os.makedirs(output_dir, exist_ok=True)

# Physical constants
c0 = 3e8  # m/s
f_centre = 1.25e9  # 1.25 GHz — centre of 0.5-2 GHz band
f_max = 2e9        # 2 GHz (upper frequency)

# Grid cell size — all snapping uses this
cell = 0.002  # 2 mm

# Realistic ellipsoidal head dimensions (average adult)
# Using semi-axes: a (front-back), b (side-side), c (top-bottom)
head_semi_axes = {
    'a': 0.095,  # 9.5 cm (front-back, anterior-posterior)
    'b': 0.075,  # 7.5 cm (side-side, left-right)
    'c': 0.115   # 11.5 cm (top-bottom, superior-inferior)
}

# Head center in domain
head_center = np.array([0.25, 0.25, 0.25])

# Layer thicknesses (realistic)
scalp_skull_thickness = 0.010  # 10 mm combined scalp+skull
gray_matter_thickness = 0.003   # 3 mm cortical gray matter
# White matter fills the rest

# Coupling medium (average dielectric of head tissues for better penetration)
# Coupling medium: low-loss glycerol/water-like mixture
# High εᵣ ≈ 36 matches head tissue impedance; low σ avoids absorbing the signal
# before it enters the head (NOT an absorber — that job belongs to the PML walls).
coupling_eps_r = 36.0   # glycerol/water mixture, typical brain-imaging rig value
coupling_sigma = 0.3    # S/m — low loss, ~half the old 0.6 S/m value
coupling_thickness = 0.005  # 5 mm layer

# Wire dipole dimensions (z-directed, resonant in coupling medium at 1.25 GHz)
# λ in coupling medium = c0 / (f_centre * sqrt(εᵣ)) = 300e6 / (1.25e9 * 6) = 40 mm
# Half-wave dipole length ≈ 0.47λ ≈ 18.8 mm → use 22 mm (2×10 mm arms + 2 mm gap)
# The coupling medium loads the dipole and lowers resonant frequency into 0.5-2 GHz band
dipole_arm_len  = 0.010   # 10 mm per arm
dipole_gap      = cell    # 2 mm feed gap (1 cell)
dipole_tl_ohms  = 73      # Ω — half-wave dipole input impedance in free space

n_antennas = 16

print("="*70)
print("BRAIN EMI SIMULATION PARAMETERS")
print("="*70)
print(f"\nFrequency range: 0.5 - 2 GHz  (centre {f_centre/1e9:.2f} GHz)")
print(f"Antennas: {n_antennas} z-directed wire dipoles (physical PEC arms)")
print(f"  — arm length: {dipole_arm_len*1000:.0f} mm, gap: {dipole_gap*1000:.0f} mm, total: {(2*dipole_arm_len+dipole_gap)*1000:.0f} mm")
print(f"  — TL impedance: {dipole_tl_ohms} Ohm")
print(f"  — feed at outer surface of coupling layer, z-directed (no axis-snapping)")
print(f"\nHead model: TRUE ELLIPSOIDAL GEOMETRY (not spherical)")
print(f"  Semi-axes: a={head_semi_axes['a']*100:.1f} cm, b={head_semi_axes['b']*100:.1f} cm, c={head_semi_axes['c']*100:.1f} cm")
print(f"  Scalp/skull thickness: {scalp_skull_thickness*1000:.0f} mm")
print(f"  Gray matter thickness: {gray_matter_thickness*1000:.0f} mm")
print(f"  CSF ventricles: INCLUDED (left + right lateral)")
print(f"\nCoupling medium (low-loss glycerol/water mixture):")
print(f"  Thickness: {coupling_thickness*1000:.0f} mm")
print(f"  er = {coupling_eps_r:.1f}, sigma = {coupling_sigma:.2f} S/m")
print("="*70)

# Generate input files
for src_idx in range(n_antennas):
    src_num = src_idx + 1
    filename = os.path.join(output_dir, f"brain_tx{src_num:02d}.in")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"## Brain hemorrhage imaging - 0-2 GHz\n")
        f.write(f"## TRUE ellipsoidal head model with CSF ventricles\n")
        f.write(f"## Transmit antenna: {src_num}/16\n")
        f.write(f"#title: brain_tx{src_num:02d}\n\n")
        
        # Domain
        f.write(f"## Domain (optimized for 2 GHz max frequency)\n")
        f.write(f"#domain: 0.6 0.6 0.6\n")
        f.write(f"#dx_dy_dz: 0.002 0.002 0.002\n")
        
        # Time window for 2 GHz (half the frequency = half the period)
        # Need enough time for signal to propagate through head and back
        # Head diameter ~20 cm, velocity in tissue ~c/sqrt(50) ≈ 0.14c
        # Round-trip time ~3 ns, use 15 ns for safety
        f.write(f"#time_window: 15e-9\n\n")
        
        # Waveforms - Gaussian centred at 1.25 GHz for 0.5-2 GHz bandwidth
        # rx_null: zero-amplitude waveform required for receiver transmission lines
        f.write(f"## Waveforms (centred at 1.25 GHz for 0.5-2 GHz band)\n")
        f.write(f"#waveform: gaussian 1 1.25e9 tx_pulse\n")
        f.write(f"#waveform: gaussian 0 1.25e9 rx_null\n\n")
        
        # Materials
        f.write(f"## Materials\n")
        f.write(f"#material: {coupling_eps_r} {coupling_sigma} 1 0 coupling_medium\n")
        f.write(f"#material: 12 0.2 1 0 scalp_skull\n")
        f.write(f"#material: 52 0.97 1 0 gray_matter\n")
        f.write(f"#material: 38 0.57 1 0 white_matter\n")
        f.write(f"#material: 80 2.0 1 0 csf\n")
        f.write(f"#material: 61 1.54 1 0 blood\n\n")
        
        # TRUE ELLIPSOIDAL HEAD MODEL
        # Generate voxelized ellipsoid geometry using Python code block
        f.write(f"## Head geometry - TRUE ellipsoidal layers (NOT spherical approximation)\n\n")
        
        # Write Python code that gprMax will execute to generate ellipsoid
        f.write(f"#python:\n")
        f.write(f"import numpy as np\n\n")
        f.write(f"# Head parameters\n")
        f.write(f"head_center = np.array([{head_center[0]}, {head_center[1]}, {head_center[2]}])\n")
        f.write(f"a = {head_semi_axes['a']}  # front-back\n")
        f.write(f"b = {head_semi_axes['b']}  # side-side\n")
        f.write(f"c = {head_semi_axes['c']}  # top-bottom\n")
        f.write(f"scalp_thickness = {scalp_skull_thickness}\n")
        f.write(f"gray_thickness = {gray_matter_thickness}\n")
        f.write(f"coupling_thickness = {coupling_thickness}\n\n")
        
        f.write(f"# Generate ellipsoidal layers using box filling\n")
        f.write(f"# Resolution for geometry generation (mm)\n")
        f.write(f"geo_res = 0.004  # 4mm boxes for geometry generation\n\n")
        
        f.write(f"# Calculate domain bounds for sampling\n")
        f.write(f"x_min = head_center[0] - (a + scalp_thickness + coupling_thickness + 0.01)\n")
        f.write(f"x_max = head_center[0] + (a + scalp_thickness + coupling_thickness + 0.01)\n")
        f.write(f"y_min = head_center[1] - (b + scalp_thickness + coupling_thickness + 0.01)\n")
        f.write(f"y_max = head_center[1] + (b + scalp_thickness + coupling_thickness + 0.01)\n")
        f.write(f"z_min = head_center[2] - (c + scalp_thickness + coupling_thickness + 0.01)\n")
        f.write(f"z_max = head_center[2] + (c + scalp_thickness + coupling_thickness + 0.01)\n\n")
        
        f.write(f"# Ellipsoid test function\n")
        f.write(f"def in_ellipsoid(x, y, z, center, a, b, c):\n")
        f.write(f"    dx = (x - center[0]) / a\n")
        f.write(f"    dy = (y - center[1]) / b\n")
        f.write(f"    dz = (z - center[2]) / c\n")
        f.write(f"    return (dx*dx + dy*dy + dz*dz) <= 1.0\n\n")
        
        f.write(f"# Sample points and fill with boxes\n")
        f.write(f"x_coords = np.arange(x_min, x_max, geo_res)\n")
        f.write(f"y_coords = np.arange(y_min, y_max, geo_res)\n")
        f.write(f"z_coords = np.arange(z_min, z_max, geo_res)\n\n")
        
        f.write(f"for x in x_coords:\n")
        f.write(f"    for y in y_coords:\n")
        f.write(f"        for z in z_coords:\n")
        f.write(f"            # Check which layer this point belongs to\n")
        f.write(f"            \n")
        f.write(f"            # Coupling medium (outermost)\n")
        f.write(f"            if in_ellipsoid(x, y, z, head_center, \n")
        f.write(f"                           a + scalp_thickness + coupling_thickness,\n")
        f.write(f"                           b + scalp_thickness + coupling_thickness,\n")
        f.write(f"                           c + scalp_thickness + coupling_thickness):\n")
        f.write(f"                material = 'coupling_medium'\n")
        f.write(f"                \n")
        f.write(f"                # Scalp/skull\n")
        f.write(f"                if in_ellipsoid(x, y, z, head_center,\n")
        f.write(f"                               a + scalp_thickness, b + scalp_thickness, c + scalp_thickness):\n")
        f.write(f"                    material = 'scalp_skull'\n")
        f.write(f"                    \n")
        f.write(f"                    # Gray matter\n")
        f.write(f"                    if in_ellipsoid(x, y, z, head_center, a, b, c):\n")
        f.write(f"                        material = 'gray_matter'\n")
        f.write(f"                        \n")
        f.write(f"                        # White matter (core)\n")
        f.write(f"                        if in_ellipsoid(x, y, z, head_center,\n")
        f.write(f"                                       a - gray_thickness, b - gray_thickness, c - gray_thickness):\n")
        f.write(f"                            material = 'white_matter'\n")
        f.write(f"                \n")
        f.write(f"                # Create box for this voxel\n")
        f.write(f"                x1, y1, z1 = x, y, z\n")
        f.write(f"                x2, y2, z2 = x + geo_res, y + geo_res, z + geo_res\n")
        f.write(f"                print(f'#box: {{x1}} {{y1}} {{z1}} {{x2}} {{y2}} {{z2}} {{material}}')\n\n")
        
        f.write(f"#end_python:\n\n")
        
        # Add CSF ventricles (also ellipsoidal)
        f.write(f"## CSF Ventricles (lateral ventricles - critical anatomical feature)\n")
        f.write(f"#python:\n")
        f.write(f"# Ventricle parameters\n")
        f.write(f"vent_a = 0.020  # 2cm front-back\n")
        f.write(f"vent_b = 0.010  # 1cm side-side\n")
        f.write(f"vent_c = 0.040  # 4cm top-bottom\n")
        f.write(f"vent_separation = 0.015  # 1.5cm between centers\n\n")
        
        f.write(f"# Left ventricle\n")
        f.write(f"vent_left_center = np.array([head_center[0] - vent_separation/2, head_center[1], head_center[2]])\n")
        f.write(f"# Right ventricle\n")
        f.write(f"vent_right_center = np.array([head_center[0] + vent_separation/2, head_center[1], head_center[2]])\n\n")
        
        f.write(f"# Sample for ventricles\n")
        f.write(f"vent_x_min = head_center[0] - vent_separation/2 - vent_a - 0.005\n")
        f.write(f"vent_x_max = head_center[0] + vent_separation/2 + vent_a + 0.005\n")
        f.write(f"vent_y_min = head_center[1] - vent_b - 0.005\n")
        f.write(f"vent_y_max = head_center[1] + vent_b + 0.005\n")
        f.write(f"vent_z_min = head_center[2] - vent_c - 0.005\n")
        f.write(f"vent_z_max = head_center[2] + vent_c + 0.005\n\n")
        
        f.write(f"x_coords = np.arange(vent_x_min, vent_x_max, geo_res)\n")
        f.write(f"y_coords = np.arange(vent_y_min, vent_y_max, geo_res)\n")
        f.write(f"z_coords = np.arange(vent_z_min, vent_z_max, geo_res)\n\n")
        
        f.write(f"for x in x_coords:\n")
        f.write(f"    for y in y_coords:\n")
        f.write(f"        for z in z_coords:\n")
        f.write(f"            # Check if in left or right ventricle\n")
        f.write(f"            if (in_ellipsoid(x, y, z, vent_left_center, vent_a, vent_b, vent_c) or\n")
        f.write(f"                in_ellipsoid(x, y, z, vent_right_center, vent_a, vent_b, vent_c)):\n")
        f.write(f"                x1, y1, z1 = x, y, z\n")
        f.write(f"                x2, y2, z2 = x + geo_res, y + geo_res, z + geo_res\n")
        f.write(f"                print(f'#box: {{x1}} {{y1}} {{z1}} {{x2}} {{y2}} {{z2}} csf')\n\n")
        
        f.write(f"#end_python:\n\n")
        
        # Hemorrhagic lesion - positioned in one hemisphere
        # Place at realistic location: ~2 cm from center toward left side
        lesion_x = head_center[0] - 0.02
        lesion_y = head_center[1]
        lesion_z = head_center[2] + 0.01
        
        f.write(f"## Hemorrhagic lesion (blood clot)\n")
        f.write(f"#sphere: {lesion_x} {lesion_y} {lesion_z} 0.015 blood\n")
        f.write(f"#sphere: {lesion_x + 0.004} {lesion_y} {lesion_z} 0.01 blood\n\n")

        # ── Antenna array: 16 z-directed wire dipoles ─────────────────────────
        # Each antenna is a physical half-wave dipole:
        #   - Two PEC arms along ±z, each dipole_arm_len long (#edge)
        #   - dipole_gap (1 cell) feed gap at the equatorial plane
        #   - #transmission_line: z at the gap, dipole_tl_ohms Ω
        #
        # All dipoles are z-directed → no axis-snapping issue.
        # The TX dipole uses tx_pulse; RX dipoles use rx_null (zero amplitude,
        # TL still records induced V/I for S-parameter extraction).
        #
        # Compute feed positions in a #python: block, then emit per-antenna.
        f.write(f"## Antenna array (16 z-directed wire dipoles at equatorial ring)\n")
        f.write(f"#python:\n")
        f.write(f"import math\n")
        f.write(f"cell               = {cell}\n")
        f.write(f"n_antennas         = {n_antennas}\n")
        f.write(f"head_center        = ({head_center[0]}, {head_center[1]}, {head_center[2]})\n")
        f.write(f"a                  = {head_semi_axes['a']}\n")
        f.write(f"b                  = {head_semi_axes['b']}\n")
        f.write(f"scalp_thickness    = {scalp_skull_thickness}\n")
        f.write(f"coupling_thickness = {coupling_thickness}\n")
        f.write(f"arm                = {dipole_arm_len}\n")
        f.write(f"gap                = {dipole_gap}\n")
        f.write(f"antennas = []  # (cx, cy, cz)\n")
        f.write(f"for i in range(n_antennas):\n")
        f.write(f"    angle = 2 * math.pi * i / n_antennas\n")
        f.write(f"    cos_a = math.cos(angle)\n")
        f.write(f"    sin_a = math.sin(angle)\n")
        f.write(f"    r_x = a + scalp_thickness + coupling_thickness + cell\n")
        f.write(f"    r_y = b + scalp_thickness + coupling_thickness + cell\n")
        f.write(f"    cx = round((head_center[0] + r_x * cos_a) / cell) * cell\n")
        f.write(f"    cy = round((head_center[1] + r_y * sin_a) / cell) * cell\n")
        f.write(f"    cz = head_center[2]\n")
        f.write(f"    antennas.append((cx, cy, cz))\n")
        f.write(f"#end_python:\n\n")

        # Per-antenna commands: PEC arms + transmission_line
        f.write(f"## Antenna array - 16 z-directed wire dipoles\n")
        for ant_idx in range(n_antennas):
            ant_num = ant_idx + 1
            f.write(f"\n## Antenna {ant_num}\n")
            f.write(f"#python:\n")
            f.write(f"cx, cy, cz = antennas[{ant_idx}]\n")
            # PEC arms: lower arm (cz-arm to cz-gap/2) and upper arm (cz+gap/2 to cz+arm)
            f.write("print(f'#edge: {cx} {cy} {round(cz-arm,6)} {cx} {cy} {round(cz-gap/2,6)} pec')\n")
            f.write("print(f'#edge: {cx} {cy} {round(cz+gap/2,6)} {cx} {cy} {round(cz+arm,6)} pec')\n")
            if ant_idx == src_idx:
                # TX: transmission line drives and records at feed gap
                f.write(f"print(f'#transmission_line: z {{cx}} {{cy}} {{cz}} {dipole_tl_ohms} tx_pulse')\n")
            else:
                # RX: zero-amplitude waveform; TL records induced V/I
                f.write(f"print(f'#transmission_line: z {{cx}} {{cy}} {{cz}} {dipole_tl_ohms} rx_null')\n")
            f.write("#end_python:\n")

        f.write(f"\n## End of input file\n")

    print(f"  Created: {filename}")

print(f"\nDone. Generated {n_antennas} input files in {output_dir}/")
