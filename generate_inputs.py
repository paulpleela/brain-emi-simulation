"""
Generate brain imaging input files with:
- TRUE ellipsoidal head geometry (NOT spherical approximation)
- CSF ventricles (left and right lateral ventricles)
- Coupling medium between antennas and head
- 0-2 GHz bandwidth
- 16 monopole antennas in circular array

This generates proper ellipsoidal layers using voxel-based geometry.
Each layer (coupling, scalp/skull, gray, white matter) follows ellipsoidal shape.
"""

import os
import math
import numpy as np

# Output directory
output_dir = "brain_inputs"
os.makedirs(output_dir, exist_ok=True)

# Physical constants
c0 = 3e8  # m/s
f_max = 2e9  # 2 GHz (upper frequency)
wavelength_min = c0 / f_max  # 0.15 m = 150 mm

# For 2 GHz operation, monopole needs to be shorter
monopole_length = wavelength_min / 4  # 37.5 mm (λ/4 @ 2 GHz)
wire_radius = 0.001  # 1 mm
gp_size = wavelength_min / 2  # 75 mm (smaller ground plane for 2 GHz)
gp_half_size = gp_size / 2
gp_thickness = 0.002  # 2 mm

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
# Using average of scalp and gray matter
coupling_eps_r = (12 + 52) / 2  # ≈ 32
coupling_sigma = (0.2 + 0.97) / 2  # ≈ 0.6 S/m
coupling_thickness = 0.005  # 5 mm layer

# Antenna positions - 16 antennas in fixed positions around head
# Positioned to touch coupling medium (not floating)
n_antennas = 16

# Calculate antenna positions on ellipsoid surface
antenna_positions = []
for i in range(n_antennas):
    # Distribute around head in horizontal plane (z = head center)
    angle = 2 * math.pi * i / n_antennas
    
    # On ellipse in x-y plane
    # Parametric: x = a*cos(θ), y = b*sin(θ)
    # Add coupling thickness to position antennas just outside head
    x = head_center[0] + (head_semi_axes['a'] + scalp_skull_thickness + coupling_thickness + gp_half_size) * math.cos(angle)
    y = head_center[1] + (head_semi_axes['b'] + scalp_skull_thickness + coupling_thickness + gp_half_size) * math.sin(angle)
    z = head_center[2]  # Same z as head center (equatorial plane)
    
    antenna_positions.append((x, y, z, angle))

print("="*70)
print("BRAIN EMI SIMULATION PARAMETERS")
print("="*70)
print(f"\nFrequency range: 0 - 2 GHz")
print(f"Monopole length: {monopole_length*1000:.1f} mm (λ/4 @ 2 GHz)")
print(f"Ground plane: {gp_size*1000:.1f} × {gp_size*1000:.1f} mm")
print(f"\nHead model: TRUE ELLIPSOIDAL GEOMETRY (not spherical)")
print(f"  Semi-axes: a={head_semi_axes['a']*100:.1f} cm, b={head_semi_axes['b']*100:.1f} cm, c={head_semi_axes['c']*100:.1f} cm")
print(f"  Scalp/skull thickness: {scalp_skull_thickness*1000:.0f} mm")
print(f"  Gray matter thickness: {gray_matter_thickness*1000:.0f} mm")
print(f"  CSF ventricles: INCLUDED (left + right lateral)")
print(f"\nCoupling medium:")
print(f"  Thickness: {coupling_thickness*1000:.0f} mm")
print(f"  εr = {coupling_eps_r:.1f}, σ = {coupling_sigma:.2f} S/m")
print(f"\nAntennas: {n_antennas} monopoles in circular array")
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
        
        # Waveforms - Gaussian centered at 1 GHz for 0-2 GHz bandwidth
        # rx_null: zero-amplitude waveform required for receiver transmission lines
        f.write(f"## Waveforms (optimized for 0-2 GHz)\n")
        f.write(f"#waveform: gaussian 1 1e9 tx_pulse\n")
        f.write(f"#waveform: gaussian 0 1e9 rx_null\n\n")
        
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
        
        # Antenna array
        f.write(f"#python:\n")
        f.write(f"# Antenna geometry parameters\n")
        f.write(f"monopole_length = {monopole_length}\n")
        f.write(f"wire_radius = {wire_radius}\n")
        f.write(f"gp_half_size = {gp_half_size}\n")
        f.write(f"gp_thickness = {gp_thickness}\n")
        f.write(f"n_antennas = {n_antennas}\n\n")
        f.write(f"# Compute antenna positions (touching coupling medium)\n")
        f.write(f"antenna_positions = []\n")
        # Add required imports and aliases so gprMax's python execution sees them
        f.write(f"import math\n")
        f.write(f"head_center = ({head_center[0]}, {head_center[1]}, {head_center[2]})\n")
        f.write(f"a = {head_semi_axes['a']}\n")
        f.write(f"b = {head_semi_axes['b']}\n")
        f.write(f"scalp_thickness = {scalp_skull_thickness}\n")
        f.write(f"coupling_thickness = {coupling_thickness}\n")
        f.write(f"for i in range(n_antennas):\n")
        f.write(f"    angle = 2 * math.pi * i / n_antennas\n")
        f.write(f"    # Position on ellipse + coupling + ground plane clearance\n")
        f.write(f"    x = head_center[0] + (a + scalp_thickness + coupling_thickness + gp_half_size) * math.cos(angle)\n")
        f.write(f"    y = head_center[1] + (b + scalp_thickness + coupling_thickness + gp_half_size) * math.sin(angle)\n")
        f.write(f"    z = head_center[2]\n")
        f.write(f"    antenna_positions.append((x, y, z, angle))\n")
        f.write(f"#end_python:\n\n")
        
        # Generate all antennas
        f.write(f"## Antenna array - 16 monopoles in fixed positions\n")
        for ant_idx in range(n_antennas):
            ant_num = ant_idx + 1
            f.write(f"\n## Antenna {ant_num}\n")
            f.write(f"#python:\n")
            f.write(f"ant_idx = {ant_idx}\n")
            f.write(f"x, y, z_base, angle = antenna_positions[ant_idx]\n")
            f.write(f"gp_x1 = x - gp_half_size\n")
            f.write(f"gp_x2 = x + gp_half_size\n")
            f.write(f"gp_y1 = y - gp_half_size\n")
            f.write(f"gp_y2 = y + gp_half_size\n")
            f.write(f"gp_z1 = z_base - gp_thickness/2\n")
            f.write(f"gp_z2 = z_base + gp_thickness/2\n")
            f.write(f"mono_top = gp_z2 + monopole_length\n")
            f.write(f"feed_z = gp_z2\n\n")
            
            # Write geometry commands INSIDE the Python block using print()
            # This way Python evaluates the variables and outputs the actual commands
            is_transmitter = (ant_idx == src_idx)
            f.write("print(f'#box: {gp_x1} {gp_y1} {gp_z1} {gp_x2} {gp_y2} {gp_z2} pec')\n")
            f.write("print(f'#cylinder: {x} {y} {feed_z} {x} {y} {mono_top} {wire_radius} pec')\n")
            
            # Use transmission_line for accurate S-parameter extraction (CPU only)
            if is_transmitter:
                f.write("print(f'#transmission_line: z {x} {y} {feed_z} 50 tx_pulse')\n")
            else:
                # Receivers: zero-amplitude waveform (measures signal, injects nothing)
                f.write("print(f'#transmission_line: z {x} {y} {feed_z} 50 rx_null')\n")
            
            f.write("#end_python:\n")
        
        f.write(f"\n## End of input file\n")
    
    print(f"  Created: {filename}")

print(f"\n✓ Generated {n_antennas} input files in {output_dir}/")
