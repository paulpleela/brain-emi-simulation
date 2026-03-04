"""
Dataset Generator - Creates 300 scenarios with varied hemorrhage positions/sizes

Run this to generate full ML training dataset.
Uses the base generate_inputs.py code but loops over scenarios.
"""

import os
import math
import numpy as np

# Output directory
output_dir = "brain_inputs"
os.makedirs(output_dir, exist_ok=True)

# Constants (from generate_inputs.py)
c0 = 3e8
f_max = 2e9
wavelength_min = c0 / f_max
monopole_length = wavelength_min / 4
wire_radius = 0.001
gp_size = wavelength_min / 2
gp_half_size = gp_size / 2
gp_thickness = 0.002

head_semi_axes = {'a': 0.095, 'b': 0.075, 'c': 0.115}
head_center = np.array([0.25, 0.25, 0.25])
scalp_skull_thickness = 0.010
gray_matter_thickness = 0.003
coupling_eps_r = 32
coupling_sigma = 0.585
coupling_thickness = 0.005
n_antennas = 16

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

NUM_HEALTHY = 50        # Healthy baselines
NUM_HEMORRHAGE = 250    # Hemorrhage cases  
LESION_SIZES = [0.005, 0.010, 0.015, 0.020, 0.025]  # 5, 10, 15, 20, 25mm
NUM_POSITIONS = 50      # Positions per size

def generate_positions(n):
    """Generate systematic hemorrhage positions covering brain volume"""
    positions = []
    
    # Sample within 70% of brain to stay in gray/white matter
    max_r_xy = 0.05  # 5cm radial
    max_z = 0.08     # 8cm vertical
    
    # Create 3D grid
    n_per_dim = int(np.ceil(n ** (1/3)))
    x_vals = np.linspace(-max_r_xy, max_r_xy, n_per_dim)
    y_vals = np.linspace(-max_r_xy, max_r_xy, n_per_dim)
    z_vals = np.linspace(-max_z, max_z, n_per_dim)
    
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                # Check if inside brain (avoid ventricles)
                r = np.sqrt(x**2 + y**2)
                if r < max_r_xy and abs(z) < max_z and abs(x) > 0.015:
                    positions.append((x, y, z))
    
    # Sample if too many
    if len(positions) > n:
        idx = np.random.choice(len(positions), n, replace=False)
        positions = [positions[i] for i in idx]
    
    # Add random if too few
    while len(positions) < n:
        r = np.random.uniform(0.015, max_r_xy)
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(-max_z, max_z)
        if abs(x) > 0.015:  # Avoid ventricles
            positions.append((x, y, z))
    
    return positions[:n]

def write_scenario(scenario_id, has_lesion, lesion_size, lesion_pos):
    """Generate 16 input files for one scenario"""
    for src_idx in range(n_antennas):
        src_num = src_idx + 1
        filename = os.path.join(output_dir, f"scenario_{scenario_id:03d}_tx{src_num:02d}.in")
        
        with open(filename, 'w') as f:
            # Header
            f.write(f"## Scenario {scenario_id:03d} - Transmit {src_num}/16\n")
            if has_lesion:
                f.write(f"## Hemorrhage: {lesion_size*1000:.0f}mm at {lesion_pos}\n")
            else:
                f.write(f"## Healthy baseline\n")
            f.write(f"#title: scenario_{scenario_id:03d}_tx{src_num:02d}\n\n")
            
            # Domain and waveform
            f.write("#domain: 0.6 0.6 0.6\n")
            f.write("#dx_dy_dz: 0.002 0.002 0.002\n")
            f.write("#time_window: 15e-9\n\n")
            
            # Materials
            f.write("## Materials\n")
            f.write(f"#material: {coupling_eps_r} {coupling_sigma} 1 0 coupling_medium\n")
            f.write("#material: 12 0.2 1 0 scalp_skull\n")
            f.write("#material: 52 0.97 1 0 gray_matter\n")
            f.write("#material: 38 0.57 1 0 white_matter\n")
            f.write("#material: 80 2.0 1 0 csf\n")
            f.write("#material: 61 1.54 1 0 blood\n\n")
            
            # Ellipsoidal head geometry
            f.write("## Ellipsoidal head geometry\n#python:\n")
            f.write("import numpy as np\n")
            f.write(f"head_center = np.array([{head_center[0]}, {head_center[1]}, {head_center[2]}])\n")
            f.write(f"a, b, c = {head_semi_axes['a']}, {head_semi_axes['b']}, {head_semi_axes['c']}\n")
            f.write(f"scalp_thickness = {scalp_skull_thickness}\n")
            f.write(f"gray_thickness = {gray_matter_thickness}\n")
            f.write(f"coupling_thickness = {coupling_thickness}\n")
            f.write("geo_res = 0.004\n\n")
            
            # Ellipsoid function and layer generation
            f.write("def in_ellipsoid(x, y, z, center, a, b, c):\n")
            f.write("    dx, dy, dz = (x-center[0])/a, (y-center[1])/b, (z-center[2])/c\n")
            f.write("    return (dx*dx + dy*dy + dz*dz) <= 1.0\n\n")
            
            f.write("x_min = head_center[0] - (a + scalp_thickness + coupling_thickness + 0.01)\n")
            f.write("x_max = head_center[0] + (a + scalp_thickness + coupling_thickness + 0.01)\n")
            f.write("y_min = head_center[1] - (b + scalp_thickness + coupling_thickness + 0.01)\n")
            f.write("y_max = head_center[1] + (b + scalp_thickness + coupling_thickness + 0.01)\n")
            f.write("z_min = head_center[2] - (c + scalp_thickness + coupling_thickness + 0.01)\n")
            f.write("z_max = head_center[2] + (c + scalp_thickness + coupling_thickness + 0.01)\n\n")
            
            f.write("for x in np.arange(x_min, x_max, geo_res):\n")
            f.write("    for y in np.arange(y_min, y_max, geo_res):\n")
            f.write("        for z in np.arange(z_min, z_max, geo_res):\n")
            f.write("            if in_ellipsoid(x, y, z, head_center, a+scalp_thickness+coupling_thickness, b+scalp_thickness+coupling_thickness, c+scalp_thickness+coupling_thickness):\n")
            f.write("                material = 'coupling_medium'\n")
            f.write("                if in_ellipsoid(x, y, z, head_center, a+scalp_thickness, b+scalp_thickness, c+scalp_thickness):\n")
            f.write("                    material = 'scalp_skull'\n")
            f.write("                    if in_ellipsoid(x, y, z, head_center, a, b, c):\n")
            f.write("                        material = 'gray_matter'\n")
            f.write("                        if in_ellipsoid(x, y, z, head_center, a-gray_thickness, b-gray_thickness, c-gray_thickness):\n")
            f.write("                            material = 'white_matter'\n")
            f.write("                print(f'#box: {x} {y} {z} {x+geo_res} {y+geo_res} {z+geo_res} {material}')\n")
            f.write("#end_python:\n\n")
            
            # CSF ventricles
            f.write("## CSF Ventricles\n#python:\n")
            f.write("vent_a, vent_b, vent_c = 0.020, 0.010, 0.040\n")
            f.write("vent_left = np.array([head_center[0]-0.0075, head_center[1], head_center[2]])\n")
            f.write("vent_right = np.array([head_center[0]+0.0075, head_center[1], head_center[2]])\n")
            f.write("for x in np.arange(head_center[0]-0.035, head_center[0]+0.035, geo_res):\n")
            f.write("    for y in np.arange(head_center[1]-0.015, head_center[1]+0.015, geo_res):\n")
            f.write("        for z in np.arange(head_center[2]-0.045, head_center[2]+0.045, geo_res):\n")
            f.write("            if in_ellipsoid(x,y,z,vent_left,vent_a,vent_b,vent_c) or in_ellipsoid(x,y,z,vent_right,vent_a,vent_b,vent_c):\n")
            f.write("                print(f'#box: {x} {y} {z} {x+geo_res} {y+geo_res} {z+geo_res} csf')\n")
            f.write("#end_python:\n\n")
            
            # Hemorrhage
            if has_lesion:
                lx = head_center[0] + lesion_pos[0]
                ly = head_center[1] + lesion_pos[1]
                lz = head_center[2] + lesion_pos[2]
                f.write(f"## Hemorrhage\n#sphere: {lx} {ly} {lz} {lesion_size} blood\n\n")
            
            # Antennas: radial inward monopoles at the equatorial plane
            # 16 antennas arranged in a ring around the head at z = head_center[2].
            # Each antenna points radially INWARD toward the head center.
            #
            # Geometry per antenna at angle theta:
            #   feed (cx, cy, cz) : outer surface of coupling layer + 1 cell gap
            #   tip               : feed - monopole_length along dominant axis (inward)
            #   GP                : thin vertical PEC slab (1 cell thick) OUTSIDE the feed,
            #                       spanning gp_half_size in the transverse directions,
            #                       acting as a backplane reflector
            #   TL polarisation   : 'x' if |cos(theta)| >= |sin(theta)|, else 'y'
            #                       (dominant horizontal axis at that angle)
            #
            # This is the only approach compatible with gprMax's axis-aligned TL
            # for all 16 angles in a ring array.
            c_ax = head_semi_axes['c']  # noqa: used for reference only
            cell_size = gp_thickness    # 0.002 m

            f.write("## Waveforms\n")
            f.write(f"#waveform: gaussian 1 1e9 tx_pulse\n")
            f.write(f"#waveform: gaussian 0 1e9 rx_null\n\n")

            # Write a single shared python block that defines all antenna positions
            f.write("## Antenna array (16 radial inward monopoles at equatorial ring)\n")
            f.write("#python:\n")
            f.write("import math\n")
            f.write(f"cell            = {cell_size}\n")
            f.write(f"monopole_length = {monopole_length}\n")
            f.write(f"wire_radius     = {wire_radius}\n")
            f.write(f"gp_half_size    = {gp_half_size}\n")
            f.write(f"gp_thickness    = {gp_thickness}\n")
            f.write(f"n_antennas      = {n_antennas}\n")
            f.write(f"head_center     = ({head_center[0]}, {head_center[1]}, {head_center[2]})\n")
            f.write(f"a               = {head_semi_axes['a']}\n")
            f.write(f"b               = {head_semi_axes['b']}\n")
            f.write(f"scalp_thickness    = {scalp_skull_thickness}\n")
            f.write(f"coupling_thickness = {coupling_thickness}\n")
            f.write("antennas = []  # list of (cx,cy,cz, tip_x,tip_y,tip_z, gp_x1,gp_y1,gp_z1, gp_x2,gp_y2,gp_z2, pol)\n")
            f.write("for i in range(n_antennas):\n")
            f.write("    angle = 2 * math.pi * i / n_antennas\n")
            f.write("    cos_a = math.cos(angle)\n")
            f.write("    sin_a = math.sin(angle)\n")
            f.write("    # Feed at outer surface of coupling layer\n")
            f.write("    r_x = a + scalp_thickness + coupling_thickness + cell\n")
            f.write("    r_y = b + scalp_thickness + coupling_thickness + cell\n")
            f.write("    cx = round((head_center[0] + r_x * cos_a) / cell) * cell\n")
            f.write("    cy = round((head_center[1] + r_y * sin_a) / cell) * cell\n")
            f.write("    cz = head_center[2]\n")
            f.write("    # Dominant axis determines TL polarisation and inward direction\n")
            f.write("    if abs(cos_a) >= abs(sin_a):\n")
            f.write("        pol = 'x'\n")
            f.write("        direction = -1 if cos_a > 0 else 1\n")
            f.write("        tip_x = round((cx + direction * monopole_length) / cell) * cell\n")
            f.write("        tip_y = cy\n")
            f.write("        tip_z = cz\n")
            f.write("        # GP: vertical slab (yz-plane), 1 cell thick, OUTSIDE feed\n")
            f.write("        # outward = -direction; GP occupies [cx+(-dir)*cell, cx+(-dir)*2*cell)\n")
            f.write("        gp_outer = round((cx + (-direction) * gp_thickness) / cell) * cell\n")
            f.write("        gp_x1 = min(cx + (-direction)*cell, gp_outer)\n")
            f.write("        gp_x2 = max(cx + (-direction)*cell, gp_outer) + cell\n")
            f.write("        gp_y1 = round((cy - gp_half_size) / cell) * cell\n")
            f.write("        gp_y2 = round((cy + gp_half_size) / cell) * cell\n")
            f.write("        gp_z1 = round((cz - gp_half_size) / cell) * cell\n")
            f.write("        gp_z2 = round((cz + gp_half_size) / cell) * cell\n")
            f.write("    else:\n")
            f.write("        pol = 'y'\n")
            f.write("        direction = -1 if sin_a > 0 else 1\n")
            f.write("        tip_x = cx\n")
            f.write("        tip_y = round((cy + direction * monopole_length) / cell) * cell\n")
            f.write("        tip_z = cz\n")
            f.write("        # GP: vertical slab (xz-plane), 1 cell thick, OUTSIDE feed\n")
            f.write("        # outward = -direction; GP occupies [cy+(-dir)*cell, cy+(-dir)*2*cell)\n")
            f.write("        gp_outer = round((cy + (-direction) * gp_thickness) / cell) * cell\n")
            f.write("        gp_x1 = round((cx - gp_half_size) / cell) * cell\n")
            f.write("        gp_x2 = round((cx + gp_half_size) / cell) * cell\n")
            f.write("        gp_y1 = min(cy + (-direction)*cell, gp_outer)\n")
            f.write("        gp_y2 = max(cy + (-direction)*cell, gp_outer) + cell\n")
            f.write("        gp_z1 = round((cz - gp_half_size) / cell) * cell\n")
            f.write("        gp_z2 = round((cz + gp_half_size) / cell) * cell\n")
            f.write("    antennas.append((cx,cy,cz, tip_x,tip_y,tip_z, gp_x1,gp_y1,gp_z1, gp_x2,gp_y2,gp_z2, pol))\n")
            f.write("#end_python:\n\n")

            # Per-antenna geometry and TL
            for ant_idx in range(n_antennas):
                f.write(f"## Antenna {ant_idx+1}\n#python:\n")
                f.write(f"cx,cy,cz, tip_x,tip_y,tip_z, gp_x1,gp_y1,gp_z1, gp_x2,gp_y2,gp_z2, pol = antennas[{ant_idx}]\n")
                f.write("# GP: vertical PEC slab (backplane reflector)\n")
                f.write("print(f'#box: {gp_x1} {gp_y1} {gp_z1} {gp_x2} {gp_y2} {gp_z2} pec')\n")
                f.write("# Monopole: radial PEC wire from feed to tip (inward)\n")
                f.write("print(f'#cylinder: {cx} {cy} {cz} {tip_x} {tip_y} {tip_z} {wire_radius} pec')\n")
                if ant_idx == src_idx:
                    f.write("print(f'#transmission_line: {pol} {cx} {cy} {cz} 50 tx_pulse')\n")
                else:
                    f.write("print(f'#transmission_line: {pol} {cx} {cy} {cz} 50 rx_null')\n")
                f.write("#end_python:\n\n")

# ============================================================================
# MAIN
# ============================================================================

print("="*80)
print("BRAIN EMI DATASET GENERATOR - 300 SCENARIOS")
print("="*80)
print(f"\nConfiguration:")
print(f"  Healthy: {NUM_HEALTHY} scenarios")
print(f"  Hemorrhage: {NUM_HEMORRHAGE} scenarios ({len(LESION_SIZES)} sizes x {NUM_POSITIONS} positions)")
print(f"  Total: {NUM_HEALTHY + NUM_HEMORRHAGE} scenarios")
print(f"  Files: {(NUM_HEALTHY + NUM_HEMORRHAGE) * 16} input files")
print("="*80)
print()

# Generate positions
print("Generating hemorrhage positions...")
positions = generate_positions(NUM_POSITIONS)
print(f"  Generated {len(positions)} positions")
print()

scenario_id = 1

# Healthy baselines
print(f"Generating healthy baselines (1-{NUM_HEALTHY})...")
for i in range(NUM_HEALTHY):
    write_scenario(scenario_id, False, 0, (0,0,0))
    print(f"  Scenario {scenario_id:03d} - Healthy", end='\r')
    scenario_id += 1
print(f"  Completed {NUM_HEALTHY} healthy scenarios")
print()

# Hemorrhage cases
print(f"Generating hemorrhage dataset ({NUM_HEALTHY+1}-{NUM_HEALTHY+NUM_HEMORRHAGE})...")
for size in LESION_SIZES:
    for pos in positions:
        if scenario_id <= NUM_HEALTHY + NUM_HEMORRHAGE:
            write_scenario(scenario_id, True, size, pos)
            print(f"  Scenario {scenario_id:03d} - {size*1000:.0f}mm", end='\r')
            scenario_id += 1
print(f"  Completed {NUM_HEMORRHAGE} hemorrhage scenarios")
print()

print("="*80)
print("COMPLETE!")
print("="*80)
print(f"\nCreated {(scenario_id-1)*16} files in {output_dir}/")
print(f"  Scenarios 001-{NUM_HEALTHY:03d}: Healthy")
print(f"  Scenarios {NUM_HEALTHY+1:03d}-{scenario_id-1:03d}: Hemorrhage")
print()
print("NEXT: Update run_simulation.sh line 5:")
print(f"  #SBATCH --array=1-{(scenario_id-1)*16}%16")
print("="*80)
