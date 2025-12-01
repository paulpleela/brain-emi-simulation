"""
Visualize realistic brain imaging model with coupling medium layer.

The script parses the realistic input file (brain_monopole_realistic/brain_realistic_tx01.in),
extracts sphere layers and transmission_line ports, and creates a 2D cross-section diagram
showing the multi-layer head structure with coupling medium and antenna positions.

Usage: python visualize_head.py
"""

import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

CWD = os.path.abspath(os.path.dirname(__file__))
# Use the realistic model input file
preferred = os.path.join(CWD, 'brain_monopole_realistic', 'brain_realistic_tx01.in')

if not os.path.exists(preferred):
    raise FileNotFoundError(f'Realistic model input file not found: {preferred}')

print(f'Reading input file: {preferred}')
in_file = preferred

spheres = []  # list of dicts {x,y,z,r,material}
antenna_positions = []  # list of (x,y,z) from transmission_line ports

with open(in_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#sphere:'):
            parts = line[len('#sphere:'):].strip().split()
            if len(parts) >= 5:
                x, y, z, r = map(float, parts[:4])
                mat = parts[4] if len(parts) > 4 else ''
                spheres.append({'x': x, 'y': y, 'z': z, 'r': r, 'mat': mat})
        elif line.startswith('#transmission_line:'):
            # Parse: #transmission_line: z {x} {y} {feed_z} 50 tx_pulse
            # These have Python variables in {}, so we need to look for patterns
            parts = line[len('#transmission_line:'):].strip().split()
            if len(parts) >= 4:
                # Try to extract x, y from the line (they appear as {x} {y} in the file)
                # We'll skip these and instead compute from the geometry parameters
                pass

# Since antenna positions use Python variables, we need to compute them
# based on the parameters in the file
head_center = np.array([0.25, 0.25, 0.25])
head_semi_axes = {'a': 0.095, 'b': 0.075, 'c': 0.115}
scalp_skull_thickness = 0.010
coupling_thickness = 0.005
gp_half_size = 0.0375
n_antennas = 16

# Compute antenna positions (same logic as in generation script)
avg_radius = (head_semi_axes['a'] + head_semi_axes['b'] + head_semi_axes['c']) / 3
for i in range(n_antennas):
    angle = 2 * math.pi * i / n_antennas
    x = head_center[0] + (avg_radius + scalp_skull_thickness + coupling_thickness + gp_half_size) * math.cos(angle)
    y = head_center[1] + (avg_radius + scalp_skull_thickness + coupling_thickness + gp_half_size) * math.sin(angle)
    z = head_center[2]
    antenna_positions.append((x, y, z))

if len(antenna_positions) == 0:
    print('Warning: No antenna positions computed.')

# We'll visualize the slice at the antenna z-plane
slice_z = head_center[2]
print(f'Using slice z = {slice_z:.5f} m (equatorial plane)')

# Prepare figure
fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect('equal')

# Colors per material (updated for realistic model with coupling medium)
colors = {
    'coupling_medium': '#90ee90',  # light green
    'scalp_skull': '#f4c2c2',      # light pink
    'gray_matter': '#c9c9ff',      # light blue
    'white_matter': '#fff5cc',     # light yellow
    'blood': '#b30000',            # dark red
}

# Draw sphere intersections as circles at slice_z
max_extent = 0.0
for sph in spheres:
    dz = slice_z - sph['z']
    if abs(dz) <= sph['r']:
        r_slice = math.sqrt(max(0.0, sph['r']**2 - dz**2))
        color = colors.get(sph['mat'], '#cccccc')
        circ = Circle((sph['x'], sph['y']), r_slice, facecolor=color, edgecolor='k', alpha=0.6, linewidth=1.2)
        ax.add_patch(circ)
        max_extent = max(max_extent, sph['x']+r_slice, sph['y']+r_slice, abs(sph['x']-r_slice), abs(sph['y']-r_slice))

# Plot antennas as small markers with ground planes
# Ground plane size (visualized as small rectangles)
gp_visual_size = gp_half_size * 0.8  # Slightly smaller for clarity
for idx, (x, y, z) in enumerate(antenna_positions, start=1):
    # Draw small ground plane indicator
    gp_rect = Rectangle((x - gp_visual_size/2, y - gp_visual_size/2), 
                         gp_visual_size, gp_visual_size, 
                         facecolor='#333333', edgecolor='k', linewidth=0.5, 
                         alpha=0.4, zorder=10)
    ax.add_patch(gp_rect)
    
    # Draw antenna feed point
    antenna_marker = Circle((x, y), 0.003, facecolor='#ff6600', edgecolor='k', 
                           linewidth=0.8, zorder=12)
    ax.add_patch(antenna_marker)
    ax.text(x, y + 0.015, f'A{idx}', fontsize=8, ha='center', zorder=13, 
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# Center marker
# If there is a sphere centered at some point, mark center
if spheres:
    center = (spheres[0]['x'], spheres[0]['y'])
    ax.plot(center[0], center[1], 'kx', markersize=8)

# Title and legend entries
legend_handles = []
from matplotlib.patches import Patch
for name, col in colors.items():
    # only add if present in spheres
    if any(s['mat'] == name for s in spheres):
        legend_handles.append(Patch(facecolor=col, edgecolor='k', 
                                   label=name.replace('_',' ').title(), alpha=0.6))

# Add antenna marker to legend
legend_handles.append(Patch(facecolor='#ff6600', edgecolor='k', 
                           label='Antenna Feed (50Ω)', alpha=1.0))
legend_handles.append(Patch(facecolor='#333333', edgecolor='k', 
                           label='Ground Plane', alpha=0.4))

if legend_handles:
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)

# Set limits so the outer (largest) circle is fully visible and centered
if spheres:
    cx, cy = spheres[0]['x'], spheres[0]['y']
    # find the largest slice radius drawn
    max_r_slice = 0.0
    for sph in spheres:
        dz = slice_z - sph['z']
        if abs(dz) <= sph['r']:
            r_slice = math.sqrt(max(0.0, sph['r'] ** 2 - dz ** 2))
            if r_slice > max_r_slice:
                max_r_slice = r_slice

    if max_r_slice <= 0:
        max_r_slice = 0.12 / 2.0

    margin = max(0.02, max_r_slice * 0.12)  # add a small padding
    ax.set_xlim(cx - (max_r_slice + margin), cx + (max_r_slice + margin))
    ax.set_ylim(cy - (max_r_slice + margin), cy + (max_r_slice + margin))
else:
    ax.autoscale()

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Realistic Brain Imaging Model - Equatorial Cross-Section\n' + 
            'Monopole Array (0-2 GHz) with Coupling Medium', fontsize=11, weight='bold')
ax.grid(True, alpha=0.3)

# Add text annotation with key parameters
param_text = (f'Domain: 600×600×600 mm\n'
             f'Mesh: 2 mm\n'
             f'Antennas: 16 monopoles (37.5 mm)\n'
             f'Coupling: 5 mm tissue-equiv.\n'
             f'Head: avg radius ~95 mm')
ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
       fontsize=8, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save
outdir = os.path.join(CWD, 'figures')
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'realistic_head_diagram.png')
plt.tight_layout()
plt.savefig(outfile, dpi=200)
print(f'Saved realistic head diagram: {outfile}')
plt.close()

print('\nRealistic model visualization complete!')
print(f'  - {len(spheres)} tissue layers (including coupling medium)')
print(f'  - {len(antenna_positions)} monopole antennas with ground planes')
print(f'  - Frequency range: 0-2 GHz')
print(f'  - Monopole length: 37.5 mm (λ/4 @ 2 GHz)')

