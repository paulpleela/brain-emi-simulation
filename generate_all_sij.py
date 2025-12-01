"""
Generate S1j..S16j plots from a Touchstone .s16p file.

This script will:
- Find a .s16p file in the current directory (prefer brain_hemorrhage.s16p if present)
- Parse the Touchstone file
- For receiver ports i = 1..16, plot S_ij for j=1..16 and save PNGs in ./sij_plots/

Usage: python generate_all_sij.py

"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
CWD = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(CWD, 'sij_plots')
PREFERRED = os.path.join(CWD, 'brain_hemorrhage.s16p')

# Find s16p file
candidates = glob.glob(os.path.join(CWD, '*.s16p'))
if os.path.exists(PREFERRED):
    s16p_file = PREFERRED
elif candidates:
    s16p_file = candidates[0]
else:
    raise FileNotFoundError('No .s16p file found in workspace. Please generate a .s16p file first.')

print(f"Using S16P file: {s16p_file}")

# Read s16p file
freqs = []
S_matrix_rows = []  # We'll store rows per receiver sequentially: list of 16 lists, each a list of 16 lists of mags(dB)

# Initialize storage for 16 rows x 16 cols, each will be a list of magnitudes per frequency
S = [[[ ] for _ in range(16)] for _ in range(16)]  # S[i][j] -> list of mag_dB per freq

with open(s16p_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('!') or line.startswith('#'):
            continue
        parts = line.split()
        # First token is frequency in Hz
        try:
            freq = float(parts[0])
        except ValueError:
            continue
        freqs.append(freq)

        # After freq, the line contains 16*16*2 numeric tokens for a full S-matrix row-major
        vals = [float(x) for x in parts[1:]]
        # There should be 16*16*2 = 512 values (mag, ang) but some Touchstone files use linear/real-imag formats; we'll assume mag-angle as produced earlier
        if len(vals) < 512:
            # If file stored only a subset or different ordering, try to continue gracefully
            # We'll pad with zeros to avoid index errors
            vals += [0.0] * (512 - len(vals))

        # For each i (row) and j (col), compute index
        # Ordering used earlier: rows sequentially, each S_ij has two numbers (mag, ang)
        # Position of element S(i+1,j+1) in vals: offset = (i*16 + j) * 2
        for i in range(16):
            for j in range(16):
                idx = (i * 16 + j) * 2
                mag = vals[idx]
                ang = vals[idx+1]
                # convert mag to dB, avoid log(0)
                mag_db = 20.0 * np.log10(mag + 1e-12)
                S[i][j].append(mag_db)

freqs = np.array(freqs)

if len(freqs) == 0:
    raise RuntimeError('No frequency points parsed from the .s16p file.')

print(f"Parsed {len(freqs)} frequency points (range {freqs[0]*1e-9:.3f} - {freqs[-1]*1e-9:.3f} GHz)")

# Make output dir
os.makedirs(OUT_DIR, exist_ok=True)

# Distance from each receiver to lesion (computed from geometry check)
# Lesion at (0.12, 0.15, 0.16)
rx_to_lesion_dist = {
    1: 0.130, 2: 0.128, 3: 0.122, 4: 0.115, 5: 0.105, 6: 0.093,
    7: 0.081, 8: 0.073, 9: 0.071, 10: 0.073, 11: 0.081, 12: 0.093,
    13: 0.105, 14: 0.115, 15: 0.122, 16: 0.128
}

# Plotting function
def plot_row(i):
    # i is 0-indexed receiver port
    port_num = i + 1
    plt.figure(figsize=(11, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    for j in range(16):
        label = f'S{port_num},{j+1}'
        linewidth = 2.0 if (j == i) else 1.5
        linestyle = '--' if (j == i) else '-'
        plt.plot(freqs * 1e-9, S[i][j], label=label, color=colors[j], linewidth=linewidth, linestyle=linestyle, alpha=0.9)

    plt.xlabel('Frequency (GHz)')
    plt.ylabel(f'|S{port_num}j| (dB)')
    
    # Add distance info to title
    dist = rx_to_lesion_dist.get(port_num, 0)
    dist_label = f' (dist to lesion: {dist:.3f} m)'
    plt.title(f'S{port_num}j: Signals received at port {port_num} from ports 1..16{dist_label}')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9, loc='best')
    # set xlim to 0-3 GHz if available, else full
    if freqs[-1] >= 3e9:
        plt.xlim(0, 3)

    outname = os.path.join(OUT_DIR, f's{i+1}j.png')
    plt.tight_layout()
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close()
    return outname

# Generate plots for i=1..16
results = []
for i in range(16):
    print(f'Plotting S{i+1}j ...')
    out = plot_row(i)
    size = os.path.getsize(out)
    mtime = datetime.fromtimestamp(os.path.getmtime(out)).isoformat()
    results.append((out, size, mtime))

# Report
print('\nGenerated images:')
for path, size, mtime in results:
    print(f' - {os.path.basename(path)} : {size} bytes, modified {mtime}')

print(f'All plots saved to: {OUT_DIR}')
