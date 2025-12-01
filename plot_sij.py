"""
Plot any S-parameter row: S_ij where i is fixed and j varies from 1 to 16.

For example:
- S1j: Transmission from all ports to port 1 (j=1..16)
- S5j: Transmission from all ports to port 5 (j=1..16)
- etc.
"""

import numpy as np
import matplotlib.pyplot as plt

# Configuration
s16p_file = r'C:\Users\paudo\OneDrive\Documents\Thesis gprmax\brain_hemorrhage.s16p'
receiver_port = 12  # Which port to receive at (i in S_ij)

print(f"Plotting S{receiver_port}j (j=1..16)")
print(f"This shows: signals received at port {receiver_port} from all 16 source ports\n")

# Read S16P file
print("Reading S16P file...")
freqs = []
S_row = [[] for _ in range(16)]  # Store S_ij for j=1..16

with open(s16p_file, 'r') as f:
    for line in f:
        line = line.strip()
        
        # Skip comments and option line
        if line.startswith('!') or line.startswith('#') or not line:
            continue
        
        # Parse data line
        parts = line.split()
        freq = float(parts[0])
        freqs.append(freq)
        
        # Extract S-parameters for receiver_port
        # S-matrix is stored row by row: S11 S12 ... S1,16, S21 S22 ... S2,16, ...
        # For receiver port i (0-indexed: i-1), we want all j=0..15
        # Position: 1 + (i-1)*16*2 + j*2
        
        i = receiver_port - 1  # Convert to 0-indexed
        
        for j in range(16):
            # Position in the line for S_(i+1)(j+1)
            idx = 1 + i * 32 + j * 2  # 32 = 16 ports * 2 values/port (mag, ang)
            
            if idx+1 < len(parts):
                mag = float(parts[idx])
                ang_deg = float(parts[idx+1])
                
                # Convert to dB
                mag_dB = 20 * np.log10(mag + 1e-12)
                S_row[j].append(mag_dB)

freqs = np.array(freqs)

print(f"Loaded {len(freqs)} frequency points")
print(f"Frequency range: {freqs[0]*1e-9:.3f} to {freqs[-1]*1e-9:.3f} GHz\n")

# Create the plot
plt.figure(figsize=(12, 7))

# Plot all 16 S-parameters
colors = plt.cm.tab20(np.linspace(0, 1, 16))

for j in range(16):
    label = f'S{receiver_port},{j+1}'
    if j+1 == receiver_port:
        label += ' (reflection)'
    
    plt.plot(freqs * 1e-9, S_row[j], 
             label=label, 
             linewidth=2 if j+1 == receiver_port else 1.5,  # Thicker line for reflection
             color=colors[j],
             alpha=0.8,
             linestyle='--' if j+1 == receiver_port else '-')  # Dashed for reflection

plt.xlabel('Frequency (GHz)', fontsize=12)
plt.ylabel(f'|S{receiver_port}j| (dB)', fontsize=12)
plt.title(f'S{receiver_port}j: Signals Received at Port {receiver_port} from All Source Ports', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim([0, 3])  # Focus on 0-3 GHz range
plt.legend(ncol=2, fontsize=9, loc='best')

# Add annotation with accurate distance info
# Lesion is at (0.12, 0.15, 0.16)
# Closest receivers: RX9 (0.071m), RX8/RX10 (0.073m), RX7/RX11 (0.081m)
info_text = f'Port {receiver_port} position: '
if receiver_port == 9:
    info_text += 'closest to hemorrhage (0.071 m)'
elif receiver_port in [8, 10]:
    info_text += 'very close to hemorrhage (0.073 m)'
elif receiver_port in [7, 11]:
    info_text += 'near hemorrhage (0.081 m)'
elif receiver_port == 1:
    info_text += 'far from hemorrhage (0.130 m)'
else:
    info_text += f'distance to hemorrhage: varies'

plt.text(0.02, 0.98, info_text, 
         transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save
output_file = rf'C:\Users\paudo\OneDrive\Documents\Thesis gprmax\s{receiver_port}j_all_sources.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved: {output_file}")

plt.show()

print("\n" + "="*70)
print("INTERPRETATION GUIDE")
print("="*70)
print(f"S{receiver_port},{receiver_port} (dashed line): Reflection at port {receiver_port}")
print(f"S{receiver_port},j (j≠{receiver_port}): Transmission from port j to port {receiver_port}")
print("\nTo plot different receiver port, change 'receiver_port' at top of script:")
print(f"  receiver_port = {receiver_port}  <- change this number (1-16)")
print("\nUseful comparisons:")
print("  Port 1: Far from hemorrhage (baseline)")
print("  Port 12-13: Close to hemorrhage (should show anomalies)")
print("="*70)
