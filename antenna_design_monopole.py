"""
Design a simple monopole antenna for 1 GHz brain imaging.

A monopole is chosen because:
- Simple geometry (vertical wire above ground plane)
- Quarter-wave length at resonance
- Easy to model with gprMax wire/edge commands
- Provides defined 50Ω port impedance when properly matched

Antenna specifications:
- Center frequency: 1 GHz
- Type: Quarter-wave monopole
- Ground plane: Small square (3λ/4 × 3λ/4 to keep compact)
- Feed: Transmission line port with 50Ω impedance
"""

import math

# Constants
c0 = 3e8  # speed of light (m/s)
f_center = 1e9  # 1 GHz
wavelength = c0 / f_center  # 0.3 m = 300 mm

# Monopole dimensions
monopole_length = wavelength / 4  # Quarter-wave
wire_radius = 0.001  # 1 mm radius wire (thin wire approximation)

# Ground plane dimensions
# Using 3λ/4 × 3λ/4 to balance compactness and performance
gp_size = 3 * wavelength / 4  # ~225 mm

# Grid resolution (must match simulation)
dx = dy = dz = 0.002  # 2 mm

print("="*70)
print("MONOPOLE ANTENNA DESIGN FOR 1 GHz BRAIN IMAGING")
print("="*70)
print(f"\nCenter frequency: {f_center/1e9:.1f} GHz")
print(f"Free-space wavelength: {wavelength*1000:.1f} mm")
print(f"\nMonopole specifications:")
print(f"  Length: λ/4 = {monopole_length*1000:.1f} mm")
print(f"  Wire radius: {wire_radius*1000:.1f} mm")
print(f"  Cells along monopole: {int(monopole_length/dz)}")
print(f"\nGround plane:")
print(f"  Size: {gp_size*1000:.1f} × {gp_size*1000:.1f} mm")
print(f"  Thickness: {dz*1000:.1f} mm (1 cell)")
print(f"\nFeed point:")
print(f"  Type: #transmission_line with 50Ω impedance")
print(f"  Position: Center of ground plane, bottom of monopole")
print(f"  Polarization: z (vertical)")

# Calculate positioning for 16 antennas around head
head_center = (0.15, 0.15, 0.15)  # meters
sensor_ring_radius = 0.1  # 10 cm from center
n_antennas = 16

print(f"\n" + "="*70)
print("ANTENNA ARRAY CONFIGURATION")
print("="*70)
print(f"Number of antennas: {n_antennas}")
print(f"Ring radius: {sensor_ring_radius*1000:.0f} mm")
print(f"Head center: ({head_center[0]}, {head_center[1]}, {head_center[2]})")

# Calculate antenna positions
antenna_positions = []
for i in range(n_antennas):
    angle = 2 * math.pi * i / n_antennas
    x = head_center[0] + sensor_ring_radius * math.cos(angle)
    y = head_center[1] + sensor_ring_radius * math.sin(angle)
    z = head_center[2]  # Same z as head center
    antenna_positions.append((x, y, z))
    print(f"  Antenna {i+1:02d}: ({x:.6f}, {y:.6f}, {z:.6f}) @ {math.degrees(angle):.1f}°")

print(f"\n" + "="*70)
print("IMPLEMENTATION NOTES")
print("="*70)
print("""
1. Ground plane modeling:
   - Use #box with PEC material
   - Centered at each antenna position
   - Horizontal (x-y plane)

2. Monopole element:
   - Use #edge (thin wire) or #cylinder
   - Vertical orientation (z-direction)
   - Base at ground plane surface

3. Feed point:
   - #transmission_line: z <x> <y> <z> 50 <waveform_id>
   - Position: base of monopole, on ground plane
   - 50Ω characteristic impedance

4. Receiver configuration:
   - For transmit antenna: use #transmission_line as source
   - For receive antennas: use #transmission_line with zero waveform
     OR use #rx_array to measure fields and integrate for voltage

5. Domain size:
   - Must accommodate all antennas + ground planes + head + PML
   - Suggest: 0.4 × 0.4 × 0.4 m (increase from 0.3 m)
   - PML: 10 cells (20 mm) on all sides

6. Calibration procedure:
   a) Simulate single antenna in free space → get S11_antenna
   b) Simulate antenna pair in free space → get S21_antenna
   c) Use these to de-embed antenna effects from head measurements
""")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Create calibration input file (antenna in free space)")
print("2. Create brain imaging input with monopole antennas")
print("3. Run simulations and extract port voltages/currents")
print("4. Apply de-embedding to get calibrated S-parameters")
print("="*70)
