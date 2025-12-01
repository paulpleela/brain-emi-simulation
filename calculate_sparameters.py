"""
Calculate S-parameters from gprMax time-domain output file.

This script:
1. Reads time-domain E and H field data from .out HDF5 file
2. Applies FFT to convert to frequency domain
3. Calculates S-parameters (reflection and transmission coefficients)

For brain imaging: S11 (reflection at each sensor) and S21 (transmission between sensors)
are useful for detecting hemorrhage lesions.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Configuration
output_file = r'C:\Users\paudo\OneDrive\Documents\Thesis gprmax\simple.out'
z0 = 376.73  # Free space impedance (Ohms)

# Read the HDF5 file
print("Reading gprMax output file...")
with h5py.File(output_file, 'r') as f:
    # Get simulation parameters
    dt = f.attrs['dt']  # Time step (seconds)
    iterations = f.attrs['Iterations']
    
    print(f"Time step: {dt*1e12:.3f} ps")
    print(f"Iterations: {iterations}")
    print(f"Time window: {dt*iterations*1e9:.3f} ns")
    
    # Get number of receivers
    num_receivers = len([k for k in f['rxs'].keys() if k.startswith('rx')])
    print(f"Number of receivers: {num_receivers}")
    
    # Read all receiver data
    receivers_E = []
    receivers_H = []
    
    for i in range(1, num_receivers + 1):
        rx_group = f[f'rxs/rx{i}']
        
        # For S-parameters, we typically use the component normal to the antenna
        # Here we'll use Ez (vertical component) and Hx (horizontal magnetic)
        Ez = rx_group['Ez'][:]
        Hx = rx_group['Hx'][:]
        
        receivers_E.append(Ez)
        receivers_H.append(Hx)
    
    receivers_E = np.array(receivers_E)  # Shape: (16, 2598)
    receivers_H = np.array(receivers_H)

print(f"\nReceiver data shape: {receivers_E.shape}")

# Create time array
time = np.arange(iterations) * dt

# Apply FFT to convert to frequency domain
print("\nApplying FFT...")
freq = np.fft.rfftfreq(iterations, dt)  # Positive frequencies only
E_freq = np.fft.rfft(receivers_E, axis=1)  # FFT along time axis
H_freq = np.fft.rfft(receivers_H, axis=1)

print(f"Frequency array shape: {freq.shape}")
print(f"Frequency range: {freq[0]*1e-9:.3f} to {freq[-1]*1e-9:.3f} GHz")

# Calculate voltage and current from E and H fields
# V ~ E * length (approximation)
# I ~ H * length
# For normalized measurements, we can use E and H directly

# Calculate impedance Z = E/H
# Avoid division by zero
H_freq_safe = np.where(np.abs(H_freq) < 1e-20, 1e-20, H_freq)
Z_freq = E_freq / H_freq_safe

# Calculate S11 (reflection coefficient) for each receiver
# S11 = (Z - Z0) / (Z + Z0)
S11 = (Z_freq - z0) / (Z_freq + z0)

# Calculate S21 (transmission between receivers)
# S21[i,j] = E_j / E_i (assuming receiver i is excited)
# For brain imaging, we use the source position as reference
# S21 represents how well signal transmits from source to each receiver

# Simplified S21: ratio of received field to incident field
# Using receiver 1 as reference (closest to source at 0.24, 0.15, 0.15)
E_incident = E_freq[0, :]  # Receiver 1 (closest to source)
E_incident_safe = np.where(np.abs(E_incident) < 1e-20, 1e-20, E_incident)

S21 = np.zeros_like(E_freq)
for i in range(num_receivers):
    S21[i, :] = E_freq[i, :] / E_incident_safe

# Convert to dB
S11_dB = 20 * np.log10(np.abs(S11) + 1e-12)
S21_dB = 20 * np.log10(np.abs(S21) + 1e-12)

print("\nS-parameter calculation complete!")
print(f"S11 shape: {S11.shape} (16 receivers × {len(freq)} frequencies)")
print(f"S21 shape: {S21.shape}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: S11 magnitude for all receivers
ax = axes[0, 0]
freq_GHz = freq * 1e-9
for i in range(num_receivers):
    ax.plot(freq_GHz, S11_dB[i, :], alpha=0.7, label=f'Rx{i+1}' if i < 4 else '')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('|S11| (dB)')
ax.set_title('S11 (Reflection) - All 16 Receivers')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 3])  # Focus on 0-3 GHz
ax.legend()

# Plot 2: S21 magnitude for selected receivers
ax = axes[0, 1]
selected_rx = [1, 4, 7, 10, 13, 16]  # Show every ~3rd receiver
for i in selected_rx:
    ax.plot(freq_GHz, S21_dB[i-1, :], label=f'Rx{i}')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('|S21| (dB)')
ax.set_title('S21 (Transmission) - Selected Receivers')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 3])
ax.legend()

# Plot 3: S11 phase for all receivers at 1 GHz
ax = axes[1, 0]
target_freq = 1e9  # 1 GHz (source frequency)
freq_idx = np.argmin(np.abs(freq - target_freq))
actual_freq = freq[freq_idx] * 1e-9
S11_phase_at_1GHz = np.angle(S11[:, freq_idx], deg=True)
rx_numbers = np.arange(1, num_receivers + 1)
ax.bar(rx_numbers, S11_phase_at_1GHz, color='steelblue', alpha=0.7)
ax.set_xlabel('Receiver Number')
ax.set_ylabel('Phase (degrees)')
ax.set_title(f'S11 Phase at {actual_freq:.2f} GHz')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: S21 magnitude at 1 GHz (spatial distribution)
ax = axes[1, 1]
S21_mag_at_1GHz = np.abs(S21[:, freq_idx])
ax.bar(rx_numbers, 20*np.log10(S21_mag_at_1GHz + 1e-12), color='coral', alpha=0.7)
ax.set_xlabel('Receiver Number')
ax.set_ylabel('|S21| (dB)')
ax.set_title(f'S21 Magnitude at {actual_freq:.2f} GHz (Spatial Response)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(r'C:\Users\paudo\OneDrive\Documents\Thesis gprmax\s_parameters.png', dpi=150)
print(f"\nPlot saved: s_parameters.png")
plt.show()

# Save S-parameters to text files (Touchstone format)
print("\nSaving S-parameters to Touchstone format...")

# Save S11 for all receivers (individual .s1p files)
for i in range(num_receivers):
    filename = f'C:\\Users\\paudo\\OneDrive\\Documents\\Thesis gprmax\\rx{i+1}.s1p'
    with open(filename, 'w') as f:
        f.write(f'! S-parameter data for Receiver {i+1}\n')
        f.write(f'! Extracted from gprMax simulation: brain_hemorrhage_16sensors\n')
        f.write(f'! Frequency (Hz)  |S11| (mag)  Angle(S11) (deg)\n')
        f.write('# Hz S MA R 50\n')  # Frequency in Hz, S-parameters, Magnitude-Angle, 50 Ohm reference
        
        for j in range(len(freq)):
            mag = np.abs(S11[i, j])
            phase = np.angle(S11[i, j], deg=True)
            f.write(f'{freq[j]:.6e} {mag:.6e} {phase:.6e}\n')
    
    if i == 0:
        print(f"  rx1.s1p (example)")
    elif i == num_receivers - 1:
        print(f"  rx{num_receivers}.s1p")

# Save full 16-port S-matrix at key frequencies
key_freqs_GHz = [0.5, 1.0, 1.5, 2.0]  # Key frequencies to analyze
print(f"\nS-parameter matrix at key frequencies:")

for f_GHz in key_freqs_GHz:
    f_target = f_GHz * 1e9
    f_idx = np.argmin(np.abs(freq - f_target))
    f_actual = freq[f_idx] * 1e-9
    
    print(f"\n  @ {f_actual:.2f} GHz:")
    print(f"    Average |S11|: {np.mean(np.abs(S11[:, f_idx])):.4f} ({20*np.log10(np.mean(np.abs(S11[:, f_idx]))+1e-12):.2f} dB)")
    print(f"    Average |S21|: {np.mean(np.abs(S21[:, f_idx])):.4f} ({20*np.log10(np.mean(np.abs(S21[:, f_idx]))+1e-12):.2f} dB)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Calculated S-parameters from {iterations} time-domain samples")
print(f"✓ Frequency range: DC to {freq[-1]*1e-9:.2f} GHz")
print(f"✓ Generated 16 individual .s1p files (S11 for each receiver)")
print(f"✓ S11: Reflection coefficient (brain tissue impedance mismatch)")
print(f"✓ S21: Transmission coefficient (signal through brain/lesion)")
print(f"✓ Visualization saved: s_parameters.png")
print("="*60)
print("\nNext steps for hemorrhage detection:")
print("  1. Compare S11/S21 with baseline (healthy brain) simulation")
print("  2. Look for anomalies around hemorrhage location (receivers 12-16)")
print("  3. Use S-parameter differences for image reconstruction")
print("  4. Analyze frequency dependence (blood has high εr, affects resonance)")
