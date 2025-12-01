"""
Multi-static simulation script for generating 16-port S-parameters (.s16p file).

This script:
1. Generates 16 input files (one for each source position at receiver locations)
2. Runs all 16 gprMax simulations
3. Extracts time-domain data from all .out files
4. Calculates full 16×16 S-parameter matrix
5. Saves as .s16p Touchstone file

For brain hemorrhage imaging: Full S-matrix enables microwave tomography reconstruction.
"""

import os
import subprocess
import h5py
import numpy as np
from pathlib import Path

# Configuration
base_dir = Path(r'C:\Users\paudo\OneDrive\Documents\Thesis gprmax')
output_dir = base_dir / 's16p_simulations'
output_dir.mkdir(exist_ok=True)

# Receiver positions (from simple.in - circular array)
receiver_positions = [
    (0.250, 0.150, 0.150),  # rx1
    (0.242, 0.188, 0.150),  # rx2
    (0.220, 0.220, 0.150),  # rx3
    (0.188, 0.242, 0.150),  # rx4
    (0.150, 0.250, 0.150),  # rx5
    (0.112, 0.242, 0.150),  # rx6
    (0.080, 0.220, 0.150),  # rx7
    (0.058, 0.188, 0.150),  # rx8
    (0.050, 0.150, 0.150),  # rx9
    (0.058, 0.112, 0.150),  # rx10
    (0.080, 0.080, 0.150),  # rx11
    (0.112, 0.058, 0.150),  # rx12
    (0.150, 0.050, 0.150),  # rx13
    (0.188, 0.058, 0.150),  # rx14
    (0.220, 0.080, 0.150),  # rx15
    (0.242, 0.112, 0.150),  # rx16
]

num_ports = len(receiver_positions)

# Simulation parameters
z0 = 376.73  # Free space impedance (Ohms)
freq_start = 0.5e9  # 500 MHz
freq_stop = 3.0e9   # 3 GHz
freq_center = 1.0e9 # 1 GHz

print("="*70)
print("MULTI-STATIC S16P GENERATION SCRIPT")
print("="*70)
print(f"Output directory: {output_dir}")
print(f"Number of ports (receivers): {num_ports}")
print(f"Simulations to run: {num_ports}")
print(f"Expected total time: ~{num_ports * 2.7:.0f} minutes (@ ~2.7 min/sim)")
print("="*70)

def generate_input_file(source_idx, source_pos):
    """Generate gprMax input file with source at given position."""
    
    x, y, z = source_pos
    filename = output_dir / f'brain_source_{source_idx+1:02d}.in'
    
    content = f"""## Brain hemorrhage model - Source position {source_idx+1}/{num_ports}
## Multi-static configuration for S16P generation
#title: brain_s16p_source_{source_idx+1:02d}

## Domain and discretization
#domain: 0.3 0.3 0.3
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 10e-9

## Excitation waveform
#waveform: gaussian 1 1e9 my_pulse

## Source at receiver position {source_idx+1}
#hertzian_dipole: z {x} {y} {z} my_pulse

## All 16 receivers (circular array)
"""
    
    # Add all 16 receivers
    for i, (rx, ry, rz) in enumerate(receiver_positions):
        content += f"#rx: {rx} {ry} {rz}\n"
    
    # Add materials
    content += """
## Brain tissue materials (realistic dielectric properties at ~1 GHz)
#material: 12 0.2 1 0 scalp_skull
#material: 52 0.97 1 0 gray_matter
#material: 38 0.57 1 0 white_matter
#material: 61 1.54 1 0 blood

## Geometry - concentric spheres for head model
## Outer skull/scalp
#sphere: 0.15 0.15 0.15 0.09 scalp_skull

## Gray matter
#sphere: 0.15 0.15 0.15 0.08 gray_matter

## White matter core
#sphere: 0.15 0.15 0.15 0.06 white_matter

## Hemorrhagic lesion (blood-filled cavity, offset from center)
#sphere: 0.12 0.15 0.16 0.015 blood
#sphere: 0.124 0.15 0.16 0.01 blood

## End of input file
"""
    
    with open(filename, 'w') as f:
        f.write(content)
    
    return filename


def run_simulation(input_file):
    """Run gprMax simulation."""
    print(f"  Running: {input_file.name}")
    
    try:
        result = subprocess.run(
            ['gprmax', str(input_file)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per simulation
        )
        
        if result.returncode != 0:
            print(f"    ERROR: Simulation failed!")
            print(result.stderr)
            return False
        
        print(f"    ✓ Complete")
        return True
    
    except subprocess.TimeoutExpired:
        print(f"    ERROR: Simulation timed out!")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def extract_field_data(output_file):
    """Extract E and H field data from gprMax output file."""
    
    with h5py.File(output_file, 'r') as f:
        dt = f.attrs['dt']
        iterations = f.attrs['Iterations']
        
        # Extract data for all 16 receivers
        E_data = []
        H_data = []
        
        for i in range(1, num_ports + 1):
            rx_group = f[f'rxs/rx{i}']
            
            # Use Ez and Hx for vertical dipole excitation
            Ez = rx_group['Ez'][:]
            Hx = rx_group['Hx'][:]
            
            E_data.append(Ez)
            H_data.append(Hx)
        
        E_data = np.array(E_data)  # Shape: (16, iterations)
        H_data = np.array(H_data)
    
    return E_data, H_data, dt, iterations


def calculate_s_matrix_column(E_data, H_data, dt, iterations, source_idx):
    """
    Calculate one column of the S-matrix (all S_ij where j=source_idx).
    
    Returns S-parameters at all frequencies for this source position.
    """
    
    # FFT to frequency domain
    freq = np.fft.rfftfreq(iterations, dt)
    E_freq = np.fft.rfft(E_data, axis=1)  # Shape: (16, freq_bins)
    
    # Get incident voltage (at source position = receiver source_idx)
    V_incident = E_freq[source_idx, :]
    
    # Prevent division by zero
    V_incident_safe = np.where(np.abs(V_incident) < 1e-20, 1e-20, V_incident)
    
    # Calculate S-parameters: S_ij = V_j / V_incident (when port i is excited)
    # Column j of S-matrix represents all receivers when source j is excited
    S_column = E_freq / V_incident_safe  # Broadcasting: (16, freq) / (freq,)
    
    return S_column, freq


def save_s16p_file(S_matrix, freq, filename):
    """
    Save 16×16 S-parameter matrix to Touchstone .s16p file.
    
    S_matrix shape: (16, 16, freq_bins)
    Format: Touchstone 16-port file
    """
    
    print(f"\nSaving S16P file: {filename}")
    
    with open(filename, 'w') as f:
        # Header
        f.write('! 16-port S-parameter data\n')
        f.write('! Generated from gprMax brain hemorrhage simulation\n')
        f.write('! Multi-static measurement: 16 source positions × 16 receivers\n')
        f.write('! Geometry: Circular array around head model with hemorrhagic lesion\n')
        f.write(f'! Frequency points: {len(freq)}\n')
        f.write(f'! Frequency range: {freq[0]*1e-9:.3f} - {freq[-1]*1e-9:.3f} GHz\n')
        f.write('!\n')
        f.write('# HZ S MA R 50\n')  # Frequency in Hz, S-params, Magnitude-Angle, 50Ω ref
        f.write('!\n')
        
        # Data: Each frequency point followed by all 256 S-parameters (16×16)
        # Format: S11_mag S11_ang S12_mag S12_ang ... S1,16_mag S1,16_ang
        #         S21_mag S21_ang S22_mag S22_ang ... S2,16_mag S2,16_ang
        #         ...
        #         S16,1_mag S16,1_ang ... S16,16_mag S16,16_ang
        
        for f_idx in range(len(freq)):
            # Write frequency
            f.write(f'{freq[f_idx]:.6e}')
            
            # Write all 256 S-parameters (16×16) in row-major order
            for i in range(16):  # Receiver (row)
                for j in range(16):  # Source/port (column)
                    mag = np.abs(S_matrix[i, j, f_idx])
                    phase = np.angle(S_matrix[i, j, f_idx], deg=True)
                    f.write(f' {mag:.6e} {phase:.6e}')
            
            f.write('\n')
    
    print(f"  ✓ S16P file saved: {len(freq)} frequency points")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*70)
print("STEP 1: GENERATING INPUT FILES")
print("="*70)

input_files = []
for i, pos in enumerate(receiver_positions):
    filename = generate_input_file(i, pos)
    input_files.append(filename)
    print(f"  ✓ Generated: {filename.name}")

print(f"\n  Total input files: {len(input_files)}")

# Ask for confirmation
print("\n" + "="*70)
print(f"WARNING: About to run {num_ports} simulations!")
print(f"Estimated time: ~{num_ports * 2.7:.0f} minutes")
print("="*70)
response = input("Continue? (yes/no): ").strip().lower()

if response != 'yes':
    print("Aborted by user.")
    exit()

print("\n" + "="*70)
print("STEP 2: RUNNING SIMULATIONS")
print("="*70)

output_files = []
for i, input_file in enumerate(input_files):
    print(f"\nSimulation {i+1}/{num_ports}:")
    success = run_simulation(input_file)
    
    if success:
        # gprMax creates output in same dir as input file with name based on #title
        # Check both possible locations
        output_file = output_dir / f'brain_s16p_source_{i+1:02d}.out'
        output_file_alt = output_dir / f'brain_source_{i+1:02d}.out'  # Based on input filename
        
        if output_file.exists():
            output_files.append(output_file)
        elif output_file_alt.exists():
            output_files.append(output_file_alt)
        else:
            print(f"    WARNING: Output file not found: {output_file} or {output_file_alt}")
    else:
        print(f"    Simulation {i+1} failed. Continuing with remaining simulations...")

print(f"\n  Successful simulations: {len(output_files)}/{num_ports}")

if len(output_files) < num_ports:
    print(f"  WARNING: Only {len(output_files)} simulations completed successfully.")
    response = input("Continue with partial data? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Aborted by user.")
        exit()

print("\n" + "="*70)
print("STEP 3: EXTRACTING DATA AND CALCULATING S-MATRIX")
print("="*70)

# Initialize S-matrix storage
# We'll store it as we go, column by column
S_matrix = None
freq = None

for i, output_file in enumerate(output_files):
    print(f"\nProcessing simulation {i+1}/{len(output_files)}: {output_file.name}")
    
    # Extract field data
    E_data, H_data, dt, iterations = extract_field_data(output_file)
    print(f"  Data shape: {E_data.shape}, dt={dt*1e12:.3f} ps")
    
    # Calculate S-matrix column for this source position
    S_column, freq_temp = calculate_s_matrix_column(E_data, H_data, dt, iterations, i)
    
    # Initialize S_matrix on first iteration
    if S_matrix is None:
        freq = freq_temp
        num_freq_bins = len(freq)
        S_matrix = np.zeros((num_ports, num_ports, num_freq_bins), dtype=complex)
        print(f"  Initialized S-matrix: {S_matrix.shape}")
    
    # Store this column (source position i -> all receivers)
    S_matrix[:, i, :] = S_column
    print(f"  ✓ S-matrix column {i+1} calculated")

print("\n" + "="*70)
print("STEP 4: SAVING S16P FILE")
print("="*70)

s16p_filename = base_dir / 'brain_hemorrhage.s16p'
save_s16p_file(S_matrix, freq, s16p_filename)

print("\n" + "="*70)
print("STEP 5: ANALYSIS")
print("="*70)

# Analyze S-parameters at key frequencies
key_freqs = [0.5e9, 1.0e9, 1.5e9, 2.0e9]

for f_target in key_freqs:
    f_idx = np.argmin(np.abs(freq - f_target))
    f_actual = freq[f_idx]
    
    S_at_freq = S_matrix[:, :, f_idx]
    
    # Extract diagonal (S11, S22, ..., S16,16) - reflection coefficients
    S_diag = np.diag(S_at_freq)
    S_diag_mag = np.abs(S_diag)
    S_diag_dB = 20 * np.log10(S_diag_mag + 1e-12)
    
    # Extract off-diagonal (transmission coefficients)
    S_offdiag = S_at_freq.copy()
    np.fill_diagonal(S_offdiag, 0)
    S_offdiag_mag = np.abs(S_offdiag)
    S_offdiag_dB = 20 * np.log10(S_offdiag_mag + 1e-12)
    
    print(f"\n@ {f_actual*1e-9:.2f} GHz:")
    print(f"  Reflection (Sii diagonal):")
    print(f"    Average |Sii|: {np.mean(S_diag_mag):.4f} ({np.mean(S_diag_dB):.2f} dB)")
    print(f"    Range: {np.min(S_diag_dB):.2f} to {np.max(S_diag_dB):.2f} dB")
    print(f"  Transmission (Sij off-diagonal):")
    
    # Filter valid transmission values
    valid_offdiag_mag = S_offdiag_mag[S_offdiag_mag > 0]
    valid_offdiag_dB = S_offdiag_dB[S_offdiag_dB > -100]
    
    if len(valid_offdiag_mag) > 0:
        print(f"    Average |Sij|: {np.mean(valid_offdiag_mag):.4f} ({np.mean(valid_offdiag_dB):.2f} dB)")
        print(f"    Range: {np.min(valid_offdiag_dB):.2f} to {np.max(valid_offdiag_dB):.2f} dB")
    else:
        print(f"    No valid transmission data at this frequency")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"✓ Generated S16P file: {s16p_filename}")
print(f"✓ File size: {s16p_filename.stat().st_size / 1024:.1f} KB")
print(f"✓ Frequency points: {len(freq)}")
print(f"✓ Frequency range: {freq[0]*1e-9:.3f} - {freq[-1]*1e-9:.3f} GHz")
print(f"✓ S-matrix dimensions: {num_ports}×{num_ports} = {num_ports**2} parameters per frequency")
print("\nYou can now use this .s16p file in:")
print("  - RF/microwave design tools (ADS, HFSS, CST)")
print("  - Network analysis software")
print("  - Microwave imaging reconstruction algorithms")
print("  - Machine learning models for hemorrhage detection")
print("="*70)
