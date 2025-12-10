"""
Extract S-parameters from gprMax receiver (#rx) data.

This script works with GPU-compatible simulations using #voltage_source and #rx.
It reads E-field data from receivers and computes S-parameters for the 16-antenna array.

For monopole antennas:
- S11 (reflection): From E-field at transmit antenna position
- S21, S31, etc. (transmission): From E-field at receiver positions

Algorithm:
1. Read Ez field data from all 16 receivers
2. Compute FFT to get Ez(f)
3. Convert Ez to voltage-like quantity: V ≈ Ez × monopole_height
4. Calculate S-parameters with proper normalization
5. Build 16×16 S-matrix and save as .s16p file
"""

import os
import h5py
import numpy as np
import glob
from pathlib import Path

def read_receiver_data(hdf5_file):
    """
    Read receiver data from gprMax HDF5 output.
    
    Returns:
        receivers: dict {rx_number: {'Ex': array, 'Ey': array, 'Ez': array}}
        dt: time step
        iterations: number of time steps
    """
    receivers = {}
    
    with h5py.File(hdf5_file, 'r') as f:
        # Get time step and iterations
        dt = f.attrs['dt']
        iterations = f.attrs['Iterations']
        
        # Check if receiver data exists
        if 'rxs' not in f.keys():
            raise KeyError(f"No receiver data found in {hdf5_file}. Check if #rx commands were used.")
        
        rxs_group = f['rxs']
        
        # Iterate through all receivers
        for rx_name in rxs_group.keys():
            rx = rxs_group[rx_name]
            
            # Extract receiver number from name (e.g., 'rx1' -> 1)
            rx_num = int(rx_name.replace('rx', ''))
            
            # Read E-field components (all receivers get Ex, Ey, Ez)
            receivers[rx_num] = {
                'Ex': rx['Ex'][:] if 'Ex' in rx else None,
                'Ey': rx['Ey'][:] if 'Ey' in rx else None,
                'Ez': rx['Ez'][:] if 'Ez' in rx else None,
            }
    
    return receivers, dt, iterations

def compute_sparameters_from_efield(Ez_incident, Ez_received, monopole_height=0.0375, Z0=50.0):
    """
    Compute S-parameter from E-field data.
    
    For monopole antennas, the voltage is approximately V ≈ Ez × h
    where h is the monopole height.
    
    Args:
        Ez_incident: E-field at incident (transmit) antenna (frequency domain)
        Ez_received: E-field at received antenna (frequency domain)
        monopole_height: Height of monopole antenna (m)
        Z0: Reference impedance (Ω)
    
    Returns:
        S_parameter: Complex S-parameter array vs frequency
    """
    # Convert E-field to voltage-like quantity
    V_incident = Ez_incident * monopole_height
    V_received = Ez_received * monopole_height
    
    # S-parameter is the ratio of received to incident voltage
    # Add small epsilon to avoid division by zero
    S = V_received / (V_incident + 1e-20)
    
    return S

def process_multistatic_data(output_dir, n_antennas=16, monopole_height=0.0375, Z0=50.0):
    """
    Process all .out files from multi-static simulation to build S-matrix.
    
    Args:
        output_dir: Directory containing brain_realistic_tx*.out files
        n_antennas: Number of antennas (default 16)
        monopole_height: Monopole antenna height in meters
        Z0: Reference impedance
    
    Returns:
        S_matrix: Complex S-matrix [n_freq × n_antennas × n_antennas]
        frequencies: Frequency array
    """
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING S-PARAMETERS FROM RECEIVER DATA")
    print(f"{'='*70}\n")
    print(f"Output directory: {output_dir}")
    print(f"Number of antennas: {n_antennas}")
    print(f"Monopole height: {monopole_height*1000:.1f} mm")
    print(f"Reference impedance: {Z0} Ω\n")
    
    # Find all output files
    output_files = sorted(glob.glob(os.path.join(output_dir, "brain_realistic_tx*.out")))
    
    if len(output_files) == 0:
        raise FileNotFoundError(f"No output files found in {output_dir}")
    
    print(f"Found {len(output_files)} output files")
    
    if len(output_files) != n_antennas:
        print(f"WARNING: Expected {n_antennas} files, found {len(output_files)}")
    
    # Initialize arrays
    S_matrix = None
    frequencies = None
    
    # Process each transmit configuration
    for tx_idx, out_file in enumerate(output_files):
        tx_num = tx_idx + 1
        print(f"\nProcessing TX antenna {tx_num}/{n_antennas}...")
        print(f"  File: {os.path.basename(out_file)}")
        
        try:
            # Read receiver data
            receivers, dt, iterations = read_receiver_data(out_file)
            
            print(f"  Time step: {dt*1e12:.3f} ps")
            print(f"  Iterations: {iterations}")
            print(f"  Receivers found: {len(receivers)}")
            
            if len(receivers) != n_antennas:
                print(f"  WARNING: Expected {n_antennas} receivers, found {len(receivers)}")
            
            # Compute FFT for all receivers
            n_freq = iterations // 2 + 1
            freq = np.fft.rfftfreq(iterations, dt)
            
            if frequencies is None:
                frequencies = freq
                S_matrix = np.zeros((n_freq, n_antennas, n_antennas), dtype=complex)
            
            # Process each receiver
            for rx_num in range(1, n_antennas + 1):
                if rx_num not in receivers:
                    print(f"  WARNING: Receiver {rx_num} not found")
                    continue
                
                # Get Ez field (vertical component for monopole)
                Ez_time = receivers[rx_num]['Ez']
                
                if Ez_time is None:
                    print(f"  WARNING: Ez data not available for receiver {rx_num}")
                    continue
                
                # FFT to frequency domain
                Ez_freq = np.fft.rfft(Ez_time)
                
                # For the transmit antenna (rx_num == tx_num), this is the incident field
                # For other antennas, this is the received field
                if rx_num == tx_num:
                    # Store incident field for this TX
                    Ez_incident = Ez_freq
                    
                    # S11, S22, etc. (reflection coefficient)
                    # For reflection, we compare to a reference (could use initial value or assume normalization)
                    # Simple approach: S_ii ≈ Ez_received / Ez_max
                    S_matrix[:, tx_idx, rx_num - 1] = Ez_freq / (np.max(np.abs(Ez_freq)) + 1e-20)
                else:
                    # Sij where i != j (transmission coefficient)
                    S_matrix[:, tx_idx, rx_num - 1] = compute_sparameters_from_efield(
                        Ez_incident, Ez_freq, monopole_height, Z0
                    )
            
            print(f"  ✓ Processed {len(receivers)} receivers")
            
        except Exception as e:
            print(f"  ✗ Error processing {out_file}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"S-MATRIX EXTRACTION COMPLETE")
    print(f"{'='*70}\n")
    print(f"Frequency points: {len(frequencies)}")
    print(f"Frequency range: {frequencies[0]/1e9:.3f} - {frequencies[-1]/1e9:.3f} GHz")
    print(f"S-matrix shape: {S_matrix.shape}")
    
    return S_matrix, frequencies

def write_touchstone_s16p(filename, S_matrix, frequencies, Z0=50.0):
    """
    Write S-parameters to Touchstone .s16p file format.
    
    Args:
        filename: Output filename (.s16p)
        S_matrix: Complex S-matrix [n_freq × 16 × 16]
        frequencies: Frequency array in Hz
        Z0: Reference impedance
    """
    print(f"\nWriting Touchstone file: {filename}")
    
    n_freq, n_ports, _ = S_matrix.shape
    
    with open(filename, 'w') as f:
        # Header
        f.write(f"! Touchstone file generated from gprMax receiver data\n")
        f.write(f"! Multi-static brain imaging with {n_ports} monopole antennas\n")
        f.write(f"! Frequency range: {frequencies[0]/1e9:.3f} - {frequencies[-1]/1e9:.3f} GHz\n")
        f.write(f"! Number of frequency points: {n_freq}\n")
        f.write(f"! Reference impedance: {Z0} Ohm\n")
        f.write(f"!\n")
        
        # Option line: GHz, S-parameters, Magnitude-Angle, Reference impedance
        f.write(f"# GHz S MA R {Z0}\n")
        
        # Data: frequency followed by all S-parameters
        # Format: freq S11_mag S11_ang S12_mag S12_ang ... S16_16_mag S16_16_ang
        for i, freq in enumerate(frequencies):
            # Skip DC (freq = 0)
            if freq < 1e6:  # Skip below 1 MHz
                continue
            
            f.write(f"{freq/1e9:.6f}")  # Frequency in GHz
            
            # Write all S-parameters for this frequency
            # Row-major order: S11, S12, ..., S1N, S21, S22, ..., S2N, ..., SNN
            for tx in range(n_ports):
                for rx in range(n_ports):
                    S = S_matrix[i, tx, rx]
                    mag = np.abs(S)
                    phase = np.angle(S, deg=True)
                    f.write(f"  {mag:.6e} {phase:.6f}")
            
            f.write("\n")
    
    print(f"✓ Written {n_freq} frequency points")
    print(f"✓ File size: {os.path.getsize(filename) / 1024:.1f} KB")

def main():
    """Main execution function."""
    
    # Configuration
    output_dir = "brain_monopole_realistic"
    s16p_file = "brain_realistic_array.s16p"
    n_antennas = 16
    monopole_height = 0.0375  # 37.5 mm (λ/4 at 2 GHz)
    Z0 = 50.0  # Reference impedance
    
    # Process multi-static data
    try:
        S_matrix, frequencies = process_multistatic_data(
            output_dir, 
            n_antennas=n_antennas,
            monopole_height=monopole_height,
            Z0=Z0
        )
        
        # Write Touchstone file
        write_touchstone_s16p(s16p_file, S_matrix, frequencies, Z0)
        
        print(f"\n{'='*70}")
        print(f"SUCCESS!")
        print(f"{'='*70}\n")
        print(f"S-parameter file created: {s16p_file}")
        print(f"You can now use this file for:")
        print(f"  - Network analysis")
        print(f"  - Imaging algorithms")
        print(f"  - Visualization with plot_sij.py")
        print(f"\n{'='*70}\n")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR")
        print(f"{'='*70}\n")
        print(f"Failed to extract S-parameters: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
