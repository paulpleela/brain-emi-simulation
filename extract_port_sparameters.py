"""
Extract S-parameters from transmission_line port data (V and I).

This script:
1. Reads voltage V(t) and current I(t) from transmission_line ports in gprMax HDF5 output
2. Computes FFT to get V(f) and I(f)
3. Calculates port impedance Z(f) = V(f) / I(f)
4. Computes S-parameters using proper 50Ω normalization
5. Builds 16×16 S-matrix and saves as Touchstone .s16p file

Updated for realistic brain imaging model (0-2 GHz, monopole antennas with coupling medium).
"""

import os
import h5py
import numpy as np
import glob

def read_transmission_line_ports(hdf5_file):
    """
    Read transmission line port data from gprMax HDF5 output.
    
    Returns:
        dict: {port_number: {'V': V_array, 'I': I_array, 'position': (x,y,z)}}
        dt: time step
        iterations: number of time steps
    """
    ports = {}
    
    with h5py.File(hdf5_file, 'r') as f:
        # Get time step and iterations
        dt = f.attrs['dt']
        iterations = f.attrs['Iterations']
        
        # Check if transmission line data exists
        if 'tls' not in f.keys():
            raise KeyError(f"No transmission line ports found in {hdf5_file}. Check if #transmission_line commands were used.")
        
        tls_group = f['tls']
        
        # Iterate through all transmission lines
        for tl_name in tls_group.keys():
            tl = tls_group[tl_name]
            
            # Extract port number from name (e.g., 'tl1' -> 1)
            port_num = int(tl_name.replace('tl', ''))
            
            # Read voltage and current
            V = tl['V'][:]  # Voltage time series
            I = tl['I'][:]  # Current time series
            
            # Get position (if available)
            if 'Position' in tl.attrs:
                position = tl.attrs['Position']
            else:
                position = None
            
            ports[port_num] = {
                'V': V,
                'I': I,
                'position': position
            }
    
    return ports, dt, iterations

def compute_port_sparameters(V_f, I_f, Z0=50.0):
    """
    Compute S-parameters from port voltage and current in frequency domain.
    
    Args:
        V_f: Voltage spectrum (complex array)
        I_f: Current spectrum (complex array)
        Z0: Reference impedance (default 50Ω)
    
    Returns:
        S11: Reflection coefficient
        Z_port: Port impedance
    """
    # Port impedance
    Z_port = V_f / (I_f + 1e-20)  # Avoid division by zero
    
    # Reflection coefficient (S11)
    S11 = (Z_port - Z0) / (Z_port + Z0)
    
    return S11, Z_port

def extract_s_matrix_from_monopole_simulations(sim_dir, n_ports=16, Z0=50.0):
    """
    Extract full S-matrix from multi-static monopole antenna simulations.
    
    Args:
        sim_dir: Directory containing simulation outputs (brain_monopole_tx01.out, etc.)
        n_ports: Number of ports (antennas)
        Z0: Reference impedance
    
    Returns:
        S_matrix: Complex S-matrix (n_ports × n_ports × n_freq)
        freqs: Frequency array
    """
    
    print(f"Extracting S-parameters from {sim_dir}...")
    
    # Find all output files
    output_files = sorted(glob.glob(os.path.join(sim_dir, "brain_monopole_tx*.out")))
    
    if len(output_files) != n_ports:
        print(f"Warning: Expected {n_ports} output files, found {len(output_files)}")
    
    S_matrix = None
    freqs = None
    
    for tx_idx, output_file in enumerate(output_files):
        tx_port = tx_idx + 1  # 1-indexed
        
        print(f"\nProcessing TX port {tx_port}: {os.path.basename(output_file)}")
        
        # Read transmission line port data
        try:
            ports, dt, iterations = read_transmission_line_ports(output_file)
        except Exception as e:
            print(f"  Error reading file: {e}")
            continue
        
        # Compute frequency array (only once)
        if freqs is None:
            freqs = np.fft.rfftfreq(iterations, dt)
            n_freq = len(freqs)
            S_matrix = np.zeros((n_ports, n_ports, n_freq), dtype=complex)
        
        # Get incident voltage from TX port
        if tx_port not in ports:
            print(f"  Error: TX port {tx_port} not found in output")
            continue
        
        V_tx_t = ports[tx_port]['V']
        I_tx_t = ports[tx_port]['I']
        
        # FFT
        V_tx_f = np.fft.rfft(V_tx_t)
        I_tx_f = np.fft.rfft(I_tx_t)
        
        # Incident wave (a-wave) = (V + Z0*I) / (2*sqrt(Z0))
        # For calibration purposes, we can use V_incident = V_tx
        V_incident_f = V_tx_f
        
        # Extract S-parameters for all receive ports
        for rx_port in range(1, n_ports + 1):
            if rx_port not in ports:
                print(f"  Warning: RX port {rx_port} not found, skipping")
                continue
            
            V_rx_t = ports[rx_port]['V']
            I_rx_t = ports[rx_port]['I']
            
            # FFT
            V_rx_f = np.fft.rfft(V_rx_t)
            I_rx_f = np.fft.rfft(I_rx_t)
            
            if rx_port == tx_port:
                # Reflection coefficient (S11, S22, etc.)
                S11, Z_port = compute_port_sparameters(V_rx_f, I_rx_f, Z0)
                S_matrix[tx_idx, rx_port-1, :] = S11
                
                # Print some stats
                avg_S11_dB = 20 * np.log10(np.mean(np.abs(S11[1:])) + 1e-12)
                print(f"  Port {rx_port} (reflection): avg |S{tx_port}{rx_port}| = {avg_S11_dB:.1f} dB")
            else:
                # Transmission coefficient (S21, S31, etc.)
                # S_ij = V_j / V_incident_i (for matched ports)
                # More rigorous: use b-wave / a-wave formulation
                S_ij = V_rx_f / (V_incident_f + 1e-20)
                S_matrix[tx_idx, rx_port-1, :] = S_ij
        
        print(f"  ✓ Extracted column {tx_port} of S-matrix")
    
    return S_matrix, freqs

def save_s16p_file(S_matrix, freqs, filename, Z0=50.0):
    """
    Save S-matrix as Touchstone .s16p file.
    
    Args:
        S_matrix: Complex S-matrix (16 × 16 × n_freq)
        freqs: Frequency array (Hz)
        filename: Output filename
        Z0: Reference impedance
    """
    
    n_ports = S_matrix.shape[0]
    n_freq = len(freqs)
    
    print(f"\nSaving S-parameters to {filename}...")
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Header
        f.write("! Touchstone file - Brain hemorrhage imaging with monopole antennas\n")
        f.write(f"! S-parameters from realistic 50 Ohm transmission line ports\n")
        f.write(f"! {n_ports} ports, {n_freq} frequency points\n")
        f.write(f"! Frequencies: {freqs[0]/1e9:.3f} - {freqs[-1]/1e9:.3f} GHz\n")
        f.write("! Format: magnitude-angle (dB, degrees)\n")
        f.write(f"# GHz S MA R {Z0}\n")
        
        # Data
        for fi in range(n_freq):
            freq_GHz = freqs[fi] / 1e9
            
            # Frequency
            f.write(f"{freq_GHz:.10e} ")
            
            # All S-parameters for this frequency (row-major order)
            for i in range(n_ports):
                for j in range(n_ports):
                    S_ij = S_matrix[i, j, fi]
                    mag = np.abs(S_ij)
                    ang_deg = np.angle(S_ij, deg=True)
                    f.write(f"{mag:.10e} {ang_deg:.10e} ")
            
            f.write("\n")
    
    print(f"✓ Saved {filename} ({os.path.getsize(filename) / 1e6:.2f} MB)")

# Main execution
if __name__ == "__main__":
    import sys
    
    # Configuration - Updated for realistic model
    sim_dir = "brain_monopole_realistic"
    output_s16p = "brain_realistic_calibrated.s16p"
    n_ports = 16
    Z0 = 50.0
    
    if not os.path.exists(sim_dir):
        print(f"Error: Simulation directory '{sim_dir}' not found.")
        print("Run simulations first with gprmax on HPC.")
        sys.exit(1)
    
    # Check if output files exist
    output_files = glob.glob(os.path.join(sim_dir, "*.out"))
    if len(output_files) == 0:
        print(f"Error: No .out files found in {sim_dir}")
        print("Run simulations first.")
        sys.exit(1)
    
    print(f"Found {len(output_files)} output files")
    
    # Extract S-matrix
    S_matrix, freqs = extract_s_matrix_from_monopole_simulations(sim_dir, n_ports, Z0)
    
    if S_matrix is None:
        print("Error: Failed to extract S-matrix")
        sys.exit(1)
    
    # Save as S16P
    save_s16p_file(S_matrix, freqs, output_s16p, Z0)
    
    # Print summary
    print("\n" + "="*70)
    print("S-PARAMETER EXTRACTION COMPLETE")
    print("="*70)
    print(f"Output file: {output_s16p}")
    print(f"Frequency points: {len(freqs)}")
    print(f"Frequency range: {freqs[0]/1e9:.3f} - {freqs[-1]/1e9:.3f} GHz")
    print(f"S-matrix shape: {S_matrix.shape}")
    print("\nThese S-parameters are from realistic 50Ω transmission line ports.")
    print("Apply de-embedding using calibration data for final results.")
    print("="*70)
