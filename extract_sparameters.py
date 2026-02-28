"""
Extract S-parameters from gprMax transmission line output and save as Touchstone .s16p

For each scenario:
  - Reads 16 .out files (one per transmitter: scenario_XXX_tx01.out ... tx16.out)
  - Each .out file contains 16 transmission lines (tl1..tl16) with V(t) and I(t)
  - Computes the full 16x16 S-matrix using wave variables:
      a_j = (V_j + Z0*I_j) / (2*sqrt(Z0))   <- incident wave at port j
      b_i = (V_i - Z0*I_i) / (2*sqrt(Z0))   <- reflected/transmitted wave at port i
      S_ij = b_i / a_j                        <- column j of S-matrix from tx_j simulation
  - Saves one .s16p Touchstone file per scenario
  - Deletes the 16 .out files to save disk space

Usage:
    python extract_sparameters.py --scenario 1
    python extract_sparameters.py --all
    python extract_sparameters.py --range 1 300

Output:
    sparams/scenario_001.s16p  (one file per scenario, ~kilobytes each)
"""

import os
import sys
import h5py
import numpy as np
import argparse
import glob


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_DIR   = "brain_inputs"    # where .out files live
OUTPUT_DIR  = "sparams"         # where .s16p files will be saved
N_PORTS     = 16
Z0          = 50.0              # reference impedance (Ohms)
F_MAX       = 2e9               # max frequency to keep (Hz)
DELETE_OUT  = True              # delete .out files after successful extraction


# ============================================================================
# HDF5 READING
# ============================================================================

def read_tl_data(hdf5_path):
    """
    Read all transmission line V(t) and I(t) from a gprMax .out file.

    Returns:
        tls  : dict  {tl_index (1-based): {'V': array, 'I': array}}
        dt   : float  time step in seconds
        n_it : int    number of iterations
    """
    tls = {}
    with h5py.File(hdf5_path, 'r') as f:
        dt   = float(f.attrs['dt'])
        n_it = int(f.attrs['Iterations'])

        if 'tls' not in f:
            raise KeyError(f"No 'tls' group in {hdf5_path}. "
                           "Was #transmission_line used in the input file?")

        for name in f['tls'].keys():
            # gprMax names them tl1, tl2, ... in the order they were defined
            idx = int(name.replace('tl', ''))
            tls[idx] = {
                'V': f['tls'][name]['V'][:],
                'I': f['tls'][name]['I'][:]
            }

    return tls, dt, n_it


# ============================================================================
# S-PARAMETER COMPUTATION
# ============================================================================

def compute_s_matrix(scenario_id):
    """
    Build the full 16x16 S-matrix for one scenario.

    Column j of S comes from the simulation where antenna j is the transmitter
    (file scenario_XXX_txJJ.out). Each .out has all 16 TLs recorded.

    S_ij(f) = b_i(f) / a_j(f)
    where:
        a_j = (V_j + Z0*I_j) / (2*sqrt(Z0))   incident wave (transmitter port)
        b_i = (V_i - Z0*I_i) / (2*sqrt(Z0))   scattered wave (all ports)

    Returns:
        S      : ndarray  (N_PORTS, N_PORTS, n_freq_keep)  complex
        freqs  : ndarray  (n_freq_keep,)  Hz
    """
    S      = None
    freqs  = None

    for tx in range(1, N_PORTS + 1):
        fname = os.path.join(INPUT_DIR, f"scenario_{scenario_id:03d}_tx{tx:02d}.out")
        if not os.path.isfile(fname):
            print(f"  WARNING: missing {fname}, skipping column {tx}")
            continue

        tls, dt, n_it = read_tl_data(fname)

        # Build frequency axis on first file
        if freqs is None:
            all_freqs = np.fft.rfftfreq(n_it, dt)
            freq_mask = all_freqs <= F_MAX
            freqs     = all_freqs[freq_mask]
            n_freq    = len(freqs)
            S         = np.zeros((N_PORTS, N_PORTS, n_freq), dtype=complex)

        # Incident wave at the active transmitter port (column j = tx-1)
        V_tx = np.fft.rfft(tls[tx]['V'])[freq_mask]
        I_tx = np.fft.rfft(tls[tx]['I'])[freq_mask]
        a_j  = (V_tx + Z0 * I_tx) / (2.0 * np.sqrt(Z0))

        # Scattered wave at every port (row i)
        for rx in range(1, N_PORTS + 1):
            if rx not in tls:
                continue
            V_rx = np.fft.rfft(tls[rx]['V'])[freq_mask]
            I_rx = np.fft.rfft(tls[rx]['I'])[freq_mask]
            b_i  = (V_rx - Z0 * I_rx) / (2.0 * np.sqrt(Z0))

            # Avoid divide-by-zero at DC or near-zero incident wave
            with np.errstate(divide='ignore', invalid='ignore'):
                S[rx-1, tx-1, :] = np.where(np.abs(a_j) > 1e-30, b_i / a_j, 0.0)

    return S, freqs


# ============================================================================
# TOUCHSTONE WRITER
# ============================================================================

def write_s16p(S, freqs, out_path, scenario_id):
    """
    Write S-matrix as Touchstone .s16p (magnitude-angle, GHz, 50 Ohm).

    Touchstone 1.0 format:
        # GHz S MA R 50
        freq  S11_mag S11_ang  S21_mag S21_ang  ...  SN1_mag SN1_ang
              S12_mag S12_ang  ...
        ...
    One frequency block per line-group (all rows for that frequency).
    """
    n_ports = S.shape[0]
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    with open(out_path, 'w') as f:
        f.write(f"! Brain hemorrhage imaging - scenario {scenario_id:03d}\n")
        f.write(f"! {n_ports}-port S-parameters, {len(freqs)} frequency points\n")
        f.write(f"! Frequency range: {freqs[0]/1e9:.4f} - {freqs[-1]/1e9:.4f} GHz\n")
        f.write(f"! Generated by extract_sparameters.py\n")
        f.write(f"# GHz S MA R {Z0:.0f}\n")

        for fi, freq in enumerate(freqs):
            # Touchstone: first value is frequency, then all S-params for that freq
            # For >2 ports: S_1j row, then S_2j row, etc. on separate continuation lines
            row_vals = []
            for i in range(n_ports):
                for j in range(n_ports):
                    mag = np.abs(S[i, j, fi])
                    ang = np.degrees(np.angle(S[i, j, fi]))
                    row_vals.append(f"{mag:.8e} {ang:.6f}")

            # Write: freq then 4 S-params per line (standard Touchstone convention)
            pairs_per_line = 4
            f.write(f"{freq/1e9:.10e}")
            for k, val in enumerate(row_vals):
                if k > 0 and k % pairs_per_line == 0:
                    f.write("\n")
                f.write(f" {val}")
            f.write("\n")

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved: {out_path} ({size_kb:.1f} KB, {len(freqs)} freq pts)")


# ============================================================================
# PER-SCENARIO PIPELINE
# ============================================================================

def process_scenario(scenario_id):
    """Full pipeline for one scenario: read → compute S → write .s16p → delete .out"""

    print(f"\nScenario {scenario_id:03d}")

    # Check all 16 .out files exist
    out_files = [
        os.path.join(INPUT_DIR, f"scenario_{scenario_id:03d}_tx{tx:02d}.out")
        for tx in range(1, N_PORTS + 1)
    ]
    missing = [f for f in out_files if not os.path.isfile(f)]
    if missing:
        print(f"  SKIP: {len(missing)} .out files missing (simulations not done yet?)")
        return False

    # Compute S-matrix
    try:
        S, freqs = compute_s_matrix(scenario_id)
    except Exception as e:
        print(f"  ERROR computing S-matrix: {e}")
        return False

    if S is None or freqs is None:
        print(f"  ERROR: no data extracted")
        return False

    # Write .s16p
    out_path = os.path.join(OUTPUT_DIR, f"scenario_{scenario_id:03d}.s16p")
    try:
        write_s16p(S, freqs, out_path, scenario_id)
    except Exception as e:
        print(f"  ERROR writing .s16p: {e}")
        return False

    # Delete .out files
    if DELETE_OUT:
        for f in out_files:
            os.remove(f)
        print(f"  Deleted {len(out_files)} .out files")

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract S-parameters from gprMax output")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=int, metavar="N",
                       help="Process a single scenario number")
    group.add_argument("--all",      action="store_true",
                       help="Process all scenarios (1-300)")
    group.add_argument("--range",    type=int, nargs=2, metavar=("START", "END"),
                       help="Process scenarios START to END (inclusive)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.scenario:
        ids = [args.scenario]
    elif args.all:
        ids = range(1, 301)
    else:
        ids = range(args.range[0], args.range[1] + 1)

    ok = fail = skip = 0
    for sid in ids:
        result = process_scenario(sid)
        if result is True:
            ok   += 1
        elif result is False:
            fail += 1
        else:
            skip += 1

    print(f"\n{'='*60}")
    print(f"Done: {ok} succeeded, {fail} failed, {skip} skipped")
    print(f"S16P files in: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
