"""
Validate a .s16p file by checking basic physical properties of the S-matrix.

Usage:
    python validate_s16p.py sparams/scenario_001.s16p
"""

import sys
import numpy as np
import os

def parse_s16p(filepath):
    """Parse a Touchstone .s16p file. Returns (freqs_GHz, S) where S is (n_ports, n_ports, n_freq) complex."""
    freqs = []
    data_rows = []
    n_ports = 16

    with open(filepath) as f:
        values = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
            if line.startswith('#'):
                continue
            values.extend(line.split())

    # Each frequency block: 1 freq value + n_ports*n_ports*2 mag/angle values
    vals_per_freq = 1 + n_ports * n_ports * 2
    n_freq = len(values) // vals_per_freq

    freqs = np.zeros(n_freq)
    S = np.zeros((n_ports, n_ports, n_freq), dtype=complex)

    for fi in range(n_freq):
        base = fi * vals_per_freq
        freqs[fi] = float(values[base])
        k = base + 1
        for i in range(n_ports):
            for j in range(n_ports):
                mag = float(values[k]);  ang = float(values[k+1]);  k += 2
                S[i, j, fi] = mag * np.exp(1j * np.radians(ang))

    return freqs, S


def validate(filepath):
    print(f"\n{'='*60}")
    print(f"Validating: {filepath}")
    print(f"{'='*60}")

    freqs, S = parse_s16p(filepath)
    n_ports, _, n_freq = S.shape

    print(f"\nBasic info:")
    print(f"  Frequency points : {n_freq}")
    print(f"  Frequency range  : {freqs[0]:.4f} - {freqs[-1]:.4f} GHz")
    print(f"  Ports            : {n_ports}")

    # ── Check 1: S-matrix values are finite ──────────────────────────────────
    if np.all(np.isfinite(S)):
        print(f"\n✓ All values are finite (no NaN/Inf)")
    else:
        print(f"\n✗ WARNING: {np.sum(~np.isfinite(S))} NaN/Inf values found!")

    # ── Check 2: |S11| < 1 (passive system, no gain) ─────────────────────────
    max_mag = np.max(np.abs(S))
    if max_mag <= 1.05:
        print(f"✓ Max |Sij| = {max_mag:.4f} (≤ 1, passive system OK)")
    else:
        print(f"✗ WARNING: Max |Sij| = {max_mag:.4f} (> 1, check simulation)")

    # ── Check 3: Diagonal (reflections) should be well below 0 dB ────────────
    print(f"\nReflection coefficients |Sii| at 1 GHz:")
    fi_1ghz = np.argmin(np.abs(freqs - 1.0))
    for i in range(n_ports):
        sii = np.abs(S[i, i, fi_1ghz])
        sii_db = 20 * np.log10(sii + 1e-12)
        flag = "  ← high?" if sii_db > -3 else ""
        print(f"  S{i+1:02d}{i+1:02d} = {sii_db:6.1f} dB{flag}")

    # ── Check 4: Off-diagonal (transmission) should be much smaller ──────────
    print(f"\nSample transmission |S21| across frequency:")
    for fi in [0, n_freq//4, n_freq//2, 3*n_freq//4, n_freq-1]:
        s21 = np.abs(S[1, 0, fi])
        s21_db = 20 * np.log10(s21 + 1e-12)
        print(f"  f={freqs[fi]:.3f} GHz  |S21| = {s21_db:.1f} dB")

    # ── Check 5: Reciprocity — S_ij ≈ S_ji ───────────────────────────────────
    diff = np.abs(S - np.transpose(S, (1, 0, 2)))
    max_recip_err = np.max(diff)
    avg_recip_err = np.mean(diff)
    if max_recip_err < 0.05:
        print(f"\n✓ Reciprocity OK  (max |Sij - Sji| = {max_recip_err:.4f})")
    else:
        print(f"\n⚠ Reciprocity error: max |Sij - Sji| = {max_recip_err:.4f}, avg = {avg_recip_err:.4f}")
        print(f"  (Some asymmetry expected due to independent simulations — "
              f"values < 0.1 are acceptable)")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_s16p.py sparams/scenario_001.s16p")
        sys.exit(1)
    validate(sys.argv[1])
