"""
Validate a .s16p file produced by build_s16p.py.

Runs 10 physical sanity checks and prints a clear pass/fail summary.
Only needs numpy — no extra dependencies beyond what gprMax already installs.

Usage:
    python validate_s16p.py sparams/scenario_001.s16p
    python validate_s16p.py sparams/scenario_001.s16p --verbose
    python validate_s16p.py sparams/           # validate all .s16p files in a directory
"""

import sys
import os
import glob
import argparse
import numpy as np

N_PORTS   = 16
F_MAX_GHZ = 2.0


# ═══════════════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_s16p(filepath):
    """
    Parse a Touchstone .s16p file written by build_s16p.py.

    The writer outputs magnitude-angle pairs in GHz, 4 S-param pairs per line,
    with line-breaks inside each frequency block.  The safest approach is to
    tokenise the entire data section into one flat list and use a fixed stride
    of (1 + N*N*2) = 513 values per frequency point.

    Returns
    -------
    freqs_ghz : ndarray (n_freq,)
    S         : ndarray (n_ports, n_ports, n_freq)  complex
    """
    n_ports = N_PORTS
    tokens  = []

    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('!') or line.startswith('#'):
                continue
            tokens.extend(line.split())

    vals_per_freq = 1 + n_ports * n_ports * 2   # 513 for 16-port
    n_freq        = len(tokens) // vals_per_freq
    remainder     = len(tokens) % vals_per_freq

    if n_freq == 0:
        raise ValueError(
            f"Could not parse any frequency blocks from {filepath} "
            f"(got {len(tokens)} tokens, need multiples of {vals_per_freq})")
    if remainder != 0:
        raise ValueError(
            f"Token count {len(tokens)} is not a multiple of {vals_per_freq}. "
            f"File may be truncated or corrupted.")

    freqs_ghz = np.zeros(n_freq)
    S = np.zeros((n_ports, n_ports, n_freq), dtype=complex)

    for fi in range(n_freq):
        base = fi * vals_per_freq
        freqs_ghz[fi] = float(tokens[base])
        k = base + 1
        for i in range(n_ports):
            for j in range(n_ports):
                mag = float(tokens[k])
                ang = float(tokens[k + 1])
                k  += 2
                S[i, j, fi] = mag * np.exp(1j * np.radians(ang))

    return freqs_ghz, S


# ═══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def check_finite(S, freqs_ghz, verbose=False):
    """All values must be finite (no NaN / Inf from a failed FFT or div-by-zero)."""
    n_bad = int(np.sum(~np.isfinite(S)))
    ok  = (n_bad == 0)
    msg = "All values finite" if ok else f"{n_bad} NaN/Inf values — simulation or FFT failure"
    return ok, msg


def check_passivity(S, freqs_ghz, verbose=False):
    """
    Passive system: no element |Sij| > 1.
    A 5 % tolerance covers numerical noise at very low-power frequencies.
    """
    mag   = np.abs(S)
    max_m = mag.max()
    idx   = np.unravel_index(mag.argmax(), mag.shape)
    i, j, fi = idx
    ok  = max_m <= 1.05
    msg = f"Max |Sij| = {max_m:.5f}  (port {i+1}→{j+1}, freq index {fi})"
    if not ok:
        msg += "  ← EXCEEDS 1 — check simulation"
    return ok, msg


def check_frequency_range(S, freqs_ghz, verbose=False):
    """Frequency axis should reach at least 95 % of the 2 GHz design limit."""
    f_min = freqs_ghz[0]
    f_max = freqs_ghz[-1]
    ok  = f_max >= F_MAX_GHZ * 0.95
    msg = f"{f_min:.4f}–{f_max:.4f} GHz  ({len(freqs_ghz)} points)"
    if not ok:
        msg += f"  ← expected up to {F_MAX_GHZ} GHz"
    return ok, msg


def check_return_loss(S, freqs_ghz, verbose=False):
    """
    At 1 GHz every antenna should show |Sii| < 0 dB.
    Flag any port where |Sii| > −1 dB (very poor match / likely issue).
    """
    fi      = np.argmin(np.abs(freqs_ghz - 1.0))
    f_act   = freqs_ghz[fi]
    diag    = np.array([np.abs(S[i, i, fi]) for i in range(N_PORTS)])
    diag_db = 20 * np.log10(diag + 1e-12)

    bad = [i + 1 for i, db in enumerate(diag_db) if db > -1]
    ok  = len(bad) == 0
    msg = f"At {f_act:.3f} GHz: "
    if ok:
        msg += f"all ports < −1 dB  (range {diag_db.min():.1f} to {diag_db.max():.1f} dB)"
    else:
        msg += f"ports {bad} have |Sii| > −1 dB"

    if verbose:
        for i, db in enumerate(diag_db):
            flag = "  ← high?" if db > -1 else ""
            print(f"      S{i+1:02d}{i+1:02d} = {db:6.1f} dB{flag}")

    return ok, msg


def check_port_variation(S, freqs_ghz, verbose=False):
    """
    The 16 ports should not be nearly identical at 1 GHz.

    A very small spread across the diagonal curves usually means the run is
    overly symmetric, stale, or otherwise suspicious. This is a heuristic
    warning rather than a physical law.
    """
    fi = np.argmin(np.abs(freqs_ghz - 1.0))
    diag_db = np.array([
        20 * np.log10(np.abs(S[i, i, fi]) + 1e-12)
        for i in range(N_PORTS)
    ])
    spread_db = float(diag_db.max() - diag_db.min())

    ok = spread_db >= 3.0
    msg = f"At {freqs_ghz[fi]:.3f} GHz: diagonal spread across ports = {spread_db:.2f} dB"
    if not ok:
        msg += "  ← unusually uniform across ports; check for symmetry or extraction issues"

    if verbose:
        vals = ", ".join(f"S{i+1:02d}{i+1:02d}={v:.1f} dB" for i, v in enumerate(diag_db))
        print(f"      {vals}")

    return ok, msg


def check_port_variation_band(S, freqs_ghz, verbose=False):
    """
    Across the whole band, the diagonal responses should not remain nearly
    identical from port to port.

    We compute the per-frequency standard deviation of |Sii| in dB across the
    16 ports, then summarize that distribution with the mean and minimum.
    """
    diag_db = 20 * np.log10(np.abs(np.diagonal(S, axis1=0, axis2=1)) + 1e-12)
    # diag_db has shape (n_freq, n_ports)
    std_by_freq = np.std(diag_db, axis=1)
    mean_std = float(np.mean(std_by_freq))
    min_std = float(np.min(std_by_freq))
    p10_std = float(np.percentile(std_by_freq, 10))

    ok = (mean_std >= 3.0) and (min_std >= 1.5)
    msg = (
        f"mean std={mean_std:.2f} dB, min std={min_std:.2f} dB, "
        f"p10 std={p10_std:.2f} dB"
    )
    if not ok:
        msg += "  ← diagonal curves stay too similar across the band"

    if verbose:
        print(
            f"      Per-frequency diagonal std: mean={mean_std:.2f} dB, "
            f"min={min_std:.2f} dB, p10={p10_std:.2f} dB"
        )

    return ok, msg


def check_reciprocity(S, freqs_ghz, verbose=False):
    """
    Non-magnetised medium: S_ij ≈ S_ji.
    We use a relative Frobenius-norm metric per frequency.
    Some asymmetry is expected because each column comes from an independent
    gprMax simulation; values below 15 % are acceptable.
    """
    # S shape: (n_ports, n_ports, n_freq) — transpose axes 0↔1
    diff = np.abs(S - np.transpose(S, (1, 0, 2)))

    norm_S    = np.linalg.norm(S,    axis=(0, 1))   # (n_freq,)
    norm_diff = np.linalg.norm(diff, axis=(0, 1))
    rel       = norm_diff / (norm_S + 1e-30)

    max_abs = diff.max()
    max_rel = rel.max()
    mean_rel = rel.mean()

    ok  = max_rel < 0.15
    msg = (f"max |Sij−Sji| = {max_abs:.5f}  |  "
           f"relative: max={max_rel:.4f}  mean={mean_rel:.4f}")
    if max_rel >= 0.15:
        msg += "  ← exceeds 15 % — check for mismatched .out files"
    elif max_rel >= 0.05:
        msg += "  (small asymmetry from independent sims — OK)"
    return ok, msg


def check_transmission_decay(S, freqs_ghz, verbose=False):
    """
    In a circular array around a lossy head, mean |Sij| should decrease as the
    arc-distance between ports increases.  Adjacent (sep=1) must be stronger
    than diametrically opposite (sep=8) by at least 3 dB.
    """
    fi   = np.argmin(np.abs(freqs_ghz - 1.0))
    Sabs = np.abs(S[:, :, fi])

    def mean_at_sep(sep):
        return np.mean([Sabs[i, (i + sep) % N_PORTS]
                        for i in range(N_PORTS)])

    adj_db = 20 * np.log10(mean_at_sep(1) + 1e-12)
    opp_db = 20 * np.log10(mean_at_sep(8) + 1e-12)
    delta  = adj_db - opp_db

    ok  = delta >= 3.0
    msg = (f"Adjacent(sep=1)={adj_db:.1f} dB  "
           f"Opposite(sep=8)={opp_db:.1f} dB  Δ={delta:.1f} dB")
    if not ok:
        msg += "  ← expected adjacent stronger by ≥3 dB"

    if verbose:
        print(f"      Mean |Sij| dB by port separation at {freqs_ghz[fi]:.3f} GHz:")
        for sep in range(1, N_PORTS // 2 + 1):
            m = mean_at_sep(sep)
            print(f"        sep={sep:2d}: {20*np.log10(m+1e-12):6.1f} dB")

    return ok, msg


def check_phase_continuity(S, freqs_ghz, verbose=False):
    """
    Unwrap phase along frequency and check no step exceeds 90°.
    A larger jump indicates a parsing error or numerical blow-up.
    """
    phase_unwrapped = np.unwrap(np.angle(S), axis=2)
    max_jump_deg    = np.degrees(np.abs(np.diff(phase_unwrapped, axis=2)).max())
    ok  = max_jump_deg < 190.0
    msg = f"Max |Δφ| per freq step = {max_jump_deg:.2f}°"
    if not ok:
        msg += "  ← very large jump — possible parse error or instability"
    return ok, msg


def check_diagonal_dominance(S, freqs_ghz, verbose=False):
    """
    Mean reflection (|Sii|) should exceed mean transmission (|Sij|, i≠j)
    at 1 GHz.  Fails if an entire column is zero or if the S-matrix is
    obviously wrong (e.g. all transmissions written instead of reflections).
    """
    fi   = np.argmin(np.abs(freqs_ghz - 1.0))
    Sabs = np.abs(S[:, :, fi])

    diag    = np.array([Sabs[i, i] for i in range(N_PORTS)])
    offdiag = np.array([Sabs[i, j] for i in range(N_PORTS)
                                    for j in range(N_PORTS) if i != j])
    md = diag.mean()
    mo = offdiag.mean()

    ok  = md > mo
    msg = f"Mean |Sii|={md:.5f}  Mean |Sij|(i≠j)={mo:.5f}"
    if not ok:
        msg += "  ← off-diagonal dominates — suspicious"
    return ok, msg


# ═══════════════════════════════════════════════════════════════════════════
# CHECK REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

CHECKS = [
    ("Finite values",            check_finite),
    ("Passivity  (|Sij| ≤ 1)",   check_passivity),
    ("Frequency range",          check_frequency_range),
    ("Return loss at 1 GHz",     check_return_loss),
    ("Port variation at 1 GHz",   check_port_variation),
    ("Port variation across band", check_port_variation_band),
    ("Reciprocity  (Sij ≈ Sji)", check_reciprocity),
    ("Transmission decay",       check_transmission_decay),
    ("Phase continuity",         check_phase_continuity),
    ("Diagonal dominance",       check_diagonal_dominance),
]


# ═══════════════════════════════════════════════════════════════════════════
# TOP-LEVEL VALIDATE
# ═══════════════════════════════════════════════════════════════════════════

def validate(filepath, verbose=False):
    bar = "=" * 65

    print(f"\n{bar}")
    print(f"  Validating: {filepath}")
    print(f"{bar}")

    try:
        freqs_ghz, S = parse_s16p(filepath)
    except Exception as e:
        print(f"\n  ✗ PARSE ERROR: {e}\n")
        return False

    n_ports, _, n_freq = S.shape
    print(f"\n  Ports           : {n_ports}")
    print(f"  Frequency points: {n_freq}")
    print(f"  Freq range      : {freqs_ghz[0]:.4f} – {freqs_ghz[-1]:.4f} GHz\n")

    results = {}
    for name, fn in CHECKS:
        try:
            ok, msg = fn(S, freqs_ghz, verbose=verbose)
        except Exception as e:
            ok, msg = False, f"ERROR during check: {e}"

        results[name] = ok
        tick = "✓" if ok else "✗"
        print(f"  {tick} {name:<35s}  {msg}")

    passed = sum(results.values())
    total  = len(results)
    print(f"\n{bar}")
    print(f"  Result: {passed}/{total} checks passed", end="")
    if passed == total:
        print("  — looks good ✓")
    elif passed >= total - 1:
        print("  — minor issues, review above ⚠")
    else:
        print("  — PROBLEMS FOUND, check simulation output ✗")
    print(f"{bar}\n")

    return passed == total


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate .s16p S-parameter files from brain EMI simulations")
    parser.add_argument("path",
        help=".s16p file  OR  directory of .s16p files")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Print per-port breakdown for return-loss and transmission checks")
    args = parser.parse_args()

    path = args.path

    # Directory mode: validate every .s16p inside it
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.s16p")))
        if not files:
            print(f"No .s16p files found in {path}")
            sys.exit(1)
        print(f"Found {len(files)} .s16p file(s) in {path}")
        ok_count = sum(validate(fp, verbose=args.verbose) for fp in files)
        print(f"\n{'='*65}")
        print(f"  Batch: {ok_count}/{len(files)} files passed all checks")
        print(f"{'='*65}\n")
        sys.exit(0 if ok_count == len(files) else 1)

    # Single file mode
    if not os.path.isfile(path):
        print(f"ERROR: not a file or directory: {path}")
        sys.exit(1)

    ok = validate(path, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
