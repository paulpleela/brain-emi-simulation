"""
Plot S-parameters from a .s16p file.

Shows two panels:
  Top:    S11 return loss for all 16 ports (diagonal) — reveals antenna resonance
  Bottom: Off-diagonal transmission S-params grouped by port separation

Usage:
    python visualise_s16p.py sparams/scenario_001.s16p
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

filepath = sys.argv[1] if len(sys.argv) > 1 else "sparams/scenario_001.s16p"

# Load
ntwk = rf.Network(filepath)
freqs_ghz = ntwk.f / 1e9
S = ntwk.s  # shape: (n_freq, n_ports, n_ports)

n_ports = S.shape[1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
fig.suptitle(os.path.basename(filepath), fontsize=13)

# ── Top panel: S11 return loss (diagonal) ────────────────────────────────────
for i in range(n_ports):
    mag_db = 20 * np.log10(np.abs(S[:, i, i]) + 1e-12)
    ax1.plot(freqs_ghz, mag_db, linewidth=1.0, alpha=0.7, label=f"S{i+1}{i+1}")

ax1.axhline(-10, color='red', linestyle='--', linewidth=0.8, label='-10 dB threshold')
ax1.set_ylabel("Return Loss |Sii| (dB)", fontsize=11)
ax1.set_title("Return loss — antenna resonance should appear as a dip below −10 dB", fontsize=10)
ax1.legend(fontsize=6, loc='lower right', ncol=4, title='Port')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-40, 5)

# ── Bottom panel: off-diagonal transmission ───────────────────────────────────
seen_seps = set()
for i in range(n_ports):
    for j in range(n_ports):
        if i == j:
            continue
        mag_db = 20 * np.log10(np.abs(S[:, i, j]) + 1e-12)
        sep = min((j - i) % n_ports, (i - j) % n_ports)
        label = f"sep={sep}" if sep not in seen_seps else None
        seen_seps.add(sep)
        ax2.plot(freqs_ghz, mag_db, linewidth=0.7, alpha=0.5, label=label)

ax2.set_xlabel("Frequency (GHz)", fontsize=11)
ax2.set_ylabel("Transmission |Sij| (dB)", fontsize=11)
ax2.set_title("Transmission (off-diagonal)", fontsize=10)
ax2.set_xlim(freqs_ghz[0], freqs_ghz[-1])
ax2.legend(title="Port separation", fontsize=8, loc="lower left", ncol=4)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = filepath.replace(".s16p", "_sparams.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
