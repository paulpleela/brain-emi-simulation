"""
Plot S-parameters from a .s16p file.

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

# Plot
fig, ax = plt.subplots(figsize=(11, 6))

n_ports = S.shape[1]
seen_seps = set()
for i in range(n_ports):
    for j in range(n_ports):
        if i == j:
            continue  # skip Sii (reflection terms)
        mag_db = 20 * np.log10(np.abs(S[:, i, j]) + 1e-12)
        sep = min((j - i) % n_ports, (i - j) % n_ports)
        label = f"sep={sep}" if sep not in seen_seps else None
        seen_seps.add(sep)
        ax.plot(freqs_ghz, mag_db, linewidth=0.7, alpha=0.5, label=label)

ax.set_xlabel("Frequency (GHz)", fontsize=12)
ax.set_ylabel("Magnitude (dB)", fontsize=12)
ax.set_title(f"Transmission S-parameters (off-diagonal) — {os.path.basename(filepath)}", fontsize=13)
ax.set_xlim(freqs_ghz[0], freqs_ghz[-1])
ax.legend(title="Port separation", fontsize=8, loc="lower left", ncol=4)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = filepath.replace(".s16p", "_transmission.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
