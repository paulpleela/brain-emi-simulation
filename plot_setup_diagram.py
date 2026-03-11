"""
Generate a single annotated diagram of the brain-EMI simulation setup.
Shows the equatorial (XY) cross-section with all tissue layers and all 16 antennas.
Each antenna is a z-directed wire dipole: shown as a vertical ↕ bar at the feed point.
"""
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── Geometry constants (must match generate_inputs.py) ──────────────────────
cell               = 0.002          # 2 mm grid
head_cx, head_cy   = 0.25, 0.25     # head centre (m)
a_head             = 0.095          # semi-axis X (front-back)
b_head             = 0.075          # semi-axis Y (left-right)
scalp_thick        = 0.010
gray_thick         = 0.003
coupling_thick     = 0.005
n_antennas         = 16
dipole_arm_len     = 0.056          # 56 mm per arm (resonance 1.25 GHz in free space)
dipole_gap         = 0.002          # 2 mm gap (1 cell)

# ── Derived layer semi-axes ──────────────────────────────────────────────────
layers = {
    # (a, b, facecolor, edgecolor, zorder)
    "coupling":   (a_head + scalp_thick + coupling_thick,
                   b_head + scalp_thick + coupling_thick,
                   "#d4e9ff", "#5599cc", 2),
    "scalp_skull":(a_head + scalp_thick,
                   b_head + scalp_thick,
                   "#f5c28a", "#b07030", 3),
    "gray":       (a_head,         b_head,
                   "#c8a0c8", "#804080", 4),
    "white":      (a_head - gray_thick,
                   b_head - gray_thick,
                   "#e8e8f8", "#6060a0", 5),
}

# Ventricles (equatorial slice of the 3-D ellipsoids; project onto XY plane)
vent_a, vent_b = 0.020, 0.010
vent_sep = 0.015
vent_left_cx  = head_cx - vent_sep / 2
vent_right_cx = head_cx + vent_sep / 2

# Hemorrhagic lesion
lesion_x, lesion_y, lesion_r = head_cx - 0.02, head_cy, 0.015

# ── Compute antenna feed positions (same logic as generate_inputs.py) ────────
antennas = []
for i in range(n_antennas):
    angle = 2 * math.pi * i / n_antennas
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    r_x = a_head + scalp_thick + coupling_thick + cell
    r_y = b_head + scalp_thick + coupling_thick + cell
    cx = round((head_cx + r_x * cos_a) / cell) * cell
    cy = round((head_cy + r_y * sin_a) / cell) * cell
    pol = 'x' if abs(cos_a) >= abs(sin_a) else 'y'
    antennas.append(dict(cx=cx, cy=cy, pol=pol, angle=angle, idx=i + 1))

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')

# Convert to cm for nicer axis labels
def m2cm(v):
    return np.asarray(v) * 100

def ellipse(ax, cx, cy, a, b, **kwargs):
    theta = np.linspace(0, 2 * np.pi, 400)
    x = cx + a * np.cos(theta)
    y = cy + b * np.sin(theta)
    ax.fill(*m2cm([x, y]), **kwargs)
    ax.plot(*m2cm([x, y]), color=kwargs.get('edgecolor', 'k'), lw=0.8)

# Draw tissue layers outside → inside
for name, (a, b, fc, ec, zo) in layers.items():
    ellipse(ax, head_cx, head_cy, a, b, facecolor=fc, edgecolor=ec, zorder=zo)

# Ventricles (CSF)
theta = np.linspace(0, 2 * np.pi, 200)
for vx in [vent_left_cx, vent_right_cx]:
    x = vx + vent_a * np.cos(theta)
    y = head_cy + vent_b * np.sin(theta)
    ax.fill(*m2cm([x, y]), color='#80d0ff', zorder=6)
    ax.plot(*m2cm([x, y]), color='#0060b0', lw=0.8, zorder=6)

# Hemorrhagic lesion
lesion = plt.Circle(m2cm([lesion_x, lesion_y]), m2cm(lesion_r),
                    color='#cc2222', zorder=7, label='Hemorrhage (blood, εᵣ=61)')
ax.add_patch(lesion)

# Antennas: z-directed wire dipoles.
# In the XY equatorial cross-section, the dipole arms go into/out of the page (±z).
# Represent each as a vertical double-headed arrow (↕) in the diagram plane,
# centred at the feed point, with length proportional to arm length scaled to diagram.
# Physical arm = 56 mm → scale to 2.0 cm in diagram (fits without overlap).
arm_cm = 2.0   # visual length per arm in diagram (cm)  — physical: 56 mm each
gap_cm = dipole_gap * 100      # 2 mm → 0.2 cm

for ant in antennas:
    cx_cm, cy_cm = m2cm(ant['cx']), m2cm(ant['cy'])

    # Upper arm (toward +z, shown as upward bar)
    ax.annotate('', xy=(cx_cm, cy_cm + arm_cm), xytext=(cx_cm, cy_cm + gap_cm / 2),
                arrowprops=dict(arrowstyle='->', color='#0044cc', lw=1.5), zorder=9)
    # Lower arm (toward -z, shown as downward bar)
    ax.annotate('', xy=(cx_cm, cy_cm - arm_cm), xytext=(cx_cm, cy_cm - gap_cm / 2),
                arrowprops=dict(arrowstyle='->', color='#0044cc', lw=1.5), zorder=9)
    # Feed gap dot
    ax.plot(cx_cm, cy_cm, 'o', color='#0044cc', ms=4, zorder=10,
            markeredgecolor='#002288', markeredgewidth=0.5)

    # Antenna number label (offset outward along radial direction)
    angle = ant['angle']
    lx = head_cx + (a_head + scalp_thick + coupling_thick + 0.028) * math.cos(angle)
    ly = head_cy + (b_head + scalp_thick + coupling_thick + 0.028) * math.sin(angle)
    ax.text(m2cm(lx), m2cm(ly), str(ant['idx']),
            ha='center', va='center', fontsize=6.5, color='#222222',
            fontweight='bold', zorder=11)

# Head-centre cross
ax.plot(*m2cm([head_cx, head_cy]), '+', color='black', ms=8, mew=1.2, zorder=12)

# ── Legend ───────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(fc='#d4e9ff', ec='#5599cc', label='Coupling medium (εᵣ=36, σ=0.3 S/m, 5 mm)'),
    mpatches.Patch(fc='#f5c28a', ec='#b07030', label='Scalp + Skull (εᵣ=12, σ=0.2 S/m, 10 mm)'),
    mpatches.Patch(fc='#c8a0c8', ec='#804080', label='Gray matter (εᵣ=52, σ=0.97 S/m, 3 mm)'),
    mpatches.Patch(fc='#e8e8f8', ec='#6060a0', label='White matter (εᵣ=38, σ=0.57 S/m)'),
    mpatches.Patch(fc='#80d0ff', ec='#0060b0', label='CSF ventricles (εᵣ=80, σ=2.0 S/m)'),
    mpatches.Patch(fc='#cc2222', ec='#cc2222', label='Hemorrhage / blood (εᵣ=61, σ=1.54 S/m)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0044cc',
               markersize=6, label='Dipole feed gap (TL, 73 Ω)'),
    plt.Line2D([0], [0], color='#0044cc', lw=1.5,
               label='Dipole arms ±z (56 mm each, PEC wire)'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=7.5,
          framealpha=0.92, edgecolor='#aaaaaa', title='Materials & elements', title_fontsize=8)

# ── Dimension annotations ────────────────────────────────────────────────────
# Semi-axis b arrow (horizontal, top of head)
arrow_y = m2cm(head_cy + b_head + scalp_thick + coupling_thick + 0.045)
ax.annotate('', xy=(m2cm(head_cx + b_head), arrow_y),
            xytext=(m2cm(head_cx), arrow_y),
            arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.2))
ax.text(m2cm(head_cx + b_head / 2), arrow_y + 0.4, 'b = 7.5 cm',
        ha='center', va='bottom', fontsize=8, color='#444444')

# Semi-axis a arrow (vertical, right side)
arrow_x = m2cm(head_cx + a_head + scalp_thick + coupling_thick + 0.045)
ax.annotate('', xy=(arrow_x, m2cm(head_cy + a_head)),
            xytext=(arrow_x, m2cm(head_cy)),
            arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.2))
ax.text(arrow_x + 0.4, m2cm(head_cy + a_head / 2), 'a = 9.5 cm',
        ha='left', va='center', fontsize=8, color='#444444')

# ── Axes ─────────────────────────────────────────────────────────────────────
margin = 6
all_x = [m2cm(a['cx']) for a in antennas]
all_y = [m2cm(a['cy']) for a in antennas]
ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
ax.set_xlabel('x  (cm) — anterior ↔ posterior', fontsize=10)
ax.set_ylabel('y  (cm) — left ↔ right', fontsize=10)
ax.set_title(
    'Brain-EMI simulation — equatorial cross-section (XY plane at z = 25 cm)\n'
    '16 z-directed wire dipoles (114 mm total, resonant 1.25 GHz)  |  600×600×600 mm domain  |  2 mm grid  |  0.5–2 GHz  |  60 ns / 90 pts',
    fontsize=10, pad=10
)
ax.grid(True, alpha=0.25, lw=0.5)
ax.tick_params(labelsize=8)

plt.tight_layout()
out = 'setup_diagram.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f'Saved: {out}')
plt.show()
