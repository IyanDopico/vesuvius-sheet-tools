"""Publication-quality figure for the layer-count QA profile.

Two panels, one x-domain (z along the scroll axis):
  A: median distinct-wraps-per-ray with p10-p90 band, annotated.
  B: full (z, theta) heatmap, single-hue sequential.

Usage: python scripts/make_profile_figure.py output/scroll_run
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK = "#3d4451"
MUTED = "#7a8494"
LINE = "#2f6db3"  # mid-blue from a Blues ramp
BAND = "#2f6db3"

run = Path(sys.argv[1])
with open(run / "layer_count.json") as fh:
    data = json.load(fh)

profile = {int(k): np.array(v) for k, v in data["profile"].items()}
z0s = sorted(profile)
z_mid = np.array(z0s) + 128  # mid-slice of each slab
med = np.array([np.median(profile[z]) for z in z0s])
p10 = np.array([np.percentile(profile[z], 10) for z in z0s])
p90 = np.array([np.percentile(profile[z], 90) for z in z0s])
mat = np.array([profile[z] for z in z0s])
theta_step = data["theta_step_deg"]

fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(9.2, 7.2), dpi=150,
    gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.32},
)
fig.patch.set_facecolor("white")

# --- panel A: median + band ---
ax.fill_between(z_mid, p10, p90, color=BAND, alpha=0.16, linewidth=0,
                label="p10–p90 across rays")
ax.plot(z_mid, med, color=LINE, linewidth=2, marker="o", markersize=4.5,
        markerfacecolor="white", markeredgewidth=1.4, label="median")
# anchor: interior slices only (exclude the tip ramp), minimal RELATIVE spread
interior = med >= 30
rel_spread = np.where(interior, (p90 - p10) / np.maximum(med, 1), np.inf)
anchor_i = int(np.argmin(rel_spread))
ax.scatter([z_mid[anchor_i]], [med[anchor_i]], s=90, facecolor="white",
           edgecolor=INK, zorder=5, linewidth=1.6)
ax.annotate("anchor candidate\n(tightest spread)",
            (z_mid[anchor_i], med[anchor_i]),
            xytext=(z_mid[anchor_i] + 220, med[anchor_i] - 16),
            fontsize=8.5, color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9})
ax.annotate("scroll tip:\nwinding count ramps up", (z_mid[1], med[1]),
            xytext=(z_mid[1] + 150, med[1] + 24), fontsize=8.5, color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9})
plateau = med[4:].mean()
ax.axhspan(37, 46, color=MUTED, alpha=0.06, zorder=0)
ax.text(z_mid[-1], 47.5, "interior plateau N ≈ 37–46", ha="right",
        fontsize=8.5, color=MUTED)
ax.set_ylabel("distinct wraps crossed per ray", fontsize=9, color=INK)
ax.set_title(
    "PHerc1218 · layer-count QA over stitched lower half "
    "(26 slabs, mid-slices, 180 rays each)",
    fontsize=10.5, color=INK, loc="left", pad=10)
ax.legend(frameon=False, fontsize=8.5, loc="lower right",
          labelcolor=INK)
ax.set_xlim(z_mid[0] - 60, z_mid[-1] + 60)
ax.set_ylim(0, max(p90) * 1.12)

# --- panel B: heatmap ---
im = ax2.imshow(
    mat, aspect="auto", cmap="Blues", origin="lower",
    extent=[0, 360, z_mid[0], z_mid[-1]], interpolation="nearest",
)
cb = fig.colorbar(im, ax=ax2, pad=0.012, fraction=0.035)
cb.set_label("wraps per ray", fontsize=8.5, color=INK)
cb.ax.tick_params(labelsize=8, colors=MUTED)
cb.outline.set_visible(False)
ax2.set_xlabel("ray angle θ (degrees)", fontsize=9, color=INK)
ax2.set_ylabel("z (L1 slice)", fontsize=9, color=INK)
ax2.set_xticks([0, 90, 180, 270, 360])

for a in (ax, ax2):
    a.tick_params(labelsize=8, colors=MUTED)
    for s in a.spines.values():
        s.set_color("#d5dae2")
    a.set_facecolor("white")
ax.grid(axis="y", color="#eceff3", linewidth=0.8)
ax.set_axisbelow(True)

fig.text(0.01, 0.005,
         "1 slice = 17.28 µm · rays cast from section centroid · "
         "counts are stitched global instance ids (local segments, not full windings)",
         fontsize=7.5, color=MUTED)
out = run / "layer_count_figure.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
