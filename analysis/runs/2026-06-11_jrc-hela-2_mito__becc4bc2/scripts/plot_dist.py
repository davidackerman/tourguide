"""Mitochondria morphometrics plots for jrc_hela-2 mito_seg.

Run from the run-folder root:
    uv run --project /Users/ackermand/Documents/programming/tourguide/analysis \
        python scripts/plot_dist.py
Reads tables/mito.csv, writes plots/mito_dist.png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
csv = os.path.join(RUN, "tables", "mito.csv")
out = os.path.join(RUN, "plots", "mito_dist.png")

df = pd.read_csv(csv)
vol_um3 = df["volume_nm_3"] / 1e9
eq_diam_um = (2.0 * (3.0 * df["volume_nm_3"] / (4.0 * np.pi)) ** (1.0 / 3.0)) / 1000.0

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("jrc_hela-2 mito_seg - mitochondria morphometrics (n=%d, scale s4)" % len(df),
             fontsize=13, fontweight="bold")

ax = axes[0, 0]
ax.hist(vol_um3, bins=40, color="#4C72B0", edgecolor="white")
ax.set_xscale("log")
ax.set_xlabel("Volume (um^3)"); ax.set_ylabel("Count")
ax.set_title("Volume distribution (log scale)")
ax.axvline(vol_um3.median(), color="crimson", ls="--", lw=1,
           label="median = %.3f um^3" % vol_um3.median()); ax.legend(fontsize=8)

ax = axes[0, 1]
ax.hist(eq_diam_um, bins=40, color="#55A868", edgecolor="white")
ax.set_xlabel("Equivalent-sphere diameter (um)"); ax.set_ylabel("Count")
ax.set_title("Equivalent diameter distribution")
ax.axvline(eq_diam_um.median(), color="crimson", ls="--", lw=1,
           label="median = %.2f um" % eq_diam_um.median()); ax.legend(fontsize=8)

ax = axes[1, 0]
sv = np.sort(vol_um3.values)[::-1]
cum = np.cumsum(sv) / sv.sum() * 100.0
ax.plot(np.arange(1, len(sv) + 1), cum, color="#C44E52", lw=2)
ax.set_xlabel("Mitochondrion rank (largest -> smallest)")
ax.set_ylabel("Cumulative volume (%)")
ax.set_title("Cumulative volume contribution"); ax.grid(alpha=0.3)
n50 = int(np.searchsorted(cum, 50.0)) + 1
ax.axhline(50, color="gray", ls=":", lw=1)
ax.annotate("top %d objects = 50%% of volume" % n50, xy=(n50, 50),
            xytext=(len(sv) * 0.25, 35), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"))

ax = axes[1, 1]
sc = ax.scatter(df["com_z_nm"] / 1000.0, vol_um3, c=vol_um3,
                cmap="viridis", s=18, norm=matplotlib.colors.LogNorm())
ax.set_yscale("log")
ax.set_xlabel("Centroid z (um)"); ax.set_ylabel("Volume (um^3, log)")
ax.set_title("Volume vs. depth")
fig.colorbar(sc, ax=ax, label="Volume (um^3)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out, dpi=130)
print("WROTE", out)
