"""Render grafik diagnosis: recon R^2 naik tapi delta AP tidak pulih -> PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "diagnosis_replace_v.png"

# full-data, 800 trees
latent = [16, 32, 64]
recon_r2 = [0.9015, 0.9326, 0.9454]
delta_ap = [-0.038987, -0.034141, -0.035466]

fig, ax1 = plt.subplots(figsize=(9, 5.2))

c1, c2 = "#1f3b57", "#b5482f"
ax1.set_xlabel("Dimensi laten autoencoder (kompresi makin kecil →)", fontsize=11)
ax1.set_ylabel("Recon R² (kualitas rekonstruksi V)", color=c1, fontsize=11)
l1 = ax1.plot(latent, recon_r2, "o-", color=c1, lw=2.5, ms=9, label="Recon R²")
ax1.tick_params(axis="y", labelcolor=c1)
ax1.set_ylim(0.88, 0.97)
for x, y in zip(latent, recon_r2):
    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), color=c1, fontsize=9, ha="center")

ax2 = ax1.twinx()
ax2.set_ylabel("Δ PR-AUC vs baseline", color=c2, fontsize=11)
l2 = ax2.plot(latent, delta_ap, "s--", color=c2, lw=2.5, ms=9, label="Δ PR-AUC vs baseline")
ax2.tick_params(axis="y", labelcolor=c2)
ax2.set_ylim(-0.05, 0.0)
ax2.axhline(0, color="#888", lw=1, ls=":")
for x, y in zip(latent, delta_ap):
    ax2.annotate(f"{y:+.4f}", (x, y), textcoords="offset points", xytext=(0, -16), color=c2, fontsize=9, ha="center")

ax1.set_xticks(latent)
ax1.set_title("Diagnosis desain awal: rekonstruksi membaik, tapi PR-AUC TIDAK pulih",
              fontsize=13, fontweight="bold", pad=14)

lines = l1 + l2
ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=10)

fig.text(0.5, -0.02,
         "R² naik 0.90→0.95 (rekonstruksi makin akurat) namun Δ PR-AUC tetap ~−0.035 (tidak membaik). "
         "Penyebab: AE meminimalkan rata-rata error → menghaluskan deviasi halus penanda fraud.\n"
         "Protokol: stratified 60/20/20, seed 42, LightGBM 800 trees. Baseline AP=0.7659; kontribusi importance fitur V = 22.4%.",
         ha="center", fontsize=8.3, color="#445")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
