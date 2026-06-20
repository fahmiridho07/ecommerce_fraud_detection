"""Render atribusi Part 6 (usulan) vs Part 7 (kontrol) -> PNG bar chart ΔAP."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures" / "atribusi_relasional.png"

groups = ["Relasional\n(profil entitas)", "Kategorikal\n(high-cardinality)"]
usulan = [0.01710, 0.01876]      # entity_ae, cat_embedding
usulan_lbl = ["entity_ae (AE)", "cat_embedding (embedding)"]
kontrol = [0.02071, 0.00903]     # entity_raw, target_encode
kontrol_lbl = ["entity_raw (agregasi, tanpa AE)", "target_encode (tanpa embedding)"]

x = np.arange(len(groups)); w = 0.34
fig, ax = plt.subplots(figsize=(10, 5.6))
b1 = ax.bar(x - w/2, usulan, w, label="Usulan (AE / embedding)", color="#1f6fb2")
b2 = ax.bar(x + w/2, kontrol, w, label="Kontrol (tanpa AE)", color="#e08214")

ax.axhline(0, color="#888", lw=1)
ax.set_ylabel("Δ PR-AUC vs baseline (0.82292)", fontsize=11)
ax.set_title("Atribusi: apakah kenaikan dari AUTOENCODER atau dari FITUR-nya?\n"
             "(IEEE-CIS, stratified, full data, paired bootstrap; semua p<0.001)",
             fontsize=12.5, fontweight="bold", pad=14)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11)
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, 0.026)

for bars, vals, lbls in ((b1, usulan, usulan_lbl), (b2, kontrol, kontrol_lbl)):
    for bar, v, l in zip(bars, vals, lbls):
        ax.annotate(f"+{v:.4f}", (bar.get_x()+bar.get_width()/2, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, fontweight="bold")
        ax.annotate(l, (bar.get_x()+bar.get_width()/2, 0), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, rotation=90, color="white", va="bottom")

# panah kesimpulan per grup
ax.annotate("kontrol MENANG →\nAE tidak berkontribusi", (0, 0.0235), ha="center", fontsize=8.5,
            color="#b5482f", fontweight="bold")
ax.annotate("usulan menang,\ntapi itu embedding\nsupervised (bukan AE)", (1, 0.0215), ha="center",
            fontsize=8.5, color="#2e6b2e", fontweight="bold")

fig.text(0.5, -0.02,
         "Relasional: agregasi entitas MENTAH (+0.0207) > AE-compressed (+0.0171) → kenaikan berasal dari FITUR relasional, bukan autoencoder.\n"
         "Kategorikal: embedding (+0.0188) > target encoding (+0.0090), namun embedding bersifat supervised — secara teknis bukan autoencoder.",
         ha="center", fontsize=8.3, color="#445")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
