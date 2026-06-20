"""Render tabel validasi/replikasi Ding (ULB creditcard) -> PNG."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures" / "validasi_ding.png"

# (skenario, desk, ROC, PRAUC, dAP, p, warna) — REPLIKASI FAITHFUL metode Ding (one-class AE, error feature)
ROWS = [
    ("baseline", "LightGBM fitur asli", "0,9334", "0,7210", "—", "—", "base"),
    ("ae_error", "raw + error rekonstruksi AE (integrasi Ding)", "0,9232", "0,6697", "−0,0513", "0,857", "worse"),
    ("ae_error_only", "hanya error AE (detektor anomali)", "0,9159", "0,0732", "−0,6478", "1,000", "worse"),
    ("smote", "SMOTE oversampling", "0,9798", "0,8652", "+0,1443", "0,000", "win"),
    ("ae_error_smote", "AE error + SMOTE (klaim penuh Ding)", "0,9802", "0,8659", "+0,1450", "0,000", "win"),
]
cols = ["Skenario", "Deskripsi", "ROC-AUC", "PR-AUC", "Δ PR-AUC", "p(Δ≤0)"]
cw = [2000, 3500, 1250, 1250, 1300, 1000]

fig, ax = plt.subplots(figsize=(13, 0.6 * (len(ROWS) + 1) + 1.5))
ax.axis("off")
ax.set_title("Validasi Implementasi — Replikasi FAITHFUL metode Ding et al. (ULB credit-card)\n"
             "(AE one-class + fitur error rekonstruksi, persis kode Ding; beliau melaporkan ROC-AUC ~0,968)",
             fontsize=12.5, fontweight="bold", pad=16, loc="left")

cell = [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in ROWS]
t = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1, 1.9)
palette = {"base": "#dceaf5", "tie": "#fdf3d0", "worse": "#f7e3e0", "win": "#d5ead5"}
for (ri, ci), c in t.get_celld().items():
    c.set_edgecolor("#9aa7b4")
    if ri == 0:
        c.set_facecolor("#1f3b57"); c.set_text_props(color="white", fontweight="bold")
    else:
        f = ROWS[ri-1][6]; c.set_facecolor(palette[f])
        if f in ("base", "win"): c.set_text_props(fontweight="bold")
    if ci in (0, 1):
        c.set_text_props(ha="left"); c._loc = "left"
t.auto_set_column_width(col=list(range(len(cols))))

note = ("KESIMPULAN: (1) Dengan SMOTE, ROC-AUC 0,980 ≈/melebihi Ding (0,968) memakai arsitektur & metode Ding yang PERSIS -> implementasi BENAR. "
        "(2) Fitur error AE TIDAK membantu (−0,051); menambah AE di atas SMOTE = nol praktis (0,8659 vs 0,8652). "
        "(3) Seluruh peningkatan berasal dari OVERSAMPLING (SMOTE +0,144), bukan dari autoencoder.\n"
        "Protokol: stratified 60/20/20, seed 42, paired bootstrap 2000. Hijau = menang signifikan, merah = tidak membantu.")
fig.text(0.04, 0.015, note, fontsize=8.4, color="#445", va="bottom")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
