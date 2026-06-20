"""Render tabel master 14 varian AE-fitur (Kaggle full-data) -> PNG."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures" / "tabel_master_14varian.png"

BASE = 0.822917
# (varian, deskripsi, PR-AUC, dAP, p)
ROWS = [
    ("baseline", "LightGBM fitur asli (acuan)", 0.822917, None, None),
    ("ae_mlp_stack", "Skor neural AE->MLP (OOF stacking)", 0.821738, -0.001179, 0.807),
    ("iforest_latent", "IsolationForest pada laten AE", 0.818510, -0.004407, 1.0),
    ("vae_anomaly", "Variational AE, skor anomali", 0.818042, -0.004875, 1.0),
    ("recon_error", "Error rekonstruksi (AE semua V)", 0.817680, -0.005237, 1.0),
    ("one_class_anomaly", "Error AE one-class (normal)", 0.816896, -0.006021, 1.0),
    ("contrast_anomaly", "Error normal-AE vs fraud-AE", 0.816674, -0.006244, 1.0),
    ("latent_distance", "Jarak Mahalanobis laten", 0.816332, -0.006585, 1.0),
    ("allnum_anomaly", "Anomali AE semua fitur numerik", 0.814295, -0.008622, 1.0),
    ("blockwise_ae", "AE per-blok korelasi V", 0.813142, -0.009775, 1.0),
    ("missingness_ae", "Embedding pola missing", 0.811310, -0.011607, 1.0),
    ("concat_latent", "V asli + 32 laten AE", 0.798514, -0.024403, 1.0),
    ("sae_latent", "Supervised AE laten (V)", 0.768797, -0.054120, 1.0),
    ("perfeat_anomaly", "Error per-fitur (339 dim)", 0.738178, -0.084739, 1.0),
    ("sae_allnum", "Supervised AE semua numerik", 0.733550, -0.089367, 1.0),
]

cols = ["Varian", "Deskripsi", "PR-AUC", "Δ vs baseline", "p(Δ≤0)", "Verdict"]
cell, flags = [], []
for name, desc, ap, d, p in ROWS:
    if d is None:
        cell.append([name, desc, f"{ap:.5f}", "—", "—", "acuan"]); flags.append("base")
    else:
        verdict = "seri" if p < 0.95 else "lebih buruk"
        cell.append([name, desc, f"{ap:.5f}", f"{d:+.5f}", f"{p:.3f}", verdict])
        flags.append("tie" if p < 0.95 else "worse")

fig, ax = plt.subplots(figsize=(13.5, 0.55 * (len(cell) + 1) + 1.4))
ax.axis("off")
ax.set_title("Tabel Master — 14 Pendekatan Autoencoder sebagai Penyedia Fitur untuk LightGBM\n"
             "(IEEE-CIS, stratified 60/20/20, seed 42, full data, paired bootstrap 2000)",
             fontsize=13, fontweight="bold", pad=16, loc="left")

t = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.75)
header = "#1f3b57"
for (r, c), cell_obj in t.get_celld().items():
    cell_obj.set_edgecolor("#9aa7b4")
    if r == 0:
        cell_obj.set_facecolor(header); cell_obj.set_text_props(color="white", fontweight="bold")
    else:
        f = flags[r - 1]
        bg = {"base": "#dceaf5", "tie": "#fdf3d0", "worse": "#f7e3e0"}[f]
        cell_obj.set_facecolor(bg)
        if f in ("base", "tie"):
            cell_obj.set_text_props(fontweight="bold")
    if c in (0, 1):
        cell_obj.set_text_props(ha="left"); cell_obj._loc = "left"
t.auto_set_column_width(col=list(range(len(cols))))

note = ("Temuan: dari 14 pendekatan, TIDAK ADA yang mengalahkan baseline. Terbaik (ae_mlp_stack) hanya SERI (p=0.807). "
        "Pola: makin banyak campur tangan AE -> makin buruk. LightGBM sudah mengekstrak hampir seluruh sinyal fitur "
        "IEEE-CIS yang telah direkayasa Vesta; kontribusi AE bersifat redundan. Kuning = seri, merah = signifikan lebih buruk.")
fig.text(0.04, 0.015, note, fontsize=8.2, color="#445", va="bottom", wrap=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
