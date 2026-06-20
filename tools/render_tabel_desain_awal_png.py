"""Render tabel 'desain awal' (AE merekonstruksi/mengganti blok fitur V) -> PNG.

Desain awal proposal: blok fitur V DIGANTI oleh hasil rekonstruksi autoencoder,
lalu digabung dengan fitur non-V dan dilatih LightGBM, dibanding baseline.
Angka dibaca dari run STRATIFIED (sebanding dengan Tabel 1 & Tabel 2).

Run:
    python tools/render_tabel_desain_awal_png.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"
SRC = ROOT / "outputs/stratified_reset/initial_design_replace_v/experiment_summary.json"

ROWS = [
    ("baseline", "Baseline — fitur V asli dipertahankan", False),
    ("replace_v_ae_reconstruction", "Ganti V → rekonstruksi AE + non-V (desain awal)", True),
]


def main() -> None:
    res = json.loads(SRC.read_text())["results"]
    base_ap = res["baseline"]["test_average_precision"]

    cell_text, flags = [], []
    for key, label, is_awal in ROWS:
        r = res[key]
        ap = r["test_average_precision"]
        if key == "baseline":
            delta, p = "-", "-"
        else:
            delta = f"{ap - base_ap:+.6f}"
            p = f"{r['bootstrap_vs_baseline']['p_delta_le_0']:.3f}"
        cell_text.append([
            label,
            f"{ap:.6f}",
            f"{r['test_roc_auc']:.6f}",
            f"{r['test_f1']:.6f}",
            f"{r['test_mcc']:.6f}",
            delta,
            p,
        ])
        flags.append(is_awal)

    columns = ["Desain", "PR-AUC", "ROC-AUC", "F1", "MCC", "Δ PR-AUC\nvs baseline", "p(Δ≤0)"]

    fig, ax = plt.subplots(figsize=(12.5, 0.62 * (len(cell_text) + 1) + 1.3))
    ax.axis("off")
    ax.set_title(
        "Tabel. Desain awal usulan — Autoencoder mengganti blok fitur V (stratified)",
        fontsize=14, fontweight="bold", pad=18, loc="left",
    )

    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    header_color = "#1f3b57"
    base_color = "#dceaf5"
    awal_color = "#f7e3e0"
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#9aa7b4")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if not flags[row - 1]:
                cell.set_facecolor(base_color)
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor(awal_color)
                cell.set_text_props(fontweight="bold")
        if col == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"

    table.auto_set_column_width(col=list(range(len(columns))))

    note = (
        "Protokol: stratified holdout 60/20/20, seed 42, LightGBM full budget, scale_pos_weight dari data. "
        "Metrik utama PR-AUC; threshold dipilih di validation (MCC). Signifikansi: paired bootstrap 2000 resample.\n"
        "Protokol SAMA dengan Tabel 1 & Tabel 2 (baseline 0.821840), sehingga ketiga tabel dapat disandingkan. "
        "Temuan: mengganti V dengan rekonstruksi AE MENURUNKAN PR-AUC (Δ −0.0529, p=1.000) — desain awal tidak memperbaiki baseline."
    )
    fig.text(0.05, 0.02, note, fontsize=8.3, color="#445", va="bottom")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "tabel_desain_awal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}  ({len(cell_text)} rows)")


if __name__ == "__main__":
    main()
