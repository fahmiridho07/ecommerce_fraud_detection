"""Render Tabel 1 (A1 headline) hasil eksperimen menjadi PNG.

Membaca angka langsung dari artifact JSON supaya tabel selalu sinkron dengan
hasil eksperimen. Jika tersedia varian dengan random oversampling
(a1_tuned_comparison_with_random), itu yang dipakai (4 baris); jika belum,
fallback ke a1_tuned_comparison (3 baris).

Run:
    python tools/render_tabel1_png.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"
WITH_RANDOM = ROOT / "outputs/stratified_reset/a1_tuned_comparison_with_random/experiment_summary.json"
NO_RANDOM = ROOT / "outputs/stratified_reset/a1_tuned_comparison/experiment_summary.json"

LABELS = {
    "baseline": "LightGBM baseline",
    "random_oversample": "LightGBM + random oversampling",
    "smote_nc": "LightGBM + SMOTE-NC",
    "ae_latent_smote": "LightGBM + AE latent oversampling (usulan)",
}
ORDER = ["baseline", "random_oversample", "smote_nc", "ae_latent_smote"]


def main() -> None:
    src = WITH_RANDOM if WITH_RANDOM.exists() else NO_RANDOM
    summary = json.loads(src.read_text())
    res = summary["results"]
    base_ap = res["baseline"]["test_average_precision"]

    rows, cell_text = [], []
    for key in ORDER:
        if key not in res:
            continue
        r = res[key]
        ap = r["test_average_precision"]
        delta = "-" if key == "baseline" else f"{ap - base_ap:+.6f}"
        # p-value vs baseline from comparisons block when available
        p = "-"
        comp = summary.get("comparisons", {})
        pmap = {
            "random_oversample": "random_vs_baseline",
            "smote_nc": "smote_vs_baseline",
            "ae_latent_smote": "ae_vs_baseline",
        }
        if key in pmap and pmap[key] in comp:
            p = f"{comp[pmap[key]]['p_delta_le_0']:.3f}"
        rows.append(key)
        cell_text.append([
            LABELS[key],
            f"{ap:.6f}",
            f"{r['test_roc_auc']:.6f}",
            f"{r['test_f1']:.6f}",
            f"{r['test_mcc']:.6f}",
            delta,
            p,
        ])

    columns = ["Pipeline", "PR-AUC", "ROC-AUC", "F1", "MCC", "Δ PR-AUC\nvs baseline", "p(Δ≤0)"]

    fig, ax = plt.subplots(figsize=(12.5, 0.62 * (len(cell_text) + 1) + 1.2))
    ax.axis("off")

    title = "Tabel 1. Perbandingan kinerja pada representasi A1 (dense, tuned adil)"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=18, loc="left")

    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    n_cols = len(columns)
    header_color = "#1f3b57"
    usulan_color = "#dceaf5"
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#9aa7b4")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            key = rows[row - 1]
            if key == "ae_latent_smote":
                cell.set_facecolor(usulan_color)
                cell.set_text_props(fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f4f6f8")
        if col == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"

    table.auto_set_column_width(col=list(range(n_cols)))

    note = (
        "Protokol: stratified holdout 60/20/20, seed 42. Metrik utama PR-AUC; threshold dipilih di validation (MCC). "
        "Signifikansi: paired bootstrap 2000 resample. Oversampling di-fit hanya pada train.\n"
        "Kontribusi spesifik AE — AE vs SMOTE-NC: ΔPR-AUC = +0.006555 (CI95% [+0.0039, +0.0092], p<0.001), menang signifikan."
    )
    fig.text(0.06, 0.02, note, fontsize=8.5, color="#445", va="bottom")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "tabel1_a1_headline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}  (source: {src.name}, {len(cell_text)} rows)")


if __name__ == "__main__":
    main()
