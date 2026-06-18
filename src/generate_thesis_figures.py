"""Generate thesis-ready figures from the active stratified_reset results.

Reads the experiment summary JSONs produced by the AE/augmentation harnesses and
exports the key Bab 4 figures for the active (stratified) protocol:
- AE feature-integration ablation (all variants hurt the baseline)
- Augmentation vs baseline and the AE-vs-SMOTE contrast on A0 (tie) vs A1 (win)
- Representation-dependence of the AE advantage (core finding)

Missing inputs are skipped gracefully. Run:
    python src/generate_thesis_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PROJECT_ROOT

RESET = PROJECT_ROOT / "outputs" / "stratified_reset"
OUTDIR = RESET / "thesis_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
C = {"base": "#64748B", "smote": "#2A9D8F", "ae": "#E76F51", "neg": "#C2410C", "pos": "#4B8B3B"}


def load(path: Path):
    p = Path(path)
    if not p.exists():
        print(f"[skip] missing {p}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name):
    out = OUTDIR / name
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print("saved", out)


def fig_feature_integration():
    d = load(RESET / "ae_integration_experiment_normal_ld32" / "experiment_summary.json")
    if not d:
        return
    r = d["results"]
    order = ["recon_global", "recon_grouped", "latent", "recon_grouped_plus_latent", "score_ensemble"]
    labels = ["+recon\nglobal", "+recon\ngrouped", "+latent", "+recon\n+latent", "score\nensemble"]
    deltas = [r[k]["bootstrap_vs_baseline"]["observed_delta_ap"] for k in order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, deltas, color=[C["neg"] if v < 0 else C["base"] for v in deltas])
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Delta test PR-AUC vs baseline A0")
    ax.set_title("AE sebagai fitur: semua varian menurunkan/menyamai baseline")
    for b, v in zip(bars, deltas):
        ax.annotate(f"{v:+.4f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="top" if v < 0 else "bottom", xytext=(0, -3 if v < 0 else 3),
                    textcoords="offset points", fontsize=8)
    save(fig, "fig_ae_feature_integration_ablation.png")


def fig_repr_dependence():
    a0 = load(RESET / "fair_augmentation_comparison" / "experiment_summary.json")
    a1 = load(RESET / "strong_baseline_full_budget" / "experiment_summary.json")
    if a1 is None:
        # fall back to one 800-tree split if full budget not ready
        a1 = load(RESET / "strong_baseline_augmentation" / "experiment_summary.json")
    if not a0 or not a1:
        return
    d0 = a0["ae_specific_contribution"]["ae_vs_smote_nc"]
    d1 = a1["ae_vs_smote_nc"]
    reps = ["A0\n(raw, NaN-native)", "A1\n(dense, freq-encoded)"]
    deltas = [d0["observed_delta_ap"], d1["observed_delta_ap"]]
    los = [d0["observed_delta_ap"] - d0["ci_2_5"], d1["observed_delta_ap"] - d1["ci_2_5"]]
    his = [d0["ci_97_5"] - d0["observed_delta_ap"], d1["ci_97_5"] - d1["observed_delta_ap"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(reps, deltas, yerr=[los, his], capsize=6,
                  color=[C["base"], C["pos"]])
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Delta test PR-AUC: AE - SMOTE-NC")
    ax.set_title("Keunggulan AE atas SMOTE bergantung representasi")
    for b, v in zip(bars, deltas):
        ax.annotate(f"{v:+.4f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", xytext=(0, 8), textcoords="offset points", fontsize=9)
    save(fig, "fig_representation_dependence_ae_vs_smote.png")


def fig_augmentation_levels():
    a1 = load(RESET / "strong_baseline_full_budget" / "experiment_summary.json")
    if a1 is None:
        a1 = load(RESET / "strong_baseline_augmentation" / "experiment_summary.json")
    if not a1:
        return
    r = a1["results"]
    names = ["baseline", "smote_nc", "ae_latent_smote"]
    labels = ["Baseline\n(A1)", "+SMOTE-NC", "+AE\n(usulan)"]
    aps = [r[k]["test_average_precision"] for k in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, aps, color=[C["base"], C["smote"], C["ae"]])
    ax.set_ylabel("Test PR-AUC")
    ax.set_ylim(min(aps) - 0.02, max(aps) + 0.02)
    ax.set_title("Augmentasi pada representasi padat (A1)")
    for b, v in zip(bars, aps):
        ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)
    save(fig, "fig_a1_augmentation_levels.png")


def fig_protocol_comparison():
    strat = load(RESET / "ae_augmentation_experiment" / "experiment_summary.json")
    chrono = load(RESET / "ae_augmentation_chronological" / "experiment_summary.json")
    if not strat or not chrono:
        return
    pairs = [("Stratified", strat), ("Chronological\n(temporal)", chrono)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(pairs)); w = 0.35
    base = [p[1]["results"]["baseline"]["test_average_precision"] for p in pairs]
    aug = [p[1]["results"]["augment_no_spw"]["test_average_precision"] for p in pairs]
    ax.bar(x - w / 2, base, w, label="Baseline", color=C["base"])
    ax.bar(x + w / 2, aug, w, label="+AE augmentation", color=C["ae"])
    ax.set_xticks(x, [p[0] for p in pairs])
    ax.set_ylabel("Test PR-AUC")
    ax.set_title("Augmentasi membantu di kedua protokol")
    for i in range(len(pairs)):
        ax.annotate(f"{base[i]:.3f}", (i - w / 2, base[i]), ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", fontsize=8)
        ax.annotate(f"{aug[i]:.3f}", (i + w / 2, aug[i]), ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", fontsize=8)
    ax.legend()
    save(fig, "fig_protocol_stratified_vs_temporal.png")


def fig_tuned_comparison():
    d = load(RESET / "a1_tuned_comparison" / "experiment_summary.json")
    if not d:
        return
    r = d["results"]
    names = ["baseline", "smote_nc", "ae_latent_smote"]
    labels = ["Baseline\n(tuned)", "+SMOTE-NC\n(tuned)", "+AE\n(tuned, usulan)"]
    aps = [r[k]["test_average_precision"] for k in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, aps, color=[C["base"], C["smote"], C["ae"]])
    ax.set_ylabel("Test PR-AUC")
    ax.set_ylim(min(aps) - 0.01, max(aps) + 0.01)
    ax.set_title("Tuned-vs-tuned (A1): keunggulan AE bertahan setelah tuning fair")
    for b, v in zip(bars, aps):
        ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)
    save(fig, "fig_a1_tuned_vs_tuned.png")


def main():
    fig_feature_integration()
    fig_repr_dependence()
    fig_augmentation_levels()
    fig_protocol_comparison()
    fig_tuned_comparison()
    print(f"\nFigures in: {OUTDIR}")


if __name__ == "__main__":
    main()
