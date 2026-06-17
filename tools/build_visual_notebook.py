"""Build the thesis visualization notebook.

The notebook is generated from this script so the report can be rebuilt when
experiment artifacts change, while keeping the `.ipynb` self-contained.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "notebook.ipynb"


def _source(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


cells = [
    md(
        """
        # Notebook Visualisasi Draft Tugas Akhir

        Notebook ini adalah generator visual untuk draft skripsi IEEE-CIS Fraud Detection.
        Fokusnya bukan melatih ulang model, tetapi membaca artefak canonical di
        `outputs/initial_proposal/` lalu mengekspor gambar dan tabel siap pakai untuk
        Bab 3, Bab 4, Bab 5, dan lampiran.

        Narasi yang dijaga:

        - **P01-P04 historis**: tuned LightGBM P02 mengalahkan latent replacement AE P03/P04.
        - **AE-05 transisi**: hybrid top-25 `V*` + latent LD32 + reconstruction error pertama kali mengalahkan P02 lama.
        - **Kandidat aktif terbaru**: fixed score-level ensemble memberi Test AP tertinggi, yaitu 0.529114.
        """
    ),
    md(
        """
        ## Best Practice Yang Diikuti

        1. Split temporal, bukan random split, karena fraud detection rentan dataset shift.
        2. Average Precision / PR-AUC sebagai metrik utama untuk data sangat imbalanced.
        3. ROC-AUC, F1, MCC, dan confusion matrix ditampilkan sebagai metrik pendukung.
        4. Angka lintas-paper tidak dibandingkan mentah jika split, sampling, dan feature engineering berbeda.
        5. Paired bootstrap dipakai untuk klaim delta PR-AUC karena model dinilai pada baris test yang sama.
        6. Autoencoder diposisikan sebagai sinyal komplementer, bukan pengganti fitur tabular mentah.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import warnings

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from IPython.display import display
        from sklearn.metrics import (
            average_precision_score,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )

        warnings.filterwarnings("ignore", category=FutureWarning)

        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

        PALETTE = {
            "blue": "#2F6B9A",
            "teal": "#2A9D8F",
            "orange": "#E76F51",
            "green": "#4B8B3B",
            "purple": "#7A5195",
            "gray": "#64748B",
            "red": "#C2410C",
        }


        def find_repo_root(start=None):
            start = Path.cwd() if start is None else Path(start)
            for candidate in [start, *start.parents]:
                if (candidate / "src").exists() and (candidate / "outputs").exists():
                    return candidate
            raise RuntimeError("Repo root tidak ditemukan. Jalankan notebook dari dalam ecommerce_fraud_detection/.")


        REPO_ROOT = find_repo_root()
        WORKSPACE_ROOT = REPO_ROOT.parent
        ARTIFACT_ROOT = REPO_ROOT / "outputs" / "initial_proposal"
        LAMPIRAN_ROOT = WORKSPACE_ROOT / "3. Lampiran Gambar dan Tabel" / "generated" / "draft_assets"
        FIG_DIR = LAMPIRAN_ROOT / "figures"
        TABLE_DIR = LAMPIRAN_ROOT / "tables"
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)

        print("Repo root:", REPO_ROOT)
        print("Export figures:", FIG_DIR)
        print("Export tables:", TABLE_DIR)


        def read_csv(path, **kwargs):
            path = Path(path)
            if not path.exists():
                print(f"[skip] Missing: {path}")
                return pd.DataFrame()
            return pd.read_csv(path, **kwargs)


        def read_json(path):
            path = Path(path)
            if not path.exists():
                print(f"[skip] Missing: {path}")
                return {}
            return json.loads(path.read_text(encoding="utf-8"))


        def export_fig(fig, filename):
            out = FIG_DIR / filename
            fig.savefig(out, bbox_inches="tight", facecolor="white")
            print("saved", out.relative_to(WORKSPACE_ROOT))
            return out


        def export_table(df, filename, index=False):
            out = TABLE_DIR / filename
            df.to_csv(out, index=index)
            print("saved", out.relative_to(WORKSPACE_ROOT))
            return out


        def clean_label(label):
            replacements = {
                "P01 baseline default": "P01\\nBaseline default",
                "P02 tuned baseline": "P02\\nTuned baseline",
                "Best preprocessing baseline": "Best preprocessing\\nbaseline",
                "AE all-mask LD32 latent add-on": "AE LD32 latent\\nadd-on",
                "Fixed 0.50 score ensemble": "Fixed 0.50\\nscore ensemble",
                "Tuned alpha 10-trial ensemble": "Tuned alpha\\nensemble",
            }
            return replacements.get(str(label), str(label).replace("_", " "))


        def add_bar_labels(ax, orient="v", digits=4):
            if orient == "v":
                for patch in ax.patches:
                    value = patch.get_height()
                    if np.isfinite(value):
                        ax.annotate(
                            f"{value:.{digits}f}",
                            (patch.get_x() + patch.get_width() / 2, value),
                            ha="center",
                            va="bottom",
                            xytext=(0, 3),
                            textcoords="offset points",
                            fontsize=8,
                        )
            else:
                for patch in ax.patches:
                    value = patch.get_width()
                    if np.isfinite(value):
                        ax.annotate(
                            f"{value:.{digits}f}",
                            (value, patch.get_y() + patch.get_height() / 2),
                            ha="left",
                            va="center",
                            xytext=(3, 0),
                            textcoords="offset points",
                            fontsize=8,
                        )
        """
    ),
    code(
        """
        PATHS = {
            "initial_comparison": ARTIFACT_ROOT / "final_comparison" / "initial_proposal_comparison.csv",
            "extended_comparison": ARTIFACT_ROOT / "final_comparison" / "extended_proposal_comparison.csv",
            "preprocessing_ablation": ARTIFACT_ROOT / "preprocessing_ablation" / "preprocessing_ablation_extended_comparison.csv",
            "final_candidate": ARTIFACT_ROOT / "preprocessing_ablation" / "final_candidate_comparison.csv",
            "split_summary": ARTIFACT_ROOT / "preprocessing_diagnostics" / "split_summary.csv",
            "missingness_group": ARTIFACT_ROOT / "preprocessing_diagnostics" / "missingness_by_group_split.csv",
            "categorical_unknown": ARTIFACT_ROOT / "preprocessing_diagnostics" / "categorical_unknown_rates.csv",
            "numeric_shift": ARTIFACT_ROOT / "preprocessing_diagnostics" / "numeric_distribution_shift.csv",
            "top_v_sweep": ARTIFACT_ROOT / "representation_ablation" / "top_v_retention_sweep.csv",
            "bootstrap_final": ARTIFACT_ROOT / "preprocessing_ablation" / "score_ensemble_baseline_all_masked_ld32_fixed_050_canonical" / "paired_bootstrap_pr_auc_delta.csv",
            "bootstrap_final_summary": ARTIFACT_ROOT / "preprocessing_ablation" / "score_ensemble_baseline_all_masked_ld32_fixed_050_canonical" / "paired_bootstrap_summary.json",
            "final_scores_test": ARTIFACT_ROOT / "preprocessing_ablation" / "score_ensemble_baseline_all_masked_ld32_fixed_050_canonical" / "scores_test.csv",
            "final_threshold": ARTIFACT_ROOT / "preprocessing_ablation" / "score_ensemble_baseline_all_masked_ld32_fixed_050_canonical" / "threshold_selection.csv",
            "final_confusion": ARTIFACT_ROOT / "preprocessing_ablation" / "score_ensemble_baseline_all_masked_ld32_fixed_050_canonical" / "confusion_matrix_test.csv",
            "baseline_threshold": ARTIFACT_ROOT / "preprocessing_ablation" / "baseline_frequency_missingness_time_amount_fixed_p02" / "threshold_selection.csv",
            "baseline_confusion": ARTIFACT_ROOT / "preprocessing_ablation" / "baseline_frequency_missingness_time_amount_fixed_p02" / "confusion_matrix_test.csv",
            "p02_importance": ARTIFACT_ROOT / "optuna" / "baseline_lgbm_tuned" / "feature_importance.csv",
            "best_baseline_importance": ARTIFACT_ROOT / "preprocessing_ablation" / "baseline_frequency_missingness_time_amount_fixed_p02" / "feature_importance.csv",
            "ae_component_importance": ARTIFACT_ROOT / "preprocessing_ablation" / "baseline_latent_all_masked_ld32_frequency_missingness_time_amount_fixed_p02" / "feature_importance.csv",
            "ae_drift": ARTIFACT_ROOT / "diagnostics" / "ae_reconstruction_drift_by_split.csv",
            "ae_error_by_class": ARTIFACT_ROOT / "diagnostics" / "ae_reconstruction_error_by_fraud_class.csv",
            "topk_capture": ARTIFACT_ROOT / "preprocessing_ablation" / "score_ensemble_baseline_all_masked_ld32_diagnostics" / "topk_overlap_and_fraud_capture.csv",
            "score_complementarity": ARTIFACT_ROOT / "preprocessing_ablation" / "score_ensemble_baseline_all_masked_ld32_diagnostics" / "score_complementarity_summary.json",
        }

        initial_cmp = read_csv(PATHS["initial_comparison"])
        extended_cmp = read_csv(PATHS["extended_comparison"])
        prep_cmp = read_csv(PATHS["preprocessing_ablation"])
        final_cmp = read_csv(PATHS["final_candidate"])
        split_summary = read_csv(PATHS["split_summary"])
        missingness_group = read_csv(PATHS["missingness_group"])
        cat_unknown = read_csv(PATHS["categorical_unknown"])
        numeric_shift = read_csv(PATHS["numeric_shift"])
        top_v_sweep = read_csv(PATHS["top_v_sweep"])
        bootstrap_delta = read_csv(PATHS["bootstrap_final"])
        bootstrap_summary = read_json(PATHS["bootstrap_final_summary"])
        score_summary = read_json(PATHS["score_complementarity"])
        topk_capture = read_csv(PATHS["topk_capture"])

        for name, df in {
            "initial_cmp": initial_cmp,
            "extended_cmp": extended_cmp,
            "prep_cmp": prep_cmp,
            "final_cmp": final_cmp,
            "split_summary": split_summary,
        }.items():
            print(f"{name}: {df.shape}")

        if not initial_cmp.empty:
            export_table(initial_cmp, "tabel_4_1_initial_proposal_comparison.csv")
        if not extended_cmp.empty:
            export_table(extended_cmp, "tabel_4_2_extended_proposal_comparison_ae05.csv")
        if not prep_cmp.empty:
            export_table(prep_cmp, "tabel_4_3_preprocessing_ablation_extended.csv")
        if not final_cmp.empty:
            export_table(final_cmp, "tabel_4_4_final_candidate_comparison.csv")
        if not topk_capture.empty:
            export_table(topk_capture, "tabel_4_5_topk_fraud_capture.csv")
        if bootstrap_summary:
            export_table(pd.DataFrame([bootstrap_summary]), "tabel_4_6_bootstrap_final_summary.csv")
        """
    ),
    md(
        """
        ## Visual Bab 3 - Alur Metodologi

        Dua diagram berikut cocok untuk Bab 3: alur penelitian umum dan detail final score-level ensemble.
        """
    ),
    code(
        """
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


        def draw_box(ax, x, y, w, h, text, color, fontsize=9):
            patch = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                linewidth=1.1,
                edgecolor=color,
                facecolor=color,
                alpha=0.12,
            )
            ax.add_patch(patch)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
            return patch


        def arrow(ax, start, end, color="#334155"):
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.2,
                    color=color,
                    shrinkA=4,
                    shrinkB=4,
                )
            )


        fig, ax = plt.subplots(figsize=(12, 5.2))
        ax.set_axis_off()
        for box in [
            (0.03, 0.63, 0.15, 0.22, "IEEE-CIS\\ntransaction + identity", PALETTE["blue"]),
            (0.24, 0.63, 0.15, 0.22, "Merge by\\nTransactionID", PALETTE["teal"]),
            (0.45, 0.63, 0.15, 0.22, "Chronological split\\nby TransactionDT", PALETTE["orange"]),
            (0.66, 0.73, 0.14, 0.18, "Train", PALETTE["gray"]),
            (0.66, 0.49, 0.14, 0.18, "Validation", PALETTE["gray"]),
            (0.66, 0.25, 0.14, 0.18, "Test", PALETTE["gray"]),
            (0.85, 0.63, 0.12, 0.22, "Modeling +\\nevaluation", PALETTE["green"]),
        ]:
            draw_box(ax, *box)
        arrow(ax, (0.18, 0.74), (0.24, 0.74))
        arrow(ax, (0.39, 0.74), (0.45, 0.74))
        arrow(ax, (0.60, 0.74), (0.66, 0.82))
        arrow(ax, (0.60, 0.74), (0.66, 0.58))
        arrow(ax, (0.60, 0.74), (0.66, 0.34))
        arrow(ax, (0.80, 0.82), (0.85, 0.76))
        arrow(ax, (0.80, 0.58), (0.85, 0.74))
        arrow(ax, (0.80, 0.34), (0.85, 0.68))
        ax.text(
            0.5,
            0.08,
            "Output: PR-AUC/AP utama, ROC-AUC/F1/MCC pendukung, confusion matrix, paired bootstrap",
            ha="center",
            fontsize=10,
            color="#334155",
        )
        ax.set_title("Alur Penelitian dan Evaluasi Temporal", pad=14)
        export_fig(fig, "gambar_3_1_alur_penelitian_temporal.png")
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.set_axis_off()
        for box in [
            (0.04, 0.62, 0.16, 0.20, "Train/validation/test\\nchronological split", PALETTE["blue"]),
            (0.27, 0.72, 0.20, 0.18, "Preprocessing-strengthened\\nLightGBM", PALETTE["teal"]),
            (0.27, 0.40, 0.20, 0.18, "Mask-aware denoising AE\\nLD32 on V*", PALETTE["orange"]),
            (0.53, 0.40, 0.17, 0.18, "AE latent\\nLightGBM score", PALETTE["orange"]),
            (0.53, 0.72, 0.17, 0.18, "Baseline\\nfraud score", PALETTE["teal"]),
            (0.76, 0.56, 0.18, 0.22, "Fixed score ensemble\\n0.5 baseline + 0.5 AE", PALETTE["green"]),
        ]:
            draw_box(ax, *box)
        arrow(ax, (0.20, 0.72), (0.27, 0.81))
        arrow(ax, (0.20, 0.68), (0.27, 0.49))
        arrow(ax, (0.47, 0.81), (0.53, 0.81))
        arrow(ax, (0.47, 0.49), (0.53, 0.49))
        arrow(ax, (0.70, 0.81), (0.76, 0.69))
        arrow(ax, (0.70, 0.49), (0.76, 0.63))
        ax.text(
            0.5,
            0.12,
            "Klaim utama: AE memberi sinyal skor komplementer, bukan menggantikan LightGBM tabular.",
            ha="center",
            fontsize=10,
            color="#334155",
        )
        ax.set_title("Pipeline Kandidat Final: LGBM + AE-LGBM Score Ensemble", pad=14)
        export_fig(fig, "gambar_3_2_pipeline_score_ensemble.png")
        plt.show()
        """
    ),
    md(
        """
        ## Visual Bab 4 - Data, Split, dan Imbalance

        Bagian ini menampilkan alasan metodologis untuk split temporal dan PR-AUC.
        """
    ),
    code(
        """
        if not split_summary.empty:
            df = split_summary.copy()
            df["legit_count"] = df["rows"] - df["fraud_count"]
            fig, ax1 = plt.subplots(figsize=(9, 4.8))
            x = np.arange(len(df))
            ax1.bar(x - 0.18, df["rows"], width=0.36, color=PALETTE["blue"], label="Rows")
            ax1.bar(x + 0.18, df["fraud_count"], width=0.36, color=PALETTE["orange"], label="Fraud rows")
            ax1.set_xticks(x, df["split"].str.title())
            ax1.set_ylabel("Jumlah baris")
            ax1.ticklabel_format(axis="y", style="plain")
            ax2 = ax1.twinx()
            ax2.plot(x, df["fraud_rate"], color=PALETTE["green"], marker="o", linewidth=2.2, label="Fraud rate")
            ax2.set_ylabel("Fraud rate")
            ax2.set_ylim(0, max(df["fraud_rate"]) * 1.35)
            ax2.set_yticklabels([f"{v:.1%}" for v in ax2.get_yticks()])
            for i, rate in enumerate(df["fraud_rate"]):
                ax2.annotate(f"{rate:.2%}", (i, rate), ha="center", va="bottom", xytext=(0, 6), textcoords="offset points", fontsize=9)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper right")
            ax1.set_title("Komposisi Split Temporal dan Fraud Rate")
            export_fig(fig, "gambar_4_1_split_temporal_fraud_rate.png")
            plt.show()

            plot_df = df.melt(id_vars="split", value_vars=["legit_count", "fraud_count"], var_name="class", value_name="count")
            plot_df["class"] = plot_df["class"].replace({"legit_count": "Legitimate", "fraud_count": "Fraud"})
            fig, ax = plt.subplots(figsize=(8, 4.8))
            sns.barplot(data=plot_df, x="split", y="count", hue="class", ax=ax, palette={"Legitimate": PALETTE["blue"], "Fraud": PALETTE["orange"]})
            ax.set_yscale("log")
            ax.set_xlabel("Split")
            ax.set_ylabel("Jumlah baris (log scale)")
            ax.set_title("Ketidakseimbangan Kelas per Split")
            ax.legend(title="Kelas")
            export_fig(fig, "gambar_4_2_class_imbalance_log.png")
            plt.show()

        raw_transaction = REPO_ROOT / "data" / "raw" / "train_transaction.csv"
        raw_train_small = pd.DataFrame()
        if raw_transaction.exists():
            usecols = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]
            raw_train_small = pd.read_csv(raw_transaction, usecols=usecols)
            raw_train_small["TransactionAmt_log1p"] = np.log1p(raw_train_small["TransactionAmt"])
            raw_train_small["time_bin"] = pd.qcut(raw_train_small["TransactionDT"], q=24, duplicates="drop")
            print("raw train subset:", raw_train_small.shape)
        else:
            print("[skip] data/raw/train_transaction.csv tidak tersedia")

        if not raw_train_small.empty:
            sample_df = raw_train_small.sample(n=min(120_000, len(raw_train_small)), random_state=42)
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            sns.histplot(data=sample_df, x="TransactionAmt_log1p", hue="isFraud", bins=60, stat="density", common_norm=False, element="step", ax=ax, palette={0: PALETTE["blue"], 1: PALETTE["orange"]})
            ax.set_xlabel("log1p(TransactionAmt)")
            ax.set_ylabel("Density")
            ax.set_title("Distribusi TransactionAmt Menurut Kelas Fraud")
            export_fig(fig, "gambar_4_3_transaction_amount_by_class.png")
            plt.show()

            trend = raw_train_small.groupby("time_bin", observed=True).agg(fraud_rate=("isFraud", "mean"), rows=("isFraud", "size")).reset_index()
            trend["bin_id"] = np.arange(1, len(trend) + 1)
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.plot(trend["bin_id"], trend["fraud_rate"], color=PALETTE["green"], marker="o", linewidth=2)
            ax.set_xlabel("Urutan waktu TransactionDT (24 bin)")
            ax.set_ylabel("Fraud rate")
            ax.set_title("Fraud Rate Sepanjang Waktu pada Train Raw")
            ax.set_yticklabels([f"{v:.1%}" for v in ax.get_yticks()])
            export_fig(fig, "gambar_4_4_temporal_fraud_rate_train.png")
            plt.show()
        """
    ),
    code(
        """
        if not missingness_group.empty:
            df = missingness_group.copy()
            rate_cols = [c for c in df.columns if "missing" in c.lower() and ("rate" in c.lower() or "mean" in c.lower())]
            group_col = "feature_group" if "feature_group" in df.columns else df.columns[0]
            split_col = "split" if "split" in df.columns else None
            if rate_cols and split_col:
                fig, ax = plt.subplots(figsize=(10, 5.2))
                sns.barplot(data=df, x=group_col, y=rate_cols[0], hue=split_col, ax=ax, palette=[PALETTE["blue"], PALETTE["teal"], PALETTE["orange"]])
                ax.set_xlabel("Feature group")
                ax.set_ylabel("Missing rate")
                ax.set_title("Missingness by Feature Group and Split")
                ax.tick_params(axis="x", rotation=35)
                ax.set_yticklabels([f"{v:.0%}" for v in ax.get_yticks()])
                export_fig(fig, "gambar_4_5_missingness_group_drift.png")
                plt.show()

        if not cat_unknown.empty:
            df = cat_unknown.copy()
            rate_cols = [c for c in df.columns if "unknown" in c.lower() and "rate" in c.lower()]
            feature_col = "feature" if "feature" in df.columns else df.columns[0]
            if rate_cols:
                top = df.sort_values(rate_cols[0], ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(8.5, 5.5))
                sns.barplot(data=top, y=feature_col, x=rate_cols[0], ax=ax, color=PALETTE["orange"])
                ax.set_xlabel("Unknown/unseen rate")
                ax.set_ylabel("Categorical feature")
                ax.set_title("Top Kategori dengan Unknown Rate Tertinggi")
                ax.set_xticklabels([f"{v:.0%}" for v in ax.get_xticks()])
                export_fig(fig, "gambar_4_6_categorical_unknown_rates.png")
                plt.show()

        if not numeric_shift.empty:
            if {"feature", "abs_median_shift_over_train_iqr"}.issubset(numeric_shift.columns):
                top = numeric_shift.sort_values("abs_median_shift_over_train_iqr", ascending=False).head(18)
                fig, ax = plt.subplots(figsize=(8.5, 6))
                sns.barplot(data=top, y="feature", x="abs_median_shift_over_train_iqr", ax=ax, color=PALETTE["purple"])
                ax.set_xlabel("Absolute median shift / train IQR")
                ax.set_ylabel("Feature")
                ax.set_title("Numeric Distribution Shift Tertinggi (Train vs Test)")
                export_fig(fig, "gambar_4_7_numeric_distribution_shift.png")
                plt.show()
        """
    ),
    md(
        """
        ## Visual Bab 4 - Hasil Eksperimen Utama

        Bagian ini menampilkan progres eksperimen secara berlapis: P01-P04 historis,
        AE-05 sebagai transisi, preprocessing ablation, lalu kandidat final.
        """
    ),
    code(
        """
        if not initial_cmp.empty:
            df = initial_cmp.copy()
            df["label"] = df["legacy_id"] + " / " + df["canonical_id"]
            fig, ax = plt.subplots(figsize=(8, 4.8))
            colors = [PALETTE["green"] if x == "P02" else PALETTE["blue"] for x in df["legacy_id"]]
            sns.barplot(data=df, x="label", y="test_average_precision", ax=ax, palette=colors)
            ax.set_xlabel("Model")
            ax.set_ylabel("Test Average Precision / PR-AUC")
            ax.set_title("Blok Proposal Historis P01-P04")
            ax.set_ylim(0.45, max(df["test_average_precision"]) + 0.035)
            add_bar_labels(ax)
            export_fig(fig, "gambar_4_8_p01_p04_test_ap.png")
            plt.show()

            metric_cols = ["validation_average_precision", "test_average_precision", "test_roc_auc", "test_f1", "test_mcc"]
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            sns.heatmap(df.set_index("legacy_id")[metric_cols], annot=True, fmt=".4f", cmap="YlGnBu", cbar=False, ax=ax)
            ax.set_title("Heatmap Metrik P01-P04")
            ax.set_xlabel("Metric")
            ax.set_ylabel("Model")
            export_fig(fig, "gambar_4_9_p01_p04_metric_heatmap.png")
            plt.show()

        if not extended_cmp.empty:
            df = extended_cmp.copy()
            id_col = "legacy_id" if "legacy_id" in df.columns else "canonical_id"
            ap_col = "test_average_precision" if "test_average_precision" in df.columns else "test_ap"
            df["label"] = df[id_col].fillna(df.get("canonical_id", "")).astype(str)
            fig, ax = plt.subplots(figsize=(9, 4.8))
            colors = [PALETTE["green"] if "AE-05" in str(x) else PALETTE["gray"] for x in df["label"]]
            sns.barplot(data=df, x="label", y=ap_col, ax=ax, palette=colors)
            ax.set_xlabel("Experiment")
            ax.set_ylabel("Test Average Precision / PR-AUC")
            ax.set_title("Transisi AE-05 terhadap Blok P01-P04")
            ax.set_ylim(0.46, max(df[ap_col]) + 0.035)
            add_bar_labels(ax)
            export_fig(fig, "gambar_4_10_ae05_transition.png")
            plt.show()

        if not prep_cmp.empty:
            df = prep_cmp.copy()
            model_col = "model" if "model" in df.columns else df.columns[0]
            ap_col = "test_ap" if "test_ap" in df.columns else "test_average_precision"
            label_map = {
                "baseline_frequency_missingness_time_amount_fixed_p02": "Best baseline: freq + missing + time/amount",
                "baseline_recon_frequency_missingness_time_amount_fixed_p02": "Baseline + AE recon error",
                "baseline_frequency_missingness_fixed_p02": "Baseline: freq + missing",
                "baseline_frequency_fixed_p02": "Baseline: frequency",
                "baseline_enhanced_fixed_p02": "Enhanced baseline",
                "ae05_enhanced_fixed_ae05": "AE-05 + enhanced preprocessing",
                "ae05_frequency_missingness_time_amount_fixed_ae05": "AE-05 + freq/missing/time/amount",
                "AE05_hybrid_recon": "AE-05 hybrid + recon error",
                "P02_baseline_tuned": "P02 tuned baseline",
            }
            sub = df.copy()
            sub["model_label"] = sub[model_col].map(label_map).fillna(sub[model_col].astype(str))
            sub = sub.sort_values(ap_col, ascending=True)
            fig, ax = plt.subplots(figsize=(10, 6.2))
            colors = [PALETTE["green"] if "Best baseline" in label else PALETTE["teal"] for label in sub["model_label"]]
            sns.barplot(data=sub, y="model_label", x=ap_col, ax=ax, palette=colors)
            ax.set_xlabel("Test Average Precision / PR-AUC")
            ax.set_ylabel("")
            ax.set_title("Ablasi Preprocessing dan AE terhadap Test AP")
            ax.set_xlim(max(0, sub[ap_col].min() - 0.01), sub[ap_col].max() + 0.012)
            add_bar_labels(ax, orient="h")
            export_fig(fig, "gambar_4_11_preprocessing_ablation_progression.png")
            plt.show()

        if not final_cmp.empty:
            df = final_cmp.copy()
            df["label"] = df["model"].map(clean_label)
            metric_map = {"test_ap": "Test AP", "test_roc_auc": "ROC-AUC", "test_f1": "F1", "test_mcc": "MCC"}
            plot_df = df.melt(id_vars=["model", "label"], value_vars=list(metric_map.keys()), var_name="metric", value_name="value")
            plot_df["metric"] = plot_df["metric"].map(metric_map)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
            for ax, metric in zip(axes.ravel(), metric_map.values()):
                sub = plot_df[plot_df["metric"] == metric]
                colors = [PALETTE["green"] if "Fixed 0.50" in m else PALETTE["blue"] for m in sub["model"]]
                sns.barplot(data=sub, y="label", x="value", ax=ax, palette=colors)
                ax.set_title(metric)
                ax.set_xlabel("Score")
                ax.set_ylabel("")
                ax.set_xlim(max(0, sub["value"].min() - 0.04), min(1.0, sub["value"].max() + 0.04))
                add_bar_labels(ax, orient="h")
            fig.suptitle("Perbandingan Kandidat Final", y=1.02, fontsize=15)
            fig.tight_layout()
            export_fig(fig, "gambar_4_12_final_candidate_metrics.png")
            plt.show()
        """
    ),
    md(
        """
        ## Visual Bab 4 - Kurva Ranking, Threshold, dan Confusion Matrix

        Kurva PR dan ROC dihitung dari score test final. Untuk fraud imbalanced,
        interpretasi utama tetap pada PR curve/AP.
        """
    ),
    code(
        """
        scores_test = read_csv(PATHS["final_scores_test"])
        if not scores_test.empty:
            y_true = scores_test["isFraud"].astype(int).to_numpy()
            score_cols = {
                "Best preprocessing baseline": "baseline_score",
                "AE-LGBM LD32 component": "ae_score",
                "Fixed 0.50 score ensemble": "ensemble_score",
            }
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            for name, col in score_cols.items():
                precision, recall, _ = precision_recall_curve(y_true, scores_test[col])
                ap = average_precision_score(y_true, scores_test[col])
                ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.4f})")
            prevalence = y_true.mean()
            ax.axhline(prevalence, color=PALETTE["gray"], linestyle="--", linewidth=1, label=f"Fraud rate={prevalence:.3f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve pada Test Set")
            ax.legend(loc="best")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            export_fig(fig, "gambar_4_13_pr_curve_final_components.png")
            plt.show()

            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            for name, col in score_cols.items():
                fpr, tpr, _ = roc_curve(y_true, scores_test[col])
                auc = roc_auc_score(y_true, scores_test[col])
                ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.4f})")
            ax.plot([0, 1], [0, 1], color=PALETTE["gray"], linestyle="--", linewidth=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve pada Test Set")
            ax.legend(loc="lower right")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            export_fig(fig, "gambar_4_14_roc_curve_final_components.png")
            plt.show()

        final_threshold = read_csv(PATHS["final_threshold"])
        baseline_threshold = read_csv(PATHS["baseline_threshold"])
        if not final_threshold.empty:
            fig, ax = plt.subplots(figsize=(9, 5.2))
            for metric, color in [("precision", PALETTE["blue"]), ("recall", PALETTE["orange"]), ("f1", PALETTE["green"]), ("mcc", PALETTE["purple"])]:
                ax.plot(final_threshold["threshold"], final_threshold[metric], marker="o", markersize=3, linewidth=1.6, color=color, label=metric.upper())
            if "selected" in final_threshold.columns and final_threshold["selected"].any():
                selected = final_threshold.loc[final_threshold["selected"].astype(bool), "threshold"].iloc[0]
                ax.axvline(selected, color=PALETTE["red"], linestyle="--", linewidth=1.5, label=f"selected={selected:.2f}")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Metric value")
            ax.set_title("Trade-off Threshold Kandidat Final")
            ax.legend(ncol=2)
            export_fig(fig, "gambar_4_15_threshold_tradeoff_final.png")
            plt.show()

        if not final_threshold.empty and not baseline_threshold.empty:
            fig, ax = plt.subplots(figsize=(9, 5.2))
            ax.plot(baseline_threshold["threshold"], baseline_threshold["f1"], color=PALETTE["blue"], marker="o", markersize=3, label="Baseline F1")
            ax.plot(final_threshold["threshold"], final_threshold["f1"], color=PALETTE["green"], marker="o", markersize=3, label="Ensemble F1")
            ax.plot(baseline_threshold["threshold"], baseline_threshold["mcc"], color=PALETTE["orange"], marker="o", markersize=3, label="Baseline MCC")
            ax.plot(final_threshold["threshold"], final_threshold["mcc"], color=PALETTE["purple"], marker="o", markersize=3, label="Ensemble MCC")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Metric value")
            ax.set_title("Thresholded Metrics: Baseline vs Score Ensemble")
            ax.legend(ncol=2)
            export_fig(fig, "gambar_4_16_threshold_baseline_vs_ensemble.png")
            plt.show()

        def confusion_matrix_from_csv(df, threshold_type="selected"):
            sub = df[df["threshold_type"].astype(str).str.lower() == threshold_type]
            mat = np.zeros((2, 2), dtype=int)
            for _, row in sub.iterrows():
                mat[int(row["true_label"]), int(row["predicted_label"])] = int(row["count"])
            return mat

        final_confusion = read_csv(PATHS["final_confusion"])
        baseline_confusion = read_csv(PATHS["baseline_confusion"])
        if not final_confusion.empty:
            mats = [("Final score ensemble", confusion_matrix_from_csv(final_confusion))]
            if not baseline_confusion.empty:
                mats.insert(0, ("Best preprocessing baseline", confusion_matrix_from_csv(baseline_confusion)))
            fig, axes = plt.subplots(1, len(mats), figsize=(5.2 * len(mats), 4.5))
            axes = [axes] if len(mats) == 1 else axes
            for ax, (title, mat) in zip(axes, mats):
                sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
                ax.set_title(title)
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
            fig.suptitle("Confusion Matrix pada Threshold Terpilih", y=1.02, fontsize=14)
            fig.tight_layout()
            export_fig(fig, "gambar_4_17_confusion_matrix_selected.png")
            plt.show()
        """
    ),
    md(
        """
        ## Visual Bab 4 - Feature Importance dan Diagnostik AE

        Score ensemble tidak punya feature importance sendiri karena ia menggabungkan skor dua model.
        Karena itu importance ditampilkan untuk komponen pembentuknya.
        """
    ),
    code(
        """
        def plot_feature_importance(path, title, filename, top_n=18, color=PALETTE["blue"]):
            df = read_csv(path)
            if df.empty:
                return
            gain_col = "importance_gain" if "importance_gain" in df.columns else df.columns[-1]
            feature_col = "feature" if "feature" in df.columns else df.columns[0]
            top = df.sort_values(gain_col, ascending=False).head(top_n).copy()
            fig, ax = plt.subplots(figsize=(8.5, 6.2))
            sns.barplot(data=top, y=feature_col, x=gain_col, ax=ax, color=color)
            ax.set_xlabel("Gain importance")
            ax.set_ylabel("Feature")
            ax.set_title(title)
            export_fig(fig, filename)
            plt.show()


        plot_feature_importance(PATHS["p02_importance"], "Top Feature Importance P02 Tuned Baseline", "gambar_4_18_feature_importance_p02.png", color=PALETTE["blue"])
        plot_feature_importance(PATHS["best_baseline_importance"], "Top Feature Importance Best Preprocessing Baseline", "gambar_4_19_feature_importance_best_baseline.png", color=PALETTE["teal"])
        plot_feature_importance(PATHS["ae_component_importance"], "Top Feature Importance AE-LGBM LD32 Component", "gambar_4_20_feature_importance_ae_component.png", color=PALETTE["orange"])

        ae_drift = read_csv(PATHS["ae_drift"])
        if not ae_drift.empty:
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            sns.barplot(data=ae_drift, x="split", y="mean_mse", hue="latent_dim", ax=ax, palette=[PALETTE["orange"], PALETTE["blue"]])
            ax.set_xlabel("Split")
            ax.set_ylabel("Mean masked reconstruction MSE")
            ax.set_title("AE Reconstruction Drift antar Split")
            ax.legend(title="Latent dim")
            export_fig(fig, "gambar_4_21_ae_reconstruction_drift.png")
            plt.show()

        ae_error_class = read_csv(PATHS["ae_error_by_class"])
        if not ae_error_class.empty:
            df = ae_error_class[ae_error_class["class"].isin(["non_fraud", "fraud"])].copy()
            fig, ax = plt.subplots(figsize=(9, 5.2))
            sns.barplot(data=df, x="split", y="mean_mse", hue="class", ax=ax, palette={"non_fraud": PALETTE["blue"], "fraud": PALETTE["orange"]})
            ax.set_xlabel("Split")
            ax.set_ylabel("Mean reconstruction MSE")
            ax.set_title("Reconstruction Error by Fraud Class")
            ax.legend(title="Class")
            export_fig(fig, "gambar_4_22_ae_reconstruction_error_by_class.png")
            plt.show()

        if not top_v_sweep.empty:
            df = top_v_sweep.copy()
            numeric = pd.to_numeric(df["top_k"], errors="coerce")
            sweep = df[numeric.notna()].copy()
            sweep["top_k_num"] = numeric[numeric.notna()].astype(int)
            if not sweep.empty:
                fig, ax = plt.subplots(figsize=(8.5, 5.2))
                ax.plot(sweep["top_k_num"], sweep["test_average_precision"], marker="o", linewidth=2, color=PALETTE["teal"], label="Test AP")
                ax.plot(sweep["top_k_num"], sweep["validation_average_precision"], marker="s", linewidth=2, color=PALETTE["gray"], label="Validation AP")
                refs = df[df["top_k"].isin(["P02", "P03"])]
                for _, row in refs.iterrows():
                    ax.axhline(row["test_average_precision"], linestyle="--", linewidth=1.2, color=PALETTE["green"] if row["top_k"] == "P02" else PALETTE["orange"], label=f"{row['top_k']} test AP")
                ax.set_xlabel("Top-K raw V* features retained")
                ax.set_ylabel("Average Precision / PR-AUC")
                ax.set_title("Ablasi Retensi Top-K V* pada Hybrid AE")
                ax.legend()
                export_fig(fig, "gambar_4_23_top_v_retention_sweep.png")
                plt.show()
        """
    ),
    md(
        """
        ## Visual Bab 4 - Signifikansi, Robustness, dan Complementarity

        Bagian ini menjaga klaim akhir: peningkatan AP kecil, tetapi stabil pada bootstrap
        dan tidak bergantung pada satu bobot ensemble yang rapuh.
        """
    ),
    code(
        """
        if not bootstrap_delta.empty:
            deltas = bootstrap_delta.iloc[:, 0].dropna()
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            sns.histplot(deltas, bins=40, kde=True, ax=ax, color=PALETTE["green"])
            if bootstrap_summary:
                obs = bootstrap_summary.get("observed_delta_ap", np.nan)
                lo = bootstrap_summary.get("ci_2_5", np.nan)
                hi = bootstrap_summary.get("ci_97_5", np.nan)
                ax.axvline(obs, color=PALETTE["red"], linewidth=2, label=f"Observed delta={obs:.4f}")
                ax.axvline(lo, color=PALETTE["gray"], linestyle="--", linewidth=1.5, label=f"95% CI [{lo:.4f}, {hi:.4f}]")
                ax.axvline(hi, color=PALETTE["gray"], linestyle="--", linewidth=1.5)
            ax.axvline(0, color="black", linestyle=":", linewidth=1.2, label="No improvement")
            ax.set_xlabel("Bootstrap delta AP (ensemble - baseline)")
            ax.set_ylabel("Count")
            ax.set_title("Paired Bootstrap Delta PR-AUC Kandidat Final")
            ax.legend()
            export_fig(fig, "gambar_4_24_bootstrap_delta_final.png")
            plt.show()

        alpha_rows = []
        alpha_paths = {
            0.25: ARTIFACT_ROOT / "preprocessing_ablation" / "score_ensemble_baseline_all_masked_ld32_fixed_025_canonical",
            0.50: ARTIFACT_ROOT / "preprocessing_ablation" / "score_ensemble_baseline_all_masked_ld32_fixed_050_canonical",
            0.75: ARTIFACT_ROOT / "preprocessing_ablation" / "score_ensemble_baseline_all_masked_ld32_fixed_075_canonical",
            "tuned_10": ARTIFACT_ROOT / "preprocessing_ablation" / "score_ensemble_baseline_all_masked_ld32_alpha_tuned_10trials",
        }
        for alpha, folder in alpha_paths.items():
            metrics = read_json(folder / "metrics_test_selected_threshold.json")
            summary = read_json(folder / "paired_bootstrap_summary.json")
            run_config = read_json(folder / "run_config.json")
            if metrics:
                alpha_rows.append({
                    "alpha_label": str(alpha),
                    "selected_alpha": run_config.get("selected_alpha", alpha) if isinstance(run_config, dict) else alpha,
                    "test_ap": metrics.get("average_precision"),
                    "test_roc_auc": metrics.get("roc_auc"),
                    "test_f1": metrics.get("f1"),
                    "test_mcc": metrics.get("mcc"),
                    "delta_ap": summary.get("observed_delta_ap"),
                    "ci_2_5": summary.get("ci_2_5"),
                    "ci_97_5": summary.get("ci_97_5"),
                })
        alpha_df = pd.DataFrame(alpha_rows)
        if not alpha_df.empty:
            export_table(alpha_df, "tabel_4_7_alpha_robustness.csv")
            order = ["0.25", "0.5", "0.75", "tuned_10"]
            alpha_df["alpha_label"] = pd.Categorical(alpha_df["alpha_label"], categories=order, ordered=True)
            plot_df = alpha_df.sort_values("alpha_label")
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            sns.pointplot(data=plot_df, x="alpha_label", y="test_ap", ax=ax, color=PALETTE["green"], markers="o", linestyles="-")
            ax.set_xlabel("AE score weight")
            ax.set_ylabel("Test Average Precision / PR-AUC")
            ax.set_title("Robustness Bobot Score Ensemble")
            for i, row in enumerate(plot_df.itertuples(index=False)):
                ax.annotate(f"{row.test_ap:.4f}", (i, row.test_ap), ha="center", va="bottom", xytext=(0, 6), textcoords="offset points", fontsize=9)
            export_fig(fig, "gambar_4_25_alpha_robustness.png")
            plt.show()
        """
    ),
    code(
        """
        if not scores_test.empty:
            sample = scores_test.sample(n=min(80_000, len(scores_test)), random_state=7)
            fig, ax = plt.subplots(figsize=(7.2, 6))
            hb = ax.hexbin(sample["baseline_score"], sample["ae_score"], gridsize=45, bins="log", cmap="YlGnBu", mincnt=1)
            ax.set_xlabel("Baseline score")
            ax.set_ylabel("AE-LGBM score")
            ax.set_title("Complementarity: Baseline Score vs AE Score")
            fig.colorbar(hb, ax=ax, label="log10(count)")
            if score_summary:
                corr = score_summary.get("score_correlations", {})
                ax.text(
                    0.03,
                    0.97,
                    f"Pearson={corr.get('pearson_baseline_ae', np.nan):.4f}\\nSpearman={corr.get('spearman_baseline_ae', np.nan):.4f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )
            export_fig(fig, "gambar_4_26_score_complementarity_hexbin.png")
            plt.show()

        if not topk_capture.empty:
            df = topk_capture.copy()
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            ax.plot(df["top_k"], df["baseline_frauds"], marker="o", linewidth=2, color=PALETTE["blue"], label="Baseline")
            ax.plot(df["top_k"], df["ae_frauds"], marker="s", linewidth=2, color=PALETTE["orange"], label="AE-LGBM")
            ax.plot(df["top_k"], df["ensemble_frauds"], marker="^", linewidth=2.2, color=PALETTE["green"], label="Ensemble")
            ax.set_xscale("log")
            ax.set_xlabel("Top-K highest-risk rows")
            ax.set_ylabel("Fraud captured")
            ax.set_title("Fraud Capture pada Top-K Ranking")
            ax.legend()
            export_fig(fig, "gambar_4_27_topk_fraud_capture.png")
            plt.show()

        if score_summary and "fraud_rank_movement" in score_summary:
            movement = score_summary["fraud_rank_movement"]
            move_df = pd.DataFrame([
                {"movement": "Improved rank", "count": movement.get("frauds_improved_rank", np.nan)},
                {"movement": "Worsened rank", "count": movement.get("frauds_worsened_rank", np.nan)},
                {"movement": "Unchanged rank", "count": movement.get("frauds_unchanged_rank", np.nan)},
            ])
            fig, ax = plt.subplots(figsize=(7, 4.8))
            sns.barplot(data=move_df, x="movement", y="count", ax=ax, palette=[PALETTE["green"], PALETTE["orange"], PALETTE["gray"]])
            ax.set_xlabel("Rank movement for fraud rows")
            ax.set_ylabel("Fraud row count")
            ax.set_title("Pergerakan Ranking Fraud: Ensemble vs Baseline")
            add_bar_labels(ax, digits=0)
            export_fig(fig, "gambar_4_28_fraud_rank_movement.png")
            plt.show()
        """
    ),
    md(
        """
        ## Tabel Accountability Literatur dan Manifest Output

        Tabel ini memastikan setiap keputusan metodologi punya anchor literatur yang jelas.
        Manifest di akhir membantu memilih aset mana yang masuk body skripsi dan mana yang cukup di lampiran.
        """
    ),
    code(
        """
        literature_rows = [
            {
                "decision": "Chronological split using TransactionDT",
                "literature_anchor": "Dal Pozzolo et al. (2018); Lucas et al. (2019)",
                "thesis_use": "Bab 3 - evaluasi realistis dan temporal drift",
                "implementation_anchor": "src/splitting.py; split_summary.csv",
            },
            {
                "decision": "Average Precision / PR-AUC as primary metric",
                "literature_anchor": "Saito & Rehmsmeier (2015); Davis & Goadrich (2006)",
                "thesis_use": "Bab 2/3 - imbalanced fraud evaluation",
                "implementation_anchor": "src/evaluation.py",
            },
            {
                "decision": "LightGBM supervised tabular baseline",
                "literature_anchor": "Ke et al. (2017)",
                "thesis_use": "Bab 2/3 - classifier utama",
                "implementation_anchor": "src/train_baseline_lgbm.py; src/train_enhanced_preprocessing_lgbm.py",
            },
            {
                "decision": "Optuna hyperparameter search",
                "literature_anchor": "Akiba et al. (2019)",
                "thesis_use": "Bab 3 - Bayesian/TPE tuning",
                "implementation_anchor": "src/tune_lgbm_optuna.py",
            },
            {
                "decision": "No pre-split SMOTE/ADASYN",
                "literature_anchor": "Kabane & Ouali (2024)",
                "thesis_use": "Bab 3 - leakage prevention",
                "implementation_anchor": "Pipeline split before modeling; no sampler in active path",
            },
            {
                "decision": "Frequency/missingness/time/amount features",
                "literature_anchor": "Moradi et al. (2025); Alharbi et al. (2026)",
                "thesis_use": "Bab 3/4 - preprocessing-strengthened baseline",
                "implementation_anchor": "src/enhanced_preprocessing.py",
            },
            {
                "decision": "AE as complementary representation signal",
                "literature_anchor": "Jiang et al. (2023); Ding et al. (2024); Du et al. (2023)",
                "thesis_use": "Bab 2/3/5 - AE-LGBM framing",
                "implementation_anchor": "src/train_autoencoder_normal_masked.py; src/train_score_ensemble.py",
            },
        ]
        lit_df = pd.DataFrame(literature_rows)
        export_table(lit_df, "tabel_2_1_literature_accountability.csv")
        display(lit_df)

        manifest_rows = []
        for kind, folder in [("figure", FIG_DIR), ("table", TABLE_DIR)]:
            for path in sorted(folder.glob("*")):
                manifest_rows.append({
                    "kind": kind,
                    "filename": path.name,
                    "relative_path": str(path.relative_to(WORKSPACE_ROOT)).replace("\\\\", "/"),
                    "size_bytes": path.stat().st_size,
                })
        manifest = pd.DataFrame(manifest_rows)
        manifest_path = LAMPIRAN_ROOT / "asset_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print("Manifest:", manifest_path.relative_to(WORKSPACE_ROOT))
        display(manifest)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Cells: {len(cells)}")


if __name__ == "__main__":
    main()
