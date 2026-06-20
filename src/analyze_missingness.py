"""Analisis missingness IEEE-CIS: apakah POLA MISSING berkorelasi dengan fraud?

Memandu keputusan apakah autoencoder pada pola missingness (sinyal struktural yang
LightGBM tak lihat secara gabungan) layak dikejar. Ringan memori (statistik saja;
membaca subset baris via nrows).

Output:
- missing rate per blok fitur (V, C, D, M, id, card, addr, dist, email, ...)
- untuk tiap kolom: fraud-rate saat MISSING vs saat ADA, dan lift -> apakah
  "missing" itu sendiri prediktif terhadap fraud
- co-missingness: berapa blok yang hilang bersamaan (identity dll.)

Run:
    python src/analyze_missingness.py --nrows 120000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import TARGET_COL
from data_loader import TRAIN_TRANSACTION_FILE, load_train_identity, merge_train_data
from utils import ensure_dir, log, save_json


def block_of(col: str) -> str:
    if re.fullmatch(r"V\d+", col):
        return "V"
    if re.fullmatch(r"C\d+", col):
        return "C"
    if re.fullmatch(r"D\d+", col):
        return "D"
    if re.fullmatch(r"M\d+", col):
        return "M"
    if col.startswith("id_") or col in ("DeviceType", "DeviceInfo"):
        return "identity"
    if col.startswith("card"):
        return "card"
    if col.startswith("addr"):
        return "addr"
    if col.startswith("dist"):
        return "dist"
    if "emaildomain" in col:
        return "email"
    return "other"


def main(nrows: int, output_dir: Path) -> dict:
    output_dir = ensure_dir(output_dir)
    log(f"Reading first {nrows} transaction rows + identity.")
    tx = pd.read_csv(TRAIN_TRANSACTION_FILE, nrows=nrows)
    df = merge_train_data(tx, load_train_identity())
    del tx
    y = df[TARGET_COL].to_numpy()
    base_rate = float(y.mean())
    feature_cols = [c for c in df.columns if c not in (TARGET_COL, "TransactionID")]
    log(f"rows={len(df)} fraud_rate={base_rate:.4f} features={len(feature_cols)}")

    # ---- per-column missingness + fraud lift of "is missing" ----
    rows = []
    for c in feature_cols:
        m = df[c].isna().to_numpy()
        mr = float(m.mean())
        if mr <= 0.0 or mr >= 1.0:
            fr_missing = fr_present = lift = float("nan")
        else:
            fr_missing = float(y[m].mean())
            fr_present = float(y[~m].mean())
            lift = fr_missing / base_rate if base_rate > 0 else float("nan")
        rows.append({"col": c, "block": block_of(c), "missing_rate": mr,
                     "fraud_rate_when_missing": fr_missing, "fraud_rate_when_present": fr_present,
                     "fraud_lift_when_missing": lift})
    feat = pd.DataFrame(rows)

    # ---- block-level summary ----
    blocks = (feat.groupby("block")
              .agg(n_cols=("col", "size"), mean_missing_rate=("missing_rate", "mean"),
                   max_missing_rate=("missing_rate", "max"))
              .sort_values("mean_missing_rate", ascending=False))

    # ---- columns where MISSINGNESS most discriminates fraud ----
    disc = feat.dropna(subset=["fraud_lift_when_missing"]).copy()
    disc["abs_lift_dev"] = (disc["fraud_lift_when_missing"] - 1.0).abs()
    top_disc = disc.sort_values("abs_lift_dev", ascending=False).head(20)

    # ---- co-missingness: how many distinct missingness PATTERNS dominate ----
    # Use a compact signature over blocks: fraction missing per block per row.
    miss = df[feature_cols].isna()
    block_map = {c: block_of(c) for c in feature_cols}
    block_miss = miss.T.groupby(pd.Series(block_map)).mean().T  # rows x blocks: frac missing
    # binarize (block "absent" if >50% of its cols missing) and count unique patterns
    pattern = (block_miss > 0.5).astype(int)
    pat_str = pattern.astype(str).agg("".join, axis=1)
    top_patterns = pat_str.value_counts(normalize=True).head(10)

    summary = {
        "nrows": int(len(df)), "fraud_rate": base_rate, "n_features": len(feature_cols),
        "block_summary": blocks.reset_index().to_dict(orient="records"),
        "block_columns_order": list(pattern.columns),
        "top_block_missingness_patterns": [
            {"pattern": k, "share": float(v)} for k, v in top_patterns.items()
        ],
        "n_distinct_block_patterns": int(pat_str.nunique()),
        "top20_missingness_fraud_discriminators": top_disc[
            ["col", "block", "missing_rate", "fraud_rate_when_missing", "fraud_rate_when_present", "fraud_lift_when_missing"]
        ].to_dict(orient="records"),
    }
    save_json(summary, output_dir / "missingness_analysis.json")
    feat.to_csv(output_dir / "per_column_missingness.csv", index=False)

    print("\n=== MISSINGNESS PER BLOK (rata-rata) ===")
    print(blocks.round(3).to_string())
    print(f"\nfraud_rate global = {base_rate:.4f}")
    print(f"\n=== POLA MISSING per-blok: {summary['n_distinct_block_patterns']} pola unik; 10 teratas (share) ===")
    for p in summary["top_block_missingness_patterns"]:
        print(f"  {p['pattern']}  {p['share']:.3f}")
    print(f"  (urutan blok: {summary['block_columns_order']})")
    print("\n=== 15 kolom yang MISSING-nya paling membedakan fraud (lift vs fraud-rate global) ===")
    print(top_disc.head(15)[["col", "block", "missing_rate", "fraud_rate_when_missing", "fraud_lift_when_missing"]].round(4).to_string(index=False))
    print(f"\nSaved: {output_dir / 'missingness_analysis.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analisis missingness IEEE-CIS.")
    p.add_argument("--nrows", type=int, default=120000)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/stratified_reset/missingness_analysis"))
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    main(nrows=a.nrows, output_dir=a.output_dir)
