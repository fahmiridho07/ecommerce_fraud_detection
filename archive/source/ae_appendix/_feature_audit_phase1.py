"""Phase 1 feature eligibility audit for selected-numerical AE experiment."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ID_COL, TARGET_COL, TIME_COL
from data_loader import load_labeled_train_data
from preprocessing import get_categorical_columns, get_v_feature_columns
from utils import ensure_dir, save_json

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "selected_numerical_ae_feature_audit"

EXCLUDED_IDENTIFIER = {ID_COL}
EXCLUDED_TARGET = {TARGET_COL}
EXCLUDED_RAW_TIME = {TIME_COL}

EXCLUDED_CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
]

EXCLUDED_NUMERIC_CODED_CATEGORICAL = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "id_05",
    "id_07",
    "id_08",
    "id_09",
    "id_10",
    "id_11",
    "id_32",
]

AE_ELIGIBLE_GROUPS = {
    "TransactionAmt": ["TransactionAmt"],
    "C1-C14": [f"C{i}" for i in range(1, 15)],
    "D1-D15": [f"D{i}" for i in range(1, 16)],
    "V1-V339": [f"V{i}" for i in range(1, 340)],
    "dist1-dist2": ["dist1", "dist2"],
    "id_numerical_continuous": [
        "id_01",
        "id_02",
        "id_03",
        "id_04",
        "id_06",
        "id_13",
        "id_14",
        "id_17",
        "id_18",
        "id_19",
        "id_20",
        "id_21",
        "id_22",
        "id_24",
        "id_25",
        "id_26",
    ],
}

REVIEW_REQUIRED = {
    "id_05": "Binary 0/1 anomaly flag; treated as numeric-coded categorical and excluded from AE.",
    "id_07": "Sparse discrete offset code (>99% missing, 84 levels); excluded as numeric-coded categorical.",
    "id_08": "Sparse discrete offset code (>99% missing, 94 levels); excluded as numeric-coded categorical.",
    "id_09": "Low-cardinality timezone code; excluded as numeric-coded categorical.",
    "id_10": "OS version code with modest cardinality; excluded as numeric-coded categorical.",
    "id_11": "Browser version code; excluded as numeric-coded categorical.",
    "id_32": "Device code with modest cardinality; excluded as numeric-coded categorical.",
    "card1": "Issuer identifier code, not a continuous quantity; excluded.",
    "card2": "Issuer identifier code, not a continuous quantity; excluded.",
    "card3": "Issuer identifier code, not a continuous quantity; excluded.",
    "card5": "Issuer identifier code, not a continuous quantity; excluded.",
    "addr1": "Address region code, not a continuous quantity; excluded.",
    "addr2": "Address region code, not a continuous quantity; excluded.",
}


def column_stats(df: pd.DataFrame, column: str) -> dict[str, object]:
    series = df[column]
    non_null = series.dropna()
    return {
        "dtype": str(series.dtype),
        "missing_rate": float(series.isna().mean()),
        "cardinality": int(series.nunique(dropna=True)),
        "min": float(non_null.min()) if len(non_null) else None,
        "max": float(non_null.max()) if len(non_null) else None,
    }


def build_selected_feature_list(df: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for group_columns in AE_ELIGIBLE_GROUPS.values():
        for column in group_columns:
            if column not in df.columns:
                raise KeyError(f"Expected column missing from merged training data: {column}")
            selected.append(column)
    return selected


def main() -> dict[str, object]:
    df = load_labeled_train_data()
    predictors = [column for column in df.columns if column not in {ID_COL, TARGET_COL}]
    v_columns = get_v_feature_columns(df)
    dtype_categorical = get_categorical_columns(df.drop(columns=[TARGET_COL, ID_COL]))

    selected_features = build_selected_feature_list(df)
    additional_numerical = [column for column in selected_features if column not in v_columns]

    group_summaries = []
    for group_name, group_columns in AE_ELIGIBLE_GROUPS.items():
        stats = [column_stats(df, column) for column in group_columns]
        group_summaries.append(
            {
                "feature_group": group_name,
                "columns": group_columns,
                "column_count": len(group_columns),
                "dtype_summary": sorted({item["dtype"] for item in stats}),
                "missing_rate_range": [
                    min(item["missing_rate"] for item in stats),
                    max(item["missing_rate"] for item in stats),
                ],
                "cardinality_range": [
                    min(item["cardinality"] for item in stats),
                    max(item["cardinality"] for item in stats),
                ],
            }
        )

    payload = {
        "feature_names": selected_features,
        "feature_count": len(selected_features),
        "v_feature_names": v_columns,
        "v_feature_count": len(v_columns),
        "additional_numerical_feature_names": additional_numerical,
        "additional_numerical_feature_count": len(additional_numerical),
        "excluded_time_features": sorted(EXCLUDED_RAW_TIME),
        "excluded_identifier_features": sorted(EXCLUDED_IDENTIFIER),
        "excluded_categorical_features": EXCLUDED_CATEGORICAL,
        "excluded_numeric_coded_categorical_features": EXCLUDED_NUMERIC_CODED_CATEGORICAL,
        "review_required_features": REVIEW_REQUIRED,
        "selection_policy": (
            "Include amount, count, duration, distance, anonymous V, and identity "
            "columns that represent measurable numerical quantities. Exclude "
            "identifiers, target, raw TransactionDT, categorical strings, and "
            "numeric-coded categorical identifiers. Classification uses column "
            "semantics, dtype, cardinality, and repository baseline preprocessing; "
            "no validation or test performance was used."
        ),
        "target_not_used": True,
        "validation_not_used_for_selection": True,
        "test_not_used_for_selection": True,
        "total_raw_predictor_count": len(predictors),
        "group_summaries": group_summaries,
        "dtype_categorical_count": len(dtype_categorical),
    }

    leakage_checks = {
        "target_in_ae_input": any(column in selected_features for column in EXCLUDED_TARGET),
        "identifier_in_ae_input": any(column in selected_features for column in EXCLUDED_IDENTIFIER),
        "transactiondt_in_ae_input": TIME_COL in selected_features,
    }
    if any(leakage_checks.values()):
        raise ValueError(f"Leakage check failed: {leakage_checks}")

    output_dir = ensure_dir(OUTPUT_DIR)
    save_json(payload, output_dir / "selected_numerical_features.json")

    print("=" * 60)
    print("PHASE 1 FEATURE ELIGIBILITY AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total raw predictor count              : {len(predictors)}")
    print(f"Selected AE numerical feature count    : {len(selected_features)}")
    print(f"V-feature count                        : {len(v_columns)}")
    print(f"Additional numerical count             : {len(additional_numerical)}")
    print(f"Excluded categorical count             : {len(EXCLUDED_CATEGORICAL)}")
    print(f"Excluded numeric-coded categorical     : {len(EXCLUDED_NUMERIC_CODED_CATEGORICAL)}")
    print(f"Ambiguous features reviewed            : {len(REVIEW_REQUIRED)}")
    print()
    print("Included feature groups:")
    for summary in group_summaries:
        print(
            f"  - {summary['feature_group']}: {summary['column_count']} columns, "
            f"miss [{summary['missing_rate_range'][0]:.3f}, {summary['missing_rate_range'][1]:.3f}]"
        )
    print()
    print("Excluded ambiguous / coded columns (conservative):")
    for column, reason in REVIEW_REQUIRED.items():
        print(f"  - {column}: {reason}")
    print()
    print("Leakage checks:")
    print(f"  Target in AE input       : {leakage_checks['target_in_ae_input']}")
    print(f"  TransactionID in AE input: {leakage_checks['identifier_in_ae_input']}")
    print(f"  TransactionDT in AE input  : {leakage_checks['transactiondt_in_ae_input']}")
    print()
    print(f"Audit JSON saved to: {output_dir / 'selected_numerical_features.json'}")
    return payload


if __name__ == "__main__":
    main()