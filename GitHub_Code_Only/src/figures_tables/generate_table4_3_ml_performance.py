#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Table 4-3: grouped cross-validation classification performance.

The table uses Step3's fold-level summary metrics. PR-AUC and Brier score are
retained as supplementary probability-based metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "rf": "RF",
    "xgb": "XGB",
}

METRIC_COLUMNS = [
    ("f1_macro", "Macro-F1"),
    ("recall_pod", "High-risk POD"),
    ("far", "FAR"),
    ("csi", "CSI"),
    ("pr_auc_high", "PR-AUC"),
    ("brier_high", "Brier score"),
]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def fmt3(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.3f}"


def build_table(step3_outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = pd.read_csv(step3_outdir / "summary_compare_models.csv")
    rows = []
    detail_rows = []
    for model_key, model_label in MODEL_LABELS.items():
        sub = src[src["model"].eq(model_key)].set_index("metric")
        row = {"Model": model_label}
        for metric, label in METRIC_COLUMNS:
            if metric not in sub.index:
                row[label] = ""
                continue
            metric_row = sub.loc[metric]
            row[label] = fmt3(float(metric_row["mean"]))
            detail_rows.append(
                {
                    "Model": model_label,
                    "Metric": label,
                    "source_metric": metric,
                    "mean": float(metric_row["mean"]),
                    "ci95_lo": float(metric_row["ci95_lo"]),
                    "ci95_hi": float(metric_row["ci95_hi"]),
                    "n_folds": int(metric_row["n_folds"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def to_markdown(table: pd.DataFrame) -> str:
    lines = [
        "# Table 4-3. Grouped cross-validation performance of ML risk classifiers",
        "",
        table.to_markdown(index=False, disable_numparse=True),
        "",
        "Note. Metrics are fold-level means from GroupKFold cross-validation. "
        "High-risk POD, FAR, and CSI are emphasized for operational risk detection. "
        "PR-AUC and Brier score are reported as supplementary probability-based metrics.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(outdir: Path, table: pd.DataFrame, detail: pd.DataFrame, args: argparse.Namespace) -> None:
    table.to_csv(outdir / "table4_3_ml_performance.csv", index=False)
    detail.to_csv(outdir / "table4_3_ml_performance_detail.csv", index=False)
    (outdir / "table4_3_ml_performance.md").write_text(to_markdown(table), encoding="utf-8")
    try:
        with pd.ExcelWriter(outdir / "table4_3_ml_performance.xlsx") as writer:
            table.to_excel(writer, sheet_name="Table 4-3", index=False)
            detail.to_excel(writer, sheet_name="Detail with CI", index=False)
    except Exception as exc:
        (outdir / "table4_3_excel_write_warning.txt").write_text(str(exc), encoding="utf-8")

    meta = {
        "table": "Table 4-3",
        "source": str(Path(args.step3_outdir) / "summary_compare_models.csv"),
        "metric_scope": "Step3 grouped cross-validation fold-level means.",
        "primary_operational_metrics": ["Macro-F1", "High-risk POD", "FAR", "CSI"],
        "supplementary_probability_metrics": ["PR-AUC", "Brier score"],
        "caveat": "Metric n_folds can differ when a fold lacks positive high-risk cases or predicted high-risk cases, making probability metrics or FAR undefined.",
    }
    (outdir / "table4_3_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Table 4-3 from Step3 outputs.")
    p.add_argument(
        "--step3-outdir",
        default="outputs/step3",
        help="Directory containing Step3 ML outputs.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/tables/table4_3",
        help="Output directory.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    outdir = ensure_dir(args.outdir)
    table, detail = build_table(Path(args.step3_outdir))
    write_outputs(outdir, table, detail, args)
    print(f"[Table4-3] Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
