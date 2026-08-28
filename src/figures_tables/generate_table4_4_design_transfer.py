#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Table 4-4: Design-storm transfer metrics.

Panel A reports RF external classification performance on DESIGN_CHICAGO
events. Panel B reports return-period response diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def fmt3(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.3f}"


def build_panel_a(step3_outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(step3_outdir / "external_design_metrics_rf.csv").iloc[0]
    thresholds = pd.read_csv(step3_outdir / "external_design_threshold_metrics_rf.csv")
    safety = thresholds[thresholds["criterion"].eq("recall_floor_0.9")]
    if safety.empty:
        safety = thresholds[thresholds["criterion"].eq("min_cost_norm")]
    safety = safety.iloc[0]

    panel = pd.DataFrame(
        [
            {
                "Model": "RF",
                "Accuracy": fmt3(metrics["accuracy"]),
                "Macro-F1": fmt3(metrics["f1_macro"]),
                "High-risk POD": fmt3(metrics["recall_pod"]),
                "FAR": fmt3(metrics["far"]),
                "CSI": fmt3(metrics["csi"]),
            },
            {
                "Model": "RF, safety operating point",
                "Accuracy": "-",
                "Macro-F1": "-",
                "High-risk POD": fmt3(safety["recall_pod"]),
                "FAR": fmt3(safety["far"]),
                "CSI": fmt3(safety["csi"]),
            },
        ]
    )
    detail = pd.DataFrame(
        [
            {
                "row": "RF",
                "source_file": "external_design_metrics_rf.csv",
                "threshold": "",
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "tn": int(metrics["tn"]),
                "fn": int(metrics["fn"]),
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "recall_pod": float(metrics["recall_pod"]),
                "far": float(metrics["far"]),
                "csi": float(metrics["csi"]),
            },
            {
                "row": "RF, safety operating point",
                "source_file": "external_design_threshold_metrics_rf.csv",
                "threshold": float(safety["threshold"]),
                "tp": int(safety["tp"]),
                "fp": int(safety["fp"]),
                "tn": int(safety["tn"]),
                "fn": int(safety["fn"]),
                "accuracy": "",
                "f1_macro": "",
                "recall_pod": float(safety["recall_pod"]),
                "far": float(safety["far"]),
                "csi": float(safety["csi"]),
            },
        ]
    )
    return panel, detail


def build_panel_b(step3_outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rp = pd.read_csv(step3_outdir / "external_design_return_period_monotonicity_rf.csv")
    rp = rp.sort_values("Return_Period_yr").copy()
    panel = pd.DataFrame(
        {
            "Return period": [f"{int(v)} yr" for v in rp["Return_Period_yr"]],
            "True mean ratio": [fmt3(v) for v in rp["true_mean_ratio"]],
            "Predicted high-risk probability": [fmt3(v) for v in rp["pred_mean_high_prob"]],
            "True high-risk rate": [fmt3(v) for v in rp["true_high_rate"]],
            "Predicted high-risk rate": [fmt3(v) for v in rp["pred_high_rate"]],
        }
    )
    return panel, rp


def to_markdown(panel_a: pd.DataFrame, panel_b: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Table 4-4. Design-storm transfer metrics",
            "",
            "Panel A. External classification performance",
            "",
            panel_a.to_markdown(index=False, disable_numparse=True),
            "",
            "Panel B. Return-period response",
            "",
            panel_b.to_markdown(index=False, disable_numparse=True),
            "",
            "Note. The safety operating point uses the RF high-risk probability threshold selected from idealized-event validation to prioritize recall. "
            "The return-period response indicates that observed risk increased monotonically with return period, whereas RF-predicted high-risk probability and predicted high-risk rate did not.",
            "",
        ]
    )


def write_outputs(
    outdir: Path,
    panel_a: pd.DataFrame,
    panel_a_detail: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_b_detail: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    panel_a.to_csv(outdir / "table4_4_panel_a_external_classification.csv", index=False)
    panel_a_detail.to_csv(outdir / "table4_4_panel_a_external_classification_detail.csv", index=False)
    panel_b.to_csv(outdir / "table4_4_panel_b_return_period_response.csv", index=False)
    panel_b_detail.to_csv(outdir / "table4_4_panel_b_return_period_response_detail.csv", index=False)
    (outdir / "table4_4_design_transfer.md").write_text(to_markdown(panel_a, panel_b), encoding="utf-8")
    try:
        with pd.ExcelWriter(outdir / "table4_4_design_transfer.xlsx") as writer:
            panel_a.to_excel(writer, sheet_name="Panel A", index=False)
            panel_a_detail.to_excel(writer, sheet_name="Panel A detail", index=False)
            panel_b.to_excel(writer, sheet_name="Panel B", index=False)
            panel_b_detail.to_excel(writer, sheet_name="Panel B detail", index=False)
    except Exception as exc:
        (outdir / "table4_4_excel_write_warning.txt").write_text(str(exc), encoding="utf-8")

    meta = {
        "table": "Table 4-4",
        "panel_a_sources": [
            str(Path(args.step3_outdir) / "external_design_metrics_rf.csv"),
            str(Path(args.step3_outdir) / "external_design_threshold_metrics_rf.csv"),
        ],
        "panel_b_source": str(Path(args.step3_outdir) / "external_design_return_period_monotonicity_rf.csv"),
        "model": "RF",
    }
    (outdir / "table4_4_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Table 4-4 from Step3 RF external-validation outputs.")
    p.add_argument(
        "--step3-outdir",
        default="outputs/step3",
        help="Directory containing Step3 ML outputs.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/tables/table4_4",
        help="Output directory.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    outdir = ensure_dir(args.outdir)
    panel_a, panel_a_detail = build_panel_a(Path(args.step3_outdir))
    panel_b, panel_b_detail = build_panel_b(Path(args.step3_outdir))
    write_outputs(outdir, panel_a, panel_a_detail, panel_b, panel_b_detail, args)
    print(f"[Table4-4] Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
