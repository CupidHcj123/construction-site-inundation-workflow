#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 10: RF permutation feature importance.

The figure shows the main RF classifier only. Bars denote mean permutation
importance under Macro-F1 scoring, and error bars denote between-fold standard
deviation of fold-level permutation importance.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(__file__).resolve().parent / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib as mpl
import matplotlib.pyplot as plt


REQUESTED_FEATURES = [
    "Rainfall_mm",
    "RF_over_MaxDepth",
    "RF_over_AvgDepth",
    "Rainfall_Intensity_mm_per_h",
    "RainIntensity_over_MaxDepth",
    "Rainfall_Pattern",
    "Rainfall_Duration_s",
    "Pit_Max_Depth_m",
    "Pit_Avg_Depth_m",
    "Pit_Area_m2",
]

FEATURE_LABELS = {
    "Rainfall_mm": r"Total rainfall, $P$",
    "RF_over_MaxDepth": r"Rainfall / max. depth, $\rho_{\mathrm{max}}$",
    "RF_over_AvgDepth": r"Rainfall / avg. depth, $\rho_{\mathrm{avg}}$",
    "Rainfall_Intensity_mm_per_h": "Mean rainfall intensity",
    "RainIntensity_over_MaxDepth": "Intensity / max. depth",
    "Rainfall_Pattern": "Rainfall pattern",
    "Rainfall_Duration_s": "Duration",
    "Pit_Max_Depth_m": r"Max. pit depth, $h_{\mathrm{max}}$",
    "Pit_Avg_Depth_m": r"Avg. pit depth, $h_{\mathrm{avg}}$",
    "Pit_Area_m2": r"Pit area, $A$",
}


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
        }
    )


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_publication_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def build_plot_table(step3_outdir: Path, top_n: int) -> pd.DataFrame:
    agg = pd.read_csv(step3_outdir / "perm_importance_rf_agg.csv")
    byfold = pd.read_csv(step3_outdir / "perm_importance_rf_byfold.csv")

    agg = agg[agg["feature"].isin(REQUESTED_FEATURES)].copy()
    if agg.empty:
        raise ValueError("No requested RF feature-importance rows were found.")

    fold_n = byfold.groupby("feature")["fold"].nunique().rename("n_folds").reset_index()
    table = agg.merge(fold_n, on="feature", how="left")
    table["label"] = table["feature"].map(FEATURE_LABELS).fillna(table["feature"])
    table = table.sort_values("importance_mean", ascending=False).head(top_n).copy()
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def plot_feature_importance(table: pd.DataFrame, outdir: Path) -> None:
    plot_df = table.sort_values("importance_mean", ascending=True).copy()
    y = np.arange(len(plot_df))

    fig_h = max(3.2, 0.34 * len(plot_df) + 1.15)
    fig, ax = plt.subplots(figsize=(4.9, fig_h))

    colors = ["#2F6F8F" if r <= 5 else "#9EB8C4" for r in plot_df["rank"]]
    ax.barh(
        y,
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
        color=colors,
        edgecolor="none",
        height=0.66,
        error_kw={
            "ecolor": "#4A4A4A",
            "elinewidth": 0.85,
            "capsize": 2.2,
            "capthick": 0.75,
        },
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=7)
    ax.set_xlabel("Permutation importance (decrease in Macro-F1)", fontsize=7)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax.tick_params(axis="x", labelsize=6.5)

    lower = float((plot_df["importance_mean"] - plot_df["importance_std"]).min())
    upper = float((plot_df["importance_mean"] + plot_df["importance_std"]).max())
    span = upper - min(0.0, lower)
    ax.set_xlim(min(0.0, lower) - 0.05 * span, upper + 0.08 * span)

    for yi, (_, row) in enumerate(plot_df.iterrows()):
        x = row["importance_mean"] + row["importance_std"] + 0.012 * span
        ax.text(x, yi, f"{row['importance_mean']:.3f}", va="center", ha="left", fontsize=6.2, color="#333333")

    fig.subplots_adjust(left=0.36, right=0.97, top=0.96, bottom=0.18)
    save_publication_figure(fig, outdir / "figure10_rf_feature_importance")
    plt.close(fig)


def write_metadata(outdir: Path, args: argparse.Namespace, table: pd.DataFrame) -> None:
    meta = {
        "figure": "Figure 10",
        "claim": "The RF classifier is mainly driven by rainfall magnitude and rainfall-to-pit-depth ratios.",
        "model": "RF",
        "importance_type": "Permutation importance",
        "scoring": "Macro-F1",
        "error_bars": "Between-fold standard deviation of fold-level permutation importance means.",
        "input_agg": str(Path(args.step3_outdir) / "perm_importance_rf_agg.csv"),
        "input_byfold": str(Path(args.step3_outdir) / "perm_importance_rf_byfold.csv"),
        "n_features_plotted": int(len(table)),
        "requested_features": REQUESTED_FEATURES,
    }
    (outdir / "figure10_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Figure 10 RF feature importance.")
    p.add_argument(
        "--step3-outdir",
        default="outputs/step3",
        help="Directory containing Step3 ML outputs.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/figures/figure10",
        help="Output directory.",
    )
    p.add_argument("--top-n", type=int, default=10, help="Number of requested features to plot.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_matplotlib()
    outdir = ensure_dir(args.outdir)
    table = build_plot_table(Path(args.step3_outdir), args.top_n)
    table.to_csv(outdir / "figure10_rf_feature_importance_source_data.csv", index=False)
    write_metadata(outdir, args, table)
    plot_feature_importance(table, outdir)
    print(f"[Figure10] Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
