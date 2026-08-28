#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7: pit-scale inundation depth under idealized rainfall patterns.

Default output:
  1) A 3 x 3 heatmap grid:
     rows = rainfall durations, columns = FRONT / BACK / UNIFORM,
     x = rainfall amount, y = Pit_ID, color = final inundation depth.
  2) A duration-faceted grouped boxplot:
     boxes summarize pit-scale depths across pits for each rainfall-pattern pair.

The heatmap is recommended as the main Figure 7 because it preserves pit-level
heterogeneity while still showing rainfall, duration, and pattern effects.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(__file__).resolve().parent / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PATTERN_ORDER = ["FRONT", "BACK", "UNIFORM"]
PATTERN_COLORS = {
    "FRONT": "#C65A4A",
    "BACK": "#386FA4",
    "UNIFORM": "#5B9E6D",
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
            "axes.linewidth": 0.75,
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


def validate_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_idealized_data(input_path: Path, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required = [
        "Pit_ID",
        "Rainfall_mm",
        "Rainfall_Duration_s",
        "Rainfall_Pattern",
        "Scenario_Type",
        target_col,
    ]
    validate_columns(df, required)

    out = df.copy()
    out["Rainfall_Pattern"] = out["Rainfall_Pattern"].astype(str).str.upper().str.strip()
    out["Scenario_Type"] = out["Scenario_Type"].astype(str).str.upper().str.strip()
    out = out[out["Scenario_Type"].eq("IDEALIZED")].copy()
    out = out[out["Rainfall_Pattern"].isin(PATTERN_ORDER)].copy()

    for col in ["Pit_ID", "Rainfall_mm", "Rainfall_Duration_s", target_col]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Pit_ID", "Rainfall_mm", "Rainfall_Duration_s", target_col])
    out["Pit_ID"] = out["Pit_ID"].astype(int)
    return out


def write_source_tables(df: pd.DataFrame, outdir: Path, target_col: str) -> None:
    df.to_csv(outdir / "figure7_idealized_pit_scale_source_data.csv", index=False)
    summary = (
        df.groupby(["Rainfall_Duration_s", "Rainfall_Pattern", "Rainfall_mm"], observed=True)[target_col]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
    )
    summary.to_csv(outdir / "figure7_idealized_depth_summary.csv", index=False)


def target_label(target_col: str) -> str:
    labels = {
        "Final_Inundation_Depth_m": "Final inundation depth (m)",
        "Hmax_m": "Maximum inundation depth (m)",
        "y_ratio_raw": "Relative inundation ratio (-)",
        "Peak_to_Potential_Ratio": "Peak-to-potential ratio (-)",
    }
    return labels.get(target_col, target_col)


def plot_heatmap_grid(
    df: pd.DataFrame,
    outdir: Path,
    target_col: str,
    cmap: str = "YlGnBu",
    vmax_quantile: float = 0.98,
) -> None:
    durations = sorted(df["Rainfall_Duration_s"].dropna().unique())
    rainfalls = sorted(df["Rainfall_mm"].dropna().unique())
    pits = sorted(df["Pit_ID"].dropna().unique())

    vmax = float(df[target_col].quantile(vmax_quantile))
    vmax = max(vmax, float(df[target_col].max()), 1e-9) if vmax <= 0 else vmax
    vmin = 0.0 if df[target_col].min() >= 0 else float(df[target_col].min())

    fig, axes = plt.subplots(
        nrows=len(durations),
        ncols=len(PATTERN_ORDER),
        figsize=(11.6, 7.3),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.asarray(axes)

    image = None
    for i, duration in enumerate(durations):
        for j, pattern in enumerate(PATTERN_ORDER):
            ax = axes[i, j]
            sub = df[(df["Rainfall_Duration_s"].eq(duration)) & (df["Rainfall_Pattern"].eq(pattern))]
            mat = (
                sub.pivot_table(index="Pit_ID", columns="Rainfall_mm", values=target_col, aggfunc="mean")
                .reindex(index=pits, columns=rainfalls)
            )
            image = ax.imshow(mat.values, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(pattern if i == 0 else "", color=PATTERN_COLORS.get(pattern, "black"), fontsize=8, pad=5)
            if j == 0:
                ax.set_ylabel(f"{int(duration)} s\nPit ID", fontsize=7)
            if i == len(durations) - 1:
                ax.set_xlabel("Rainfall amount (mm)", fontsize=7)

            ax.set_xticks(np.arange(len(rainfalls)))
            ax.set_xticklabels([f"{int(r)}" if float(r).is_integer() else f"{r:g}" for r in rainfalls], rotation=90, fontsize=5.8)
            ax.set_yticks(np.arange(len(pits)))
            ax.set_yticklabels([str(p) for p in pits], fontsize=6)
            ax.tick_params(length=2, pad=1)

            # Light gridlines make individual pit-rainfall cells auditable.
            ax.set_xticks(np.arange(-0.5, len(rainfalls), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(pits), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.25, alpha=0.7)
            ax.tick_params(which="minor", bottom=False, left=False)

    cax = fig.add_axes([0.925, 0.16, 0.018, 0.70])
    cb = fig.colorbar(image, cax=cax)
    cb.set_label(target_label(target_col), fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)

    fig.subplots_adjust(left=0.075, right=0.90, top=0.955, bottom=0.10, wspace=0.08, hspace=0.16)
    save_publication_figure(fig, outdir / "figure7_idealized_depth_heatmap")
    plt.close(fig)


def plot_boxplot_grid(df: pd.DataFrame, outdir: Path, target_col: str) -> None:
    durations = sorted(df["Rainfall_Duration_s"].dropna().unique())
    rainfalls = sorted(df["Rainfall_mm"].dropna().unique())
    offsets = {"FRONT": -0.24, "BACK": 0.0, "UNIFORM": 0.24}
    width = 0.20

    fig, axes = plt.subplots(
        nrows=len(durations),
        ncols=1,
        figsize=(11.6, 6.8),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)

    for ax, duration in zip(axes, durations):
        for pattern in PATTERN_ORDER:
            data = []
            positions = []
            for k, rainfall in enumerate(rainfalls):
                vals = df[
                    df["Rainfall_Duration_s"].eq(duration)
                    & df["Rainfall_Pattern"].eq(pattern)
                    & df["Rainfall_mm"].eq(rainfall)
                ][target_col].dropna().values
                data.append(vals)
                positions.append(k + offsets[pattern])
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                medianprops={"color": "#222222", "linewidth": 0.7},
                whiskerprops={"color": "#555555", "linewidth": 0.5},
                capprops={"color": "#555555", "linewidth": 0.5},
                boxprops={"linewidth": 0.55, "color": "#555555"},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(PATTERN_COLORS[pattern])
                patch.set_alpha(0.72)

        ax.set_ylabel(f"{int(duration)} s\n{target_label(target_col)}", fontsize=7)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.8)
        ax.tick_params(axis="both", labelsize=6, length=2)

    axes[-1].set_xticks(np.arange(len(rainfalls)))
    axes[-1].set_xticklabels([f"{int(r)}" if float(r).is_integer() else f"{r:g}" for r in rainfalls], rotation=90, fontsize=5.8)
    axes[-1].set_xlabel("Rainfall amount (mm)", fontsize=7)

    handles = [Patch(facecolor=PATTERN_COLORS[p], edgecolor="#555555", label=p, alpha=0.72) for p in PATTERN_ORDER]
    axes[0].legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.20))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.12, hspace=0.20)
    save_publication_figure(fig, outdir / "figure7_idealized_depth_boxplot")
    plt.close(fig)


def write_figure_metadata(outdir: Path, args: argparse.Namespace, df: pd.DataFrame) -> None:
    meta = {
        "figure": "Figure 7",
        "claim": (
            "Idealized rainfall temporal patterns produce distinct pit-scale inundation responses "
            "across rainfall amount and duration."
        ),
        "scenario_filter": "Scenario_Type == IDEALIZED",
        "patterns": PATTERN_ORDER,
        "n_rows": int(len(df)),
        "n_pits": int(df["Pit_ID"].nunique()),
        "rainfall_levels": [float(x) for x in sorted(df["Rainfall_mm"].dropna().unique())],
        "duration_levels_s": [float(x) for x in sorted(df["Rainfall_Duration_s"].dropna().unique())],
        "target_col": args.target_col,
        "input": str(args.input),
    }
    with open(outdir / "figure7_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Figure 7 for idealized rainfall pattern pit-scale inundation.")
    p.add_argument(
        "--input",
        default="data/events_extracted.csv",
        help="Input events table generated by event_extraction.py.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/figures/figure7",
        help="Output directory for Figure 7 files.",
    )
    p.add_argument(
        "--target-col",
        default="Final_Inundation_Depth_m",
        help="Column used as plotted water-depth metric. Common choices: Final_Inundation_Depth_m, Hmax_m, y_ratio_raw.",
    )
    p.add_argument(
        "--vmax-quantile",
        type=float,
        default=0.98,
        help="Upper color-scale quantile for heatmaps. Use 1.0 to force the absolute maximum.",
    )
    p.add_argument(
        "--skip-boxplot",
        action="store_true",
        help="Only generate the heatmap version.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_matplotlib()
    outdir = ensure_dir(args.outdir)
    df = load_idealized_data(Path(args.input), args.target_col)
    if df.empty:
        raise SystemExit("No IDEALIZED rows found after filtering.")

    write_source_tables(df, outdir, args.target_col)
    write_figure_metadata(outdir, args, df)
    plot_heatmap_grid(df, outdir, args.target_col, vmax_quantile=args.vmax_quantile)
    if not args.skip_boxplot:
        plot_boxplot_grid(df, outdir, args.target_col)
    print(f"[Figure7] Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
