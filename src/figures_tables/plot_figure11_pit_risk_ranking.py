#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 11: pit-level risk ranking under design-storm transfer.

Bars show true pit-level mean relative inundation ratio, sorted from low to
high. The line shows RF mean high-risk probability for the same pit order.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(__file__).resolve().parent / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib as mpl
import matplotlib.pyplot as plt


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


def build_pit_table(step3_outdir: Path) -> tuple[pd.DataFrame, pd.Series]:
    pred = pd.read_csv(step3_outdir / "external_design_predictions_rf.csv")
    rank = pd.read_csv(step3_outdir / "external_design_pit_rank_rf.csv").iloc[0]
    pit = (
        pred.groupby("Pit_ID", as_index=False)
        .agg(
            true_mean_ratio=("y_ratio_raw", "mean"),
            pred_mean_high_prob=("p_high", "mean"),
            true_high_rate=("true", lambda s: float((s == 2).mean())),
            pred_high_rate=("pred", lambda s: float((s == 2).mean())),
            n_events=("Pit_ID", "size"),
        )
        .sort_values("true_mean_ratio", ascending=True)
        .reset_index(drop=True)
    )
    pit["Pit_Label"] = pit["Pit_ID"].map(lambda x: f"Pit {int(x)}")
    pit["rank_by_true_ratio"] = range(1, len(pit) + 1)
    return pit, rank


def plot_pit_ranking(pit: pd.DataFrame, rank: pd.Series, outdir: Path) -> None:
    x = range(len(pit))
    bar_color = "#9EB8C4"
    line_color = "#B84A3A"

    fig, ax1 = plt.subplots(figsize=(5.8, 3.35))
    bars = ax1.bar(
        x,
        pit["true_mean_ratio"],
        color=bar_color,
        width=0.64,
        edgecolor="none",
        label="True mean relative ratio",
        zorder=2,
    )
    ax1.set_ylabel("True mean relative ratio", color="#2F4F5F", fontsize=7)
    ax1.tick_params(axis="y", labelcolor="#2F4F5F", labelsize=6.5)
    ax1.set_ylim(0, max(0.95, float(pit["true_mean_ratio"].max()) * 1.18))
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(pit["Pit_Label"], fontsize=6.5)
    ax1.set_xlabel("Pit ID sorted by true mean relative inundation ratio", fontsize=7)
    ax1.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8, zorder=0)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        pit["pred_mean_high_prob"],
        color=line_color,
        marker="o",
        markersize=4,
        linewidth=1.55,
        label="RF mean high-risk probability",
        zorder=3,
    )
    ax2.set_ylabel("RF mean high-risk probability", color=line_color, fontsize=7)
    ax2.tick_params(axis="y", labelcolor=line_color, labelsize=6.5)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.set_ylim(0, max(0.95, float(pit["pred_mean_high_prob"].max()) * 1.18))

    rho = float(rank["pit_rank_spearman_prob"])
    p = float(rank["pit_rank_p_prob"])
    p_txt = f"{p:.6f}" if p < 0.001 else f"{p:.3f}"
    ax1.text(
        0.03,
        0.94,
        rf"Spearman $\rho$ = {rho:.3f}" + "\n" + rf"$p$ = {p_txt}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D0D0D0", "linewidth": 0.6},
    )

    handles = [bars, ax2.lines[0]]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.58, 1.02), ncol=2, fontsize=6.4)

    fig.subplots_adjust(left=0.12, right=0.88, top=0.91, bottom=0.18)
    save_publication_figure(fig, outdir / "figure11_pit_level_risk_ranking")
    plt.close(fig)


def write_metadata(outdir: Path, args: argparse.Namespace, pit: pd.DataFrame, rank: pd.Series) -> None:
    meta = {
        "figure": "Figure 11",
        "claim": "RF preserves pit-level risk ranking under design-storm transfer despite weak return-period gradient tracking.",
        "model": "RF",
        "x_order": "Pit ID sorted by true mean relative inundation ratio.",
        "bar": "True mean relative inundation ratio.",
        "line": "RF mean high-risk probability.",
        "spearman_rho": float(rank["pit_rank_spearman_prob"]),
        "spearman_p": float(rank["pit_rank_p_prob"]),
        "input_predictions": str(Path(args.step3_outdir) / "external_design_predictions_rf.csv"),
        "input_rank_metrics": str(Path(args.step3_outdir) / "external_design_pit_rank_rf.csv"),
        "n_pits": int(len(pit)),
    }
    (outdir / "figure11_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Figure 11 pit-level RF risk ranking.")
    p.add_argument(
        "--step3-outdir",
        default="outputs/step3",
        help="Directory containing Step3 ML outputs.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/figures/figure11",
        help="Output directory.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_matplotlib()
    outdir = ensure_dir(args.outdir)
    pit, rank = build_pit_table(Path(args.step3_outdir))
    pit.to_csv(outdir / "figure11_pit_level_risk_ranking_source_data.csv", index=False)
    write_metadata(outdir, args, pit, rank)
    plot_pit_ranking(pit, rank, outdir)
    print(f"[Figure11] Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
