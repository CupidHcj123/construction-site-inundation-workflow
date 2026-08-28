#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Figure 8 panels as separate standalone figures.

This script reuses the same data preparation and plotting functions as the
combined Figure 8 script, so the statistics remain identical.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(__file__).resolve().parent / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib.pyplot as plt

from plot_figure8_response_controls import (
    build_lme_table,
    build_partial_response_table,
    build_spearman_table,
    ensure_dir,
    load_idealized_events,
    plot_lme_panel,
    plot_spearman_panel,
    plot_spline_panel,
    save_publication_figure,
    setup_matplotlib,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export Figure 8 panels A/B/C as separate figures.")
    p.add_argument(
        "--step2-outdir",
        default="outputs/step2",
        help="Directory containing Step2 statistical outputs.",
    )
    p.add_argument(
        "--events",
        default="data/events_extracted.csv",
        help="Event-level table used to fit Spline-OLS partial responses and compute LME scaling.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/figures/figure8_split_panels",
        help="Output directory for split panels.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_matplotlib()
    outdir = ensure_dir(args.outdir)

    step2_outdir = Path(args.step2_outdir)
    events_df = load_idealized_events(Path(args.events))
    spearman_table = build_spearman_table(step2_outdir)
    lme_table = build_lme_table(step2_outdir, events_df)
    spline_table = build_partial_response_table(events_df)

    spearman_table.to_csv(outdir / "figure8a_spearman_source_data.csv", index=False)
    lme_table.to_csv(outdir / "figure8b_lme_effect_source_data.csv", index=False)
    spline_table.to_csv(outdir / "figure8c_spline_partial_response_source_data.csv", index=False)

    fig_a, ax_a = plt.subplots(figsize=(4.8, 3.65))
    plot_spearman_panel(ax_a, spearman_table)
    if ax_a.get_legend() is not None:
        ax_a.get_legend().remove()
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.55, 1.14), ncol=2, fontsize=6.3)
    fig_a.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.30)
    save_publication_figure(fig_a, outdir / "figure8a_spearman_controls")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(5.05, 3.65))
    plot_lme_panel(ax_b, lme_table)
    fig_b.subplots_adjust(left=0.34, right=0.98, top=0.93, bottom=0.17)
    save_publication_figure(fig_b, outdir / "figure8b_lme_fixed_effects")
    plt.close(fig_b)

    fig_c, (ax_c1, ax_c2) = plt.subplots(2, 1, figsize=(4.1, 4.45))
    plot_spline_panel(ax_c1, ax_c2, spline_table)
    fig_c.subplots_adjust(left=0.18, right=0.98, top=0.94, bottom=0.13, hspace=0.52)
    save_publication_figure(fig_c, outdir / "figure8c_spline_partial_response")
    plt.close(fig_c)

    meta = {
        "figure": "Figure 8 split panels",
        "source_combined_script": "plot_figure8_response_controls.py",
        "panel_a": "Spearman correlations with selected controls.",
        "panel_b": "Standardized interaction LME fixed effects with 95% CI.",
        "panel_c": "Spline-OLS partial response curves.",
        "statistics_identical_to_combined_figure": True,
        "input_step2_outdir": str(step2_outdir),
        "input_events": str(args.events),
    }
    (outdir / "figure8_split_panels_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Figure8 split] Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
