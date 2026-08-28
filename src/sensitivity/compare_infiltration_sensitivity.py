#!/usr/bin/env python3
"""Compare pit-scale hydrodynamic responses across infiltration assumptions.

The script compares simulation-derived final inundation responses, not RF
probabilities. It is intended for a targeted sensitivity subset, for example
10--100 mm rainfall at 3600 s under FRONT, UNIFORM and BACK rainfall patterns.

Inputs must be pit-level event CSVs produced by event_extraction.py. The
baseline and every alternative must contain the same selected combinations of
Pit_ID, Rainfall_mm, Rainfall_Pattern and Rainfall_Duration_s.

Example
-------
python compare_infiltration_sensitivity.py \
  --baseline-csv events_fc8.csv \
  --case fc4=events_fc4.csv \
  --case fc12=events_fc12.csv \
  --case no_infiltration=events_no_infiltration.csv \
  --duration-s 3600 \
  --rainfall-mm 10 20 30 40 50 60 70 80 90 100 \
  --outdir infiltration_sensitivity
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


KEYS = ["Pit_ID", "Rainfall_mm", "Rainfall_Pattern", "Rainfall_Duration_s"]
REQUIRED_COLUMNS = set(KEYS + ["Final_Inundation_Depth_m", "Pit_Max_Depth_m"])
DEFAULT_PATTERNS = ("FRONT", "UNIFORM", "BACK")


def parse_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser()
    if not label:
        raise argparse.ArgumentTypeError("Case label cannot be empty")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Case CSV not found: {path}")
    return label, path


def canonicalize_events(
    path: Path,
    duration_s: int | None,
    rainfall_mm: list[float] | None,
    patterns: list[str],
    assume_duration_s: int | None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Rainfall_Pattern"] = df["Rainfall_Pattern"].astype(str).str.upper()
    for column in ["Rainfall_mm", "Rainfall_Duration_s", "Pit_ID", "Final_Inundation_Depth_m", "Pit_Max_Depth_m"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if assume_duration_s is not None:
        df["Rainfall_Duration_s"] = df["Rainfall_Duration_s"].fillna(assume_duration_s)
    df = df.dropna(subset=KEYS + ["Final_Inundation_Depth_m", "Pit_Max_Depth_m"])
    df = df[df["Pit_Max_Depth_m"] > 0].copy()

    if duration_s is not None:
        df = df[np.isclose(df["Rainfall_Duration_s"], duration_s)]
    if rainfall_mm:
        keep = np.zeros(len(df), dtype=bool)
        for rainfall in rainfall_mm:
            keep |= np.isclose(df["Rainfall_mm"], rainfall)
        df = df[keep]
    if patterns:
        df = df[df["Rainfall_Pattern"].isin(patterns)]

    df["Relative_Inundation_Ratio"] = (
        df["Final_Inundation_Depth_m"] / df["Pit_Max_Depth_m"]
    )
    if not np.isfinite(df["Relative_Inundation_Ratio"]).all():
        raise ValueError(f"{path} contains non-finite relative inundation ratios")
    if df.duplicated(KEYS).any():
        example = df.loc[df.duplicated(KEYS, keep=False), KEYS].head(8)
        raise ValueError(
            f"{path} has duplicate event rows after filtering. Example:\n{example.to_string(index=False)}"
        )
    if df.empty:
        raise ValueError(f"No records remain after applying filters to {path}")
    return df[KEYS + ["Final_Inundation_Depth_m", "Relative_Inundation_Ratio"]].copy()


def risk_class(values: pd.Series, thresholds: tuple[float, float]) -> pd.Categorical:
    low_medium, medium_high = thresholds
    return pd.Categorical(
        np.select(
            [values < low_medium, values < medium_high],
            ["Low", "Medium"],
            default="High",
        ),
        categories=["Low", "Medium", "High"],
        ordered=True,
    )


def spearman_rho(x: Iterable[float], y: Iterable[float]) -> float:
    x_series = pd.Series(x, dtype=float)
    y_series = pd.Series(y, dtype=float)
    if len(x_series) < 2 or x_series.nunique() < 2 or y_series.nunique() < 2:
        return np.nan
    return float(x_series.corr(y_series, method="spearman"))


def high_risk_metrics(base: pd.Series, case: pd.Series) -> dict[str, float | int]:
    base_high = base.eq("High")
    case_high = case.eq("High")
    tp = int((base_high & case_high).sum())
    fp = int((~base_high & case_high).sum())
    fn = int((base_high & ~case_high).sum())
    tn = int((~base_high & ~case_high).sum())
    denom_jaccard = tp + fp + fn
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "High_Risk_Recall": tp / (tp + fn) if tp + fn else np.nan,
        "High_Risk_Precision": tp / (tp + fp) if tp + fp else np.nan,
        "High_Risk_Jaccard": tp / denom_jaccard if denom_jaccard else np.nan,
    }


def compare_case(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    label: str,
    thresholds: tuple[float, float],
) -> pd.DataFrame:
    merged = baseline.merge(candidate, on=KEYS, how="outer", suffixes=("_Baseline", "_Case"), indicator=True)
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        example = unmatched[KEYS + ["_merge"]].head(12).to_string(index=False)
        raise ValueError(
            f"{label}: selected scenarios do not match the baseline. Example:\n{example}"
        )
    merged = merged.drop(columns="_merge")
    merged["Setting"] = label
    merged["Ratio_Delta"] = (
        merged["Relative_Inundation_Ratio_Case"]
        - merged["Relative_Inundation_Ratio_Baseline"]
    )
    merged["Absolute_Depth_Delta_m"] = (
        merged["Final_Inundation_Depth_m_Case"]
        - merged["Final_Inundation_Depth_m_Baseline"]
    )
    merged["Risk_Baseline"] = risk_class(merged["Relative_Inundation_Ratio_Baseline"], thresholds)
    merged["Risk_Case"] = risk_class(merged["Relative_Inundation_Ratio_Case"], thresholds)
    merged["Risk_Class_Changed"] = merged["Risk_Baseline"].ne(merged["Risk_Case"])
    merged["High_Risk_Baseline"] = merged["Risk_Baseline"].eq("High")
    merged["High_Risk_Case"] = merged["Risk_Case"].eq("High")
    return merged


def summarise_conditions(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    group_cols = ["Setting", "Rainfall_mm", "Rainfall_Pattern", "Rainfall_Duration_s"]
    for group, frame in pairs.groupby(group_cols, sort=True, observed=True):
        setting, rainfall, pattern, duration = group
        metrics = high_risk_metrics(frame["Risk_Baseline"], frame["Risk_Case"])
        rows.append(
            {
                "Setting": setting,
                "Rainfall_mm": rainfall,
                "Rainfall_Pattern": pattern,
                "Rainfall_Duration_s": duration,
                "N_Pits": len(frame),
                "Mean_Baseline_Ratio": frame["Relative_Inundation_Ratio_Baseline"].mean(),
                "Mean_Case_Ratio": frame["Relative_Inundation_Ratio_Case"].mean(),
                "Mean_Ratio_Delta": frame["Ratio_Delta"].mean(),
                "Median_Ratio_Delta": frame["Ratio_Delta"].median(),
                "Mean_Absolute_Depth_Delta_m": frame["Absolute_Depth_Delta_m"].mean(),
                "Risk_Class_Agreement": 1.0 - frame["Risk_Class_Changed"].mean(),
                "Pit_Rank_Spearman_Rho": spearman_rho(
                    frame["Relative_Inundation_Ratio_Baseline"],
                    frame["Relative_Inundation_Ratio_Case"],
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarise_settings(pairs: pd.DataFrame, condition_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for setting, frame in pairs.groupby("Setting", sort=True, observed=True):
        subset = condition_summary[condition_summary["Setting"] == setting]
        metrics = high_risk_metrics(frame["Risk_Baseline"], frame["Risk_Case"])
        rows.append(
            {
                "Setting": setting,
                "N_Paired_Records": len(frame),
                "N_Conditions": len(subset),
                "Mean_Ratio_Delta": frame["Ratio_Delta"].mean(),
                "Median_Ratio_Delta": frame["Ratio_Delta"].median(),
                "Mean_Absolute_Depth_Delta_m": frame["Absolute_Depth_Delta_m"].mean(),
                "Risk_Class_Agreement": 1.0 - frame["Risk_Class_Changed"].mean(),
                "Median_Pit_Rank_Spearman_Rho": subset["Pit_Rank_Spearman_Rho"].median(),
                "Minimum_Pit_Rank_Spearman_Rho": subset["Pit_Rank_Spearman_Rho"].min(),
                "Median_High_Risk_Jaccard": subset["High_Risk_Jaccard"].median(),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("Setting").reset_index(drop=True)


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())


def display_label(text: str) -> str:
    labels = {
        "fc4": r"$f_c=4$",
        "fc8": r"$f_c=8$",
        "fc12": r"$f_c=12$",
        "no_infiltration": "No infiltration",
    }
    return labels.get(text, text.replace("_", " "))


def make_figure(pairs: pd.DataFrame, condition_summary: pd.DataFrame, outpath: Path) -> None:
    settings = sorted(pairs["Setting"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(settings), 1)))
    color_map = dict(zip(settings, colors))
    pattern_order = [p for p in DEFAULT_PATTERNS if p in pairs["Rainfall_Pattern"].unique()]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(178 / 25.4, 75 / 25.4), constrained_layout=False)
    single_pattern = len(pattern_order) == 1
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.86,
        bottom=0.20 if single_pattern else 0.30,
        wspace=0.38,
    )

    # (a) Distribution of pairwise changes in relative inundation ratio.
    ax = axes[0]
    for idx, setting in enumerate(settings):
        values = pairs.loc[pairs["Setting"] == setting, "Ratio_Delta"].dropna().to_numpy()
        ax.boxplot(
            values,
            positions=[idx],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color_map[setting], "alpha": 0.55, "edgecolor": color_map[setting]},
            medianprops={"color": "#202020", "linewidth": 1.0},
            whiskerprops={"color": color_map[setting], "linewidth": 0.9},
            capprops={"color": color_map[setting], "linewidth": 0.9},
        )
    ax.axhline(0, color="#4a4a4a", linewidth=0.8)
    ax.set_xticks(range(len(settings)), [display_label(setting) for setting in settings], rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta$ relative inundation ratio")
    ax.set_title("(a) Pit-level response change", loc="left", fontweight="bold")

    # (b) Baseline rank agreement for each rainfall-pattern condition.
    ax = axes[1]
    for setting in settings:
        subset = condition_summary[condition_summary["Setting"] == setting]
        for pattern in pattern_order:
            block = subset[subset["Rainfall_Pattern"] == pattern].sort_values("Rainfall_mm")
            if block.empty:
                continue
            linestyle = {"FRONT": "-", "UNIFORM": "--", "BACK": ":"}.get(pattern, "-")
            ax.plot(
                block["Rainfall_mm"],
                block["Pit_Rank_Spearman_Rho"],
                color=color_map[setting],
                linestyle=linestyle,
                marker="o",
                markersize=2.5,
                linewidth=1.1,
            )
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.9, color="#8a8a8a", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Total rainfall (mm)")
    ax.set_ylabel(r"Pit-rank Spearman $\rho$")
    ax.set_title("(b) Rank agreement", loc="left", fontweight="bold")
    ax.legend(
        handles=[
            Line2D([0], [0], color=color_map[setting], linewidth=1.5, label=display_label(setting))
            for setting in settings
        ],
        title="Infiltration setting",
        loc="lower right",
        fontsize=6,
        title_fontsize=6.5,
        frameon=False,
    )

    # (c) High-risk agreement for each infiltration setting.
    ax = axes[2]
    bar_data = []
    for setting in settings:
        frame = pairs[pairs["Setting"] == setting]
        metrics = high_risk_metrics(frame["Risk_Baseline"], frame["Risk_Case"])
        bar_data.append((setting, metrics["High_Risk_Recall"], metrics["High_Risk_Precision"], metrics["High_Risk_Jaccard"]))
    x = np.arange(len(settings))
    width = 0.24
    for offset, metric_index, label, color in [
        (-width, 1, "Recall", "#4C78A8"),
        (0, 2, "Precision", "#F58518"),
        (width, 3, "Jaccard", "#54A24B"),
    ]:
        heights = [row[metric_index] for row in bar_data]
        ax.bar(x + offset, heights, width=width, label=label, color=color)
    ax.set_xticks(x, [display_label(row[0]) for row in bar_data], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Agreement metric")
    ax.set_title("(c) High-risk agreement", loc="left", fontweight="bold")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        frameon=False,
        fontsize=5.8,
        ncol=1,
        labelspacing=0.25,
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.55)
        ax.set_axisbelow(True)
    if not single_pattern:
        fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#4a4a4a",
                    linestyle={"FRONT": "-", "UNIFORM": "--", "BACK": ":"}.get(pattern, "-"),
                    linewidth=1.1,
                    label=pattern,
                )
                for pattern in pattern_order
            ],
            title="Line style: rainfall pattern",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(pattern_order),
            fontsize=5.8,
            title_fontsize=6.5,
            frameon=False,
        )
    fig.savefig(outpath, dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    # Matplotlib commonly writes RGBA TIFFs. Flatten to opaque RGB for submission.
    with Image.open(outpath) as image:
        rgb = Image.new("RGB", image.size, "white")
        if image.mode == "RGBA":
            rgb.paste(image, mask=image.getchannel("A"))
        else:
            rgb.paste(image.convert("RGB"))
        rgb.save(outpath, format="TIFF", compression="tiff_lzw", dpi=(600, 600))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-csv", required=True, type=Path, help="Event CSV for the baseline assumption, e.g. fc=8 mm h^-1.")
    parser.add_argument("--case", required=True, action="append", type=parse_case, help="Alternative setting in LABEL=PATH form. Repeat for each setting.")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--duration-s", type=int, default=3600, help="Rainfall duration to analyse; set to -1 to retain all durations.")
    parser.add_argument(
        "--assume-duration-s",
        type=int,
        default=None,
        help="Fill missing Rainfall_Duration_s values only when every record in an input CSV is known to share this duration.",
    )
    parser.add_argument("--rainfall-mm", nargs="+", type=float, default=None, help="Optional selected rainfall totals in mm.")
    parser.add_argument("--patterns", nargs="+", default=list(DEFAULT_PATTERNS), help="Rainfall patterns to retain.")
    parser.add_argument("--risk-thresholds", nargs=2, type=float, default=[0.30, 0.60], metavar=("LOW_MEDIUM", "MEDIUM_HIGH"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.baseline_csv.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {args.baseline_csv}")
    if args.risk_thresholds[0] >= args.risk_thresholds[1]:
        raise ValueError("Risk thresholds must satisfy LOW_MEDIUM < MEDIUM_HIGH")

    duration = None if args.duration_s < 0 else args.duration_s
    patterns = [pattern.upper() for pattern in args.patterns]
    thresholds = (float(args.risk_thresholds[0]), float(args.risk_thresholds[1]))
    args.outdir.mkdir(parents=True, exist_ok=True)

    baseline = canonicalize_events(
        args.baseline_csv, duration, args.rainfall_mm, patterns, args.assume_duration_s
    )
    pair_tables = []
    for label, path in args.case:
        candidate = canonicalize_events(
            path, duration, args.rainfall_mm, patterns, args.assume_duration_s
        )
        pair_tables.append(compare_case(baseline, candidate, label, thresholds))
    pairs = pd.concat(pair_tables, ignore_index=True)
    condition_summary = summarise_conditions(pairs)
    setting_summary = summarise_settings(pairs, condition_summary)

    pairs.to_csv(args.outdir / "infiltration_sensitivity_paired_records.csv", index=False)
    condition_summary.to_csv(args.outdir / "infiltration_sensitivity_by_condition.csv", index=False)
    setting_summary.to_csv(args.outdir / "infiltration_sensitivity_summary.csv", index=False)
    make_figure(pairs, condition_summary, args.outdir / "Figure_S1_infiltration_sensitivity.tiff")

    print(f"Wrote paired records: {args.outdir / 'infiltration_sensitivity_paired_records.csv'}")
    print(f"Wrote condition summary: {args.outdir / 'infiltration_sensitivity_by_condition.csv'}")
    print(f"Wrote setting summary: {args.outdir / 'infiltration_sensitivity_summary.csv'}")
    print(f"Wrote Figure S1: {args.outdir / 'Figure_S1_infiltration_sensitivity.tiff'}")


if __name__ == "__main__":
    main()
