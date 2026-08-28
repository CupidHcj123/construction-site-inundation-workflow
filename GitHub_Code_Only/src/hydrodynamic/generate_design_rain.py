#!/usr/bin/env python3
"""Generate a Chicago design hyetograph for the inundation model.

Output format consumed by Inertial_Adaptive_LimitQ_Horton.cpp:
start_s  end_s  intensity_mm_h

Beijing parameters come from DB11/T 969-2016. If --total-mm is supplied,
the Chicago shape is scaled to that exact event depth; otherwise the total
depth is calculated from the selected return period and duration.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


BEIJING_PARAMS = {
    # q = A * (1 + C * log10(P)) / (t + b) ** n, q in L/(s*hm2), t in min
    "beijing-I": {
        "short": (1558.0, 0.955, 5.551, 0.835),
        "long": (2719.0, 0.960, 11.591, 0.902),
    },
    "beijing-II": {
        "short": (591.0, 0.893, 1.859, 0.436),
        "long": (1602.0, 1.037, 11.593, 0.681),
    },
}


def rain_q_lps_hm2(duration_min: float, return_period: float, region: str) -> float:
    if duration_min <= 0.0:
        return 0.0
    band = "short" if duration_min <= 5.0 else "long"
    a, c, b, n = BEIJING_PARAMS[region][band]
    return a * (1.0 + c * math.log10(return_period)) / ((duration_min + b) ** n)


def idf_depth_mm(duration_min: float, return_period: float, region: str) -> float:
    """Depth for an IDF duration. 1 L/(s*hm2) = 0.006 mm/min."""
    if duration_min <= 0.0:
        return 0.0
    return rain_q_lps_hm2(duration_min, return_period, region) * 0.006 * duration_min


def chicago_cumulative_mm(
    elapsed_min: float,
    total_min: float,
    return_period: float,
    region: str,
    peak_ratio: float,
) -> float:
    peak_min = total_min * peak_ratio
    design_depth = idf_depth_mm(total_min, return_period, region)

    if elapsed_min <= peak_min:
        pseudo_min = (peak_min - elapsed_min) / peak_ratio
        return peak_ratio * (design_depth - idf_depth_mm(pseudo_min, return_period, region))

    pseudo_min = (elapsed_min - peak_min) / (1.0 - peak_ratio)
    return peak_ratio * design_depth + (1.0 - peak_ratio) * idf_depth_mm(
        pseudo_min, return_period, region
    )


def build_intervals(args: argparse.Namespace) -> list[tuple[float, float, float, float]]:
    total_min = args.duration_s / 60.0
    n_steps = math.ceil(args.duration_s / args.interval_s)
    raw_depths: list[tuple[float, float, float]] = []

    for idx in range(n_steps):
        start_s = idx * args.interval_s
        end_s = min((idx + 1) * args.interval_s, args.duration_s)
        start_min = start_s / 60.0
        end_min = end_s / 60.0
        d0 = chicago_cumulative_mm(
            start_min, total_min, args.return_period, args.region, args.peak_ratio
        )
        d1 = chicago_cumulative_mm(
            end_min, total_min, args.return_period, args.region, args.peak_ratio
        )
        raw_depths.append((start_s, end_s, max(0.0, d1 - d0)))

    raw_total = sum(depth for _, _, depth in raw_depths)
    target_total = args.total_mm if args.total_mm is not None else raw_total
    scale = target_total / raw_total if raw_total > 0.0 else 0.0

    intervals = []
    for start_s, end_s, depth_mm in raw_depths:
        scaled_depth = depth_mm * scale
        intensity_mm_h = scaled_depth / ((end_s - start_s) / 3600.0)
        intervals.append((start_s, end_s, intensity_mm_h, scaled_depth))
    return intervals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region", choices=sorted(BEIJING_PARAMS), default="beijing-II")
    parser.add_argument("--return-period", type=float, default=50.0)
    parser.add_argument("--duration-s", type=float, default=7200.0)
    parser.add_argument("--interval-s", type=float, default=300.0)
    parser.add_argument("--peak-ratio", type=float, default=0.40)
    parser.add_argument("--total-mm", type=float, default=None)
    args = parser.parse_args()

    if args.return_period <= 0.0:
        parser.error("--return-period must be positive")
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    if args.interval_s <= 0.0:
        parser.error("--interval-s must be positive")
    if not 0.0 < args.peak_ratio < 1.0:
        parser.error("--peak-ratio must be between 0 and 1")
    if args.total_mm is not None and args.total_mm <= 0.0:
        parser.error("--total-mm must be positive")
    return args


def main() -> int:
    args = parse_args()
    intervals = build_intervals(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_depth = sum(row[3] for row in intervals)
    peak_intensity = max(row[2] for row in intervals)
    with args.output.open("w", encoding="utf-8") as f:
        f.write("# start_s\tend_s\tintensity_mm_h\tdepth_mm\n")
        f.write(
            f"# region={args.region} return_period={args.return_period:g}a "
            f"duration_s={args.duration_s:g} interval_s={args.interval_s:g} "
            f"peak_ratio={args.peak_ratio:g} total_depth_mm={total_depth:.6f}\n"
        )
        for start_s, end_s, intensity_mm_h, depth_mm in intervals:
            f.write(f"{start_s:.0f}\t{end_s:.0f}\t{intensity_mm_h:.6f}\t# {depth_mm:.6f}\n")

    print(
        f"Wrote {args.output} depth={total_depth:.3f} mm "
        f"peak_intensity={peak_intensity:.3f} mm/h",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
