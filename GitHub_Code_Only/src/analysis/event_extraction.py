#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

EARTH_EQ_M_PER_DEG = 111320.0
PATTERN_NAMES = ("FRONT", "UNIFORM", "BACK", "CHICAGO", "REAL")


def progress(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


@dataclass
class Grid:
    meta: Dict[str, float]
    data: np.ndarray


@dataclass
class RunMeta:
    run_dir: Path
    rainfall_mm: Optional[float]
    pattern: str
    rainfall_duration_s: Optional[int]
    simulation_duration_s: Optional[int]
    output_step_s: int
    scenario_type: str
    rain_file_path: Optional[str]
    rain_file_total_mm: Optional[float]
    rain_file_peak_intensity_mm_h: Optional[float]
    rain_file_interval_s: Optional[float]
    design_region: Optional[str]
    return_period_yr: Optional[float]
    chicago_peak_ratio: Optional[float]


def read_ascii_grid(path: str | Path) -> Grid:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        header: Dict[str, float] = {}
        for _ in range(6):
            parts = f.readline().strip().split()
            if len(parts) < 2:
                raise ValueError(f"Malformed ASCII header: {path}")
            header[parts[0].lower()] = float(parts[1])
        ncols = int(header["ncols"])
        nrows = int(header["nrows"])
        data = np.loadtxt(f, dtype=float).reshape((nrows, ncols))
    return Grid(
        meta={
            "ncols": ncols,
            "nrows": nrows,
            "xllcorner": header.get("xllcorner", header.get("xllcenter", np.nan)),
            "yllcorner": header.get("yllcorner", header.get("yllcenter", np.nan)),
            "cellsize": header["cellsize"],
            "NODATA_value": header.get("nodata_value", -9999.0),
        },
        data=data,
    )


def spacing_m(meta: Dict[str, float], override_cellsize_m: Optional[float] = None) -> Tuple[float, float, float]:
    if override_cellsize_m is not None and override_cellsize_m > 0:
        return float(override_cellsize_m), float(override_cellsize_m), float(override_cellsize_m) ** 2

    cs = float(meta["cellsize"])
    xll = float(meta["xllcorner"])
    yll = float(meta["yllcorner"])
    is_geo = (-180 <= xll <= 180) and (-90 <= yll <= 90) and (0 < cs < 0.1)
    if not is_geo:
        return cs, cs, cs * cs

    lat_center = yll + cs * (int(meta["nrows"]) / 2.0)
    dy = cs * EARTH_EQ_M_PER_DEG
    dx = cs * EARTH_EQ_M_PER_DEG * math.cos(math.radians(lat_center))
    return dx, dy, dx * dy


def normalize_number(value: object) -> float:
    text = str(value).strip().replace("\u2212", "-")
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    return float(text)


def read_thresholds(path: str | Path) -> Dict[int, float]:
    with Path(path).open("r", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Empty threshold CSV: {path}")
        lower = {name.strip().lower(): name for name in reader.fieldnames}
        pid_col = lower.get("pit_id") or lower.get("id") or lower.get("pitid")
        thr_col = lower.get("threshold") or lower.get("threshold_elev") or lower.get("threshold_elevation")
        if pid_col is None or thr_col is None:
            raise ValueError(f"Threshold CSV needs pit_id/id and threshold columns; got {reader.fieldnames}")
        out: Dict[int, float] = {}
        for row in reader:
            out[int(normalize_number(row[pid_col]))] = normalize_number(row[thr_col])
        return out


def parse_input_info(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    meta: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                meta[parts[0].strip()] = parts[1].strip()
    return meta


def first_float(meta: Dict[str, str], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in meta:
            try:
                return float(meta[key])
            except ValueError:
                continue
    return None


def parse_key_value_comment(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
        out[key.lower()] = value
    return out


def parse_float_token(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    text = re.sub(r"(?i)a$", "", text)
    try:
        return float(text)
    except ValueError:
        return None


def parse_rain_file(path: Optional[Path]) -> Dict[str, Optional[float] | Optional[str]]:
    result: Dict[str, Optional[float] | Optional[str]] = {
        "total_mm": None,
        "peak_intensity_mm_h": None,
        "interval_s": None,
        "duration_s": None,
        "region": None,
        "return_period_yr": None,
        "peak_ratio": None,
    }
    if path is None or not path.exists():
        return result

    total_from_rows = 0.0
    peak_intensity = None
    intervals: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                kv = parse_key_value_comment(stripped)
                if "total_depth_mm" in kv:
                    result["total_mm"] = parse_float_token(kv["total_depth_mm"])
                if "duration_s" in kv:
                    result["duration_s"] = parse_float_token(kv["duration_s"])
                if "interval_s" in kv:
                    result["interval_s"] = parse_float_token(kv["interval_s"])
                if "region" in kv:
                    result["region"] = kv["region"]
                if "return_period" in kv:
                    result["return_period_yr"] = parse_float_token(kv["return_period"])
                if "peak_ratio" in kv:
                    result["peak_ratio"] = parse_float_token(kv["peak_ratio"])
                continue

            data_part = stripped.split("#", 1)[0].strip()
            parts = re.split(r"[\s,]+", data_part)
            if len(parts) < 3:
                continue
            try:
                start_s = float(parts[0])
                end_s = float(parts[1])
                intensity = float(parts[2])
            except ValueError:
                continue
            duration = end_s - start_s
            if duration <= 0:
                continue
            intervals.append(duration)
            total_from_rows += intensity * duration / 3600.0
            peak_intensity = intensity if peak_intensity is None else max(peak_intensity, intensity)

    if result["total_mm"] is None and total_from_rows > 0:
        result["total_mm"] = total_from_rows
    if result["peak_intensity_mm_h"] is None and peak_intensity is not None:
        result["peak_intensity_mm_h"] = peak_intensity
    if result["interval_s"] is None and intervals:
        result["interval_s"] = min(intervals)
    if result["duration_s"] is None and intervals:
        result["duration_s"] = sum(intervals)
    return result


def parse_design_from_name(run_dir: Path) -> Dict[str, Optional[float] | Optional[str]]:
    name = run_dir.name
    out: Dict[str, Optional[float] | Optional[str]] = {
        "region": None,
        "return_period_yr": None,
        "peak_ratio": None,
        "total_mm": None,
    }
    m = re.search(r"(?i)(beijing-[A-Za-z0-9]+)", name)
    if m:
        out["region"] = m.group(1)
    m = re.search(r"(?i)(\d+(?:\.\d+)?)\s*a(?:$|[_\-\s])", name)
    if m:
        out["return_period_yr"] = float(m.group(1))
    m = re.search(r"(?i)(\d+(?:\.\d+)?)\s*mm", name)
    if m:
        out["total_mm"] = float(m.group(1))
    m = re.search(r"(?i)(?:^|[_\-\s])r(\d{2,3})(?:$|[_\-\s])", name)
    if m:
        out["peak_ratio"] = float(m.group(1)) / 100.0
    return out


def parse_run_dir_name(run_dir: Path) -> Tuple[Optional[float], Optional[str], Optional[int]]:
    name = run_dir.name.upper()
    rain = None
    duration = None
    pattern = None

    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*MM(?![A-Z])", name)
    if m:
        rain = float(m.group(1))
    m = re.search(r"(?<!\d)(\d{3,6})\s*S(?![A-Z])", name)
    if m:
        duration = int(m.group(1))
    for candidate in PATTERN_NAMES:
        if re.search(rf"(^|[_\-\s]){candidate}($|[_\-\s])", name):
            pattern = candidate
            break
    return rain, pattern, duration


def parse_run_meta(run_dir: Path) -> RunMeta:
    info = parse_input_info(run_dir / "INPUT_INFO")
    rain_from_name, pattern_from_name, duration_from_name = parse_run_dir_name(run_dir)
    design_from_name = parse_design_from_name(run_dir)

    rain_file_raw = info.get("RAIN_FILE_PATH")
    rain_file_path = None
    if rain_file_raw and rain_file_raw.upper() != "NONE":
        candidate = Path(rain_file_raw)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        rain_file_path = candidate
    rain_info = parse_rain_file(rain_file_path)

    water_depth_m = first_float(info, ["WATER_DEPTH(m)", "WATER_DEPTH", "RAIN_TOTAL_M", "TOTAL_RAIN_M"])
    rainfall_mm = None
    if water_depth_m is not None and water_depth_m > 0:
        rainfall_mm = water_depth_m * 1000.0
    elif rain_info["total_mm"] is not None:
        rainfall_mm = float(rain_info["total_mm"])
    elif design_from_name["total_mm"] is not None:
        rainfall_mm = float(design_from_name["total_mm"])
    else:
        rainfall_mm = rain_from_name

    info_pattern = (
        info.get("RAIN_PATTERN")
        or info.get("HYETOGRAPH")
        or info.get("RAIN_MODE")
    )
    info_pattern = str(info_pattern).upper() if info_pattern is not None else None

    # Older idealized runs can carry a generic RAIN_MODE=REAL/FILE in INPUT_INFO.
    # The folder name is more specific for FRONT/BACK/UNIFORM matrix runs.
    if pattern_from_name in {"FRONT", "UNIFORM", "BACK"}:
        pattern = pattern_from_name
    else:
        pattern = info_pattern or pattern_from_name or "UNIFORM"

    if pattern in {"FILE", "EXTERNAL"}:
        has_design_meta = (
            rain_info["region"] is not None
            or rain_info["return_period_yr"] is not None
            or design_from_name["region"] is not None
            or design_from_name["return_period_yr"] is not None
        )
        pattern = "CHICAGO" if has_design_meta else "REAL"

    rain_duration = first_float(
        info,
        [
            "RAIN_DURATION(s)",
            "RAINFALL_DURATION(s)",
            "DURATION(s)",
            "RAIN_DURATION",
            "RAINFALL_DURATION",
        ],
    )
    sim_duration = first_float(info, ["SIM_DURATION(s)", "SIM_TIME(s)", "T_END(s)", "END_TIME(s)"])
    if sim_duration is None:
        sim_duration = first_float(info, ["TOTAL_TIME(s)", "TOTAL_TIME"])
    output_step = first_float(info, ["FDR_OUTSTEP", "OUTPUT_STEP(s)", "OUTSTEP"])
    if rain_duration is None and rain_info["duration_s"] is not None:
        rain_duration = float(rain_info["duration_s"])
    # Matrix runs commonly use TOTAL_TIME(s) as both the simulation and rainfall
    # duration. This fallback preserves the duration when the folder name omits it.
    if rain_duration is None:
        rain_duration = sim_duration

    region = rain_info["region"] or design_from_name["region"]
    return_period = rain_info["return_period_yr"] or design_from_name["return_period_yr"]
    peak_ratio = rain_info["peak_ratio"] or design_from_name["peak_ratio"]
    if pattern == "CHICAGO":
        scenario_type = "DESIGN_CHICAGO"
    elif pattern == "REAL":
        scenario_type = "REAL_EVENT"
    else:
        scenario_type = "IDEALIZED"

    return RunMeta(
        run_dir=run_dir,
        rainfall_mm=rainfall_mm,
        pattern=pattern,
        rainfall_duration_s=int(round(rain_duration)) if rain_duration is not None else duration_from_name,
        simulation_duration_s=int(round(sim_duration)) if sim_duration is not None else None,
        output_step_s=int(round(output_step)) if output_step is not None else 60,
        scenario_type=scenario_type,
        rain_file_path=str(rain_file_path) if rain_file_path is not None else None,
        rain_file_total_mm=float(rain_info["total_mm"]) if rain_info["total_mm"] is not None else None,
        rain_file_peak_intensity_mm_h=float(rain_info["peak_intensity_mm_h"]) if rain_info["peak_intensity_mm_h"] is not None else None,
        rain_file_interval_s=float(rain_info["interval_s"]) if rain_info["interval_s"] is not None else None,
        design_region=str(region) if region is not None else None,
        return_period_yr=float(return_period) if return_period is not None else None,
        chicago_peak_ratio=float(peak_ratio) if peak_ratio is not None else None,
    )


def discover_run_dirs(runs_root: str | Path, recursive: bool = False, max_runs: Optional[int] = None) -> List[Path]:
    root = Path(runs_root)
    candidates = root.rglob("INPUT_INFO") if recursive else root.glob("*/INPUT_INFO")
    dirs = sorted({p.parent for p in candidates})
    if max_runs is not None:
        dirs = dirs[:max_runs]
    return dirs


def discover_time_from_name(path: Path) -> Optional[int]:
    m = re.search(r"(?<!\d)(\d{2,6})S?(?=\.(?:ASC|TIF)$)", path.name.upper())
    return int(m.group(1)) if m else None


def sorted_process_files(run_dir: Path, output_step_s: int) -> List[Tuple[Path, int]]:
    proc_dir = run_dir / "process"
    files = sorted(proc_dir.glob("*.asc"))
    times = [discover_time_from_name(p) for p in files]
    if any(t is None for t in times):
        times = [output_step_s * (i + 1) for i in range(len(files))]
    return [(p, int(t)) for p, t in zip(files, times) if t is not None]


def depth_to_m(data: np.ndarray, unit: str) -> np.ndarray:
    if unit == "m":
        return data.astype(float)
    if unit == "mm":
        return data.astype(float) / 1000.0
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return data.astype(float)
    return data.astype(float) / 1000.0 if np.nanpercentile(finite, 99) > 20 else data.astype(float)


def clean_nodata(grid: Grid) -> np.ndarray:
    data = grid.data.astype(float)
    nodata = grid.meta.get("NODATA_value")
    if nodata is not None and np.isfinite(nodata):
        data[data == float(nodata)] = np.nan
    return data


def outside_ring(mask: np.ndarray, pit_id: int) -> np.ndarray:
    pit = mask == pit_id
    ring = np.zeros_like(pit, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.zeros_like(pit, dtype=bool)
            src_i = slice(max(0, -di), pit.shape[0] - max(0, di))
            dst_i = slice(max(0, di), pit.shape[0] - max(0, -di))
            src_j = slice(max(0, -dj), pit.shape[1] - max(0, dj))
            dst_j = slice(max(0, dj), pit.shape[1] - max(0, -dj))
            shifted[dst_i, dst_j] = pit[src_i, src_j]
            ring |= shifted
    return ring & ~pit


def compute_geometry(
    dsm: np.ndarray,
    mask: np.ndarray,
    cell_area_m2: float,
    thresholds: Optional[Dict[int, float]],
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, np.ndarray]]:
    pit_ids = sorted(int(v) for v in np.unique(mask) if np.isfinite(v) and v != 0)
    pit_masks = {pid: mask == pid for pid in pit_ids}
    geom: Dict[int, Dict[str, float]] = {}

    for pid in pit_ids:
        pit_mask = pit_masks[pid]
        z = dsm[pit_mask].astype(float)
        z = z[np.isfinite(z)]
        if z.size == 0:
            continue

        if thresholds is not None:
            if pid not in thresholds:
                raise ValueError(f"Missing threshold for Pit_ID={pid}")
            threshold_elev = thresholds[pid]
            rim_elev = np.nan
        else:
            ring = outside_ring(mask, pid)
            ring_z = dsm[ring].astype(float)
            ring_z = ring_z[np.isfinite(ring_z)]
            if ring_z.size == 0:
                raise ValueError(f"No outside ring cells for Pit_ID={pid}")
            threshold_elev = float(np.nanmin(ring_z))
            rim_elev = threshold_elev

        potential_depth = np.maximum(0.0, threshold_elev - z)
        area = float(pit_mask.sum()) * cell_area_m2
        volume = float(np.nansum(potential_depth)) * cell_area_m2
        max_depth = float(np.nanmax(potential_depth)) if potential_depth.size else 0.0
        avg_depth = volume / area if area > 0 else 0.0

        geom[pid] = {
            "Pit_Area_m2": area,
            "Threshold_Elev": float(threshold_elev),
            "Rim_Elev_m": float(rim_elev) if np.isfinite(rim_elev) else np.nan,
            "Pit_Min_Elev_m": float(np.nanmin(z)),
            "Pit_Mean_Elev_m": float(np.nanmean(z)),
            "Pit_Max_Depth_m": max_depth,
            "Pit_Avg_Depth_m": avg_depth,
            "Pit_Volume_m3": volume,
            "Shape_Ratio": avg_depth / max_depth if max_depth > 0 else 0.0,
        }
    return geom, pit_masks


def max_and_mean_by_pit(depth_m: np.ndarray, pit_masks: Dict[int, np.ndarray]) -> Tuple[Dict[int, float], Dict[int, float]]:
    maxes: Dict[int, float] = {}
    means: Dict[int, float] = {}
    for pid, pit_mask in pit_masks.items():
        vals = depth_m[pit_mask]
        vals = vals[np.isfinite(vals)]
        maxes[pid] = float(np.nanmax(vals)) if vals.size else np.nan
        means[pid] = float(np.nanmean(vals)) if vals.size else np.nan
    return maxes, means


def extract_events(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, object]]:
    t_all = time.monotonic()
    progress("[Extract] Loading DSM and pit mask...", args.quiet)
    dsm_grid = read_ascii_grid(args.dsm)
    mask_grid = read_ascii_grid(args.mask)
    if dsm_grid.data.shape != mask_grid.data.shape:
        raise ValueError(f"DSM shape {dsm_grid.data.shape} != mask shape {mask_grid.data.shape}")

    _, _, cell_area = spacing_m(dsm_grid.meta, args.override_cellsize_m)
    progress("[Extract] Computing pit geometry...", args.quiet)
    thresholds = read_thresholds(args.seeds) if args.seeds else None
    geom, pit_masks = compute_geometry(clean_nodata(dsm_grid), mask_grid.data, cell_area, thresholds)
    progress(f"[Extract] Pit count: {len(geom)}", args.quiet)

    run_roots = args.runs if isinstance(args.runs, list) else [args.runs]
    progress(f"[Extract] Discovering run folders under {len(run_roots)} root(s)...", args.quiet)
    run_dirs: List[Path] = []
    for root in run_roots:
        run_dirs.extend(discover_run_dirs(root, recursive=args.recursive, max_runs=None))
    run_dirs = sorted(set(run_dirs))
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]
    progress(f"[Extract] Discovered {len(run_dirs)} run folder(s).", args.quiet)
    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, str]] = []

    progress_every = max(1, int(args.progress_every))
    total_runs = len(run_dirs)
    for idx, run_dir in enumerate(run_dirs, 1):
        t_run = time.monotonic()
        meta = parse_run_meta(run_dir)
        if meta.rainfall_mm is None:
            skipped.append({"run_dir": str(run_dir), "reason": "cannot infer rainfall_mm"})
            if idx == 1 or idx % progress_every == 0 or idx == total_runs:
                progress(f"[Extract] {idx}/{total_runs} SKIP {run_dir.name}: cannot infer rainfall_mm", args.quiet)
            continue

        proc_files = sorted_process_files(run_dir, meta.output_step_s)
        final_files = sorted(run_dir.glob("final_*.asc"))
        final_path = final_files[0] if final_files else None
        if idx == 1 or idx % progress_every == 0 or idx == total_runs:
            rain = f"{meta.rainfall_mm:.3f}".rstrip("0").rstrip(".")
            duration = f"{meta.rainfall_duration_s}s" if meta.rainfall_duration_s is not None else "NA"
            progress(
                "[Extract] "
                f"{idx}/{total_runs} START {run_dir.name} "
                f"pattern={meta.pattern} scenario={meta.scenario_type} "
                f"rain={rain}mm duration={duration} process_files={len(proc_files)} final_files={len(final_files)}",
                args.quiet,
            )

        hmax = {pid: 0.0 for pid in geom}
        t_peak = {pid: None for pid in geom}
        start_mean_gt_avg = {pid: None for pid in geom}

        for depth_path, t in proc_files:
            depth = depth_to_m(clean_nodata(read_ascii_grid(depth_path)), args.depth_unit)
            maxes, means = max_and_mean_by_pit(depth, pit_masks)
            for pid in geom:
                cur = maxes[pid]
                if np.isfinite(cur) and cur > hmax[pid]:
                    hmax[pid] = cur
                    t_peak[pid] = t
                if start_mean_gt_avg[pid] is None and np.isfinite(means[pid]) and means[pid] > geom[pid]["Pit_Avg_Depth_m"]:
                    start_mean_gt_avg[pid] = t

        # ``Final_Inundation_Depth_m`` must always represent the terminal
        # water-depth field.  Some batch runs retain only process/*.asc and
        # omit final_*.asc; in that case use the latest process output rather
        # than the temporal maximum (Hmax_m), which is a different quantity.
        final_maxes: Dict[int, float]
        if final_path is not None:
            terminal_depth_path = final_path
        elif proc_files:
            terminal_depth_path = max(proc_files, key=lambda item: item[1])[0]
        else:
            terminal_depth_path = None

        if terminal_depth_path is not None:
            final_depth = depth_to_m(
                clean_nodata(read_ascii_grid(terminal_depth_path)), args.depth_unit
            )
            final_maxes, _ = max_and_mean_by_pit(final_depth, pit_masks)
        else:
            final_maxes = hmax

        sim_duration = meta.simulation_duration_s
        if sim_duration is None and proc_files:
            sim_duration = max(t for _, t in proc_files)

        for pid, g in geom.items():
            rain_tag = f"{meta.rainfall_mm:.3f}".rstrip("0").rstrip(".")
            run_id_parts = [f"pit{pid}", f"{rain_tag}mm", meta.pattern]
            if meta.rainfall_duration_s is not None:
                run_id_parts.append(f"{meta.rainfall_duration_s}s")
            if meta.return_period_yr is not None:
                rp_tag = f"{meta.return_period_yr:g}a"
                run_id_parts.append(rp_tag)
            if meta.chicago_peak_ratio is not None:
                run_id_parts.append(f"r{int(round(meta.chicago_peak_ratio * 100)):03d}")
            run_id = "_".join(run_id_parts)
            peak_ratio = hmax[pid] / g["Pit_Max_Depth_m"] if g["Pit_Max_Depth_m"] > 0 else np.nan

            rows.append(
                {
                    "Run_ID": run_id,
                    "Source_Run_Dir": str(run_dir),
                    "Pit_ID": pid,
                    "Rainfall_mm": meta.rainfall_mm,
                    "Rainfall_Pattern": meta.pattern,
                    "Rainfall_Duration_s": meta.rainfall_duration_s,
                    "Simulation_Duration_s": sim_duration,
                    "Scenario_Type": meta.scenario_type,
                    "Rain_File_Path": meta.rain_file_path,
                    "Rain_File_Total_mm": meta.rain_file_total_mm,
                    "Rain_File_Peak_Intensity_mm_h": meta.rain_file_peak_intensity_mm_h,
                    "Rain_File_Interval_s": meta.rain_file_interval_s,
                    "Design_Region": meta.design_region,
                    "Return_Period_yr": meta.return_period_yr,
                    "Chicago_Peak_Ratio": meta.chicago_peak_ratio,
                    **g,
                    "Final_Inundation_Depth_m": final_maxes[pid],
                    "Hmax_m": hmax[pid],
                    "T_to_Hmax_s": t_peak[pid] if t_peak[pid] is not None else -1,
                    "StartTime_MeanGT_AvgDepth_s": start_mean_gt_avg[pid] if start_mean_gt_avg[pid] is not None else -1,
                    "Peak_to_Potential_Ratio": peak_ratio,
                }
            )
        if idx == 1 or idx % progress_every == 0 or idx == total_runs:
            elapsed = time.monotonic() - t_run
            progress(
                f"[Extract] {idx}/{total_runs} DONE {run_dir.name} "
                f"elapsed={elapsed:.1f}s rows_total={len(rows)} skipped={len(skipped)}",
                args.quiet,
            )

    df = pd.DataFrame(rows)
    report = {
        "n_runs_discovered": len(run_dirs),
        "run_roots": [str(r) for r in run_roots],
        "n_runs_skipped": len(skipped),
        "n_rows": int(len(df)),
        "n_pits": int(df["Pit_ID"].nunique()) if not df.empty else 0,
        "n_scenarios": int(df["Source_Run_Dir"].nunique()) if not df.empty else 0,
        "elapsed_s": round(time.monotonic() - t_all, 3),
        "skipped": skipped,
    }
    progress(f"[Extract] Finished extraction in {report['elapsed_s']:.1f}s; rows={len(df)} skipped={len(skipped)}", args.quiet)
    return df, report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract pit-scale event CSV from hydrodynamic simulation outputs.")
    p.add_argument("--dsm", required=True, help="DSM ASCII grid aligned with pit mask")
    p.add_argument("--mask", required=True, help="Pit ID mask ASCII grid aligned with DSM")
    p.add_argument("--runs", required=True, nargs="+", help="One or more root directories containing run subdirectories")
    p.add_argument("--out", required=True, help="Output events CSV")
    p.add_argument("--seeds", default=None, help="Optional seeds/threshold CSV; if omitted, rim elevation is used")
    p.add_argument("--override-cellsize-m", type=float, default=None)
    p.add_argument("--depth-unit", choices=["mm", "m", "auto"], default="mm", help="Unit of output depth ASC files")
    p.add_argument("--recursive", action="store_true", help="Search INPUT_INFO recursively below --runs")
    p.add_argument("--max-runs", type=int, default=None, help="Debug option: process only first N runs")
    p.add_argument("--report", default=None, help="Optional extraction report JSON path")
    p.add_argument("--progress-every", type=int, default=1, help="Print progress every N run folders")
    p.add_argument("--quiet", action="store_true", help="Disable progress messages")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df, report = extract_events(args)
    df.to_csv(out, index=False)

    report_path = Path(args.report) if args.report else out.with_suffix(".report.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Extract] Wrote {len(df)} rows to {out}")
    print(f"[Extract] Report: {report_path}")


if __name__ == "__main__":
    main()
