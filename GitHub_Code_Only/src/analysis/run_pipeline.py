#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent


def resolve_path(value: str | None, base: Path) -> str | None:
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = base / path
    return str(path)


def load_config(path: str | Path) -> tuple[Dict[str, Any], Path]:
    cfg_path = Path(path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f), cfg_path.parent


def run_cmd(cmd: List[str], label: str) -> None:
    print(f"\n[{label}] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def bool_flag(cmd: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def extract_cmd(cfg: Dict[str, Any], base: Path) -> List[str]:
    ex = cfg["extract"]
    runs = ex["runs"]
    run_values = runs if isinstance(runs, list) else [runs]
    cmd = [
        sys.executable,
        str(ROOT / "event_extraction.py"),
        "--dsm",
        resolve_path(ex["dsm"], base),
        "--mask",
        resolve_path(ex["mask"], base),
        "--runs",
        *[resolve_path(v, base) for v in run_values],
        "--out",
        resolve_path(ex["out"], base),
        "--depth-unit",
        ex.get("depth_unit", "mm"),
    ]
    optional = {
        "--seeds": ex.get("seeds"),
        "--report": ex.get("report"),
    }
    for flag, value in optional.items():
        if value:
            cmd.extend([flag, resolve_path(value, base)])
    if ex.get("override_cellsize_m") is not None:
        cmd.extend(["--override-cellsize-m", str(ex["override_cellsize_m"])])
    if ex.get("max_runs") is not None:
        cmd.extend(["--max-runs", str(ex["max_runs"])])
    if ex.get("progress_every") is not None:
        cmd.extend(["--progress-every", str(ex["progress_every"])])
    bool_flag(cmd, "--recursive", bool(ex.get("recursive", False)))
    bool_flag(cmd, "--quiet", bool(ex.get("quiet", False)))
    return cmd


def step2_cmd(cfg: Dict[str, Any], base: Path) -> List[str]:
    s2 = cfg["step2"]
    cmd = [
        sys.executable,
        str(ROOT / "step2_pipeline.py"),
        "--input",
        resolve_path(s2["input"], base),
        "--outdir",
        resolve_path(s2["outdir"], base),
        "--n-splits",
        str(s2.get("n_splits", 5)),
        "--target-col",
        s2.get("target_col", "Final_Inundation_Depth_m"),
        "--spline-df-rf",
        str(s2.get("spline_df_rf", 5)),
        "--spline-df-rainfall",
        str(s2.get("spline_df_rainfall", 5)),
    ]
    if s2.get("sheet_name") is not None:
        cmd.extend(["--sheet-name", str(s2["sheet_name"])])
    if s2.get("lme_formula"):
        cmd.extend(["--lme-formula", s2["lme_formula"]])
    if s2.get("train_scenario_type"):
        cmd.extend(["--train-scenario-type", str(s2["train_scenario_type"])])
    if s2.get("external_scenario_type"):
        cmd.extend(["--external-scenario-type", str(s2["external_scenario_type"])])
    return cmd


def step3_cmd(cfg: Dict[str, Any], base: Path) -> List[str]:
    s3 = cfg["step3"]
    cmd = [
        sys.executable,
        str(ROOT / "step3_pipeline.py"),
        "--input",
        resolve_path(s3["input"], base),
        "--outdir",
        resolve_path(s3["outdir"], base),
        "--label-mode",
        s3.get("label_mode", "relative"),
        "--n-splits",
        str(s3.get("n_splits", 5)),
        "--inner-splits",
        str(s3.get("inner_splits", 3)),
        "--perm-repeats",
        str(s3.get("perm_repeats", 20)),
        "--n-jobs",
        str(s3.get("n_jobs", 1)),
        "--cost-fn",
        str(s3.get("cost_fn", 3.0)),
        "--cost-fp",
        str(s3.get("cost_fp", 1.0)),
        "--recall-floor",
        str(s3.get("recall_floor", 0.90)),
    ]
    models = s3.get("models", ["rf", "xgb"])
    cmd.extend(["--models", *models])
    if s3.get("rel_thresh"):
        cmd.extend(["--rel-thresh", *[str(v) for v in s3["rel_thresh"]]])
    if s3.get("abs_thresh"):
        cmd.extend(["--abs-thresh", *[str(v) for v in s3["abs_thresh"]]])
    if s3.get("sheet_name") is not None:
        cmd.extend(["--sheet-name", str(s3["sheet_name"])])
    if s3.get("train_scenario_type"):
        cmd.extend(["--train-scenario-type", str(s3["train_scenario_type"])])
    if s3.get("external_scenario_type"):
        cmd.extend(["--external-scenario-type", str(s3["external_scenario_type"])])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extraction, Step2 statistics, and Step3 ML from one JSON config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", choices=["all", "extract", "step2", "step3"], default="all")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-step2", action="store_true")
    parser.add_argument("--skip-step3", action="store_true")
    args = parser.parse_args()

    cfg, base = load_config(args.config)

    run_extract = args.only in {"all", "extract"} and not args.skip_extract
    run_step2 = args.only in {"all", "step2"} and not args.skip_step2
    run_step3 = args.only in {"all", "step3"} and not args.skip_step3

    if run_extract:
        run_cmd(extract_cmd(cfg, base), "Extract")
    if run_step2:
        run_cmd(step2_cmd(cfg, base), "Step2")
    if run_step3:
        run_cmd(step3_cmd(cfg, base), "Step3")


if __name__ == "__main__":
    main()
