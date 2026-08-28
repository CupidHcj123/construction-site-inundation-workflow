#!/usr/bin/env python3
"""Split event-extraction output from the 15-case Horton sensitivity batch.

The batch script names run folders as:
  fc4_030mm_UNIFORM_1800s
  fc12_030mm_UNIFORM_1800s
  no_infiltration_030mm_UNIFORM_1800s

This utility uses Source_Run_Dir to retain the sensitivity-setting label and
writes one CSV per setting for compare_infiltration_sensitivity.py.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SETTINGS = ("fc4", "fc12", "no_infiltration")


def setting_from_path(value: object) -> str | None:
    name = Path(str(value)).name
    match = re.match(r"^(fc4|fc12|no_infiltration)_", name, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split extracted events from the 15-case Horton sensitivity batch."
    )
    parser.add_argument("--input", required=True, help="Combined event CSV from event_extraction.py")
    parser.add_argument("--outdir", required=True, help="Directory for events_fc4.csv, events_fc12.csv, and events_no_infiltration.csv")
    args = parser.parse_args()

    source = Path(args.input)
    outdir = Path(args.outdir)
    df = pd.read_csv(source)
    if "Source_Run_Dir" not in df.columns:
        raise ValueError(f"{source} has no Source_Run_Dir column")

    df = df.copy()
    df["Sensitivity_Setting"] = df["Source_Run_Dir"].map(setting_from_path)
    unknown = df.loc[df["Sensitivity_Setting"].isna(), "Source_Run_Dir"].drop_duplicates()
    if not unknown.empty:
        raise ValueError(
            "Could not identify a sensitivity setting from these run directories:\n"
            + "\n".join(unknown.head(10).astype(str))
        )

    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for setting in SETTINGS:
        subset = df.loc[df["Sensitivity_Setting"] == setting].drop(columns="Sensitivity_Setting")
        if subset.empty:
            raise ValueError(f"No rows found for setting '{setting}'")
        output = outdir / f"events_{setting}.csv"
        subset.to_csv(output, index=False)
        summary_rows.append(
            {
                "Setting": setting,
                "Rows": len(subset),
                "Pits": subset["Pit_ID"].nunique(),
                "Rainfall_totals": subset["Rainfall_mm"].nunique(),
                "Patterns": "; ".join(sorted(subset["Rainfall_Pattern"].astype(str).unique())),
                "Durations_s": "; ".join(
                    str(int(v)) for v in sorted(pd.to_numeric(subset["Rainfall_Duration_s"]).dropna().unique())
                ),
                "Output": str(output),
            }
        )
        print(f"Wrote {len(subset)} rows: {output}")

    summary = pd.DataFrame(summary_rows)
    summary_path = outdir / "sensitivity_event_split_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nWrote split summary: {summary_path}")


if __name__ == "__main__":
    main()
