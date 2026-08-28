#!/usr/bin/env python3
import argparse
import math
from pathlib import Path


def read_asc(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = [next(f).strip() for _ in range(6)]
        vals = []
        for line in f:
            if line.strip():
                vals.extend(float(x) for x in line.split())
    meta = {}
    for row in header:
        parts = row.split()
        if len(parts) >= 2:
            meta[parts[0].lower()] = parts[1]
    ncols = int(float(meta.get("ncols", 0)))
    nrows = int(float(meta.get("nrows", 0)))
    cellsize = float(meta.get("cellsize", 1.0))
    nodata = float(meta.get("nodata_value", -9999))
    if ncols and nrows and len(vals) != ncols * nrows:
        raise ValueError(f"{path}: expected {ncols*nrows} values, got {len(vals)}")
    return header, vals, cellsize, nodata


def main():
    ap = argparse.ArgumentParser(description="Compare two final ESRI ASCII grids in mm.")
    ap.add_argument("base", type=Path, help="trusted/original final ASC")
    ap.add_argument("test", type=Path, help="new/experimental final ASC")
    ap.add_argument("--wet-threshold-mm", type=float, default=1e-6)
    args = ap.parse_args()

    _, a, cell_a, nodata_a = read_asc(args.base)
    _, b, cell_b, nodata_b = read_asc(args.test)
    if len(a) != len(b):
        raise ValueError(f"grid sizes differ: {len(a)} vs {len(b)}")
    if abs(cell_a - cell_b) > 1e-12:
        raise ValueError(f"cellsize differs: {cell_a} vs {cell_b}")

    n = 0
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    base_wet = 0
    test_wet = 0
    changed_wet = 0
    vol_base_m3 = 0.0
    vol_test_m3 = 0.0
    cell_area = cell_a * cell_a

    for va, vb in zip(a, b):
        if va == nodata_a or vb == nodata_b:
            continue
        d = vb - va
        ad = abs(d)
        n += 1
        max_abs = max(max_abs, ad)
        sum_abs += ad
        sum_sq += d * d
        aw = va > args.wet_threshold_mm
        bw = vb > args.wet_threshold_mm
        base_wet += int(aw)
        test_wet += int(bw)
        changed_wet += int(aw != bw)
        vol_base_m3 += va / 1000.0 * cell_area
        vol_test_m3 += vb / 1000.0 * cell_area

    mae = sum_abs / n if n else 0.0
    rmse = math.sqrt(sum_sq / n) if n else 0.0
    print(f"cells_compared={n}")
    print(f"max_abs_diff_mm={max_abs:.12g}")
    print(f"mae_mm={mae:.12g}")
    print(f"rmse_mm={rmse:.12g}")
    print(f"base_wet_cells={base_wet}")
    print(f"test_wet_cells={test_wet}")
    print(f"changed_wet_cells={changed_wet}")
    print(f"base_volume_m3={vol_base_m3:.12g}")
    print(f"test_volume_m3={vol_test_m3:.12g}")
    print(f"volume_diff_m3={vol_test_m3 - vol_base_m3:.12g}")
    if vol_base_m3:
        print(f"volume_diff_pct={(vol_test_m3 - vol_base_m3) / vol_base_m3 * 100.0:.12g}")


if __name__ == "__main__":
    main()
