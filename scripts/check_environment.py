#!/usr/bin/env python3
"""Minimal import/API check for the xiongan analysis environment."""

from __future__ import annotations

import importlib
import json
import platform
import sys


PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    "patsy",
    "matplotlib",
    "joblib",
    "openpyxl",
    "xgboost",
    "rasterio",
]


def version_of(package_name: str) -> str:
    module = importlib.import_module(package_name)
    return str(getattr(module, "__version__", "unknown"))


def check_sklearn_api() -> None:
    from sklearn.preprocessing import OneHotEncoder

    OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def main() -> int:
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
    }
    failures = []

    for package_name in PACKAGES:
        try:
            report["packages"][package_name] = version_of(package_name)
        except Exception as exc:
            failures.append(f"{package_name}: {exc}")

    try:
        check_sklearn_api()
    except Exception as exc:
        failures.append(f"sklearn API check: {exc}")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
