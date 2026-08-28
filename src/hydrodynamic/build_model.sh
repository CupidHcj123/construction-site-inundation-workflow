#!/usr/bin/env bash
set -euo pipefail

CXX="${CXX:-g++}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-${SCRIPT_DIR}/Inertial_Adaptive_LimitQ_Chicago_fast.cpp}"
OUT="${OUT:-${SCRIPT_DIR}/run_Horton}"

"$CXX" -std=c++11 -O3 -fopenmp "$SRC" -o "$OUT"
chmod +x "$OUT"
echo "Built $OUT from $SRC"
