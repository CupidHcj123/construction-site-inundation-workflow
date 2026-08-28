#!/usr/bin/env bash
# Focused Horton-infiltration sensitivity experiment:
# 3 settings x 5 rainfall totals, all under 1800 s UNIFORM rainfall.
set -euo pipefail

# ---------- Model and fixed numerical inputs ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXEC="${EXEC:-${SCRIPT_DIR}/../hydrodynamic/run_Horton}"
ROW="${ROW:-1077}"
COL="${COL:-1517}"
RESOLUTION="${RESOLUTION:-1}"
DSM_PATH="${DSM_PATH:?Set DSM_PATH to an aligned DSM ASCII grid}"
LC_PATH="${LC_PATH:-NONE}"
STORAGE_PATH="${STORAGE_PATH:-NONE}"
TIME_ADAPTIVE="${TIME_ADAPTIVE:-ON}"
DT_USER="${DT_USER:-1}"
FDR_OUTSTEP="${FDR_OUTSTEP:-60}"
ROUGHNESS_MODE="${ROUGHNESS_MODE:-LANDCOVER}"
MANNING_N_UNIFORM="${MANNING_N_UNIFORM:-0.025}"

# Keep these fixed across every setting.  Only fc or the infiltration switch changes.
HORTON_F0="${HORTON_F0:-60}"
HORTON_K="${HORTON_K:-3}"
DURATION_S="${DURATION_S:-1800}"
RAIN_MODE="${RAIN_MODE:-UNIFORM}"
RAINS_MM=(${RAINS_MM_STR:-30 40 50 60 70})
SETTINGS=(${SETTINGS_STR:-fc4 fc12 no_infiltration})
OUT_ROOT="${OUT_ROOT:-./runs_horton_infiltration_sensitivity_1800s_uniform}"

# Eight OpenMP threads per simulation.  JOBS may be supplied explicitly,
# otherwise the script uses only as many concurrent simulations as the host can support.
TPJ="${TPJ:-8}"

if [[ ! -x "$EXEC" ]]; then
  echo "ERR: executable not found or not executable: $EXEC" >&2
  exit 1
fi

EXEC_ABS="$(readlink -f "$EXEC")"
DSM_ABS="$(readlink -f "$DSM_PATH")"
if [[ ! -f "$DSM_ABS" ]]; then
  echo "ERR: DSM not found: $DSM_PATH" >&2
  exit 1
fi

if [[ -f "$LC_PATH" ]]; then
  LC_ABS="$(readlink -f "$LC_PATH")"
else
  LC_ABS="NONE"
  echo "WARN: land-cover grid not found; model will fall back to uniform Manning n." >&2
fi

mkdir -p "$OUT_ROOT"
OUT_ROOT_ABS="$(readlink -f "$OUT_ROOT")"

if command -v nproc >/dev/null 2>&1; then
  TOTAL_THREADS="$(nproc --all)"
else
  TOTAL_THREADS=8
fi
MAX_JOBS_BY_CPU=$(( TOTAL_THREADS / TPJ ))
if (( MAX_JOBS_BY_CPU < 1 )); then MAX_JOBS_BY_CPU=1; fi
JOBS="${JOBS:-$MAX_JOBS_BY_CPU}"
if (( JOBS > MAX_JOBS_BY_CPU )); then JOBS="$MAX_JOBS_BY_CPU"; fi

echo "Sensitivity cases: ${#SETTINGS[@]} settings x ${#RAINS_MM[@]} rainfall totals = $(( ${#SETTINGS[@]} * ${#RAINS_MM[@]} ))"
echo "Host threads: $TOTAL_THREADS; concurrent jobs: $JOBS; OpenMP threads per job: $TPJ"
echo "Output root: $OUT_ROOT_ABS"

setting_parameters() {
  local setting="$1"
  case "$setting" in
    fc4)
      INFILTRATION="ON"
      HORTON_FC=4
      ;;
    fc12)
      INFILTRATION="ON"
      HORTON_FC=12
      ;;
    no_infiltration)
      INFILTRATION="OFF"
      HORTON_FC=8  # Recorded for traceability; ignored by the model when infiltration is OFF.
      ;;
    *)
      echo "ERR: unknown sensitivity setting: $setting" >&2
      return 2
      ;;
  esac
}

run_one() {
  local setting="$1"
  local mm="$2"
  local h tag rundir final_out logfile hb_pid rc
  local INFILTRATION HORTON_FC

  setting_parameters "$setting"
  h="$(awk -v mm="$mm" 'BEGIN { printf "%.3f", mm / 1000.0 }')"
  tag="${setting}_$(printf '%03d' "$mm")mm_${RAIN_MODE}_${DURATION_S}s"
  rundir="$OUT_ROOT_ABS/$tag"
  final_out="$rundir/final_${mm}mm_${RAIN_MODE}_${DURATION_S}s.asc"
  logfile="$rundir/run.log"
  mkdir -p "$rundir/process"

  if [[ -s "$final_out" ]]; then
    echo "[SKIP] $tag"
    return 0
  fi

  cat > "$rundir/INPUT_INFO" <<EOF
ROW	$ROW
COL	$COL
TOTAL_TIME(s)	$DURATION_S
DSM_PATH	$DSM_ABS
LANDCOVER_PATH	$LC_ABS
OUTPUT_PATH	$final_out
STORAGE_PATH	$STORAGE_PATH
WATER_DEPTH(m)	$h
TIME_STEP(s)	$DT_USER
TIME_ADAPTIVE	$TIME_ADAPTIVE
RESOLUTION	$RESOLUTION
FDR_OUTSTEP	$FDR_OUTSTEP
FDR_PATH	$rundir/process/
RAIN_MODE	$RAIN_MODE
INFILTRATION	$INFILTRATION
HORTON_F0(mm/h)	$HORTON_F0
HORTON_FC(mm/h)	$HORTON_FC
HORTON_K(1/h)	$HORTON_K
ROUGHNESS_MODE	$ROUGHNESS_MODE
MANNING_N_UNIFORM	$MANNING_N_UNIFORM
EOF

  echo "[RUN ] $tag (OMP=$TPJ)"
  {
    echo "[$(date '+%F %T')] START $tag"
    echo "setting=$setting infiltration=$INFILTRATION f0=$HORTON_F0 fc=$HORTON_FC k=$HORTON_K"
  } > "$logfile"

  (
    while true; do
      sleep 120
      echo "[$(date '+%F %T')] HEARTBEAT $tag still running" >> "$logfile"
    done
  ) &
  hb_pid=$!

  if (
    cd "$rundir"
    export OMP_NUM_THREADS="$TPJ"
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
    export OMP_WAIT_POLICY=PASSIVE
    if command -v stdbuf >/dev/null 2>&1; then
      nice -n 10 stdbuf -oL -eL "$EXEC_ABS"
    else
      nice -n 10 "$EXEC_ABS"
    fi
  ) >> "$logfile" 2>&1; then
    rc=0
  else
    rc=$?
  fi

  kill "$hb_pid" 2>/dev/null || true
  wait "$hb_pid" 2>/dev/null || true
  echo "[$(date '+%F %T')] END $tag rc=$rc" >> "$logfile"

  if (( rc != 0 )); then
    echo "[FAIL] $tag; see $logfile" >&2
    return "$rc"
  fi
  if [[ ! -s "$final_out" ]]; then
    echo "[FAIL] $tag finished without $final_out" >&2
    return 1
  fi
  echo "[DONE] $tag"
}

TASKS_FILE="$(mktemp)"
trap 'rm -f "$TASKS_FILE"' EXIT
for setting in "${SETTINGS[@]}"; do
  for mm in "${RAINS_MM[@]}"; do
    printf '%s\t%s\n' "$setting" "$mm" >> "$TASKS_FILE"
  done
done

export -f setting_parameters run_one
export OUT_ROOT_ABS EXEC_ABS ROW COL RESOLUTION DSM_ABS LC_ABS STORAGE_PATH TIME_ADAPTIVE DT_USER FDR_OUTSTEP
export HORTON_F0 HORTON_K DURATION_S RAIN_MODE ROUGHNESS_MODE MANNING_N_UNIFORM TPJ

if command -v parallel >/dev/null 2>&1; then
  parallel -j "$JOBS" --colsep '\t' --bar \
    --joblog "$OUT_ROOT_ABS/joblog.tsv" run_one {1} {2} :::: "$TASKS_FILE"
else
  echo "WARN: GNU parallel unavailable; using Bash job control." >&2
  pids=()
  while IFS=$'\t' read -r setting mm; do
    run_one "$setting" "$mm" &
    pids+=("$!")
    while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 1; done
  done < "$TASKS_FILE"

  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
  done
  if (( failed > 0 )); then
    echo "ERR: $failed sensitivity task(s) failed." >&2
    exit 1
  fi
fi

echo "All 15 sensitivity tasks completed: $OUT_ROOT_ABS"
