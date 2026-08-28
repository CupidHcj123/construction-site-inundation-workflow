#!/usr/bin/env bash
set -euo pipefail

# Run near-realistic design storms with a Beijing DB11/T 969 Chicago hyetograph.
# Default: several return-period storms. To scale the same shape to fixed depths:
#   TOTAL_MMS_STR="50 75 100" RETURN_PERIODS_STR="50" ./matrix_design_rain.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXEC="${EXEC:-${SCRIPT_DIR}/run_Horton}"
GEN="${GEN:-${SCRIPT_DIR}/generate_design_rain.py}"
if [[ ! -x "$EXEC" ]]; then
  echo "ERR: $EXEC does not exist or is not executable"
  exit 1
fi
if [[ ! -f "$GEN" ]]; then
  echo "ERR: $GEN does not exist"
  exit 1
fi
EXEC_ABS="$(readlink -f "$EXEC")"
GEN_ABS="$(readlink -f "$GEN")"

ROW="${ROW:-1077}"
COL="${COL:-1517}"
RESOLUTION="${RESOLUTION:-1}"
DSM_PATH="${DSM_PATH:?Set DSM_PATH to an aligned DSM ASCII grid}"
LC_PATH="${LC_PATH:-NONE}"
STORAGE_PATH="${STORAGE_PATH:-NONE}"
TIME_ADAPTIVE="${TIME_ADAPTIVE:-ON}"
DT_USER="${DT_USER:-1}"
FDR_OUTSTEP="${FDR_OUTSTEP:-60}"
OUT_ROOT="${OUT_ROOT:-./runs_horton_design_rain}"

INFILTRATION="${INFILTRATION:-ON}"
HORTON_F0="${HORTON_F0:-60}"
HORTON_FC="${HORTON_FC:-8}"
HORTON_K="${HORTON_K:-3}"
ROUGHNESS_MODE="${ROUGHNESS_MODE:-LANDCOVER}"
MANNING_N_UNIFORM="${MANNING_N_UNIFORM:-0.025}"

REGION="${REGION:-beijing-II}"
RETURN_PERIODS_STR="${RETURN_PERIODS_STR:-20 50 100}"
DURATIONS_S_STR="${DURATIONS_S_STR:-1800 3600 7200}"
PEAK_RATIOS_STR="${PEAK_RATIOS_STR:-0.40}"
INTERVAL_S="${INTERVAL_S:-300}"
TOTAL_MMS_STR="${TOTAL_MMS_STR:-}"

DSM_ABS="$(readlink -f "$DSM_PATH")"
if [[ ! -f "$DSM_ABS" ]]; then
  echo "ERR: cannot find DSM: $DSM_PATH"
  exit 1
fi
if [[ -f "$LC_PATH" ]]; then
  LC_ABS="$(readlink -f "$LC_PATH")"
else
  LC_ABS="NONE"
  echo "WARN: cannot find LC file, LANDCOVER_PATH will be NONE"
fi

mkdir -p "$OUT_ROOT"
OUT_ROOT_ABS="$(readlink -f "$OUT_ROOT")"

if command -v nproc >/dev/null 2>&1; then
  TOTAL_THREADS="$(nproc --all)"
else
  TOTAL_THREADS=8
fi

TPJ="${TPJ:-16}"
JOBS_MIN="${JOBS_MIN:-3}"
JOBS_MAX="${JOBS_MAX:-5}"
if (( JOBS_MIN < 1 )); then JOBS_MIN=1; fi
if (( JOBS_MAX < JOBS_MIN )); then JOBS_MAX="$JOBS_MIN"; fi
MAX_JOBS_BY_CPU=$(( TOTAL_THREADS / TPJ ))
if (( MAX_JOBS_BY_CPU < 1 )); then MAX_JOBS_BY_CPU=1; fi

if [[ -n "${JOBS:-}" ]]; then
  if (( JOBS > MAX_JOBS_BY_CPU )); then
    echo "WARN: specified JOBS=$JOBS exceeds CPU capacity, reduced to $MAX_JOBS_BY_CPU"
    JOBS=$MAX_JOBS_BY_CPU
  fi
else
  JOBS=$MAX_JOBS_BY_CPU
  if (( JOBS > JOBS_MAX )); then JOBS=$JOBS_MAX; fi
  if (( JOBS < JOBS_MIN )); then
    echo "WARN: current CPU only supports JOBS=$JOBS (<$JOBS_MIN), running with $JOBS"
  fi
fi

echo "TOTAL_THREADS=$TOTAL_THREADS -> $JOBS concurrent design storms, OMP=$TPJ (target ${JOBS_MIN}-${JOBS_MAX})"
echo "REGION=$REGION RETURN_PERIODS=[$RETURN_PERIODS_STR] DURATIONS=[$DURATIONS_S_STR] PEAK_RATIOS=[$PEAK_RATIOS_STR] TOTAL_MMS=[$TOTAL_MMS_STR]"

TASKS_FILE="$(mktemp)"
trap 'rm -f "$TASKS_FILE"' EXIT

if [[ -n "$TOTAL_MMS_STR" ]]; then
  for MM in $TOTAL_MMS_STR; do
    for RP in $RETURN_PERIODS_STR; do
      for DUR in $DURATIONS_S_STR; do
        for RATIO in $PEAK_RATIOS_STR; do
          printf "%s\t%s\t%s\t%s\n" "$RP" "$DUR" "$RATIO" "$MM" >> "$TASKS_FILE"
        done
      done
    done
  done
else
  for RP in $RETURN_PERIODS_STR; do
    for DUR in $DURATIONS_S_STR; do
      for RATIO in $PEAK_RATIOS_STR; do
        printf "%s\t%s\t%s\tNA\n" "$RP" "$DUR" "$RATIO" >> "$TASKS_FILE"
      done
    done
  done
fi
echo "Total design cases: $(wc -l < "$TASKS_FILE")"

run_one() {
  local RP="$1" DUR="$2" RATIO="$3" TOTAL_MM="$4"
  local ratio_tag tag rundir rain_file final_out
  local -a total_arg

  ratio_tag="$(awk -v r="$RATIO" 'BEGIN{printf "r%03d", int(r*100+0.5)}')"
  if [[ "$TOTAL_MM" == "NA" ]]; then
    tag="$(printf "%s_%03da_%ds_%s" "$REGION" "$RP" "$DUR" "$ratio_tag")"
    total_arg=()
  else
    tag="$(printf "%s_%03dmm_shape%03da_%ds_%s" "$REGION" "$TOTAL_MM" "$RP" "$DUR" "$ratio_tag")"
    total_arg=(--total-mm "$TOTAL_MM")
  fi

  rundir="$OUT_ROOT_ABS/$tag"
  mkdir -p "$rundir" "$rundir/process"
  rain_file="$rundir/rain.tsv"
  final_out="$rundir/final_${tag}.asc"

  if [[ -s "$final_out" ]]; then
    echo "[SKIP] $tag already exists"
    return 0
  fi

  python3 "$GEN_ABS" \
    --output "$rain_file" \
    --region "$REGION" \
    --return-period "$RP" \
    --duration-s "$DUR" \
    --interval-s "$INTERVAL_S" \
    --peak-ratio "$RATIO" \
    "${total_arg[@]}" \
    > "$rundir/rain_generator.log" 2>&1

  cat > "$rundir/INPUT_INFO" <<EOF
ROW	$ROW
COL	$COL
TOTAL_TIME(s)	$DUR
DSM_PATH	$DSM_ABS
LANDCOVER_PATH	$LC_ABS
OUTPUT_PATH	$final_out
STORAGE_PATH	$STORAGE_PATH
WATER_DEPTH(m)	0
TIME_STEP(s)	$DT_USER
TIME_ADAPTIVE	$TIME_ADAPTIVE
RESOLUTION	$RESOLUTION
FDR_OUTSTEP	$FDR_OUTSTEP
FDR_PATH	$rundir/process/
RAIN_MODE	FILE
RAIN_FILE_PATH	$rain_file
INFILTRATION	$INFILTRATION
HORTON_F0(mm/h)	$HORTON_F0
HORTON_FC(mm/h)	$HORTON_FC
HORTON_K(1/h)	$HORTON_K
ROUGHNESS_MODE	$ROUGHNESS_MODE
MANNING_N_UNIFORM	$MANNING_N_UNIFORM
EOF

  echo "[RUN ] $tag -> $rundir (OMP=$TPJ)"
  (
    cd "$rundir"
    export OMP_NUM_THREADS="$TPJ"
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
    export OMP_WAIT_POLICY=PASSIVE
    nice -n 10 "$EXEC_ABS"
  ) > "$rundir/run.log" 2>&1

  if [[ -s "$final_out" ]]; then
    echo "[DONE] $tag -> $final_out"
  else
    echo "[FAIL] $tag, see $rundir/run.log"
    return 1
  fi
}

export -f run_one
export OUT_ROOT_ABS EXEC_ABS GEN_ABS ROW COL RESOLUTION DSM_ABS LC_ABS STORAGE_PATH TIME_ADAPTIVE DT_USER FDR_OUTSTEP TPJ
export INFILTRATION HORTON_F0 HORTON_FC HORTON_K ROUGHNESS_MODE MANNING_N_UNIFORM REGION INTERVAL_S

if command -v parallel >/dev/null 2>&1; then
  parallel -j "$JOBS" --colsep '\t' --bar \
    --joblog "$OUT_ROOT_ABS/joblog.tsv" --resume-failed \
    run_one {1} {2} {3} {4} :::: "$TASKS_FILE"
else
  echo "WARN: GNU parallel not found, using Bash native concurrency"
  FAILS=0
  pids=()
  while IFS=$'\t' read -r RP DUR RATIO TOTAL_MM; do
    run_one "$RP" "$DUR" "$RATIO" "$TOTAL_MM" &
    pids+=("$!")
    while (( $(jobs -rp | wc -l) >= JOBS )); do
      sleep 1
    done
  done < "$TASKS_FILE"

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      FAILS=$((FAILS+1))
    fi
  done

  if (( FAILS > 0 )); then
    echo "ERR: $FAILS design cases failed"
    exit 1
  fi
fi

echo "All design-rain tasks finished. Joblog: $OUT_ROOT_ABS/joblog.tsv"
