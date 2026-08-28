#!/usr/bin/env bash
set -euo pipefail

# Required inputs are passed as environment variables, so no personal paths are
# stored in the public release. Example:
# DSM_PATH=/path/to/dsm.asc LC_PATH=/path/to/landcover.asc bash run_idealized_matrix.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXEC="${EXEC:-${SCRIPT_DIR}/run_Horton}"
if [[ ! -x "$EXEC" ]]; then
  echo "ERR: $EXEC 不存在或不可执行"
  exit 1
fi
EXEC_ABS="$(readlink -f "$EXEC")"

# === 固定输入（按你当前 1 网格设置） ===
ROW="${ROW:-1077}"
COL="${COL:-1517}"
RESOLUTION="${RESOLUTION:-1}"
DSM_PATH="${DSM_PATH:?Set DSM_PATH to an aligned DSM ASCII grid}"
LC_PATH="${LC_PATH:-NONE}"
STORAGE_PATH="${STORAGE_PATH:-NONE}"
TIME_ADAPTIVE="${TIME_ADAPTIVE:-ON}"
DT_USER="${DT_USER:-1}"
FDR_OUTSTEP="${FDR_OUTSTEP:-60}"
OUT_ROOT="${OUT_ROOT:-./runs_horton_5mm_3mode_3dur}"

# === Horton + 统一裸土参数 ===
INFILTRATION="${INFILTRATION:-ON}"
HORTON_F0="${HORTON_F0:-60}"
HORTON_FC="${HORTON_FC:-8}"
HORTON_K="${HORTON_K:-3}"
# 粗糙率模式：LANDCOVER 或 UNIFORM
ROUGHNESS_MODE="${ROUGHNESS_MODE:-LANDCOVER}"
MANNING_N_UNIFORM="${MANNING_N_UNIFORM:-0.025}"

# 绝对路径与存在性检查
DSM_ABS="$(readlink -f "$DSM_PATH")"
if [[ ! -f "$DSM_ABS" ]]; then
  echo "ERR: 找不到 DSM: $DSM_PATH"
  exit 1
fi
if [[ -f "$LC_PATH" ]]; then
  LC_ABS="$(readlink -f "$LC_PATH")"
else
  LC_ABS="NONE"
  echo "WARN: 找不到 LC 文件，LANDCOVER_PATH 将写 NONE"
fi

mkdir -p "$OUT_ROOT"
OUT_ROOT_ABS="$(readlink -f "$OUT_ROOT")"

# === 参数矩阵 ===
# 0mm 不跑，从 100mm 开始，每 5mm 递减到 5mm
RAINS_MM=($(seq 100 -5 5))
RAIN_MODES=(UNIFORM FRONT BACK)
DURATIONS_S=(1800 3600 7200)   # 0.5h, 1h, 2h

# === 并行设置 ===
if command -v nproc >/dev/null 2>&1; then
  TOTAL_THREADS="$(nproc --all)"
else
  TOTAL_THREADS=8
fi

# 默认目标：每情景 16 线程，并发 3~5 个情景
# 可通过环境变量覆盖，例如：TPJ=16 JOBS=4 ./run_horton_cases.sh
TPJ="${TPJ:-16}"             # 每个情景 OpenMP 线程数
JOBS_MIN="${JOBS_MIN:-3}"
JOBS_MAX="${JOBS_MAX:-5}"

if (( JOBS_MIN < 1 )); then JOBS_MIN=1; fi
if (( JOBS_MAX < JOBS_MIN )); then JOBS_MAX="$JOBS_MIN"; fi

MAX_JOBS_BY_CPU=$(( TOTAL_THREADS / TPJ ))
if (( MAX_JOBS_BY_CPU < 1 )); then MAX_JOBS_BY_CPU=1; fi

if [[ -n "${JOBS:-}" ]]; then
  if (( JOBS > MAX_JOBS_BY_CPU )); then
    echo "WARN: 指定 JOBS=$JOBS 超过 CPU 可承载上限，自动降为 $MAX_JOBS_BY_CPU"
    JOBS=$MAX_JOBS_BY_CPU
  fi
else
  JOBS=$MAX_JOBS_BY_CPU
  if (( JOBS > JOBS_MAX )); then JOBS=$JOBS_MAX; fi
  if (( JOBS < JOBS_MIN )); then
    echo "WARN: 当前 CPU 仅支持 JOBS=$JOBS (<$JOBS_MIN)，将按 $JOBS 运行"
  fi
fi

echo "TOTAL_THREADS=$TOTAL_THREADS -> $JOBS 并发情景, 每情景 OMP=$TPJ (目标区间 ${JOBS_MIN}-${JOBS_MAX})"

# === 生成任务列表 ===
TASKS_FILE="$(mktemp)"
trap 'rm -f "$TASKS_FILE"' EXIT
for MM in "${RAINS_MM[@]}"; do
  for MODE in "${RAIN_MODES[@]}"; do
    for DUR in "${DURATIONS_S[@]}"; do
      printf "%s\t%s\t%s\n" "$MM" "$MODE" "$DUR" >> "$TASKS_FILE"
    done
  done
done
echo "Total cases: $(wc -l < "$TASKS_FILE")"

# === 单任务函数 ===
run_one() {
  local MM="$1" MODE="$2" DUR="$3"
  local H tag rundir final_out
  H="$(awk -v mm="$MM" 'BEGIN{printf "%.3f", mm/1000.0}')"
  tag="$(printf "%03dmm_%s_%ds" "$MM" "$MODE" "$DUR")"
  rundir="$OUT_ROOT_ABS/$tag"
  mkdir -p "$rundir" "$rundir/process"

  final_out="$rundir/final_${MM}mm_${MODE}_${DUR}s.asc"
  if [[ -s "$final_out" ]]; then
    echo "[SKIP] $tag 已存在 $final_out"
    return 0
  fi

  cat > "$rundir/INPUT_INFO" <<EOF
ROW	$ROW
COL	$COL
TOTAL_TIME(s)	$DUR
DSM_PATH	$DSM_ABS
LANDCOVER_PATH	$LC_ABS
OUTPUT_PATH	$final_out
STORAGE_PATH	$STORAGE_PATH
WATER_DEPTH(m)	$H
TIME_STEP(s)	$DT_USER
TIME_ADAPTIVE	$TIME_ADAPTIVE
RESOLUTION	$RESOLUTION
FDR_OUTSTEP	$FDR_OUTSTEP
FDR_PATH	$rundir/process/
RAIN_MODE	$MODE
INFILTRATION	$INFILTRATION
HORTON_F0(mm/h)	$HORTON_F0
HORTON_FC(mm/h)	$HORTON_FC
HORTON_K(1/h)	$HORTON_K
ROUGHNESS_MODE	$ROUGHNESS_MODE
MANNING_N_UNIFORM	$MANNING_N_UNIFORM
EOF

  echo "[RUN ] $tag -> $rundir (OMP=$TPJ)"
  : > "$rundir/run.log"
  {
    echo "[$(date '+%F %T')] START $tag"
    echo "[$(date '+%F %T')] EXEC=$EXEC_ABS OMP_NUM_THREADS=$TPJ"
  } >> "$rundir/run.log"

  # 心跳日志，避免长算例阶段看不到任何输出
  (
    while true; do
      sleep 120
      echo "[$(date '+%F %T')] HEARTBEAT $tag still running" >> "$rundir/run.log"
    done
  ) &
  hb_pid=$!

  (
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
  ) >> "$rundir/run.log" 2>&1
  rc=$?

  kill "$hb_pid" 2>/dev/null || true
  wait "$hb_pid" 2>/dev/null || true

  echo "[$(date '+%F %T')] END $tag rc=$rc" >> "$rundir/run.log"
  if (( rc != 0 )); then
    return "$rc"
  fi

  if [[ -s "$final_out" ]]; then
    echo "[DONE] $tag -> $final_out"
  else
    echo "[FAIL] $tag（请看 $rundir/run.log）"
    return 1
  fi
}

export -f run_one
export OUT_ROOT_ABS EXEC_ABS ROW COL RESOLUTION DSM_ABS LC_ABS STORAGE_PATH TIME_ADAPTIVE DT_USER FDR_OUTSTEP TPJ
export INFILTRATION HORTON_F0 HORTON_FC HORTON_K ROUGHNESS_MODE MANNING_N_UNIFORM

# === 并发执行 ===
if command -v parallel >/dev/null 2>&1; then
  parallel -j "$JOBS" --colsep '\t' --bar \
    --joblog "$OUT_ROOT_ABS/joblog.tsv" --resume-failed \
    run_one {1} {2} {3} :::: "$TASKS_FILE"
else
  echo "WARN: 未找到 GNU parallel，回退到 Bash 原生并发执行"
  FAILS=0
  pids=()
  while IFS=$'\t' read -r MM MODE DUR; do
    run_one "$MM" "$MODE" "$DUR" &
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
    echo "ERR: 有 $FAILS 个任务失败，请检查各子目录 run.log"
    exit 1
  fi
fi

echo "All tasks finished. Joblog: $OUT_ROOT_ABS/joblog.tsv"
