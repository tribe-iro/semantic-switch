#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PACK_KIND="${PACK_KIND:-zt411}"
case "$PACK_KIND" in
  zt411|support) ;;
  *) echo "PACK_KIND must be zt411 or support" >&2; exit 2 ;;
esac
PACK="${PACK:-packs/$PACK_KIND}"
WORK="${WORK:-work/$PACK_KIND}"
KEY_FILE="${KEY_FILE:-$ROOT/../keys/openrouter.txt}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
REVISION="${REVISION:-3e22461f65e89153144f8adb70e3b8c2cc9845a7}"
MARGIN="${MARGIN:-0}"
EPOCHS="${EPOCHS:-4}"
MIN_EPOCHS="${MIN_EPOCHS:-}"
DEFINITION="${DEFINITION:-}"
EVAL_EVERY_EXAMPLES="${EVAL_EVERY_EXAMPLES:-}"
PROFILE="${PROFILE:-}"
PROFILE_STEPS="${PROFILE_STEPS:-}"
PROFILE_WARMUP="${PROFILE_WARMUP:-1}"
PROFILE_CADENCES="${PROFILE_CADENCES:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
MAX_MONITOR_TOKENS="${MAX_MONITOR_TOKENS:-}"
MAX_SECONDS="${MAX_SECONDS:-18000}"
FEASIBILITY_UPDATES="${FEASIBILITY_UPDATES:-8}"
FIT_MEMORY="${FIT_MEMORY:-22G}"
GUARD_GIB="${GUARD_GIB:-1.5}"
UNTIL="${UNTIL:-}"
ATTEMPT="${ATTEMPT:-}"
SMOKE="${SMOKE:-}"
REFERENCE="${REFERENCE:-}"
EVALUATION_BATCH_SIZE="${EVALUATION_BATCH_SIZE:-1}"
RESULTS="${RESULTS:-$WORK/results}"

PY="$ROOT/.venv/bin"
XPU="$ROOT/.venv-xpu/bin"
if [[ ! -f "$PACK/benchmark.json" ]]; then
  echo "training requires an existing published pack: $PACK/benchmark.json; corpus construction is a separate operation" >&2; exit 2
fi
mkdir -p "$WORK"

available_gib() { awk '/MemAvailable/ {printf "%.1f", $2 / 1048576}' /proc/meminfo; }
guarded() {
  "$@" &
  local pid=$!
  trap 'kill -TERM "$pid" 2>/dev/null || true; wait "$pid" || true; trap - INT TERM; return 130' INT TERM
  while kill -0 "$pid" 2>/dev/null; do
    if awk -v a="$(available_gib)" -v g="$GUARD_GIB" 'BEGIN {exit !(a < g)}'; then
      echo "MemAvailable below ${GUARD_GIB} GiB; stopping $1 to protect the desktop" >&2
      kill -TERM "$pid"; wait "$pid" || true; trap - INT TERM; return 99
    fi
    sleep 2
  done
  local result=0
  wait "$pid" || result=$?
  trap - INT TERM
  return "$result"
}
field() { "$PY/python" -c 'import functools, json, sys; print(functools.reduce(lambda v, k: v[k], sys.argv[2:], json.load(open(sys.argv[1]))))' "$@"; }

if [[ -e "$WORK/margin" && "$(cat "$WORK/margin")" != "$MARGIN" ]]; then
  echo "required improvement $(cat "$WORK/margin") was predeclared for $WORK; a different MARGIN needs a new WORK directory" >&2; exit 2
fi
echo "$MARGIN" > "$WORK/margin"

"$PY/python" -m bench.evidence stage "$PACK" --under "$WORK/stages" --binding "$WORK/stage.json" > /dev/null
CORPUS="$(field "$WORK/stage.json" manifest_sha256)"
STAGED="$(field "$WORK/stage.json" definition)"
LABELS="$(field "$WORK/stage.json" labels)"
EVIDENCE="$(field "$WORK/stage.json" evidence)"
STAGE="$(dirname "$STAGED")"
TASK="$STAGED"
if [[ -n "$DEFINITION" ]]; then
  TASK="$WORK/definitions/$("$PY/python" -c 'import sys; from semantic_switch import files; print(files.sha256_file(sys.argv[1])[:16])' "$DEFINITION")/$(basename "$DEFINITION")"
  mkdir -p "$(dirname "$TASK")"
  [[ -f "$TASK" ]] || cp "$DEFINITION" "$TASK"
fi

SPECIALIZATION="${SPECIALIZATION:-$WORK/specializations/$(basename "$STAGE")}"
SPECIALIZED="$WORK/specialization.json"
PROFILING=()
if [[ -n "$PROFILE_STEPS" ]]; then PROFILING+=(--profile-steps "$PROFILE_STEPS" --profile-warmup "$PROFILE_WARMUP"); fi
BUDGET_ARGS=(--budget "max_seconds=$MAX_SECONDS" --budget "feasibility_updates=$FEASIBILITY_UPDATES")
if [[ -n "$MAX_TOKENS" ]]; then BUDGET_ARGS+=(--budget "max_tokens=$MAX_TOKENS"); fi
if [[ -n "$MAX_MONITOR_TOKENS" ]]; then BUDGET_ARGS+=(--budget "max_monitor_tokens=$MAX_MONITOR_TOKENS"); fi
for cadence in $PROFILE_CADENCES; do PROFILING+=(--profile-cadence "$cadence"); done
SPECIALIZE_ARGS=("$PY/ss" --json specialize "$TASK" --examples "$LABELS" --to "$SPECIALIZATION" --trainer "$XPU/semantic-trainer"
  --device xpu:0 --native-device gpu --model "$MODEL" --revision "$REVISION" --quant Q8_0 --task-family "bench:$PACK_KIND" --fit-situations 100
  --recipe selection_unit=complete_situation --recipe seed=0 --recipe "epochs=$EPOCHS" --recipe "evaluation_batch_size=$EVALUATION_BATCH_SIZE"
  "${BUDGET_ARGS[@]}" "${PROFILING[@]}")
if [[ -n "$MIN_EPOCHS" ]]; then SPECIALIZE_ARGS+=(--recipe "min_epochs=$MIN_EPOCHS"); fi
if [[ -n "$PROFILE" ]]; then SPECIALIZE_ARGS+=(--profile "$PROFILE"); fi
if [[ -n "$EVAL_EVERY_EXAMPLES" ]]; then SPECIALIZE_ARGS+=(--cadence "$EVAL_EVERY_EXAMPLES"); fi
if [[ -n "$ATTEMPT" ]]; then SPECIALIZE_ARGS+=(--attempt "$ATTEMPT"); fi
status=0
if [[ "$UNTIL" == freeze ]]; then
  guarded "${SPECIALIZE_ARGS[@]}" --until freeze > "$SPECIALIZED"
  exit 0
fi
guarded "${SPECIALIZE_ARGS[@]}" --until assets > "$SPECIALIZED"
BASE="$(field "$SPECIALIZED" base path)"
guarded "$PY/python" -m bench.snapshots "$PACK_KIND" --pack "$PACK" --evidence "$EVIDENCE" --baseline "base=$BASE" \
  --role model_select --kind workflow --condition structured --corpus "$CORPUS" --native device=gpu --native threads=4 --margin "$MARGIN" --results "$RESULTS"
if [[ "$UNTIL" == assets ]]; then exit 0; fi
guarded systemd-run --user --scope -q -p MemoryMax="$FIT_MEMORY" -p MemorySwapMax=0 \
  "${SPECIALIZE_ARGS[@]}" ${UNTIL:+--until "$UNTIL"} > "$SPECIALIZED" || status=$?
if [[ ! -s "$SPECIALIZED" ]]; then echo "ss specialize failed before the trainer produced a result" >&2; exit "$status"; fi
STATE="$(field "$SPECIALIZED" state)"
case "$STATE" in
  completed) ;;
  awaiting_configuration)
    "$PY/python" -c 'import json, sys
for c in json.load(open(sys.argv[1]))["choices"]:
    print("  eval_every_examples=%(eval_every_examples)6d  points=%(evaluation_points)3d  monitor_tokens=%(monitor_tokens)10d  seconds=%(seconds)8.0f  feasible=%(feasible)s" % c)' "$SPECIALIZED"
    echo "choose the monitoring cadence from this measured profile and rerun with EVAL_EVERY_EXAMPLES=N" >&2; exit 3 ;;
  stopped) echo "stopped after $(field "$SPECIALIZED" stage): $SPECIALIZED"; exit 0 ;;
  *) echo "specialization $STATE at $(field "$SPECIALIZED" stage): $(field "$SPECIALIZED" error)" >&2; exit "$(( status ? status : 1 ))" ;;
esac
RECEIPT="$(field "$SPECIALIZED" receipt)"
BASE="$(field "$SPECIALIZED" base path)"

ARMS=(--baseline base="$BASE" --specialist specialist="$RECEIPT")
REFERENCE_ARGS=()
if [[ -n "$REFERENCE" ]]; then
  REFERENCE_ARGS+=(--reference "$REFERENCE")
  if [[ -n "$KEY_FILE" ]]; then REFERENCE_ARGS+=(--key-file "$KEY_FILE"); fi
fi
if [[ -n "$SMOKE" ]]; then
  guarded "$PY/python" -m bench.snapshots "$PACK_KIND" --pack "$PACK" --evidence "$EVIDENCE" "${ARMS[@]}" --role model_select --kind workflow \
    --corpus "$CORPUS" --native device=gpu --native threads=4 --margin "$MARGIN" --results "$RESULTS"
fi

guarded "$PY/python" -m bench.snapshots "$PACK_KIND" --pack "$PACK" --evidence "$EVIDENCE" "${ARMS[@]}" \
  "${REFERENCE_ARGS[@]}" --condition structured --kind workflow --corpus "$CORPUS" --native device=gpu --native threads=4 --margin "$MARGIN" --results "$RESULTS" --check "$WORK/checks"

echo "== finished: specialization $SPECIALIZED; comparison and derived check are named by the experiment plan in $RESULTS and $WORK/checks"
echo "corpus $CORPUS review status: $(field "$WORK/stage.json" review)"
