#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PACK_KIND="${PACK_KIND:-zt411}"
case "$PACK_KIND" in
  zt411) BUILDER=bench.procedures; DEFINITION=printer.yaml; SOURCES=(bench/procedures.py bench/evidence.py bench/data/zt411) ;;
  support) BUILDER=bench.support; DEFINITION=support.yaml; SOURCES=(bench/support.py bench/evidence.py bench/data/support) ;;
  *) echo "PACK_KIND must be zt411 or support" >&2; exit 2 ;;
esac
STEM="${DEFINITION%.yaml}"
PACK="${PACK:-packs/$PACK_KIND}"
WORK="${WORK:-work/$PACK_KIND}"
REVIEWS="${REVIEWS:-}"
KEY_FILE="${KEY_FILE:-$ROOT/../keys/openrouter.txt}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
REVISION="${REVISION:-3e22461f65e89153144f8adb70e3b8c2cc9845a7}"
MARGIN="${MARGIN:-0}"
EPOCHS="${EPOCHS:-4}"
EVAL_EVERY_EXAMPLES="${EVAL_EVERY_EXAMPLES:-}"
PROFILE="${PROFILE:-}"
PROFILE_STEPS="${PROFILE_STEPS:-3}"
PROFILE_WARMUP="${PROFILE_WARMUP:-1}"
PROFILE_CADENCES="${PROFILE_CADENCES:-}"
PROVISIONAL_CADENCE=1000000
MAX_TOKENS="${MAX_TOKENS:-2000000}"
MAX_MONITOR_TOKENS="${MAX_MONITOR_TOKENS:-20000000}"
MAX_SECONDS="${MAX_SECONDS:-18000}"
FEASIBILITY_UPDATES="${FEASIBILITY_UPDATES:-8}"
FIT_MEMORY="${FIT_MEMORY:-22G}"
FIT_MIN_AVAILABLE_GIB="${FIT_MIN_AVAILABLE_GIB:-16}"
GUARD_GIB="${GUARD_GIB:-1.5}"
UNTIL="${UNTIL:-}"

PY="$ROOT/.venv/bin"
XPU="$ROOT/.venv-xpu/bin"
DONE="$WORK/done"
LOGS="$WORK/logs"
mkdir -p "$DONE" "$LOGS"

available_gib() { awk '/MemAvailable/ {printf "%.1f", $2 / 1048576}' /proc/meminfo; }

guarded() {
  "$@" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if awk -v a="$(available_gib)" -v g="$GUARD_GIB" 'BEGIN {exit !(a < g)}'; then
      echo "MemAvailable below ${GUARD_GIB} GiB; stopping $1 to protect the desktop" >&2
      kill -TERM "$pid"; wait "$pid" || true; return 99
    fi
    sleep 2
  done
  wait "$pid"
}

identity() { printf '%s\n' "$@" | sha256sum | cut -c1-16; }
tree() { find "$@" -type f -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16; }
sha() { sha256sum "$1" | cut -c1-16; }
fresh() { if [[ -e "$1" ]]; then mv "$1" "$1.interrupted.$(date +%s)"; fi; }
require_memory() {
  local free; free=$(available_gib)
  if awk -v a="$free" -v m="$FIT_MIN_AVAILABLE_GIB" 'BEGIN {exit !(a < m)}'; then
    echo "$1 needs ${FIT_MIN_AVAILABLE_GIB} GiB available (the iGPU shares host memory); ${free} GiB now. Close browsers/apps and rerun." >&2; return 1
  fi
}
scoped() { guarded systemd-run --user --scope -q -p MemoryMax="$FIT_MEMORY" -p MemorySwapMax=0 "$@"; }

step() {
  local name=$1 id=$2; shift 2
  if [[ "$(cat "$DONE/$name" 2>/dev/null)" == "$id" ]]; then echo "== $name $id: done"; return; fi
  echo "== $name $id ($(date +%H:%M:%S), MemAvailable $(available_gib) GiB)"
  "$@" 2>&1 | tee "$LOGS/$name-$id.log"
  echo "$id" > "$DONE/$name"
  if [[ "$UNTIL" == "$name" ]]; then echo "stopped after $name"; exit 0; fi
}

json() { "$PY/python" -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {'d': d}))" "$@"; }

if [[ -e "$WORK/margin" && "$(cat "$WORK/margin")" != "$MARGIN" ]]; then
  echo "required improvement $(cat "$WORK/margin") was predeclared for $WORK; a different MARGIN needs a new WORK directory" >&2; exit 2
fi
echo "$MARGIN" > "$WORK/margin"

pack() { "$PY/python" -m "$BUILDER" pack --output "$PACK" ${REVIEWS:+--reviews "$REVIEWS"}; }
step pack "$(identity "$(tree "${SOURCES[@]}")" "${REVIEWS:+$(sha "$REVIEWS")}")" pack

STAGE="$WORK/stages/$(sha "$PACK/benchmark.json")"
LABELS="$STAGE/$STEM.test.jsonl"
EVIDENCE="$STAGE/$STEM.dataset.json"
stage() { fresh "$STAGE"; "$PY/python" -m bench.evidence stage "$PACK" --to "$STAGE"; }
step stage "$(basename "$STAGE")" stage

ASSETS="$WORK/assets-$(identity "$MODEL" Q8_0 "$REVISION").json"
assets() { "$PY/semantic-trainer" assets --model "$MODEL" --quant Q8_0 --revision "$REVISION" | tail -n 1 > "$ASSETS"; cat "$ASSETS"; }
step assets "$(basename "$ASSETS" .json)" assets
base() { "$PY/python" -c "import json,sys; m=json.load(open(sys.argv[1] + '/manifests/' + sys.argv[2] + '.json')); print(sys.argv[1] + '/' + m['gguf']['Q8_0']['path'])" \
  "$(json "$ASSETS" "d['root']")" "$(json "$ASSETS" "d['manifest']")"; }

PRESENTATION="$("$PY/python" -c "import json; from semantic_switch.presentation import identity; print(json.dumps(identity(), sort_keys=True))")"
EXPORT="$WORK/exports/$(identity "$(basename "$STAGE")" "$(json "$ASSETS" "d['manifest']")" "$PRESENTATION")"
export_native() {
  fresh "$EXPORT"; mkdir -p "$EXPORT"
  "$PY/ss" --judge "$(base)" --device gpu export "$STAGE/$DEFINITION" --native --task-family "bench:$PACK_KIND" --labels "$LABELS" --to "$EXPORT/$STEM.export.jsonl"
}
step export "$(basename "$EXPORT")" export_native

CONTRACT="$("$XPU/python" -c "import json; from semantic_trainer.jobs import FAMILIES, FITTING, IDENTITY, LOSS, PROFILE_SCHEMA, RESOLVED_SCHEMA, SCHEMA; print(json.dumps([LOSS, SCHEMA, RESOLVED_SCHEMA, FAMILIES, FITTING, IDENTITY, PROFILE_SCHEMA], sort_keys=True))")"
SHARED=(--quant Q8_0 --device xpu:0 --native-device gpu --fit-situations 100
  --recipe selection_unit=complete_situation --recipe seed=0 --recipe "epochs=$EPOCHS"
  --budget "max_tokens=$MAX_TOKENS" --budget "max_monitor_tokens=$MAX_MONITOR_TOKENS" --budget "max_seconds=$MAX_SECONDS"
  --budget "feasibility_updates=$FEASIBILITY_UPDATES")
prepare_job() {
  local to=$1; shift
  fresh "$to"
  "$XPU/semantic-trainer" prepare "$STAGE/$DEFINITION" "$EXPORT/$STEM.export.jsonl.manifest.json" --assets "$(json "$ASSETS" "d['manifest']")" \
    "${SHARED[@]}" "$@" --to "$to"
}
cadences() {
  "$PY/python" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(f\"profile {d['id'][:16]} on {d['device']}: update {d['costs']['update_seconds_per_token'] * 1e3:.3f} ms/token, point {d['costs']['point_seconds_per_token'] * 1e3:.3f} ms/token, checkpoint {d['costs']['checkpoint_seconds']:.1f} s\")
for c in d['cadences']:
    print(f\"  eval_every_examples={c['eval_every_examples']:>6}  points={c['evaluation_points']:>3}  monitor_tokens={c['monitor_tokens']:>10}  hours={c['seconds'] / 3600:6.2f}  feasible={c['feasible']}\")
" "$1"
}

if [[ -z "$PROFILE" ]]; then
  PROVISIONAL="$WORK/provisional/$(identity "$(basename "$EXPORT")" "$CONTRACT" "${SHARED[@]}" "$PROVISIONAL_CADENCE")"
  step provisional "$(basename "$PROVISIONAL")" prepare_job "$PROVISIONAL" --recipe "eval_every_examples=$PROVISIONAL_CADENCE"
  PROFILE="$WORK/profiles/$(identity "$(sha "$PROVISIONAL/job.yaml")" "$PROFILE_STEPS" "$PROFILE_WARMUP" "$PROFILE_CADENCES").json"
  profile_fit() {
    require_memory profiling
    fresh "$PROFILE"; mkdir -p "$(dirname "$PROFILE")"
    local flags=(); for cadence in $PROFILE_CADENCES; do flags+=(--cadence "$cadence"); done
    scoped "$XPU/semantic-trainer" profile "$PROVISIONAL/job.yaml" --to "$PROFILE" --steps "$PROFILE_STEPS" --warmup "$PROFILE_WARMUP" "${flags[@]}" > /dev/null
    cadences "$PROFILE"
  }
  step profile "$(basename "$PROFILE" .json)" profile_fit
fi
if [[ ! -f "$PROFILE" ]]; then echo "PROFILE $PROFILE does not exist" >&2; exit 2; fi
if [[ -z "$EVAL_EVERY_EXAMPLES" ]]; then
  cadences "$PROFILE"
  echo "choose the monitoring cadence from this measured profile and rerun with EVAL_EVERY_EXAMPLES=N PROFILE=$PROFILE" >&2; exit 3
fi
SETTINGS=(--recipe "eval_every_examples=$EVAL_EVERY_EXAMPLES" --profile "$PROFILE")
JOB="$WORK/jobs/$(identity "$(basename "$EXPORT")" "$CONTRACT" "${SHARED[@]}" "$EVAL_EVERY_EXAMPLES" "$(sha "$PROFILE")")"
step prepare "$(basename "$JOB")" prepare_job "$JOB" "${SETTINGS[@]}"

RUN="$JOB/runs/$(sha "$JOB/job.yaml")"
RECEIPT="$RUN/deployment/receipt.json"
fit() {
  require_memory fitting
  fresh "$RUN"
  "$XPU/semantic-trainer" lora "$JOB/job.yaml" --to "$RUN" --dry-run
  scoped "$XPU/semantic-trainer" lora "$JOB/job.yaml" --to "$RUN"
}
step fit "$(basename "$RUN")" fit

select_deployment() { guarded "$XPU/semantic-trainer" select "$JOB/job.yaml" --run "$RUN"; }
step select "$(basename "$RUN")" select_deployment

CHECK="$WORK/checks/$(identity "$(sha "$RECEIPT")" "$(sha "$EVIDENCE")")"
check() { fresh "$CHECK"; guarded "$PY/semantic-trainer" check "$RECEIPT" "$LABELS" --evidence "$EVIDENCE" --to "$CHECK"; }
step check "$(basename "$CHECK")" check

SNAPSHOTS="$("$PY/python" -c "from bench.snapshots import SCHEMA, DECISION; from semantic_switch.metrics import METRICS; print(SCHEMA, DECISION, METRICS)")"
COMPARE="$(identity "$(basename "$CHECK")" "$(sha "$PACK/benchmark.json")" "$(base)" "$MARGIN" "$SNAPSHOTS")"
RESULT="bench/results/$PACK_KIND-specialist-$COMPARE.json"
compare() {
  guarded "$PY/python" -m bench.snapshots "$PACK_KIND" --pack "$PACK" --evidence "$EVIDENCE" --baseline base="$(base)" --specialist specialist="$RECEIPT" \
    --reference jev=openrouter:typesafe/jev-1.13 --native device=gpu --key-file "$KEY_FILE" --margin "$MARGIN" --output "$RESULT"
}
step compare "$COMPARE" compare

echo "== finished: progress $RUN/progress.jsonl, check $CHECK, comparison $RESULT"
echo "pack review status: $(json "$PACK/benchmark.json" "d.get('review')")"
