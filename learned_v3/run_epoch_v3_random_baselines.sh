#!/usr/bin/env bash
set -Eeuo pipefail

BASE=/root/autodl-tmp/epoch_horizons/v2
SCRIPTS="$BASE/scripts"
RUNS="$BASE/runs"
RESULTS="$BASE/results"
PY=/root/autodl-tmp/epoch_env_torch214/bin/python
MAIN_DONE="$RESULTS/EPOCH_V3_PIPELINE_COMPLETE"
MAIN_FAILED="$RESULTS/PIPELINE_FAILED.txt"
OLD_RUN="$RUNS/random_pre1900_v3_20260907"
NEW_RUN="$RUNS/random_pre1950_v3_20260907"
OLD_RESULT="$RESULTS/random_phase_a_pre1900_v3_evaluation.json"
NEW_RESULT="$RESULTS/random_phase_b_pre1950_v3_evaluation.json"
CLOCK_RESULT="$RESULTS/random_epoch_clock_v3_comparison.json"
LOG=/root/autodl-tmp/epoch_v3_random_baselines.log
EXPECTED_MAKER_SHA=754f97a130dafe61de35129106a5ca4fe27801e8e0f383f01dcdc153af5947ac
EXPECTED_EVALUATOR_SHA=831f45cbb919e148337d88265fc66a61915e5b8bd5bcecc28fa80b80bd804288
EXPECTED_COMPARE_SHA=df7750993c1390c581c7071da0c43a0597a4c04bcdb9d2f86c471bf40df9f46c
EXPECTED_ADDENDUM_SHA=5da3e3ae0ee70723882a1a7acd9b8e23190893edbeb701c56bd92c3c58b1585a
EXPECTED_COMPAT_ADDENDUM_SHA=2014c07dcf7228e3e2a190c527da0059b1992d110eb138dba3e89c33882eb1d6

mkdir -p "$RESULTS" "$OLD_RUN" "$NEW_RUN"
exec 9>"$BASE/.epoch_v3_random_baselines.lock"
flock -n 9 || exit 0

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$LOG"
}

fail() {
  log "FAIL: $*"
  printf '%s\n' "$*" > "$RESULTS/RANDOM_BASELINES_FAILED.txt"
  exit 1
}

require_hash() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || fail "hash mismatch for $path: $actual"
}

validate_eval() {
  "$PY" - "$1" "$2" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert d["status"] == "PASS", d
assert int(d["knowledge_cutoff"]) == int(sys.argv[2]), d
assert d["primary_metric"] == "one_class_ensemble", d
assert d["future_horizon_used_for_training_or_score_fitting"] is False, d
print(json.dumps(d["comparisons"], sort_keys=True))
PY
}

trap 'code=$?; trap - ERR; fail "random-baseline supervisor stopped at line $LINENO (exit $code)"' ERR
require_hash "$SCRIPTS/make_random_epoch_checkpoint_v3.py" "$EXPECTED_MAKER_SHA"
require_hash "$SCRIPTS/evaluate_epoch_horizon_v3.py" "$EXPECTED_EVALUATOR_SHA"
require_hash "$SCRIPTS/compare_epoch_clock_v3.py" "$EXPECTED_COMPARE_SHA"
require_hash "$BASE/EPOCH_LEARNED_V3_RANDOM_BASELINE_PREREG.json" "$EXPECTED_ADDENDUM_SHA"
require_hash "$BASE/EPOCH_LEARNED_V3_EVALUATOR_COMPATIBILITY_ADDENDUM.json" "$EXPECTED_COMPAT_ADDENDUM_SHA"

log "waiting for completion of the frozen trained-model pipeline"
while [[ ! -e "$MAIN_DONE" ]]; do
  [[ ! -e "$MAIN_FAILED" ]] || fail "trained-model pipeline failed"
  sleep 30
done

cd "$SCRIPTS"
if [[ ! -s "$OLD_RUN/checkpoint_random.pt" ]]; then
  "$PY" make_random_epoch_checkpoint_v3.py \
    --config "$BASE/run_configs/train_pre1900.json" \
    --output "$OLD_RUN/checkpoint_random.pt" --seed 20260907 >> "$LOG" 2>&1
fi
if [[ ! -s "$NEW_RUN/checkpoint_random.pt" ]]; then
  "$PY" make_random_epoch_checkpoint_v3.py \
    --config "$BASE/run_configs/train_pre1950_joint.json" \
    --output "$NEW_RUN/checkpoint_random.pt" --seed 20260907 >> "$LOG" 2>&1
fi

if [[ ! -s "$OLD_RESULT" ]]; then
  log "evaluating untrained 1899-horizon baseline"
  OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 "$PY" evaluate_epoch_horizon_v3.py \
    --config "$BASE/run_configs/train_pre1900.json" \
    --checkpoint "$OLD_RUN/checkpoint_random.pt" --output "$OLD_RESULT" \
    --batch-size 1024 --graph-batch-size 256 --workers 8 --points 64 \
    --top-k 8 --bootstrap 2000 --seed 20260907 \
    > /root/autodl-tmp/epoch_random_phase_a_eval_v3.log 2>&1
fi
validate_eval "$OLD_RESULT" 1899 | tee -a "$LOG"
sha256sum "$OLD_RESULT" > "$OLD_RESULT.sha256"

if [[ ! -s "$NEW_RESULT" ]]; then
  log "evaluating untrained 1950-horizon baseline"
  OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 "$PY" evaluate_epoch_horizon_v3.py \
    --config "$BASE/run_configs/train_pre1950_joint.json" \
    --checkpoint "$NEW_RUN/checkpoint_random.pt" --output "$NEW_RESULT" \
    --batch-size 1024 --graph-batch-size 256 --workers 8 --points 64 \
    --top-k 8 --bootstrap 2000 --seed 20260907 \
    > /root/autodl-tmp/epoch_random_phase_b_eval_v3.log 2>&1
fi
validate_eval "$NEW_RESULT" 1950 | tee -a "$LOG"
sha256sum "$NEW_RESULT" > "$NEW_RESULT.sha256"

"$PY" compare_epoch_clock_v3.py \
  --pre1900 "$OLD_RESULT" --pre1950 "$NEW_RESULT" \
  --output "$CLOCK_RESULT" --bootstrap 10000 --seed 20260907 \
  >> "$LOG" 2>&1
sha256sum "$CLOCK_RESULT" > "$CLOCK_RESULT.sha256"
touch "$RESULTS/EPOCH_V3_RANDOM_BASELINES_COMPLETE"
log "PASS: untrained architecture-matched controls complete"
