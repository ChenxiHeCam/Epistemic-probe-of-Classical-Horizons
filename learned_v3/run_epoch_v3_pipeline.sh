#!/usr/bin/env bash
set -Eeuo pipefail

# Fail-closed orchestration for the two frozen EPOCH horizon runs.  This script
# never changes training or scoring code; it only enforces ordering, hashes and
# completion checks around the preregistered Phase A -> Phase B sequence.

BASE=/root/autodl-tmp/epoch_horizons/v2
SCRIPTS="$BASE/scripts"
RUNS="$BASE/runs"
RESULTS="$BASE/results"
PY=/root/autodl-tmp/epoch_env_torch214/bin/python
PHASE_A_RUN="$RUNS/phase_a_pre1900_strict_v3_20260907"
PHASE_B_RUN="$RUNS/phase_b_pre1950_joint_v3_20260907"
PHASE_A_RESULT="$RESULTS/phase_a_pre1900_v3_evaluation.json"
PHASE_B_RESULT="$RESULTS/phase_b_pre1950_v3_evaluation.json"
CLOCK_RESULT="$RESULTS/epoch_clock_v3_comparison.json"
PIPELINE_LOG=/root/autodl-tmp/epoch_v3_pipeline_supervisor.log
PHASE_B_LOG=/root/autodl-tmp/epoch_phase_b_pre1950_joint_v3_20260907.log
PHASE_A_EVAL_LOG=/root/autodl-tmp/epoch_phase_a_eval_v3.log
PHASE_B_EVAL_LOG=/root/autodl-tmp/epoch_phase_b_eval_v3.log
EXPECTED_EVALUATOR_SHA=831f45cbb919e148337d88265fc66a61915e5b8bd5bcecc28fa80b80bd804288
EXPECTED_TRAINER_SHA=39caa71c3436b6316e798cd783e2f8b750791d0210507b11e8da52ad628d1895
EXPECTED_COMPARE_SHA=df7750993c1390c581c7071da0c43a0597a4c04bcdb9d2f86c471bf40df9f46c
EXPECTED_COMPAT_ADDENDUM_SHA=2014c07dcf7228e3e2a190c527da0059b1992d110eb138dba3e89c33882eb1d6

mkdir -p "$RESULTS"
exec 9>"$BASE/.epoch_v3_pipeline.lock"
if ! flock -n 9; then
  printf '%s supervisor already active\n' "$(date -Iseconds)" >> "$PIPELINE_LOG"
  exit 0
fi

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$PIPELINE_LOG"
}

fail() {
  log "FAIL: $*"
  printf '%s\n' "$*" > "$RESULTS/PIPELINE_FAILED.txt"
  exit 1
}

require_hash() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || fail "hash mismatch for $path: $actual"
}

validate_training() {
  local summary=$1
  local cutoff=$2
  "$PY" - "$summary" "$cutoff" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
assert d["status"] == "completed", d
assert int(d["knowledge_cutoff"]) == int(sys.argv[2]), d
assert int(d["epochs_completed"]) == 30, d
assert int(d["global_batches"]) > 0, d
assert d["future_horizon_used_for_optimization"] is False, d
checkpoint = pathlib.Path(d["final_checkpoint"])
assert checkpoint.is_file(), checkpoint
print(json.dumps({k: d[k] for k in (
    "status", "knowledge_cutoff", "global_batches", "epochs_completed",
    "mean_clouds_per_second", "final_checkpoint_sha256"
)}, sort_keys=True))
PY
}

validate_evaluation() {
  local result=$1
  local cutoff=$2
  "$PY" - "$result" "$cutoff" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
assert d["status"] == "PASS", d
assert int(d["knowledge_cutoff"]) == int(sys.argv[2]), d
assert d["primary_metric"] == "one_class_ensemble", d
assert d["future_horizon_used_for_training_or_score_fitting"] is False, d
assert d["comparisons"], d
print(json.dumps(d["comparisons"], sort_keys=True))
PY
}

trap 'fail "supervisor stopped at line $LINENO"' ERR

require_hash "$SCRIPTS/train_epoch_horizon_v3.py" "$EXPECTED_TRAINER_SHA"
require_hash "$SCRIPTS/evaluate_epoch_horizon_v3.py" "$EXPECTED_EVALUATOR_SHA"
require_hash "$SCRIPTS/compare_epoch_clock_v3.py" "$EXPECTED_COMPARE_SHA"
require_hash "$BASE/EPOCH_LEARNED_V3_EVALUATOR_COMPATIBILITY_ADDENDUM.json" "$EXPECTED_COMPAT_ADDENDUM_SHA"

log "waiting for frozen Phase A training"
while [[ ! -s "$PHASE_A_RUN/training_summary.json" ]]; do
  if ! pgrep -f 'train_epoch_horizon_v3.py.*phase_a_pre1900_strict_v3_20260907' >/dev/null; then
    fail "Phase A process disappeared before writing training_summary.json"
  fi
  sleep 30
done
while pgrep -f 'train_epoch_horizon_v3.py.*phase_a_pre1900_strict_v3_20260907' >/dev/null; do
  sleep 5
done
validate_training "$PHASE_A_RUN/training_summary.json" 1899 | tee -a "$PIPELINE_LOG"
log "Phase A training verified"

if [[ ! -s "$PHASE_A_RESULT" ]]; then
  log "starting frozen Phase A evaluation"
  cd "$SCRIPTS"
  OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 "$PY" evaluate_epoch_horizon_v3.py \
    --config "$BASE/run_configs/train_pre1900.json" \
    --checkpoint "$PHASE_A_RUN/checkpoint_final.pt" \
    --output "$PHASE_A_RESULT" \
    --batch-size 1024 --graph-batch-size 256 --workers 8 --points 64 \
    --top-k 8 --bootstrap 2000 --seed 20260907 \
    > "$PHASE_A_EVAL_LOG" 2>&1
fi
validate_evaluation "$PHASE_A_RESULT" 1899 | tee -a "$PIPELINE_LOG"
sha256sum "$PHASE_A_RESULT" > "$PHASE_A_RESULT.sha256"
log "Phase A evaluation verified and hash-frozen"

if [[ ! -s "$PHASE_B_RUN/training_summary.json" ]]; then
  if pgrep -f 'train_epoch_horizon_v3.py.*phase_b_pre1950_joint_v3_20260907' >/dev/null; then
    fail "unmanaged Phase B process already exists"
  fi
  log "starting cumulative pre-1900 + pre-1950 Phase B training"
  mkdir -p "$PHASE_B_RUN"
  cd "$SCRIPTS"
  OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 "$PY" train_epoch_horizon_v3.py \
    --config "$BASE/run_configs/train_pre1950_joint.json" \
    --output "$PHASE_B_RUN" \
    --epochs 30 --batch-size 1024 --workers 8 --lr 0.0003 \
    --weight-decay 0.05 --warmup-steps 1000 --queue-size 16384 \
    --view-weight 0.25 --clip-grad 1.0 --log-every 20 --save-every 5000 \
    --validation-batches 32 --seed 20260907 \
    --phase-a-freeze "$PHASE_A_RESULT" \
    > "$PHASE_B_LOG" 2>&1
fi
validate_training "$PHASE_B_RUN/training_summary.json" 1950 | tee -a "$PIPELINE_LOG"
log "Phase B training verified"

if [[ ! -s "$PHASE_B_RESULT" ]]; then
  log "starting frozen Phase B evaluation"
  cd "$SCRIPTS"
  OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 "$PY" evaluate_epoch_horizon_v3.py \
    --config "$BASE/run_configs/train_pre1950_joint.json" \
    --checkpoint "$PHASE_B_RUN/checkpoint_final.pt" \
    --output "$PHASE_B_RESULT" \
    --batch-size 1024 --graph-batch-size 256 --workers 8 --points 64 \
    --top-k 8 --bootstrap 2000 --seed 20260907 \
    > "$PHASE_B_EVAL_LOG" 2>&1
fi
validate_evaluation "$PHASE_B_RESULT" 1950 | tee -a "$PIPELINE_LOG"
sha256sum "$PHASE_B_RESULT" > "$PHASE_B_RESULT.sha256"
log "Phase B evaluation verified and hash-frozen"

cd "$SCRIPTS"
"$PY" compare_epoch_clock_v3.py \
  --pre1900 "$PHASE_A_RESULT" --pre1950 "$PHASE_B_RESULT" \
  --output "$CLOCK_RESULT" --bootstrap 10000 --seed 20260907 \
  >> "$PIPELINE_LOG" 2>&1
"$PY" - "$CLOCK_RESULT" <<'PY' | tee -a "$PIPELINE_LOG"
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert d["status"] == "PASS", d
assert d["families"] > 0, d
print(json.dumps({k: d[k] for k in (
    "families", "pre1900_median", "pre1950_median", "median_collapse",
    "median_collapse_ci95", "fraction_decreased_among_non_ties",
    "exact_two_sided_sign_test_p"
)}, sort_keys=True))
PY
sha256sum "$CLOCK_RESULT" > "$CLOCK_RESULT.sha256"
touch "$RESULTS/EPOCH_V3_PIPELINE_COMPLETE"
log "PASS: complete two-horizon pipeline"
