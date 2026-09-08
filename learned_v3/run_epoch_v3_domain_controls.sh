#!/usr/bin/env bash
set -Eeuo pipefail

BASE=/root/autodl-tmp/epoch_horizons/v2
SCRIPTS="$BASE/scripts"
RUNS="$BASE/runs"
RESULTS="$BASE/results"
CONTROL="$BASE/domain_controls"
PY=/root/autodl-tmp/epoch_env_torch214/bin/python
LOG=/root/autodl-tmp/epoch_v3_domain_controls.log
DONE="$RESULTS/EPOCH_V3_RANDOM_BASELINES_COMPLETE"
MAIN_FAIL="$RESULTS/PIPELINE_FAILED.txt"
RANDOM_FAIL="$RESULTS/RANDOM_BASELINES_FAILED.txt"
TRAINED_RESULT="$RESULTS/matched_pre1900_domain_control_trained_v3.json"
RANDOM_RESULT="$RESULTS/matched_pre1900_domain_control_random_v3.json"

EXPECTED_SCORER_SHA=be165a5f707790c5c811830b35d4bd4f6691647351b69559426b33ca6beb65fd
EXPECTED_EVALUATOR_SHA=831f45cbb919e148337d88265fc66a61915e5b8bd5bcecc28fa80b80bd804288
EXPECTED_REGISTRATION_SHA=36a45108959a19cba886aff7bc904e9e47edcd44b7ade9d5b2d1ca89ff1593ad
EXPECTED_ARCHIVE_SHA=125c9f4e30b8e1ada6c86c5b0d513d9a5911161c699dc3d276dfd58a889dd70c
EXPECTED_MANIFEST_SHA=6f33f1f4888b437702c5cfb43f8bbbd1759c66876db36c76455a3cbd908ee46a

mkdir -p "$RESULTS"
exec 9>"$BASE/.epoch_v3_domain_controls.lock"
flock -n 9 || exit 0

log() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$LOG"; }
fail() {
  log "FAIL: $*"
  printf '%s\n' "$*" > "$RESULTS/DOMAIN_CONTROLS_FAILED.txt"
  exit 1
}
require_hash() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || fail "hash mismatch for $path: $actual"
}
validate() {
  "$PY" - "$1" <<'PY'
import json, pathlib, sys
d=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert d["status"] == "PASS", d
assert d["schema_version"] == "3.0-post-score-domain-audit", d
assert int(d["knowledge_cutoff"]) == 1899, d
assert d["future_horizon_used_for_training_or_score_fitting"] is False, d
assert d["sets"]["generator_matched_pre1900_controls"]["groups"] == 9, d
assert d["comparisons"], d
print(json.dumps(d["comparisons"], sort_keys=True))
PY
}

trap 'code=$?; trap - ERR; fail "domain-control supervisor stopped at line $LINENO (exit $code)"' ERR
require_hash "$SCRIPTS/evaluate_matched_pre1900_clock_controls_v3.py" "$EXPECTED_SCORER_SHA"
require_hash "$SCRIPTS/evaluate_epoch_horizon_v3.py" "$EXPECTED_EVALUATOR_SHA"
require_hash "$BASE/EPOCH_LEARNED_V3_DOMAIN_SHIFT_DIAGNOSTIC_REGISTRATION.json" "$EXPECTED_REGISTRATION_SHA"
require_hash "$CONTROL/matched_pre1900_controls_v3.npz" "$EXPECTED_ARCHIVE_SHA"
require_hash "$CONTROL/matched_pre1900_controls_v3_manifest.json" "$EXPECTED_MANIFEST_SHA"

log "waiting for trained and random primary pipelines"
while [[ ! -e "$DONE" ]]; do
  [[ ! -e "$MAIN_FAIL" ]] || fail "trained-model pipeline failed"
  [[ ! -e "$RANDOM_FAIL" ]] || fail "random-baseline pipeline failed"
  sleep 30
done

cd "$SCRIPTS"
if [[ ! -s "$TRAINED_RESULT" ]]; then
  log "scoring generator-matched controls with trained Phase-A checkpoint"
  OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 "$PY" evaluate_matched_pre1900_clock_controls_v3.py \
    --config "$BASE/run_configs/train_pre1900.json" \
    --checkpoint "$RUNS/phase_a_pre1900_strict_v3_20260907/checkpoint_final.pt" \
    --primary-result "$RESULTS/phase_a_pre1900_v3_evaluation.json" \
    --archive "$CONTROL/matched_pre1900_controls_v3.npz" \
    --manifest "$CONTROL/matched_pre1900_controls_v3_manifest.json" \
    --output "$TRAINED_RESULT" --batch-size 1024 --graph-batch-size 256 \
    --workers 8 --points 64 --top-k 8 --bootstrap 2000 --seed 20260907 \
    > /root/autodl-tmp/epoch_domain_control_trained_v3.log 2>&1
fi
validate "$TRAINED_RESULT" | tee -a "$LOG"
sha256sum "$TRAINED_RESULT" > "$TRAINED_RESULT.sha256"

if [[ ! -s "$RANDOM_RESULT" ]]; then
  log "scoring generator-matched controls with zero-optimization Phase-A checkpoint"
  OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 "$PY" evaluate_matched_pre1900_clock_controls_v3.py \
    --config "$BASE/run_configs/train_pre1900.json" \
    --checkpoint "$RUNS/random_pre1900_v3_20260907/checkpoint_random.pt" \
    --primary-result "$RESULTS/random_phase_a_pre1900_v3_evaluation.json" \
    --archive "$CONTROL/matched_pre1900_controls_v3.npz" \
    --manifest "$CONTROL/matched_pre1900_controls_v3_manifest.json" \
    --output "$RANDOM_RESULT" --batch-size 1024 --graph-batch-size 256 \
    --workers 8 --points 64 --top-k 8 --bootstrap 2000 --seed 20260907 \
    > /root/autodl-tmp/epoch_domain_control_random_v3.log 2>&1
fi
validate "$RANDOM_RESULT" | tee -a "$LOG"
sha256sum "$RANDOM_RESULT" > "$RANDOM_RESULT.sha256"
touch "$RESULTS/EPOCH_V3_DOMAIN_CONTROLS_COMPLETE"
log "PASS: post-score generator-domain diagnostic complete"
