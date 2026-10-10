#!/usr/bin/env bash
# RQ3 runs (12): train on general+math, test on code (cross-domain) vs general+math (in-domain).
# Runs whose results JSON already exists are skipped, so the script can be resumed.
# DRY_RUN=1 bash scripts/run_rq3.sh   prints the commands without training.
set -euo pipefail
cd "$(dirname "$0")/.."

ts() { date "+%Y-%m-%d %H:%M:%S"; }

# run <name> <train.py args...>; <name> must match run_name() in src/train.py
run() {
  local name="$1"; shift
  if [[ -f "results/${name}.json" ]]; then
    echo "[$(ts)] skip   ${name} (results/${name}.json exists)"
    return
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[$(ts)] dry    ${name}: python src/train.py $*"
    return
  fi
  echo "[$(ts)] start  ${name}"
  python src/train.py "$@"
  echo "[$(ts)] finish ${name}"
}

TR="general+math"
for seed in 0 1 2; do
  for te in code "general math"; do
    te_name="${te// /+}"
    # $te is intentionally unquoted below so "general math" becomes two arguments
    run "cnn_output_s${seed}_tr-${TR}_te-${te_name}" \
      --model cnn --mode output --seed "$seed" --train_domains general math --test_domains $te
    run "lstm_output_s${seed}_tr-${TR}_te-${te_name}_pat4" \
      --model lstm --mode output --seed "$seed" --train_domains general math --test_domains $te --patience 4
  done
done
echo "[$(ts)] rq3 runs done"
