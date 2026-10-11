#!/usr/bin/env bash
# Extra runs (3): cnn output with the drop_first_sentence ablation, seeds 0-2.
# Runs whose results JSON already exists are skipped, so the script can be resumed.
# DRY_RUN=1 bash scripts/run_extra.sh   prints the commands without training.
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

for seed in 0 1 2; do
  run "cnn_output_s${seed}_drop_first_sentence" --model cnn --mode output --seed "$seed" --ablation drop_first_sentence
done
echo "[$(ts)] extra runs done"
