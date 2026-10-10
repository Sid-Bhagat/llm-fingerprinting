#!/usr/bin/env bash
# RQ4 runs (18): output-mode ablations, an input-length control, and an early-stopping check.
# Runs whose results JSON already exists are skipped, so the script can be resumed.
# DRY_RUN=1 bash scripts/run_rq4.sh   prints the commands without training.
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
  for model in cnn lstm; do
    # markdown stripped / output truncated to 100 tokens
    for abl in strip_markdown truncate100; do
      run "${model}_output_s${seed}_${abl}" --model "$model" --mode output --seed "$seed" --ablation "$abl"
    done
  done
  # RQ2 control: output-only with the same 192 output tokens as "both" mode
  run "cnn_output_s${seed}_maxout192" --model cnn --mode output --seed "$seed" --max_out_len 192
  # early-stopping sensitivity
  run "lstm_output_s${seed}_pat4" --model lstm --mode output --seed "$seed" --patience 4
done
echo "[$(ts)] rq4 runs done"
