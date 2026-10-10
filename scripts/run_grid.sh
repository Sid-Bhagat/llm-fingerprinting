#!/usr/bin/env bash
# RQ1/RQ2 grid: {cnn,lstm} x {input,output,both} x seeds {0,1,2}, ablation none, all domains.
# Runs whose results JSON already exists are skipped, so the grid can be resumed.
set -euo pipefail
cd "$(dirname "$0")/.."

ts() { date "+%Y-%m-%d %H:%M:%S"; }

for model in cnn lstm; do
  for mode in input output both; do
    for seed in 0 1 2; do
      name="${model}_${mode}_s${seed}"   # matches run_name() in src/train.py for default args
      if [[ -f "results/${name}.json" ]]; then
        echo "[$(ts)] skip   ${name} (results/${name}.json exists)"
        continue
      fi
      echo "[$(ts)] start  ${name}"
      python src/train.py --model "$model" --mode "$mode" --seed "$seed" --ablation none
      echo "[$(ts)] finish ${name}"
    done
  done
done
echo "[$(ts)] grid done"
