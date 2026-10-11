#!/usr/bin/env bash
# Reruns (6) that save test predictions to results/preds/{run}.csv: cnn output and lstm output
# --patience 4, seeds 0-2. Their result JSONs go to results/preds/runs/ so the originals in results/
# are not overwritten; at the end, src/verify_preds.py checks the test metrics match the originals exactly.
# Runs whose predictions CSV and JSON already exist are skipped, so the script can be resumed.
# DRY_RUN=1 bash scripts/run_preds.sh   prints the commands without training.
set -euo pipefail
cd "$(dirname "$0")/.."

ts() { date "+%Y-%m-%d %H:%M:%S"; }

PREDS_DIR="results/preds"
RUNS_DIR="results/preds/runs"

# run <name> <train.py args...>; <name> must match run_name() in src/train.py
run() {
  local name="$1"; shift
  if [[ -f "${PREDS_DIR}/${name}.csv" && -f "${RUNS_DIR}/${name}.json" ]]; then
    echo "[$(ts)] skip   ${name} (${PREDS_DIR}/${name}.csv exists)"
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

COMMON=(--save_preds --preds_dir "$PREDS_DIR" --results_dir "$RUNS_DIR")
for seed in 0 1 2; do
  run "cnn_output_s${seed}" --model cnn --mode output --seed "$seed" "${COMMON[@]}"
  run "lstm_output_s${seed}_pat4" --model lstm --mode output --seed "$seed" --patience 4 "${COMMON[@]}"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[$(ts)] dry    verify: python src/verify_preds.py"
else
  python src/verify_preds.py
fi
echo "[$(ts)] preds runs done"
