"""Check that the --save_preds reruns reproduce the original results exactly.

For each results/preds/{run}.csv: the rerun JSON (results/preds/runs/{run}.json) must have the same test
accuracy, macro-F1, per-class F1 and confusion matrix as results/{run}.json, and the CSV must agree with
the rerun JSON. Prints one line per run; exits 1 on any mismatch or missing file.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ["accuracy", "macro_f1", "per_class_f1", "confusion_matrix"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds_dir", default=os.path.join(ROOT, "results", "preds"))
    p.add_argument("--runs_dir", default=os.path.join(ROOT, "results", "preds", "runs"))
    p.add_argument("--ref_dir", default=os.path.join(ROOT, "results"))
    a = p.parse_args()

    paths = sorted(glob.glob(os.path.join(a.preds_dir, "*.csv")))
    if not paths:
        sys.exit(f"no prediction CSVs in {a.preds_dir}")
    bad = 0
    for path in paths:
        name = os.path.basename(path)[:-4]
        new_path, ref_path = os.path.join(a.runs_dir, f"{name}.json"), os.path.join(a.ref_dir, f"{name}.json")
        if not (os.path.exists(new_path) and os.path.exists(ref_path)):
            print(f"MISSING {name}: need {new_path} and {ref_path}")
            bad += 1
            continue
        new, ref = json.load(open(new_path)), json.load(open(ref_path))
        problems = [f"{k}: original {ref['test'][k]} vs rerun {new['test'][k]}" for k in KEYS if new["test"][k] != ref["test"][k]]
        if new["best_epoch"] != ref["best_epoch"]:
            problems.append(f"best_epoch: original {ref['best_epoch']} vs rerun {new['best_epoch']}")
        preds = pd.read_csv(path)
        labels = new["labels"]
        cm = confusion_matrix(preds["family"], preds["pred"], labels=labels).tolist()
        if cm != new["test"]["confusion_matrix"] or (preds["family"] == preds["pred"]).tolist() != preds["correct"].tolist():
            problems.append("predictions CSV disagrees with the rerun JSON")
        if problems:
            bad += 1
            print(f"MISMATCH {name}:\n  " + "\n  ".join(problems))
        else:
            print(f"OK       {name}: acc {new['test']['accuracy']:.4f} macro_f1 {new['test']['macro_f1']:.4f} "
                  f"(best epoch {new['best_epoch']}) identical to {ref_path}")
    print(f"{len(paths) - bad}/{len(paths)} runs match")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
