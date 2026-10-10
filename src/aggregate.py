"""Aggregate results/*.json into results/summary.csv and seed-0 confusion-matrix plots."""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
GROUP = ["model", "mode", "ablation", "train_domains", "test_domains", "max_out_len", "patience"]
DEFAULT_OUT_LEN = {"input": 0, "output": 256, "both": 192}  # mirrors train.py; 0 = no output (input mode)


def load_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if "test" not in r or r.get("train_frac", 1.0) < 1:
            continue  # not a training result, or a subsampled smoke run
        runs.append(r)
    return runs


def summarize(runs):
    df = pd.DataFrame([{
        "model": r["model"], "mode": r["mode"], "ablation": r["ablation"],
        "train_domains": "+".join(r["train_domains"]), "test_domains": "+".join(r["test_domains"]),
        # older JSONs predate --max_out_len; input mode stores None
        "max_out_len": r.get("max_out_len") or DEFAULT_OUT_LEN[r["mode"]], "patience": r.get("patience", 2),
        "seed": r["seed"], "accuracy": r["test"]["accuracy"], "macro_f1": r["test"]["macro_f1"],
    } for r in runs])
    g = df.groupby(GROUP, sort=True)
    out = g.agg(
        n_seeds=("seed", "nunique"), seeds=("seed", lambda s: " ".join(map(str, sorted(s)))),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    # std is undefined (NaN) for a single seed; report 0 so the "±" column still reads
    for m in ["accuracy", "macro_f1"]:
        out[f"{m}_std"] = out[f"{m}_std"].fillna(0.0)
        out[m] = [f"{a:.4f} ± {s:.4f}" for a, s in zip(out[f"{m}_mean"], out[f"{m}_std"])]
    return out


def plot_confusion(r, path):
    labels = r["labels"]
    cm = np.array(r["test"]["confusion_matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # row-normalized: recall per true family
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title(f"{r['run_name']}\nacc {r['test']['accuracy']:.3f}  macro-F1 {r['test']['macro_f1']:.3f}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        print(f"no result JSONs in {RESULTS_DIR}")
        return

    summary = summarize(runs)
    path = os.path.join(RESULTS_DIR, "summary.csv")
    summary.to_csv(path, index=False)
    print(f"wrote {path} ({len(summary)} configs from {len(runs)} runs)")
    print(summary[GROUP + ["n_seeds", "accuracy", "macro_f1"]].to_string(index=False))

    for r in runs:
        if r["seed"] == 0:
            fig_path = os.path.join(FIG_DIR, f"cm_{r['run_name']}.png")
            plot_confusion(r, fig_path)
            print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
