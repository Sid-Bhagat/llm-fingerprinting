"""Per-underlying-model error analysis from the saved test predictions (scripts/run_preds.sh).

For cnn output and lstm output --patience 4, averaged over the seeds found in results/preds/:
test count, accuracy, and the share of each model's responses predicted as each family.
Writes results/error_analysis/per_model.csv, per_model_heatmap.png, and examples.md
(misclassified GPT->Llama and Llama->GPT outputs, cnn seed 0; contains dataset text, so gitignored).
"""
import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
FAMILIES = ["GPT", "Claude", "Llama", "PaLM"]
MODELS = ["gpt-4", "gpt-3.5-turbo", "claude-v1", "claude-instant-v1", "vicuna-13b", "vicuna-7b", "koala-13b",
          "alpaca-13b", "wizardlm-13b", "gpt4all-13b-snoozy", "guanaco-33b", "palm-2"]
# arch -> run-name pattern for seed s (matches run_name() in src/train.py)
RUNS = {"cnn": "cnn_output_s{}", "lstm": "lstm_output_s{}_pat4"}
SEEDS = [0, 1, 2]
LOW_COUNT = 30
N_EXAMPLES, N_CHARS = 5, 300


def load_preds(preds_dir, arch):
    out = []
    for s in SEEDS:
        path = os.path.join(preds_dir, RUNS[arch].format(s) + ".csv")
        if os.path.exists(path):
            out.append(pd.read_csv(path).assign(seed=s))
    return out


def per_model(preds_by_seed, arch):
    rows = []
    for m in MODELS:
        per_seed = [p[p["model"] == m] for p in preds_by_seed]
        counts = {len(d) for d in per_seed}
        assert len(counts) == 1, f"{arch} {m}: test count differs across seeds {counts}"
        n = counts.pop()
        acc = [d["correct"].mean() for d in per_seed]
        shares = np.mean([[(d["pred"] == f).mean() for f in FAMILIES] for d in per_seed], axis=0)
        rows.append({"arch": arch, "run": RUNS[arch].format("N"), "model": m, "family": per_seed[0]["family"].iloc[0],
                     "n_test": n, "low_count": n < LOW_COUNT, "n_seeds": len(per_seed),
                     "accuracy_mean": np.mean(acc), "accuracy_std": np.std(acc, ddof=1) if len(acc) > 1 else 0.0,
                     **{f"pred_{f}": v for f, v in zip(FAMILIES, shares)}})
    return pd.DataFrame(rows)


def heatmap(table, path):
    archs = list(table["arch"].unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(3.6 * len(archs) + 3.0, 6.2), squeeze=False)
    cols = [f"pred_{f}" for f in FAMILIES]
    for ax, arch in zip(axes[0], archs):
        t = table[table["arch"] == arch]
        m = t[cols].to_numpy()
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > 0.55 else "#222222")
        ax.set_xticks(range(len(FAMILIES)), FAMILIES)
        # rows are the same models in every panel, so only the first panel is labeled
        ax.set_yticks(range(len(t)), [f"{r.model} ({r.family}, n={r.n_test}){' *' if r.low_count else ''}"
                                      for r in t.itertuples()] if ax is axes[0][0] else [], fontsize=8)
        # separate the families with thin lines
        fam = t["family"].tolist()
        for i in range(1, len(fam)):
            if fam[i] != fam[i - 1]:
                ax.axhline(i - 0.5, color="white", linewidth=2)
        n_seeds = t["n_seeds"].iloc[0]
        ax.set_title(f"{arch} ({RUNS[arch].format('N')}, {n_seeds} seed{'s' if n_seeds > 1 else ''})", fontsize=10)
        ax.set_xlabel("predicted family")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02, label="share of responses (mean over seeds)")
    fig.suptitle(f"Predicted family per underlying model (test set; * = fewer than {LOW_COUNT} test examples)", fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fence(text):
    run = max((len(r) for r in re.findall(r"`+", text)), default=0)
    return "`" * max(3, run + 1)


def examples(preds, test, path):
    df = preds.merge(test[["question_id", "model", "output"]], on=["question_id", "model"], how="left", validate="1:1")
    lines = ["# Misclassified examples (cnn output, seed 0)", "",
             f"First {N_EXAMPLES} test examples of each error type, in test-set order; first {N_CHARS} chars of the output.",
             "Contains dataset text; gitignored.", ""]
    for true, pred in [("GPT", "Llama"), ("Llama", "GPT")]:
        sel = df[(df["family"] == true) & (df["pred"] == pred)]
        lines += [f"## True {true}, predicted {pred} ({len(sel)} such errors in total)", ""]
        for i, r in enumerate(sel.head(N_EXAMPLES).itertuples(), 1):
            snippet = r.output[:N_CHARS]
            f = fence(snippet)
            lines += [f"### {i}. {r.model}, domain {r.domain}, question_id {r.question_id}", "", f, snippet, f, ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds_dir", default=os.path.join(ROOT, "results", "preds"))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "error_analysis"))
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    preds = {arch: load_preds(a.preds_dir, arch) for arch in RUNS}
    missing = [arch for arch, v in preds.items() if not v]
    if missing:
        raise SystemExit(f"no saved predictions for {missing} in {a.preds_dir}; run scripts/run_preds.sh first")
    table = pd.concat([per_model(preds[arch], arch) for arch in RUNS], ignore_index=True)
    table.to_csv(os.path.join(a.out_dir, "per_model.csv"), index=False)
    heatmap(table, os.path.join(a.out_dir, "per_model_heatmap.png"))

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.drop(columns=["run"]).to_string(index=False))

    cnn_s0 = [d for d in preds["cnn"] if d["seed"].iloc[0] == 0]
    if cnn_s0:
        test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
        examples(cnn_s0[0], test, os.path.join(a.out_dir, "examples.md"))
    else:
        print("cnn seed 0 predictions missing; examples.md not written")
    print(f"wrote {a.out_dir}/per_model.csv, per_model_heatmap.png" + (", examples.md" if cnn_s0 else ""))


if __name__ == "__main__":
    main()
