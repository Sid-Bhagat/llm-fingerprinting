"""Report figures from results/*.json -> results/report_figs/ (PNG, 300 dpi). Also prints the
seed-averaged confusion matrices and the trainable-parameter counts of both models."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "report_figs")
VOCAB_PATH = os.path.join(ROOT, "data", "processed", "vocab.json")
FAMILIES = ["GPT", "Claude", "Llama", "PaLM"]
ALL_DOMAINS = ["general", "code", "math"]
SEEDS = [0, 1, 2]
DEFAULT_OUT_LEN = {"input": None, "output": 256, "both": 192}  # input-mode JSONs store None; grid JSONs predate the key

# family colors match src/features.py; models use other slots so a hue always means one thing
FAMILY_COLORS = dict(zip(FAMILIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
MODEL_COLORS = {"cnn": "#6250d6", "lstm": "#008300"}
MODEL_NAMES = {"cnn": "CNN", "lstm": "BiLSTM"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e5e4df"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(model, mode, seed, ablation="none", patience=2, max_out_len=None, train_domains=ALL_DOMAINS,
         test_domains=ALL_DOMAINS):
    """Load one full-data run (name built like train.run_name); None if its JSON is missing."""
    name = f"{model}_{mode}_s{seed}"
    if ablation != "none":
        name += f"_{ablation}"
    if train_domains != ALL_DOMAINS or test_domains != ALL_DOMAINS:
        name += f"_tr-{'+'.join(train_domains)}_te-{'+'.join(test_domains)}"
    if max_out_len is not None and max_out_len != DEFAULT_OUT_LEN[mode]:
        name += f"_maxout{max_out_len}"
    if patience != 2:
        name += f"_pat{patience}"
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        r = json.load(f)
    # guard against the file name lying about the config
    assert r["model"] == model and r["mode"] == mode and r["seed"] == seed and r["ablation"] == ablation, path
    assert r.get("patience", 2) == patience and r.get("train_frac", 1.0) == 1.0, path
    assert r["train_domains"] == train_domains and r["test_domains"] == test_domains, path
    assert (r.get("max_out_len") or DEFAULT_OUT_LEN[mode]) == (max_out_len or DEFAULT_OUT_LEN[mode]), path
    assert r["labels"] == FAMILIES, path
    return r


def load_seeds(*args, **kw):
    runs = [load(args[0], args[1], s, *args[2:], **kw) for s in SEEDS]
    missing = [s for s, r in zip(SEEDS, runs) if r is None]
    if missing:
        print(f"  warning: {args} {kw} missing seeds {missing}")
    return [r for r in runs if r is not None]


def f1_stats(runs):
    f1 = np.array([r["test"]["macro_f1"] for r in runs])
    return f1.mean(), (f1.std(ddof=1) if len(f1) > 1 else 0.0), len(f1)


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def bar_label(ax, x, mean, std):
    ax.text(x, mean + std + 0.012, f"{mean:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK_2)


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def fig_rq1_rq2():
    modes = ["input", "output", "both"]
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k, model in enumerate(["cnn", "lstm"]):
        for i, mode in enumerate(modes):
            m, s, n = f1_stats(load_seeds(model, mode))
            x = i + (k - 1) * width
            ax.bar(x, m, width * 0.92, yerr=s, capsize=3, color=MODEL_COLORS[model],
                   label=MODEL_NAMES[model] if i == 0 else None, error_kw={"ecolor": INK, "lw": 1})
            bar_label(ax, x, m, s)
    # lstm output, patience 4: third bar in the output group
    m, s, n = f1_stats(load_seeds("lstm", "output", patience=4))
    x = 1 + width
    ax.bar(x, m, width * 0.92, yerr=s, capsize=3, facecolor="white", edgecolor=MODEL_COLORS["lstm"],
           hatch="///", linewidth=1.2, label="BiLSTM (patience 4)", error_kw={"ecolor": INK, "lw": 1})
    bar_label(ax, x, m, s)
    ax.axhline(0.25, color=INK_2, linestyle="--", linewidth=1)
    ax.text(2.45, 0.255, "chance", ha="right", va="bottom", color=INK_2, fontsize=9)
    ax.set_xticks(range(len(modes)), ["Input only", "Output only", "Input + output"])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Family classification by input mode (mean ± std, 3 seeds)")
    ax.legend(loc="upper left", frameon=False)
    style_axis(ax)
    save(fig, "rq1_rq2_bars.png")


def mean_norm_cm(runs):
    cms = [np.array(r["test"]["confusion_matrix"], dtype=float) for r in runs]
    return np.mean([cm / cm.sum(axis=1, keepdims=True) for cm in cms], axis=0)


def print_cm(title, cm):
    print(f"\n{title} (rows = true, cols = predicted, row-normalized, mean over seeds)")
    print(" " * 8 + "".join(f"{f:>8}" for f in FAMILIES))
    for f, row in zip(FAMILIES, cm):
        print(f"{f:>8}" + "".join(f"{v:8.3f}" for v in row))


def fig_confusion():
    panels = [("CNN, output only", load_seeds("cnn", "output")),
              ("BiLSTM, output only (patience 4)", load_seeds("lstm", "output", patience=4))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for ax, (title, runs) in zip(axes, panels):
        cm = mean_norm_cm(runs)
        print_cm(f"{title}, seeds {[r['seed'] for r in runs]}", cm)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(FAMILIES)):
            for j in range(len(FAMILIES)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(FAMILIES)), FAMILIES)
        ax.set_yticks(range(len(FAMILIES)), FAMILIES)
        ax.set_xlabel("Predicted family")
        ax.set_ylabel("True family")
        ax.set_title(f"{title}\n(mean of {len(runs)} seeds)")
        for side in ax.spines.values():
            side.set_visible(False)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="Share of true-family row")
    save(fig, "confusion_side_by_side.png")


def fig_ablations():
    conds = [("Full output\n(256 tokens)", dict()),
             ("First 192\ntokens", dict(max_out_len=192)),
             ("Markdown\nstripped", dict(ablation="strip_markdown")),
             ("First 100\ntokens", dict(ablation="truncate100"))]
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k, model in enumerate(["cnn", "lstm"]):
        for i, (_, kw) in enumerate(conds):
            if model == "lstm" and "max_out_len" in kw:
                continue  # the 192-token control was run for the CNN only
            runs = load_seeds(model, "output", **kw)
            if not runs:
                continue
            m, s, n = f1_stats(runs)
            x = i + (k - 0.5) * width
            ax.bar(x, m, width * 0.92, yerr=s, capsize=3, color=MODEL_COLORS[model],
                   label=MODEL_NAMES[model] if i == 0 else None, error_kw={"ecolor": INK, "lw": 1})
            bar_label(ax, x, m, s)
    ax.text(1 + 0.5 * width, 0.02, "CNN only", ha="center", va="bottom", fontsize=8.5, color=INK_2, rotation=90)
    ax.axhline(0.25, color=INK_2, linestyle="--", linewidth=1, label="chance (0.25)")
    ax.set_xticks(range(len(conds)), [c[0] for c in conds])
    ax.set_xlim(-0.6, 3.65)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Output-only ablations (mean ± std, 3 seeds, patience 2)")
    ax.legend(loc="upper center", ncol=3, frameon=False)
    style_axis(ax)
    save(fig, "ablations.png")


def fig_training_curves():
    # patience 4 is drawn first: both lstm runs are identical until patience 2 stops, so the dashed
    # patience-2 line goes on top; xytext places each best-epoch label clear of the curves
    curves = [("BiLSTM, seed 2, patience 4", load("lstm", "output", 2, patience=4), "-", "o", 2.4, (0, -32)),
              ("BiLSTM, seed 2, patience 2", load("lstm", "output", 2), "--", "s", 1.6, (45, -55)),
              ("CNN, seed 0", load("cnn", "output", 0), "-", "o", 2.0, (0, 12))]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for label, r, ls, marker, lw, offset in curves:
        if r is None:
            print(f"  warning: missing run for {label}")
            continue
        color = MODEL_COLORS[r["model"]]
        ep = [h["epoch"] for h in r["history"]]
        loss = [h["train_loss"] for h in r["history"]]
        f1 = [h["val_macro_f1"] for h in r["history"]]
        b = r["best_epoch"]
        for ax, y in [(ax_loss, loss), (ax_f1, f1)]:
            ax.plot(ep, y, color=color, linestyle=ls, linewidth=lw, marker=marker, markersize=5, label=label)
            # best epoch: large ring on the curve
            ax.plot(b, y[ep.index(b)], marker="o", markersize=13, markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=1.8, linestyle="none")
        ax_f1.annotate(f"best ep {b}: {r['best_val_macro_f1']:.3f}", (b, f1[ep.index(b)]),
                       textcoords="offset points", xytext=offset, ha="center", va="center",
                       fontsize=8.5, color=INK_2,
                       arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8, "shrinkB": 8} if ls == "--" else None)
    lo, hi = ax_f1.get_ylim()
    ax_f1.set_ylim(lo, hi + 0.25 * (hi - lo))  # headroom for the top label
    ax_f1.plot([], [], marker="o", markersize=10, markerfacecolor="none", markeredgecolor=INK_2,
               linestyle="none", label="best epoch (checkpoint kept)")
    for ax, title, ylabel in [(ax_loss, "Training loss", "Cross-entropy"),
                              (ax_f1, "Validation macro-F1", "Macro-F1")]:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 11))
        style_axis(ax)
    ax_f1.legend(loc="lower right", frameon=False)
    fig.suptitle("Output-only training curves", fontsize=13)
    save(fig, "training_curves.png")


def fig_rq3_per_family():
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    seen, unseen = ["general", "math"], ["code"]
    tests = [("Seen domains (general + math)", seen, None, 1.0), ("Unseen domain (code)", unseen, "///", 0.55)]
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, constrained_layout=True)
    for ax, model in zip(axes, ["cnn", "lstm"]):
        patience = 4 if model == "lstm" else 2  # RQ3 used patience 4 for the lstm
        macros = []
        for k, (_, test_domains, hatch, alpha) in enumerate(tests):
            runs = load_seeds(model, "output", patience=patience, train_domains=seen, test_domains=test_domains)
            f1 = np.array([[r["test"]["per_class_f1"][f] for f in FAMILIES] for r in runs])
            mean, std = f1.mean(axis=0), f1.std(axis=0, ddof=1)
            macros.append(np.mean([r["test"]["macro_f1"] for r in runs]))
            x = np.arange(len(FAMILIES)) + (k - 0.5) * width
            ax.bar(x, mean, width * 0.92, yerr=std, capsize=3, color=[FAMILY_COLORS[f] for f in FAMILIES],
                   alpha=alpha, hatch=hatch, edgecolor="white", linewidth=0, error_kw={"ecolor": INK, "lw": 1})
            for xi, m, sd in zip(x, mean, std):
                bar_label(ax, xi, m, sd)
        ax.axhline(0.25, color=INK_2, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(FAMILIES)), FAMILIES)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{MODEL_NAMES[model]}" + (" (patience 4)" if patience == 4 else "")
                     + f"\nmacro-F1: seen {macros[0]:.3f}, unseen {macros[1]:.3f}")
        style_axis(ax)
    axes[0].set_ylabel("Test F1 (mean ± std, 3 seeds)")
    # neutral-gray proxies: hatch/opacity, not hue, encodes seen vs unseen; hue is the family
    handles = [Patch(facecolor="#8a8985", alpha=alpha, hatch=hatch, edgecolor="white", label=label)
               for label, _, hatch, alpha in tests]
    handles.append(Line2D([], [], color=INK_2, linestyle="--", linewidth=1, label="chance (0.25)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("RQ3: per-family F1, trained on general + math", fontsize=13)
    save(fig, "rq3_per_family.png")


def print_param_counts():
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH) as f:
            vocab_size = len(json.load(f))
    else:
        vocab_size = 30000  # train.MAX_VOCAB; the cap is hit (notes.md §2), so this matches vocab.json
        print(f"\n{VOCAB_PATH} not found (run src/train.py once); using vocab size {vocab_size}")
    print(f"\nTrainable parameters (vocab size {vocab_size}, {len(FAMILIES)} classes):")
    for name in ["cnn", "lstm"]:
        model = build_model(name, vocab_size, len(FAMILIES))
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emb = model.embedding.weight.numel()
        print(f"  {type(model).__name__:8} {total:>11,}  (embedding {emb:,}, rest {total - emb:,})")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_rq1_rq2()
    fig_confusion()
    fig_ablations()
    fig_training_curves()
    fig_rq3_per_family()
    print_param_counts()


if __name__ == "__main__":
    main()
