"""RQ4: per-family stylistic features of model outputs -> results/rq4/.

Writes feature_stats.csv (mean/median per family), feature_separation.csv (Kruskal-Wallis H
per feature), box plots of the 4 most family-separating features, and openings.csv.
"""
import os
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
OUT_DIR = os.path.join(ROOT, "results", "rq4")
FAMILIES = ["GPT", "Claude", "Llama", "PaLM"]
COLORS = dict(zip(FAMILIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # fixed slot per family
TTR_WINDOW = 100

WORD_RE = re.compile(r"\w+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")

# name -> description; per-100-word rates normalize for response length
FEATURES = {
    "word_count": "words (\\w+ tokens)",
    "char_count": "characters",
    "ttr_100": f"type-token ratio over the first {TTR_WINDOW} lowercased words",
    "sentence_count": "sentences (split on .!? + whitespace, or newlines)",
    "avg_sentence_len": "words per sentence",
    "newline_rate": "newlines per 100 words",
    "list_rate": "share of non-empty lines that are bullet or numbered items",
    "header_rate": "share of non-empty lines that are markdown headers",
    "code_block_rate": "fenced code blocks per response",
    "bold_rate": "**bold** spans per 100 words",
    "question_rate": "'?' per 100 words",
    "exclamation_rate": "'!' per 100 words",
}
AXIS_LABELS = {
    "word_count": "words", "char_count": "characters", "ttr_100": "TTR (first 100 words)",
    "sentence_count": "sentences", "avg_sentence_len": "words per sentence",
    "newline_rate": "newlines per 100 words", "list_rate": "share of lines that are list items",
    "header_rate": "share of lines that are headers", "code_block_rate": "code blocks per output",
    "bold_rate": "bold spans per 100 words", "question_rate": "'?' per 100 words",
    "exclamation_rate": "'!' per 100 words",
}


def text_features(text):
    words = WORD_RE.findall(text)
    n_words = len(words)
    per100 = 100 / max(n_words, 1)
    lines = [l for l in text.split("\n") if l.strip()]
    n_lines = max(len(lines), 1)
    sentences = [s for s in SENT_SPLIT_RE.split(text) if WORD_RE.search(s)]
    head = [w.lower() for w in words[:TTR_WINDOW]]
    return {
        "word_count": n_words,
        "char_count": len(text),
        "ttr_100": len(set(head)) / max(len(head), 1),
        "sentence_count": len(sentences),
        "avg_sentence_len": n_words / max(len(sentences), 1),
        "newline_rate": text.count("\n") * per100,
        "list_rate": sum(bool(BULLET_RE.match(l)) for l in lines) / n_lines,
        "header_rate": sum(bool(HEADER_RE.match(l)) for l in lines) / n_lines,
        "code_block_rate": len(FENCE_RE.findall(text)) // 2,
        "bold_rate": len(BOLD_RE.findall(text)) * per100,
        "question_rate": text.count("?") * per100,
        "exclamation_rate": text.count("!") * per100,
    }


def featurize(df):
    return pd.DataFrame([text_features(t) for t in df["output"]], index=df.index)


def load_all():
    return pd.concat(
        [pd.read_parquet(os.path.join(DATA_DIR, f"{s}.parquet")).assign(split=s) for s in ["train", "val", "test"]],
        ignore_index=True,
    )


def opening(text, n=3):
    words = text.split()
    return " ".join(words[:n]) if len(words) >= n else None


def box_plots(feats, families, top):
    """Box per family; a feature that is 0 in most outputs of every family (box collapses to a line)
    is shown as the share of outputs where it is > 0 instead."""
    fig, axes = plt.subplots(1, len(top), figsize=(3.4 * len(top), 3.8))
    for ax, name in zip(axes, top):
        data = [feats.loc[families == f, name] for f in FAMILIES]
        if all(d.quantile(0.75) == 0 for d in data):
            shares = [(d > 0).mean() for d in data]
            bars = ax.bar(FAMILIES, shares, color=[COLORS[f] for f in FAMILIES], width=0.6)
            ax.bar_label(bars, labels=[f"{v:.1%}" for v in shares], fontsize=8, padding=2)
            ax.set_ylabel(f"share of outputs with {name} > 0", fontsize=8)
            ax.set_title(f"{name} (mostly zero)", fontsize=10)
        else:
            bp = ax.boxplot(data, tick_labels=FAMILIES, showfliers=False, patch_artist=True, widths=0.6,
                            medianprops={"color": "#1a1a19", "linewidth": 1.5})
            for patch, f in zip(bp["boxes"], FAMILIES):
                patch.set(facecolor=COLORS[f], alpha=0.85, edgecolor="#1a1a19", linewidth=0.8)
            ax.set_ylabel(AXIS_LABELS.get(name, name), fontsize=8)
            ax.set_title(name, fontsize=10)
        ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("Most family-separating output features (Kruskal-Wallis H; box outliers hidden)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "top4_features_boxplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_all()
    feats = featurize(df)
    fam = df["family"]
    print(f"{len(df)} outputs (train+val+test): " + ", ".join(f"{f}={(fam == f).sum()}" for f in FAMILIES))

    stats = feats.groupby(fam).agg(["mean", "median"]).reindex(FAMILIES)
    stats.columns = [f"{feat}_{stat}" for feat, stat in stats.columns]
    stats.T.round(4).to_csv(os.path.join(OUT_DIR, "feature_stats.csv"), index_label="feature_stat")

    sep = pd.DataFrame([
        {"feature": name, "kruskal_H": kruskal(*[feats.loc[fam == f, name] for f in FAMILIES]).statistic,
         "description": desc}
        for name, desc in FEATURES.items()
    ]).sort_values("kruskal_H", ascending=False)
    sep.to_csv(os.path.join(OUT_DIR, "feature_separation.csv"), index=False)
    short = (feats["word_count"] < TTR_WINDOW).groupby(fam).mean().reindex(FAMILIES)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\nmean (median) per family:")
        print(pd.DataFrame({f: [f"{stats.loc[f, n + '_mean']:.3f} ({stats.loc[f, n + '_median']:.3f})"
                                for n in FEATURES] for f in FAMILIES}, index=list(FEATURES)).to_string())
        print("\nfeatures ranked by Kruskal-Wallis H:")
        print(sep[["feature", "kruskal_H"]].to_string(index=False))
    print(f"\nshare of outputs under {TTR_WINDOW} words (ttr_100 uses the whole text there): "
          + ", ".join(f"{f}={v:.2f}" for f, v in short.items()))

    top = sep["feature"].head(4).tolist()
    print(f"\nwrote {box_plots(feats, fam, top)}")

    rows = []
    print("\ntop 10 opening 3-word phrases per family:")
    for f in FAMILIES:
        ops = df.loc[fam == f, "output"].map(opening).dropna()
        print(f"  {f} (n={len(ops)})")
        for rank, (phrase, count) in enumerate(Counter(ops).most_common(10), 1):
            rows.append({"family": f, "rank": rank, "phrase": phrase, "count": count, "share": count / len(ops)})
            print(f"    {rank:2d}. {count:4d} ({count / len(ops):.1%})  {phrase!r}")
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT_DIR, "openings.csv"), index=False)
    print(f"wrote {OUT_DIR}/feature_stats.csv, feature_separation.csv, openings.csv")


if __name__ == "__main__":
    main()
