"""TF-IDF + logistic regression baseline on outputs only -> results/tfidf/.

Two variants, each with at most 50k features and case kept:
  word: 1-2 grams over the project tokenizer (train.tokenize: words, punctuation, <nl> for newlines)
  char: 3-5 grams over the raw text (whitespace and newlines included)
Fit on train, C picked by val macro-F1, evaluated once on test. Outputs are not truncated.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from train import DATA_DIR, FAMILIES, ROOT, tokenize

OUT_DIR = os.path.join(ROOT, "results", "tfidf")
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MAX_FEATURES = 50000
TOP_K = 10
SEED = 0
VARIANTS = {
    "word_1-2": dict(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2)),
    "char_3-5": dict(analyzer="char", ngram_range=(3, 5)),
}


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=range(len(FAMILIES)), zero_division=0)


def fit(C, X, y):
    return LogisticRegression(C=C, max_iter=5000, random_state=SEED).fit(X, y)


def run_variant(name, kwargs, splits):
    vec = TfidfVectorizer(lowercase=False, max_features=MAX_FEATURES, sublinear_tf=False, **kwargs)
    Xtr = vec.fit_transform(splits["train"]["output"])
    Xva, Xte = vec.transform(splits["val"]["output"]), vec.transform(splits["test"]["output"])
    ytr, yva, yte = (splits[s]["family"].map(FAMILIES.index).to_numpy() for s in ["train", "val", "test"])

    val_scores = {C: macro_f1(yva, fit(C, Xtr, ytr).predict(Xva)) for C in C_GRID}
    best_C = max(C_GRID, key=lambda C: val_scores[C])  # ties -> smallest C (first in grid)
    model = fit(best_C, Xtr, ytr)
    p = model.predict(Xte)

    vocab = vec.get_feature_names_out()
    top = {fam: [{"ngram": vocab[i], "coef": float(model.coef_[k, i])} for i in np.argsort(-model.coef_[k])[:TOP_K]]
           for k, fam in enumerate(FAMILIES)}
    result = {
        "variant": name, "max_features": MAX_FEATURES, "n_features": len(vocab), "lowercase": False,
        "C_grid": C_GRID, "val_macro_f1_by_C": {str(c): s for c, s in val_scores.items()}, "best_C": best_C,
        "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
        "test": {
            "accuracy": accuracy_score(yte, p), "macro_f1": macro_f1(yte, p),
            "per_class_f1": dict(zip(FAMILIES, f1_score(yte, p, average=None, labels=range(len(FAMILIES)), zero_division=0).tolist())),
            "confusion_matrix": confusion_matrix(yte, p, labels=range(len(FAMILIES))).tolist(),
        },
        f"top{TOP_K}_ngrams": top, "labels": FAMILIES,
    }
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)

    t = result["test"]
    print(f"[{name}] {len(vocab)} features; val macro-F1 by C: "
          + ", ".join(f"{c}={s:.4f}" for c, s in val_scores.items()) + f" -> best C={best_C}")
    print(f"[{name}] test accuracy {t['accuracy']:.4f}  macro-F1 {t['macro_f1']:.4f}  per-class F1 "
          + ", ".join(f"{k}={v:.4f}" for k, v in t["per_class_f1"].items()))
    for fam, items in top.items():
        print(f"  {fam:>6}: " + " | ".join(repr(d["ngram"]) for d in items))
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    splits = {s: pd.read_parquet(os.path.join(DATA_DIR, f"{s}.parquet")) for s in ["train", "val", "test"]}
    results = {name: run_variant(name, kw, splits) for name, kw in VARIANTS.items()}

    summary = pd.DataFrame([{"variant": n, "best_C": r["best_C"], "n_features": r["n_features"],
                             "accuracy": r["test"]["accuracy"], "macro_f1": r["test"]["macro_f1"],
                             **{f"{f}_f1": r["test"]["per_class_f1"][f] for f in FAMILIES}} for n, r in results.items()])
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    top = pd.DataFrame([{"variant": n, "family": fam, "rank": i + 1, "ngram": d["ngram"], "coef": d["coef"]}
                        for n, r in results.items() for fam, items in r[f"top{TOP_K}_ngrams"].items()
                        for i, d in enumerate(items)])
    top.to_csv(os.path.join(OUT_DIR, "top_ngrams.csv"), index=False)
    print(f"wrote {OUT_DIR}/{{{','.join(VARIANTS)}}}.json, summary.csv, top_ngrams.csv")


if __name__ == "__main__":
    main()
