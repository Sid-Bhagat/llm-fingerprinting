"""RQ4: logistic regression on the handcrafted features from features.py -> results/rq4/feature_lr.json.

Standardized features, fit on train, C picked by val macro-F1, evaluated once on test.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import DATA_DIR, FAMILIES, OUT_DIR, featurize

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 0


def load(split):
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{split}.parquet"))
    return featurize(df), df["family"].map(FAMILIES.index).to_numpy()


def fit(C, X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=SEED)).fit(X, y)


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=range(len(FAMILIES)), zero_division=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load("train"), load("val"), load("test")
    features = list(Xtr.columns)

    val_scores = {C: macro_f1(yva, fit(C, Xtr, ytr).predict(Xva)) for C in C_GRID}
    best_C = max(C_GRID, key=lambda C: val_scores[C])  # ties -> smallest C (first in grid)
    model = fit(best_C, Xtr, ytr)
    p = model.predict(Xte)
    coef = model[-1].coef_  # (n_classes, n_features), on standardized features

    top = {}
    for k, fam in enumerate(FAMILIES):
        order = np.argsort(-coef[k])[:3]
        top[fam] = [{"feature": features[i], "coef": float(coef[k, i])} for i in order if coef[k, i] > 0]

    result = {
        "features": features, "C_grid": C_GRID, "val_macro_f1_by_C": {str(c): s for c, s in val_scores.items()},
        "best_C": best_C, "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
        "test": {
            "accuracy": accuracy_score(yte, p), "macro_f1": macro_f1(yte, p),
            "per_class_f1": dict(zip(FAMILIES, f1_score(yte, p, average=None, labels=range(len(FAMILIES)), zero_division=0).tolist())),
            "confusion_matrix": confusion_matrix(yte, p, labels=range(len(FAMILIES))).tolist(),
        },
        "top3_positive_coefs": top, "coef": {fam: dict(zip(features, coef[k].tolist())) for k, fam in enumerate(FAMILIES)},
        "labels": FAMILIES,
    }
    path = os.path.join(OUT_DIR, "feature_lr.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print("val macro-F1 by C: " + ", ".join(f"{c}={s:.4f}" for c, s in val_scores.items()) + f"  -> best C={best_C}")
    t = result["test"]
    print(f"test accuracy {t['accuracy']:.4f}  macro-F1 {t['macro_f1']:.4f}")
    print("per-class F1: " + ", ".join(f"{k}={v:.4f}" for k, v in t["per_class_f1"].items()))
    print("confusion matrix (rows = true, cols = predicted, order " + "/".join(FAMILIES) + "):")
    for fam, row in zip(FAMILIES, t["confusion_matrix"]):
        print(f"  {fam:>6} {row}")
    print("top 3 positive coefficients (standardized features):")
    for fam, items in top.items():
        print(f"  {fam:>6}: " + ", ".join(f"{d['feature']} {d['coef']:+.3f}" for d in items))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
