"""Train a TextCNN or BiLSTM family classifier and write results/{run_name}.json."""
import argparse
import json
import os
import random
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from models import PAD_ID, build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
FAMILIES = ["GPT", "Claude", "Llama", "PaLM"]
DOMAINS = ["general", "code", "math"]

SPECIALS = ["<pad>", "<unk>", "<nl>", "<sep>"]
UNK_ID, NL, SEP = 1, "<nl>", "<sep>"
MIN_FREQ, MAX_VOCAB = 2, 30000
INPUT_LEN, OUTPUT_LEN, BOTH_INPUT_LEN, BOTH_OUTPUT_LEN = 64, 256, 64, 192
DEFAULT_OUT_LEN = {"input": None, "output": OUTPUT_LEN, "both": BOTH_OUTPUT_LEN}  # input mode has no output
DEFAULT_PATIENCE = 2

# newline, word, or single punctuation char; case is kept
TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]")


def tokenize(text):
    return [NL if t == "\n" else t for t in TOKEN_RE.findall(text)]


def build_vocab(train):
    """Built from train inputs + outputs only, so every mode and run shares one vocab."""
    counts = Counter()
    for text in pd.concat([train["input"], train["output"]]):
        counts.update(tokenize(text))
    words = [w for w, c in counts.most_common() if c >= MIN_FREQ and w not in SPECIALS]
    itos = SPECIALS + words[: MAX_VOCAB - len(SPECIALS)]
    with open(VOCAB_PATH, "w") as f:
        json.dump(itos, f)
    print(f"built vocab: {len(itos)} tokens -> {VOCAB_PATH}")
    return itos


def load_vocab(train):
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH) as f:
            return json.load(f)
    return build_vocab(train)


# ---------- ablations (applied to the model output text, in every split) ----------
MD_FENCE_RE = re.compile(r"```[^\n]*")
MD_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
MD_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
MD_EMPH_RE = re.compile(r"(\*\*|__|\*|`)")
MULTI_NL_RE = re.compile(r"\n{2,}")


def strip_markdown(text):
    text = MD_FENCE_RE.sub("", text)
    text = MD_HEADER_RE.sub("", text)
    text = MD_BULLET_RE.sub("", text)
    text = MD_EMPH_RE.sub("", text)
    return MULTI_NL_RE.sub("\n", text)


def apply_ablation(df, ablation):
    df = df.copy()
    if ablation == "strip_markdown":
        df["output"] = df["output"].apply(strip_markdown)
    return df


def encode(df, stoi, mode, ablation, max_out_len):
    def ids(tokens, n):
        return [stoi.get(t, UNK_ID) for t in tokens[:n]]

    out_cap = 100 if ablation == "truncate100" else None
    seqs = []
    for inp, outp in zip(df["input"], df["output"]):
        out_tok = tokenize(outp)[:out_cap]
        if mode == "input":
            s = ids(tokenize(inp), INPUT_LEN)
        elif mode == "output":
            s = ids(out_tok, max_out_len)
        else:
            s = ids(tokenize(inp), BOTH_INPUT_LEN) + [stoi[SEP]] + ids(out_tok, max_out_len)
        seqs.append(s or [UNK_ID])
    labels = [FAMILIES.index(f) for f in df["family"]]
    return seqs, labels


def batches(seqs, labels, batch_size, shuffle, gen=None):
    order = torch.randperm(len(seqs), generator=gen).tolist() if shuffle else range(len(seqs))
    order = list(order)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        width = max(5, max(len(seqs[j]) for j in idx))  # CNN needs >= largest filter size
        x = torch.full((len(idx), width), PAD_ID, dtype=torch.long)
        for r, j in enumerate(idx):
            x[r, : len(seqs[j])] = torch.tensor(seqs[j])
        yield x, torch.tensor([labels[j] for j in idx])


@torch.no_grad()
def predict(model, seqs, labels, device, batch_size):
    model.eval()
    preds = []
    for x, _ in batches(seqs, labels, batch_size, shuffle=False):
        preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
    return preds


def metrics(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", labels=range(len(FAMILIES)), zero_division=0),
        "per_class_f1": dict(zip(FAMILIES, f1_score(y, p, average=None, labels=range(len(FAMILIES)), zero_division=0).tolist())),
        "confusion_matrix": confusion_matrix(y, p, labels=range(len(FAMILIES))).tolist(),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_name(a):
    name = f"{a.model}_{a.mode}_s{a.seed}"
    if a.ablation != "none":
        name += f"_{a.ablation}"
    if a.train_domains != DOMAINS or a.test_domains != DOMAINS:
        name += f"_tr-{'+'.join(a.train_domains)}_te-{'+'.join(a.test_domains)}"
    if a.max_out_len != DEFAULT_OUT_LEN[a.mode]:
        name += f"_maxout{a.max_out_len}"
    if a.patience != DEFAULT_PATIENCE:
        name += f"_pat{a.patience}"
    if a.train_frac < 1:
        name += f"_frac{a.train_frac}"
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "lstm"], required=True)
    p.add_argument("--mode", choices=["input", "output", "both"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablation", choices=["none", "strip_markdown", "truncate100"], default="none")
    p.add_argument("--train_domains", nargs="+", choices=DOMAINS, default=DOMAINS)
    p.add_argument("--test_domains", nargs="+", choices=DOMAINS, default=DOMAINS)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--max_out_len", type=int, default=None,
                   help=f"output tokens kept (default {OUTPUT_LEN} in output mode, {BOTH_OUTPUT_LEN} in both mode)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_frac", type=float, default=1.0, help="subsample train (smoke tests)")
    p.add_argument("--log_every", type=int, default=0, help="print train loss every N batches (0 = off)")
    p.add_argument("--results_dir", default=os.path.join(ROOT, "results"))
    a = p.parse_args()
    if a.max_out_len is None:
        a.max_out_len = DEFAULT_OUT_LEN[a.mode]
    elif a.mode == "input":
        p.error("--max_out_len has no effect in input mode")
    a.train_domains, a.test_domains = sorted(a.train_domains, key=DOMAINS.index), sorted(a.test_domains, key=DOMAINS.index)

    set_seed(a.seed)
    os.makedirs(a.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = run_name(a)

    splits = {s: pd.read_parquet(os.path.join(DATA_DIR, f"{s}.parquet")) for s in ["train", "val", "test"]}
    itos = load_vocab(splits["train"])  # always the full train split, before any filtering
    stoi = {w: i for i, w in enumerate(itos)}

    # val follows the training domains (it drives early stopping); test uses the test domains
    splits["train"] = splits["train"][splits["train"]["domain"].isin(a.train_domains)]
    splits["val"] = splits["val"][splits["val"]["domain"].isin(a.train_domains)]
    splits["test"] = splits["test"][splits["test"]["domain"].isin(a.test_domains)]
    if a.train_frac < 1:
        splits["train"] = splits["train"].sample(frac=a.train_frac, random_state=a.seed)
    data = {s: encode(apply_ablation(df, a.ablation), stoi, a.mode, a.ablation, a.max_out_len) for s, df in splits.items()}
    print(f"[{name}] device={device} vocab={len(itos)} " + " ".join(f"{s}={len(d[1])}" for s, d in data.items()))

    model = build_model(a.model, len(itos), len(FAMILIES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(a.seed)

    best_f1, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(batches(*data["train"], a.batch_size, True, gen), 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
            if a.log_every and step % a.log_every == 0:
                print(f"  epoch {epoch} batch {step} loss {loss.item():.4f}")
        val_f1 = metrics(data["val"][1], predict(model, *data["val"], device, a.batch_size))["macro_f1"]
        secs = time.time() - t0
        history.append({"epoch": epoch, "train_loss": total / n, "val_macro_f1": val_f1, "seconds": secs})
        print(f"  epoch {epoch} train_loss {total / n:.4f} val_macro_f1 {val_f1:.4f} ({secs:.1f}s)")
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    test = metrics(data["test"][1], predict(model, *data["test"], device, a.batch_size))
    result = {
        "run_name": name, **{k: v for k, v in vars(a).items() if k != "results_dir"},
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "history": history,
        "n_train": len(data["train"][1]), "n_val": len(data["val"][1]), "n_test": len(data["test"][1]),
        "test": test, "labels": FAMILIES,
    }
    path = os.path.join(a.results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{name}] test acc {test['accuracy']:.4f} macro_f1 {test['macro_f1']:.4f} -> {path}")


if __name__ == "__main__":
    main()
