"""Build data/processed/{train,val,test}.parquet from lmsys/chatbot_arena_conversations."""
import os
import re

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "data", "processed")
SEED = 42

FAMILY_MAP = {
    "gpt-4": "GPT", "gpt-3.5-turbo": "GPT",
    "claude-v1": "Claude", "claude-instant-v1": "Claude",
    "vicuna-13b": "Llama", "vicuna-7b": "Llama", "koala-13b": "Llama", "alpaca-13b": "Llama",
    "wizardlm-13b": "Llama", "gpt4all-13b-snoozy": "Llama", "guanaco-33b": "Llama",
    "palm-2": "PaLM",
}
FAMILIES = ["GPT", "Claude", "Llama", "PaLM"]

SELF_ID_RE = re.compile(
    r"\b(?:claude|anthropic|openai|chatgpt|gpt-?4|gpt-?3|vicuna|lmsys|koala|alpaca|palm|bard|google|trained by)\b",
    re.IGNORECASE,
)

CODE_RE = re.compile(
    r"```|#include|console\.log|print\(|=>|\b(def|import|return|function|class|public|void|"
    r"python|javascript|typescript|java|c\+\+|c#|sql|html|css|bash|regex|code|script|compile|"
    r"debug|api|json|array|variable|loop|programming|program)\b",
    re.IGNORECASE,
)
MATH_RE = re.compile(
    r"\b(math|maths|mathematical|equation|equations|calculate|integral|derivative|algebra|geometry|"
    r"arithmetic|probability|factorial|prime numbers?|square root|logarithm|theorem|integers?|"
    r"hexadecimal|polynomial|calculus|divisible|divided by|multiplied by|multiply|subtract|"
    r"percent|percentage of|average of|fraction|ratio of|solve for)\b",
    re.IGNORECASE,
)
# digit-operator-digit, e.g. "2+2", "3 * 4", "x^2", "5 = 7"; minus and "x" only with spaces so ranges/sizes don't count
ARITH_RE = re.compile(r"\d\s*[+*/×^=]\s*\d|\d \- \d|\d x \d")
WORD_PROBLEM_RE = re.compile(r"\bhow (many|much|long|old|far)\b", re.IGNORECASE)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
DATE_RE = re.compile(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}")


def domain_of(prompt):
    if CODE_RE.search(prompt):
        return "code"
    chars = [c for c in prompt if not c.isspace()]
    digit_frac = sum(c.isdigit() or c in "+-*/^=().%" for c in chars) / max(len(chars), 1)
    word_problem = WORD_PROBLEM_RE.search(prompt) and len(NUM_RE.findall(prompt)) >= 2
    no_dates = DATE_RE.sub(" ", prompt)
    if MATH_RE.search(prompt) or ARITH_RE.search(no_dates) or word_problem or digit_frac > 0.5:
        return "math"
    return "general"


def counts(df, title):
    print(f"\n--- {title} (total {len(df)})")
    print(df["family"].value_counts().reindex(FAMILIES, fill_value=0).to_string())


def load_responses():
    load_dotenv(os.path.join(ROOT, ".env"))
    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train",
                      token=os.environ["HF_TOKEN"], cache_dir=RAW_DIR)
    df = ds.to_pandas()
    df = df[(df["language"] == "English") & (df["turn"] == 1)]
    rows = []
    for side in ("a", "b"):
        part = pd.DataFrame({
            "question_id": df["question_id"],
            "model": df[f"model_{side}"],
            "input": df[f"conversation_{side}"].apply(lambda c: c[0]["content"]),
            "output": df[f"conversation_{side}"].apply(lambda c: c[1]["content"]),
        })
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out["family"] = out["model"].map(FAMILY_MAP)
    out = out.dropna(subset=["family"])
    out = out[out["output"].str.strip() != ""]
    out = out.drop_duplicates(subset=["input", "output"]).reset_index(drop=True)
    return out


def split_by_question(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, rest_idx = next(gss.split(df, groups=df["question_id"]))
    train, rest = df.iloc[train_idx], df.iloc[rest_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(gss.split(rest, groups=rest["question_id"]))
    return {"train": train, "val": rest.iloc[val_idx], "test": rest.iloc[test_idx]}


def balance(df):
    n = df["family"].value_counts().min()
    out = df.groupby("family", group_keys=False).sample(n=n, random_state=SEED)
    return out.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_responses()
    counts(df, "after filtering (single-turn English, mapped families, non-empty, deduped)")

    leaked = df["output"].str.contains(SELF_ID_RE)
    print("\n--- self-ID responses dropped per family")
    print(df.loc[leaked, "family"].value_counts().reindex(FAMILIES, fill_value=0).to_string())
    df = df[~leaked].reset_index(drop=True)
    counts(df, "after self-ID drop")

    df["domain"] = df["input"].apply(domain_of)

    splits = {name: balance(part) for name, part in split_by_question(df).items()}
    cols = ["question_id", "model", "family", "input", "output", "domain"]
    for name, part in splits.items():
        counts(part, f"final {name}")
        part[cols].to_parquet(os.path.join(OUT_DIR, f"{name}.parquet"), index=False)

    ids = {name: set(part["question_id"]) for name, part in splits.items()}
    print("\n--- question_id overlap between splits")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"{a} & {b}: {len(ids[a] & ids[b])}")

    final = pd.concat(splits.values(), ignore_index=True)
    print("\n--- domain counts per family (all splits)")
    print(pd.crosstab(final["family"], final["domain"]).reindex(FAMILIES).to_string())

    print("\n--- 2 random final examples per family")
    for fam in FAMILIES:
        for _, r in final[final["family"] == fam].sample(n=2, random_state=SEED).iterrows():
            print(f"\n[{fam} / {r['model']} / {r['domain']}]")
            print(f"INPUT: {r['input'][:300]!r}")
            print(f"OUTPUT: {r['output'][:300]!r}")

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
