"""Step 1: build the modeling table from the JailbreakBench attack artifacts.

Stacks the jailbreak rows of every attack method and target LLM, drops rows
without a prompt or response, and adds
  - prompt features: length, mean word length, non-ASCII share, character
    entropy, and a 10-d TF-IDF embedding
  - benchmark scores of the target LLM
  - one-hot attack method columns
  - benchmark x feature interactions (used by the linear models)

Writes data/processed_data.csv, which cross_validation.py and analysis.py read.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw_jailbenchmark" / "attack-artifacts"
PROCESSED_PATH = DATA_DIR / "processed_data.csv"

# DSN: Don't say no, GCG: greedy coordinate gradient, JBC: Jailbreak chat,
# PAIR: Prompt Automatic Iterative Refinement, prompt_with_random_search (PRS)
FOLDERS = ["DSN", "GCG", "JBC", "PAIR", "prompt_with_random_search"]
MODELS = {
    "llama": "llama-2-7b-chat-hf.json",
    "vicuna": "vicuna-13b-v1.5.json",
    "gpt3.5": "gpt-3.5-turbo-1106.json",
    "gpt4": "gpt-4-0125-preview.json",
}
NULL_COLS = ["number_of_queries", "queries_to_jailbreak", "prompt_tokens", "response_tokens"]
# Columns with too many NaNs, and the row index inside each JSON file
DROP_COLS = NULL_COLS + ["jailbroken_llama_guard1", "index"]

# Benchmark scores of each target LLM
# Arena expert: https://github.com/lmarena/arena-catalog/blob/main/data/leaderboard-text.json
ARENA_EXPERT = {"llama": 1032, "vicuna": 992, "gpt3.5": 1067, "gpt4": 1203}
# EQ-Bench: https://huggingface.co/spaces/sam-paech/EQ-Bench-Leaderboard/raw/main/script.js
EQ_BENCH = {"llama": 36, "vicuna": 67, "gpt3.5": 72, "gpt4": 84}
# SORRY-Bench: https://arxiv.org/html/2406.14598v2, table 4. This might be a forward-looking bias
SORRY_BENCH = {"llama": 14, "vicuna": 32, "gpt3.5": 11, "gpt4": 26}

EMB_COLS = [f"emb_{i}" for i in range(10)]
NUMERIC_COLS = ["prompt_words", "avg_word_len", "non_ascii_share", "char_entropy"] + EMB_COLS
CAPABILITY_COLS = ["arena_expert", "EQ_bench"]
BENCH_COLS = CAPABILITY_COLS + ["sorry_bench"]
METHOD_COLS = [f"method_{folder}" for folder in FOLDERS]  # same order as pd.get_dummies
INTERACTION_COLS = [f"{b}_x_{c}" for cols in (NUMERIC_COLS, METHOD_COLS)
                    for b in BENCH_COLS for c in cols]


def char_share(text, test):
    """Share of non-space characters that pass `test`."""
    chars = [c for c in text if not c.isspace()]
    return sum(test(c) for c in chars) / len(chars) if chars else 0.0


def char_entropy(text):
    """Shannon entropy of the character distribution (how unpredictable the mix of characters is)."""
    if not text:
        return 0.0
    _, counts = np.unique(list(text), return_counts=True)
    prob = counts / counts.sum()
    return float(-(prob * np.log2(prob)).sum())


def load_raw_attacks():
    """Stack the jailbreak rows of every attack method and target LLM."""
    frames = []
    for folder in FOLDERS:
        for model, filename in MODELS.items():
            path = RAW_DIR / folder / filename
            if path.exists():  # not every attack was run on every LLM
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                df = pd.DataFrame(data["jailbreaks"])
                df[NULL_COLS] = df[NULL_COLS].astype("float64")  # all-null columns become NaN floats
                df["generating_method"] = folder
                df["model"] = model
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_prompt_features(attack_df):
    """Length, non-ASCII share, and character entropy of the prompt.

    Prompt/goal length ratio, type-token ratio, and directive counts were
    dropped because they were highly correlated with these features.
    """
    prompt = attack_df["prompt"]
    attack_df = attack_df.copy()
    attack_df["prompt_words"] = prompt.str.split().str.len()
    # Mean characters per word
    attack_df["avg_word_len"] = (
        prompt.str.replace(r"\s+", "", regex=True).str.len()
        / attack_df["prompt_words"].clip(lower=1)
    )
    # Share of characters outside plain English
    attack_df["non_ascii_share"] = prompt.apply(char_share, test=lambda c: ord(c) > 127)
    attack_df["char_entropy"] = prompt.apply(char_entropy)
    return attack_df


def add_prompt_embeddings(attack_df):
    """10 PCA components of a 50-d TF-IDF SVD of the prompt (small, so it doesn't overwhelm the other features)."""
    attack_df = attack_df.copy()
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))
    prompt_svd = TruncatedSVD(n_components=50, random_state=0).fit_transform(
        tfidf.fit_transform(attack_df["prompt"])
    )
    attack_df[EMB_COLS] = PCA(n_components=10, random_state=0).fit_transform(prompt_svd)
    return attack_df


def add_benchmarks_and_interactions(attack_df):
    """Benchmark scores of the target LLM, one-hot attack method, and their interactions."""
    attack_df = attack_df.copy()
    attack_df["arena_expert"] = attack_df["model"].map(ARENA_EXPERT)
    attack_df["EQ_bench"] = attack_df["model"].map(EQ_BENCH)
    attack_df["sorry_bench"] = attack_df["model"].map(SORRY_BENCH)
    attack_df = attack_df.join(
        pd.get_dummies(attack_df["generating_method"], prefix="method", dtype=int)
    )
    interactions = {f"{b}_x_{c}": attack_df[b] * attack_df[c]
                    for cols in (NUMERIC_COLS, METHOD_COLS) for b in BENCH_COLS for c in cols}
    return attack_df.assign(**interactions)


def load_attack_df():
    """Return the modeling table: one row per prompt, with the features and the `jailbroken` label."""
    attack_df = load_raw_attacks()
    # Remove rows without prompt or response
    attack_df = attack_df[~attack_df["prompt"].isna()]
    attack_df = attack_df[~attack_df["response"].isna()]
    attack_df = attack_df.drop(columns=DROP_COLS).reset_index(drop=True)
    attack_df = add_prompt_features(attack_df)
    attack_df = add_prompt_embeddings(attack_df)
    return add_benchmarks_and_interactions(attack_df)


if __name__ == "__main__":
    attack_df = load_attack_df()
    attack_df.to_csv(PROCESSED_PATH, index=False)
    print(f"Wrote {len(attack_df)} rows and {len(attack_df.columns)} columns to {PROCESSED_PATH}")
    print(attack_df.groupby(["model", "generating_method"]).size().unstack(fill_value=0))
    # Rows and jailbreak rate by target LLM and by attack method
    for col in ["model", "generating_method"]:
        print(attack_df.groupby(col)["jailbroken"].agg(rows="size", jailbroken="mean").round(2))
    print("jailbroken rate:", attack_df["jailbroken"].astype(int).mean())
