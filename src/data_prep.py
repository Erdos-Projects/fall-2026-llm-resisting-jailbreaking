"""Build the attack table used by cv_code.py.

Extracted from data_cleaning_v2.ipynb. `generating_method` is left as a
column so cv_code.py can one-hot encode it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw_jailbenchmark"
    / "attack-artifacts"
)

FOLDERS = ["DSN", "GCG", "JBC", "PAIR", "prompt_with_random_search"]
MODELS = {
    "llama": "llama-2-7b-chat-hf.json",
    "vicuna": "vicuna-13b-v1.5.json",
    "gpt3.5": "gpt-3.5-turbo-1106.json",
    "gpt4": "gpt-4-0125-preview.json",
}
NULL_COLS = [
    "number_of_queries",
    "queries_to_jailbreak",
    "prompt_tokens",
    "response_tokens",
]
DROP_COLS = NULL_COLS + ["jailbroken_llama_guard1", "index"]

# Model benchmark scores. Sources still need to be recorded.
ARENA_CREATIVE_WRITING = {
    "llama": 1035.9,
    "vicuna": 1059.3,
    "gpt3.5": 1036.2,
    "gpt4": 1233.7,
}
EQ_BENCH = {"llama": 36, "vicuna": 67, "gpt3.5": 72, "gpt4": 84}

BENCH_COLS = ["arena_creative_writing", "EQ_bench"]
EMB_COLS = [f"emb_{i}" for i in range(10)]


def char_share(text, test):
    """Share of non-space characters that pass `test`."""
    chars = [c for c in text if not c.isspace()]
    return sum(test(c) for c in chars) / len(chars) if chars else 0.0


def char_entropy(text):
    """Shannon entropy of the character distribution."""
    if not text:
        return 0.0
    _, counts = np.unique(list(text), return_counts=True)
    prob = counts / counts.sum()
    return float(-(prob * np.log2(prob)).sum())


def load_raw_attacks(data_dir=DATA_DIR):
    """Stack jailbreak rows from each attack method and model."""
    frames = []
    for folder in FOLDERS:
        for model, filename in MODELS.items():
            path = Path(data_dir) / folder / filename
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data["jailbreaks"])
            present = [c for c in NULL_COLS if c in df.columns]
            df[present] = df[present].astype("float64")
            df["generating_method"] = folder
            df["model"] = model
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No attack artifacts found under {data_dir}")
    return pd.concat(frames, ignore_index=True)


def add_prompt_features(attack_df):
    """Length, non-ASCII share, and character entropy of the prompt.

    Prompt/goal length ratio, type-token ratio, and directive counts were
    dropped because they were highly correlated with these features.
    """
    prompt = attack_df["prompt"]
    attack_df = attack_df.copy()
    attack_df["prompt_words"] = prompt.str.split().str.len()
    attack_df["avg_word_len"] = (
        prompt.str.replace(r"\s+", "", regex=True).str.len()
        / attack_df["prompt_words"].clip(lower=1)
    )
    attack_df["non_ascii_share"] = prompt.apply(
        char_share, test=lambda c: ord(c) > 127
    )
    attack_df["char_entropy"] = prompt.apply(char_entropy)
    return attack_df


def add_prompt_embeddings(attack_df):
    """10 PCA components of a 50-d TF-IDF SVD of the prompt."""
    attack_df = attack_df.copy()
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))
    prompt_svd = TruncatedSVD(n_components=50, random_state=0).fit_transform(
        tfidf.fit_transform(attack_df["prompt"])
    )
    attack_df[EMB_COLS] = PCA(n_components=10, random_state=0).fit_transform(
        prompt_svd
    )
    return attack_df


def add_benchmarks_and_interactions(attack_df):
    """Map each model to benchmark scores and multiply them by each embedding."""
    attack_df = attack_df.copy()
    attack_df["arena_creative_writing"] = attack_df["model"].map(
        ARENA_CREATIVE_WRITING
    )
    attack_df["EQ_bench"] = attack_df["model"].map(EQ_BENCH)
    interactions = {
        f"{bench}_emb_{i}": attack_df[bench] * attack_df[f"emb_{i}"]
        for bench in BENCH_COLS
        for i in range(10)
    }
    return attack_df.assign(**interactions)


def load_attack_df(data_dir=DATA_DIR):
    """Return the modeling table, still with a `generating_method` column."""
    attack_df = load_raw_attacks(data_dir)
    attack_df = attack_df[~attack_df["prompt"].isna()]
    attack_df = attack_df[~attack_df["response"].isna()]
    attack_df = attack_df.drop(
        columns=[c for c in DROP_COLS if c in attack_df.columns]
    )
    attack_df = attack_df.reset_index(drop=True)
    attack_df = add_prompt_features(attack_df)
    attack_df = add_prompt_embeddings(attack_df)
    attack_df = add_benchmarks_and_interactions(attack_df)
    return attack_df


if __name__ == "__main__":
    df = load_attack_df()
    print(f"rows={len(df)} cols={len(df.columns)}")
    print(df.groupby(["model", "generating_method"]).size().unstack(fill_value=0))
    print("jailbroken rate:", df["jailbroken"].astype(int).mean())
