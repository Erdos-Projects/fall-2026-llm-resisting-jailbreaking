"""Step 3: the chosen models on the held-out test rows.

Item 1: test results vs. two simple baselines, and how the tree models do with fewer features
Item 2: gradient boosting results for each attack method and LLM
Item 3: does mixing all four LLMs together help? One model on all four vs. one model per LLM
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from cross_validation import (FINAL_MODELS, THRESHOLD, get_metrics, load_split, predict_score,
                              scale_and_clip)
from data_prep import BENCH_COLS, CAPABILITY_COLS, FOLDERS, METHOD_COLS, NUMERIC_COLS

LLM_ORDER = ["llama", "vicuna", "gpt3.5", "gpt4"]
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
TREE_MODELS = ["Decision tree", "Random forest", "Gradient boosting", "XGBoost"]
# Item 1: feature sets for the tree models
FEATURE_SETS = {
    "All features":      NUMERIC_COLS + BENCH_COLS + METHOD_COLS,
    "No sorry_bench":    NUMERIC_COLS + CAPABILITY_COLS + METHOD_COLS,
    "No benchmarks":     NUMERIC_COLS + METHOD_COLS,
    "No method columns": NUMERIC_COLS + BENCH_COLS,
}


def get_scores(train_part, test_part, models):
    """Fit every (model, features) pair on train_part; one row per model and test prompt."""
    train_rows, test_rows = scale_and_clip(train_part, test_part)
    y_part = train_part["jailbroken"].astype(int)
    output = []
    for model_name, (model, features) in models.items():
        fitted = clone(model).fit(train_rows[features], y_part)
        output.append(pd.DataFrame({"ml_model": model_name,
                                    "llm": test_part["model"],
                                    "generating_method": test_part["generating_method"],
                                    "actual": test_part["jailbroken"].astype(int),
                                    "score": predict_score(fitted, test_rows[features])}))
    return pd.concat(output)


def metrics_by(scores, keys):
    """get_metrics for each group of test rows; one row per group, one column per metric."""
    return scores.groupby(keys)[["actual", "score"]].apply(
        lambda group: pd.Series(get_metrics(group["actual"], group["score"])))


def attack_llm_metrics(group):
    """Item 2 metrics for one attack method and LLM."""
    actual, predicted = group["actual"], (group["score"] >= THRESHOLD).astype(int)
    return pd.Series({"f1":        f1_score(actual, predicted, zero_division=np.nan),
                      "accuracy":  accuracy_score(actual, predicted),
                      "precision": precision_score(actual, predicted, zero_division=np.nan),
                      "recall":    recall_score(actual, predicted, zero_division=np.nan)})


def show(metrics, title, rows, columns, names=METRICS):
    """One table per metric: `rows` down, the last grouping key (`columns`) across."""
    for metric in names:
        print(f"\n{title}: {metric}")
        print(metrics[metric].unstack().loc[rows, columns].round(3).to_string())


def item1(train_df, test_df, pooled):
    """Test results of the chosen models and two baselines, then a feature ablation."""
    # Baselines: score each test row with its group's jailbreak rate on the training rows
    baselines = {"Baseline: attack method": "generating_method",
                 "Baseline: attack + LLM":  "attack_and_llm"}
    y_train = train_df["jailbroken"].astype(int)
    y_test = test_df["jailbroken"].astype(int)
    baseline_rows = {}
    for name, group_col in baselines.items():
        rate = y_train.groupby(train_df[group_col]).mean()
        baseline_rows[name] = get_metrics(y_test, test_df[group_col].map(rate))
    results = pd.concat([pd.DataFrame(baseline_rows).T,
                         metrics_by(pooled, ["ml_model"]).loc[list(FINAL_MODELS)]])
    print("\nItem 1: test results")
    print(results.round(3).to_string())

    # How do the tree models do with fewer features?
    ablation = pd.concat([
        get_scores(train_df, test_df, {name: (FINAL_MODELS[name][0], features)
                                       for name in TREE_MODELS}).assign(features=set_name)
        for set_name, features in FEATURE_SETS.items()])
    show(metrics_by(ablation, ["features", "ml_model"]), "Item 1: test, fewer features",
         list(FEATURE_SETS), TREE_MODELS)


def item2(pooled):
    """Gradient boosting (all features) on each attack method and LLM of the test rows."""
    gb = pooled[pooled["ml_model"] == "Gradient boosting"]
    by_group = gb.groupby(["generating_method", "llm"])[["actual", "score"]].apply(attack_llm_metrics)
    # NaN means the score can't be calculated:
    #   every metric: no rows for this attack and LLM
    #   recall:       no real jailbreaks
    #   precision:    nothing was flagged as a jailbreak
    #   f1:           no real jailbreaks and nothing was flagged
    show(by_group, "Item 2: gradient boosting, test", FOLDERS, LLM_ORDER,
         ["f1", "accuracy", "precision", "recall"])


def item3(train_df, test_df, pooled):
    """One model trained on all four LLMs vs. a separate model for each LLM, on the same test rows."""
    # Benchmarks and their interactions have one value per LLM, so they're left out when training on one LLM
    own_llm_models = {name: (model, [c for c in features if c in NUMERIC_COLS + METHOD_COLS])
                      for name, (model, features) in FINAL_MODELS.items()}
    single = [get_scores(train_df[train_df["model"] == llm], test_df[test_df["model"] == llm],
                         own_llm_models)
              for llm in LLM_ORDER]
    all_scores = pd.concat([pooled.assign(training="All four LLMs"),
                            pd.concat(single).assign(training="Own LLM only")])
    # "all" scores every test row together, so the four separate models can be compared with one pooled model
    all_scores = pd.concat([all_scores, all_scores.assign(llm="all")])
    show(metrics_by(all_scores, ["llm", "training", "ml_model"]), "Item 3: test",
         LLM_ORDER + ["all"], list(FINAL_MODELS))


def main():
    train_df, test_df = load_split()
    for df in (train_df, test_df):
        df["attack_and_llm"] = df["generating_method"] + " / " + df["model"]
    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    # Every chosen model trained on all four LLMs; used by all three items
    pooled = get_scores(train_df, test_df, FINAL_MODELS)
    item1(train_df, test_df, pooled)
    item2(pooled)
    item3(train_df, test_df, pooled)


if __name__ == "__main__":
    main()
