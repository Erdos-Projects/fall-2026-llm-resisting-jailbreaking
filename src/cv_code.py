from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, precision_score,
                             recall_score, accuracy_score, f1_score, roc_auc_score, log_loss)

from data_prep import load_attack_df

attack_df = load_attack_df()

# ---------------------------------------------------------------
# 1. One-hot encode generating_method -> 5 columns of 0/1 (method_DSN, method_GCG, ...)
# ---------------------------------------------------------------
attack_df = pd.get_dummies(attack_df, columns=["generating_method"], prefix="method", dtype=int)
method_cols = [c for c in attack_df.columns if c.startswith("method_")]

# ---------------------------------------------------------------
# 2. Feature lists
# ---------------------------------------------------------------
numeric_cols = ['prompt_words', 'avg_word_len', 'non_ascii_share', 'char_entropy',
                'emb_0', 'emb_1', 'emb_2', 'emb_3', 'emb_4',
                'emb_5', 'emb_6', 'emb_7', 'emb_8', 'emb_9']
#bench_cols = ['arena_math', 'eqbench_v2']          # your benchmark columns (use at most 2, see notes)
#interaction_cols = [f"{b}_x_emb_{i}" for b in bench_cols for i in range(10)]  # your benchmark x embedding columns
bench_cols = ['arena_creative_writing', 'EQ_bench']
interaction_cols = [f"{b}_emb_{i}" for b in bench_cols for i in range(10)]
# 'model' is NOT a feature: GPT-4 is the test set, so its category never appears in training.
# The benchmark columns are what tell the model "this is a different LLM".

# Linear models: drop one method column. The intercept already covers it, so keeping all 5 is perfectly collinear.
linear_features = numeric_cols + bench_cols + interaction_cols + method_cols[1:]
# Trees: keep all 5 method columns, skip the interaction columns (trees build interactions on their own).
tree_features = numeric_cols + bench_cols + method_cols
# Columns to standardize (never the 0/1 method columns)
scale_cols = numeric_cols + bench_cols + interaction_cols

# ---------------------------------------------------------------
# 3. Test set = GPT-4 (kept aside until the very end). Everything else is used for 5-fold CV.
#    (No train_test_split anymore: the 5 folds replace the single validation split.)
# ---------------------------------------------------------------
train_val_data = attack_df[attack_df["model"] != "gpt4"].copy()
data_test = attack_df[attack_df["model"] == "gpt4"].copy()

y_cv = train_val_data["jailbroken"].astype(int)
# Random 5-fold split of the rows. shuffle=True matters: without it, KFold takes rows in order,
# and since the data is sorted by method/model, each fold would be mostly one method.
kf = KFold(n_splits=5, shuffle=True, random_state=120)


def scale(train_part, other_part, features):
    """Standardize using the mean/std of the training part only, then apply them to the other part."""
    train_part, other_part = train_part.copy(), other_part.copy()
    cols = [c for c in scale_cols if c in features]
    scaler = StandardScaler()
    train_part[cols] = scaler.fit_transform(train_part[cols])
    other_part[cols] = scaler.transform(other_part[cols])
    return train_part, other_part


def score(model, X):
    """Probability of 'jailbroken' for classifiers; the raw prediction for LinearRegression."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def metrics(model, X, y):
    """AUC (higher = better) and log loss (lower = better) for one set of rows."""
    s = score(model, X)
    # Log loss needs probabilities; LinearRegression's outputs aren't, so it gets NaN.
    ll = log_loss(y, s) if hasattr(model, "predict_proba") else np.nan
    return roc_auc_score(y, s), ll


def cv_metrics(model, features):
    """Train and validation AUC / log loss, averaged over the 5 folds."""
    rows = []
    for train_idx, val_idx in kf.split(train_val_data):
        tr, va = scale(train_val_data.iloc[train_idx], train_val_data.iloc[val_idx], features)
        model.fit(tr[features], y_cv.iloc[train_idx])
        tr_auc, tr_ll = metrics(model, tr[features], y_cv.iloc[train_idx])
        va_auc, va_ll = metrics(model, va[features], y_cv.iloc[val_idx])
        rows.append([tr_auc, va_auc, tr_ll, va_ll])
    tr_auc, va_auc, tr_ll, va_ll = np.mean(rows, axis=0)
    return {"train_auc": tr_auc, "val_auc": va_auc, "train_logloss": tr_ll, "val_logloss": va_ll}


# ---------------------------------------------------------------
# 4. Grid search: try every combination of the listed hyperparameter values.
#    record() runs the 5-fold CV for one combination and saves one row:
#    the method, its hyperparameter values, and the averaged train/validation metrics.
# ---------------------------------------------------------------
results = {}   # method name -> list of rows (one row per hyperparameter combination)

def record(method, params, model, features):
    row = {**params, **cv_metrics(model, features)}   # e.g. {"C": 0.1, "train_auc": ..., "val_auc": ...}
    results.setdefault(method, []).append(row)

# Linear regression: no hyperparameters. CV just gives an honest validation score.
record("OLS", {}, LinearRegression(), linear_features)

# Logistic regression: tune C (smaller C = stronger penalty)
for C in [0.01, 0.1, 1, 10]:
    record("Logistic", {"C": C}, LogisticRegression(C=C, max_iter=1000), linear_features)

# Decision tree: limit depth and leaf size, otherwise it memorizes the training data
for depth in [2, 4, 6, None]:
    for leaf in [1, 10, 30]:
        record("Tree", {"max_depth": str(depth), "min_samples_leaf": leaf},  # str() so None prints as "None"
               DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=120),
               tree_features)

# Random forest: more trees never hurts; tune columns per split and leaf size
for mf in ["sqrt", 0.5]:
    for leaf in [1, 10]:
        record("RF", {"max_features": mf, "min_samples_leaf": leaf},
               RandomForestClassifier(n_estimators=300, max_features=mf, min_samples_leaf=leaf,
                                      random_state=120, n_jobs=-1),
               tree_features)

# Gradient boosting: learning rate, tree depth and number of rounds work together
for lr in [0.05, 0.1]:
    for depth in [2, 3]:
        for n in [100, 300]:
            record("GB", {"learning_rate": lr, "max_depth": depth, "n_estimators": n},
                   GradientBoostingClassifier(learning_rate=lr, max_depth=depth, n_estimators=n,
                                              random_state=120),
                   tree_features)

# One table per method: its hyperparameter columns + train/val metrics, best val_auc first.
# Pick the setting with the best validation score (not the smallest train-val gap).
tables = {method: pd.DataFrame(rows) for method, rows in results.items()}
report_lines = []
for method, table in tables.items():
    block = f"\n=== {method} ===\n" + table.sort_values("val_auc", ascending=False).round(3).to_string(index=False)
    report_lines.append(block)
# tables["GB"] etc. are ordinary DataFrames if you want to plot or save them.

# ---------------------------------------------------------------
# 5. Refit the chosen setting on ALL train+val rows, then evaluate on GPT-4 once
# ---------------------------------------------------------------
best_model = LogisticRegression(C=0.1, max_iter=1000)   # <- replace with your chosen model + settings
best_features = linear_features                          # <- linear_features or tree_features to match

train_full, test = scale(train_val_data, data_test, best_features)
best_model.fit(train_full[best_features], train_full["jailbroken"].astype(int))

X_test = test[best_features]
y_test = test["jailbroken"].astype(int)
y_test_score = score(best_model, X_test)
y_pred = (y_test_score >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
ConfusionMatrixDisplay(cm, display_labels=["not jailbroken", "jailbroken"]).plot()

out_dir = Path(__file__).resolve().parent.parent / "deliverables"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
plt.close()

tn, fp, fn, tp = cm.ravel()
test_lines = [
    "GPT-4 test, LogisticRegression(C=0.1)",
    f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
    f"Precision: {precision_score(y_test, y_pred)}",
    f"Recall:    {recall_score(y_test, y_pred)}",
    f"Accuracy:  {accuracy_score(y_test, y_pred)}",
    f"F1:        {f1_score(y_test, y_pred)}",
    f"AUC ROC:   {roc_auc_score(y_test, y_test_score)}",
]
(out_dir / "cv_results.txt").write_text(
    "\n".join(report_lines + ["", *test_lines]) + "\n", encoding="utf-8"
)