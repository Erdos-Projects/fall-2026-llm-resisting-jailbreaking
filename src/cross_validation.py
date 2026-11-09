"""Step 2: 5-fold cross-validated grid search on the training rows.

Prints one table per model, best validation AUC first. analysis.py uses the
settings in CHOSEN_PARAMS, so update them by hand if the search picks
different ones.

Also holds the setup shared with analysis.py: the train/test split,
scaling, metrics, feature lists, and the chosen models.
"""

from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_prep import BENCH_COLS, INTERACTION_COLS, METHOD_COLS, NUMERIC_COLS, PROCESSED_PATH

SEED = 120
# Choosing threshold as 0.5 to "maximize" recall (don't want false negatives for jailbreak attempts)
THRESHOLD = 0.5

# The linear models drop the first method column to avoid collinearity
LINEAR_FEATURES = NUMERIC_COLS + BENCH_COLS + INTERACTION_COLS + METHOD_COLS[1:]
TREE_FEATURES = NUMERIC_COLS + BENCH_COLS + METHOD_COLS
SCALE_COLS = NUMERIC_COLS + BENCH_COLS + INTERACTION_COLS  # everything except the 0/1 method columns

BOOSTING_GRID = [{"learning_rate": lr, "max_depth": depth, "n_estimators": n}
                 for lr in [0.05, 0.1] for depth in [2, 3, 4] for n in [100, 300]]
# name: (model with its fixed settings, features, hyperparameter grid)
SEARCHES = {
    "OLS": (LinearRegression, LINEAR_FEATURES, [{}]),
    # smaller C = stronger penalty
    "Logistic": (partial(LogisticRegression, max_iter=1000), LINEAR_FEATURES,
                 [{"C": C} for C in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]]),
    "Decision tree": (partial(DecisionTreeClassifier, random_state=SEED), TREE_FEATURES,
                      [{"max_depth": depth, "min_samples_leaf": leaf}
                       for depth in [2, 3, 4, 5, 6, None] for leaf in [1, 5, 10, 20, 30]]),
    "Random forest": (partial(RandomForestClassifier, n_estimators=300, random_state=SEED, n_jobs=-1),
                      TREE_FEATURES,
                      [{"max_features": mf, "min_samples_leaf": leaf, "max_depth": depth}
                       for mf in [0.1, 0.3, 0.5] for depth in [2, 3, 4, None] for leaf in [1, 5, 10]]),
    "Gradient boosting": (partial(GradientBoostingClassifier, random_state=SEED), TREE_FEATURES,
                          BOOSTING_GRID),
    "XGBoost": (partial(XGBClassifier, random_state=SEED), TREE_FEATURES, BOOSTING_GRID),
}

# Settings used by analysis.py, picked from the grid-search tables
CHOSEN_PARAMS = {
    "OLS": {},
    "Logistic": {"C": 5},
    "Decision tree": {"max_depth": 3, "min_samples_leaf": 10},
    "Random forest": {"max_features": 0.1, "min_samples_leaf": 5},
    "Gradient boosting": {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 100},
    "XGBoost": {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 300},
}
# name: (unfitted model, features)
FINAL_MODELS = {name: (SEARCHES[name][0](**params), SEARCHES[name][1])
                for name, params in CHOSEN_PARAMS.items()}


def load_split():
    """Train and test rows (80/20). All four LLMs are mixed together."""
    return train_test_split(pd.read_csv(PROCESSED_PATH), test_size=0.2, random_state=SEED)


def scale_and_clip(fit_df, other_df):
    """Fit the scaler on fit_df only, then scale both and clip to [-2, 2] to limit outliers."""
    scaler = StandardScaler()
    fit_df, other_df = fit_df.copy(), other_df.copy()
    fit_df[SCALE_COLS] = np.clip(scaler.fit_transform(fit_df[SCALE_COLS]), -2, 2)
    other_df[SCALE_COLS] = np.clip(scaler.transform(other_df[SCALE_COLS]), -2, 2)
    return fit_df, other_df


def predict_score(model, X):
    """Score for "jailbroken". OLS has no predict_proba, so use its raw prediction."""
    if isinstance(model, LinearRegression):
        return model.predict(X)
    # predict_proba gives [P(not jailbroken), P(jailbroken)]; keep the second column
    return model.predict_proba(X)[:, 1]


def get_metrics(y_true, y_score):
    """AUC uses the raw scores; the others use 0/1 predictions at THRESHOLD."""
    y_pred = (y_score >= THRESHOLD).astype(int)
    return {"accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "auc":       roc_auc_score(y_true, y_score)}


def run_cv(model, features, train_df):
    """5-fold CV for one model with one hyperparameter setting."""
    train_rows, val_rows = [], []
    for train_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=SEED).split(train_df):
        train_part, val_part = scale_and_clip(train_df.iloc[train_idx], train_df.iloc[val_idx])
        y_train = train_part["jailbroken"].astype(int)
        y_val = val_part["jailbroken"].astype(int)
        model.fit(train_part[features], y_train)
        train_rows.append(get_metrics(y_train, predict_score(model, train_part[features])))
        val_rows.append(get_metrics(y_val, predict_score(model, val_part[features])))
    return train_rows, val_rows


def summarize(params, train_rows, val_rows):
    """One table row: the hyperparameters + 5-fold averages of each train and val metric."""
    row = dict(params)
    row.update(pd.DataFrame(train_rows).mean().add_prefix("train_").to_dict())
    row.update(pd.DataFrame(val_rows).mean().add_prefix("val_").to_dict())
    return row


if __name__ == "__main__":
    train_df, _ = load_split()
    for name, (make_model, features, grid) in SEARCHES.items():
        rows = [summarize(params, *run_cv(make_model(**params), features, train_df))
                for params in grid]
        table = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
        print(f"\n{name}: 5-fold CV, best validation AUC first")
        print(table.round(3).to_string(index=False, na_rep="None"))  # max_depth None shows as None
