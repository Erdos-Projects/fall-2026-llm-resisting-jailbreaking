# src

Predicting whether a jailbreak attempt succeeds on an LLM, from the JailbreakBench
attack artifacts. Run the three steps in order from the repository root:

| Step | Command | Does | Output |
|---|---|---|---|
| 1 | `python src/data_prep.py` | Loads `data/raw_jailbenchmark/attack-artifacts/`, drops rows without a prompt or response, adds the prompt, benchmark, attack-method and interaction features | `data/processed_data.csv` |
| 2 | `python src/cross_validation.py` | 80/20 train/test split, then a 5-fold CV grid search on the training rows for OLS, logistic regression, decision tree, random forest, gradient boosting and XGBoost | One table per model, printed, best validation AUC first |
| 3 | `python src/analysis.py` | Fits the chosen settings on the training rows and scores the test rows | Item 1: test results vs. two lookup baselines, and a feature ablation. Item 2: gradient boosting by attack method and LLM. Item 3: one model on all four LLMs vs. one model per LLM |

Step 3 uses the settings in `CHOSEN_PARAMS` (in `cross_validation.py`). They are
not read from step 2's output, so update them by hand if the grid search picks
different ones.

`cross_validation.py` also holds what both later steps share: the split, the
scaling (fit on training rows only, clipped to [-2, 2]), the metrics at a 0.5
threshold, and the feature lists.

Requires pandas, numpy, scikit-learn (1.3 or later) and xgboost.

These scripts replace `data_cleaning.ipynb` (archived), `data_cleaning_v2.ipynb`,
`cv_v2.ipynb` and `analysis_v2.ipynb`.
