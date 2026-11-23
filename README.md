# Predicting Whether an LLM Can Resist a Jailbreak Attempt

Ruoyu (Tony) Guo, Yu Xin, Vera Andersson

A jailbreak attempt is a prompt that tries to convince an LLM to provide harmful information. Given an attempt and a target LLM, we use ML methods to predict whether the attempt will succeed. Safety training of LLMs to resist jailbreaks can take up a significant amount of time and resources, so our project could potentially make the process more efficient by prioritizing the most dangerous jailbreak attempts first.

## Data

We use 1,637 prompt–response pairs from the [JailbreakBench attack artifacts](https://github.com/JailbreakBench/artifacts). The data have 5 attack methods against 4 target LLMs, and 57% of attempts are successful jailbreaks.

- **LLMs**:
    - Llama-2-7B-chat
    - Vicuna-13B v1.5
    - GPT-3.5 Turbo
    - GPT-4 Turbo
- **Attacks**:
    - DSN, Don't Say No
    - GCG, Greedy Coordinate Gradient
    - JBC, Jailbreak Chat
    - PAIR, Prompt Automatic Iterative Refinement
    - PRS, prompt with random search

**Features**

- Prompt-related features including 10-d embedding of the prompt
- One-hot attack method embedding
- LLM benchmark scores from [LMArena](https://github.com/lmarena/arena-catalog/blob/main/data/leaderboard-text.json) (expert category), [EQ-Bench](https://huggingface.co/spaces/sam-paech/EQ-Bench-Leaderboard) and [SORRY-Bench](https://arxiv.org/abs/2406.14598)
- Interactions between each benchmark scores and each prompt-related feature and attack column

## Results on the test set

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| OLS | 0.857 | 0.835 | 0.937 | 0.883 | 0.921 |
| Logistic regression | 0.863 | 0.836 | **0.947** | 0.888 | 0.923 |
| Decision tree | 0.866 | 0.844 | 0.942 | 0.890 | 0.912 |
| Random forest | 0.866 | 0.854 | 0.926 | 0.888 | 0.929 |
| **Gradient boosting** | **0.890** | **0.877** | 0.942 | **0.908** | **0.943** |
| XGBoost | 0.866 | 0.840 | **0.947** | 0.891 | 0.942 |

## Analysis

**1. LLM benchmarks are necessary.**

We refit the tree models with groups of features removed and report the AUC scores:

| Features | Decision tree | Random forest | Gradient boosting | XGBoost |
|---|---|---|---|---|
| All features | 0.912 | 0.929 | 0.943 | 0.942 |
| No SORRY-Bench | 0.912 | 0.927 | 0.945 | 0.939 |
| No attack one-hots | 0.872 | 0.923 | 0.934 | 0.937 |
| No LLM benchmarks | 0.837 | 0.828 | 0.850 | 0.851 |

**2. Training on all LLMs together is better than training on LLMs separately.**

Average AUC on the test set shows more data, even though on different LLMs, could slightly improve ML performance.

| Training | OLS | Logistic | Decision tree | Random forest | Gradient boosting | XGBoost |
|---|---|---|---|---|---|---|
| All four LLMs | 0.921 | 0.923 | 0.912 | 0.929 | 0.943 | 0.942 |
| Own LLM only | 0.919 | 0.921 | 0.916 | 0.911 | 0.923 | 0.922 |

## Limitations and next steps

- With only four LLMs, the benchmark columns take four distinct values, so they also act as an LLM identifier. However, the interaction columns are most helpful for linear models.
- A preliminary experiment that held out GPT-4 as the test set (not part of the scripts below) shows weak generalization to LLMs the model has not seen.

## Repository and how to run

```
src/data_prep.py          Step 1: builds data/processed_data.csv from the raw JailbreakBench JSON files
src/cross_validation.py   Step 2: 5-fold CV grid search; also holds the shared split, scaling, metrics and chosen models
src/analysis.py           Step 3: test-set results, baselines, feature ablation, per-attack/LLM and pooled vs. per-LLM comparisons
data/raw_jailbenchmark/attack-artifacts/<attack>/<llm>.json   raw input
deliverables/             write-up and presentation
```

```bash
python data_prep.py          # writes data/processed_data.csv
python cross_validation.py   # prints one grid-search table per model
python analysis.py           # prints all test-set results
```

`analysis.py` uses the settings in `CHOSEN_PARAMS` in `cross_validation.py`. If the grid search picks different settings, update `CHOSEN_PARAMS` by hand.

## Data licenses
- Third-party data from [JailbreakBench artifacts](https://github.com/JailbreakBench/artifacts) retains its [upstream MIT license](data/raw/UPSTREAM_LICENSE.txt).
- The benchmark scores are taken from the public LMArena, EQ-Bench and SORRY-Bench sources linked above.
