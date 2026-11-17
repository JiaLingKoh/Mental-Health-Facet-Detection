# S1 Aspect Detection

This directory contains the Python source code for the **S1 aspect-detection experiment**. The workflow prepares segment-level data, trains multiple Chinese BERT binary classifiers, combines their segment predictions using a model-priority strategy, and produces article-level predictions and evaluation reports across five rounds.

This README documents the **functions and execution order** represented in the supplied S1 shell script (`run.sh`). The script itself does **not** need to be published. All commands should be run from the **S1 project directory** after the required datasets and local modules have been provided.

> **Reproducibility status:** This guide has been checked against the supplied Python source and shell commands, but the complete workflow has **not** been executed end to end. Some original commands are incompatible with the present Python command-line interfaces. These cases are identified explicitly below; they should not be mistaken for tested reproduction commands.

## 1. Source files and functions

| Published filename | Function |
| --- | --- |
| `preprocess_modify.py` | Prepare stage-1 and stage-2 training/test data. Use `-s 1` or `-s 2` to select the stage, and `-p np` for the optional NP experiment. |
| `train.py` | Train label-specific Chinese BERT binary classifiers and save checkpoints selected by **precision** and **F1 score**. |
| `evaluation_nolabel5.py` | Predict with the five S1 classifiers (`-2`, `-1`, `1`, `2`, `0`) using **precision-selected** checkpoints; export per-segment predictions. |
| `evaluation.py` | Expanded evaluation, including the additional `5` and `-3` labels in its default configuration, or labels `0` and `5` in NP mode. |
| `ModelPriorityStrategy.py` | Model-priority aggregation for segment and article labels; calculate classification metrics, export results, and invoke case analysis. **This is the implementation called by the supplied shell script.** |
| `ModelPriorityStrategy_S1.py` | **Later S1-specific variant** of model-priority aggregation. Uses an S1-specific classifier order and reads earlier article predictions to influence final decisions; see compatibility warning in Section 4. |
| `S1_Article_decision_rules.py` | Rule-based article-label selection from the segment labels; imported by both model-priority scripts. |
| `ResultsParsing.py` | Extract correct and incorrect article/segment predictions for error analysis. |
| `combineResult.py` | Merge first-stage article predictions and subsequent article-label predictions; compute the overall S1 report and confusion matrix. |

The uploaded filenames with suffixes such as `(2)` or `(1)` are upload copies. The **published S1 directory should use the unsuffixed filenames** shown above, matching the imports in the source code.

The programs additionally require the local modules `lib/preprocess.py`, `lib/model.py`, and `lib/utils.py`, which are imported by the supplied scripts but were **not included in this S1 upload**. Include these files in the release.

## 2. Environment and input/output layout

The source depends on Python, PyTorch, Hugging Face Transformers, pandas, NumPy, scikit-learn, Matplotlib, and Seaborn, and uses **`bert-base-chinese`**. Install the versions validated for the original experiment; no tested, fully pinned S1 environment file was included in this upload. If a validated `requirements.txt` is available, use:

```bash
python -m pip install -r requirements.txt
```

Expected paths (five rounds: `round_0` through `round_4`):

| Relative path | Role |
| --- | --- |
| `data/data_4train/round_<n>/` | Training inputs such as `train<id>_p.csv`, `train<id>_n.csv`, and `test<id>.csv` used in model selection. |
| `data/data_4test/round_<n>/` | Classifier test files (`test<id>.csv`) and the aggregate `test.csv`. |
| `model_precision/round_<n>/<id>.ckpt` | Precision-selected checkpoints loaded by the S1 evaluation scripts. |
| `model_f1/round_<n>/<id>.ckpt` | F1-selected checkpoints also written by the training code. |
| `result/round_<n>/prediction_all.csv` | Segment predictions from the five binary classifiers. |
| `result/round_<n>_SegmentLabel.csv` | Aggregated segment predictions. |
| `result/round_<n>_ArticleLabel.csv` | Aggregated article predictions. |
| `S1/article/prediction<n>.csv` | First-stage article predictions needed for the final merging step. |
| `result/all_results/` | Combined results and correct/incorrect prediction examples. |

The creation of some intermediate files depends on the local `lib/` modules and the input data, so their complete formats cannot be verified from this upload alone.

## 3. Execution sequence from the original S1 shell script

The original file describes the following order: **training preprocessing → model training → test preprocessing → initial prediction/aggregation → filtered test preparation → final prediction/aggregation → result merging**.

### Step 1 — Generate training data

```bash
python preprocess_modify.py -a S1 -s 1
python preprocess_modify.py -a S1 -s 2
```

Stage 1 prepares label-specific data; stage 2 creates training inputs for the binary classifiers. These commands match the shell script and the `-s` options defined in `preprocess_modify.py`.

### Step 2 — Train the binary classifiers

```bash
python train.py
```

The supplied `train.py` defaults to five classifiers (`-2`, `-1`, `+1`, `+2`, and `0`) over five rounds, with 15 epochs, `bert-base-chinese`, and Adam with learning rate `1e-5`. It saves both `model_precision` and `model_f1` checkpoints. The default five-model configuration matches `evaluation_nolabel5.py`.

### Step 3 — Generate ordinary test inputs

```bash
python preprocess_modify.py -t test -a S1 -s 1
python preprocess_modify.py -t test -a S1 -s 2
```

This produces the test data required by the classifier-evaluation step.

### Step 4 — Evaluate the first-stage predictions

The **original shell commands** are:

```bash
python evaluation_nolabel5.py
python ModelPriorityStrategy.py -m 5
```

`evaluation_nolabel5.py` loads `model_precision/round_<n>/<id>.ckpt` and writes `result/round_<n>/prediction_all.csv`. The S1 prediction thresholds include `0.997` for classifier `-2` and `0.9` for classifiers `-1`, `1`, `2`, and `0`.

**Command-line mismatch:** The supplied `ModelPriorityStrategy.py` **does not accept `-m`**. The original `-m 5` command therefore will not run with this version. The available decision-mode argument is `-d` (e.g., `-d rule`), but replacing `-m 5` with `-d rule` is **not a verified equivalent** to the originally intended first-stage setting. Confirm the intended first-stage model-priority invocation and resulting `S1/article/prediction<n>.csv` files before running the later steps.

### Step 5 — Prepare filtered test inputs using prior article predictions

```bash
python preprocess_modify.py -t test -m combine_article -a S1/article -s 1
```

This corresponds to the script comment indicating that later-stage testing uses articles selected from prior article-detection results. The actual filtering logic is in `lib/preprocess.py`. The required earlier-stage files under `S1/article/` must already exist.

### Step 6 — Re-evaluate and aggregate the filtered predictions

```bash
python evaluation_nolabel5.py
python ModelPriorityStrategy.py -d rule
```

The second evaluation call produces classifier predictions for the filtered test set. **Although a comment in the shell script says label `5` is restored at this point, it still invokes `evaluation_nolabel5.py`, whose default label list excludes `5`.** `ModelPriorityStrategy.py` applies the following **segment-level classifier priority**:

```text
0 → 2 → -2 → -1 → 1
```

Each segment is assigned the first classifier label whose binary prediction is positive; if none is positive, the segment is assigned `0`. With `-d rule`, article predictions are determined using the `rule` branch in `S1_Article_decision_rules.py`. The strategy script also computes per-round and overall segment/article metrics and invokes `ResultsParsing.py` to export correctly and incorrectly classified examples.

**Implementation distinction:** The later `ModelPriorityStrategy_S1.py` uses a different priority, `0 → 2 → 1 → -2 → -1`, and incorporates earlier article predictions. It is **not the script called in this shell sequence**. See Section 4 before attempting to use it for the final reported S1 results.

### Step 7 — Combine the article-level predictions

```bash
python combineResult.py -c S1/article
```

For each round, this script joins `S1/article/prediction<n>.csv` with `result/round_<n>_ArticleLabel.csv` on article ID. It exports `result/all_results/Article_all_result.csv`, a confusion-matrix image (`result/_Article_all_result_cm.png`), and classification scores (`result/_Article_all_result_Score.csv`). It compares the final predictions against the **`S1`** column in the merged data.

Ensure that `result/all_results/` exists and that the input files contain the expected columns (`id`, `model_pred`, `S1`, `ModelPred_articleLabel`, etc.).

## 4. Later S1-specific model-priority version

The supplied directory also contains `ModelPriorityStrategy_S1.py` (the later S1-specific implementation). It differs substantively from `ModelPriorityStrategy.py`:

- Classifier priority: `0 → 2 → 1 → -2 → -1`, instead of `0 → 2 → -2 → -1 → 1`.
- Additional input: `S1_0307/article/prediction<n>.csv`, including the earlier `model_pred` values.
- Article decision: if the earlier article prediction is `0`, the script sets the article output to `0`; otherwise, it uses segment-level aggregation and tries to pass the earlier article prediction to the decision function.

**This later version is currently not a drop-in replacement.** `ModelPriorityStrategy_S1.py` calls `det_article_label(post_id, seg_labels, article_predict, article_decision)` with **four arguments**, but the uploaded `S1_Article_decision_rules.py` defines `det_article_label(p_id, seg_labels, article_decision)` with **three arguments**. Its hard-coded `S1_0307/article/` path also differs from the renamed `S1/article/` path. Those inconsistencies must be resolved and verified against the experiment before this later implementation can be used to reproduce the final results. This README does **not** silently modify the algorithm or claim that the later version runs as supplied.

## 5. Optional NP-model and other evaluations

The remaining blocks in the original shell script are **additional executable experiments**, not comments. They should be run separately because they may overwrite intermediate evaluation files.

### NP-model training and evaluation

The original shell commands omit `-s` for preprocessing, but the supplied `preprocess_modify.py` uses `stage_num` without defining a default. The following explicit stages are **proposed corrections**, subject to confirming the intended data flow:

```bash
python preprocess_modify.py -p np -a S1 -s 1
python preprocess_modify.py -p np -a S1 -s 2
python preprocess_modify.py -p np -t test -a S1 -s 1
python preprocess_modify.py -p np -t test -a S1 -s 2
python train.py -p np
python evaluation.py -m np
```

`train.py -p np` selects the `+5` classifier. `evaluation.py -m np` selects the `0` and `5` label list. The original shell also calls `python ModelPriorityStrategy.py -m np`, but **`-m` is not supported** by the supplied model-priority script; the intended NP aggregation command must be established separately.

### Expanded hierarchical and combined evaluation

The shell script additionally runs preprocessing in both NP and ordinary modes, then calls `evaluation.py` and `ModelPriorityStrategy.py`. Its last block also prepares combined article-based test data. The original preprocessing commands in those blocks lack `-s` and require correction to match the actual intended stage. In addition, the default article-decision mode in `ModelPriorityStrategy.py` is `for_s`, which is not one of the named decision branches in `S1_Article_decision_rules.py`; pass and validate an intended decision mode rather than assuming its default reproduces the experiment. Do not combine their outputs with the main evaluation without first checking which data and checkpoints belong to each experimental condition.

