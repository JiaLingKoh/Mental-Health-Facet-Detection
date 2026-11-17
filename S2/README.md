# S2 Aspect Detection

This directory contains the Python code for the **S2 aspect-detection experiment**. The workflow prepares segment-level data, trains multiple Chinese BERT binary classifiers, aggregates their predictions at segment and article levels, and produces five-round evaluation results. In the S2-specific article-decision stage, predictions from an earlier article-level model (including its `NP_pred` output) are also used.

This README documents the **execution order and intended functions** of the supplied S2 shell script. The shell file itself need not be included in the public release. Run the commands from the **S2 project directory**, with the required input data, first-stage predictions, and local `lib/` modules available.

> **Reproducibility status:** The commands and source interfaces below were checked against the uploaded files; the complete experiment has **not** been executed end to end. The original shell sequence contains a missing model-priority program and some paths/arguments that are incompatible with the uploaded Python code. These are identified below rather than silently changed.

## 1. Source files

| Published filename | Purpose |
| --- | --- |
| `preprocess_modify.py` | Prepare stage-1 and stage-2 training and test data. Select a stage using `-s 1` or `-s 2`; use `-p np` for the optional NP-model condition. |
| `train.py` | Train five label-specific Chinese BERT binary classifiers (`-2`, `-1`, `+1`, `+2`, and `0`) for five rounds; save checkpoints selected by precision and F1 score. |
| `evaluation_nolabel5.py` | Evaluate the **five** S2 label classifiers with **F1-selected** checkpoints and S2-specific probability thresholds; export per-segment predictions. |
| `evaluation.py` | Extended evaluation script: its default label list also includes `5` and `-3`, and NP mode selects `0` and `5`. This script uses **precision-selected** checkpoints. |
| `ModelPriorityStrategy_S2.py` | S2-specific segment-priority aggregation and article decisions using previous article/NP predictions; write per-round and overall metrics. |
| `S2_Article_decision_rules.py` | S2 article-level rule function used by `ModelPriorityStrategy_S2.py`. |
| `ResultsParsing.py` | Export correctly and incorrectly classified article/segment examples for error analysis. |
| `combineResult.py` | Merge first-stage article predictions with S2 predictions and compute the final classification report and confusion matrix. |

The upload suffixes `(3)` and `(1)` are file-copy artifacts. **Use the unsuffixed filenames above in the published directory** so imports and command names match the code.

**Required but not included in this S2 upload:** The scripts import `lib/preprocess.py`, `lib/model.py`, and `lib/utils.py`. The original shell script also calls **`ModelPriorityStrategy.py`**, but that separate file was not supplied in this S2 set. Do not substitute `ModelPriorityStrategy_S2.py` for it without confirming the intended first-stage behavior.

## 2. Requirements and data layout

The Python code uses PyTorch, Hugging Face Transformers, pandas, NumPy, scikit-learn, Matplotlib, and Seaborn. The base language model is **`bert-base-chinese`**. A fully validated environment/lockfile was not included in this S2 upload. Install the versions validated for the experiment; if a tested requirements file is supplied separately, it may be installed with:

```bash
python -m pip install -r requirements.txt
```

The default experiment uses **five rounds** (`round_0` to `round_4`). Its relative paths include:

| Path | Contents / role |
| --- | --- |
| `data/data_4train/round_<n>/` | Training inputs: `train<id>_p.csv`, `train<id>_n.csv`, and `test<id>.csv`. |
| `data/data_4test/round_<n>/` | Testing inputs: `test<id>.csv` and combined `test.csv`. |
| `model_f1/round_<n>/<id>.ckpt` | F1-selected classifiers loaded by `evaluation_nolabel5.py`. |
| `model_precision/round_<n>/<id>.ckpt` | Precision-selected classifiers loaded by `evaluation.py`. |
| `result/round_<n>/prediction_all.csv` | Combined classifier predictions for each segment. |
| `result/round_<n>_SegmentLabel.csv` | Aggregated segment-label predictions. |
| `result/round_<n>_ArticleLabel.csv` | Aggregated article-label predictions. |
| `S2/article/prediction<n>.csv` | Earlier article-level results used by preprocessing and `combineResult.py`. |
| `S2_0326/article/prediction<n>.csv` | **Legacy hard-coded path actually read by** `ModelPriorityStrategy_S2.py`; see Section 4. |
| `result/all_results/` | Merged article results and correct/incorrect prediction examples. |

The precise input CSV format and the creation of upstream files depend on `lib/preprocess.py` and the earlier-stage article-detection pipeline, which are not fully documented by this upload.

## 3. Main workflow (in the order of the original shell script)

### Step 1 — Prepare training data

```bash
python preprocess_modify.py -a S2 -s 1
python preprocess_modify.py -a S2 -s 2
```

Stage 1 generates label-specific inputs; stage 2 builds binary-classifier training data. The ordinary S2 path prepares labels `0`, `2`, `-2`, `1`, and `-1`. The stage flag is required: `preprocess_modify.py` does **not** initialize `stage_num` when `-s` is omitted.

### Step 2 — Train S2 binary classifiers

```bash
python train.py
```

As supplied, the script trains the `-2`, `-1`, `+1`, `+2`, and `0` classifiers over five rounds, using **15 epochs**, **`bert-base-chinese`**, and Adam at learning rate **`1e-5`**. It saves both F1- and precision-selected checkpoints. `evaluation_nolabel5.py` uses the **F1-selected** ones.

### Step 3 — Prepare ordinary test data

```bash
python preprocess_modify.py -t test -a S2 -s 1
python preprocess_modify.py -t test -a S2 -s 2
```

These commands create the test inputs required by the individual classifiers.

### Step 4 — First prediction / aggregation pass

The original shell file records:

```bash
python evaluation_nolabel5.py
python ModelPriorityStrategy.py -m 5
```

The evaluation script loads `model_f1/round_<n>/<id>.ckpt` and writes `result/round_<n>/prediction_all.csv`. Its five default labels are `-2`, `-1`, `1`, `2`, and `0`; the associated S2 thresholds are **0.995** (`-2`), **0.9** (`-1`), **0.9** (`1`), **0.995** (`2`), and **0.99** (`0`).

**Unresolved original command:** `ModelPriorityStrategy.py` is **not present** in the supplied S2 files. Its `-m 5` interface therefore cannot be confirmed. The available `ModelPriorityStrategy_S2.py` instead accepts `-d` (article-decision mode) and does **not** implement `-m`. This first-stage command must be reconciled with the original experiment before claiming a complete reproduction.

### Step 5 — Prepare the article-filtered test set

```bash
python preprocess_modify.py -t test -m combine_article -a S2/article -s 1
```

This shell step selects the downstream test set based on earlier article-level detection results (described in the original shell comments as articles predicted positive and nonzero). Filtering itself is implemented in `lib/preprocess.py`. The upstream `S2/article/prediction<n>.csv` files must already exist.

### Step 6 — Predict and apply S2-specific article rules

```bash
python evaluation_nolabel5.py
python ModelPriorityStrategy_S2.py -d rule
```

The second call to `evaluation_nolabel5.py` predicts on the article-filtered test set. Despite the shell comment about adding label `5` back, the **actual script still evaluates only the five default labels**.

`ModelPriorityStrategy_S2.py` applies the segment classifier priority:

```text
0 → 2 → 1 → -2 → -1
```

For each segment, the first positive binary-classifier decision determines the segment label; if no classifier is positive, the label is `0`. At article level, the S2 implementation uses the earlier `model_pred` and `NP_pred` values. An earlier `model_pred` of `0` directly gives an article prediction of `0`; otherwise the script calls `S2_Article_decision_rules.det_article_label(...)`, passing the segment labels and both earlier predictions. This function's **five-argument signature matches** the call from the supplied S2 strategy script.

The strategy exports segment/article CSVs, confusion-matrix summaries, overall metrics, and error-analysis examples via `ResultsParsing.py`.

**Important path mismatch:** The strategy currently reads `S2_0326/article/prediction<n>.csv`, whereas the renamed shell sequence uses `S2/article/`. Align that path to the location of the **actual matching earlier-stage predictions** before running this step. Do not mix different experimental versions.

### Step 7 — Merge and evaluate S2 article predictions

```bash
python combineResult.py -c S2/article
```

The program joins each `S2/article/prediction<n>.csv` with `result/round_<n>_ArticleLabel.csv` on `id`, concatenates the five rounds, and evaluates predictions against the **`S2`** ground-truth column. It writes:

- `result/all_results/Article_all_result.csv`
- `result/_Article_all_result_cm.png`
- `result/_Article_all_result_Score.csv`

The first-stage CSVs must contain the expected fields, including `id`, `S2`, and `model_pred`. The current `combineResult.py` also expects columns named `Unnamed: 0`, `article_label`, `備註`, `NP_pred`, and `final_pred` when dropping them; a missing field will raise an error unless the input schema or source is adjusted. Confirm that `result/all_results/` exists before this step.

## 4. S2-specific compatibility notes

The following points are significant for reproduction, rather than merely cosmetic naming differences:

1. **Earlier-prediction path:** `ModelPriorityStrategy_S2.py` reads from `S2_0326/article/`, while preprocessing and final merging use `S2/article/`. All stages must refer to the same experiment and fold assignments.
2. **Previous-model inputs:** The S2 strategy expects `id`, `S2`, `model_pred`, and **`NP_pred`** in the prior article-level files. These are substantive inputs to article decisions, not optional metadata.
3. **Article/segment alignment:** In the S2 strategy, an article predicted `0` by the earlier model bypasses the segment-decision function; the source later computes segment metrics from accumulated arrays. Verify that the arrays and IDs remain aligned for such articles. The source was not changed here.
4. **ID types:** `ModelPriorityStrategy_S2.py` converts test article IDs to strings, but the earlier dataframe's index may retain its original type. Confirm matching IDs before running; otherwise prior predictions may not be found.
5. **First-stage program:** `ModelPriorityStrategy.py` called in the shell is not provided in this S2 upload. Its intended output and command-line arguments remain unverified.

## 5. Optional NP-model and other evaluations

The remaining shell-script blocks are **additional executable experiments**, not shell comments, and should be run separately from the main evaluation because they may reuse or overwrite intermediate result files.

### NP-model preparation and evaluation

The original NP preprocessing commands omit `-s`, but the supplied Python script requires it. The following expands them into **suggested explicit stages**, not independently tested replacements:

```bash
python preprocess_modify.py -p np -a S2 -s 1
python preprocess_modify.py -p np -a S2 -s 2
python preprocess_modify.py -p np -t test -a S2 -s 1
python preprocess_modify.py -p np -t test -a S2 -s 2
python train.py -p np
python evaluation.py -m np
```

In NP mode, `train.py` selects the `+5` classifier, whereas `evaluation.py -m np` evaluates labels `0` and `5`; ensure the `0` checkpoint is available from the appropriate model training. The original shell additionally calls `ModelPriorityStrategy.py -m np`, but the required script is not present here, so NP aggregation cannot be specified confidently from these files alone.

### Expanded hierarchical / combined evaluations

The original shell also performs NP and ordinary test preprocessing, runs `evaluation.py`, and calls `ModelPriorityStrategy.py`; its final block creates filtered test sets and repeats the evaluation. Those preprocessing commands omit `-s`, so the correct stage(s) must be supplied after confirming the intended condition. `evaluation.py` defaults to **seven** labels (`-2`, `-1`, `1`, `2`, `0`, `5`, `-3`) and loads **precision-selected** checkpoints, unlike the five-label F1-based main workflow. The additional labels/checkpoints and missing model-priority program must be supplied before running this optional evaluation.

 
