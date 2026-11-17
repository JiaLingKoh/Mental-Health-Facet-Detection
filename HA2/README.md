# HA2 Aspect Detection

This directory contains the Python programs for the **HA2 aspect-detection experiment**. The system prepares segment-level data, trains Chinese BERT binary classifiers, combines their predictions using a model-priority strategy, and generates article-level results and evaluation reports over five rounds.

This README documents the **execution order and purpose of the commands** in the original HA2 shell script. A `.sh` file is **not needed** in the published code. Execute the Python commands in the order described below from the root of the HA2 directory.

> **Reproduction status:** This is a source-based guide, **not a verified end-to-end reproduction**. The original shell script references `ModelPriorityStrategy.py`, which is absent from the supplied HA2 files; its first command also uses an obsolete filename. Some preprocessing commands omit the required `-s` argument. The commands below clearly identify the corresponding adjustments and outstanding prerequisites.

## 1. Program files and their functions

| File in the published HA2 directory | Function |
| --- | --- |
| `preprocess_modify.py` | Prepare training or testing data. Its `-s 1` and `-s 2` options select the two preprocessing stages, while `-p np` selects the optional NP-model workflow. |
| `train.py` | Train Chinese BERT binary classifiers and save checkpoints selected by F1 score or precision. |
| `evaluation_nolabel5.py` | Load F1-selected checkpoints and output per-classifier predictions. The default label list is `-2`, `-1`, `1`, `2`, `0`, `-3` (excluding `5`). |
| `evaluation.py` | Alternative evaluation code, including label `5`, used in the NP and expanded-evaluation blocks. |
| `ModelPriorityStrategy_ha_common.py` | **Main model-priority script invoked by the shell file.** Combine segment predictions, apply the article-decision method, calculate metrics and confusion matrices, and export error examples. |
| `ModelPriorityStrategy_ha2.py` | HA2-specific alternative of the model-priority script. Its article-decision logic is not identical to the `_ha_common` version; it is **not** the script selected in the main shell sequence. |
| `ModelPriorityStrategy_ha_test.py` | Experimental version of model-priority processing; not required by the main shell sequence. |
| `Article_decision_functions.py` | Alternative article-label decision routines. Not called directly by the main shell sequence. |
| `ResultsParsing.py` | Extract correctly and incorrectly classified articles/segments; imported by the model-priority scripts. |
| `combineResult.py` | Merge earlier article predictions with later-stage outputs and calculate the final overall classification report and confusion matrix. |
| `ModelPriorityStrategy.py` | **Missing dependency**: invoked by the earlier-stage and NP/extended-evaluation commands in the original shell script but not supplied in this HA2 batch. |

The uploaded names ending in `(1)`, `(2)`, or `(3)` are treated as upload-name suffixes. The published files should use the unsuffixed names shown above so Python imports and commands match.

The programs also import `lib/preprocess.py`, `lib/model.py`, and `lib/utils.py`. These local modules must be provided in the HA2 release along with any necessary input datasets and previously generated predictions.

## 2. Environment, data, and expected files

The source programs use Python, **PyTorch**, **Transformers**, **pandas**, **NumPy**, **scikit-learn**, **Matplotlib**, and **Seaborn**. Their pretrained backbone is `bert-base-chinese`. Provide a validated `requirements.txt` for the actual research environment; the complete HA2 environment specification was not supplied with this batch.

Once a validated dependency file is provided:

```bash
python -m pip install -r requirements.txt
```

All commands below assume that the current directory is the HA2 project root. The code uses relative paths such as:

| Path | Content or purpose |
| --- | --- |
| `data/data_4train/round_<n>/` | Classifier training files `train<id>_p.csv`, `train<id>_n.csv`, and validation/test files `test<id>.csv`. |
| `data/data_4test/round_<n>/` | Classifier test files `test<id>.csv` and aggregated `test.csv`. |
| `model_f1/round_<n>/<id>.ckpt` | F1-selected model checkpoints loaded by the supplied evaluation scripts. |
| `model_precision/round_<n>/<id>.ckpt` | Precision-selected checkpoints saved during training. |
| `result/round_<n>/prediction_all.csv` | Combined binary-classifier predictions for each test segment. |
| `result/round_<n>_SegmentLabel.csv` | Combined segment-level classifications. |
| `result/round_<n>_ArticleLabel.csv` | Combined article-level classifications. |
| `ha2/article/prediction<n>.csv` | Earlier-stage article predictions required by the final integration step. |
| `result/all_results/` | Merged results and exported correct/incorrect examples. |

Here `<n>` is `0`–`4` in the default five-round setting. The exact original dataset format and creation of earlier-stage article predictions depend on the missing local `lib/` modules and `ModelPriorityStrategy.py`.

## 3. Main HA2 execution sequence

The sequence below follows the HA2 shell file: **preprocessing → training → first-stage testing → earlier-stage decision → later-stage testing → model-priority decision → final integration**. Where the original commands do not match the supplied source code, the necessary change is noted rather than silently assumed.

### Step 1 — Prepare training data

The shell script begins with `python preprocess_modify_0119.py -a ha2`, but that dated filename is not present in this renamed HA2 directory. The supplied `preprocess_modify.py` selects preprocessing using `-s`; both stages are shown explicitly here:

```bash
python preprocess_modify.py -a ha2 -s 1
python preprocess_modify.py -a ha2 -s 2
```

Stage 1 prepares label-specific data; stage 2 constructs the files needed to train the binary classifiers. **These are corrections proposed from the supplied program interface**, not verbatim commands from the shell file. Verify the input data and stage transitions using `lib/preprocess.py`.

### Step 2 — Train the classifiers

The original shell command is:

```bash
python train.py
```

**Important:** In the supplied `train.py`, the default `train_models` list contains **only `-3`**. This command therefore does not train the complete set of six classifiers used by `evaluation_nolabel5.py`.

If the goal is to retrain *all six classifiers* rather than reuse existing checkpoints, the script supports the explicit selection:

```bash
python train.py -m "-2 -1 +1 +2 0 -3"
```

Select the command that matches the actual experiment. The script specifies five rounds, 15 epochs, `bert-base-chinese`, and Adam (`1e-5`), and writes checkpoints to `model_f1/` and `model_precision/`.

### Step 3 — Prepare initial test data

The shell file contains `python preprocess_modify.py -t test -a ha2`, with no stage number. Because the supplied program requires `-s`, the explicit-stage form is:

```bash
python preprocess_modify.py -t test -a ha2 -s 1
python preprocess_modify.py -t test -a ha2 -s 2
```

Verify the required order and the generated test files before executing the next stage.

### Step 4 — Generate initial binary-classifier and article predictions

```bash
python evaluation_nolabel5.py
python ModelPriorityStrategy.py -m 5
```

The first command reads the test data and `model_f1/` checkpoints, then writes `result/round_<n>/prediction_all.csv`.

**The second command cannot currently be run from the supplied HA2 files:** `ModelPriorityStrategy.py` was referenced by the shell script but was not provided. Do not assume that `ModelPriorityStrategy_ha2.py` or `ModelPriorityStrategy_ha_common.py` is a drop-in replacement: their available command-line options and decision logic must first be compared with the intended initial-stage program.

### Step 5 — Prepare later-stage test data from prior article predictions

```bash
python preprocess_modify.py -t test -m combine_article -a ha2/article -s 1
```

This corresponds to the original shell file's filtered test-data preparation. It depends on the earlier article-prediction outputs under `ha2/article/`. The details of which articles are selected are implemented by the local preprocessing library.

### Step 6 — Generate final segment- and article-level predictions

```bash
python evaluation_nolabel5.py
python ModelPriorityStrategy_ha_common.py -d rule
```

The evaluation command runs again on the prepared later-stage test data. The model-priority command combines predictions using the classifier priority:

```text
0 → 2 → -2 → -1 → 1 → -3
```

For a segment, the model-priority classifier selects the first positive binary prediction in this order, or label `0` if none is positive. The `-d rule` argument selects the rule-based article-label decision. The script produces segment/article classification outputs and evaluation metrics, and calls `ResultsParsing.py` to export correct/incorrect examples.

**HA2-specific alternative:** `ModelPriorityStrategy_ha2.py` contains somewhat different article-decision rules, especially for combinations of classes `-1` and `1`. Although it is included in this HA2 batch, **the provided shell file explicitly invokes `ModelPriorityStrategy_ha_common.py` at this stage**. Retain the HA2-specific file as an alternative only if the manuscript reports results from it, rather than silently substituting it here.

### Step 7 — Merge earlier and later-stage results

```bash
python combineResult.py -c ha2/article
```

This program combines `ha2/article/prediction<n>.csv` with `result/round_<n>_ArticleLabel.csv` for each round. It writes merged article predictions and evaluation outputs under `result/`, including `result/all_results/Article_all_result.csv` and `result/_Article_all_result_Score.csv`.

**Data check:** `combineResult.py` expects a ground-truth column named `stage2SCORE` and drops several fixed column names. Confirm that these columns appear in the actual HA2 prediction files before running it.

## 4. Optional NP-model and additional evaluation procedures

The shell file contains further executable commands after the main HA2 procedure. They are documented separately because they can represent distinct experimental conditions, and running all of them in sequence may replace intermediate test files and results.

### 4.1 NP-model preprocessing, training, and evaluation

The original workflow is to prepare NP data, train the NP classifier, then evaluate it. The following commands add the explicit stage options required by the supplied preprocessing script:

```bash
python preprocess_modify.py -p np -a ha2 -s 1
python preprocess_modify.py -p np -a ha2 -s 2
python preprocess_modify.py -p np -t test -a ha2 -s 1
python preprocess_modify.py -p np -t test -a ha2 -s 2
python train.py -p np
python evaluation.py -m np
python ModelPriorityStrategy.py -m np
```

In the supplied training script, `-p np` selects the `+5` classifier. In `evaluation.py`, `-m np` changes the evaluated label list to `0` and `5`. The final step **still requires the missing** `ModelPriorityStrategy.py`.

### 4.2 Extended hierarchical and combined evaluations

The shell file also describes alternative tests combining NP and ordinary model outputs and using `evaluation.py` rather than `evaluation_nolabel5.py`. These are not part of the main HA2 execution sequence documented above. Apply their corresponding preprocessing stages before running the evaluations, and do not run them without their prerequisite checkpoints and the missing model-priority script.

**Possible copied path in the source shell file:** the final `combine_article` block refers to `ha1/article` (rather than `ha2/article`). This should be checked against the actual experiment; it has **not** been automatically changed in this README.

