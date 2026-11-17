# HA1 Aspect Detection

This directory contains the code used for the **HA1 aspect-detection experiment**. The pipeline uses Chinese BERT binary classifiers to predict labels for text segments, combines their outputs into article-level predictions, and reports classification results across five rounds.

This README documents the execution order represented by the original shell script. **The shell script is not required for the code release**; execute the Python commands in the order described below.

> **Reproduction status:** The uploaded files document the intended workflow, but a complete end-to-end run has **not** been verified. In particular, `ModelPriorityStrategy.py` is referenced by the shell script but was not included in this upload, and some original preprocessing commands omit a required `-s` argument. See [Items to verify before publication](#items-to-verify-before-publication).

## 1. Main program files

| File | Function |
| --- | --- |
| `preprocess_modify.py` | Prepare training and test data; supports stage 1 (`-s 1`), stage 2 (`-s 2`), and optional NP-model preparation (`-p np`). |
| `train.py` | Train binary Chinese BERT classifiers and save checkpoints selected by precision and F1 score. |
| `evaluation_nolabel5.py` | Load F1-selected checkpoints and generate predictions for labels `-2`, `-1`, `1`, `2`, `0`, and `-3` (excluding label `5`). |
| `evaluation.py` | Alternative prediction script that includes label `5` and supports the NP-model mode. |
| `ModelPriorityStrategy_ha_common.py` | Combine binary-classifier predictions according to a fixed model priority; determine segment/article labels and calculate evaluation metrics. |
| `ModelPriorityStrategy_ha1.py` | HA1-named version of the model-priority code; the two supplied versions have identical content. The main workflow calls the `_ha_common` version. |
| `ResultsParsing.py` | Export correctly and incorrectly classified examples for error analysis; imported by the model-priority script. |
| `combineResult.py` | Merge earlier article-level predictions with the later-stage article predictions and calculate overall results. |
| `ModelPriorityStrategy.py` | **Required by the earlier-stage and NP commands in the shell script, but not supplied in this upload.** |

The Python scripts also depend on the local modules `lib/preprocess.py`, `lib/model.py`, and `lib/utils.py`. These must be included in the released directory.

## 2. Environment and required inputs

- Python with PyTorch, Hugging Face Transformers, pandas, scikit-learn, Matplotlib, and Seaborn.
- Pretrained model: `bert-base-chinese` (downloaded through Hugging Face Transformers unless already cached).
- The `lib/` modules named above.
- The source dataset and any prerequisite article-level predictions referenced below.
- Run commands from the **root of this directory**, because the programs use relative paths.

The uploaded dependency file needs correction before it can be used as a reproducible installation manifest: `seaborn-0.13.2` should be expressed as `seaborn==0.13.2`, PyTorch is not listed, and several package versions are unpinned. After validating the actual experimental environment, provide a working `requirements.txt` and install it with:

```bash
python -m pip install -r requirements.txt
```

### Expected input/output locations

The scripts refer to the following paths:

| Location | Purpose |
| --- | --- |
| `data/data_4train/round_<n>/` | Training data (`train<id>_p.csv`, `train<id>_n.csv`) and validation data (`test<id>.csv`) for each binary classifier. |
| `data/data_4test/round_<n>/` | Test data (`test<id>.csv`) and combined test data (`test.csv`). |
| `model_f1/round_<n>/<id>.ckpt` | Checkpoints used by `evaluation.py` and `evaluation_nolabel5.py`. |
| `model_precision/round_<n>/<id>.ckpt` | Alternative precision-selected checkpoints saved during training. |
| `result/round_<n>/prediction_all.csv` | Per-segment outputs from the binary classifiers. |
| `result/round_<n>_SegmentLabel.csv` and `result/round_<n>_ArticleLabel.csv` | Combined segment/article predictions. |
| `ha1/article/prediction<n>.csv` | Earlier-stage article predictions required by final result merging. |
| `result/all_results/` | Combined results and correctly/incorrectly classified examples. |

Here, `<n>` runs from `0` to `4` by default. The source dataset layout before preprocessing is determined by `lib/preprocess.py`, which was not included in this upload.

## 3. Main HA1 execution sequence

The sequence below follows the **training → first evaluation → later-stage evaluation → final integration** structure in the supplied shell script. Two adjustments are explicitly indicated: the preprocessing program requires `-s`, and the full set of classifiers must be specified if training from scratch.

### Step 1 — Prepare training data

`preprocess_modify.py` dispatches preprocessing by `-s 1` or `-s 2`. The original shell script omitted this flag; the commands below explicitly select both preprocessing stages. Verify these stages against `lib/preprocess.py` before use.

```bash
python preprocess_modify.py -a ha1 -s 1
python preprocess_modify.py -a ha1 -s 2
```

The first stage prepares label-specific data; the second stage assembles the classifier training data, including negative examples.

### Step 2 — Train binary classifiers

```bash
python train.py -m "-2 -1 +1 +2 0 -3"
```

The command explicitly requests all six classifiers used by `evaluation_nolabel5.py`. **Important:** the uploaded `train.py` defaults to training only `-3` when `-m` is omitted, even though the original shell script simply runs `python train.py`. For exact reproduction of an existing experiment, verify whether the other checkpoints were trained previously and whether this command reflects the intended training setup.

Training defaults in the script include five rounds, 15 epochs, `bert-base-chinese`, and Adam with learning rate `1e-5`. Both F1-selected and precision-selected checkpoints may be saved.

### Step 3 — Prepare first-stage test data

```bash
python preprocess_modify.py -t test -a ha1 -s 1
python preprocess_modify.py -t test -a ha1 -s 2
```

These commands expand the shell script's original test-preparation command to specify the required stage argument. They require validation against the actual dataset layout.

### Step 4 — Generate first-stage predictions and article-level results

```bash
python evaluation_nolabel5.py
python ModelPriorityStrategy.py -m 5
```

`evaluation_nolabel5.py` generates `result/round_<n>/prediction_all.csv`. The second command appears in the shell script and is needed for the earlier-stage results, but **`ModelPriorityStrategy.py` is missing from the supplied directory**. This stage cannot be declared reproducible until that script (or a verified equivalent) is provided.

### Step 5 — Prepare later-stage test data

The shell script next selects inputs using the earlier-stage article predictions:

```bash
python preprocess_modify.py -t test -m combine_article -a ha1/article -s 1
```

This stage depends on the preceding article-level prediction files. Its precise filtering behavior is implemented inside `lib/preprocess.py`.

### Step 6 — Predict and combine segment/article labels

```bash
python evaluation_nolabel5.py
python ModelPriorityStrategy_ha_common.py -d rule
```

The first command generates binary-classifier predictions for the later-stage test data. The second command applies the classifier priority order:

```text
0 → 2 → -2 → -1 → 1 → -3
```

It also applies the `rule` article-decision method, computes classification metrics and confusion matrices, and calls `ResultsParsing.py` to extract correct/incorrect examples.

### Step 7 — Integrate and evaluate final results

```bash
python combineResult.py -c ha1/article
```

This command merges the article predictions from `ha1/article/prediction<n>.csv` with `result/round_<n>_ArticleLabel.csv` for the five rounds. It writes the merged article results and overall scores/confusion matrix under `result/`.

## 4. Optional NP and extended evaluation workflows

The original shell script also contains **additional commands**, not just the main HA1 run. They are documented separately because they represent optional NP training/evaluation and alternative testing setups. Running all blocks in one sequence may overwrite files in `data/` or `result/`.

### 4.1 NP-model training and evaluation

The shell script includes preprocessing and training in NP mode, followed by evaluation:

```bash
python preprocess_modify.py -p np -a ha1 -s 1
python preprocess_modify.py -p np -a ha1 -s 2
python preprocess_modify.py -p np -t test -a ha1 -s 1
python preprocess_modify.py -p np -t test -a ha1 -s 2
python train.py -p np
python evaluation.py -m np
python ModelPriorityStrategy.py -m np
```

The preprocessing stage flags are added here because they are required by the current `preprocess_modify.py`. The NP-specific evaluation still depends on the missing `ModelPriorityStrategy.py` and must be verified with the relevant NP test data.

### 4.2 Other hierarchical/combined evaluation experiments

The remaining commands in the shell script cover testing with both NP and non-NP data, including a `combine_article` variant, and then call `evaluation.py` and `ModelPriorityStrategy.py`. These are **alternative experiment configurations**, not additional mandatory steps after the main pipeline. Do not run them as part of the primary reproduction sequence unless the corresponding results are reported in the paper and the missing script is supplied.

