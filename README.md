# Aspect Detection Experiments (HA1, HA2, S1, and S2)

This repository contains the Python implementations for four aspect-detection experiments: **HA1**, **HA2**, **S1**, and **S2**. Each aspect has its own directory, model settings, decision rules, and execution instructions.

The experiments use Chinese BERT-based binary classifiers to predict labels for text segments. Their outputs are combined into segment-level and article-level predictions, followed by evaluation across **five rounds**. Some configurations also use predictions from an earlier article-level stage or an auxiliary **NP** classifier.

> **Start with the README in the aspect directory you want to reproduce.** These are four aspect-specific experiments, not four commands to run sequentially. The original shell scripts were used as references for the documented command sequences and do not need to be included in the release.

## Repository structure

```text
.
├── README.md                         # This overview
├── HA1/
│   ├── README.md                     # HA1 setup and execution
│   └── ...                           # HA1 Python programs
├── HA2/
│   ├── README.md                     # HA2 setup and execution
│   └── ...                           # HA2 Python programs
├── S1/
│   ├── README.md                     # S1 setup and execution
│   └── ...                           # S1 Python programs
└── S2/
    ├── README.md                     # S2 setup and execution
    └── ...                           # S2 Python programs
```

| Directory | Experiment | Detailed instructions |
| --- | --- | --- |
| [`HA1/`](HA1/) | HA1 aspect detection; includes the common model-priority aggregation workflow and optional NP experiments. | [HA1 README](HA1/README.md) |
| [`HA2/`](HA2/) | HA2 aspect detection; includes its own model-priority variants and optional NP experiments. | [HA2 README](HA2/README.md) |
| [`S1/`](S1/) | S1 aspect detection; includes the original strategy and a later S1-specific strategy variant. | [S1 README](S1/README.md) |
| [`S2/`](S2/) | S2 aspect detection; its S2-specific article decision also uses previous article predictions and `NP_pred`. | [S2 README](S2/README.md) |

**Do not assume that identically named files are interchangeable across directories.** In particular, classifier thresholds, checkpoint selection, model-priority order, preprocessing settings, and article-decision logic may differ by aspect. Keep the relevant Python programs, models, data, and outputs together within each aspect directory.

## General experimental workflow

The directories follow a broadly similar sequence, with aspect-specific details documented locally:

1. **Preprocess training data.** Prepare label-specific segments and construct binary-classifier training inputs. The preprocessing programs generally distinguish stage 1 (`-s 1`) and stage 2 (`-s 2`).
2. **Train the classifiers.** Fine-tune `bert-base-chinese` classifiers for the relevant labels and save selected model checkpoints.
3. **Prepare and evaluate test data.** Generate per-classifier segment predictions, with aspect-specific probability thresholds and checkpoint choices.
4. **Aggregate predictions.** Resolve segment labels using a specified classifier-priority order, then determine article labels using the corresponding decision rules. Some variants incorporate earlier article-level predictions.
5. **Combine results.** Merge earlier and later-stage predictions, calculate classification scores, and export results for error analysis.

Optional NP-model and expanded-evaluation commands are described in the individual READMEs; they should not be assumed to be mandatory steps in every main experiment.

## Requirements and data

The scripts reference **Python**, **PyTorch**, **Hugging Face Transformers**, **pandas**, **NumPy**, **scikit-learn**, **Matplotlib**, and **Seaborn**, and use the pretrained model **`bert-base-chinese`**. Exact package versions should be taken from a validated experimental environment rather than inferred from these source files.

Before running an experiment, provide the following in the appropriate aspect directory:

- The required training and test datasets, with the layout and columns expected by that aspect's preprocessing and evaluation programs.
- The local Python modules `lib/preprocess.py`, `lib/model.py`, and `lib/utils.py`, which are imported by the scripts.
- Any prerequisite article-level prediction CSVs or pretrained checkpoints required by the selected execution path.

Generated files commonly appear under `data/data_4train/`, `data/data_4test/`, `model_f1/`, `model_precision/`, and `result/`, with fold-specific subdirectories or filenames. Exact inputs and outputs are specified in the relevant aspect README.

## How to run an aspect

Choose **one** aspect and execute the commands in its README **from that aspect's own directory**, because the programs use relative file paths. For example:

```bash
cd S2
# Follow the ordered Python commands in S2/README.md.
```

The directory READMEs distinguish the main workflow from optional experiments and identify commands that require adjustment or confirmation. Do not run all optional blocks consecutively without checking whether they overwrite shared intermediate files.

## Reproducibility notes

The included READMEs were prepared by inspecting the supplied Python programs and their original shell-script execution sequences. **An end-to-end reproduction has not been verified.** Before using the repository to reproduce published results, check the following points against the original experimental environment:

- Availability of the required `lib/` code, datasets, earlier-stage predictions, and checkpoints.
- Consistency of renamed files and legacy hard-coded paths, especially in the S1/S2 strategy variants.
- Differences between the original shell commands and the current command-line arguments; some optional commands reference programs not supplied with their aspect directory.
- Consistency of fold assignments, article identifiers, input CSV columns, and reported results.

See the **validation/compatibility notes** in [HA1](HA1/README.md), [HA2](HA2/README.md), [S1](S1/README.md), and [S2](S2/README.md) before claiming exact reproducibility.
