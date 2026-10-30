# Reproduction of the recorded experiments

## Scope and evidence

The repository contains the author’s original cleaned feature tables and two selected classifiers, exported from the archived experiment states into CSV and JSON. Their source checksums are recorded in [the manifest](../data/manifest.json). Model inference does not load a Python pickle.

`rot-reproduce replay` verifies file checksums, reconstructs the original predictions, checks them against independently recorded prediction hashes, and recalculates accuracy, precision, recall, F1, MCC, and confusion matrices. It covers 728 RoT-DS test rows, 704 KU-PCP test rows, and all 1,115 usable AVT rows.

`rot-reproduce refit` reconstructs each selected classifier from its training rows and retained feature subset. Both refits give exactly the original test labels. The RoT-DS command also fits its SelectKBest selector separately within each of the original five validation folds and checks every score.

The paper includes further comparisons, feature ablations, and combined-dataset experiments. The two released classifier states and their AVT evaluation do not by themselves reproduce those additional experiments or the results of other authors.

## Original evaluation design

The original notebook uses `train_test_split(test_size=0.2, random_state=42)` on each cleaned table, without stratification. Rows with missing values were removed before this split. In particular, the KU-PCP experiment pools the dataset’s original training and test partitions, then makes a new 80/20 split. Its numbers must not be described as evaluation on the dataset’s official test partition.

The historical search compares classifiers and selectors using the resulting test F1. Some random-forest feature selectors have no fixed seed, and a prefit Lasso selector was applied before cross-validation. These choices constrain the interpretation and repeatability of the original search. This repository preserves the selected feature subsets and reports a faithful reconstruction of the selected models. It does not rerun that stochastic search or present these test scores as an independent estimate after model selection. For a new study, choose features and hyperparameters using training folds and evaluate the chosen model once on an untouched test set.

## Validation score and paper discrepancy

The original RoT-DS logistic-regression validation scores, independently reproduced by refitting, are:

| Fold | F1 |
|---|---:|
| 1 | 0.7419898819561551 |
| 2 | 0.7650085763293311 |
| 3 | 0.7566666666666667 |
| 4 | 0.7450331125827815 |
| 5 | 0.7105719237435009 |
| **Mean** | **0.7438540322556871** |

The test F1 is 0.747638326585695. The paper’s prose reports 0.748 as the validation mean; that rounded value corresponds to the test F1, not the mean of these recorded folds. Reports generated here keep the two quantities separate.

## Features and numerical conventions

The recovered functions implement the original ResNet50 Grad-CAM, box selection, SAM segmentation, and geometry. [Source provenance](../data/provenance.json) records the original file hashes. Heavy image dependencies are optional; replay and classifier refits run without them.

Historical geometry retains these conventions because changing them would change the classifier inputs:

- Mask centerlines truncate coordinates to integers. Some grid lines use integer thirds, while box features use fractional thirds.
- The horizontal pointwise-distance function reverses the `(x, y)` tuple before evaluating its horizontal coordinate.
- The original table conversion sums the lengths of three-element tuples describing transitions. Its stored line-switch features therefore equal three times the number of recorded events.
- YOLO saliency crops can exclude the final row or column; the stored saliency sum preserves that crop.

Corrected geometry uses fractional thirds consistently, the intended coordinate axes, and actual transition counts. It changes the feature definition and requires refitting the classifiers. The original fitted states should only be used with historical features.

## Environment and repeatability

`requirements-lock.txt` pins the tested CPU environment on Python 3.12, including scikit-learn 1.6.1. The fitted gradient-boosting predictor standardizes inputs in float64, then casts to float32 before traversing trees, matching scikit-learn. The portable model implementation preserves that order.

The image pipeline additionally depends on checkpoint identity, preprocessing, PyTorch/CUDA, and image-decoding libraries. [Image setup](image_pipeline.md) records those choices. Replayed predictions from archived features and newly extracted image features are distinct checks; small changes in a detector or segmentation can alter downstream features.

## New image extraction

The released pipeline was run on all 2,547 recorded evaluation images using historical geometry. Every image produced all 22 finite features, with zero extraction errors. Checkpoint identity and every input image hash were verified. An independent comparison using the public comparison script gives the same counts and metrics as the validation analysis.

| Evaluation | Images | Predictions changed | Archived accuracy | Recomputed accuracy | Archived F1 | Recomputed F1 |
|---|---:|---:|---:|---:|---:|---:|
| RoT-DS | 728 | 1 | 0.743132 | 0.744505 | 0.747638 | 0.748649 |
| KU-PCP | 704 | 0 | 0.931818 | 0.931818 | 0.834483 | 0.834483 |
| AVT | 1,115 | 1 | 0.756054 | 0.756951 | 0.339806 | 0.343826 |

Extraction coverage passes. Exact feature and prediction parity with the archived study **differs**. All 22 features agree within `rtol=1e-5, atol=1e-7` for 436 images; at least one field differs beyond that tolerance for the other images. The [validation report](../results/image_validation.json) records differences for each field, complete metrics, source hashes, checkpoint hashes, and resource use. [Per-image predictions](../results/image_predictions.csv) and [recomputed features](../results/recomputed_image_features.csv.gz) are included.

The original archived tables remain the reference for reconstructing the paper’s recorded classifier results. The new image measurements provide a separate, versioned check of the released pipeline; they do not replace those archived results.

Two RTX 3090 processes used two CPU threads each. Their elapsed times were approximately 39.5 and 30.2 minutes. A completely fresh GPU environment then passed dependency checks, the full archived-prediction replay, and a three-image check covering all datasets. Those three feature vectors matched the extraction runtime exactly. [Software validation](software_validation.json) records the installation checks. The [image setup](image_pipeline.md#recompute-the-recorded-evaluation-images) explains how to reproduce the full image check.
