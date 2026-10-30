# Cross-Chemistry SOH Estimation — MATLAB Code

Code accompanying: R. A.-J. Mohammed and M. A. Sharif, "Machine Learning for
Cross-Chemistry SOH Estimation in Second-Life LiFePO4 Batteries," AIICRTA-2026.

This repository provides the MATLAB scripts used to (1) build the training
feature table from the public NASA PCoE Li-ion battery aging dataset, (2)
build the external-validation feature table from the authors' own
second-life LiFePO4 hardware measurements, (3) train and cross-validate the
two regression models reported in the paper (Gaussian Process Regression and
an LSBoost ensemble), and (4) reproduce every quantitative revision analysis
requested during peer review (trivial baseline, corrected SOH-definition
retrain, leave-one-battery-out cross-validation, and the normalization
ablation).

Requires MATLAB with the Statistics and Machine Learning Toolbox
(`fitrgp`, `fitrensemble`).

## Contents

```
scripts/
  nasa_feature_extraction.m      Builds the NASA PCoE training feature table
                                  from the public dataset (metadata.csv +
                                  per-cycle CSVs). Produces
                                  nasa_health_indicator_features.csv.
  parse_triplet_log.m            Reads one charge/discharge run logged as
                                  three two-column text files (time, value)
                                  -- the Arduino rig's raw log format.
  extract_health_indicators.m    Computes the thirteen scalar health
                                  indicators (Table I in the paper) from one
                                  parsed charge run.
  main_pipeline_example.m        Worked example: builds the two-row,
                                  real-hardware external-validation feature
                                  table (28C and 18C) from the raw Arduino
                                  logs. Produces
                                  health_indicator_features.csv.
  train_soh_models.m             Trains GPR and LSBoost on the NASA feature
                                  table, runs 5-fold cross-validation
                                  (Table II), and evaluates both models
                                  against the real-hardware external
                                  validation set (Table III). This is the
                                  script that produces every headline number
                                  reported in the paper.

  revision_analyses/             Scripts written during peer review to
                                  answer specific reviewer questions. Each
                                  one re-derives the relevant feature table
                                  internally (same filters as the main
                                  pipeline) rather than depending on a saved
                                  intermediate file, so each can be run
                                  independently.
    reviewer2_point1_analysis.m        Trivial mean-prediction baseline;
                                        GPR predictive variance / 95%
                                        interval on the hardware rows.
    reviewer2_point2_soh_definition.m  Retrains both models after
                                        rescaling NASA's BOL-relative SOH
                                        labels to the same nameplate-relative
                                        definition used for the hardware
                                        ground truth.
    reviewer2_point3_lobo_cv.m         Leave-one-battery-out
                                        cross-validation (replaces the
                                        random cycle-level 5-fold split).
    reviewer2_point5_normalization_ablation.m
                                        Capacity-based (C-rate) 
                                        normalization of the current/charge
                                        features, in addition to the
                                        existing voltage-ratio
                                        normalization; full pipeline +
                                        external validation.
    reviewer2_point5_cv_only.m         Lighter re-derivation of just the
                                        5-fold CV numbers from the ablation
                                        feature table.

data/
  health_indicator_features_hardware.csv
                                  The two-row (28C, 18C) real-hardware
                                  external-validation feature table actually
                                  used in Table III, including the
                                  StartVoltage_ratio / EndVoltage_ratio /
                                  EarlyVoltageSlope_ratioPerH columns
                                  train_soh_models.m's predictor list
                                  requires.
```

## Data availability

- **NASA PCoE training data** are public: B. Saha and K. Goebel, "Battery
  Data Set," NASA Ames Prognostics Data Repository, NASA Ames Research
  Center, Moffett Field, CA, 2007. The derived training feature table
  (`nasa_health_indicator_features.csv`, produced by
  `scripts/nasa_feature_extraction.m`) is large (several hundred thousand
  rows across all cycles) and is not included in this repository; it is
  regenerated directly from the public dataset by running
  `nasa_feature_extraction.m` against a local copy.
- **Real-hardware external-validation data**: the processed, two-row feature
  table used in the paper's Table III is included in `data/` above. The raw
  Arduino serial-monitor logs (multi-hour charge/discharge time series) are
  available from the authors upon reasonable request.

## Reproducing the paper's reported numbers

1. Download the NASA PCoE dataset and run `nasa_feature_extraction.m`
   against it to produce `nasa_health_indicator_features.csv`.
2. Place that file alongside
   `data/health_indicator_features_hardware.csv` (renamed to
   `health_indicator_features.csv`, as `train_soh_models.m` expects) and run
   `train_soh_models.m`. This reproduces Table II (5-fold cross-validation)
   and Table III (external validation on the real hardware), and is also
   the script used to obtain the GPR hyperparameter values (ARD length
   scales, signal and noise standard deviations) reported in response to
   peer review.
3. Each script under `scripts/revision_analyses/` can be run independently
   against the same NASA dataset to reproduce the corresponding revision
   analysis.
