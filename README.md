# Cross-Cycle Explanation Drift in Solar Flare Forecasting

> **SHAP Stability from Solar Cycle 24 to Solar Cycle 25**

A reproducible research project investigating whether a solar-flare prediction model can maintain similar predictive performance across solar cycles while its internal feature-attribution pattern changes.

---

## Overview

Machine-learning models for solar-flare forecasting are commonly trained on historical active-region data. A model may appear to generalize well to a newer solar cycle based on metrics such as the **True Skill Statistic (TSS)**, but predictive accuracy alone does not tell us whether the model is relying on the same magnetic features.

This project studies that second question.

We train flare-prediction models using **Solar Cycle 24-era SHARP active-region data**, freeze those models, and evaluate them on:

1. an unseen **Cycle 24 holdout**, and  
2. **Solar Cycle 25 temporal blocks**.

We then compare both:

- **Predictive behavior** — TSS and complementary metrics
- **Explanation behavior** — SHAP feature-attribution structure

The goal is to determine whether **prediction drift** and **explanation drift** occur together or independently.

---

## Research Question

> **When a solar-flare prediction model trained only on Solar Cycle 24 SHARP data is applied unchanged to Solar Cycle 25, do its predictive skill and feature-attribution structure remain stable, or do they drift independently?**

A secondary question is:

> If SHAP explanations change, are those changes larger than normal within-cycle sampling variability, and do they remain meaningful after considering covariate shift, label-prevalence shift, and correlated SHARP features?

---

## Why This Research Matters

A model can preserve a similar prediction score while changing the features it relies on.

For example:

```text
Cycle 24
R_VALUE  -> most important
TOTUSJH  -> second
USFLUX   -> third

Cycle 25
USFLUX   -> most important
AREA     -> second
R_VALUE  -> third
```

If predictive performance remains similar while attribution structure changes substantially, a normal performance-only evaluation may miss an important model-behavior shift.

This project therefore evaluates **what the model predicts** and **how the model attributes importance to its inputs**.

---

## Research Gap

Previous research has already studied:

- solar-flare prediction across different solar-cycle periods,
- machine-learning models using SHARP parameters,
- SHAP and other explainability methods in flare forecasting,
- public Cycle 25 datasets and benchmarks.

This project **does not claim** to be:

- the first cross-cycle flare-prediction study,
- the first use of SHAP in solar-flare forecasting,
- the first Cycle 25 benchmark.

The narrower contribution is:

> **A controlled cross-cycle explanation-stability experiment in which the same frozen Cycle 24-trained model is evaluated on Cycle 24 and Cycle 25, while predictive drift and SHAP attribution drift are quantified separately and compared against a within-Cycle 24 null baseline.**

---

## Hypotheses

The study tests the following hypotheses rather than assuming them as results.

- **H1:** Predictive skill on Cycle 25 differs measurably from Cycle 24 holdout performance.
- **H2:** SHAP attribution structure differs across cycles beyond normal within-cycle resampling variability.
- **H3:** Explanation drift can occur without a commensurate loss in predictive skill.
- **H0:** Observed cross-cycle explanation differences are consistent with ordinary sampling and distribution-shift variability.

Both stable and drifting outcomes are scientifically informative.

---

## Data Sources

### Magnetic Features
**SDO/HMI SHARP** active-region parameters from JSOC.

Example features include:

- `R_VALUE`
- `TOTUSJH`
- `USFLUX`
- `AREA`
- additional SHARP magnetic parameters

### Flare Labels
A single, consistent **GOES flare catalog** source will be used for event labels.

### Study Windows

- **Cycle 24-era training window:** May 2010 – December 2019
- **Cycle 25 evaluation window:** 2020 onward, divided into temporal blocks

These are analysis windows aligned with HMI/SHARP availability and the December 2019 solar minimum.

### Prediction Task

Binary classification:

```text
Will this active region produce a >= M-class flare
within the next 24 hours?

Yes -> 1
No  -> 0
```

---

## Models

The primary models are:

- Logistic Regression
- Random Forest
- XGBoost

All models are trained **only on Cycle 24 training data**.

After training, the models are frozen before cross-cycle evaluation.

---

## Experimental Flow

```text
JSOC SHARP Data + GOES Flare Labels
                |
                v
      Common Dataset Pipeline
                |
                v
     Cleaning + Quality Control
                |
                v
     Active-Region-Level Split
                |
                v
      Cycle 24 Training Data
                |
                v
   LR / Random Forest / XGBoost
                |
                v
          Freeze Models
             /     \
            /       \
           v         v
 Cycle 24 Holdout   Cycle 25 Blocks
           |         |
           v         v
 Prediction Metrics
           |         |
           v         v
         SHAP Explanations
              \     /
               \   /
                v
       Cross-Cycle Comparison
                |
                v
     Distribution-Shift Audit
                |
                v
 Within-Cycle-24 Null Baseline
                |
                v
 Correlated-Feature Robustness
                |
                v
      Final Drift Interpretation
```

---

# 7 Research Phases

## Phase 1 — Data Collection & Dataset Design

### What
Collect SHARP magnetic features and GOES flare labels for Cycle 24 and Cycle 25.

### Why
Both cycles must be built using the same data definition and labeling logic so that any observed difference is not caused by inconsistent dataset construction.

### How
- Query JSOC for SHARP parameters
- Retrieve flare events from one consistent GOES catalog
- Record query dates and data provenance
- Define one fixed forecast window
- Build reproducible Cycle 24 and Cycle 25 datasets

---

## Phase 2 — Cleaning, Preprocessing & Leakage-Safe Splitting

### What
Clean the data and create train, validation, and test partitions.

### Why
Samples from the same active region are highly related. Allowing the same AR to appear in both training and testing can create data leakage and inflated performance.

### How
- Remove invalid / poor-quality observations
- Handle missing values
- Apply justified limb filtering
- Fit preprocessing only on Cycle 24 training data
- Split by **HARP / active region**
- Ensure no active region overlaps between train, validation, and test groups

---

## Phase 3 — Cycle 24 Training & Baseline Evaluation

### What
Train the three primary models using only Cycle 24 training data.

### Why
Cycle 24 provides the historical training regime. The Cycle 24 holdout gives a clean in-era baseline before the model encounters Cycle 25.

### How
- Train Logistic Regression
- Train Random Forest
- Train XGBoost
- Tune using Cycle 24 validation data only
- Freeze the final models
- Evaluate on unseen Cycle 24 holdout data

---

## Phase 4 — Frozen-Model Evaluation on Cycle 25

### What
Apply the exact same frozen models to Cycle 25.

### Why
This tests genuine temporal / cross-cycle generalization.

### How
- No retraining
- No Cycle 25 fine-tuning
- Same threshold-selection rule
- Evaluate multiple Cycle 25 temporal blocks
- Compare Cycle 25 performance against the Cycle 24 holdout baseline

---

## Phase 5 — Distribution Shift & Null Control

### What
Measure how Cycle 24 and Cycle 25 differ at the data level and establish normal within-cycle SHAP variability.

### Why
A SHAP difference does not automatically imply meaningful explanation drift. It may be caused by sampling noise, changed feature ranges, changed flare prevalence, or different active-region populations.

### How
Compare:

- feature distributions,
- missingness,
- active-region properties,
- positive-label prevalence.

Then repeatedly resample Cycle 24 holdout data and measure how much SHAP normally changes **within the same cycle**.

This creates the **null baseline**.

---

## Phase 6 — SHAP Explanation-Drift Analysis

### What
Compare the feature-attribution structure of the same frozen model on Cycle 24 and Cycle 25.

### Why
This is the central explainability experiment.

### How
Use a fixed, documented SHAP background / reference derived from Cycle 24 training data.

Measure:

- mean `|SHAP|` changes,
- attribution sign / direction,
- Spearman rank correlation,
- Kendall rank correlation,
- top-k feature overlap,
- attribution-distribution distances,
- statistical significance with multiple-comparison correction.

Cross-cycle explanation drift is considered meaningful only when it exceeds the expected within-Cycle-24 baseline.

---

## Phase 7 — Robustness & Scientific Interpretation

### What
Combine predictive results, SHAP results, data-shift evidence, and correlated-feature analysis.

### Why
SHAP describes **model behavior**, not causal solar physics.

### How
Strongly correlated SHARP parameters will be analyzed both individually and as feature groups.

Possible outcomes:

| Predictive Skill | Explanation Stability | Interpretation |
|---|---|---|
| Stable | Stable | Model transfers with a stable attribution structure |
| Stable | Drifts | Possible explanation drift despite similar aggregate skill |
| Drops | Drifts | Predictive and attribution behavior are both unstable |
| Drops | Stable | Similar features remain important, but predictive performance degrades |

We will **not** claim that SHAP proves that solar physics changed.

---

## Evaluation Metrics

### Predictive Metrics

Primary:

- **TSS — True Skill Statistic**

Complementary:

- HSS and/or F1
- PR-AUC
- Brier score
- calibration analysis

Confidence intervals will be obtained using **active-region-level bootstrap resampling** rather than treating individual time samples as independent.

### Explanation-Drift Metrics

- Mean absolute SHAP change
- SHAP sign consistency
- Spearman correlation
- Kendall correlation
- Top-k overlap
- Attribution-distribution distance
- Within-cycle null comparison
- Feature-group robustness

---

## Critical Methodological Rules

This project follows several non-negotiable rules:

- No active region appears in both training and test sets.
- Cycle 25 is never used to train the primary cross-cycle models.
- Preprocessing parameters are fitted only on Cycle 24 training data.
- The same frozen model is compared across cycles.
- SHAP background selection is fixed and documented.
- SHAP ranking alone is not treated as sufficient evidence.
- Correlated SHARP features are explicitly considered.
- Cross-cycle SHAP differences are compared against a within-cycle null baseline.
- Predictive metrics include confidence intervals.
- SHAP is interpreted as a model-explanation tool, **not causal evidence of solar physics**.
- No result is assumed before running the experiment.

---

## Reproducibility

The repository will aim to include:

```text
.
├── README.md
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── configs/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── shap_analysis/
│   └── utils/
├── experiments/
├── results/
│   ├── metrics/
│   ├── shap/
│   └── figures/
├── manifests/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── tests/
├── requirements.txt
└── environment.yml
```

The final release should preserve:

- dataset query scripts,
- data provenance,
- HARP split manifests,
- configuration files,
- random seeds,
- package versions,
- trained-model metadata,
- metric-generation scripts,
- SHAP analysis scripts,
- figure-generation scripts.

---

## Expected Figures

The analysis is expected to generate figures such as:

1. Cycle 24 vs Cycle 25 predictive performance with confidence intervals
2. Feature-distribution comparisons
3. SHAP summary plots for each evaluation regime
4. Side-by-side SHAP rankings
5. Cross-cycle drift versus within-cycle null variability
6. Feature-group attribution stability
7. Temporal Cycle 25 performance / attribution trends
8. Local SHAP case studies for selected active regions

---

## Current Status

**Research framing:** Locked  
**Primary research question:** Locked  
**Dataset strategy:** Defined  
**Evaluation strategy:** Defined  
**SHAP protocol:** Defined  
**Null-control strategy:** Defined  
**Implementation:** In progress / not yet completed  
**Scientific result:** Not yet known

> We do not assume that explanation drift exists. Stable and drifting results are both valid scientific outcomes.

---

## Research Scope

### In Scope

- Solar flare forecasting
- SDO/HMI SHARP parameters
- GOES flare labels
- Solar Cycles 24 and 25
- Logistic Regression
- Random Forest
- XGBoost
- TSS and complementary forecasting metrics
- SHAP explainability
- explanation-drift analysis
- distribution-shift analysis
- within-cycle null analysis
- correlated-feature robustness

### Out of Scope

Unless the scope is formally revised:

- claiming SHAP reveals causal solar physics,
- using Cycle 25 to train the primary cross-cycle model,
- claiming this is the first cross-cycle flare-prediction study,
- claiming this is the first use of SHAP in flare forecasting,
- claiming prediction stability automatically means explanation stability,
- declaring a result before completing the experiment.

---

## Literature Foundation

The project builds on prior research including:

1. Sun, Z. et al. (2022). *Predicting Solar Flares Using CNN and LSTM on Two Solar Cycles of Active Region Data.* The Astrophysical Journal, 931, 163.
2. Goodwin, G. T., Sadykov, V. M., & Martens, P. C. (2024). *Investigating Performance Trends of Simulated Real-time Solar Flare Predictions.* The Astrophysical Journal, 964, 163.
3. Hu, K. et al. (2025). *Data Quality Issues in Flare Prediction using Machine Learning Models.* arXiv:2512.13417.
4. Wu, Z. et al. (2026). *Prediction of major solar flares using interpretable class-dependent reward framework.* MNRAS.
5. Li, X. et al. (2026). *Operational Solar Flare Forecasting System Using an Explainable Large Language Model.* Space Weather.
6. Roy, S. et al. (2026). *SuryaBench: Benchmark Dataset for Advancing Machine Learning in Heliophysics.* Scientific Data.
7. Liu, N. et al. (2026). *FlareDB: A Database of Significant Flares in Solar Cycles 24 and 25.* Scientific Data.
8. Chen, Y. et al. (2019). *Identifying Solar Flare Precursors Using Time Series of SDO/HMI Images and SHARP Parameters.* Space Weather.

The literature search will be re-checked before submission, especially for any novelty-sensitive claim.

---

## Project Checkpoint — `Duck`

This project uses **`Duck`** as a research checkpoint command.

Whenever a checkpoint is performed, the project should be audited against the locked research design for:

- scope consistency,
- data-source consistency,
- HARP-level leakage prevention,
- Cycle 24-only primary training,
- frozen-model Cycle 25 evaluation,
- predictive metrics,
- distribution-shift analysis,
- within-Cycle-24 SHAP null baseline,
- SHAP protocol consistency,
- correlated-feature robustness,
- novelty / claim discipline,
- reproducibility,
- current phase status,
- risks and deviations,
- immediate next actions.

Checkpoint output:

```text
ON TRACK
MISSING
RISKS / DEVIATIONS
NEXT ACTIONS
FINAL DECISION:
Continue / Fix before continuing / Re-scope
```

---

## One-Line Summary

> **Train on Cycle 24, freeze the model, test it on Cycle 24 and Cycle 25, compare both prediction performance and SHAP explanations, then determine whether any cross-cycle explanation change is larger than normal sampling and data-distribution variation.**

---

## Disclaimer

This repository contains an ongoing scientific research project.

Results, conclusions, and novelty claims are considered provisional until:

- experiments are complete,
- statistical robustness checks are passed,
- the literature is re-verified,
- and the manuscript undergoes scientific review.

SHAP explanations are interpreted as **model-attribution evidence**, not as direct proof of causal solar physics.

---

## License

A license should be selected before public release.

Recommended options:

- **MIT License** for code
- Appropriate data-license / citation requirements for externally sourced scientific datasets
- **CC BY 4.0** for original documentation, if desired

Always preserve the original terms and attribution requirements of JSOC, SDO/HMI, GOES, and other external data sources.

---

## Author

**Swayam Jain**

Research area: **Space Weather · Machine Learning · Explainable AI**

---

## Citation

A formal repository citation and `CITATION.cff` file will be added when the first stable research release is published.
