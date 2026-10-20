# Flarex

## Cross-Cycle Explanation Drift in Solar Flare Forecasting

**SHAP Stability from Solar Cycle 24 to Solar Cycle 25**

Flarex is a reproducible machine-learning research project that studies whether a solar-flare forecasting model can preserve similar predictive performance across solar cycles while its internal feature-attribution behavior changes.

The study trains three model families only on **Solar Cycle 24-era SDO/HMI SHARP data**, freezes the complete modeling pipeline, and evaluates the same frozen models on:

- an unseen **Cycle 24 holdout**, and
- **Solar Cycle 25** temporal blocks.

The core question is:

> **When a flare-prediction model trained only on Cycle 24 SHARP data is applied unchanged to Cycle 25, do predictive skill and SHAP feature-attribution structure remain stable, or can they change independently?**

---

## Project Status

The scientific computation is complete and frozen.

| Component | Status |
|---|---|
| Dataset v2 construction and audit | Complete / Frozen |
| Leakage-safe Cycle 24 split | Complete / Frozen |
| Preprocessing | Complete / Frozen |
| Logistic Regression training | Complete / Frozen |
| Random Forest training | Complete / Frozen |
| XGBoost training | Complete / Frozen |
| Cycle 24 holdout evaluation | Complete / Frozen |
| Cycle 25 frozen-model evaluation | Complete / Frozen |
| Distribution-shift controls | Complete / Frozen |
| Within-Cycle-24 SHAP null baseline | Complete / Frozen |
| Cross-cycle SHAP analysis | Complete / Frozen |
| Featurewise BH-FDR inference | Complete / Frozen |
| Common-support / reweighting sensitivity | Complete / Frozen |
| Correlated-feature robustness | Complete / Frozen |
| Final scientific synthesis | Complete / Frozen |
| Historical inference demo | Complete |
| Manuscript writing | Pending professor guidance |
| Publication figures / final paper tables | Pending |
| Dependency lock / environment export | Pending |
| License / `CITATION.cff` | Pending before public release |

---

## Final Scientific Result

The frozen study conclusion is:

> **Predictive-change detection and explanation-change evidence did not necessarily move together.**

Across all three frozen models and all four Cycle 25 evaluation scopes, **no measurable change in the primary True Skill Statistic (TSS) was detected under the predefined 95% bootstrap confidence-interval criterion**.

This is **not** a proof of predictive equivalence or strict predictive stability.

At the explanation level:

| Model | Final explanation evidence |
|---|---|
| Logistic Regression | TIER 0 |
| Random Forest | TIER 2 |
| XGBoost | TIER 2 |

Random Forest and XGBoost showed both global explanation-structure null exceedance and featurewise BH-FDR-significant attribution changes. Logistic Regression did not exceed the frozen global structure null, although featurewise changes were still present.

Therefore, the final cross-model interpretation is:

> **Majority-model evidence of prediction-explanation separation with model dependence.**

The result is not universal across all model families.

---

## Hypothesis Outcomes

| Hypothesis | Frozen conclusion |
|---|---|
| **H0** — observed explanation differences are consistent with ordinary Cycle 24 variability and/or shift | Mixed; insufficient as a universal explanation |
| **H1** — predictive behavior differs measurably across cycles | Not supported by the predefined primary TSS criterion |
| **H2** — SHAP explanation structure differs beyond within-Cycle-24 variability | Supported with model dependence |
| **H3** — explanation behavior can change without a commensurate measurable change in primary predictive skill | Supported with qualification by majority-model evidence |

Important interpretation rules:

- failure to detect a TSS change does not establish equivalence,
- TIER 0 is not proof of explanation stability,
- SHAP is interpreted as model-attribution evidence, not causal solar physics.

---

# Research Design

## Prediction Task

For each SHARP active-region observation at time `t`, predict:

> **Will the associated active region produce at least one GOES M1.0+ flare within `(t, t + 24h]`?**

Binary target:

```text
1 = >= M1.0 flare occurs within the next 24 hours
0 = no >= M1.0 flare occurs within the next 24 hours
```

---

## Data Sources

### SDO/HMI SHARP

The project uses definitive SDO/HMI SHARP active-region data from:

```text
hmi.sharp_cea_720s
```

Primary sampling:

```text
120-minute deterministic TAI grid
```

### GOES flare events

GOES flare information is used to construct the prediction labels.

### NOAA/NCEI Solar Region Summary

Daily Solar Region Summary information is used to temporally validate active-region associations and reduce stale or ambiguous HARP-to-NOAA label assignments.

---

## Dataset v2

The frozen model input contains **25 magnetic / Lorentz-force-related features**:

```text
TOTUSJH
TOTPOT
TOTUSJZ
ABSNJZH
SAVNCPP
USFLUX
AREA_ACR
MEANPOT
R_VALUE
SHRGT45
MEANSHR
MEANGAM
MEANGBT
MEANGBZ
MEANGBH
MEANJZH
MEANJZD
MEANALP
TOTBSQ
TOTFX
TOTFY
TOTFZ
EPSX
EPSY
EPSZ
```

Dataset summary:

| Dataset | Rows | Positive | Negative | Entity groups |
|---|---:|---:|---:|---:|
| Cycle 24 | 88,435 | 3,462 | 84,973 | 1,174 |
| Cycle 25 | 79,361 | 4,836 | 74,525 | 1,011 |

Frozen files:

```text
cross-cycle-flare-drift/data/final/dataset_v2/cycle24_dataset_v2.parquet
cross-cycle-flare-drift/data/final/dataset_v2/cycle25_dataset_v2.parquet
```

Dataset contract:

```text
cross-cycle-flare-drift/data_manifests/dataset_v2_contract.json
```

---

# Leakage Control

A major methodological requirement is preventing connected active-region identities from appearing across development partitions.

The frozen leakage unit is:

> **Connected HARP-NOAA entity group (`ENTITY_GROUP_ID`)**

Cycle 24 is split deterministically into:

| Split | Rows | Positive | Negative | Entity groups |
|---|---:|---:|---:|---:|
| Train | 62,064 | 2,420 | 59,644 | 822 |
| Validation | 13,299 | 515 | 12,784 | 176 |
| Holdout | 13,072 | 527 | 12,545 | 176 |

Integrity checks confirm:

- zero entity-group overlap,
- zero HARP overlap,
- zero sample-key overlap,
- exact source partitioning.

Cycle 25 is reserved as a **temporal transfer domain** and is not used for model development.

---

# Preprocessing

The 25-feature representation is frozen.

### Logistic Regression

Uses a `StandardScaler` fitted on **Cycle 24 TRAIN only**.

After fitting:

```text
validation -> transform only
holdout    -> transform only
Cycle 25   -> transform only
```

No refitting is allowed.

### Random Forest

Uses the raw frozen feature representation.

### XGBoost

Uses the raw frozen feature representation.

The pipeline does not apply:

- PCA,
- post-hoc feature selection,
- imputation,
- winsorization,
- outlier clipping,
- log transforms,
- polynomial feature engineering.

---

# Models

Three model families are retained. No overall cross-family winner was selected.

## Logistic Regression

Selected model:

```text
LR_02_C0p01_BALANCED
```

Key configuration:

```text
C = 0.01
class_weight = balanced
penalty = L2
solver = lbfgs
max_iter = 5000
```

Frozen threshold:

```text
0.4548081408
```

---

## Random Forest

Selected model:

```text
RF_04_D12_L5_BALSUB
```

Key configuration:

```text
n_estimators = 500
max_depth = 12
min_samples_leaf = 5
class_weight = balanced_subsample
max_features = sqrt
bootstrap = True
```

Frozen threshold:

```text
0.1196946373
```

---

## XGBoost

Selected model:

```text
XGB_02_LR0p03_D3_BAL
```

Key configuration:

```text
n_estimators = 500
learning_rate = 0.03
max_depth = 3
subsample = 0.8
colsample_bytree = 0.8
scale_pos_weight = 24.6462809917
tree_method = hist
```

Frozen threshold:

```text
0.2638225555
```

---

# Model Selection Policy

Model candidates are trained on:

```text
Cycle 24 TRAIN only
```

Model and threshold selection use:

```text
Cycle 24 VALIDATION only
```

Cycle 24 HOLDOUT and Cycle 25 are not used for:

- hyperparameter tuning,
- threshold optimization,
- feature selection,
- preprocessing fitting,
- candidate substitution,
- retraining.

---

# Predictive Results

## Cycle 24 Holdout

| Model | TSS | 95% group-bootstrap CI | PR-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.7190 | [0.5774, 0.8221] | 0.5219 |
| Random Forest | 0.7096 | [0.5863, 0.7935] | 0.3562 |
| XGBoost | 0.7010 | [0.5776, 0.7841] | 0.5175 |

The primary metric is **TSS**.

Bootstrap uncertainty uses entity-group resampling rather than treating every time sample as independent.

---

## Cycle 25 Evaluation

Cycle 25 is evaluated in three predefined temporal blocks:

```text
B1 = 2020-2021
B2 = 2022-2023
B3 = 2024-2025
```

plus a pooled 2020-2025 summary.

### TSS by temporal block

| Model | B1 | B2 | B3 | Pooled |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7283 | 0.6785 | 0.7237 | 0.7156 |
| Random Forest | 0.7231 | 0.6600 | 0.7084 | 0.7019 |
| XGBoost | 0.6990 | 0.6368 | 0.7001 | 0.6873 |

### Pooled C25 vs C24 holdout

| Model | C24 holdout TSS | C25 pooled TSS | Delta |
|---|---:|---:|---:|
| Logistic Regression | 0.7190 | 0.7156 | -0.0034 |
| Random Forest | 0.7096 | 0.7019 | -0.0077 |
| XGBoost | 0.7010 | 0.6873 | -0.0137 |

All 12 model-by-C25-scope primary TSS delta confidence intervals contain zero.

Formal primary classification:

```text
NO_MEASURABLE_CHANGE_AT_95CI
```

This must not be described as a formal equivalence result.

---

# Distribution Shift and Robustness Controls

The study does not treat a raw Cycle 24-vs-Cycle 25 SHAP difference as sufficient evidence.

The frozen control pipeline includes:

- marginal feature-shift analysis,
- label-prevalence analysis,
- active-region population analysis,
- domain-classifier shift diagnostics,
- common-support analysis,
- C25-to-C24 reweighting sensitivity,
- correlated-feature grouping,
- class-conditional SHAP comparisons,
- within-Cycle-24 SHAP null distributions,
- featurewise empirical inference with BH-FDR.

---

## Label Prevalence

Positive prevalence changes substantially across the temporal domains.

Representative frozen values:

| Scope | Positive prevalence |
|---|---:|
| Cycle 24 holdout | 0.0403 |
| C25 B1: 2020-2021 | 0.0128 |
| C25 B2: 2022-2023 | 0.0544 |
| C25 B3: 2024-2025 | 0.0810 |
| C25 pooled | 0.0609 |

Prevalence is reported as a real distribution shift. No post-hoc prevalence-standardized SHAP rule is introduced.

---

# Correlated Feature Robustness

Feature correlation structure is estimated from **Cycle 24 TRAIN only**.

Rule:

```text
absolute Spearman rho >= 0.80
```

Connected components are used as robustness groups.

The final design contains 10 feature groups, including four multi-feature groups and six singletons.

These groups are an attribution-robustness device only and do not imply that correlated SHARP variables are physically identical or causally interchangeable.

---

# SHAP Protocol

SHAP is used to analyze model behavior, not causal solar physics.

The fixed SHAP background contains:

```text
512 Cycle 24 TRAIN-derived observations
```

The same frozen source background is used for cross-cycle attribution comparison.

Model-specific explainers:

- Logistic Regression -> linear explainer
- Random Forest -> tree-based SHAP
- XGBoost -> tree-based SHAP / raw-margin-compatible evaluation

Raw SHAP magnitudes are not pooled across different model families because their output spaces differ.

---

# Within-Cycle-24 SHAP Null Baseline

A key contribution of the study is the use of a frozen within-cycle explanation-variability baseline.

Population:

```text
Cycle 24 HOLDOUT
```

Bootstrap unit:

```text
ENTITY_GROUP_ID
```

Replicates:

```text
2000
```

Strata:

```text
OVERALL
TARGET_0
TARGET_1
```

Metrics include:

- Spearman correlation,
- Kendall tau-b,
- top-5 overlap,
- top-10 overlap,
- normalized L1 distance,
- cosine distance,
- signed cosine similarity,
- sign agreement.

A future cross-cycle difference is interpreted relative to this frozen same-cycle baseline.

---

# Cross-Cycle SHAP Results

Final evidence tiers:

| Model | Global structure result | Featurewise evidence | Final tier |
|---|---|---|---|
| Logistic Regression | No global null exceedance | Present | TIER 0 |
| Random Forest | Global null exceedance | Present | TIER 2 |
| XGBoost | Global null exceedance | Present | TIER 2 |

Overall featurewise BH-FDR evidence includes:

```text
LR  : 13 significant hypotheses / 8 unique features
RF  : 23 significant hypotheses / 14 unique features
XGB : 11 significant hypotheses / 9 unique features
```

TIER 0 does not mean "stable." It only means the predefined global structure-null criterion was not exceeded.

---

# Final Interpretation

The strongest defensible conclusion is:

> **Within the frozen Cycle 24-to-Cycle 25 experimental design, predictive-change detection and explanation-change evidence do not necessarily move together.**

Primary TSS showed no measurable cross-cycle change under the predefined criterion, while Random Forest and XGBoost exhibited explanation changes beyond the frozen Cycle 24 attribution baseline.

Therefore:

> Cross-cycle evaluation should examine both predictive performance and explanation behavior rather than using predictive metrics alone as evidence that a model's internal attribution structure remained unchanged.

---

# Historical Inference Demo

The project includes:

```text
cross-cycle-flare-drift/main.py
```

It provides a reproducible historical inference interface using the frozen artifacts.

Workflow:

```text
Historical date
    |
    v
Available NOAA active regions
    |
    v
Observation time
    |
    v
Exact 25 frozen features
    |
    +--> Logistic Regression
    |
    +--> Random Forest
    |
    +--> XGBoost
    |
    v
Probability + frozen threshold + model-specific prediction
```

Run from the research directory:

```powershell
cd cross-cycle-flare-drift
python main.py --date 2021-04-18
```

The program verifies frozen model/scaler artifacts before inference.

It reports the three models individually and intentionally does **not** generate a majority-vote ensemble because no ensemble rule was frozen during model development.

---

## Example Historical Output

For one Cycle 25 positive historical observation:

```text
Date      : 2021-04-18
NOAA AR   : 12816
HARPNUM   : 7570
Observation: 2021-04-18 23:59:23 UTC

Logistic Regression
Probability = 0.418768
Threshold   = 0.454808
Prediction  = NO FLARE

Random Forest
Probability = 0.077669
Threshold   = 0.119695
Prediction  = NO FLARE

XGBoost
Probability = 0.407424
Threshold   = 0.263823
Prediction  = FLARE

Actual historical target:
FLARE >= M1.0 within the next 24 hours
```

This model disagreement is expected and is not treated as a software error.

---

# Prospective / Live Inference

The current validated demo is historical.

A date outside the frozen Cycle 24 / Cycle 25 datasets is intentionally rejected.

Example:

```text
2026-01-01
```

produces a controlled message indicating that a separate SHARP data-fetch pipeline would be required.

No 2026 live-data performance claim is made.

---

# Repository Structure

```text
Flarex/
|
+-- README.md
|
+-- docs/
|   +-- Solar_Flare_Master_Research_Reference_v2.pdf
|   +-- 01_WHAT_We_Are_Doing_BASE_v2.pdf
|   +-- 02_WHY_We_Are_Doing_BASE_v2.pdf
|   +-- 03_HOW_We_Are_Doing_BASE_v2.pdf
|
+-- cross-cycle-flare-drift/
    |
    +-- main.py
    +-- pyproject.toml
    +-- configs/
    +-- data/
    +-- data_manifests/
    +-- artifacts/
    |   +-- splitting/
    |   +-- preprocessing/
    |   +-- models/
    |       +-- logistic/
    |       +-- random_forest/
    |       +-- xgboost/
    |       +-- evaluation/
    |       +-- shift_controls/
    |       +-- cross_cycle_shap/
    |       +-- final_synthesis/
    |
    +-- src/
        +-- data/
        +-- modeling/
```

---

# Key Frozen Artifacts

## Dataset contract

```text
cross-cycle-flare-drift/data_manifests/dataset_v2_contract.json
```

## Cycle 24 split contract

```text
cross-cycle-flare-drift/artifacts/splitting/contracts/cycle24_split_contract.json
```

## Preprocessing contract

```text
cross-cycle-flare-drift/artifacts/preprocessing/contracts/preprocessing_contract.json
```

## Frozen selected models

```text
cross-cycle-flare-drift/artifacts/models/selection/contracts/selected_models_contract.json
```

## Predictive drift contract

```text
cross-cycle-flare-drift/artifacts/models/evaluation/contracts/cross_cycle_predictive_drift_contract.json
```

## Distribution-shift and SHAP null controls

```text
cross-cycle-flare-drift/artifacts/models/shift_controls/contracts/chapter9_shift_null_controls_contract.json
```

## Cross-cycle SHAP package

```text
cross-cycle-flare-drift/artifacts/models/cross_cycle_shap/contracts/chapter10_cross_cycle_shap_contract.json
```

## Final scientific synthesis

```text
cross-cycle-flare-drift/artifacts/models/final_synthesis/contracts/chapter11_final_synthesis_contract.json
cross-cycle-flare-drift/artifacts/models/final_synthesis/contracts/chapter11_final_scientific_conclusion_contract.json
```

---

# Frozen Runtime Artifacts

The historical demo depends on the following frozen files:

```text
artifacts/models/logistic/models/LR_02_C0p01_BALANCED.joblib
artifacts/models/random_forest/models/RF_04_D12_L5_BALSUB.joblib
artifacts/models/xgboost/models/XGB_02_LR0p03_D3_BAL.joblib

artifacts/preprocessing/scalers/logistic_standard_scaler.joblib

data/final/dataset_v2/cycle24_dataset_v2.parquet
data/final/dataset_v2/cycle25_dataset_v2.parquet
```

Artifact hashes are verified against frozen contracts during the workflow.

---

# Scientific Guardrails

This project does **not** claim:

- predictive equivalence,
- strict predictive stability,
- universal explanation drift,
- universal explanation stability,
- SHAP causality,
- changed solar physics from SHAP alone,
- a "right for the wrong reasons" mechanism,
- that this is the first cross-cycle flare forecasting study,
- that this is the first use of SHAP in flare forecasting.

The contribution is narrower:

> **A controlled frozen-model cross-cycle experiment that evaluates predictive change and explanation change separately, using explicit within-cycle explanation baselines and robustness controls.**

---

# Reproducibility

The project targets:

```text
Python >= 3.11, < 3.12
```

The current repository preserves:

- frozen datasets,
- split contracts,
- data provenance manifests,
- model-training protocols,
- frozen preprocessing metadata,
- model-selection outputs,
- prediction artifacts,
- bootstrap outputs,
- SHAP artifacts,
- null distributions,
- final synthesis contracts,
- frozen runtime models,
- the historical inference interface.

A complete pinned dependency lock is still recommended before final archival/public release.

---

# Master Research Reference

A separate master reference document has been prepared for the full project history, including:

- dataset construction,
- model development,
- Chapters 1-12,
- key outputs,
- graphs,
- tables,
- example predictions,
- SHAP analysis,
- final scientific conclusions.

The manuscript phase should use the frozen repository artifacts and the master reference together as the source of truth.

---

# Manuscript Phase

The next research phase is paper preparation, subject to professor guidance.

Planned work:

```text
1. Publication-ready tables
2. Publication-ready figures
3. Final literature and novelty audit
4. Manuscript writing
5. Scientific claim audit
6. Journal selection
7. Journal-specific formatting
8. Submission package
```

The frozen scientific pipeline should not be changed merely to obtain a more attractive paper result.

---

# Current Repository Polish Still Pending

Before a formal archival/public release, the following should still be completed:

- exact dependency lock / `requirements.txt`,
- lightweight automated smoke tests,
- `LICENSE`,
- `CITATION.cff`,
- final paper figures,
- final paper tables,
- final manuscript,
- literature / novelty verification.

These are repository and publication tasks; they do not change the frozen scientific result.

---

# Source-of-Truth Documents

```text
docs/Solar_Flare_Master_Research_Reference_v2.pdf
docs/01_WHAT_We_Are_Doing_BASE_v2.pdf
docs/02_WHY_We_Are_Doing_BASE_v2.pdf
docs/03_HOW_We_Are_Doing_BASE_v2.pdf
```

The Master Research Reference v2 supersedes earlier research-reference versions.

---

# Author

**Swayam Jain**

Research area:

**Space Weather · Machine Learning · Explainable AI**

---

# Citation

A formal repository citation and `CITATION.cff` should be added with the first archival/public research release.

---

# License

A repository license has not yet been finalized.

Any future public release should preserve the original attribution and usage requirements associated with SDO/HMI, JSOC, GOES, NOAA/NCEI, and other external scientific data sources.

---

## One-Line Summary

> **Train on Cycle 24, freeze the complete pipeline, test unchanged on Cycle 24 holdout and Cycle 25, and determine whether predictive performance and SHAP explanation behavior move together or separately.**
