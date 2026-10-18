# Pre-Model Methodology Closure

Generated: 2026-10-18T09:43:05.052212+00:00

## Status

**PRE-MODEL GATE: OPEN**

Primary modeling dataset: **Dataset v2.0**

---

## Dataset v2 Final State

### Cycle 24

- Rows: 88,435
- Positive: 3,462
- Negative: 84,973

### Cycle 25

- Rows: 79,361
- Positive: 4,836
- Negative: 74,525

- Model features: 25
- Cross-cycle ENTITY_GROUP_ID overlap: 0
- Cross-cycle HARPNUM overlap: 0
- Dataset v2 full audit: PASS

---

## QC Rule — QUALITY

Primary rule:

`QUALITY == 0`

This is a conservative nominal-quality-only rule.

The HMI QUALITY keyword is a bit-mask describing known
acquisition and processing problems. Records with QUALITY=0
have no known quality flags set.

This study intentionally applies the stricter zero-only rule
rather than attempting to optimize acceptable QUALITY bit
combinations.

This decision was fixed before model fitting and was not
selected using Cycle25 predictive outcomes.

---

## Disk-Position Rule

Primary rule:

`|LON_FWT| <= 68 degrees`

The restriction is used to reduce projection and
foreshortening effects near the solar limb.

The threshold is treated as a fixed literature/operational
convention rather than a data-optimized cutoff.

---

## NOAA/SRS Temporal Association

SHARP `NOAA_ARS` was found to be effectively static over the
lifetime of a HARP and therefore cannot be treated as a
timestamp-specific active-region association.

Dataset v2 uses daily NOAA/NCEI Solar Region Summary (SRS)
reports as a temporal validation control for positive labels.

Final positive-label audit:

- Supported positive rows: 8,298
- Temporal-ambiguity censored: 393
- Missing-SRS-source censored: 42
- Stale positives relabeled negative: 95
- Stale association + unassigned M1+ censored: 8

SRS reports are operational daily products describing regions
observed during the preceding UTC day.

Historical warehouse reports are preliminary and may contain
errors or later revisions. Therefore SRS information is used
only for region-association validation/censoring and is never
used as a predictive feature.

---

## Leakage Unit

Primary leakage unit:

`connected HARP-NOAA ENTITY_GROUP_ID`

An entity component must never be split between Cycle24 and
Cycle25.

Current cross-cycle overlap:

- ENTITY_GROUP_ID: 0
- HARPNUM: 0

---

## Cycle 25 Isolation Rule

Cycle25 is a frozen temporal/domain transfer evaluation set.

Cycle25 must not be used for:

- preprocessing fitting
- feature selection
- model training
- hyperparameter tuning
- threshold tuning

Cycle25 may only be used after the Cycle24 model pipeline is
fully frozen.

---

## Model Gate

Dataset v2 has passed:

- schema validation
- 25-feature validation
- finite-value validation
- target validation
- 120-minute-grid validation
- 24-hour-horizon validation
- lineage validation
- SRS label correction validation
- cross-cycle leakage validation
- checksum freeze

**MODEL TRAINING MAY BEGIN.**
