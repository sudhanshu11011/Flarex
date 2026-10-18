from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

MANIFEST_DIR = Path(
    "data_manifests"
)


DATASET_CONTRACT_FILE = (
    MANIFEST_DIR
    /
    "dataset_v2_contract.json"
)

DATASET_FILES_FILE = (
    MANIFEST_DIR
    /
    "dataset_v2_files.csv"
)

DATASET_AUDIT_FILE = (
    MANIFEST_DIR
    /
    "pre_model_dataset_v2_full_audit.csv"
)

SRS_SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "pre_model_srs_all_positive_summary.csv"
)


METHOD_FILE = (
    MANIFEST_DIR
    /
    "pre_model_methodology_closure.md"
)

GATE_FILE = (
    MANIFEST_DIR
    /
    "pre_model_gate.json"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# REQUIRED INPUTS
# ============================================================

required = [
    DATASET_CONTRACT_FILE,
    DATASET_FILES_FILE,
    DATASET_AUDIT_FILE,
    SRS_SUMMARY_FILE,
]


for path in required:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# LOAD
# ============================================================

with DATASET_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    contract = json.load(
        file
    )


files = pd.read_csv(
    DATASET_FILES_FILE
)


audit = pd.read_csv(
    DATASET_AUDIT_FILE
)


srs_summary = pd.read_csv(
    SRS_SUMMARY_FILE
)


# ============================================================
# VERIFY DATASET v2
# ============================================================

if contract.get("status") != "FROZEN":

    raise RuntimeError(
        "Dataset v2 is not frozen."
    )


if contract.get(
    "dataset_version"
) != "2.0":

    raise RuntimeError(
        "Unexpected dataset version."
    )


if len(audit) != 2:

    raise RuntimeError(
        "Dataset audit should contain "
        "two cycle rows."
    )


if not audit[
    "cycle_audit_pass"
].astype(bool).all():

    raise RuntimeError(
        "Dataset v2 full audit is not PASS."
    )


if len(files) != 2:

    raise RuntimeError(
        "Dataset v2 file manifest invalid."
    )


# ============================================================
# EXPECTED FINAL COUNTS
# ============================================================

expected = {

    "cycle24": {
        "rows": 88435,
        "positive": 3462,
        "negative": 84973,
    },

    "cycle25": {
        "rows": 79361,
        "positive": 4836,
        "negative": 74525,
    },
}


for cycle in [
    "cycle24",
    "cycle25",
]:

    row = files[
        files["cycle"]
        ==
        cycle
    ]


    if len(row) != 1:

        raise RuntimeError(
            f"Missing file manifest row: {cycle}"
        )


    row = row.iloc[0]


    for field in [
        "rows",
        "positive",
        "negative",
    ]:

        if int(
            row[field]
        ) != expected[
            cycle
        ][
            field
        ]:

            raise RuntimeError(
                f"{cycle}: unexpected "
                f"{field} count."
            )


# ============================================================
# METHODOLOGY DECISIONS
# ============================================================

methodology = {

    "quality_filter": {

        "rule":
            "QUALITY == 0",

        "status":
            "FROZEN",

        "interpretation":
            (
                "Only HMI/SHARP records with no known "
                "QUALITY flags set are retained."
            ),

        "rationale":
            (
                "HMI QUALITY is a bit-mask describing "
                "known acquisition or processing issues. "
                "QUALITY=0 represents a conservative "
                "nominal-quality-only selection."
            ),

        "important_note":
            (
                "This is more conservative than some "
                "published flare-prediction pipelines that "
                "permit selected non-zero QUALITY states. "
                "The threshold was not optimized using "
                "Cycle25 outcomes."
            ),
    },

    "disk_filter": {

        "rule":
            "|LON_FWT| <= 68 degrees",

        "status":
            "FROZEN",

        "rationale":
            (
                "Restricts analysis to active regions "
                "sufficiently far from the solar limb to "
                "reduce projection and foreshortening "
                "effects."
            ),

        "important_note":
            (
                "The 68-degree threshold is treated as "
                "a fixed literature/operational convention, "
                "not a threshold tuned against flare labels "
                "or Cycle25 results."
            ),
    },

    "srs_region_validation": {

        "status":
            "FROZEN",

        "purpose":
            (
                "Resolve time-specific NOAA region "
                "association for positive flare labels."
            ),

        "method":
            (
                "Use the Solar Region Summary reports "
                "bracketing the UTC observation date. "
                "A matched NOAA flare region must be "
                "present in both bracketing reports to "
                "receive high-confidence temporal support."
            ),

        "supported_positive_rows":
            8298,

        "temporal_ambiguity_censored":
            393,

        "source_missing_censored":
            42,

        "stale_association_relabeled_negative":
            95,

        "stale_with_unassigned_m1_censored":
            8,

        "limitation":
            (
                "Historical SRS warehouse products are "
                "preliminary operational reports and may "
                "contain errors or later revisions. "
                "They are therefore used as a temporal "
                "association control with explicit censoring, "
                "not as model input features."
            ),
    },

    "cycle25_policy": {

        "status":
            "FROZEN",

        "allowed":
            [
                "final frozen transfer evaluation",
                "distribution-shift analysis",
                "cross-cycle SHAP comparison",
            ],

        "forbidden":
            [
                "model training",
                "preprocessing fitting",
                "feature selection",
                "hyperparameter tuning",
                "decision-threshold tuning",
            ],
    },

    "modeling_dataset": {

        "primary_version":
            "Dataset v2.0",

        "dataset_v1_status":
            (
                "Preserved as an immutable audit and "
                "sensitivity-analysis artifact."
            ),

        "cycle24_role":
            (
                "Development domain for entity-group-safe "
                "train, validation, and independent holdout."
            ),

        "cycle25_role":
            (
                "Frozen temporal/domain transfer test."
            ),
    },
}


# ============================================================
# WRITE MARKDOWN
# ============================================================

markdown = f"""# Pre-Model Methodology Closure

Generated: {datetime.now(timezone.utc).isoformat()}

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
"""


METHOD_FILE.write_text(
    markdown,
    encoding="utf-8",
)


# ============================================================
# WRITE MACHINE-READABLE GATE
# ============================================================

gate = {

    "status":
        "OPEN_FOR_MODELING",

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "primary_dataset_version":
        "2.0",

    "dataset_contract_sha256":
        sha256_file(
            DATASET_CONTRACT_FILE
        ),

    "dataset_files_manifest_sha256":
        sha256_file(
            DATASET_FILES_FILE
        ),

    "dataset_full_audit_sha256":
        sha256_file(
            DATASET_AUDIT_FILE
        ),

    "srs_summary_sha256":
        sha256_file(
            SRS_SUMMARY_FILE
        ),

    "methodology":
        methodology,

    "modeling_rules": {

        "cycle24_only_for_development":
            True,

        "cycle25_tuning_allowed":
            False,

        "entity_group_safe_splitting_required":
            True,

        "preprocessing_fit_train_only":
            True,

        "threshold_tuning_cycle24_validation_only":
            True,

        "cycle25_evaluation_after_freeze_only":
            True,
    },
}


with GATE_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        gate,
        file,
        indent=2,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "===================================="
)

print(
    "FINAL PRE-MODEL METHODOLOGY CLOSURE"
)

print(
    "===================================="
)


print(
    "Dataset v2 status: FROZEN"
)


print(
    "Dataset v2 audit: PASS"
)


print(
    "QUALITY rule: FROZEN"
)


print(
    "Disk-position rule: FROZEN"
)


print(
    "SRS temporal association rule: FROZEN"
)


print(
    "Cycle25 isolation rule: FROZEN"
)


print(
    "Cross-cycle leakage: 0"
)


print(
    "\nSaved:"
)

print(
    METHOD_FILE
)

print(
    GATE_FILE
)


print(
    "\n===================================="
)

print(
    "PRE-MODEL GATE: OPEN FOR MODELING"
)

print(
    "===================================="
)