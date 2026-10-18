from pathlib import Path
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "data/final/dataset_v2/cycle24_dataset_v2.parquet"
)

ASSIGNMENT_FILE = Path(
    "data_manifests/model_ch1_split_assignments.csv"
)

ASSIGNMENT_SUMMARY_FILE = Path(
    "data_manifests/model_ch1_split_assignment_summary.csv"
)

PRE_MODEL_GATE = Path(
    "data_manifests/pre_model_gate.json"
)


OUTPUT_DIR = Path(
    "artifacts/splitting/manifests"
)

AUDIT_OUTPUT = (
    OUTPUT_DIR
    /
    "model_ch1_split_audit_summary.csv"
)

YEAR_OUTPUT = (
    OUTPUT_DIR
    /
    "model_ch1_split_year_balance.csv"
)

OVERLAP_OUTPUT = (
    OUTPUT_DIR
    /
    "model_ch1_split_overlap_audit.csv"
)


# ============================================================
# FROZEN SPLIT EXPECTATIONS
# ============================================================

EXPECTED_GROUPS = {
    "TRAIN": 822,
    "VALIDATION": 176,
    "HOLDOUT": 176,
}

EXPECTED_SPLITS = {
    "TRAIN",
    "VALIDATION",
    "HOLDOUT",
}


# Broad audit tolerances.
# These are integrity/balance checks, not optimization targets.

MAX_ROW_FRACTION_DEVIATION = 0.02
MAX_POSITIVE_RATE_DEVIATION_PCT = 0.50

MIN_POSITIVE_GROUPS = 20


TARGET_RATIOS = {
    "TRAIN": 0.70,
    "VALIDATION": 0.15,
    "HOLDOUT": 0.15,
}


# ============================================================
# INPUT VALIDATION
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 1.3 — SPLIT AUDIT"
)

print(
    "===================================="
)


for path in [
    DATA_FILE,
    ASSIGNMENT_FILE,
    ASSIGNMENT_SUMMARY_FILE,
    PRE_MODEL_GATE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


with PRE_MODEL_GATE.open(
    "r",
    encoding="utf-8",
) as file:

    gate = json.load(file)


if gate.get("status") != "OPEN_FOR_MODELING":

    raise RuntimeError(
        "Pre-model gate is not open."
    )


df = pd.read_parquet(
    DATA_FILE
)


assignments = pd.read_csv(
    ASSIGNMENT_FILE
)


original_summary = pd.read_csv(
    ASSIGNMENT_SUMMARY_FILE
)


# ============================================================
# DATASET STATE
# ============================================================

required_data_columns = {
    "HARPNUM",
    "T_REC",
    "ENTITY_GROUP_ID",
    "observation_time_utc",
    "target",
}


missing = (
    required_data_columns
    -
    set(df.columns)
)


if missing:

    raise RuntimeError(
        f"Dataset missing columns: {sorted(missing)}"
    )


required_assignment_columns = {
    "ENTITY_GROUP_ID",
    "split",
}


missing = (
    required_assignment_columns
    -
    set(assignments.columns)
)


if missing:

    raise RuntimeError(
        "Assignment manifest missing columns: "
        f"{sorted(missing)}"
    )


df["_year"] = pd.to_datetime(
    df["observation_time_utc"],
    utc=True,
    errors="raise",
).dt.year


total_rows = len(df)

total_positive = int(
    df["target"].sum()
)

total_negative = (
    total_rows
    -
    total_positive
)

overall_positive_rate = (
    total_positive
    /
    total_rows
    *
    100
)


dataset_groups = set(
    df[
        "ENTITY_GROUP_ID"
    ].astype(str)
)


assignments[
    "ENTITY_GROUP_ID"
] = assignments[
    "ENTITY_GROUP_ID"
].astype(str)


assignment_groups = set(
    assignments[
        "ENTITY_GROUP_ID"
    ]
)


# ============================================================
# ASSIGNMENT MANIFEST INTEGRITY
# ============================================================

duplicate_assignment_groups = int(
    assignments[
        "ENTITY_GROUP_ID"
    ]
    .duplicated()
    .sum()
)


unexpected_splits = (
    set(
        assignments[
            "split"
        ].unique()
    )
    -
    EXPECTED_SPLITS
)


missing_dataset_groups = (
    dataset_groups
    -
    assignment_groups
)


extra_assignment_groups = (
    assignment_groups
    -
    dataset_groups
)


if duplicate_assignment_groups != 0:

    raise RuntimeError(
        "Duplicate ENTITY_GROUP_ID assignments found."
    )


if unexpected_splits:

    raise RuntimeError(
        "Unexpected split names: "
        f"{sorted(unexpected_splits)}"
    )


if missing_dataset_groups:

    raise RuntimeError(
        f"{len(missing_dataset_groups)} "
        "dataset groups have no split assignment."
    )


if extra_assignment_groups:

    raise RuntimeError(
        f"{len(extra_assignment_groups)} "
        "assigned groups do not exist in dataset."
    )


# ============================================================
# ATTACH SPLIT TO EVERY ROW
# ============================================================

split_map = dict(
    zip(
        assignments[
            "ENTITY_GROUP_ID"
        ],
        assignments[
            "split"
        ],
    )
)


df["_entity_string"] = (
    df[
        "ENTITY_GROUP_ID"
    ].astype(str)
)


df["_split"] = (
    df[
        "_entity_string"
    ].map(
        split_map
    )
)


missing_split_rows = int(
    df[
        "_split"
    ].isna()
    .sum()
)


if missing_split_rows != 0:

    raise RuntimeError(
        "Some Cycle24 rows are not assigned."
    )


# ============================================================
# SAMPLE-KEY INTEGRITY
# ============================================================

duplicate_sample_keys = int(
    df[
        [
            "HARPNUM",
            "T_REC",
        ]
    ]
    .duplicated()
    .sum()
)


if duplicate_sample_keys != 0:

    raise RuntimeError(
        "Duplicate Cycle24 sample keys found."
    )


# ============================================================
# BUILD SPLIT GROUP SETS
# ============================================================

group_sets = {}


for split in [
    "TRAIN",
    "VALIDATION",
    "HOLDOUT",
]:

    group_sets[
        split
    ] = set(
        assignments.loc[
            assignments[
                "split"
            ]
            ==
            split,
            "ENTITY_GROUP_ID",
        ]
    )


train_validation_group_overlap = (
    group_sets["TRAIN"]
    &
    group_sets["VALIDATION"]
)


train_holdout_group_overlap = (
    group_sets["TRAIN"]
    &
    group_sets["HOLDOUT"]
)


validation_holdout_group_overlap = (
    group_sets["VALIDATION"]
    &
    group_sets["HOLDOUT"]
)


# ============================================================
# HARP-LEVEL OVERLAP
# ============================================================

harp_sets = {}


for split in [
    "TRAIN",
    "VALIDATION",
    "HOLDOUT",
]:

    subset = df[
        df[
            "_split"
        ]
        ==
        split
    ]


    harp_sets[
        split
    ] = set(
        pd.to_numeric(
            subset[
                "HARPNUM"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )


train_validation_harp_overlap = (
    harp_sets["TRAIN"]
    &
    harp_sets["VALIDATION"]
)


train_holdout_harp_overlap = (
    harp_sets["TRAIN"]
    &
    harp_sets["HOLDOUT"]
)


validation_holdout_harp_overlap = (
    harp_sets["VALIDATION"]
    &
    harp_sets["HOLDOUT"]
)


# ============================================================
# SPLIT SUMMARY
# ============================================================

audit_rows = []


for split in [
    "TRAIN",
    "VALIDATION",
    "HOLDOUT",
]:

    subset = df[
        df[
            "_split"
        ]
        ==
        split
    ]


    rows = len(subset)


    positive = int(
        subset[
            "target"
        ].sum()
    )


    negative = (
        rows
        -
        positive
    )


    groups = int(
        subset[
            "ENTITY_GROUP_ID"
        ].nunique()
    )


    positive_group_count = int(
        subset.groupby(
            "ENTITY_GROUP_ID"
        )[
            "target"
        ]
        .sum()
        .gt(0)
        .sum()
    )


    row_fraction = (
        rows
        /
        total_rows
    )


    positive_rate = (
        positive
        /
        rows
        *
        100
    )


    target_ratio = TARGET_RATIOS[
        split
    ]


    row_fraction_deviation = abs(
        row_fraction
        -
        target_ratio
    )


    positive_rate_deviation = abs(
        positive_rate
        -
        overall_positive_rate
    )


    group_count_ok = (
        groups
        ==
        EXPECTED_GROUPS[
            split
        ]
    )


    row_fraction_ok = (
        row_fraction_deviation
        <=
        MAX_ROW_FRACTION_DEVIATION
    )


    positive_rate_ok = (
        positive_rate_deviation
        <=
        MAX_POSITIVE_RATE_DEVIATION_PCT
    )


    positive_group_ok = (
        positive_group_count
        >=
        MIN_POSITIVE_GROUPS
    )


    split_pass = (
        group_count_ok
        and
        row_fraction_ok
        and
        positive_rate_ok
        and
        positive_group_ok
    )


    audit_rows.append(
        {
            "split":
                split,

            "rows":
                rows,

            "row_fraction":
                row_fraction,

            "target_ratio":
                target_ratio,

            "row_fraction_deviation":
                row_fraction_deviation,

            "positive":
                positive,

            "negative":
                negative,

            "positive_rate_pct":
                positive_rate,

            "overall_positive_rate_pct":
                overall_positive_rate,

            "positive_rate_deviation_pct":
                positive_rate_deviation,

            "entity_groups":
                groups,

            "positive_groups":
                positive_group_count,

            "group_count_ok":
                group_count_ok,

            "row_fraction_ok":
                row_fraction_ok,

            "positive_rate_ok":
                positive_rate_ok,

            "positive_group_ok":
                positive_group_ok,

            "split_audit_pass":
                split_pass,
        }
    )


audit_df = pd.DataFrame(
    audit_rows
)


# ============================================================
# RECONCILE ORIGINAL CHAPTER 1.2 SUMMARY
# ============================================================

for split in EXPECTED_SPLITS:

    original = original_summary[
        original_summary[
            "split"
        ]
        ==
        split
    ]


    current = audit_df[
        audit_df[
            "split"
        ]
        ==
        split
    ]


    if (
        len(original) != 1
        or
        len(current) != 1
    ):

        raise RuntimeError(
            f"Summary row mismatch: {split}"
        )


    original = original.iloc[0]
    current = current.iloc[0]


    if (
        int(original["rows"])
        !=
        int(current["rows"])
        or
        int(original["positive"])
        !=
        int(current["positive"])
        or
        int(original["negative"])
        !=
        int(current["negative"])
        or
        int(original["entity_groups"])
        !=
        int(current["entity_groups"])
    ):

        raise RuntimeError(
            f"Chapter 1.2 reconciliation failed: "
            f"{split}"
        )


# ============================================================
# YEAR-WISE BALANCE
# ============================================================

year_rows = []


all_years = sorted(
    df[
        "_year"
    ].unique()
)


for split in [
    "TRAIN",
    "VALIDATION",
    "HOLDOUT",
]:

    subset = df[
        df[
            "_split"
        ]
        ==
        split
    ]


    for year in all_years:

        year_subset = subset[
            subset[
                "_year"
            ]
            ==
            year
        ]


        rows = len(
            year_subset
        )


        positive = int(
            year_subset[
                "target"
            ].sum()
        )


        negative = (
            rows
            -
            positive
        )


        year_rows.append(
            {
                "split":
                    split,

                "year":
                    int(year),

                "rows":
                    rows,

                "positive":
                    positive,

                "negative":
                    negative,

                "positive_rate_pct":
                    (
                        positive
                        /
                        rows
                        *
                        100
                        if rows > 0
                        else 0
                    ),

                "entity_groups":
                    int(
                        year_subset[
                            "ENTITY_GROUP_ID"
                        ].nunique()
                    ),
            }
        )


year_df = pd.DataFrame(
    year_rows
)


# Every split must contain observations from every Cycle24 year.

missing_year_cells = int(
    (
        year_df[
            "rows"
        ]
        ==
        0
    ).sum()
)


# ============================================================
# TOTAL ACCOUNTING
# ============================================================

reconciled_rows = int(
    audit_df[
        "rows"
    ].sum()
)


reconciled_positive = int(
    audit_df[
        "positive"
    ].sum()
)


reconciled_negative = int(
    audit_df[
        "negative"
    ].sum()
)


total_accounting_ok = (
    reconciled_rows
    ==
    total_rows
    and
    reconciled_positive
    ==
    total_positive
    and
    reconciled_negative
    ==
    total_negative
)


# ============================================================
# OVERLAP AUDIT TABLE
# ============================================================

overlap_df = pd.DataFrame(
    [
        {
            "overlap_type":
                "ENTITY_GROUP_ID",

            "pair":
                "TRAIN_VALIDATION",

            "count":
                len(
                    train_validation_group_overlap
                ),
        },

        {
            "overlap_type":
                "ENTITY_GROUP_ID",

            "pair":
                "TRAIN_HOLDOUT",

            "count":
                len(
                    train_holdout_group_overlap
                ),
        },

        {
            "overlap_type":
                "ENTITY_GROUP_ID",

            "pair":
                "VALIDATION_HOLDOUT",

            "count":
                len(
                    validation_holdout_group_overlap
                ),
        },

        {
            "overlap_type":
                "HARPNUM",

            "pair":
                "TRAIN_VALIDATION",

            "count":
                len(
                    train_validation_harp_overlap
                ),
        },

        {
            "overlap_type":
                "HARPNUM",

            "pair":
                "TRAIN_HOLDOUT",

            "count":
                len(
                    train_holdout_harp_overlap
                ),
        },

        {
            "overlap_type":
                "HARPNUM",

            "pair":
                "VALIDATION_HOLDOUT",

            "count":
                len(
                    validation_holdout_harp_overlap
                ),
        },
    ]
)


zero_overlap = (
    overlap_df[
        "count"
    ].eq(0).all()
)


# ============================================================
# FINAL GATE
# ============================================================

split_rows_pass = (
    audit_df[
        "split_audit_pass"
    ].all()
)


overall_pass = (
    duplicate_assignment_groups == 0
    and
    len(missing_dataset_groups) == 0
    and
    len(extra_assignment_groups) == 0
    and
    missing_split_rows == 0
    and
    duplicate_sample_keys == 0
    and
    zero_overlap
    and
    missing_year_cells == 0
    and
    total_accounting_ok
    and
    split_rows_pass
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


audit_df.to_csv(
    AUDIT_OUTPUT,
    index=False,
)


year_df.to_csv(
    YEAR_OUTPUT,
    index=False,
)


overlap_df.to_csv(
    OVERLAP_OUTPUT,
    index=False,
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "SPLIT BALANCE AUDIT"
)

print(
    "===================================="
)


print(
    audit_df.to_string(
        index=False
    )
)


print(
    "\n===================================="
)

print(
    "OVERLAP AUDIT"
)

print(
    "===================================="
)


print(
    overlap_df.to_string(
        index=False
    )
)


print(
    "\n===================================="
)

print(
    "YEAR-WISE COVERAGE"
)

print(
    "===================================="
)


print(
    year_df.to_string(
        index=False
    )
)


print(
    "\n===================================="
)

print(
    "ACCOUNTING"
)

print(
    "===================================="
)


print(
    "Dataset rows:",
    total_rows
)

print(
    "Assigned rows:",
    reconciled_rows
)

print(
    "Dataset positives:",
    total_positive
)

print(
    "Assigned positives:",
    reconciled_positive
)

print(
    "Missing year cells:",
    missing_year_cells
)

print(
    "Total accounting:",
    (
        "PASS"
        if total_accounting_ok
        else
        "FAIL"
    )
)


print(
    "\n===================================="
)

print(
    "CHAPTER 1.3 RESULT"
)

print(
    "===================================="
)


print(
    "ENTITY_GROUP leakage:",
    0 if zero_overlap else "FAIL"
)

print(
    "HARP leakage:",
    0 if zero_overlap else "FAIL"
)

print(
    "Overall split audit:",
    (
        "PASS"
        if overall_pass
        else
        "FAIL"
    )
)


print(
    "\nSaved:"
)

print(
    AUDIT_OUTPUT
)

print(
    YEAR_OUTPUT
)

print(
    OVERLAP_OUTPUT
)


if not overall_pass:

    raise RuntimeError(
        "Chapter 1.3 split audit failed."
    )


print(
    "\nMODEL CHAPTER 1.3 COMPLETE."
)