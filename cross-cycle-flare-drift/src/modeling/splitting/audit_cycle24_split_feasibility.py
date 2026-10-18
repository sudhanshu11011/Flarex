from pathlib import Path
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "data/final/dataset_v2/cycle24_dataset_v2.parquet"
)

PRE_MODEL_GATE = Path(
    "data_manifests/pre_model_gate.json"
)

DATASET_CONTRACT = Path(
    "data_manifests/dataset_v2_contract.json"
)

MANIFEST_DIR = Path(
    "data_manifests"
)


GROUP_OUTPUT = (
    MANIFEST_DIR
    /
    "model_ch1_cycle24_group_summary.csv"
)

SUMMARY_OUTPUT = (
    MANIFEST_DIR
    /
    "model_ch1_split_feasibility_summary.csv"
)


# ============================================================
# TARGET SPLIT
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
HOLDOUT_RATIO = 0.15


# ============================================================
# VERIFY PRE-MODEL GATE
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 1.1 — SPLIT FEASIBILITY AUDIT"
)

print(
    "===================================="
)


for path in [
    DATA_FILE,
    PRE_MODEL_GATE,
    DATASET_CONTRACT,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


with PRE_MODEL_GATE.open(
    "r",
    encoding="utf-8",
) as file:

    gate = json.load(
        file
    )


with DATASET_CONTRACT.open(
    "r",
    encoding="utf-8",
) as file:

    contract = json.load(
        file
    )


if gate.get(
    "status"
) != "OPEN_FOR_MODELING":

    raise RuntimeError(
        "Pre-model gate is not open."
    )


if contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Dataset v2 is not frozen."
    )


print(
    "Pre-model gate: OPEN"
)

print(
    "Dataset v2: FROZEN"
)


# ============================================================
# LOAD CYCLE24
# ============================================================

df = pd.read_parquet(
    DATA_FILE
)


required_columns = {
    "HARPNUM",
    "T_REC",
    "observation_time_utc",
    "ENTITY_GROUP_ID",
    "target",
}


missing = (
    required_columns
    -
    set(df.columns)
)


if missing:

    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


if not df[
    "target"
].isin(
    [0, 1]
).all():

    raise RuntimeError(
        "Invalid target values found."
    )


if df[
    "ENTITY_GROUP_ID"
].isna().any():

    raise RuntimeError(
        "Missing ENTITY_GROUP_ID found."
    )


df[
    "_observation_time"
] = pd.to_datetime(
    df[
        "observation_time_utc"
    ],
    utc=True,
    errors="coerce",
)


if df[
    "_observation_time"
].isna().any():

    raise RuntimeError(
        "Invalid observation timestamps found."
    )


# ============================================================
# BASIC DATASET STATE
# ============================================================

total_rows = len(
    df
)

total_positive = int(
    (
        df[
            "target"
        ]
        ==
        1
    ).sum()
)

total_negative = int(
    (
        df[
            "target"
        ]
        ==
        0
    ).sum()
)

total_groups = int(
    df[
        "ENTITY_GROUP_ID"
    ].nunique()
)


print(
    "\n===================================="
)

print(
    "CYCLE24 DATASET"
)

print(
    "===================================="
)


print(
    "Rows:",
    total_rows
)

print(
    "Positive:",
    total_positive
)

print(
    "Negative:",
    total_negative
)

print(
    "Positive rate:",
    f"{total_positive / total_rows * 100:.4f}%"
)

print(
    "ENTITY_GROUP_ID count:",
    total_groups
)


# ============================================================
# GROUP-LEVEL SUMMARY
# ============================================================

group_rows = []


for group_id, group in df.groupby(
    "ENTITY_GROUP_ID",
    sort=True,
):

    rows = len(
        group
    )

    positive = int(
        (
            group[
                "target"
            ]
            ==
            1
        ).sum()
    )

    negative = int(
        (
            group[
                "target"
            ]
            ==
            0
        ).sum()
    )


    first_time = group[
        "_observation_time"
    ].min()

    last_time = group[
        "_observation_time"
    ].max()


    active_days = (
        last_time
        -
        first_time
    ).total_seconds() / 86400.0


    years = sorted(
        group[
            "_observation_time"
        ]
        .dt.year
        .unique()
        .tolist()
    )


    group_rows.append(
        {
            "ENTITY_GROUP_ID":
                group_id,

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
                ),

            "has_positive":
                positive > 0,

            "unique_harpnums":
                int(
                    group[
                        "HARPNUM"
                    ].nunique()
                ),

            "first_observation_utc":
                first_time,

            "last_observation_utc":
                last_time,

            "span_days":
                active_days,

            "first_year":
                min(
                    years
                ),

            "last_year":
                max(
                    years
                ),

            "year_count":
                len(
                    years
                ),
        }
    )


groups = pd.DataFrame(
    group_rows
)


# ============================================================
# GROUP DISTRIBUTION
# ============================================================

positive_groups = groups[
    groups[
        "has_positive"
    ]
].copy()


negative_only_groups = groups[
    ~groups[
        "has_positive"
    ]
].copy()


positive_group_count = len(
    positive_groups
)

negative_only_group_count = len(
    negative_only_groups
)


print(
    "\n===================================="
)

print(
    "ENTITY-GROUP LABEL DISTRIBUTION"
)

print(
    "===================================="
)


print(
    "Positive-containing groups:",
    positive_group_count
)

print(
    "Negative-only groups:",
    negative_only_group_count
)

print(
    "Positive-group fraction:",
    f"{positive_group_count / total_groups * 100:.4f}%"
)


# ============================================================
# GROUP SIZE DISTRIBUTION
# ============================================================

print(
    "\n===================================="
)

print(
    "GROUP SIZE DISTRIBUTION"
)

print(
    "===================================="
)


print(
    groups[
        "rows"
    ].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    ).to_string()
)


print(
    "\nPositive rows per positive group:"
)


print(
    positive_groups[
        "positive"
    ].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    ).to_string()
)


# ============================================================
# POSITIVE CONCENTRATION
# ============================================================

positive_ranked = positive_groups.sort_values(
    "positive",
    ascending=False,
).reset_index(
    drop=True
)


positive_ranked[
    "cumulative_positive"
] = positive_ranked[
    "positive"
].cumsum()


positive_ranked[
    "cumulative_positive_pct"
] = (
    positive_ranked[
        "cumulative_positive"
    ]
    /
    total_positive
    *
    100
)


def groups_needed_for_fraction(
    fraction
):

    threshold = (
        total_positive
        *
        fraction
    )


    hit = positive_ranked[
        positive_ranked[
            "cumulative_positive"
        ]
        >=
        threshold
    ]


    if len(
        hit
    ) == 0:

        return None


    return int(
        hit.index[
            0
        ]
        +
        1
    )


groups_for_50pct_positive = (
    groups_needed_for_fraction(
        0.50
    )
)

groups_for_80pct_positive = (
    groups_needed_for_fraction(
        0.80
    )
)

groups_for_90pct_positive = (
    groups_needed_for_fraction(
        0.90
    )
)


print(
    "\n===================================="
)

print(
    "POSITIVE CONCENTRATION"
)

print(
    "===================================="
)


print(
    "Groups containing 50% of positives:",
    groups_for_50pct_positive
)

print(
    "Groups containing 80% of positives:",
    groups_for_80pct_positive
)

print(
    "Groups containing 90% of positives:",
    groups_for_90pct_positive
)


# ============================================================
# YEAR-WISE GROUP / CLASS COVERAGE
# ============================================================

year_summary = (
    df.assign(
        year=df[
            "_observation_time"
        ].dt.year
    )
    .groupby(
        "year"
    )
    .agg(
        rows=(
            "target",
            "size",
        ),

        positive=(
            "target",
            "sum",
        ),

        entity_groups=(
            "ENTITY_GROUP_ID",
            "nunique",
        ),
    )
    .reset_index()
)


year_summary[
    "negative"
] = (
    year_summary[
        "rows"
    ]
    -
    year_summary[
        "positive"
    ]
)


year_summary[
    "positive_rate_pct"
] = (
    year_summary[
        "positive"
    ]
    /
    year_summary[
        "rows"
    ]
    *
    100
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
    year_summary.to_string(
        index=False
    )
)


# ============================================================
# TARGET SPLIT ESTIMATES
# ============================================================

target_train_rows = round(
    total_rows
    *
    TRAIN_RATIO
)

target_validation_rows = round(
    total_rows
    *
    VALIDATION_RATIO
)

target_holdout_rows = (
    total_rows
    -
    target_train_rows
    -
    target_validation_rows
)


target_train_positive = round(
    total_positive
    *
    TRAIN_RATIO
)

target_validation_positive = round(
    total_positive
    *
    VALIDATION_RATIO
)

target_holdout_positive = (
    total_positive
    -
    target_train_positive
    -
    target_validation_positive
)


target_train_groups = round(
    total_groups
    *
    TRAIN_RATIO
)

target_validation_groups = round(
    total_groups
    *
    VALIDATION_RATIO
)

target_holdout_groups = (
    total_groups
    -
    target_train_groups
    -
    target_validation_groups
)


# ============================================================
# FEASIBILITY GATE
# ============================================================

# Conservative minimum:
# each subset should have many positive-containing groups.
#
# This does NOT create the split.
# It only checks whether group-safe stratification is plausible.

minimum_positive_groups_per_subset = 20


positive_group_feasible = (
    positive_group_count
    >=
    minimum_positive_groups_per_subset
    *
    3
)


split_feasible = (
    total_groups
    >=
    100
    and
    positive_group_feasible
    and
    total_positive
    >=
    300
)


summary = pd.DataFrame(
    [
        {
            "dataset":
                "cycle24_dataset_v2",

            "rows":
                total_rows,

            "positive":
                total_positive,

            "negative":
                total_negative,

            "positive_rate_pct":
                (
                    total_positive
                    /
                    total_rows
                    *
                    100
                ),

            "entity_groups":
                total_groups,

            "positive_groups":
                positive_group_count,

            "negative_only_groups":
                negative_only_group_count,

            "groups_for_50pct_positive":
                groups_for_50pct_positive,

            "groups_for_80pct_positive":
                groups_for_80pct_positive,

            "groups_for_90pct_positive":
                groups_for_90pct_positive,

            "target_train_ratio":
                TRAIN_RATIO,

            "target_validation_ratio":
                VALIDATION_RATIO,

            "target_holdout_ratio":
                HOLDOUT_RATIO,

            "target_train_rows":
                target_train_rows,

            "target_validation_rows":
                target_validation_rows,

            "target_holdout_rows":
                target_holdout_rows,

            "target_train_positive":
                target_train_positive,

            "target_validation_positive":
                target_validation_positive,

            "target_holdout_positive":
                target_holdout_positive,

            "target_train_groups":
                target_train_groups,

            "target_validation_groups":
                target_validation_groups,

            "target_holdout_groups":
                target_holdout_groups,

            "minimum_positive_groups_per_subset":
                minimum_positive_groups_per_subset,

            "positive_group_feasible":
                positive_group_feasible,

            "split_feasible":
                split_feasible,
        }
    ]
)


# ============================================================
# SAVE
# ============================================================

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


groups.drop(
    columns=[],
).to_csv(
    GROUP_OUTPUT,
    index=False,
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n===================================="
)

print(
    "TARGET 70 / 15 / 15 SPLIT"
)

print(
    "===================================="
)


print(
    "Target TRAIN:"
)

print(
    " Rows:",
    target_train_rows
)

print(
    " Positive:",
    target_train_positive
)

print(
    " Groups:",
    target_train_groups
)


print(
    "\nTarget VALIDATION:"
)

print(
    " Rows:",
    target_validation_rows
)

print(
    " Positive:",
    target_validation_positive
)

print(
    " Groups:",
    target_validation_groups
)


print(
    "\nTarget HOLDOUT:"
)

print(
    " Rows:",
    target_holdout_rows
)

print(
    " Positive:",
    target_holdout_positive
)

print(
    " Groups:",
    target_holdout_groups
)


print(
    "\n===================================="
)

print(
    "SPLIT FEASIBILITY"
)

print(
    "===================================="
)


print(
    "Positive-group feasibility:",
    (
        "PASS"
        if positive_group_feasible
        else
        "FAIL"
    )
)


print(
    "Overall split feasibility:",
    (
        "PASS"
        if split_feasible
        else
        "FAIL"
    )
)


print(
    "\nSaved:"
)

print(
    GROUP_OUTPUT
)

print(
    SUMMARY_OUTPUT
)


if not split_feasible:

    raise RuntimeError(
        "70/15/15 group-safe split feasibility failed."
    )


print(
    "\nMODEL CHAPTER 1.1 COMPLETE."
)