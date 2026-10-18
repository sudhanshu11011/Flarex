from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "data/final/dataset_v2/cycle24_dataset_v2.parquet"
)

GROUP_SUMMARY_FILE = Path(
    "data_manifests/model_ch1_cycle24_group_summary.csv"
)

FEASIBILITY_FILE = Path(
    "data_manifests/model_ch1_split_feasibility_summary.csv"
)

PRE_MODEL_GATE = Path(
    "data_manifests/pre_model_gate.json"
)

MANIFEST_DIR = Path(
    "data_manifests"
)


ASSIGNMENT_OUTPUT = (
    MANIFEST_DIR
    /
    "model_ch1_split_assignments.csv"
)

SEARCH_OUTPUT = (
    MANIFEST_DIR
    /
    "model_ch1_split_search_summary.csv"
)

SUMMARY_OUTPUT = (
    MANIFEST_DIR
    /
    "model_ch1_split_assignment_summary.csv"
)


# ============================================================
# FROZEN SPLIT DESIGN
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
HOLDOUT_RATIO = 0.15

SPLIT_NAMES = [
    "TRAIN",
    "VALIDATION",
    "HOLDOUT",
]

SPLIT_RATIOS = np.array(
    [
        TRAIN_RATIO,
        VALIDATION_RATIO,
        HOLDOUT_RATIO,
    ],
    dtype=float,
)


# Deterministic candidate search.
#
# We do NOT search using:
# - features
# - model performance
# - Cycle25
#
# Only Cycle24 group/label/year balance.

BASE_SEED = 24025
N_CANDIDATES = 10000


# Conservative minimum established during 1.1.
MIN_POSITIVE_GROUPS_PER_SPLIT = 20


# ============================================================
# HELPERS
# ============================================================

def relative_error(actual, expected):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    expected = np.asarray(
        expected,
        dtype=float,
    )


    valid = expected > 0


    if not valid.any():

        return 0.0


    return float(
        np.mean(
            np.abs(
                actual[valid]
                -
                expected[valid]
            )
            /
            expected[valid]
        )
    )


# ============================================================
# VERIFY INPUT STATE
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 1.2 — GROUP-SAFE SPLIT ASSIGNMENT"
)

print(
    "===================================="
)


required_files = [
    DATA_FILE,
    GROUP_SUMMARY_FILE,
    FEASIBILITY_FILE,
    PRE_MODEL_GATE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


with PRE_MODEL_GATE.open(
    "r",
    encoding="utf-8",
) as file:

    gate = json.load(
        file
    )


if gate.get(
    "status"
) != "OPEN_FOR_MODELING":

    raise RuntimeError(
        "Pre-model gate is not open."
    )


feasibility = pd.read_csv(
    FEASIBILITY_FILE
)


if len(
    feasibility
) != 1:

    raise RuntimeError(
        "Unexpected feasibility manifest."
    )


if not bool(
    feasibility.iloc[0][
        "split_feasible"
    ]
):

    raise RuntimeError(
        "Chapter 1.1 split feasibility did not PASS."
    )


print(
    "Pre-model gate: OPEN"
)

print(
    "Chapter 1.1 feasibility: PASS"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(
    DATA_FILE
)


df[
    "_year"
] = pd.to_datetime(
    df[
        "observation_time_utc"
    ],
    utc=True,
    errors="raise",
).dt.year


total_rows = len(
    df
)

total_positive = int(
    df[
        "target"
    ].sum()
)

total_groups = int(
    df[
        "ENTITY_GROUP_ID"
    ].nunique()
)


# ============================================================
# BUILD EXACT GROUP TABLE
# ============================================================

basic_groups = (
    df
    .groupby(
        "ENTITY_GROUP_ID",
        sort=True,
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

        first_year=(
            "_year",
            "min",
        ),

        last_year=(
            "_year",
            "max",
        ),

        unique_harpnums=(
            "HARPNUM",
            "nunique",
        ),
    )
    .reset_index()
)


basic_groups[
    "negative"
] = (
    basic_groups[
        "rows"
    ]
    -
    basic_groups[
        "positive"
    ]
)


basic_groups[
    "has_positive"
] = (
    basic_groups[
        "positive"
    ]
    >
    0
)


if len(
    basic_groups
) != total_groups:

    raise RuntimeError(
        "Group-count reconciliation failed."
    )


# ============================================================
# EXACT YEAR CONTRIBUTION PER GROUP
# ============================================================

group_year_rows = (
    df
    .groupby(
        [
            "ENTITY_GROUP_ID",
            "_year",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


group_year_positive = (
    df
    .groupby(
        [
            "ENTITY_GROUP_ID",
            "_year",
        ]
    )[
        "target"
    ]
    .sum()
    .unstack(
        fill_value=0
    )
)


years = sorted(
    df[
        "_year"
    ]
    .unique()
    .tolist()
)


group_year_rows = (
    group_year_rows
    .reindex(
        index=basic_groups[
            "ENTITY_GROUP_ID"
        ],
        columns=years,
        fill_value=0,
    )
)


group_year_positive = (
    group_year_positive
    .reindex(
        index=basic_groups[
            "ENTITY_GROUP_ID"
        ],
        columns=years,
        fill_value=0,
    )
)


# ============================================================
# NUMPY ARRAYS
# ============================================================

group_ids = basic_groups[
    "ENTITY_GROUP_ID"
].astype(str).to_numpy()


group_rows = basic_groups[
    "rows"
].to_numpy(
    dtype=int
)


group_positive = basic_groups[
    "positive"
].to_numpy(
    dtype=int
)


group_has_positive = (
    basic_groups[
        "has_positive"
    ]
    .astype(int)
    .to_numpy()
)


year_rows_matrix = (
    group_year_rows
    .to_numpy(
        dtype=int
    )
)


year_positive_matrix = (
    group_year_positive
    .to_numpy(
        dtype=int
    )
)


year_total_rows = (
    year_rows_matrix
    .sum(
        axis=0
    )
)


year_total_positive = (
    year_positive_matrix
    .sum(
        axis=0
    )
)


total_positive_groups = int(
    group_has_positive.sum()
)


# ============================================================
# EXACT GROUP COUNTS
# ============================================================

train_group_count = round(
    total_groups
    *
    TRAIN_RATIO
)


validation_group_count = round(
    total_groups
    *
    VALIDATION_RATIO
)


holdout_group_count = (
    total_groups
    -
    train_group_count
    -
    validation_group_count
)


target_group_counts = np.array(
    [
        train_group_count,
        validation_group_count,
        holdout_group_count,
    ],
    dtype=int,
)


print(
    "\nDataset rows:",
    total_rows
)

print(
    "Positive rows:",
    total_positive
)

print(
    "Entity groups:",
    total_groups
)

print(
    "Positive-containing groups:",
    total_positive_groups
)


print(
    "\nExact target group counts:"
)

print(
    "TRAIN:",
    train_group_count
)

print(
    "VALIDATION:",
    validation_group_count
)

print(
    "HOLDOUT:",
    holdout_group_count
)


# ============================================================
# TARGETS
# ============================================================

target_rows = (
    total_rows
    *
    SPLIT_RATIOS
)


target_positive = (
    total_positive
    *
    SPLIT_RATIOS
)


target_positive_groups = (
    total_positive_groups
    *
    SPLIT_RATIOS
)


target_year_rows = (
    SPLIT_RATIOS[
        :,
        None
    ]
    *
    year_total_rows[
        None,
        :
    ]
)


target_year_positive = (
    SPLIT_RATIOS[
        :,
        None
    ]
    *
    year_total_positive[
        None,
        :
    ]
)


# ============================================================
# CANDIDATE EVALUATION
# ============================================================

def evaluate_candidate(
    permutation
):

    train_idx = permutation[
        :
        train_group_count
    ]


    validation_idx = permutation[
        train_group_count
        :
        train_group_count
        +
        validation_group_count
    ]


    holdout_idx = permutation[
        train_group_count
        +
        validation_group_count
        :
    ]


    split_indices = [
        train_idx,
        validation_idx,
        holdout_idx,
    ]


    actual_rows = np.array(
        [
            group_rows[
                indices
            ].sum()

            for indices
            in split_indices
        ],
        dtype=float,
    )


    actual_positive = np.array(
        [
            group_positive[
                indices
            ].sum()

            for indices
            in split_indices
        ],
        dtype=float,
    )


    actual_positive_groups = np.array(
        [
            group_has_positive[
                indices
            ].sum()

            for indices
            in split_indices
        ],
        dtype=float,
    )


    actual_year_rows = np.vstack(
        [
            year_rows_matrix[
                indices
            ].sum(
                axis=0
            )

            for indices
            in split_indices
        ]
    )


    actual_year_positive = np.vstack(
        [
            year_positive_matrix[
                indices
            ].sum(
                axis=0
            )

            for indices
            in split_indices
        ]
    )


    # --------------------------------------------------------
    # HARD SAFETY CONSTRAINTS
    # --------------------------------------------------------

    if (
        actual_positive_groups.min()
        <
        MIN_POSITIVE_GROUPS_PER_SPLIT
    ):

        return None


    if (
        actual_positive.min()
        <=
        0
    ):

        return None


    # Every split must contain observations from every year.
    if (
        actual_year_rows
        <=
        0
    ).any():

        return None


    # --------------------------------------------------------
    # BALANCE OBJECTIVE
    #
    # This uses only:
    # rows
    # labels
    # positive-group counts
    # year coverage
    #
    # No model or feature information.
    # --------------------------------------------------------

    row_error = relative_error(
        actual_rows,
        target_rows,
    )


    positive_error = relative_error(
        actual_positive,
        target_positive,
    )


    positive_group_error = relative_error(
        actual_positive_groups,
        target_positive_groups,
    )


    year_row_error = relative_error(
        actual_year_rows,
        target_year_rows,
    )


    year_positive_error = relative_error(
        actual_year_positive,
        target_year_positive,
    )


    score = (
        row_error
        +
        positive_error
        +
        positive_group_error
        +
        0.5
        *
        year_row_error
        +
        0.5
        *
        year_positive_error
    )


    return {

        "score":
            float(
                score
            ),

        "row_error":
            float(
                row_error
            ),

        "positive_error":
            float(
                positive_error
            ),

        "positive_group_error":
            float(
                positive_group_error
            ),

        "year_row_error":
            float(
                year_row_error
            ),

        "year_positive_error":
            float(
                year_positive_error
            ),

        "actual_rows":
            actual_rows.astype(
                int
            ),

        "actual_positive":
            actual_positive.astype(
                int
            ),

        "actual_positive_groups":
            actual_positive_groups.astype(
                int
            ),

        "split_indices":
            split_indices,
    }


# ============================================================
# DETERMINISTIC SEARCH
# ============================================================

print(
    "\n===================================="
)

print(
    "DETERMINISTIC CANDIDATE SEARCH"
)

print(
    "===================================="
)


best = None

search_rows = []


for candidate_number in range(
    N_CANDIDATES
):

    seed = (
        BASE_SEED
        +
        candidate_number
    )


    rng = np.random.default_rng(
        seed
    )


    permutation = rng.permutation(
        total_groups
    )


    result = evaluate_candidate(
        permutation
    )


    if result is None:
        continue


    search_rows.append(
        {
            "seed":
                seed,

            "score":
                result[
                    "score"
                ],

            "row_error":
                result[
                    "row_error"
                ],

            "positive_error":
                result[
                    "positive_error"
                ],

            "positive_group_error":
                result[
                    "positive_group_error"
                ],

            "year_row_error":
                result[
                    "year_row_error"
                ],

            "year_positive_error":
                result[
                    "year_positive_error"
                ],

            "train_rows":
                result[
                    "actual_rows"
                ][0],

            "validation_rows":
                result[
                    "actual_rows"
                ][1],

            "holdout_rows":
                result[
                    "actual_rows"
                ][2],

            "train_positive":
                result[
                    "actual_positive"
                ][0],

            "validation_positive":
                result[
                    "actual_positive"
                ][1],

            "holdout_positive":
                result[
                    "actual_positive"
                ][2],

            "train_positive_groups":
                result[
                    "actual_positive_groups"
                ][0],

            "validation_positive_groups":
                result[
                    "actual_positive_groups"
                ][1],

            "holdout_positive_groups":
                result[
                    "actual_positive_groups"
                ][2],
        }
    )


    if (
        best is None
        or
        result[
            "score"
        ]
        <
        best[
            "result"
        ][
            "score"
        ]
    ):

        best = {
            "seed":
                seed,

            "permutation":
                permutation.copy(),

            "result":
                result,
        }


if best is None:

    raise RuntimeError(
        "No valid group-safe candidate split found."
    )


search_df = pd.DataFrame(
    search_rows
)


search_df = search_df.sort_values(
    [
        "score",
        "seed",
    ]
).reset_index(
    drop=True
)


# Save only strongest candidates for auditability.
search_df.head(
    100
).to_csv(
    SEARCH_OUTPUT,
    index=False,
)


# ============================================================
# BUILD FINAL ASSIGNMENT
# ============================================================

best_seed = best[
    "seed"
]

best_result = best[
    "result"
]


assignment_map = {}


for split_name, indices in zip(
    SPLIT_NAMES,
    best_result[
        "split_indices"
    ],
):

    for index in indices:

        assignment_map[
            group_ids[
                index
            ]
        ] = split_name


basic_groups[
    "split"
] = (
    basic_groups[
        "ENTITY_GROUP_ID"
    ]
    .astype(str)
    .map(
        assignment_map
    )
)


if basic_groups[
    "split"
].isna().any():

    raise RuntimeError(
        "Some entity groups were not assigned."
    )


if basic_groups[
    "ENTITY_GROUP_ID"
].duplicated().any():

    raise RuntimeError(
        "Duplicate entity-group assignment found."
    )


# ============================================================
# FINAL ASSIGNMENT SUMMARY
# ============================================================

summary_rows = []


for split_name, ratio in zip(
    SPLIT_NAMES,
    SPLIT_RATIOS,
):

    group_subset = basic_groups[
        basic_groups[
            "split"
        ]
        ==
        split_name
    ]


    ids = set(
        group_subset[
            "ENTITY_GROUP_ID"
        ]
    )


    row_subset = df[
        df[
            "ENTITY_GROUP_ID"
        ]
        .isin(
            ids
        )
    ]


    rows = len(
        row_subset
    )


    positive = int(
        row_subset[
            "target"
        ].sum()
    )


    negative = (
        rows
        -
        positive
    )


    positive_groups = int(
        (
            group_subset[
                "positive"
            ]
            >
            0
        )
        .sum()
    )


    summary_rows.append(
        {
            "split":
                split_name,

            "target_ratio":
                ratio,

            "entity_groups":
                len(
                    group_subset
                ),

            "positive_groups":
                positive_groups,

            "rows":
                rows,

            "row_fraction":
                rows
                /
                total_rows,

            "positive":
                positive,

            "negative":
                negative,

            "positive_rate_pct":
                positive
                /
                rows
                *
                100,

            "positive_fraction":
                positive
                /
                total_positive,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# GROUP OVERLAP CHECK
# ============================================================

split_group_sets = {}


for split_name in SPLIT_NAMES:

    split_group_sets[
        split_name
    ] = set(
        basic_groups.loc[
            basic_groups[
                "split"
            ]
            ==
            split_name,
            "ENTITY_GROUP_ID",
        ]
    )


train_validation_overlap = (
    split_group_sets[
        "TRAIN"
    ]
    &
    split_group_sets[
        "VALIDATION"
    ]
)


train_holdout_overlap = (
    split_group_sets[
        "TRAIN"
    ]
    &
    split_group_sets[
        "HOLDOUT"
    ]
)


validation_holdout_overlap = (
    split_group_sets[
        "VALIDATION"
    ]
    &
    split_group_sets[
        "HOLDOUT"
    ]
)


if (
    train_validation_overlap
    or
    train_holdout_overlap
    or
    validation_holdout_overlap
):

    raise RuntimeError(
        "ENTITY_GROUP_ID leakage detected."
    )


# ============================================================
# SAVE ASSIGNMENT
# ============================================================

assignment_columns = [
    "ENTITY_GROUP_ID",
    "split",
    "rows",
    "positive",
    "negative",
    "has_positive",
    "unique_harpnums",
    "first_year",
    "last_year",
]


basic_groups[
    assignment_columns
].sort_values(
    [
        "split",
        "ENTITY_GROUP_ID",
    ]
).to_csv(
    ASSIGNMENT_OUTPUT,
    index=False,
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print(
    "Candidates evaluated:",
    N_CANDIDATES
)

print(
    "Valid candidates:",
    len(
        search_df
    )
)

print(
    "Selected seed:",
    best_seed
)

print(
    "Selected score:",
    f"{best_result['score']:.8f}"
)


print(
    "\n===================================="
)

print(
    "SELECTED SPLIT SUMMARY"
)

print(
    "===================================="
)


print(
    summary.to_string(
        index=False
    )
)


print(
    "\n===================================="
)

print(
    "LEAKAGE CHECK"
)

print(
    "===================================="
)


print(
    "TRAIN ∩ VALIDATION:",
    len(
        train_validation_overlap
    )
)

print(
    "TRAIN ∩ HOLDOUT:",
    len(
        train_holdout_overlap
    )
)

print(
    "VALIDATION ∩ HOLDOUT:",
    len(
        validation_holdout_overlap
    )
)


print(
    "\n===================================="
)

print(
    "REPRODUCIBILITY"
)

print(
    "===================================="
)


print(
    "Base seed:",
    BASE_SEED
)

print(
    "Candidate count:",
    N_CANDIDATES
)

print(
    "Selected seed:",
    best_seed
)

print(
    "Cycle25 used:",
    False
)

print(
    "Model performance used:",
    False
)

print(
    "Feature values used:",
    False
)


print(
    "\nSaved:"
)

print(
    ASSIGNMENT_OUTPUT
)

print(
    SEARCH_OUTPUT
)

print(
    SUMMARY_OUTPUT
)


print(
    "\nMODEL CHAPTER 1.2 COMPLETE."
)