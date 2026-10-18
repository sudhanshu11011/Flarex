from pathlib import Path
import re

import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = Path(
    "data/final/dataset_v1"
)

MANIFEST_DIR = Path(
    "data_manifests"
)


C24_FILE = (
    DATASET_DIR
    /
    "cycle24_dataset_v1.parquet"
)

C25_FILE = (
    DATASET_DIR
    /
    "cycle25_dataset_v1.parquet"
)


SECONDARY_DETAIL_FILE = (
    MANIFEST_DIR
    /
    "pre_model_secondary_noaa_positive_events.csv"
)


SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "pre_model_single_noaa_candidate_summary.csv"
)


SECONDARY_CHECK_FILE = (
    MANIFEST_DIR
    /
    "pre_model_single_noaa_secondary_check.csv"
)


# ============================================================
# HELPERS
# ============================================================

def parse_primary_region(value):

    if pd.isna(value):
        return None

    try:

        region = int(
            float(value)
        )

    except Exception:

        return None

    if region <= 0:
        return None

    return region


def parse_region_list(value):

    if pd.isna(value):
        return []

    numbers = re.findall(
        r"\d+",
        str(value)
    )

    regions = []

    for number in numbers:

        region = int(
            number
        )

        if region > 0:

            regions.append(
                region
            )

    return sorted(
        set(
            regions
        )
    )


def classify_association(
    primary_region,
    region_list
):

    if (
        primary_region is None
        or
        len(region_list) == 0
    ):

        return "MISSING_ASSOCIATION"


    if len(region_list) == 1:

        if region_list[0] == primary_region:

            return "SINGLE_CONSISTENT"

        return "SINGLE_MISMATCH"


    if len(region_list) > 1:

        if primary_region in region_list:

            return "MULTI_REGION"

        return "MULTI_PRIMARY_NOT_IN_LIST"


    return "UNKNOWN"


# ============================================================
# LOAD SECONDARY-ONLY POSITIVE ROWS
# ============================================================

if not SECONDARY_DETAIL_FILE.exists():

    raise FileNotFoundError(
        f"Missing Fix C output: "
        f"{SECONDARY_DETAIL_FILE}"
    )


secondary_detail = pd.read_csv(
    SECONDARY_DETAIL_FILE
)


required_secondary_columns = {
    "cycle",
    "HARPNUM",
    "T_REC",
    "positive_source",
}


missing_secondary = (
    required_secondary_columns
    -
    set(
        secondary_detail.columns
    )
)


if missing_secondary:

    raise RuntimeError(
        "Secondary audit file missing columns: "
        f"{sorted(missing_secondary)}"
    )


secondary_only = (
    secondary_detail[
        secondary_detail[
            "positive_source"
        ]
        ==
        "SECONDARY_ONLY"
    ]
    [
        [
            "cycle",
            "HARPNUM",
            "T_REC",
        ]
    ]
    .drop_duplicates()
    .copy()
)


secondary_only[
    "HARPNUM"
] = pd.to_numeric(
    secondary_only[
        "HARPNUM"
    ],
    errors="coerce"
).astype(
    "Int64"
)


print(
    "===================================="
)

print(
    "PRE-MODEL FIX D — SINGLE-NOAA CANDIDATE AUDIT"
)

print(
    "===================================="
)


print(
    "Secondary-only positive rows from Fix C:",
    len(
        secondary_only
    )
)


# ============================================================
# AUDIT DATASETS
# ============================================================

datasets = {

    "cycle24":
        C24_FILE,

    "cycle25":
        C25_FILE,
}


summary_rows = []
secondary_check_rows = []


for cycle, path in datasets.items():

    if not path.exists():

        raise FileNotFoundError(
            f"Missing Dataset v1 file: "
            f"{path}"
        )


    df = pd.read_parquet(
        path
    )


    required_columns = {
        "HARPNUM",
        "T_REC",
        "NOAA_AR",
        "NOAA_ARS",
        "ENTITY_GROUP_ID",
        "target",
    }


    missing = (
        required_columns
        -
        set(
            df.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"{cycle} missing columns: "
            f"{sorted(missing)}"
        )


    work = df[
        [
            "HARPNUM",
            "T_REC",
            "NOAA_AR",
            "NOAA_ARS",
            "ENTITY_GROUP_ID",
            "target",
        ]
    ].copy()


    work[
        "cycle"
    ] = cycle


    work[
        "_primary_noaa"
    ] = work[
        "NOAA_AR"
    ].apply(
        parse_primary_region
    )


    work[
        "_noaa_list"
    ] = work[
        "NOAA_ARS"
    ].apply(
        parse_region_list
    )


    work[
        "noaa_region_count"
    ] = work[
        "_noaa_list"
    ].apply(
        len
    )


    work[
        "association_class"
    ] = [

        classify_association(
            primary,
            regions,
        )

        for primary, regions
        in zip(
            work[
                "_primary_noaa"
            ],
            work[
                "_noaa_list"
            ],
        )
    ]


    # ========================================================
    # BASIC CLASS COUNTS
    # ========================================================

    total_rows = len(
        work
    )


    single_consistent = work[
        work[
            "association_class"
        ]
        ==
        "SINGLE_CONSISTENT"
    ].copy()


    multi_region = work[
        work[
            "association_class"
        ]
        ==
        "MULTI_REGION"
    ].copy()


    single_mismatch = work[
        work[
            "association_class"
        ]
        ==
        "SINGLE_MISMATCH"
    ].copy()


    missing_association = work[
        work[
            "association_class"
        ]
        ==
        "MISSING_ASSOCIATION"
    ].copy()


    multi_primary_missing = work[
        work[
            "association_class"
        ]
        ==
        "MULTI_PRIMARY_NOT_IN_LIST"
    ].copy()


    # ========================================================
    # PRIMARY CANDIDATE COUNTS
    # ========================================================

    candidate_rows = len(
        single_consistent
    )


    candidate_positive = int(
        (
            single_consistent[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    candidate_negative = int(
        (
            single_consistent[
                "target"
            ]
            ==
            0
        )
        .sum()
    )


    candidate_rate = (
        candidate_positive
        /
        candidate_rows
        *
        100

        if candidate_rows
        else 0
    )


    excluded_rows = (
        total_rows
        -
        candidate_rows
    )


    excluded_pct = (
        excluded_rows
        /
        total_rows
        *
        100

        if total_rows
        else 0
    )


    # ========================================================
    # MULTI-REGION IMPACT
    # ========================================================

    multi_positive = int(
        (
            multi_region[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    multi_negative = int(
        (
            multi_region[
                "target"
            ]
            ==
            0
        )
        .sum()
    )


    # ========================================================
    # CHECK SECONDARY-ONLY POSITIVES
    # ========================================================

    cycle_secondary = secondary_only[
        secondary_only[
            "cycle"
        ]
        ==
        cycle
    ].copy()


    work[
        "HARPNUM"
    ] = pd.to_numeric(
        work[
            "HARPNUM"
        ],
        errors="coerce"
    ).astype(
        "Int64"
    )


    secondary_join = cycle_secondary.merge(
        work[
            [
                "HARPNUM",
                "T_REC",
                "association_class",
                "NOAA_AR",
                "NOAA_ARS",
                "ENTITY_GROUP_ID",
            ]
        ],
        on=[
            "HARPNUM",
            "T_REC",
        ],
        how="left",
        validate="one_to_one",
    )


    secondary_check_rows.append(
        secondary_join
    )


    secondary_in_single = int(
        (
            secondary_join[
                "association_class"
            ]
            ==
            "SINGLE_CONSISTENT"
        )
        .sum()
    )


    secondary_in_multi = int(
        (
            secondary_join[
                "association_class"
            ]
            ==
            "MULTI_REGION"
        )
        .sum()
    )


    secondary_unresolved = int(
        secondary_join[
            "association_class"
        ]
        .isna()
        .sum()
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows.append(
        {
            "cycle":
                cycle,

            "dataset_v1_rows":
                total_rows,

            "single_consistent_rows":
                candidate_rows,

            "single_consistent_pct":
                (
                    candidate_rows
                    /
                    total_rows
                    *
                    100
                    if total_rows
                    else 0
                ),

            "single_consistent_positive":
                candidate_positive,

            "single_consistent_negative":
                candidate_negative,

            "single_consistent_positive_rate_pct":
                candidate_rate,

            "single_consistent_harps":
                int(
                    single_consistent[
                        "HARPNUM"
                    ]
                    .nunique()
                ),

            "single_consistent_entity_groups":
                int(
                    single_consistent[
                        "ENTITY_GROUP_ID"
                    ]
                    .nunique()
                ),

            "multi_region_rows":
                len(
                    multi_region
                ),

            "multi_region_positive":
                multi_positive,

            "multi_region_negative":
                multi_negative,

            "multi_region_harps":
                int(
                    multi_region[
                        "HARPNUM"
                    ]
                    .nunique()
                ),

            "single_mismatch_rows":
                len(
                    single_mismatch
                ),

            "missing_association_rows":
                len(
                    missing_association
                ),

            "multi_primary_not_in_list_rows":
                len(
                    multi_primary_missing
                ),

            "candidate_excluded_rows":
                excluded_rows,

            "candidate_excluded_pct":
                excluded_pct,

            "secondary_only_positive_rows":
                len(
                    cycle_secondary
                ),

            "secondary_only_in_single_candidate":
                secondary_in_single,

            "secondary_only_in_multi_region":
                secondary_in_multi,

            "secondary_only_unresolved":
                secondary_unresolved,
        }
    )


    print(
        "\n===================================="
    )

    print(
        cycle.upper()
    )

    print(
        "===================================="
    )


    print(
        "Dataset v1 rows:",
        total_rows
    )


    print(
        "Single-consistent candidate rows:",
        candidate_rows
    )


    print(
        "Candidate retained:",
        f"{candidate_rows / total_rows * 100:.4f}%"
    )


    print(
        "Candidate positives:",
        candidate_positive
    )


    print(
        "Candidate negatives:",
        candidate_negative
    )


    print(
        "Candidate positive rate:",
        f"{candidate_rate:.4f}%"
    )


    print(
        "Multi-region rows:",
        len(
            multi_region
        )
    )


    print(
        "Multi-region positives:",
        multi_positive
    )


    print(
        "Multi-region negatives:",
        multi_negative
    )


    print(
        "Single mismatch rows:",
        len(
            single_mismatch
        )
    )


    print(
        "Missing association rows:",
        len(
            missing_association
        )
    )


    print(
        "Primary-not-in-list multi rows:",
        len(
            multi_primary_missing
        )
    )


    print(
        "Secondary-only positives:",
        len(
            cycle_secondary
        )
    )


    print(
        "Secondary-only retained by candidate:",
        secondary_in_single
    )


    print(
        "Secondary-only classified multi-region:",
        secondary_in_multi
    )


# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


secondary_check = pd.concat(
    secondary_check_rows,
    ignore_index=True
)


MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


secondary_check.to_csv(
    SECONDARY_CHECK_FILE,
    index=False
)


# ============================================================
# FINAL DECISION VIEW
# ============================================================

print(
    "\n===================================="
)

print(
    "SINGLE-NOAA PRIMARY CANDIDATE SUMMARY"
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
    "SECONDARY-POSITIVE SAFETY CHECK"
)

print(
    "===================================="
)


total_secondary = int(
    summary[
        "secondary_only_positive_rows"
    ].sum()
)


retained_secondary = int(
    summary[
        "secondary_only_in_single_candidate"
    ].sum()
)


multi_secondary = int(
    summary[
        "secondary_only_in_multi_region"
    ].sum()
)


unresolved_secondary = int(
    summary[
        "secondary_only_unresolved"
    ].sum()
)


print(
    "Total secondary-only positives:",
    total_secondary
)


print(
    "Retained by single-NOAA candidate:",
    retained_secondary
)


print(
    "Classified as multi-region:",
    multi_secondary
)


print(
    "Unresolved:",
    unresolved_secondary
)


if (
    retained_secondary == 0
    and
    unresolved_secondary == 0
):

    print(
        "Safety check: PASS"
    )

else:

    print(
        "Safety check: FAIL"
    )


print(
    "\nSaved:"
)

print(
    SUMMARY_FILE
)

print(
    SECONDARY_CHECK_FILE
)


print(
    "\nPre-model Fix D audit complete."
)