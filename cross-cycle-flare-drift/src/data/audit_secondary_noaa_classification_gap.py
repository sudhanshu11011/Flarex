from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data_manifests/pre_model_single_noaa_secondary_check.csv"
)

OUTPUT_FILE = Path(
    "data_manifests/pre_model_secondary_noaa_classification_gap.csv"
)


print(
    "===================================="
)

print(
    "PRE-MODEL FIX E — SECONDARY CLASSIFICATION GAP"
)

print(
    "===================================="
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Missing: {INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


required = {
    "cycle",
    "HARPNUM",
    "T_REC",
    "association_class",
    "NOAA_AR",
    "NOAA_ARS",
    "ENTITY_GROUP_ID",
}


missing = (
    required
    -
    set(
        df.columns
    )
)


if missing:

    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


print(
    "Total secondary-only rows:",
    len(df)
)


# ============================================================
# CLASSIFICATION COUNTS
# ============================================================

counts = (
    df
    .groupby(
        [
            "cycle",
            "association_class",
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="rows"
    )
)


print(
    "\n===================================="
)

print(
    "ASSOCIATION CLASS COUNTS"
)

print(
    "===================================="
)


print(
    counts.to_string(
        index=False
    )
)


# ============================================================
# GLOBAL ACCOUNTING
# ============================================================

expected_total = len(
    df
)


accounted_total = int(
    counts[
        "rows"
    ].sum()
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
    "Expected secondary-only rows:",
    expected_total
)


print(
    "Accounted rows:",
    accounted_total
)


print(
    "Difference:",
    expected_total
    -
    accounted_total
)


# ============================================================
# NON-STANDARD CASES
# ============================================================

standard_classes = {
    "SINGLE_CONSISTENT",
    "MULTI_REGION",
}


special = df[
    ~df[
        "association_class"
    ].isin(
        standard_classes
    )
].copy()


print(
    "\n===================================="
)

print(
    "NON-STANDARD SECONDARY-ONLY CASES"
)

print(
    "===================================="
)


print(
    "Rows:",
    len(
        special
    )
)


if len(
    special
) > 0:

    print(
        special[
            [
                "cycle",
                "HARPNUM",
                "T_REC",
                "association_class",
                "NOAA_AR",
                "NOAA_ARS",
                "ENTITY_GROUP_ID",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# SUMMARY BY CYCLE
# ============================================================

summary_rows = []


for cycle in sorted(
    df[
        "cycle"
    ].dropna().unique()
):

    subset = df[
        df[
            "cycle"
        ]
        ==
        cycle
    ]


    class_counts = (
        subset[
            "association_class"
        ]
        .value_counts(
            dropna=False
        )
    )


    summary_rows.append(
        {
            "cycle":
                cycle,

            "secondary_only_rows":
                len(
                    subset
                ),

            "single_consistent":
                int(
                    class_counts.get(
                        "SINGLE_CONSISTENT",
                        0
                    )
                ),

            "multi_region":
                int(
                    class_counts.get(
                        "MULTI_REGION",
                        0
                    )
                ),

            "single_mismatch":
                int(
                    class_counts.get(
                        "SINGLE_MISMATCH",
                        0
                    )
                ),

            "missing_association":
                int(
                    class_counts.get(
                        "MISSING_ASSOCIATION",
                        0
                    )
                ),

            "multi_primary_not_in_list":
                int(
                    class_counts.get(
                        "MULTI_PRIMARY_NOT_IN_LIST",
                        0
                    )
                ),

            "other_or_missing_class":
                int(
                    len(
                        subset
                    )
                    -
                    class_counts.get(
                        "SINGLE_CONSISTENT",
                        0
                    )
                    -
                    class_counts.get(
                        "MULTI_REGION",
                        0
                    )
                    -
                    class_counts.get(
                        "SINGLE_MISMATCH",
                        0
                    )
                    -
                    class_counts.get(
                        "MISSING_ASSOCIATION",
                        0
                    )
                    -
                    class_counts.get(
                        "MULTI_PRIMARY_NOT_IN_LIST",
                        0
                    )
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print(
    "\n===================================="
)

print(
    "SECONDARY-ONLY ACCOUNTING SUMMARY"
)

print(
    "===================================="
)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


special.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


if accounted_total != expected_total:

    raise RuntimeError(
        "Secondary-only accounting mismatch."
    )


print(
    "\nPre-model Fix E complete."
)