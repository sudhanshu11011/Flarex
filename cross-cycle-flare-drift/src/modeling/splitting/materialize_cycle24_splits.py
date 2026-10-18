from pathlib import Path
import hashlib

import pandas as pd


# ============================================================
# PATHS
# ============================================================

SOURCE_FILE = Path(
    "data/final/dataset_v2/cycle24_dataset_v2.parquet"
)

ASSIGNMENT_FILE = Path(
    "data_manifests/model_ch1_split_assignments.csv"
)

AUDIT_FILE = Path(
    "artifacts/splitting/manifests/model_ch1_split_audit_summary.csv"
)


OUTPUT_DIR = Path(
    "artifacts/splitting/data"
)

MANIFEST_DIR = Path(
    "artifacts/splitting/manifests"
)


TRAIN_FILE = (
    OUTPUT_DIR
    /
    "cycle24_train.parquet"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    /
    "cycle24_validation.parquet"
)

HOLDOUT_FILE = (
    OUTPUT_DIR
    /
    "cycle24_holdout.parquet"
)


SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "model_ch1_materialization_summary.csv"
)


# ============================================================
# EXPECTED COUNTS
# ============================================================

EXPECTED = {

    "TRAIN": {
        "rows": 62064,
        "positive": 2420,
        "negative": 59644,
        "groups": 822,
    },

    "VALIDATION": {
        "rows": 13299,
        "positive": 515,
        "negative": 12784,
        "groups": 176,
    },

    "HOLDOUT": {
        "rows": 13072,
        "positive": 527,
        "negative": 12545,
        "groups": 176,
    },
}


OUTPUT_FILES = {
    "TRAIN": TRAIN_FILE,
    "VALIDATION": VALIDATION_FILE,
    "HOLDOUT": HOLDOUT_FILE,
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path):

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


def sample_key_set(df):

    return set(
        zip(
            pd.to_numeric(
                df["HARPNUM"],
                errors="raise",
            ).astype(int),

            df["T_REC"].astype(str),
        )
    )


# ============================================================
# VERIFY INPUT FILES
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 1.4 — MATERIALIZE SPLITS"
)

print(
    "===================================="
)


for path in [
    SOURCE_FILE,
    ASSIGNMENT_FILE,
    AUDIT_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# VERIFY CHAPTER 1.3 PASSED
# ============================================================

audit = pd.read_csv(
    AUDIT_FILE
)


if len(audit) != 3:

    raise RuntimeError(
        "Chapter 1.3 audit should contain "
        "exactly three split rows."
    )


if not audit[
    "split_audit_pass"
].astype(bool).all():

    raise RuntimeError(
        "Chapter 1.3 audit has not passed."
    )


print(
    "Chapter 1.3 audit: PASS"
)


# ============================================================
# LOAD SOURCE + ASSIGNMENTS
# ============================================================

source = pd.read_parquet(
    SOURCE_FILE
)


assignments = pd.read_csv(
    ASSIGNMENT_FILE
)


source_columns = list(
    source.columns
)


source_dtypes = [
    str(
        source[column].dtype
    )

    for column
    in source_columns
]


assignments[
    "ENTITY_GROUP_ID"
] = assignments[
    "ENTITY_GROUP_ID"
].astype(str)


source[
    "_ENTITY_STRING"
] = source[
    "ENTITY_GROUP_ID"
].astype(str)


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


source[
    "_SPLIT"
] = source[
    "_ENTITY_STRING"
].map(
        split_map
    )


missing_split_rows = int(
    source[
        "_SPLIT"
    ]
    .isna()
    .sum()
)


if missing_split_rows != 0:

    raise RuntimeError(
        f"{missing_split_rows} rows have "
        "no split assignment."
    )


unexpected_splits = (
    set(
        source[
            "_SPLIT"
        ].unique()
    )
    -
    {
        "TRAIN",
        "VALIDATION",
        "HOLDOUT",
    }
)


if unexpected_splits:

    raise RuntimeError(
        f"Unexpected split names: "
        f"{sorted(unexpected_splits)}"
    )


# ============================================================
# MATERIALIZE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


split_frames = {}

summary_rows = []


for split in [
    "TRAIN",
    "VALIDATION",
    "HOLDOUT",
]:

    output_path = OUTPUT_FILES[
        split
    ]


    subset = source[
        source[
            "_SPLIT"
        ]
        ==
        split
    ].copy()


    # Remove audit-only temporary columns.
    subset.drop(
        columns=[
            "_ENTITY_STRING",
            "_SPLIT",
        ],
        inplace=True,
    )


    # Preserve exact source schema/order.
    subset = subset[
        source_columns
    ]


    # ========================================================
    # COUNTS
    # ========================================================

    rows = len(
        subset
    )


    positive = int(
        (
            subset[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    negative = int(
        (
            subset[
                "target"
            ]
            ==
            0
        )
        .sum()
    )


    groups = int(
        subset[
            "ENTITY_GROUP_ID"
        ]
        .nunique()
    )


    expected = EXPECTED[
        split
    ]


    if rows != expected[
        "rows"
    ]:

        raise RuntimeError(
            f"{split}: row count mismatch."
        )


    if positive != expected[
        "positive"
    ]:

        raise RuntimeError(
            f"{split}: positive count mismatch."
        )


    if negative != expected[
        "negative"
    ]:

        raise RuntimeError(
            f"{split}: negative count mismatch."
        )


    if groups != expected[
        "groups"
    ]:

        raise RuntimeError(
            f"{split}: group count mismatch."
        )


    # ========================================================
    # SCHEMA
    # ========================================================

    if list(
        subset.columns
    ) != source_columns:

        raise RuntimeError(
            f"{split}: schema column mismatch."
        )


    subset_dtypes = [
        str(
            subset[column].dtype
        )

        for column
        in subset.columns
    ]


    if subset_dtypes != source_dtypes:

        raise RuntimeError(
            f"{split}: dtype mismatch."
        )


    # ========================================================
    # SAMPLE INTEGRITY
    # ========================================================

    duplicate_keys = int(
        subset[
            [
                "HARPNUM",
                "T_REC",
            ]
        ]
        .duplicated()
        .sum()
    )


    if duplicate_keys != 0:

        raise RuntimeError(
            f"{split}: duplicate sample keys."
        )


    # ========================================================
    # SAVE
    # ========================================================

    subset.to_parquet(
        output_path,
        index=False,
    )


    # Reload to independently verify written file.
    verified = pd.read_parquet(
        output_path
    )


    if len(
        verified
    ) != rows:

        raise RuntimeError(
            f"{split}: written row count mismatch."
        )


    if list(
        verified.columns
    ) != source_columns:

        raise RuntimeError(
            f"{split}: written schema mismatch."
        )


    digest = sha256_file(
        output_path
    )


    split_frames[
        split
    ] = verified


    summary_rows.append(
        {
            "split":
                split,

            "file":
                str(
                    output_path
                ),

            "rows":
                rows,

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

            "entity_groups":
                groups,

            "unique_harpnums":
                int(
                    verified[
                        "HARPNUM"
                    ]
                    .nunique()
                ),

            "columns":
                len(
                    verified.columns
                ),

            "duplicate_sample_keys":
                duplicate_keys,

            "sha256":
                digest,

            "file_size_bytes":
                output_path
                .stat()
                .st_size,
        }
    )


# ============================================================
# CROSS-SPLIT ENTITY OVERLAP
# ============================================================

group_sets = {

    split:
        set(
            frame[
                "ENTITY_GROUP_ID"
            ].astype(str)
        )

    for split, frame
    in split_frames.items()
}


if (
    group_sets["TRAIN"]
    &
    group_sets["VALIDATION"]
):

    raise RuntimeError(
        "TRAIN/VALIDATION group leakage."
    )


if (
    group_sets["TRAIN"]
    &
    group_sets["HOLDOUT"]
):

    raise RuntimeError(
        "TRAIN/HOLDOUT group leakage."
    )


if (
    group_sets["VALIDATION"]
    &
    group_sets["HOLDOUT"]
):

    raise RuntimeError(
        "VALIDATION/HOLDOUT group leakage."
    )


# ============================================================
# CROSS-SPLIT HARP OVERLAP
# ============================================================

harp_sets = {

    split:
        set(
            pd.to_numeric(
                frame[
                    "HARPNUM"
                ],
                errors="raise",
            ).astype(int)
        )

    for split, frame
    in split_frames.items()
}


for left, right in [
    ("TRAIN", "VALIDATION"),
    ("TRAIN", "HOLDOUT"),
    ("VALIDATION", "HOLDOUT"),
]:

    overlap = (
        harp_sets[left]
        &
        harp_sets[right]
    )


    if overlap:

        raise RuntimeError(
            f"{left}/{right} HARPNUM leakage."
        )


# ============================================================
# EXACT SOURCE PARTITION CHECK
# ============================================================

source_clean = source.drop(
    columns=[
        "_ENTITY_STRING",
        "_SPLIT",
    ]
)


source_keys = sample_key_set(
    source_clean
)


train_keys = sample_key_set(
    split_frames[
        "TRAIN"
    ]
)


validation_keys = sample_key_set(
    split_frames[
        "VALIDATION"
    ]
)


holdout_keys = sample_key_set(
    split_frames[
        "HOLDOUT"
    ]
)


if (
    train_keys
    &
    validation_keys
):

    raise RuntimeError(
        "TRAIN/VALIDATION sample-key overlap."
    )


if (
    train_keys
    &
    holdout_keys
):

    raise RuntimeError(
        "TRAIN/HOLDOUT sample-key overlap."
    )


if (
    validation_keys
    &
    holdout_keys
):

    raise RuntimeError(
        "VALIDATION/HOLDOUT sample-key overlap."
    )


combined_keys = (
    train_keys
    |
    validation_keys
    |
    holdout_keys
)


if combined_keys != source_keys:

    raise RuntimeError(
        "Materialized datasets do not form "
        "an exact partition of Cycle24 source."
    )


# ============================================================
# TOTAL ACCOUNTING
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


if int(
    summary[
        "rows"
    ].sum()
) != len(
    source_clean
):

    raise RuntimeError(
        "Total row accounting mismatch."
    )


if int(
    summary[
        "positive"
    ].sum()
) != int(
    source_clean[
        "target"
    ].sum()
):

    raise RuntimeError(
        "Positive accounting mismatch."
    )


if int(
    summary[
        "negative"
    ].sum()
) != int(
    (
        source_clean[
            "target"
        ]
        ==
        0
    ).sum()
):

    raise RuntimeError(
        "Negative accounting mismatch."
    )


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "MATERIALIZED SPLIT SUMMARY"
)

print(
    "===================================="
)


print(
    summary[
        [
            "split",
            "rows",
            "positive",
            "negative",
            "positive_rate_pct",
            "entity_groups",
            "unique_harpnums",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n===================================="
)

print(
    "PARTITION VERIFICATION"
)

print(
    "===================================="
)


print(
    "ENTITY_GROUP overlap: 0"
)

print(
    "HARPNUM overlap: 0"
)

print(
    "Sample-key overlap: 0"
)

print(
    "Source rows:",
    len(
        source_clean
    )
)

print(
    "Materialized rows:",
    int(
        summary[
            "rows"
        ].sum()
    )
)

print(
    "Exact source partition: PASS"
)


print(
    "\n===================================="
)

print(
    "FILE HASHES"
)

print(
    "===================================="
)


for _, row in summary.iterrows():

    print(
        row[
            "split"
        ],
        ":",
        row[
            "sha256"
        ],
    )


print(
    "\nSaved:"
)


for path in OUTPUT_FILES.values():

    print(
        path
    )


print(
    SUMMARY_FILE
)


print(
    "\nMODEL CHAPTER 1.4 COMPLETE."
)