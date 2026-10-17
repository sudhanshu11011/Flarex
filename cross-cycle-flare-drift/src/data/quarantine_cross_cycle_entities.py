from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path(
    "data/processed"
)

MANIFEST_DIR = Path(
    "data_manifests"
)


C24_FILE = (
    PROCESSED_DIR
    /
    "cycle24_model_ready.parquet"
)

C25_FILE = (
    PROCESSED_DIR
    /
    "cycle25_model_ready.parquet"
)


C24_OUTPUT = (
    PROCESSED_DIR
    /
    "cycle24_model_ready_leakage_safe.parquet"
)

C25_OUTPUT = (
    PROCESSED_DIR
    /
    "cycle25_model_ready_leakage_safe.parquet"
)


SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "phase11_cross_cycle_quarantine_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

if not C24_FILE.exists():

    raise FileNotFoundError(
        f"Missing: {C24_FILE}"
    )


if not C25_FILE.exists():

    raise FileNotFoundError(
        f"Missing: {C25_FILE}"
    )


c24 = pd.read_parquet(
    C24_FILE
)

c25 = pd.read_parquet(
    C25_FILE
)


required = {
    "ENTITY_GROUP_ID",
    "HARPNUM",
    "T_REC",
    "target",
}


for label, dataframe in [
    ("Cycle24", c24),
    ("Cycle25", c25),
]:

    missing = (
        required
        -
        set(
            dataframe.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"{label} missing columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# STANDARDIZE ENTITY IDS
# ============================================================

c24[
    "ENTITY_GROUP_ID"
] = (
    c24[
        "ENTITY_GROUP_ID"
    ]
    .astype(str)
)


c25[
    "ENTITY_GROUP_ID"
] = (
    c25[
        "ENTITY_GROUP_ID"
    ]
    .astype(str)
)


# ============================================================
# FIND CROSS-CYCLE ENTITY OVERLAP
# ============================================================

c24_groups = set(
    c24[
        "ENTITY_GROUP_ID"
    ]
)


c25_groups = set(
    c25[
        "ENTITY_GROUP_ID"
    ]
)


overlap_groups = sorted(
    c24_groups
    &
    c25_groups
)


print(
    "===================================="
)

print(
    "PHASE 11F — CROSS-CYCLE QUARANTINE"
)

print(
    "===================================="
)


print(
    "Overlapping entity groups:",
    len(
        overlap_groups
    )
)


print(
    "Groups:",
    overlap_groups
)


if len(
    overlap_groups
) == 0:

    raise RuntimeError(
        "No cross-cycle entity overlap found. "
        "Quarantine is unnecessary."
    )


# ============================================================
# AUDIT EACH QUARANTINED GROUP
# ============================================================

summary_rows = []


for group_id in overlap_groups:

    c24_group = c24[
        c24[
            "ENTITY_GROUP_ID"
        ]
        ==
        group_id
    ].copy()


    c25_group = c25[
        c25[
            "ENTITY_GROUP_ID"
        ]
        ==
        group_id
    ].copy()


    c24_positive = int(
        (
            c24_group[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    c24_negative = int(
        (
            c24_group[
                "target"
            ]
            ==
            0
        )
        .sum()
    )


    c25_positive = int(
        (
            c25_group[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    c25_negative = int(
        (
            c25_group[
                "target"
            ]
            ==
            0
        )
        .sum()
    )


    c24_harps = sorted(
        pd.to_numeric(
            c24_group[
                "HARPNUM"
            ],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


    c25_harps = sorted(
        pd.to_numeric(
            c25_group[
                "HARPNUM"
            ],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


    c24_time = pd.to_datetime(
        c24_group[
            "observation_time_utc"
        ],
        utc=True,
        errors="coerce"
    )


    c25_time = pd.to_datetime(
        c25_group[
            "observation_time_utc"
        ],
        utc=True,
        errors="coerce"
    )


    summary_rows.append(
        {
            "ENTITY_GROUP_ID":
                group_id,

            "quarantine_reason":
                "CROSS_CYCLE_ENTITY_OVERLAP",

            "cycle24_rows_removed":
                len(
                    c24_group
                ),

            "cycle24_positive_removed":
                c24_positive,

            "cycle24_negative_removed":
                c24_negative,

            "cycle24_harps":
                ",".join(
                    map(
                        str,
                        c24_harps
                    )
                ),

            "cycle24_first_utc":
                c24_time.min(),

            "cycle24_last_utc":
                c24_time.max(),

            "cycle25_rows_removed":
                len(
                    c25_group
                ),

            "cycle25_positive_removed":
                c25_positive,

            "cycle25_negative_removed":
                c25_negative,

            "cycle25_harps":
                ",".join(
                    map(
                        str,
                        c25_harps
                    )
                ),

            "cycle25_first_utc":
                c25_time.min(),

            "cycle25_last_utc":
                c25_time.max(),
        }
    )


# ============================================================
# QUARANTINE
# ============================================================

c24_safe = c24[
    ~c24[
        "ENTITY_GROUP_ID"
    ].isin(
        overlap_groups
    )
].copy()


c25_safe = c25[
    ~c25[
        "ENTITY_GROUP_ID"
    ].isin(
        overlap_groups
    )
].copy()


# ============================================================
# VERIFY NO OVERLAP REMAINS
# ============================================================

remaining_overlap = sorted(
    set(
        c24_safe[
            "ENTITY_GROUP_ID"
        ]
    )
    &
    set(
        c25_safe[
            "ENTITY_GROUP_ID"
        ]
    )
)


if remaining_overlap:

    raise RuntimeError(
        "Cross-cycle entity overlap still remains: "
        f"{remaining_overlap}"
    )


# ============================================================
# BASIC INTEGRITY CHECKS
# ============================================================

c24_duplicates = int(
    c24_safe[
        [
            "HARPNUM",
            "T_REC",
        ]
    ]
    .duplicated()
    .sum()
)


c25_duplicates = int(
    c25_safe[
        [
            "HARPNUM",
            "T_REC",
        ]
    ]
    .duplicated()
    .sum()
)


if (
    c24_duplicates != 0
    or
    c25_duplicates != 0
):

    raise RuntimeError(
        "Duplicate key introduced during quarantine."
    )


if list(
    c24_safe.columns
) != list(
    c25_safe.columns
):

    raise RuntimeError(
        "Cross-cycle schemas differ after quarantine."
    )


# ============================================================
# SAVE
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


c24_safe.to_parquet(
    C24_OUTPUT,
    index=False
)


c25_safe.to_parquet(
    C25_OUTPUT,
    index=False
)


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# FINAL COUNTS
# ============================================================

c24_removed = (
    len(
        c24
    )
    -
    len(
        c24_safe
    )
)


c25_removed = (
    len(
        c25
    )
    -
    len(
        c25_safe
    )
)


c24_positive = int(
    (
        c24_safe[
            "target"
        ]
        ==
        1
    )
    .sum()
)


c24_negative = int(
    (
        c24_safe[
            "target"
        ]
        ==
        0
    )
    .sum()
)


c25_positive = int(
    (
        c25_safe[
            "target"
        ]
        ==
        1
    )
    .sum()
)


c25_negative = int(
    (
        c25_safe[
            "target"
        ]
        ==
        0
    )
    .sum()
)


print(
    "\n------------------------------------"
)

print(
    "CYCLE24"
)

print(
    "------------------------------------"
)


print(
    "Before:",
    len(
        c24
    )
)


print(
    "Removed:",
    c24_removed
)


print(
    "After:",
    len(
        c24_safe
    )
)


print(
    "Positive:",
    c24_positive
)


print(
    "Negative:",
    c24_negative
)


print(
    "Unique entity groups:",
    c24_safe[
        "ENTITY_GROUP_ID"
    ].nunique()
)


print(
    "\n------------------------------------"
)

print(
    "CYCLE25"
)

print(
    "------------------------------------"
)


print(
    "Before:",
    len(
        c25
    )
)


print(
    "Removed:",
    c25_removed
)


print(
    "After:",
    len(
        c25_safe
    )
)


print(
    "Positive:",
    c25_positive
)


print(
    "Negative:",
    c25_negative
)


print(
    "Unique entity groups:",
    c25_safe[
        "ENTITY_GROUP_ID"
    ].nunique()
)


print(
    "\n===================================="
)

print(
    "QUARANTINE VALIDATION"
)

print(
    "===================================="
)


print(
    "Cross-cycle ENTITY_GROUP_ID overlap:",
    len(
        remaining_overlap
    )
)


print(
    "Cycle24 duplicate keys:",
    c24_duplicates
)


print(
    "Cycle25 duplicate keys:",
    c25_duplicates
)


print(
    "Schemas identical:",
    list(
        c24_safe.columns
    )
    ==
    list(
        c25_safe.columns
    )
)


print(
    "\nSaved:"
)


print(
    C24_OUTPUT
)


print(
    C25_OUTPUT
)


print(
    SUMMARY_FILE
)


print(
    "\nPhase 11F quarantine complete."
)