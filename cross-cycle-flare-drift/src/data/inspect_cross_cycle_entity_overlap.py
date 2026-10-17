from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

C24_FILE = Path(
    "data/processed/cycle24_model_ready.parquet"
)

C25_FILE = Path(
    "data/processed/cycle25_model_ready.parquet"
)

GROUP_FILE = Path(
    "data_manifests/harp_noaa_entity_groups.csv"
)

OUTPUT_FILE = Path(
    "data_manifests/phase11_cross_cycle_overlap_detail.csv"
)


# ============================================================
# LOAD
# ============================================================

c24 = pd.read_parquet(
    C24_FILE
)

c25 = pd.read_parquet(
    C25_FILE
)

groups = pd.read_csv(
    GROUP_FILE
)


c24[
    "ENTITY_GROUP_ID"
] = c24[
    "ENTITY_GROUP_ID"
].astype(str)


c25[
    "ENTITY_GROUP_ID"
] = c25[
    "ENTITY_GROUP_ID"
].astype(str)


groups[
    "ENTITY_GROUP_ID"
] = groups[
    "ENTITY_GROUP_ID"
].astype(str)


# ============================================================
# FIND OVERLAP
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


overlap = sorted(
    c24_groups
    &
    c25_groups
)


print(
    "===================================="
)

print(
    "PHASE 11B — CROSS-CYCLE ENTITY INSPECTION"
)

print(
    "===================================="
)


print(
    "Overlapping entity groups:",
    len(
        overlap
    )
)


if len(
    overlap
) == 0:

    print(
        "No cross-cycle entity overlap."
    )

    raise SystemExit(
        0
    )


# ============================================================
# DETAIL
# ============================================================

detail_rows = []


for entity_group in overlap:

    print(
        "\n------------------------------------"
    )

    print(
        "ENTITY GROUP:",
        entity_group
    )

    print(
        "------------------------------------"
    )


    c24_rows = c24[
        c24[
            "ENTITY_GROUP_ID"
        ]
        ==
        entity_group
    ].copy()


    c25_rows = c25[
        c25[
            "ENTITY_GROUP_ID"
        ]
        ==
        entity_group
    ].copy()


    canonical = groups[
        groups[
            "ENTITY_GROUP_ID"
        ]
        ==
        entity_group
    ].copy()


    c24_harps = sorted(
        pd.to_numeric(
            c24_rows[
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
            c25_rows[
                "HARPNUM"
            ],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


    canonical_harps = sorted(
        pd.to_numeric(
            canonical[
                "HARPNUM"
            ],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


    canonical_noaa = sorted(
        pd.to_numeric(
            canonical[
                "NOAA_AR"
            ],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


    c24_time = pd.to_datetime(
        c24_rows[
            "observation_time_utc"
        ],
        utc=True,
        errors="coerce"
    )


    c25_time = pd.to_datetime(
        c25_rows[
            "observation_time_utc"
        ],
        utc=True,
        errors="coerce"
    )


    c24_positive = int(
        (
            c24_rows[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    c25_positive = int(
        (
            c25_rows[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    print(
        "Cycle24 rows:",
        len(
            c24_rows
        )
    )


    print(
        "Cycle24 HARPs:",
        c24_harps
    )


    print(
        "Cycle24 first UTC:",
        c24_time.min()
    )


    print(
        "Cycle24 last UTC:",
        c24_time.max()
    )


    print(
        "Cycle24 positives:",
        c24_positive
    )


    print()


    print(
        "Cycle25 rows:",
        len(
            c25_rows
        )
    )


    print(
        "Cycle25 HARPs:",
        c25_harps
    )


    print(
        "Cycle25 first UTC:",
        c25_time.min()
    )


    print(
        "Cycle25 last UTC:",
        c25_time.max()
    )


    print(
        "Cycle25 positives:",
        c25_positive
    )


    print()


    print(
        "Canonical HARPs:",
        canonical_harps
    )


    print(
        "Canonical NOAA ARs:",
        canonical_noaa
    )


    detail_rows.append(
        {
            "ENTITY_GROUP_ID":
                entity_group,

            "cycle24_rows":
                len(
                    c24_rows
                ),

            "cycle25_rows":
                len(
                    c25_rows
                ),

            "cycle24_positive":
                c24_positive,

            "cycle25_positive":
                c25_positive,

            "cycle24_harps":
                ",".join(
                    map(
                        str,
                        c24_harps
                    )
                ),

            "cycle25_harps":
                ",".join(
                    map(
                        str,
                        c25_harps
                    )
                ),

            "canonical_harps":
                ",".join(
                    map(
                        str,
                        canonical_harps
                    )
                ),

            "canonical_noaa_ars":
                ",".join(
                    map(
                        str,
                        canonical_noaa
                    )
                ),

            "cycle24_first_utc":
                c24_time.min(),

            "cycle24_last_utc":
                c24_time.max(),

            "cycle25_first_utc":
                c25_time.min(),

            "cycle25_last_utc":
                c25_time.max(),
        }
    )


# ============================================================
# SAVE
# ============================================================

detail = pd.DataFrame(
    detail_rows
)


detail.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n===================================="
)

print(
    "SUMMARY"
)

print(
    "===================================="
)


print(
    detail.to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\nPhase 11B inspection complete."
)