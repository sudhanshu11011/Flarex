from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RAW_ROOT = Path(
    "data/raw/jsoc"
)

EDGE_FILE = Path(
    "data_manifests/harp_noaa_edges.csv"
)

AUDIT_FILE = Path(
    "data_manifests/phase11_harp_noaa_temporal_edge_audit.csv"
)

SUMMARY_FILE = Path(
    "data_manifests/phase11_harp_noaa_temporal_edge_summary.csv"
)


# Specific suspicious cross-cycle edge discovered in Phase 11
TARGET_HARP = 13354
TARGET_NOAA = 11114


# ============================================================
# HELPERS
# ============================================================

def parse_trec(value):

    if pd.isna(value):
        return pd.NaT

    text = str(
        value
    ).strip()

    text = text.replace(
        "_TAI",
        ""
    )

    text = (
        text
        .replace(
            ".",
            "-",
            2
        )
        .replace(
            "_",
            "T",
            1
        )
    )

    return pd.to_datetime(
        text,
        format="%Y-%m-%dT%H:%M:%S",
        errors="coerce"
    )


def parse_noaa_ar(value):

    if pd.isna(value):
        return None

    try:

        number = int(
            float(
                value
            )
        )

    except Exception:

        return None

    if number <= 0:
        return None

    return number


def parse_noaa_list(value):

    if pd.isna(value):
        return []

    text = str(
        value
    )

    numbers = []

    for token in re.findall(
        r"\d+",
        text
    ):

        number = int(
            token
        )

        if number > 0:

            numbers.append(
                number
            )

    return sorted(
        set(
            numbers
        )
    )


def interval_gap_days(
    first_a,
    last_a,
    first_b,
    last_b
):

    if (
        pd.isna(
            first_a
        )
        or
        pd.isna(
            last_a
        )
        or
        pd.isna(
            first_b
        )
        or
        pd.isna(
            last_b
        )
    ):

        return np.nan


    # A occurs completely before B
    if last_a < first_b:

        return (
            first_b
            -
            last_a
        ).total_seconds() / 86400


    # B occurs completely before A
    if last_b < first_a:

        return (
            first_a
            -
            last_b
        ).total_seconds() / 86400


    # Intervals overlap
    return 0.0


# ============================================================
# LOAD FULL SHARP METADATA
# ============================================================

print(
    "===================================="
)

print(
    "PHASE 11E — GLOBAL TEMPORAL EDGE AUDIT"
)

print(
    "===================================="
)


frames = []


for cycle in [
    "cycle24",
    "cycle25",
]:

    directory = (
        RAW_ROOT
        /
        cycle
        /
        "sharp"
    )


    files = sorted(
        directory.glob(
            "*.parquet"
        )
    )


    if not files:

        raise FileNotFoundError(
            f"No SHARP parquet files found: "
            f"{directory}"
        )


    print(
        cycle,
        "files:",
        len(
            files
        )
    )


    for index, file in enumerate(
        files,
        start=1
    ):

        df = pd.read_parquet(
            file,
            columns=[
                "HARPNUM",
                "T_REC",
                "NOAA_AR",
                "NOAA_ARS",
            ],
        )


        df[
            "cycle"
        ] = cycle


        frames.append(
            df
        )


        if (
            index % 30 == 0
            or
            index == len(
                files
            )
        ):

            print(
                cycle,
                index,
                "/",
                len(
                    files
                )
            )


if not frames:

    raise RuntimeError(
        "No SHARP metadata was loaded."
    )


sharp = pd.concat(
    frames,
    ignore_index=True
)


sharp[
    "HARPNUM"
] = pd.to_numeric(
    sharp[
        "HARPNUM"
    ],
    errors="coerce"
)


sharp[
    "_T_REC"
] = sharp[
    "T_REC"
].apply(
    parse_trec
)


sharp[
    "_PRIMARY_NOAA"
] = sharp[
    "NOAA_AR"
].apply(
    parse_noaa_ar
)


# ============================================================
# HARP TEMPORAL INFORMATION
# ============================================================

harp_rows = []


for harpnum, group in sharp.groupby(
    "HARPNUM"
):

    if pd.isna(
        harpnum
    ):
        continue


    harpnum = int(
        harpnum
    )


    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # Pandas may convert Python None values into float NaN.
    #
    # Therefore:
    #
    #     value is not None
    #
    # is NOT sufficient.
    #
    # Explicit pd.notna() filtering prevents:
    #
    # ValueError:
    # cannot convert float NaN to integer
    # --------------------------------------------------------

    primary_noaa = sorted(
        {
            int(
                value
            )

            for value
            in group[
                "_PRIMARY_NOAA"
            ]

            if (
                pd.notna(
                    value
                )
                and
                float(
                    value
                )
                >
                0
            )
        }
    )


    all_noaa = sorted(
        {
            region

            for value
            in group[
                "NOAA_ARS"
            ]

            for region
            in parse_noaa_list(
                value
            )
        }
    )


    valid_times = group[
        "_T_REC"
    ].dropna()


    first_trec = (
        valid_times.min()
        if len(
            valid_times
        )
        >
        0
        else pd.NaT
    )


    last_trec = (
        valid_times.max()
        if len(
            valid_times
        )
        >
        0
        else pd.NaT
    )


    harp_rows.append(
        {
            "HARPNUM":
                harpnum,

            "first_trec":
                first_trec,

            "last_trec":
                last_trec,

            "primary_noaa":
                primary_noaa,

            "all_noaa":
                all_noaa,

            "cycles":
                sorted(
                    group[
                        "cycle"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                ),
        }
    )


harp_info = pd.DataFrame(
    harp_rows
)


harp_lookup = {

    int(
        row[
            "HARPNUM"
        ]
    ):
        row

    for _, row
    in harp_info.iterrows()
}


print(
    "\nObserved HARPs:",
    len(
        harp_lookup
    )
)


# ============================================================
# PRIMARY NOAA TEMPORAL ANCHORS
# ============================================================
#
# NOAA_AR is used here ONLY as a temporal anchor.
#
# This script:
#
# - does NOT modify labels
# - does NOT remove edges
# - does NOT rebuild entity groups
#
# It only diagnoses temporal consistency.
# ============================================================

primary_anchors = {}


for _, row in harp_info.iterrows():

    harpnum = int(
        row[
            "HARPNUM"
        ]
    )


    primary_regions = row[
        "primary_noaa"
    ]


    for noaa in primary_regions:

        # Additional defensive check
        if pd.isna(
            noaa
        ):
            continue


        noaa = int(
            noaa
        )


        primary_anchors.setdefault(
            noaa,
            []
        ).append(
            {
                "HARPNUM":
                    harpnum,

                "first_trec":
                    row[
                        "first_trec"
                    ],

                "last_trec":
                    row[
                        "last_trec"
                    ],
            }
        )


print(
    "NOAA primary temporal anchors:",
    len(
        primary_anchors
    )
)


# ============================================================
# LOAD CANONICAL HARP ↔ NOAA EDGES
# ============================================================

if not EDGE_FILE.exists():

    raise FileNotFoundError(
        f"Missing canonical edge file: "
        f"{EDGE_FILE}"
    )


edges = pd.read_csv(
    EDGE_FILE
)


required_edge_columns = {
    "HARPNUM",
    "NOAA_AR",
}


missing_edge_columns = (
    required_edge_columns
    -
    set(
        edges.columns
    )
)


if missing_edge_columns:

    raise RuntimeError(
        "Canonical edge file missing columns: "
        f"{sorted(missing_edge_columns)}"
    )


edges[
    "HARPNUM"
] = pd.to_numeric(
    edges[
        "HARPNUM"
    ],
    errors="coerce"
)


edges[
    "NOAA_AR"
] = pd.to_numeric(
    edges[
        "NOAA_AR"
    ],
    errors="coerce"
)


edges = edges.dropna(
    subset=[
        "HARPNUM",
        "NOAA_AR",
    ]
).copy()


edges[
    "HARPNUM"
] = (
    edges[
        "HARPNUM"
    ]
    .astype(int)
)


edges[
    "NOAA_AR"
] = (
    edges[
        "NOAA_AR"
    ]
    .astype(int)
)


# ============================================================
# AUDIT EACH CANONICAL EDGE
# ============================================================

audit_rows = []


for _, edge in edges.iterrows():

    harpnum = int(
        edge[
            "HARPNUM"
        ]
    )


    noaa = int(
        edge[
            "NOAA_AR"
        ]
    )


    harp = harp_lookup.get(
        harpnum
    )


    # --------------------------------------------------------
    # HARP is outside our extracted C24/C25 SHARP dataset
    # --------------------------------------------------------

    if harp is None:

        audit_rows.append(
            {
                "HARPNUM":
                    harpnum,

                "NOAA_AR":
                    noaa,

                "harp_observed":
                    False,

                "harp_first_trec":
                    pd.NaT,

                "harp_last_trec":
                    pd.NaT,

                "harp_primary_noaa":
                    "",

                "is_primary_edge":
                    False,

                "primary_anchor_count":
                    0,

                "minimum_anchor_gap_days":
                    np.nan,

                "nearest_anchor_harp":
                    pd.NA,

                "extreme_gap_gt_365d":
                    False,

                "audit_status":
                    "HARP_NOT_IN_EXTRACTED_DOMAIN",
            }
        )

        continue


    primary_set = set(
        harp[
            "primary_noaa"
        ]
    )


    is_primary = (
        noaa
        in
        primary_set
    )


    anchors = primary_anchors.get(
        noaa,
        []
    )


    # ========================================================
    # CASE 1:
    # Edge is directly supported by this HARP's NOAA_AR
    # ========================================================

    if is_primary:

        minimum_gap = 0.0

        nearest_harp = harpnum

        status = (
            "PRIMARY_SUPPORTED"
        )


    # ========================================================
    # CASE 2:
    # Secondary NOAA association with available temporal anchor
    # ========================================================

    elif len(
        anchors
    ) > 0:

        candidates = []


        for anchor in anchors:

            gap = interval_gap_days(
                harp[
                    "first_trec"
                ],
                harp[
                    "last_trec"
                ],
                anchor[
                    "first_trec"
                ],
                anchor[
                    "last_trec"
                ],
            )


            candidates.append(
                (
                    gap,
                    anchor[
                        "HARPNUM"
                    ],
                )
            )


        candidates = [
            item

            for item in candidates

            if not pd.isna(
                item[
                    0
                ]
            )
        ]


        if candidates:

            candidates.sort(
                key=lambda item:
                    item[
                        0
                    ]
            )


            minimum_gap = float(
                candidates[
                    0
                ][
                    0
                ]
            )


            nearest_harp = int(
                candidates[
                    0
                ][
                    1
                ]
            )


            if minimum_gap == 0:

                status = (
                    "SECONDARY_TEMPORALLY_OVERLAPS_PRIMARY_ANCHOR"
                )


            else:

                status = (
                    "SECONDARY_SEPARATED_FROM_PRIMARY_ANCHOR"
                )


        else:

            minimum_gap = np.nan

            nearest_harp = pd.NA

            status = (
                "SECONDARY_ANCHOR_TIME_UNAVAILABLE"
            )


    # ========================================================
    # CASE 3:
    # No NOAA_AR temporal anchor available in extracted domain
    # ========================================================

    else:

        minimum_gap = np.nan

        nearest_harp = pd.NA

        status = (
            "SECONDARY_WITHOUT_PRIMARY_ANCHOR"
        )


    extreme_gap = bool(
        (
            not pd.isna(
                minimum_gap
            )
        )
        and
        (
            minimum_gap
            >
            365
        )
    )


    audit_rows.append(
        {
            "HARPNUM":
                harpnum,

            "NOAA_AR":
                noaa,

            "harp_observed":
                True,

            "harp_first_trec":
                harp[
                    "first_trec"
                ],

            "harp_last_trec":
                harp[
                    "last_trec"
                ],

            "harp_primary_noaa":
                ",".join(
                    map(
                        str,
                        harp[
                            "primary_noaa"
                        ],
                    )
                ),

            "is_primary_edge":
                is_primary,

            "primary_anchor_count":
                len(
                    anchors
                ),

            "minimum_anchor_gap_days":
                minimum_gap,

            "nearest_anchor_harp":
                nearest_harp,

            "extreme_gap_gt_365d":
                extreme_gap,

            "audit_status":
                status,
        }
    )


audit = pd.DataFrame(
    audit_rows
)


# ============================================================
# SAVE FULL EDGE AUDIT
# ============================================================

AUDIT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


audit.to_csv(
    AUDIT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

observed = audit[
    audit[
        "harp_observed"
    ]
].copy()


primary_edges = int(
    observed[
        "is_primary_edge"
    ].sum()
)


secondary_edges = int(
    (
        ~observed[
            "is_primary_edge"
        ]
    ).sum()
)


extreme_edges = observed[
    observed[
        "extreme_gap_gt_365d"
    ]
].copy()


unanchored_secondary = int(
    (
        observed[
            "audit_status"
        ]
        ==
        "SECONDARY_WITHOUT_PRIMARY_ANCHOR"
    ).sum()
)


temporally_overlapping_secondary = int(
    (
        observed[
            "audit_status"
        ]
        ==
        "SECONDARY_TEMPORALLY_OVERLAPS_PRIMARY_ANCHOR"
    ).sum()
)


temporally_separated_secondary = int(
    (
        observed[
            "audit_status"
        ]
        ==
        "SECONDARY_SEPARATED_FROM_PRIMARY_ANCHOR"
    ).sum()
)


summary = pd.DataFrame(
    [
        {
            "canonical_edges_total":
                len(
                    audit
                ),

            "edges_with_observed_harp":
                len(
                    observed
                ),

            "primary_supported_edges":
                primary_edges,

            "secondary_edges":
                secondary_edges,

            "secondary_temporally_overlapping":
                temporally_overlapping_secondary,

            "secondary_temporally_separated":
                temporally_separated_secondary,

            "secondary_without_primary_anchor":
                unanchored_secondary,

            "extreme_gap_gt_365d":
                len(
                    extreme_edges
                ),
        }
    ]
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PRINT GLOBAL RESULTS
# ============================================================

print(
    "\n===================================="
)

print(
    "GLOBAL EDGE SUMMARY"
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
# EXTREME TEMPORAL GAPS
# ============================================================

print(
    "\n===================================="
)

print(
    "EXTREME TEMPORAL GAPS > 365 DAYS"
)

print(
    "===================================="
)


if len(
    extreme_edges
) == 0:

    print(
        "None"
    )


else:

    display_columns = [
        "HARPNUM",
        "NOAA_AR",
        "harp_first_trec",
        "harp_last_trec",
        "harp_primary_noaa",
        "minimum_anchor_gap_days",
        "nearest_anchor_harp",
        "audit_status",
    ]


    print(
        extreme_edges[
            display_columns
        ]
        .sort_values(
            "minimum_anchor_gap_days",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# TARGET EDGE
# ============================================================

print(
    "\n===================================="
)

print(
    "TARGET EDGE 13354 → 11114"
)

print(
    "===================================="
)


target = audit[
    (
        audit[
            "HARPNUM"
        ]
        ==
        TARGET_HARP
    )
    &
    (
        audit[
            "NOAA_AR"
        ]
        ==
        TARGET_NOAA
    )
]


if len(
    target
) == 1:

    print(
        target.to_string(
            index=False
        )
    )


elif len(
    target
) == 0:

    print(
        "Target edge not found."
    )


else:

    print(
        "Target edge appears more than once."
    )


# ============================================================
# TARGET HARP CONTEXT
# ============================================================

print(
    "\n===================================="
)

print(
    "TARGET HARP 13354 CONTEXT"
)

print(
    "===================================="
)


target_harp_info = harp_info[
    harp_info[
        "HARPNUM"
    ]
    ==
    TARGET_HARP
]


if len(
    target_harp_info
) == 1:

    row = target_harp_info.iloc[
        0
    ]


    print(
        "First T_REC:",
        row[
            "first_trec"
        ]
    )


    print(
        "Last T_REC:",
        row[
            "last_trec"
        ]
    )


    print(
        "NOAA_AR values:",
        row[
            "primary_noaa"
        ]
    )


    print(
        "NOAA_ARS values:",
        row[
            "all_noaa"
        ]
    )


else:

    print(
        "Target HARP unavailable or duplicated."
    )


# ============================================================
# SAVED FILES
# ============================================================

print(
    "\n===================================="
)

print(
    "SAVED"
)

print(
    "===================================="
)


print(
    AUDIT_FILE
)


print(
    SUMMARY_FILE
)


print(
    "\nPhase 11E audit complete."
)