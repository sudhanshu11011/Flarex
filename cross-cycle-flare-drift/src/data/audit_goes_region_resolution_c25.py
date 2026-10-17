from pathlib import Path
import re

import numpy as np
import pandas as pd


GOES_DIR = Path(
    "data/raw/goes"
)

EDGE_FILE = Path(
    "data_manifests/harp_noaa_edges.csv"
)

AUDIT_FILE = Path(
    "data_manifests/goes_c25_region_resolution_audit.csv"
)

SUMMARY_FILE = Path(
    "data_manifests/goes_c25_region_resolution_summary.csv"
)


YEARS = list(
    range(
        2020,
        2026
    )
)


C25_START = pd.Timestamp(
    "2020-01-01T00:00:00Z"
)

C25_END_EXCLUSIVE = pd.Timestamp(
    "2026-01-01T00:00:00Z"
)


# ============================================================
# HELPERS
# ============================================================

def parse_flare_class(value):

    if pd.isna(value):
        return None

    text = str(
        value
    ).strip().upper()


    match = re.match(
        r"^([ABCMX])\s*([0-9]+(?:\.[0-9]+)?)",
        text
    )


    if not match:
        return None


    letter = match.group(1)

    magnitude = float(
        match.group(2)
    )


    base = {
        "A": 1e-8,
        "B": 1e-7,
        "C": 1e-6,
        "M": 1e-5,
        "X": 1e-4,
    }[
        letter
    ]


    return (
        letter,
        magnitude,
        base * magnitude,
    )


def is_m1_plus(value):

    parsed = parse_flare_class(
        value
    )


    if parsed is None:
        return False


    _, _, flux = parsed


    return (
        flux
        >=
        1e-5
    )


def find_noaa_column(df):

    candidates = [
        "NOAA_AR",
        "NOAA",
        "noaa_ar",
        "noaa",
    ]


    for column in candidates:

        if column in df.columns:
            return column


    for column in df.columns:

        if (
            "NOAA"
            in column.upper()
        ):

            return column


    raise RuntimeError(
        "Could not identify NOAA column "
        "in harp_noaa_edges.csv"
    )


# ============================================================
# LOAD CANONICAL HARP ↔ NOAA MAP
# ============================================================

if not EDGE_FILE.exists():

    raise FileNotFoundError(
        f"Missing canonical edge file: "
        f"{EDGE_FILE}"
    )


edges = pd.read_csv(
    EDGE_FILE
)


noaa_column = find_noaa_column(
    edges
)


canonical_noaa = (
    pd.to_numeric(
        edges[
            noaa_column
        ],
        errors="coerce"
    )
    .dropna()
    .astype(int)
)


canonical_noaa_set = set(
    canonical_noaa.tolist()
)


print(
    "===================================="
)

print(
    "PHASE 10A — CYCLE25 GOES REGION AUDIT"
)

print(
    "===================================="
)


print(
    "Canonical NOAA column:",
    noaa_column
)


print(
    "Canonical NOAA count:",
    len(
        canonical_noaa_set
    )
)


if len(
    canonical_noaa_set
) > 0:

    print(
        "Canonical NOAA range:",
        min(
            canonical_noaa_set
        ),
        "→",
        max(
            canonical_noaa_set
        )
    )


# ============================================================
# LOAD CYCLE25 GOES FILES
# ============================================================

frames = []


for year in YEARS:

    file = (
        GOES_DIR
        /
        f"goes_flare_{year}.csv"
    )


    if not file.exists():

        raise FileNotFoundError(
            f"Missing GOES file: {file}"
        )


    df = pd.read_csv(
        file
    )


    required = {
        "start_time",
        "flare_class",
        "active_region",
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
            f"{year} missing columns: "
            f"{sorted(missing)}"
        )


    df[
        "catalog_year"
    ] = year


    frames.append(
        df
    )


goes = pd.concat(
    frames,
    ignore_index=True
)


goes[
    "start_time_utc"
] = pd.to_datetime(
    goes[
        "start_time"
    ],
    utc=True,
    errors="coerce"
)


goes = goes[
    (
        goes[
            "start_time_utc"
        ]
        >=
        C25_START
    )
    &
    (
        goes[
            "start_time_utc"
        ]
        <
        C25_END_EXCLUSIVE
    )
].copy()


# ============================================================
# M1+ ONLY
# ============================================================

goes[
    "is_m1_plus"
] = (
    goes[
        "flare_class"
    ]
    .apply(
        is_m1_plus
    )
)


major = goes[
    goes[
        "is_m1_plus"
    ]
].copy()


major[
    "active_region_numeric"
] = pd.to_numeric(
    major[
        "active_region"
    ],
    errors="coerce"
)


major[
    "has_region"
] = (
    major[
        "active_region_numeric"
    ]
    .notna()
    &
    (
        major[
            "active_region_numeric"
        ]
        >
        0
    )
)


assigned = major[
    major[
        "has_region"
    ]
].copy()


assigned[
    "active_region_numeric"
] = (
    assigned[
        "active_region_numeric"
    ]
    .astype(int)
)


# ============================================================
# TEST REPRESENTATIONS
# ============================================================

assigned[
    "raw_noaa_candidate"
] = (
    assigned[
        "active_region_numeric"
    ]
)


assigned[
    "plus_10000_candidate"
] = np.where(
    assigned[
        "active_region_numeric"
    ]
    <
    10000,

    assigned[
        "active_region_numeric"
    ]
    +
    10000,

    assigned[
        "active_region_numeric"
    ]
)


assigned[
    "raw_in_canonical_map"
] = (
    assigned[
        "raw_noaa_candidate"
    ]
    .isin(
        canonical_noaa_set
    )
)


assigned[
    "normalized_in_canonical_map"
] = (
    assigned[
        "plus_10000_candidate"
    ]
    .isin(
        canonical_noaa_set
    )
)


# ============================================================
# FINAL NORMALIZATION RULE
# ============================================================
#
# Important:
#
# This does NOT use modulo arithmetic.
#
# GOES active_region < 10000
#     -> +10000
#
# already full NOAA >= 10000
#     -> keep as-is
#
# ============================================================

assigned[
    "normalized_noaa_ar"
] = np.where(
    assigned[
        "active_region_numeric"
    ]
    <
    10000,

    assigned[
        "active_region_numeric"
    ]
    +
    10000,

    assigned[
        "active_region_numeric"
    ]
)


assigned[
    "normalization_rule"
] = np.where(
    assigned[
        "active_region_numeric"
    ]
    <
    10000,

    "ADD_10000",

    "KEEP_FULL_NOAA"
)


# ============================================================
# SUMMARY COUNTS
# ============================================================

total_catalog_rows = len(
    goes
)


invalid_start_time = int(
    goes[
        "start_time_utc"
    ]
    .isna()
    .sum()
)


m1_total = len(
    major
)


m1_with_region = len(
    assigned
)


m1_without_region = (
    m1_total
    -
    m1_with_region
)


raw_lt_10000 = int(
    (
        assigned[
            "active_region_numeric"
        ]
        <
        10000
    )
    .sum()
)


raw_ge_10000 = int(
    (
        assigned[
            "active_region_numeric"
        ]
        >=
        10000
    )
    .sum()
)


raw_map_matches = int(
    assigned[
        "raw_in_canonical_map"
    ]
    .sum()
)


normalized_map_matches = int(
    assigned[
        "normalized_in_canonical_map"
    ]
    .sum()
)


# ============================================================
# CONTRADICTION CHECK
# ============================================================
#
# A contradictory row would be:
#
# raw value matches one canonical NOAA
# AND normalized (+10000) matches a different
# canonical NOAA.
#
# ============================================================

assigned[
    "contradictory_representation"
] = (
    assigned[
        "raw_in_canonical_map"
    ]
    &
    assigned[
        "normalized_in_canonical_map"
    ]
    &
    (
        assigned[
            "raw_noaa_candidate"
        ]
        !=
        assigned[
            "plus_10000_candidate"
        ]
    )
)


contradictions = int(
    assigned[
        "contradictory_representation"
    ]
    .sum()
)


# ============================================================
# YEAR SUMMARY
# ============================================================

year_rows = []


for year in YEARS:

    y_all = goes[
        goes[
            "catalog_year"
        ]
        ==
        year
    ]


    y_major = major[
        major[
            "catalog_year"
        ]
        ==
        year
    ]


    y_assigned = assigned[
        assigned[
            "catalog_year"
        ]
        ==
        year
    ]


    year_rows.append(
        {
            "year":
                year,

            "catalog_rows":
                len(
                    y_all
                ),

            "m1_plus":
                len(
                    y_major
                ),

            "m1_plus_with_region":
                len(
                    y_assigned
                ),

            "m1_plus_without_region":
                (
                    len(
                        y_major
                    )
                    -
                    len(
                        y_assigned
                    )
                ),

            "region_lt_10000":
                int(
                    (
                        y_assigned[
                            "active_region_numeric"
                        ]
                        <
                        10000
                    )
                    .sum()
                ),

            "region_ge_10000":
                int(
                    (
                        y_assigned[
                            "active_region_numeric"
                        ]
                        >=
                        10000
                    )
                    .sum()
                ),

            "raw_canonical_matches":
                int(
                    y_assigned[
                        "raw_in_canonical_map"
                    ]
                    .sum()
                ),

            "normalized_canonical_matches":
                int(
                    y_assigned[
                        "normalized_in_canonical_map"
                    ]
                    .sum()
                ),
        }
    )


summary = pd.DataFrame(
    year_rows
)


# ============================================================
# SAVE
# ============================================================

AUDIT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


assigned.to_csv(
    AUDIT_FILE,
    index=False
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PRINT
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "CYCLE25 REGION ENCODING"
)

print(
    "------------------------------------"
)


print(
    "Catalog rows:",
    total_catalog_rows
)


print(
    "Invalid start_time:",
    invalid_start_time
)


print(
    "M1+ events:",
    m1_total
)


print(
    "M1+ with active_region:",
    m1_with_region
)


print(
    "M1+ without active_region:",
    m1_without_region
)


print(
    "Assigned region <10000:",
    raw_lt_10000
)


print(
    "Assigned region >=10000:",
    raw_ge_10000
)


print(
    "Raw canonical-map matches:",
    raw_map_matches
)


print(
    "+10000/as-is canonical-map matches:",
    normalized_map_matches
)


print(
    "Contradictory representations:",
    contradictions
)


print(
    "\n------------------------------------"
)

print(
    "YEAR SUMMARY"
)

print(
    "------------------------------------"
)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# DECISION
# ============================================================

normalization_pass = (

    invalid_start_time
    ==
    0

    and

    contradictions
    ==
    0

    and

    m1_with_region
    >
    0

    and

    normalized_map_matches
    >=
    raw_map_matches
)


print(
    "\n===================================="
)

print(
    "PHASE 10A DECISION"
)

print(
    "===================================="
)


print(
    "Normalization rule:"
)


print(
    "active_region < 10000  → +10000"
)


print(
    "active_region >=10000 → keep as-is"
)


print(
    "Generic modulo rule: NOT USED"
)


print(
    "Canonical HARP-NOAA map is validation evidence,"
)


print(
    "not a requirement for an event to be valid."
)


print(
    "Audit status:",
    (
        "PASS"
        if normalization_pass
        else "FAIL"
    )
)


print(
    "\nSaved:"
)


print(
    AUDIT_FILE
)


print(
    SUMMARY_FILE
)


if not normalization_pass:

    raise RuntimeError(
        "Cycle25 GOES region normalization "
        "audit failed."
    )


print(
    "\nPhase 10A complete."
)