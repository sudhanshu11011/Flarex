from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

GOES_AUDIT_FILE = Path(
    "data_manifests/goes_c24_label_contract_audit.csv"
)

EDGE_FILE = Path(
    "data_manifests/harp_noaa_edges.csv"
)

OUTPUT_FILE = Path(
    "data_manifests/goes_c24_region_resolution_audit.csv"
)

SUMMARY_FILE = Path(
    "data_manifests/goes_c24_region_resolution_summary.csv"
)


# ============================================================
# VALIDATE FILES
# ============================================================

if not GOES_AUDIT_FILE.exists():

    raise FileNotFoundError(
        f"Missing GOES audit file: {GOES_AUDIT_FILE}"
    )


if not EDGE_FILE.exists():

    raise FileNotFoundError(
        f"Missing HARP-NOAA edge file: {EDGE_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

goes = pd.read_csv(
    GOES_AUDIT_FILE
)

edges = pd.read_csv(
    EDGE_FILE
)


# ============================================================
# NORMALIZE TYPES
# ============================================================

goes["start_time_utc"] = pd.to_datetime(
    goes["start_time_utc"],
    utc=True,
    errors="coerce"
)


goes["active_region_parsed"] = pd.to_numeric(
    goes["active_region_parsed"],
    errors="coerce"
)


# Convert boolean safely
goes["is_m1plus_class"] = (
    goes["is_m1plus_class"]
    .astype(str)
    .str.lower()
    .eq("true")
)


goes["has_active_region"] = (
    goes["has_active_region"]
    .astype(str)
    .str.lower()
    .eq("true")
)


edges["NOAA_AR"] = pd.to_numeric(
    edges["NOAA_AR"],
    errors="coerce"
)


edges = edges.dropna(
    subset=[
        "NOAA_AR"
    ]
)


edges["NOAA_AR"] = (
    edges["NOAA_AR"]
    .astype(int)
)


# ============================================================
# CANONICAL NOAA REGION SET
# ============================================================

canonical_noaa = sorted(
    set(
        edges["NOAA_AR"]
        .astype(int)
    )
)


canonical_noaa_set = set(
    canonical_noaa
)


print(
    "Canonical NOAA AR count:",
    len(canonical_noaa)
)


print(
    "Canonical NOAA range:",
    min(canonical_noaa),
    "→",
    max(canonical_noaa)
)


# ============================================================
# BUILD SUFFIX LOOKUP
#
# Example:
#
# 11166 % 10000 = 1166
#
# suffix_map[1166] = [11166]
# ============================================================

suffix_map = {}


for noaa in canonical_noaa:

    suffix = (
        noaa
        % 10000
    )


    suffix_map.setdefault(
        suffix,
        []
    )


    suffix_map[
        suffix
    ].append(
        noaa
    )


for suffix in suffix_map:

    suffix_map[
        suffix
    ] = sorted(
        set(
            suffix_map[
                suffix
            ]
        )
    )


# ============================================================
# GLOBAL SUFFIX COLLISION AUDIT
# ============================================================

suffix_collisions = {
    suffix:
    values

    for suffix, values
    in suffix_map.items()

    if len(values) > 1
}


print(
    "\n===================================="
)

print(
    "CANONICAL SUFFIX COLLISION AUDIT"
)

print(
    "===================================="
)


print(
    "Unique suffixes:",
    len(suffix_map)
)


print(
    "Suffixes mapping to >1 NOAA AR:",
    len(suffix_collisions)
)


if len(suffix_collisions) > 0:

    print(
        "\nFirst collisions:"
    )

    for suffix, candidates in list(
        sorted(
            suffix_collisions.items()
        )
    )[:20]:

        print(
            suffix,
            "->",
            candidates
        )


# ============================================================
# M1+ WITH REGION ONLY
# ============================================================

m1_region = goes[
    (
        goes[
            "is_m1plus_class"
        ]
    )
    &
    (
        goes[
            "has_active_region"
        ]
    )
].copy()


print(
    "\nM1+ events with GOES region:",
    len(m1_region)
)


# ============================================================
# RESOLVE GOES REGION → FULL NOAA ID
# ============================================================

resolution_rows = []


for _, row in m1_region.iterrows():

    short_region = int(
        row[
            "active_region_parsed"
        ]
    )


    # --------------------------------------------------------
    # Exact full-number candidate
    # --------------------------------------------------------

    exact_candidate = (
        short_region
        if short_region in canonical_noaa_set
        else None
    )


    # --------------------------------------------------------
    # Suffix candidate(s)
    # --------------------------------------------------------

    suffix_candidates = (
        suffix_map.get(
            short_region % 10000,
            []
        )
    )


    # --------------------------------------------------------
    # Resolution rule
    # --------------------------------------------------------

    if exact_candidate is not None:

        resolution_status = (
            "RESOLVED_EXACT"
        )

        resolved_noaa = (
            exact_candidate
        )

    elif len(
        suffix_candidates
    ) == 1:

        resolution_status = (
            "RESOLVED_UNIQUE_SUFFIX"
        )

        resolved_noaa = (
            suffix_candidates[0]
        )

    elif len(
        suffix_candidates
    ) == 0:

        resolution_status = (
            "UNRESOLVED_NO_CANDIDATE"
        )

        resolved_noaa = None

    else:

        resolution_status = (
            "AMBIGUOUS_MULTIPLE_CANDIDATES"
        )

        resolved_noaa = None


    # --------------------------------------------------------
    # Compare against simple +10000 hypothesis
    # ONLY FOR AUDIT.
    # Not our final rule yet.
    # --------------------------------------------------------

    plus_10000 = (
        short_region
        + 10000
    )


    plus_10000_exists = (
        plus_10000
        in canonical_noaa_set
    )


    resolver_equals_plus10000 = (
        resolved_noaa is not None
        and
        resolved_noaa == plus_10000
    )


    resolution_rows.append(
        {
            "start_time_utc":
                row[
                    "start_time_utc"
                ],

            "flare_class":
                row[
                    "flare_class"
                ],

            "goes_active_region":
                short_region,

            "candidate_count":
                len(
                    suffix_candidates
                ),

            "candidate_full_noaa":
                ",".join(
                    map(
                        str,
                        suffix_candidates
                    )
                ),

            "resolution_status":
                resolution_status,

            "resolved_noaa":
                resolved_noaa,

            "plus_10000":
                plus_10000,

            "plus_10000_exists":
                plus_10000_exists,

            "resolver_equals_plus10000":
                resolver_equals_plus10000,
        }
    )


resolution = pd.DataFrame(
    resolution_rows
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

print(
    "\n\n===================================="
)

print(
    "QC-8B GOES REGION RESOLUTION AUDIT"
)

print(
    "===================================="
)


print(
    "M1+ region-assigned events:",
    len(resolution)
)


status_counts = (
    resolution[
        "resolution_status"
    ]
    .value_counts()
)


print(
    "\nResolution status:"
)

print(
    status_counts.to_string()
)


resolved_mask = (
    resolution[
        "resolved_noaa"
    ]
    .notna()
)


resolved_count = int(
    resolved_mask.sum()
)


print(
    "\nResolved events:",
    resolved_count
)


print(
    "Unresolved / ambiguous:",
    len(resolution)
    - resolved_count
)


if len(resolution) > 0:

    print(
        "Resolution coverage:",
        f"{resolved_count / len(resolution) * 100:.2f}%"
    )


# ============================================================
# +10000 AUDIT
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "+10000 HYPOTHESIS AUDIT"
)

print(
    "------------------------------------"
)


print(
    "+10000 exists in canonical set:",
    int(
        resolution[
            "plus_10000_exists"
        ].sum()
    )
)


print(
    "Resolved NOAA == GOES + 10000:",
    int(
        resolution[
            "resolver_equals_plus10000"
        ].sum()
    )
)


# ============================================================
# YEAR SUMMARY
# ============================================================

resolution[
    "year"
] = (
    resolution[
        "start_time_utc"
    ]
    .dt.year
)


year_summary = (
    resolution
    .groupby(
        "year"
    )
    .agg(
        region_assigned_m1plus=(
            "goes_active_region",
            "size"
        ),

        resolved=(
            "resolved_noaa",
            lambda values:
            int(
                values.notna().sum()
            )
        ),

        ambiguous=(
            "resolution_status",
            lambda values:
            int(
                (
                    values
                    ==
                    "AMBIGUOUS_MULTIPLE_CANDIDATES"
                ).sum()
            )
        ),

        unresolved=(
            "resolution_status",
            lambda values:
            int(
                (
                    values
                    ==
                    "UNRESOLVED_NO_CANDIDATE"
                ).sum()
            )
        ),
    )
    .reset_index()
)


year_summary[
    "resolution_pct"
] = (
    year_summary[
        "resolved"
    ]
    /
    year_summary[
        "region_assigned_m1plus"
    ]
    * 100
)


print(
    "\n------------------------------------"
)

print(
    "YEAR-LEVEL REGION RESOLUTION"
)

print(
    "------------------------------------"
)


print(
    year_summary.to_string(
        index=False
    )
)


# ============================================================
# KNOWN SAMPLE CHECK
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "KNOWN 2011 SAMPLE CHECK"
)

print(
    "------------------------------------"
)


known = resolution[
    (
        resolution[
            "start_time_utc"
        ]
        ==
        pd.Timestamp(
            "2011-03-09T23:11:00Z"
        )
    )
]


if len(known) == 0:

    # Some catalog versions could differ
    # slightly in start timestamp.
    known = resolution[
        (
            resolution[
                "start_time_utc"
            ].dt.date
            ==
            pd.Timestamp(
                "2011-03-09"
            ).date()
        )
        &
        (
            resolution[
                "flare_class"
            ]
            .astype(str)
            .str.upper()
            ==
            "X2.2"
        )
    ]


if len(known) == 0:

    print(
        "Known X2.2 event not found."
    )

else:

    print(
        known[
            [
                "start_time_utc",
                "flare_class",
                "goes_active_region",
                "candidate_full_noaa",
                "resolution_status",
                "resolved_noaa",
            ]
        ]
        .to_string(
            index=False
        )
    )


    if (
        11166
        in known[
            "resolved_noaa"
        ]
        .dropna()
        .astype(int)
        .tolist()
    ):

        print(
            "PASS: GOES region 1166 "
            "resolved to NOAA 11166."
        )

    else:

        print(
            "CHECK REQUIRED:"
            " expected NOAA 11166."
        )


# ============================================================
# SHOW PROBLEM EVENTS
# ============================================================

problems = resolution[
    ~resolution[
        "resolved_noaa"
    ].notna()
]


print(
    "\n------------------------------------"
)

print(
    "UNRESOLVED / AMBIGUOUS EVENTS"
)

print(
    "------------------------------------"
)


print(
    "Count:",
    len(problems)
)


if len(problems) > 0:

    print(
        problems[
            [
                "start_time_utc",
                "flare_class",
                "goes_active_region",
                "candidate_full_noaa",
                "resolution_status",
            ]
        ]
        .head(30)
        .to_string(
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


resolution.to_csv(
    OUTPUT_FILE,
    index=False
)


year_summary.to_csv(
    SUMMARY_FILE,
    index=False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)


print(
    "\nQC-8B region resolution audit finished."
)