import re
from pathlib import Path

import drms
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

client = drms.Client()

SHARP_SERIES = "hmi.sharp_cea_720s"

EDGE_FILE = Path(
    "data_manifests/harp_noaa_edges.csv"
)

OUTPUT_FILE = Path(
    "data_manifests/harp_noaa_coverage_audit.csv"
)

SUMMARY_FILE = Path(
    "data_manifests/harp_noaa_coverage_summary.csv"
)


if not EDGE_FILE.exists():
    raise FileNotFoundError(
        f"Mapping edge file not found: {EDGE_FILE}"
    )


# ============================================================
# LOAD CANONICAL HARP ↔ NOAA EDGES
# ============================================================

edges = pd.read_csv(
    EDGE_FILE
)


required_columns = {
    "HARPNUM",
    "NOAA_AR",
}


if not required_columns.issubset(
    edges.columns
):
    raise RuntimeError(
        "harp_noaa_edges.csv does not have "
        "required HARPNUM and NOAA_AR columns."
    )


edges["HARPNUM"] = pd.to_numeric(
    edges["HARPNUM"],
    errors="coerce"
)


edges["NOAA_AR"] = pd.to_numeric(
    edges["NOAA_AR"],
    errors="coerce"
)


edges = edges.dropna(
    subset=[
        "HARPNUM",
        "NOAA_AR",
    ]
)


edges["HARPNUM"] = (
    edges["HARPNUM"]
    .astype(int)
)

edges["NOAA_AR"] = (
    edges["NOAA_AR"]
    .astype(int)
)


# ============================================================
# BUILD HARP → NOAA LOOKUP
# ============================================================

harp_to_noaa = (
    edges
    .groupby("HARPNUM")["NOAA_AR"]
    .apply(
        lambda values:
        set(
            values.astype(int)
        )
    )
    .to_dict()
)


print(
    "Canonical mapped HARPs:",
    len(harp_to_noaa)
)


# ============================================================
# NOAA PARSER
# ============================================================

def parse_noaa_values(value):
    """
    Parse NOAA_ARS safely.

    Returns a set of positive NOAA region numbers.
    """

    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    numbers = re.findall(
        r"\d+",
        text
    )

    result = set()

    for number in numbers:

        try:

            parsed = int(number)

            if parsed > 0:
                result.add(
                    parsed
                )

        except ValueError:
            pass

    return result


# ============================================================
# BUILD SAME CROSS-CYCLE AUDIT GRID
# ============================================================

DATES = []


# Cycle 24: 2010 partial year
for month in [
    6,
    9,
    12,
]:

    DATES.append(
        (
            "Cycle24",
            f"2010.{month:02d}.15_12:00:00_TAI"
        )
    )


# 2011 → 2019
for year in range(
    2011,
    2020
):

    for month in [
        3,
        6,
        9,
        12,
    ]:

        DATES.append(
            (
                "Cycle24",
                f"{year}.{month:02d}.15_12:00:00_TAI"
            )
        )


# Cycle 25 audit window used so far:
# 2020 → 2025
for year in range(
    2020,
    2026
):

    for month in [
        3,
        6,
        9,
        12,
    ]:

        DATES.append(
            (
                "Cycle25",
                f"{year}.{month:02d}.15_12:00:00_TAI"
            )
        )


# ============================================================
# AUDIT
# ============================================================

records = []


for cycle, time in DATES:

    print(
        "\n===================================="
    )

    print(
        cycle,
        "-",
        time
    )

    print(
        "===================================="
    )


    query = (
        f"{SHARP_SERIES}"
        f"[]"
        f"[{time}]"
    )


    try:

        df = client.query(
            query,
            key=(
                "T_REC,"
                "HARPNUM,"
                "NOAA_AR,"
                "NOAA_ARS"
            )
        )

    except Exception as error:

        print(
            "QUERY ERROR:",
            error
        )

        continue


    if len(df) == 0:

        print(
            "No SHARP records."
        )

        continue


    print(
        "SHARP rows:",
        len(df)
    )


    for _, row in df.iterrows():

        # ----------------------------------------------------
        # HARPNUM
        # ----------------------------------------------------

        try:

            harpnum = int(
                row["HARPNUM"]
            )

        except Exception:

            continue


        # ----------------------------------------------------
        # JSOC current NOAA associations
        # ----------------------------------------------------

        jsoc_noaa = parse_noaa_values(
            row.get(
                "NOAA_ARS"
            )
        )


        # Include NOAA_AR too
        try:

            primary_noaa = int(
                row["NOAA_AR"]
            )

            if primary_noaa > 0:

                jsoc_noaa.add(
                    primary_noaa
                )

        except Exception:
            pass


        jsoc_has_noaa = (
            len(jsoc_noaa) > 0
        )


        # ----------------------------------------------------
        # CANONICAL MAPPING
        # ----------------------------------------------------

        canonical_noaa = (
            harp_to_noaa.get(
                harpnum,
                set()
            )
        )


        harp_in_mapping = (
            harpnum
            in harp_to_noaa
        )


        # ----------------------------------------------------
        # RELATION AGREEMENT
        #
        # Canonical mapping may contain MORE NOAA regions than
        # one timestamp because it can represent lifetime
        # associations.
        #
        # Therefore timestamp NOAA set only needs to be a
        # subset of canonical mapping.
        # ----------------------------------------------------

        if not jsoc_has_noaa:

            relation_status = (
                "NO_NOAA_AT_TIMESTAMP"
            )

        elif not harp_in_mapping:

            relation_status = (
                "MISSING_FROM_CANONICAL"
            )

        elif jsoc_noaa.issubset(
            canonical_noaa
        ):

            relation_status = (
                "AGREEMENT"
            )

        else:

            relation_status = (
                "RELATION_CONFLICT"
            )


        records.append(
            {
                "cycle":
                    cycle,

                "audit_time":
                    time,

                "T_REC":
                    row.get(
                        "T_REC"
                    ),

                "HARPNUM":
                    harpnum,

                "NOAA_AR_raw":
                    row.get(
                        "NOAA_AR"
                    ),

                "NOAA_ARS_raw":
                    row.get(
                        "NOAA_ARS"
                    ),

                "jsoc_noaa_set":
                    ",".join(
                        map(
                            str,
                            sorted(
                                jsoc_noaa
                            )
                        )
                    ),

                "canonical_noaa_set":
                    ",".join(
                        map(
                            str,
                            sorted(
                                canonical_noaa
                            )
                        )
                    ),

                "jsoc_has_noaa":
                    jsoc_has_noaa,

                "harp_in_mapping":
                    harp_in_mapping,

                "relation_status":
                    relation_status,
            }
        )


# ============================================================
# DATAFRAME
# ============================================================

audit = pd.DataFrame(
    records
)


if len(audit) == 0:

    raise RuntimeError(
        "No audit records were generated."
    )


# ============================================================
# OVERALL SUMMARY
# ============================================================

print(
    "\n\n===================================="
)

print(
    "HARP ↔ NOAA COVERAGE AUDIT"
)

print(
    "===================================="
)


print(
    "Total sampled SHARP rows:",
    len(audit)
)


print(
    "Rows with NOAA association:",
    int(
        audit[
            "jsoc_has_noaa"
        ].sum()
    )
)


print(
    "Rows without NOAA association:",
    int(
        (
            ~audit[
                "jsoc_has_noaa"
            ]
        ).sum()
    )
)


linked = audit[
    audit[
        "jsoc_has_noaa"
    ]
].copy()


print(
    "NOAA-linked rows found "
    "in canonical mapping:",
    int(
        linked[
            "harp_in_mapping"
        ].sum()
    )
)


print(
    "NOAA-linked rows missing "
    "from canonical mapping:",
    int(
        (
            ~linked[
                "harp_in_mapping"
            ]
        ).sum()
    )
)


if len(linked) > 0:

    coverage = (
        linked[
            "harp_in_mapping"
        ].mean()
        * 100
    )

else:

    coverage = 0.0


print(
    "Canonical mapping coverage:",
    f"{coverage:.2f}%"
)


# ============================================================
# RELATION STATUS
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "RELATION STATUS"
)

print(
    "------------------------------------"
)


status_counts = (
    audit[
        "relation_status"
    ]
    .value_counts()
)


print(
    status_counts.to_string()
)


# ============================================================
# CYCLE SUMMARY
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "CYCLE-LEVEL COVERAGE"
)

print(
    "------------------------------------"
)


summary_rows = []


for cycle in [
    "Cycle24",
    "Cycle25",
]:

    cycle_df = audit[
        audit[
            "cycle"
        ] == cycle
    ]


    cycle_linked = cycle_df[
        cycle_df[
            "jsoc_has_noaa"
        ]
    ]


    total_rows = len(
        cycle_df
    )


    linked_rows = len(
        cycle_linked
    )


    mapped_rows = int(
        cycle_linked[
            "harp_in_mapping"
        ].sum()
    )


    conflicts = int(
        (
            cycle_linked[
                "relation_status"
            ]
            ==
            "RELATION_CONFLICT"
        ).sum()
    )


    missing_mapping = int(
        (
            cycle_linked[
                "relation_status"
            ]
            ==
            "MISSING_FROM_CANONICAL"
        ).sum()
    )


    if linked_rows > 0:

        mapping_coverage = (
            mapped_rows
            / linked_rows
            * 100
        )

    else:

        mapping_coverage = 0.0


    summary_rows.append(
        {
            "cycle":
                cycle,

            "total_rows":
                total_rows,

            "noaa_linked_rows":
                linked_rows,

            "mapped_rows":
                mapped_rows,

            "missing_mapping":
                missing_mapping,

            "relation_conflicts":
                conflicts,

            "mapping_coverage_pct":
                round(
                    mapping_coverage,
                    2
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# UNIQUE HARP COVERAGE
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "UNIQUE HARP COVERAGE"
)

print(
    "------------------------------------"
)


for cycle in [
    "Cycle24",
    "Cycle25",
]:

    cycle_linked = audit[
        (
            audit["cycle"] == cycle
        )
        &
        (
            audit["jsoc_has_noaa"]
        )
    ]


    unique_harps = set(
        cycle_linked[
            "HARPNUM"
        ].astype(int)
    )


    mapped_harps = {
        harp

        for harp in unique_harps

        if harp in harp_to_noaa
    }


    print(
        cycle
    )

    print(
        "Unique NOAA-linked HARPs:",
        len(
            unique_harps
        )
    )

    print(
        "Present in canonical mapping:",
        len(
            mapped_harps
        )
    )

    print(
        "Missing:",
        len(
            unique_harps
            -
            mapped_harps
        )
    )


# ============================================================
# IMPORTANT KNOWN HARPs
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "KNOWN HARP COVERAGE CHECK"
)

print(
    "------------------------------------"
)


KNOWN_HARPS = [
    401,
    7451,
    12519,
]


for harp in KNOWN_HARPS:

    mapping = harp_to_noaa.get(
        harp
    )


    print(
        f"HARP {harp}:",
        sorted(mapping)
        if mapping
        else "NOT IN MAPPING"
    )


# ============================================================
# SHOW MISSING MAPPINGS
# ============================================================

missing = linked[
    linked[
        "relation_status"
    ]
    ==
    "MISSING_FROM_CANONICAL"
]


print(
    "\n------------------------------------"
)

print(
    "NOAA-LINKED HARPs MISSING FROM MAP"
)

print(
    "------------------------------------"
)


if len(missing) == 0:

    print(
        "None."
    )

else:

    print(
        missing[
            [
                "cycle",
                "audit_time",
                "HARPNUM",
                "NOAA_AR_raw",
                "NOAA_ARS_raw",
            ]
        ]
        .drop_duplicates()
        .to_string(
            index=False
        )
    )


# ============================================================
# SHOW RELATION CONFLICTS
# ============================================================

conflicts = linked[
    linked[
        "relation_status"
    ]
    ==
    "RELATION_CONFLICT"
]


print(
    "\n------------------------------------"
)

print(
    "RELATION CONFLICTS"
)

print(
    "------------------------------------"
)


if len(conflicts) == 0:

    print(
        "None."
    )

else:

    print(
        conflicts[
            [
                "cycle",
                "audit_time",
                "HARPNUM",
                "jsoc_noaa_set",
                "canonical_noaa_set",
            ]
        ]
        .drop_duplicates()
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


audit.to_csv(
    OUTPUT_FILE,
    index=False
)


summary_df.to_csv(
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
    "\nHARP-NOAA coverage audit finished."
)