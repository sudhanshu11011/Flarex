from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INTERIM_DIR = Path(
    "data/interim"
)

PROCESSED_DIR = Path(
    "data/processed"
)

GROUP_FILE = Path(
    "data_manifests/harp_noaa_entity_groups.csv"
)

EDGE_FILE = Path(
    "data_manifests/harp_noaa_edges.csv"
)

SUMMARY_FILE = Path(
    "data_manifests/phase10_final_dataset_summary.csv"
)


PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

SHARP_FEATURES = [
    "TOTUSJH",
    "TOTPOT",
    "TOTUSJZ",
    "ABSNJZH",
    "SAVNCPP",
    "USFLUX",
    "AREA_ACR",
    "MEANPOT",
    "R_VALUE",
    "SHRGT45",
    "MEANSHR",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "MEANJZH",
    "MEANJZD",
    "MEANALP",
]


LORENTZ_FEATURES = [
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
]


ALL_FEATURES = (
    SHARP_FEATURES
    +
    LORENTZ_FEATURES
)


# ============================================================
# COLUMN DETECTION HELPERS
# ============================================================

def find_column(
    dataframe,
    candidates,
    contains=None
):

    lookup = {
        str(column).upper():
            column

        for column in dataframe.columns
    }


    for candidate in candidates:

        key = candidate.upper()

        if key in lookup:

            return lookup[key]


    if contains:

        token = contains.upper()

        for column in dataframe.columns:

            if token in str(
                column
            ).upper():

                return column


    return None


def require_column(
    dataframe,
    candidates,
    label,
    contains=None
):

    column = find_column(
        dataframe,
        candidates,
        contains
    )


    if column is None:

        raise RuntimeError(
            f"{label}: could not identify "
            f"required column. "
            f"Available columns: "
            f"{list(dataframe.columns)}"
        )


    return column


# ============================================================
# UNIQUE MAPPING VALIDATION
# ============================================================

def validate_unique_mapping(
    dataframe,
    key_column,
    group_column,
    label
):

    subset = dataframe[
        [
            key_column,
            group_column,
        ]
    ].dropna()


    counts = (
        subset
        .groupby(
            key_column
        )[
            group_column
        ]
        .nunique()
    )


    conflicts = counts[
        counts > 1
    ]


    if len(conflicts) > 0:

        raise RuntimeError(
            f"{label}: "
            f"{len(conflicts)} keys map "
            f"to multiple entity groups."
        )


# ============================================================
# LOAD ENTITY-GROUP CONTRACT
# ============================================================

if not GROUP_FILE.exists():

    raise FileNotFoundError(
        f"Missing entity-group file: "
        f"{GROUP_FILE}"
    )


if not EDGE_FILE.exists():

    raise FileNotFoundError(
        f"Missing HARP-NOAA edge file: "
        f"{EDGE_FILE}"
    )


groups = pd.read_csv(
    GROUP_FILE
)


edges = pd.read_csv(
    EDGE_FILE
)


print(
    "===================================="
)

print(
    "PHASE 10E — ENTITY GROUP CONTRACT"
)

print(
    "===================================="
)


print(
    "Entity-group columns:",
    list(
        groups.columns
    )
)


# ============================================================
# DETECT GROUP COLUMN
# ============================================================

group_column = require_column(
    groups,
    [
        "ENTITY_GROUP_ID",
        "entity_group_id",
        "GROUP_ID",
        "group_id",
        "ENTITY_GROUP",
    ],
    "Entity-group table",
    contains="GROUP",
)


# ============================================================
# HARP MAPPING
# ============================================================

harp_column = find_column(
    groups,
    [
        "HARPNUM",
        "harpnum",
        "HARP",
    ],
    contains="HARP",
)


node_type_column = find_column(
    groups,
    [
        "NODE_TYPE",
        "node_type",
        "TYPE",
    ]
)


node_id_column = find_column(
    groups,
    [
        "NODE_ID",
        "node_id",
        "ID",
    ]
)


if harp_column is not None:

    harp_groups = groups[
        [
            harp_column,
            group_column,
        ]
    ].copy()


    harp_groups[
        harp_column
    ] = pd.to_numeric(
        harp_groups[
            harp_column
        ],
        errors="coerce"
    )


    harp_groups = harp_groups.dropna(
        subset=[
            harp_column,
            group_column,
        ]
    )


    harp_groups[
        harp_column
    ] = (
        harp_groups[
            harp_column
        ]
        .astype(int)
    )


elif (
    node_type_column is not None
    and
    node_id_column is not None
):

    harp_groups = groups[
        groups[
            node_type_column
        ]
        .astype(str)
        .str.upper()
        .str.contains(
            "HARP"
        )
    ][
        [
            node_id_column,
            group_column,
        ]
    ].copy()


    harp_groups[
        node_id_column
    ] = pd.to_numeric(
        harp_groups[
            node_id_column
        ],
        errors="coerce"
    )


    harp_groups = harp_groups.dropna()


    harp_groups[
        node_id_column
    ] = (
        harp_groups[
            node_id_column
        ]
        .astype(int)
    )


    harp_column = node_id_column


else:

    raise RuntimeError(
        "Could not construct HARP → "
        "ENTITY_GROUP_ID mapping."
    )


harp_groups[
    group_column
] = (
    harp_groups[
        group_column
    ]
    .astype(str)
)


validate_unique_mapping(
    harp_groups,
    harp_column,
    group_column,
    "HARP mapping",
)


harp_groups = (
    harp_groups
    .drop_duplicates(
        subset=[
            harp_column
        ]
    )
)


harp_to_group = dict(
    zip(
        harp_groups[
            harp_column
        ],
        harp_groups[
            group_column
        ],
    )
)


print(
    "HARP → entity groups:",
    len(
        harp_to_group
    )
)


# ============================================================
# NOAA MAPPING
# ============================================================

noaa_column = find_column(
    groups,
    [
        "NOAA_AR",
        "NOAA",
        "NOAA_REGION",
    ],
    contains="NOAA",
)


noaa_groups = None


if noaa_column is not None:

    noaa_groups = groups[
        [
            noaa_column,
            group_column,
        ]
    ].copy()


    noaa_groups[
        noaa_column
    ] = pd.to_numeric(
        noaa_groups[
            noaa_column
        ],
        errors="coerce"
    )


    noaa_groups = noaa_groups.dropna(
        subset=[
            noaa_column,
            group_column,
        ]
    )


    noaa_groups[
        noaa_column
    ] = (
        noaa_groups[
            noaa_column
        ]
        .astype(int)
    )


else:

    # --------------------------------------------------------
    # Fall back to canonical edges:
    #
    # NOAA -> HARP -> entity group
    # --------------------------------------------------------

    edge_harp = require_column(
        edges,
        [
            "HARPNUM",
            "harpnum",
        ],
        "Edge table",
        contains="HARP",
    )


    edge_noaa = require_column(
        edges,
        [
            "NOAA_AR",
            "NOAA",
        ],
        "Edge table",
        contains="NOAA",
    )


    edge_temp = edges[
        [
            edge_harp,
            edge_noaa,
        ]
    ].copy()


    edge_temp[
        edge_harp
    ] = pd.to_numeric(
        edge_temp[
            edge_harp
        ],
        errors="coerce"
    )


    edge_temp[
        edge_noaa
    ] = pd.to_numeric(
        edge_temp[
            edge_noaa
        ],
        errors="coerce"
    )


    edge_temp = edge_temp.dropna()


    edge_temp[
        edge_harp
    ] = (
        edge_temp[
            edge_harp
        ]
        .astype(int)
    )


    edge_temp[
        edge_noaa
    ] = (
        edge_temp[
            edge_noaa
        ]
        .astype(int)
    )


    edge_temp[
        group_column
    ] = edge_temp[
        edge_harp
    ].map(
        harp_to_group
    )


    noaa_groups = edge_temp[
        [
            edge_noaa,
            group_column,
        ]
    ].dropna()


    noaa_column = edge_noaa


noaa_groups[
    group_column
] = (
    noaa_groups[
        group_column
    ]
    .astype(str)
)


validate_unique_mapping(
    noaa_groups,
    noaa_column,
    group_column,
    "NOAA mapping",
)


noaa_groups = (
    noaa_groups
    .drop_duplicates(
        subset=[
            noaa_column
        ]
    )
)


noaa_to_group = dict(
    zip(
        noaa_groups[
            noaa_column
        ],
        noaa_groups[
            group_column
        ],
    )
)


print(
    "NOAA → entity groups:",
    len(
        noaa_to_group
    )
)


# ============================================================
# NOAA LIST PARSER
# ============================================================

def parse_associated_noaa_list(
    value
):

    if pd.isna(value):

        return []


    result = []


    for item in str(
        value
    ).split(","):

        item = item.strip()


        if not item:
            continue


        try:

            region = int(
                float(
                    item
                )
            )

        except Exception:

            continue


        if region > 0:

            result.append(
                region
            )


    return sorted(
        set(
            result
        )
    )


# ============================================================
# PROCESS CYCLE
# ============================================================

def process_cycle(
    cycle_key
):

    print(
        "\n===================================="
    )

    print(
        cycle_key.upper()
    )

    print(
        "===================================="
    )


    input_file = (
        INTERIM_DIR
        /
        f"{cycle_key}_qc_labeled.parquet"
    )


    output_file = (
        PROCESSED_DIR
        /
        f"{cycle_key}_model_ready.parquet"
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Missing labeled dataset: "
            f"{input_file}"
        )


    df = pd.read_parquet(
        input_file
    )


    # ========================================================
    # ONLY PRIMARY USABLE LABELS
    # ========================================================

    usable = df[
        df[
            "label_status"
        ].isin(
            [
                "POSITIVE",
                "NEGATIVE",
            ]
        )
        &
        df[
            "qc_pass_1_6"
        ]
        &
        df[
            "target"
        ].notna()
    ].copy()


    usable[
        "target"
    ] = (
        usable[
            "target"
        ]
        .astype(int)
    )


    before_group = len(
        usable
    )


    print(
        "Usable labeled rows before "
        "entity grouping:",
        before_group
    )


    # ========================================================
    # PRIMARY HARP → GROUP
    # ========================================================

    usable[
        "HARPNUM"
    ] = pd.to_numeric(
        usable[
            "HARPNUM"
        ],
        errors="coerce"
    ).astype(
        "Int64"
    )


    usable[
        "ENTITY_GROUP_ID"
    ] = (
        usable[
            "HARPNUM"
        ]
        .map(
            harp_to_group
        )
    )


    usable[
        "entity_group_source"
    ] = np.where(
        usable[
            "ENTITY_GROUP_ID"
        ].notna(),
        "HARPNUM",
        "UNRESOLVED",
    )


    mapped_by_harp = int(
        usable[
            "ENTITY_GROUP_ID"
        ]
        .notna()
        .sum()
    )


    # ========================================================
    # NOAA FALLBACK
    # ========================================================

    fallback_mapped = 0
    unresolved = 0
    ambiguous_fallback = 0
    conflicts = 0


    for index in usable.index:

        regions = (
            parse_associated_noaa_list(
                usable.at[
                    index,
                    "associated_noaa_list"
                ]
            )
        )


        noaa_groups_found = {
            noaa_to_group[
                region
            ]

            for region in regions

            if region
            in
            noaa_to_group
        }


        existing = usable.at[
            index,
            "ENTITY_GROUP_ID"
        ]


        # ----------------------------------------------------
        # Agreement audit if HARP mapping exists
        # ----------------------------------------------------

        if pd.notna(
            existing
        ):

            if (
                len(
                    noaa_groups_found
                )
                >
                0
                and
                any(
                    str(group)
                    !=
                    str(existing)

                    for group
                    in noaa_groups_found
                )
            ):

                conflicts += 1


            continue


        # ----------------------------------------------------
        # NOAA fallback
        # ----------------------------------------------------

        if len(
            noaa_groups_found
        ) == 1:

            group = next(
                iter(
                    noaa_groups_found
                )
            )


            usable.at[
                index,
                "ENTITY_GROUP_ID"
            ] = str(
                group
            )


            usable.at[
                index,
                "entity_group_source"
            ] = "NOAA_FALLBACK"


            fallback_mapped += 1


        elif len(
            noaa_groups_found
        ) > 1:

            usable.at[
                index,
                "entity_group_source"
            ] = "AMBIGUOUS"


            ambiguous_fallback += 1


        else:

            unresolved += 1


    # ========================================================
    # CANONICAL-CONTRACT CONFLICT IS FATAL
    # ========================================================

    if conflicts > 0:

        raise RuntimeError(
            f"{cycle_key}: "
            f"{conflicts} HARP/NOAA entity-group "
            f"conflicts found."
        )


    # ========================================================
    # PRIMARY MODEL DATA REQUIRES ENTITY GROUP
    # ========================================================

    model = usable[
        usable[
            "ENTITY_GROUP_ID"
        ].notna()
    ].copy()


    model[
        "ENTITY_GROUP_ID"
    ] = (
        model[
            "ENTITY_GROUP_ID"
        ]
        .astype(str)
    )


    # ========================================================
    # FINAL INTEGRITY CHECKS
    # ========================================================

    duplicate_keys = int(
        model[
            [
                "HARPNUM",
                "T_REC",
            ]
        ]
        .duplicated()
        .sum()
    )


    target_invalid = int(
        (
            ~model[
                "target"
            ]
            .isin(
                [
                    0,
                    1,
                ]
            )
        )
        .sum()
    )


    missing_groups = int(
        model[
            "ENTITY_GROUP_ID"
        ]
        .isna()
        .sum()
    )


    feature_numeric = (
        model[
            ALL_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    nonfinite_features = int(
        (
            ~np.isfinite(
                feature_numeric
                .to_numpy(
                    dtype=float
                )
            )
            .all(
                axis=1
            )
        )
        .sum()
    )


    if duplicate_keys > 0:

        raise RuntimeError(
            f"{cycle_key}: duplicate keys "
            f"in model-ready data."
        )


    if target_invalid > 0:

        raise RuntimeError(
            f"{cycle_key}: invalid targets "
            f"in model-ready data."
        )


    if missing_groups > 0:

        raise RuntimeError(
            f"{cycle_key}: missing entity "
            f"groups remain."
        )


    if nonfinite_features > 0:

        raise RuntimeError(
            f"{cycle_key}: non-finite model "
            f"features remain."
        )


    # ========================================================
    # KEEP MODEL + AUDIT METADATA
    # ========================================================

    preferred_columns = [
        "T_REC",
        "observation_time_utc",
        "HARPNUM",
        "NOAA_AR",
        "NOAA_ARS",
        "associated_noaa_list",
        "associated_noaa_count",
        "ENTITY_GROUP_ID",
        "entity_group_source",
        "cycle",
        "QUALITY",
        "CMASK",
        "LON_FWT",
        "LAT_FWT",
    ] + ALL_FEATURES + [
        "target",
        "label_status",
    ]


    output_columns = [
        column

        for column in preferred_columns

        if column
        in
        model.columns
    ]


    model = model[
        output_columns
    ].copy()


    model.to_parquet(
        output_file,
        index=False
    )


    positives = int(
        (
            model[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    negatives = int(
        (
            model[
                "target"
            ]
            ==
            0
        )
        .sum()
    )


    final_rows = len(
        model
    )


    positive_rate = (
        positives
        /
        final_rows
        *
        100
        if final_rows > 0
        else np.nan
    )


    unique_groups = (
        model[
            "ENTITY_GROUP_ID"
        ]
        .nunique()
    )


    excluded_for_group = (
        before_group
        -
        final_rows
    )


    summary = {

        "cycle":
            cycle_key,

        "usable_labels_before_group":
            before_group,

        "mapped_by_harp_initially":
            mapped_by_harp,

        "mapped_by_noaa_fallback":
            fallback_mapped,

        "ambiguous_group_fallback":
            ambiguous_fallback,

        "unresolved_group":
            unresolved,

        "harp_noaa_group_conflicts":
            conflicts,

        "excluded_for_missing_group":
            excluded_for_group,

        "final_model_rows":
            final_rows,

        "positive":
            positives,

        "negative":
            negatives,

        "positive_rate_pct":
            positive_rate,

        "unique_entity_groups":
            unique_groups,

        "duplicate_keys":
            duplicate_keys,

        "nonfinite_feature_rows":
            nonfinite_features,

        "invalid_target_rows":
            target_invalid,

        "missing_entity_group_rows":
            missing_groups,

        "output_file":
            str(
                output_file
            ),
    }


    print(
        "\n------------------------------------"
    )

    print(
        "FINAL MODEL-READY SUMMARY"
    )

    print(
        "------------------------------------"
    )


    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )


    return summary


# ============================================================
# MAIN
# ============================================================

summary_rows = []


for cycle in [
    "cycle24",
    "cycle25",
]:

    summary_rows.append(
        process_cycle(
            cycle
        )
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


print(
    "\n===================================="
)

print(
    "PHASE 10E/10F FINAL SUMMARY"
)

print(
    "===================================="
)


print(
    summary_df[
        [
            "cycle",
            "usable_labels_before_group",
            "mapped_by_harp_initially",
            "mapped_by_noaa_fallback",
            "excluded_for_missing_group",
            "final_model_rows",
            "positive",
            "negative",
            "positive_rate_pct",
            "unique_entity_groups",
            "harp_noaa_group_conflicts",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    SUMMARY_FILE
)


print(
    "\nPhase 10E/10F complete."
)