from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = Path(
    "data/final/dataset_v1"
)

GOES_DIR = Path(
    "data/raw/goes"
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


SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "pre_model_noaa_label_sensitivity_summary.csv"
)

CHANGES_FILE = (
    MANIFEST_DIR
    /
    "pre_model_noaa_label_sensitivity_changes.csv"
)


FORECAST_HOURS = 24
MIN_GOES_CLASS = "M1.0"


# ============================================================
# HELPERS
# ============================================================

def flare_class_to_value(value):

    if pd.isna(value):
        return np.nan

    text = str(
        value
    ).strip().upper()


    match = re.match(
        r"^([ABCMX])\s*([0-9.]+)",
        text
    )


    if not match:
        return np.nan


    letter = match.group(
        1
    )


    try:

        magnitude = float(
            match.group(
                2
            )
        )

    except ValueError:

        return np.nan


    scale = {
        "A": 1e-8,
        "B": 1e-7,
        "C": 1e-6,
        "M": 1e-5,
        "X": 1e-4,
    }


    return (
        scale[
            letter
        ]
        *
        magnitude
    )


def normalize_goes_region(value):

    if pd.isna(value):
        return None


    try:

        region = int(
            float(
                value
            )
        )

    except Exception:

        return None


    if region <= 0:
        return None


    # --------------------------------------------------------
    # Audited GOES convention:
    #
    # e.g. 1166 -> NOAA 11166
    #
    # Already-full NOAA numbers remain unchanged.
    # --------------------------------------------------------

    if region < 10000:

        return (
            region
            +
            10000
        )


    return region


def normalize_sharp_primary_noaa(value):

    if pd.isna(value):
        return None


    try:

        region = int(
            float(
                value
            )
        )

    except Exception:

        return None


    if region <= 0:
        return None


    # SHARP NOAA_AR is already expected to contain
    # the full NOAA region identifier.
    return region


def find_column(
    dataframe,
    candidates,
    label
):

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate


    raise RuntimeError(
        f"Could not find {label} column. "
        f"Tried: {candidates}\n"
        f"Available columns: "
        f"{dataframe.columns.tolist()}"
    )


# ============================================================
# LOAD GOES CATALOGS
# ============================================================

print(
    "===================================="
)

print(
    "PRE-MODEL FIX A — NOAA LABEL SENSITIVITY"
)

print(
    "===================================="
)


goes_frames = []


for year in range(
    2010,
    2026
):

    file = (
        GOES_DIR
        /
        f"goes_flare_{year}.csv"
    )


    if not file.exists():

        raise FileNotFoundError(
            f"Missing GOES catalog: "
            f"{file}"
        )


    df = pd.read_csv(
        file
    )


    df[
        "_source_year"
    ] = year


    goes_frames.append(
        df
    )


if not goes_frames:

    raise RuntimeError(
        "No GOES catalogs loaded."
    )


goes = pd.concat(
    goes_frames,
    ignore_index=True
)


print(
    "GOES rows loaded:",
    len(
        goes
    )
)


# ============================================================
# DETECT GOES COLUMN NAMES
# ============================================================

start_col = find_column(
    goes,
    [
        "start_time",
        "start",
        "event_start_time",
    ],
    "GOES start_time",
)


# Actual downloaded schema uses:
#
# flare_class
#
class_col = find_column(
    goes,
    [
        "flare_class",
        "class",
        "class_type",
        "goes_class",
        "xray_class",
    ],
    "GOES flare class",
)


region_col = find_column(
    goes,
    [
        "active_region",
        "noaa_active_region",
        "region",
    ],
    "GOES active region",
)


print(
    "GOES start column:",
    start_col
)

print(
    "GOES flare-class column:",
    class_col
)

print(
    "GOES region column:",
    region_col
)


# ============================================================
# CLEAN GOES CATALOG
# ============================================================

goes[
    "_start_time"
] = pd.to_datetime(
    goes[
        start_col
    ],
    utc=True,
    errors="coerce"
)


goes[
    "_class_value"
] = goes[
    class_col
].apply(
    flare_class_to_value
)


M1_THRESHOLD = flare_class_to_value(
    MIN_GOES_CLASS
)


if pd.isna(
    M1_THRESHOLD
):

    raise RuntimeError(
        "Could not parse M1.0 threshold."
    )


goes[
    "_normalized_region"
] = goes[
    region_col
].apply(
    normalize_goes_region
)


m1plus = goes[
    (
        goes[
            "_start_time"
        ]
        .notna()
    )
    &
    (
        goes[
            "_class_value"
        ]
        >=
        M1_THRESHOLD
    )
].copy()


m1plus = (
    m1plus
    .sort_values(
        "_start_time"
    )
    .reset_index(
        drop=True
    )
)


assigned_events = m1plus[
    m1plus[
        "_normalized_region"
    ]
    .notna()
].copy()


unassigned_events = m1plus[
    m1plus[
        "_normalized_region"
    ]
    .isna()
].copy()


print(
    "M1+ events:",
    len(
        m1plus
    )
)


print(
    "Assigned M1+ events:",
    len(
        assigned_events
    )
)


print(
    "Unassigned M1+ events:",
    len(
        unassigned_events
    )
)


# ============================================================
# PRECOMPUTE EVENT ARRAYS
# ============================================================

assigned_times = np.array(
    [
        timestamp.value
        for timestamp
        in assigned_events[
            "_start_time"
        ]
    ],
    dtype=np.int64,
)


assigned_regions = (
    assigned_events[
        "_normalized_region"
    ]
    .astype(int)
    .to_numpy()
)

unassigned_times = np.array(
    [
        timestamp.value
        for timestamp
        in unassigned_events[
            "_start_time"
        ]
    ],
    dtype=np.int64,
)


# ============================================================
# SENSITIVITY LABEL FUNCTION
# ============================================================
#
# Sensitivity rule:
#
# Use ONLY NOAA_AR as the region association.
#
# Everything else remains identical to the frozen
# label contract:
#
# - threshold = M1.0+
# - event timestamp = start_time
# - forecast window = (t, t+24h]
# - matching assigned event => positive
# - if no matching assigned positive but an unassigned
#   M1+ event occurs => censor
# - otherwise => negative
#
# ============================================================

def sensitivity_label(
    observation_time,
    primary_noaa
):

    if pd.isna(
        observation_time
    ):

        return (
            "INVALID_TIME",
            np.nan,
        )


    if primary_noaa is None:

        return (
            "EXCLUDE_NOAA",
            np.nan,
        )


    start_ns = int(
        observation_time.value
    )


    end_time = (
        observation_time
        +
        pd.Timedelta(
            hours=FORECAST_HOURS
        )
    )


    end_ns = int(
        end_time.value
    )


    # --------------------------------------------------------
    # Assigned M1+ events in:
    #
    # (t, t+24h]
    # --------------------------------------------------------

    left = np.searchsorted(
        assigned_times,
        start_ns,
        side="right"
    )


    right = np.searchsorted(
        assigned_times,
        end_ns,
        side="right"
    )


    if right > left:

        window_regions = assigned_regions[
            left:right
        ]


        if np.any(
            window_regions
            ==
            primary_noaa
        ):

            return (
                "POSITIVE",
                1,
            )


    # --------------------------------------------------------
    # Assigned positive takes precedence.
    #
    # Only if no assigned matching positive was found,
    # test whether any region-unassigned M1+ event
    # censors this sample.
    # --------------------------------------------------------

    u_left = np.searchsorted(
        unassigned_times,
        start_ns,
        side="right"
    )


    u_right = np.searchsorted(
        unassigned_times,
        end_ns,
        side="right"
    )


    if u_right > u_left:

        return (
            "CENSORED_UNASSIGNED_M1PLUS",
            np.nan,
        )


    return (
        "NEGATIVE",
        0,
    )


# ============================================================
# AUDIT EACH FROZEN CYCLE
# ============================================================

summary_rows = []
change_rows = []


datasets = {

    "cycle24":
        C24_FILE,

    "cycle25":
        C25_FILE,
}


for cycle, file in datasets.items():

    print(
        "\n===================================="
    )

    print(
        cycle.upper()
    )

    print(
        "===================================="
    )


    if not file.exists():

        raise FileNotFoundError(
            f"Missing frozen dataset: "
            f"{file}"
        )


    df = pd.read_parquet(
        file
    )


    required_columns = {
        "HARPNUM",
        "T_REC",
        "observation_time_utc",
        "NOAA_AR",
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
            f"{cycle} missing required columns: "
            f"{sorted(missing)}"
        )


    observation_times = pd.to_datetime(
        df[
            "observation_time_utc"
        ],
        utc=True,
        errors="coerce"
    )


    primary_regions = df[
        "NOAA_AR"
    ].apply(
        normalize_sharp_primary_noaa
    )


    sensitivity_status = []
    sensitivity_target = []


    total_rows = len(
        df
    )


    for index, (
        observation_time,
        primary_noaa,
    ) in enumerate(
        zip(
            observation_times,
            primary_regions,
        ),
        start=1
    ):

        status, target = sensitivity_label(
            observation_time,
            primary_noaa,
        )


        sensitivity_status.append(
            status
        )


        sensitivity_target.append(
            target
        )


        if (
            index % 20000 == 0
            or
            index == total_rows
        ):

            print(
                "Processed:",
                index,
                "/",
                total_rows
            )


    # ========================================================
    # BUILD AUDIT TABLE
    # ========================================================

    audit_columns = [
        "HARPNUM",
        "T_REC",
        "observation_time_utc",
        "NOAA_AR",
        "ENTITY_GROUP_ID",
        "target",
    ]


    audit = df[
        audit_columns
    ].copy()


    if "NOAA_ARS" in df.columns:

        audit[
            "NOAA_ARS"
        ] = df[
            "NOAA_ARS"
        ]


    audit[
        "cycle"
    ] = cycle


    audit[
        "current_target"
    ] = pd.to_numeric(
        audit[
            "target"
        ],
        errors="coerce"
    )


    audit[
        "sensitivity_status"
    ] = sensitivity_status


    audit[
        "sensitivity_target"
    ] = sensitivity_target


    # ========================================================
    # COMPARISON CATEGORIES
    # ========================================================

    audit[
        "comparison"
    ] = "UNCHANGED"


    became_unusable = (
        audit[
            "sensitivity_target"
        ]
        .isna()
    )


    audit.loc[
        became_unusable,
        "comparison"
    ] = "BECOMES_CENSORED_OR_EXCLUDED"


    both_numeric = (
        audit[
            "sensitivity_target"
        ]
        .notna()
    )


    positive_to_negative = (
        both_numeric
        &
        (
            audit[
                "current_target"
            ]
            ==
            1
        )
        &
        (
            audit[
                "sensitivity_target"
            ]
            ==
            0
        )
    )


    negative_to_positive = (
        both_numeric
        &
        (
            audit[
                "current_target"
            ]
            ==
            0
        )
        &
        (
            audit[
                "sensitivity_target"
            ]
            ==
            1
        )
    )


    audit.loc[
        positive_to_negative,
        "comparison"
    ] = "POSITIVE_TO_NEGATIVE"


    audit.loc[
        negative_to_positive,
        "comparison"
    ] = "NEGATIVE_TO_POSITIVE"


    changed = audit[
        audit[
            "comparison"
        ]
        !=
        "UNCHANGED"
    ].copy()


    change_rows.append(
        changed
    )


    # ========================================================
    # COUNTS
    # ========================================================

    current_positive = int(
        (
            audit[
                "current_target"
            ]
            ==
            1
        )
        .sum()
    )


    current_negative = int(
        (
            audit[
                "current_target"
            ]
            ==
            0
        )
        .sum()
    )


    sensitivity_positive = int(
        (
            audit[
                "sensitivity_target"
            ]
            ==
            1
        )
        .sum()
    )


    sensitivity_negative = int(
        (
            audit[
                "sensitivity_target"
            ]
            ==
            0
        )
        .sum()
    )


    sensitivity_censored = int(
        (
            audit[
                "sensitivity_status"
            ]
            ==
            "CENSORED_UNASSIGNED_M1PLUS"
        )
        .sum()
    )


    sensitivity_excluded_noaa = int(
        (
            audit[
                "sensitivity_status"
            ]
            ==
            "EXCLUDE_NOAA"
        )
        .sum()
    )


    sensitivity_invalid_time = int(
        (
            audit[
                "sensitivity_status"
            ]
            ==
            "INVALID_TIME"
        )
        .sum()
    )


    pos_to_neg = int(
        positive_to_negative.sum()
    )


    neg_to_pos = int(
        negative_to_positive.sum()
    )


    unchanged = int(
        (
            audit[
                "comparison"
            ]
            ==
            "UNCHANGED"
        )
        .sum()
    )


    total_changed = int(
        len(
            changed
        )
    )


    changed_pct = (
        total_changed
        /
        len(
            audit
        )
        *
        100
        if len(
            audit
        )
        else np.nan
    )


    affected_harps = int(
        changed[
            "HARPNUM"
        ]
        .nunique()
    )


    affected_groups = int(
        changed[
            "ENTITY_GROUP_ID"
        ]
        .nunique()
    )


    summary_rows.append(
        {
            "cycle":
                cycle,

            "rows":
                len(
                    audit
                ),

            "current_positive":
                current_positive,

            "current_negative":
                current_negative,

            "sensitivity_positive":
                sensitivity_positive,

            "sensitivity_negative":
                sensitivity_negative,

            "sensitivity_censored":
                sensitivity_censored,

            "sensitivity_excluded_noaa":
                sensitivity_excluded_noaa,

            "sensitivity_invalid_time":
                sensitivity_invalid_time,

            "positive_to_negative":
                pos_to_neg,

            "negative_to_positive":
                neg_to_pos,

            "unchanged":
                unchanged,

            "total_changed_or_unusable":
                total_changed,

            "changed_pct":
                changed_pct,

            "affected_harpnums":
                affected_harps,

            "affected_entity_groups":
                affected_groups,
        }
    )


    print(
        "Rows:",
        len(
            audit
        )
    )


    print(
        "Current positives:",
        current_positive
    )


    print(
        "Current negatives:",
        current_negative
    )


    print(
        "Sensitivity positives:",
        sensitivity_positive
    )


    print(
        "Sensitivity negatives:",
        sensitivity_negative
    )


    print(
        "Positive -> Negative:",
        pos_to_neg
    )


    print(
        "Negative -> Positive:",
        neg_to_pos
    )


    print(
        "Becomes censored:",
        sensitivity_censored
    )


    print(
        "Becomes excluded (NOAA_AR missing):",
        sensitivity_excluded_noaa
    )


    print(
        "Invalid observation time:",
        sensitivity_invalid_time
    )


    print(
        "Total changed/unusable:",
        total_changed
    )


    print(
        "Changed percentage:",
        f"{changed_pct:.6f}%"
    )


    print(
        "Affected HARPs:",
        affected_harps
    )


    print(
        "Affected entity groups:",
        affected_groups
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


if change_rows:

    changes = pd.concat(
        change_rows,
        ignore_index=True
    )


else:

    changes = pd.DataFrame()


MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


changes.to_csv(
    CHANGES_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "NOAA LABEL SENSITIVITY SUMMARY"
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
    "SCIENTIFIC DECISION GATE"
)

print(
    "===================================="
)


total_rows = int(
    summary[
        "rows"
    ].sum()
)


total_changed = int(
    summary[
        "total_changed_or_unusable"
    ].sum()
)


overall_changed_pct = (
    total_changed
    /
    total_rows
    *
    100
    if total_rows
    else np.nan
)


total_pos_to_neg = int(
    summary[
        "positive_to_negative"
    ].sum()
)


total_neg_to_pos = int(
    summary[
        "negative_to_positive"
    ].sum()
)


total_censored = int(
    summary[
        "sensitivity_censored"
    ].sum()
)


total_excluded = int(
    summary[
        "sensitivity_excluded_noaa"
    ].sum()
)


print(
    "Total frozen rows:",
    total_rows
)


print(
    "Rows changed/unusable:",
    total_changed
)


print(
    "Overall affected percentage:",
    f"{overall_changed_pct:.6f}%"
)


print(
    "Total positive -> negative:",
    total_pos_to_neg
)


print(
    "Total negative -> positive:",
    total_neg_to_pos
)


print(
    "Total sensitivity censored:",
    total_censored
)


print(
    "Total sensitivity NOAA-excluded:",
    total_excluded
)


print(
    "\nSaved:"
)


print(
    SUMMARY_FILE
)


print(
    CHANGES_FILE
)


print(
    "\nPre-model NOAA sensitivity audit complete."
)