from pathlib import Path
import re

import pandas as pd


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


DETAIL_FILE = (
    MANIFEST_DIR
    /
    "pre_model_secondary_noaa_positive_events.csv"
)

SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "pre_model_secondary_noaa_positive_summary.csv"
)


FORECAST_HOURS = 24


# ============================================================
# HELPERS
# ============================================================

def parse_region(value):

    if pd.isna(value):
        return None

    try:

        region = int(
            float(value)
        )

    except Exception:

        return None

    if region <= 0:
        return None

    return region


def parse_region_list(value):

    if pd.isna(value):
        return []

    numbers = re.findall(
        r"\d+",
        str(value)
    )

    result = []

    for number in numbers:

        region = int(
            number
        )

        if region > 0:

            result.append(
                region
            )

    return sorted(
        set(
            result
        )
    )


def normalize_goes_region(value):

    region = parse_region(
        value
    )

    if region is None:
        return None

    if region < 10000:

        return (
            region
            +
            10000
        )

    return region


def is_m1plus(value):

    if pd.isna(value):
        return False

    text = str(
        value
    ).strip().upper()

    match = re.match(
        r"^([ABCMX])([0-9.]+)",
        text
    )

    if not match:
        return False

    letter = match.group(
        1
    )

    magnitude = float(
        match.group(
            2
        )
    )

    if letter == "X":
        return True

    if (
        letter == "M"
        and
        magnitude >= 1.0
    ):
        return True

    return False


# ============================================================
# LOAD GOES
# ============================================================

print(
    "===================================="
)

print(
    "PRE-MODEL FIX C — SECONDARY NOAA POSITIVE AUDIT"
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
            f"{file} missing columns: "
            f"{sorted(missing)}"
        )


    goes_frames.append(
        df
    )


goes = pd.concat(
    goes_frames,
    ignore_index=True
)


goes[
    "_event_time"
] = pd.to_datetime(
    goes[
        "start_time"
    ],
    utc=True,
    errors="coerce"
)


goes[
    "_region"
] = goes[
    "active_region"
].apply(
    normalize_goes_region
)


goes[
    "_m1plus"
] = goes[
    "flare_class"
].apply(
    is_m1plus
)


events = goes[
    (
        goes[
            "_m1plus"
        ]
    )
    &
    (
        goes[
            "_event_time"
        ]
        .notna()
    )
    &
    (
        goes[
            "_region"
        ]
        .notna()
    )
].copy()


events[
    "_region"
] = events[
    "_region"
].astype(
    int
)


events = events.sort_values(
    "_event_time"
).reset_index(
    drop=True
)


print(
    "Assigned M1+ events:",
    len(
        events
    )
)


# ============================================================
# AUDIT POSITIVE ROWS
# ============================================================

detail_rows = []


datasets = {

    "cycle24":
        C24_FILE,

    "cycle25":
        C25_FILE,
}


for cycle, path in datasets.items():

    df = pd.read_parquet(
        path
    )


    required = {
        "HARPNUM",
        "T_REC",
        "observation_time_utc",
        "NOAA_AR",
        "NOAA_ARS",
        "ENTITY_GROUP_ID",
        "target",
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
            f"{cycle} missing columns: "
            f"{sorted(missing)}"
        )


    positives = df[
        df[
            "target"
        ]
        ==
        1
    ].copy()


    print(
        "\n",
        cycle,
        "positive rows:",
        len(
            positives
        )
    )


    for index, row in positives.iterrows():

        observation_time = pd.to_datetime(
            row[
                "observation_time_utc"
            ],
            utc=True,
            errors="coerce"
        )


        if pd.isna(
            observation_time
        ):

            continue


        window_end = (
            observation_time
            +
            pd.Timedelta(
                hours=FORECAST_HOURS
            )
        )


        primary_region = parse_region(
            row[
                "NOAA_AR"
            ]
        )


        all_regions = parse_region_list(
            row[
                "NOAA_ARS"
            ]
        )


        window = events[
            (
                events[
                    "_event_time"
                ]
                >
                observation_time
            )
            &
            (
                events[
                    "_event_time"
                ]
                <=
                window_end
            )
            &
            (
                events[
                    "_region"
                ]
                .isin(
                    all_regions
                )
            )
        ]


        for _, event in window.iterrows():

            event_region = int(
                event[
                    "_region"
                ]
            )


            relation = (
                "PRIMARY_NOAA"
                if event_region
                ==
                primary_region
                else
                "SECONDARY_NOAA"
            )


            detail_rows.append(
                {
                    "cycle":
                        cycle,

                    "HARPNUM":
                        int(
                            row[
                                "HARPNUM"
                            ]
                        ),

                    "ENTITY_GROUP_ID":
                        row[
                            "ENTITY_GROUP_ID"
                        ],

                    "T_REC":
                        row[
                            "T_REC"
                        ],

                    "observation_time_utc":
                        observation_time,

                    "NOAA_AR":
                        primary_region,

                    "NOAA_ARS":
                        row[
                            "NOAA_ARS"
                        ],

                    "event_time_utc":
                        event[
                            "_event_time"
                        ],

                    "event_flare_class":
                        event[
                            "flare_class"
                        ],

                    "event_noaa_region":
                        event_region,

                    "event_relation":
                        relation,
                }
            )


detail = pd.DataFrame(
    detail_rows
)


if len(
    detail
) == 0:

    raise RuntimeError(
        "No positive-event matches found."
    )


# ============================================================
# ROW-LEVEL PRIMARY / SECONDARY CLASSIFICATION
# ============================================================

key_columns = [
    "cycle",
    "HARPNUM",
    "T_REC",
]


row_flags = (
    detail
    .groupby(
        key_columns
    )[
        "event_relation"
    ]
    .agg(
        lambda values:
            set(values)
    )
    .reset_index()
)


def classify_relation(values):

    has_primary = (
        "PRIMARY_NOAA"
        in
        values
    )

    has_secondary = (
        "SECONDARY_NOAA"
        in
        values
    )


    if (
        has_primary
        and
        has_secondary
    ):

        return (
            "PRIMARY_AND_SECONDARY"
        )


    if has_primary:

        return (
            "PRIMARY_ONLY"
        )


    if has_secondary:

        return (
            "SECONDARY_ONLY"
        )


    return (
        "NO_MATCH"
    )


row_flags[
    "positive_source"
] = row_flags[
    "event_relation"
].apply(
    classify_relation
)


detail = detail.merge(
    row_flags[
        key_columns
        +
        [
            "positive_source"
        ]
    ],
    on=key_columns,
    how="left",
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for cycle in [
    "cycle24",
    "cycle25",
]:

    subset = row_flags[
        row_flags[
            "cycle"
        ]
        ==
        cycle
    ]


    counts = (
        subset[
            "positive_source"
        ]
        .value_counts()
    )


    summary_rows.append(
        {
            "cycle":
                cycle,

            "positive_rows_with_event_match":
                len(
                    subset
                ),

            "primary_only":
                int(
                    counts.get(
                        "PRIMARY_ONLY",
                        0
                    )
                ),

            "secondary_only":
                int(
                    counts.get(
                        "SECONDARY_ONLY",
                        0
                    )
                ),

            "primary_and_secondary":
                int(
                    counts.get(
                        "PRIMARY_AND_SECONDARY",
                        0
                    )
                ),

            "secondary_only_pct":
                (
                    counts.get(
                        "SECONDARY_ONLY",
                        0
                    )
                    /
                    len(
                        subset
                    )
                    *
                    100
                    if len(
                        subset
                    )
                    else 0
                ),

            "affected_harps_secondary_only":
                int(
                    subset[
                        subset[
                            "positive_source"
                        ]
                        ==
                        "SECONDARY_ONLY"
                    ][
                        "HARPNUM"
                    ]
                    .nunique()
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE
# ============================================================

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


detail.to_csv(
    DETAIL_FILE,
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
    "\n===================================="
)

print(
    "SECONDARY NOAA POSITIVE SUMMARY"
)

print(
    "===================================="
)


print(
    summary.to_string(
        index=False
    )
)


secondary_only = detail[
    detail[
        "positive_source"
    ]
    ==
    "SECONDARY_ONLY"
]


print(
    "\nUnique secondary NOAA regions involved:",
    secondary_only[
        "event_noaa_region"
    ]
    .nunique()
)


print(
    "Unique HARPs involved:",
    secondary_only[
        "HARPNUM"
    ]
    .nunique()
)


print(
    "\nSaved:"
)

print(
    DETAIL_FILE
)

print(
    SUMMARY_FILE
)


print(
    "\nPre-model Fix C complete."
)