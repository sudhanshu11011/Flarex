from pathlib import Path
from datetime import timezone
import re

import numpy as np
import pandas as pd
from astropy.time import Time


# ============================================================
# PATHS
# ============================================================

INTERIM_DIR = Path(
    "data/interim"
)

GOES_DIR = Path(
    "data/raw/goes"
)

SUMMARY_FILE = Path(
    "data_manifests/phase10_label_summary.csv"
)

SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONTRACT
# ============================================================

FORECAST_HOURS = 24

C24_END_EXCLUSIVE = pd.Timestamp(
    "2020-01-01T00:00:00Z"
)

C25_END_EXCLUSIVE = pd.Timestamp(
    "2026-01-01T00:00:00Z"
)


# ============================================================
# FLARE HELPERS
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


    return (
        parsed[2]
        >=
        1e-5
    )


# ============================================================
# NOAA HELPERS
# ============================================================

def parse_noaa_values(
    noaa_ar,
    noaa_ars
):

    values = set()


    for value in [
        noaa_ar,
        noaa_ars,
    ]:

        if pd.isna(value):
            continue


        text = str(value)


        # NOAA_AR sometimes arrives as numeric-looking
        # strings such as "11166.0".
        for match in re.findall(
            r"\d+",
            text
        ):

            number = int(
                match
            )


            if number > 0:

                values.add(
                    number
                )


    return sorted(
        values
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


    if region < 10000:

        return (
            region
            +
            10000
        )


    return region


# ============================================================
# VECTOR TAI → UTC
# ============================================================

def convert_trec_tai_to_utc(
    series
):

    cleaned = (
        series
        .astype(
            "string"
        )
        .str.replace(
            "_TAI",
            "",
            regex=False
        )
        .str.replace(
            ".",
            "-",
            n=2,
            regex=False
        )
        .str.replace(
            "_",
            "T",
            n=1,
            regex=False
        )
    )


    valid_mask = (
        cleaned.notna()
        &
        (
            cleaned.str.len()
            >
            0
        )
    )


    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns, UTC]"
    )


    if valid_mask.sum() == 0:

        return result


    values = (
        cleaned[
            valid_mask
        ]
        .tolist()
    )


    times = Time(
        values,
        format="isot",
        scale="tai"
    )


    utc_values = (
        times
        .utc
        .to_datetime(
            timezone=timezone.utc
        )
    )


    result.loc[
        valid_mask
    ] = pd.to_datetime(
        utc_values,
        utc=True
    )


    return result


# ============================================================
# LOAD GOES EVENTS
# ============================================================

def load_goes(
    start_year,
    end_year
):

    frames = []


    for year in range(
        start_year,
        end_year + 1
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
                f"{year}: missing columns "
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
        "event_time_utc"
    ] = pd.to_datetime(
        goes[
            "start_time"
        ],
        utc=True,
        errors="coerce"
    )


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


    goes = goes[
        goes[
            "is_m1_plus"
        ]
        &
        goes[
            "event_time_utc"
        ].notna()
    ].copy()


    goes[
        "normalized_noaa_ar"
    ] = (
        goes[
            "active_region"
        ]
        .apply(
            normalize_goes_region
        )
    )


    return goes


# ============================================================
# BUILD EVENT INDEX
# ============================================================

def build_event_index(
    goes
):

    assigned = goes[
        goes[
            "normalized_noaa_ar"
        ].notna()
    ].copy()


    assigned[
        "normalized_noaa_ar"
    ] = (
        assigned[
            "normalized_noaa_ar"
        ]
        .astype(int)
    )


    by_region = {}


    for region, group in assigned.groupby(
        "normalized_noaa_ar"
    ):

        times = (
            group[
                "event_time_utc"
            ]
            .sort_values()
            .to_numpy(
                dtype="datetime64[ns]"
            )
        )


        by_region[
            int(region)
        ] = times


    unassigned_times = (
        goes[
            goes[
                "normalized_noaa_ar"
            ].isna()
        ][
            "event_time_utc"
        ]
        .sort_values()
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )


    return (
        by_region,
        unassigned_times,
    )


# ============================================================
# COUNT EVENTS IN (t, t+24h]
# ============================================================

def count_future_events(
    times,
    start,
    end
):

    if (
        times is None
        or
        len(times) == 0
    ):

        return 0


    start64 = np.datetime64(
        start.to_datetime64()
    )


    end64 = np.datetime64(
        end.to_datetime64()
    )


    left = np.searchsorted(
        times,
        start64,
        side="right"
    )


    right = np.searchsorted(
        times,
        end64,
        side="right"
    )


    return int(
        right
        -
        left
    )


# ============================================================
# LABEL ONE CYCLE
# ============================================================

def label_cycle(
    cycle_key,
    start_year,
    end_year,
    boundary
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
        f"{cycle_key}_merged_qc.parquet"
    )


    output_file = (
        INTERIM_DIR
        /
        f"{cycle_key}_qc_labeled.parquet"
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Missing: {input_file}"
        )


    df = pd.read_parquet(
        input_file
    )


    required = {
        "HARPNUM",
        "T_REC",
        "NOAA_AR",
        "NOAA_ARS",
        "qc_pass_1_6",
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
            f"{cycle_key}: missing columns "
            f"{sorted(missing)}"
        )


    print(
        "Rows:",
        len(df)
    )


    print(
        "QC-valid rows:",
        int(
            df[
                "qc_pass_1_6"
            ]
            .sum()
        )
    )


    # ========================================================
    # TAI → UTC
    # ========================================================

    print(
        "Converting T_REC TAI → UTC..."
    )


    df[
        "observation_time_utc"
    ] = convert_trec_tai_to_utc(
        df[
            "T_REC"
        ]
    )


    # ========================================================
    # CONTEMPORANEOUS NOAA ASSOCIATIONS
    # ========================================================

    print(
        "Parsing contemporaneous NOAA associations..."
    )


    associations = []


    for noaa_ar, noaa_ars in zip(
        df[
            "NOAA_AR"
        ],
        df[
            "NOAA_ARS"
        ]
    ):

        associations.append(
            parse_noaa_values(
                noaa_ar,
                noaa_ars
            )
        )


    df[
        "associated_noaa_list"
    ] = [
        ",".join(
            str(x)
            for x in values
        )

        for values in associations
    ]


    df[
        "associated_noaa_count"
    ] = [
        len(values)

        for values in associations
    ]


    # ========================================================
    # GOES
    # ========================================================

    print(
        "Loading GOES M1+ catalog..."
    )


    goes = load_goes(
        start_year,
        end_year
    )


    # Restrict to cycle temporal domain.
    #
    # This prevents outcomes outside the frozen
    # evaluation domain from influencing labels.
    goes = goes[
        goes[
            "event_time_utc"
        ]
        <
        boundary
    ].copy()


    (
        events_by_region,
        unassigned_times,
    ) = build_event_index(
        goes
    )


    print(
        "M1+ GOES events:",
        len(goes)
    )


    print(
        "Assigned regions in event index:",
        len(
            events_by_region
        )
    )


    print(
        "Unassigned M1+ events:",
        len(
            unassigned_times
        )
    )


    # ========================================================
    # INITIAL LABEL FIELDS
    # ========================================================

    df[
        "target"
    ] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int8"
    )


    df[
        "label_status"
    ] = "UNPROCESSED"


    df[
        "matching_assigned_m1_count"
    ] = 0


    df[
        "unassigned_m1_count"
    ] = 0


    # ========================================================
    # LABEL QC-VALID ROWS
    # ========================================================

    print(
        "Applying final label contract..."
    )


    qc_indices = df.index[
        df[
            "qc_pass_1_6"
        ]
    ]


    for counter, index in enumerate(
        qc_indices,
        start=1
    ):

        if (
            counter % 25000
            ==
            0
        ):

            print(
                "Labeled:",
                counter,
                "/",
                len(
                    qc_indices
                )
            )


        t = df.at[
            index,
            "observation_time_utc"
        ]


        # ----------------------------------------------------
        # Invalid time
        # ----------------------------------------------------

        if pd.isna(t):

            df.at[
                index,
                "label_status"
            ] = "EXCLUDE_INVALID_TIME"

            continue


        end = (
            t
            +
            pd.Timedelta(
                hours=FORECAST_HOURS
            )
        )


        # ----------------------------------------------------
        # Strict cycle-boundary isolation
        #
        # Require t+24h < boundary.
        # ----------------------------------------------------

        if end >= boundary:

            df.at[
                index,
                "label_status"
            ] = "EXCLUDE_BOUNDARY"

            continue


        regions = associations[
            index
        ]


        # ----------------------------------------------------
        # No contemporaneous NOAA association
        # ----------------------------------------------------

        if len(regions) == 0:

            df.at[
                index,
                "label_status"
            ] = "EXCLUDE_NOAA"

            continue


        # ----------------------------------------------------
        # Matching assigned M1+
        # ----------------------------------------------------

        matching_count = 0


        for region in regions:

            times = events_by_region.get(
                region
            )


            matching_count += (
                count_future_events(
                    times,
                    t,
                    end
                )
            )


        df.at[
            index,
            "matching_assigned_m1_count"
        ] = matching_count


        # Positive takes precedence.
        if matching_count > 0:

            df.at[
                index,
                "target"
            ] = 1


            df.at[
                index,
                "label_status"
            ] = "POSITIVE"

            continue


        # ----------------------------------------------------
        # Region-unassigned M1+ censoring
        # ----------------------------------------------------

        unassigned_count = (
            count_future_events(
                unassigned_times,
                t,
                end
            )
        )


        df.at[
            index,
            "unassigned_m1_count"
        ] = unassigned_count


        if unassigned_count > 0:

            df.at[
                index,
                "label_status"
            ] = "CENSORED_UNASSIGNED_FLARE"

            continue


        # ----------------------------------------------------
        # Negative
        # ----------------------------------------------------

        df.at[
            index,
            "target"
        ] = 0


        df.at[
            index,
            "label_status"
        ] = "NEGATIVE"


    # ========================================================
    # ALL NON-QC ROWS
    # ========================================================

    df.loc[
        ~df[
            "qc_pass_1_6"
        ],
        "label_status"
    ] = "EXCLUDE_QC"


    # ========================================================
    # SUMMARY
    # ========================================================

    counts = (
        df[
            "label_status"
        ]
        .value_counts()
    )


    positive = int(
        counts.get(
            "POSITIVE",
            0
        )
    )


    negative = int(
        counts.get(
            "NEGATIVE",
            0
        )
    )


    censored = int(
        counts.get(
            "CENSORED_UNASSIGNED_FLARE",
            0
        )
    )


    excluded_noaa = int(
        counts.get(
            "EXCLUDE_NOAA",
            0
        )
    )


    excluded_boundary = int(
        counts.get(
            "EXCLUDE_BOUNDARY",
            0
        )
    )


    excluded_qc = int(
        counts.get(
            "EXCLUDE_QC",
            0
        )
    )


    invalid_time = int(
        counts.get(
            "EXCLUDE_INVALID_TIME",
            0
        )
    )


    usable = (
        positive
        +
        negative
    )


    positive_rate = (
        positive
        /
        usable
        *
        100
        if usable > 0
        else np.nan
    )


    multi_noaa = int(
        (
            (
                df[
                    "associated_noaa_count"
                ]
                >
                1
            )
            &
            df[
                "qc_pass_1_6"
            ]
        )
        .sum()
    )


    qc_under_10000 = 0


    for index in qc_indices:

        values = associations[
            index
        ]


        if any(
            value < 10000
            for value in values
        ):

            qc_under_10000 += 1


    df.to_parquet(
        output_file,
        index=False
    )


    summary = {

        "cycle":
            cycle_key,

        "total_rows":
            len(df),

        "qc_pass_1_6":
            int(
                df[
                    "qc_pass_1_6"
                ]
                .sum()
            ),

        "positive":
            positive,

        "negative":
            negative,

        "usable_labels":
            usable,

        "positive_rate_pct":
            positive_rate,

        "censored_unassigned_flare":
            censored,

        "excluded_noaa":
            excluded_noaa,

        "excluded_boundary":
            excluded_boundary,

        "excluded_invalid_time":
            invalid_time,

        "excluded_qc":
            excluded_qc,

        "qc_rows_with_multiple_noaa":
            multi_noaa,

        "qc_rows_with_noaa_under_10000":
            qc_under_10000,

        "output_file":
            str(
                output_file
            ),
    }


    print(
        "\n------------------------------------"
    )

    print(
        "LABEL SUMMARY"
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

summaries = []


summaries.append(
    label_cycle(
        cycle_key="cycle24",
        start_year=2010,
        end_year=2019,
        boundary=C24_END_EXCLUSIVE,
    )
)


summaries.append(
    label_cycle(
        cycle_key="cycle25",
        start_year=2020,
        end_year=2025,
        boundary=C25_END_EXCLUSIVE,
    )
)


summary_df = pd.DataFrame(
    summaries
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


print(
    "\n===================================="
)

print(
    "PHASE 10D FINAL SUMMARY"
)

print(
    "===================================="
)


print(
    summary_df[
        [
            "cycle",
            "qc_pass_1_6",
            "positive",
            "negative",
            "usable_labels",
            "positive_rate_pct",
            "censored_unassigned_flare",
            "excluded_noaa",
            "excluded_boundary",
            "qc_rows_with_multiple_noaa",
            "qc_rows_with_noaa_under_10000",
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
    "\nPhase 10D complete."
)