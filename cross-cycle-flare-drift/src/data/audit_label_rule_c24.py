from pathlib import Path
from datetime import timezone
import re

import drms
import numpy as np
import pandas as pd
from astropy.time import Time


# ============================================================
# CONFIG
# ============================================================

client = drms.Client()

SHARP_SERIES = "hmi.sharp_cea_720s"
LORENTZ_SERIES = "cgem.lorentz"


GOES_FILE = Path(
    "data_manifests/goes_c24_label_contract_audit.csv"
)

OUTPUT_FILE = Path(
    "data_manifests/cycle24_label_rule_audit.csv"
)

SUMMARY_FILE = Path(
    "data_manifests/cycle24_label_rule_summary.csv"
)


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
    + LORENTZ_FEATURES
)


# ============================================================
# HELPER: NOAA SET PARSER
# ============================================================

def parse_noaa_set(value):

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

            value_int = int(number)

            if value_int > 0:
                result.add(
                    value_int
                )

        except ValueError:
            pass

    return result


# ============================================================
# HELPER: SHARP T_REC TAI -> UTC
# ============================================================

def sharp_trec_to_utc(t_rec):

    """
    Convert JSOC SHARP timestamp:

        2011.03.15_12:00:00_TAI

    into a timezone-aware UTC pandas Timestamp.

    Important:
    T_REC is TAI, so we use Astropy rather than treating it
    directly as UTC.
    """

    text = str(
        t_rec
    ).strip()


    # Remove explicit TAI suffix.
    text = text.replace(
        "_TAI",
        ""
    )


    # Convert:
    #
    # 2011.03.15_12:00:00
    #
    # to:
    #
    # 2011-03-15T12:00:00
    #
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


    tai_time = Time(
        text,
        format="isot",
        scale="tai"
    )


    utc_datetime = (
        tai_time
        .utc
        .to_datetime(
            timezone=timezone.utc
        )
    )


    return pd.Timestamp(
        utc_datetime
    )


# ============================================================
# LOAD GOES CATALOG AUDIT
# ============================================================

if not GOES_FILE.exists():

    raise FileNotFoundError(
        f"Missing GOES audit file: {GOES_FILE}"
    )


goes = pd.read_csv(
    GOES_FILE
)


required_goes_columns = {
    "start_time_utc",
    "flare_class",
    "is_m1plus_class",
    "has_active_region",
    "active_region_parsed",
}


missing_goes_columns = (
    required_goes_columns
    -
    set(goes.columns)
)


if missing_goes_columns:

    raise RuntimeError(
        "GOES audit file is missing columns: "
        f"{sorted(missing_goes_columns)}"
    )


# ============================================================
# NORMALIZE GOES TYPES
# ============================================================

goes["start_time_utc"] = pd.to_datetime(
    goes["start_time_utc"],
    utc=True,
    errors="coerce"
)


goes["is_m1plus_class"] = (
    goes["is_m1plus_class"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)


goes["has_active_region"] = (
    goes["has_active_region"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)


goes["active_region_parsed"] = pd.to_numeric(
    goes["active_region_parsed"],
    errors="coerce"
)


# ============================================================
# M1+ EVENTS ONLY
# ============================================================

m1plus = goes[
    goes["is_m1plus_class"]
].copy()


# ============================================================
# NORMALIZE GOES REGION NUMBER
#
# GOES Cycle-24 flare report stores AR as e.g. 1166,
# while SHARP uses full NOAA 11166.
#
# QC-8B verified:
#
# 1015 / 1015 resolvable events:
#
# full NOAA = GOES AR + 10000
#
# No suffix collisions were observed.
# ============================================================

m1plus["normalized_noaa"] = np.nan


has_region = (
    m1plus["has_active_region"]
    &
    m1plus["active_region_parsed"].notna()
)


m1plus.loc[
    has_region,
    "normalized_noaa"
] = (
    m1plus.loc[
        has_region,
        "active_region_parsed"
    ]
    .astype(int)
    + 10000
)


assigned_flares = (
    m1plus[
        has_region
    ]
    .copy()
)


unassigned_flares = (
    m1plus[
        ~has_region
    ]
    .copy()
)


print(
    "M1+ assigned events:",
    len(assigned_flares)
)

print(
    "M1+ unassigned events:",
    len(unassigned_flares)
)


# ============================================================
# CYCLE-24 AUDIT GRID
# ============================================================

DATES = []


# Cycle 24 begins during 2010.
for month in [
    6,
    9,
    12,
]:

    DATES.append(
        f"2010.{month:02d}.15_12:00:00_TAI"
    )


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
            f"{year}.{month:02d}.15_12:00:00_TAI"
        )


results = []

total_qc_valid_rows = 0
time_conversion_errors = 0


# ============================================================
# QUERY AUDIT SAMPLES
# ============================================================

for time in DATES:

    print(
        "\n===================================="
    )

    print(
        time
    )

    print(
        "===================================="
    )


    sharp_query = (
        f"{SHARP_SERIES}"
        f"[]"
        f"[{time}]"
    )


    lorentz_query = (
        f"{LORENTZ_SERIES}"
        f"[]"
        f"[{time}]"
    )


    sharp_keys = ",".join(
        [
            "T_REC",
            "HARPNUM",
            "NOAA_AR",
            "NOAA_ARS",
            "QUALITY",
            "CMASK",
            "LON_FWT",
            "LAT_FWT",
        ]
        + SHARP_FEATURES
    )


    lorentz_keys = ",".join(
        [
            "T_REC",
            "HARPNUM",
        ]
        + LORENTZ_FEATURES
    )


    try:

        sharp = client.query(
            sharp_query,
            key=sharp_keys
        )

        lorentz = client.query(
            lorentz_query,
            key=lorentz_keys
        )

    except Exception as error:

        print(
            "QUERY ERROR:",
            error
        )

        continue


    # ========================================================
    # EMPTY SHARP SNAPSHOT
    # ========================================================

    if len(sharp) == 0:

        print(
            "No SHARP rows."
        )

        continue


    # ========================================================
    # NORMALIZE SHARP METADATA TYPES
    # ========================================================

    for column in [
        "QUALITY",
        "CMASK",
        "LON_FWT",
        "LAT_FWT",
    ]:

        sharp[column] = pd.to_numeric(
            sharp[column],
            errors="coerce"
        )


    # ========================================================
    # SHARP ↔ LORENTZ MERGE
    # ========================================================

    if len(lorentz) > 0:

        merged = sharp.merge(
            lorentz,
            on=[
                "HARPNUM",
                "T_REC",
            ],
            how="left",
            validate="one_to_one",
            indicator=True,
        )

    else:

        merged = sharp.copy()


        for feature in LORENTZ_FEATURES:

            merged[
                feature
            ] = np.nan


        merged[
            "_merge"
        ] = "left_only"


    # ========================================================
    # QC-1 → QC-6
    # ========================================================

    numeric_features = (
        merged[
            ALL_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    merged[
        "features_finite"
    ] = (
        np.isfinite(
            numeric_features
        )
        .all(axis=1)
    )


    merged[
        "qc_pass"
    ] = (

        # QC-2:
        # exact SHARP-Lorentz match
        (
            merged[
                "_merge"
            ]
            ==
            "both"
        )

        &

        # QC-3
        (
            merged[
                "QUALITY"
            ]
            ==
            0
        )

        &

        # QC-4
        (
            merged[
                "CMASK"
            ]
            >
            0
        )

        &

        # valid position
        merged[
            "LON_FWT"
        ].notna()

        &

        merged[
            "LAT_FWT"
        ].notna()

        &

        # QC-5
        (
            merged[
                "LON_FWT"
            ]
            .abs()
            <=
            68
        )

        &

        # QC-6
        merged[
            "features_finite"
        ]
    )


    candidates = (
        merged[
            merged[
                "qc_pass"
            ]
        ]
        .copy()
    )


    print(
        "QC-valid rows:",
        len(candidates)
    )


    total_qc_valid_rows += len(
        candidates
    )


    # ========================================================
    # LABEL EVERY QC-VALID OBSERVATION
    # ========================================================

    for _, row in candidates.iterrows():

        # ----------------------------------------------------
        # QC-7:
        # contemporaneous NOAA association
        # ----------------------------------------------------

        associated_noaa = parse_noaa_set(
            row.get(
                "NOAA_ARS"
            )
        )


        # Add NOAA_AR too.
        try:

            primary_noaa = int(
                row[
                    "NOAA_AR"
                ]
            )


            if primary_noaa > 0:

                associated_noaa.add(
                    primary_noaa
                )


        except Exception:
            pass


        # ----------------------------------------------------
        # NO NOAA ASSOCIATION
        # ----------------------------------------------------

        if len(
            associated_noaa
        ) == 0:

            results.append(
                {
                    "T_REC":
                        row[
                            "T_REC"
                        ],

                    "HARPNUM":
                        int(
                            row[
                                "HARPNUM"
                            ]
                        ),

                    "NOAA_SET":
                        "",

                    "sample_time_utc":
                        pd.NaT,

                    "window_end_utc":
                        pd.NaT,

                    "label_status":
                        "EXCLUDE_NOAA",

                    "target":
                        np.nan,

                    "matching_m1plus":
                        0,

                    "unassigned_m1plus_in_window":
                        0,
                }
            )

            continue


        # ----------------------------------------------------
        # TAI -> UTC
        # ----------------------------------------------------

        try:

            sample_time = (
                sharp_trec_to_utc(
                    row[
                        "T_REC"
                    ]
                )
            )


        except Exception as error:

            time_conversion_errors += 1


            print(
                "TIME CONVERSION ERROR:",
                row[
                    "T_REC"
                ],
                error
            )


            results.append(
                {
                    "T_REC":
                        row[
                            "T_REC"
                        ],

                    "HARPNUM":
                        int(
                            row[
                                "HARPNUM"
                            ]
                        ),

                    "NOAA_SET":
                        ",".join(
                            map(
                                str,
                                sorted(
                                    associated_noaa
                                )
                            )
                        ),

                    "sample_time_utc":
                        pd.NaT,

                    "window_end_utc":
                        pd.NaT,

                    "label_status":
                        "TIME_CONVERSION_ERROR",

                    "target":
                        np.nan,

                    "matching_m1plus":
                        0,

                    "unassigned_m1plus_in_window":
                        0,
                }
            )

            continue


        # ----------------------------------------------------
        # 24-HOUR FUTURE WINDOW
        #
        # (t, t + 24 h]
        # ----------------------------------------------------

        window_end = (
            sample_time
            +
            pd.Timedelta(
                hours=24
            )
        )


        # ----------------------------------------------------
        # REGION-ASSIGNED M1+ FLARES IN WINDOW
        # ----------------------------------------------------

        assigned_window = assigned_flares[
            (
                assigned_flares[
                    "start_time_utc"
                ]
                >
                sample_time
            )
            &
            (
                assigned_flares[
                    "start_time_utc"
                ]
                <=
                window_end
            )
        ]


        # ----------------------------------------------------
        # REGION-UNASSIGNED M1+ FLARES IN WINDOW
        # ----------------------------------------------------

        unassigned_window = unassigned_flares[
            (
                unassigned_flares[
                    "start_time_utc"
                ]
                >
                sample_time
            )
            &
            (
                unassigned_flares[
                    "start_time_utc"
                ]
                <=
                window_end
            )
        ]


        # ----------------------------------------------------
        # MATCH FLARE REGION AGAINST SAMPLE NOAA SET
        # ----------------------------------------------------

        matching = assigned_window[
            assigned_window[
                "normalized_noaa"
            ]
            .isin(
                associated_noaa
            )
        ]


        matching_count = len(
            matching
        )


        unassigned_count = len(
            unassigned_window
        )


        # ----------------------------------------------------
        # QC-8 LABEL RULE
        #
        # Priority:
        #
        # 1. If matching M1+ exists -> POSITIVE.
        #
        # 2. Else if an unassigned M1+ exists in same future
        #    window -> CENSOR because we cannot safely prove
        #    that this active region was negative.
        #
        # 3. Else -> NEGATIVE.
        # ----------------------------------------------------

        if matching_count > 0:

            label_status = (
                "POSITIVE"
            )

            target = 1


        elif unassigned_count > 0:

            label_status = (
                "CENSORED_UNASSIGNED_FLARE"
            )

            target = np.nan


        else:

            label_status = (
                "NEGATIVE"
            )

            target = 0


        results.append(
            {
                "T_REC":
                    row[
                        "T_REC"
                    ],

                "HARPNUM":
                    int(
                        row[
                            "HARPNUM"
                        ]
                    ),

                "NOAA_SET":
                    ",".join(
                        map(
                            str,
                            sorted(
                                associated_noaa
                            )
                        )
                    ),

                "sample_time_utc":
                    sample_time,

                "window_end_utc":
                    window_end,

                "label_status":
                    label_status,

                "target":
                    target,

                "matching_m1plus":
                    matching_count,

                "unassigned_m1plus_in_window":
                    unassigned_count,
            }
        )


# ============================================================
# RESULT TABLE
# ============================================================

result = pd.DataFrame(
    results
)


if len(result) == 0:

    raise RuntimeError(
        "No label-audit rows generated."
    )


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n\n===================================="
)

print(
    "QC-8C LABEL RULE AUDIT"
)

print(
    "===================================="
)


print(
    "Total QC-valid candidates:",
    total_qc_valid_rows
)


print(
    "Rows recorded in label audit:",
    len(result)
)


print(
    "Time conversion errors:",
    time_conversion_errors
)


if (
    len(result)
    !=
    total_qc_valid_rows
):

    print(
        "WARNING:"
        " result row count does not match "
        "QC-valid candidate count."
    )


# ============================================================
# LABEL STATUS COUNTS
# ============================================================

print(
    "\nLabel status:"
)


status_counts = (
    result[
        "label_status"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    status_counts.to_string()
)


# ============================================================
# COUNTS
# ============================================================

positive = int(
    (
        result[
            "label_status"
        ]
        ==
        "POSITIVE"
    )
    .sum()
)


negative = int(
    (
        result[
            "label_status"
        ]
        ==
        "NEGATIVE"
    )
    .sum()
)


censored = int(
    (
        result[
            "label_status"
        ]
        ==
        "CENSORED_UNASSIGNED_FLARE"
    )
    .sum()
)


noaa_excluded = int(
    (
        result[
            "label_status"
        ]
        ==
        "EXCLUDE_NOAA"
    )
    .sum()
)


time_errors = int(
    (
        result[
            "label_status"
        ]
        ==
        "TIME_CONVERSION_ERROR"
    )
    .sum()
)


usable = (
    positive
    +
    negative
)


print(
    "\nPositive:",
    positive
)

print(
    "Negative:",
    negative
)

print(
    "Censored due to unassigned M1+:",
    censored
)

print(
    "Excluded because no NOAA association:",
    noaa_excluded
)

print(
    "Time conversion errors:",
    time_errors
)

print(
    "Usable labeled rows:",
    usable
)


if usable > 0:

    positive_rate = (
        positive
        /
        usable
        *
        100
    )

    print(
        "Positive rate among usable labels:",
        f"{positive_rate:.2f}%"
    )

else:

    positive_rate = np.nan


# ============================================================
# CENSORED EXAMPLES
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "CENSORED EXAMPLES"
)

print(
    "------------------------------------"
)


censored_rows = result[
    result[
        "label_status"
    ]
    ==
    "CENSORED_UNASSIGNED_FLARE"
]


if len(
    censored_rows
) == 0:

    print(
        "None."
    )

else:

    print(
        censored_rows[
            [
                "T_REC",
                "HARPNUM",
                "NOAA_SET",
                "sample_time_utc",
                "window_end_utc",
                "unassigned_m1plus_in_window",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# POSITIVE EXAMPLES
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "POSITIVE EXAMPLES"
)

print(
    "------------------------------------"
)


positive_rows = result[
    result[
        "label_status"
    ]
    ==
    "POSITIVE"
]


if len(
    positive_rows
) == 0:

    print(
        "None."
    )

else:

    print(
        positive_rows[
            [
                "T_REC",
                "HARPNUM",
                "NOAA_SET",
                "sample_time_utc",
                "window_end_utc",
                "matching_m1plus",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# NEGATIVE EXAMPLES
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "NEGATIVE EXAMPLES"
)

print(
    "------------------------------------"
)


negative_rows = result[
    result[
        "label_status"
    ]
    ==
    "NEGATIVE"
]


if len(
    negative_rows
) == 0:

    print(
        "None."
    )

else:

    print(
        negative_rows[
            [
                "T_REC",
                "HARPNUM",
                "NOAA_SET",
                "sample_time_utc",
                "window_end_utc",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE OUTPUT
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


result.to_csv(
    OUTPUT_FILE,
    index=False
)


summary = pd.DataFrame(
    [
        {
            "total_qc_valid_candidates":
                total_qc_valid_rows,

            "recorded_rows":
                len(result),

            "positive":
                positive,

            "negative":
                negative,

            "censored_unassigned":
                censored,

            "excluded_noaa":
                noaa_excluded,

            "time_conversion_errors":
                time_errors,

            "usable_labels":
                usable,

            "positive_rate_pct":
                positive_rate,
        }
    ]
)


summary.to_csv(
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
    "\nQC-8C label-rule audit finished."
)