from pathlib import Path
import re
import time

import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data_manifests/pre_model_secondary_noaa_positive_events.csv"
)

SRS_CACHE_DIR = Path(
    "data/raw/srs/bracketing"
)

MANIFEST_DIR = Path(
    "data_manifests"
)


EVENT_OUTPUT = (
    MANIFEST_DIR
    /
    "pre_model_srs_all_positive_event_audit.csv"
)

ROW_OUTPUT = (
    MANIFEST_DIR
    /
    "pre_model_srs_all_positive_row_audit.csv"
)

SUMMARY_OUTPUT = (
    MANIFEST_DIR
    /
    "pre_model_srs_all_positive_summary.csv"
)


BASE_URL = (
    "https://www.ngdc.noaa.gov/"
    "stp/space-weather/swpc-products/"
    "daily_reports/solar_region_summaries"
)


REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 0.05


# ============================================================
# HELPERS
# ============================================================

def build_srs_url(report_date):

    report_date = pd.Timestamp(
        report_date
    )

    year = report_date.strftime(
        "%Y"
    )

    month = report_date.strftime(
        "%m"
    )

    filename = (
        report_date.strftime(
            "%Y%m%d"
        )
        +
        "SRS.txt"
    )

    return (
        f"{BASE_URL}/"
        f"{year}/"
        f"{month}/"
        f"{filename}"
    )


def cache_path(report_date):

    report_date = pd.Timestamp(
        report_date
    )

    filename = (
        report_date.strftime(
            "%Y%m%d"
        )
        +
        "_SRS.txt"
    )

    return (
        SRS_CACHE_DIR
        /
        filename
    )


def full_noaa_to_srs_display(full_noaa):

    # ONLY for matching four-digit SRS display.
    # Do not reuse as GOES normalization.

    return int(
        full_noaa
    ) % 10000


def parse_srs_regions(text):

    regions = set()

    pattern = re.compile(
        r"^\s*"
        r"(?P<number>\d{4})"
        r"\s+"
        r"[NS]\d{2}[EW]\d{2,3}"
        r"\s+"
        r"\d{3}"
        r"(?:\s+.*)?$"
    )


    for line in text.splitlines():

        match = pattern.match(
            line
        )

        if match:

            regions.add(
                int(
                    match.group(
                        "number"
                    )
                )
            )

    return regions


def download_or_read_srs(
    report_date,
    session,
):

    report_date = pd.Timestamp(
        report_date
    ).normalize()


    path = cache_path(
        report_date
    )


    if path.exists():

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return {
            "status":
                "OK_CACHED",

            "text":
                text,

            "url":
                build_srs_url(
                    report_date
                ),
        }


    url = build_srs_url(
        report_date
    )


    try:

        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException as exc:

        return {
            "status":
                "REQUEST_ERROR",

            "text":
                None,

            "url":
                url,

            "error":
                str(
                    exc
                ),
        }


    if response.status_code != 200:

        return {
            "status":
                f"HTTP_{response.status_code}",

            "text":
                None,

            "url":
                url,
        }


    text = response.text


    if (
        "Solar Region Summary"
        not in text
        and
        "SOLAR REGION SUMMARY"
        not in text
        and
        "Joint USAF/NOAA"
        not in text
    ):

        return {
            "status":
                "INVALID_CONTENT",

            "text":
                None,

            "url":
                url,
        }


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    path.write_text(
        text,
        encoding="utf-8",
    )


    time.sleep(
        REQUEST_DELAY_SECONDS
    )


    return {
        "status":
            "OK_DOWNLOADED",

        "text":
            text,

        "url":
            url,
    }


def classify_bracket(
    present_start,
    present_end,
    start_report_ok,
    end_report_ok,
):

    if (
        not start_report_ok
        or
        not end_report_ok
    ):

        return (
            "REPORT_MISSING_OR_INVALID"
        )


    if (
        present_start
        and
        present_end
    ):

        return (
            "BRACKETED_PRESENT"
        )


    if (
        present_start
        and
        not present_end
    ):

        return (
            "START_ONLY"
        )


    if (
        not present_start
        and
        present_end
    ):

        return (
            "END_ONLY"
        )


    return (
        "ABSENT_BOTH"
    )


# ============================================================
# LOAD ALL POSITIVE EVENT MATCHES
# ============================================================

print(
    "===================================="
)

print(
    "PRE-MODEL FIX H — ALL POSITIVE SRS AUDIT"
)

print(
    "===================================="
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Missing Fix C output: "
        f"{INPUT_FILE}"
    )


detail = pd.read_csv(
    INPUT_FILE
)


required_columns = {
    "cycle",
    "HARPNUM",
    "T_REC",
    "ENTITY_GROUP_ID",
    "observation_time_utc",
    "event_time_utc",
    "event_noaa_region",
    "event_relation",
    "positive_source",
}


missing = (
    required_columns
    -
    set(
        detail.columns
    )
)


if missing:

    raise RuntimeError(
        "Missing columns: "
        f"{sorted(missing)}"
    )


# ============================================================
# CLEAN
# ============================================================

detail[
    "observation_time_utc"
] = pd.to_datetime(
    detail[
        "observation_time_utc"
    ],
    utc=True,
    errors="coerce",
)


detail[
    "event_time_utc"
] = pd.to_datetime(
    detail[
        "event_time_utc"
    ],
    utc=True,
    errors="coerce",
)


detail[
    "event_noaa_region"
] = pd.to_numeric(
    detail[
        "event_noaa_region"
    ],
    errors="coerce",
)


detail = detail[
    detail[
        "observation_time_utc"
    ].notna()
    &
    detail[
        "event_time_utc"
    ].notna()
    &
    detail[
        "event_noaa_region"
    ].notna()
].copy()


detail[
    "event_noaa_region"
] = detail[
    "event_noaa_region"
].astype(int)


# Remove accidental duplicate event matches if any.
detail = detail.drop_duplicates(
    subset=[
        "cycle",
        "HARPNUM",
        "T_REC",
        "event_time_utc",
        "event_noaa_region",
        "event_relation",
    ]
).copy()


positive_rows = (
    detail[
        [
            "cycle",
            "HARPNUM",
            "T_REC",
        ]
    ]
    .drop_duplicates()
)


print(
    "Positive event records:",
    len(
        detail
    )
)


print(
    "Unique positive sample rows:",
    len(
        positive_rows
    )
)


print(
    "Expected Dataset v1 positives:",
    3620 + 5216
)


if len(
    positive_rows
) != (
    3620
    +
    5216
):

    raise RuntimeError(
        "Positive-row count does not reconcile "
        "with Dataset v1."
    )


# ============================================================
# OBSERVATION DATES
# ============================================================

detail[
    "_observation_date"
] = (
    detail[
        "observation_time_utc"
    ]
    .dt
    .tz_convert(
        "UTC"
    )
    .dt
    .normalize()
    .dt
    .tz_localize(
        None
    )
)


required_report_dates = set()


for date in detail[
    "_observation_date"
].unique():

    date = pd.Timestamp(
        date
    )

    required_report_dates.add(
        date
    )

    required_report_dates.add(
        date
        +
        pd.Timedelta(
            days=1
        )
    )


required_report_dates = sorted(
    required_report_dates
)


print(
    "Required SRS reports:",
    len(
        required_report_dates
    )
)


# ============================================================
# DOWNLOAD / CACHE SRS
# ============================================================

session = requests.Session()

report_cache = {}


for index, report_date in enumerate(
    required_report_dates,
    start=1,
):

    result = download_or_read_srs(
        report_date,
        session,
    )


    status = result[
        "status"
    ]


    regions = set()


    if status in {
        "OK_CACHED",
        "OK_DOWNLOADED",
    }:

        regions = parse_srs_regions(
            result[
                "text"
            ]
        )


    report_cache[
        pd.Timestamp(
            report_date
        )
    ] = {
        "status":
            status,

        "regions":
            regions,

        "url":
            result[
                "url"
            ],
    }


    if (
        index % 50 == 0
        or
        index == len(
            required_report_dates
        )
    ):

        print(
            "SRS reports processed:",
            index,
            "/",
            len(
                required_report_dates
            )
        )


# ============================================================
# REPORT STATUS
# ============================================================

status_counts = {}


for report in report_cache.values():

    status = report[
        "status"
    ]

    status_counts[
        status
    ] = (
        status_counts.get(
            status,
            0
        )
        +
        1
    )


print(
    "\nSRS REPORT STATUS"
)


for status, count in sorted(
    status_counts.items()
):

    print(
        status,
        ":",
        count
    )


# ============================================================
# EVENT-LEVEL AUDIT
# ============================================================

event_rows = []


for _, row in detail.iterrows():

    observation_date = pd.Timestamp(
        row[
            "_observation_date"
        ]
    )


    start_report_date = (
        observation_date
    )


    end_report_date = (
        observation_date
        +
        pd.Timedelta(
            days=1
        )
    )


    start_report = report_cache[
        start_report_date
    ]


    end_report = report_cache[
        end_report_date
    ]


    full_noaa = int(
        row[
            "event_noaa_region"
        ]
    )


    display_noaa = (
        full_noaa_to_srs_display(
            full_noaa
        )
    )


    start_ok = (
        start_report[
            "status"
        ]
        in {
            "OK_CACHED",
            "OK_DOWNLOADED",
        }
    )


    end_ok = (
        end_report[
            "status"
        ]
        in {
            "OK_CACHED",
            "OK_DOWNLOADED",
        }
    )


    present_start = (
        start_ok
        and
        display_noaa
        in
        start_report[
            "regions"
        ]
    )


    present_end = (
        end_ok
        and
        display_noaa
        in
        end_report[
            "regions"
        ]
    )


    bracket_status = classify_bracket(
        present_start,
        present_end,
        start_ok,
        end_ok,
    )


    event_rows.append(
        {
            "cycle":
                row[
                    "cycle"
                ],

            "HARPNUM":
                int(
                    row[
                        "HARPNUM"
                    ]
                ),

            "T_REC":
                row[
                    "T_REC"
                ],

            "ENTITY_GROUP_ID":
                row[
                    "ENTITY_GROUP_ID"
                ],

            "observation_time_utc":
                row[
                    "observation_time_utc"
                ],

            "positive_source":
                row[
                    "positive_source"
                ],

            "event_relation":
                row[
                    "event_relation"
                ],

            "event_time_utc":
                row[
                    "event_time_utc"
                ],

            "event_noaa_region":
                full_noaa,

            "srs_display_region":
                f"{display_noaa:04d}",

            "start_report_date":
                start_report_date.date(),

            "end_report_date":
                end_report_date.date(),

            "start_report_status":
                start_report[
                    "status"
                ],

            "end_report_status":
                end_report[
                    "status"
                ],

            "present_start":
                present_start,

            "present_end":
                present_end,

            "bracket_status":
                bracket_status,

            "start_source_url":
                start_report[
                    "url"
                ],

            "end_source_url":
                end_report[
                    "url"
                ],
        }
    )


event_audit = pd.DataFrame(
    event_rows
)


# ============================================================
# ROW-LEVEL DECISION
# ============================================================

key_columns = [
    "cycle",
    "HARPNUM",
    "T_REC",
]


row_rows = []


for keys, group in event_audit.groupby(
    key_columns
):

    cycle, harpnum, t_rec = keys


    statuses = set(
        group[
            "bracket_status"
        ]
    )


    supported = group[
        group[
            "bracket_status"
        ]
        ==
        "BRACKETED_PRESENT"
    ]


    supported_primary = int(
        (
            supported[
                "event_relation"
            ]
            ==
            "PRIMARY_NOAA"
        )
        .sum()
    )


    supported_secondary = int(
        (
            supported[
                "event_relation"
            ]
            ==
            "SECONDARY_NOAA"
        )
        .sum()
    )


    if len(
        supported
    ) > 0:

        row_decision = (
            "POSITIVE_SUPPORTED"
        )


    elif (
        "REPORT_MISSING_OR_INVALID"
        in
        statuses
    ):

        row_decision = (
            "CENSOR_SOURCE_MISSING"
        )


    elif (
        "START_ONLY"
        in
        statuses
        or
        "END_ONLY"
        in
        statuses
    ):

        row_decision = (
            "CENSOR_TEMPORAL_AMBIGUITY"
        )


    elif statuses == {
        "ABSENT_BOTH"
    }:

        row_decision = (
            "STALE_ALL_MATCHED_ASSOCIATIONS"
        )


    else:

        row_decision = (
            "REQUIRES_REVIEW"
        )


    row_rows.append(
        {
            "cycle":
                cycle,

            "HARPNUM":
                harpnum,

            "T_REC":
                t_rec,

            "ENTITY_GROUP_ID":
                group[
                    "ENTITY_GROUP_ID"
                ].iloc[0],

            "observation_time_utc":
                group[
                    "observation_time_utc"
                ].iloc[0],

            "positive_source":
                group[
                    "positive_source"
                ].iloc[0],

            "matched_events":
                len(
                    group
                ),

            "supported_events":
                len(
                    supported
                ),

            "supported_primary_events":
                supported_primary,

            "supported_secondary_events":
                supported_secondary,

            "start_only_events":
                int(
                    (
                        group[
                            "bracket_status"
                        ]
                        ==
                        "START_ONLY"
                    )
                    .sum()
                ),

            "end_only_events":
                int(
                    (
                        group[
                            "bracket_status"
                        ]
                        ==
                        "END_ONLY"
                    )
                    .sum()
                ),

            "absent_both_events":
                int(
                    (
                        group[
                            "bracket_status"
                        ]
                        ==
                        "ABSENT_BOTH"
                    )
                    .sum()
                ),

            "missing_report_events":
                int(
                    (
                        group[
                            "bracket_status"
                        ]
                        ==
                        "REPORT_MISSING_OR_INVALID"
                    )
                    .sum()
                ),

            "row_decision":
                row_decision,
        }
    )


row_audit = pd.DataFrame(
    row_rows
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for (
    cycle,
    positive_source,
), subset in row_audit.groupby(
    [
        "cycle",
        "positive_source",
    ]
):

    counts = (
        subset[
            "row_decision"
        ]
        .value_counts()
    )


    summary_rows.append(
        {
            "cycle":
                cycle,

            "positive_source":
                positive_source,

            "positive_rows":
                len(
                    subset
                ),

            "positive_supported":
                int(
                    counts.get(
                        "POSITIVE_SUPPORTED",
                        0
                    )
                ),

            "censor_temporal_ambiguity":
                int(
                    counts.get(
                        "CENSOR_TEMPORAL_AMBIGUITY",
                        0
                    )
                ),

            "stale_all_matched_associations":
                int(
                    counts.get(
                        "STALE_ALL_MATCHED_ASSOCIATIONS",
                        0
                    )
                ),

            "censor_source_missing":
                int(
                    counts.get(
                        "CENSOR_SOURCE_MISSING",
                        0
                    )
                ),

            "requires_review":
                int(
                    counts.get(
                        "REQUIRES_REVIEW",
                        0
                    )
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
    exist_ok=True,
)


event_audit.to_csv(
    EVENT_OUTPUT,
    index=False,
)


row_audit.to_csv(
    ROW_OUTPUT,
    index=False,
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "ALL POSITIVE SRS SUMMARY"
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
    "GLOBAL POSITIVE DECISION COUNTS"
)

print(
    "===================================="
)


global_counts = (
    row_audit[
        "row_decision"
    ]
    .value_counts()
)


print(
    global_counts.to_string()
)


print(
    "\n===================================="
)

print(
    "CYCLE TOTALS"
)

print(
    "===================================="
)


cycle_totals = (
    row_audit
    .groupby(
        [
            "cycle",
            "row_decision",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


print(
    cycle_totals.to_string()
)


print(
    "\nPositive rows expected:",
    8836
)


print(
    "Positive rows audited:",
    len(
        row_audit
    )
)


print(
    "Accounted rows:",
    int(
        global_counts.sum()
    )
)


print(
    "\nSaved:"
)

print(
    EVENT_OUTPUT
)

print(
    ROW_OUTPUT
)

print(
    SUMMARY_OUTPUT
)


if len(
    row_audit
) != 8836:

    raise RuntimeError(
        "All-positive row accounting failed."
    )


if int(
    global_counts.sum()
) != len(
    row_audit
):

    raise RuntimeError(
        "Decision accounting mismatch."
    )


print(
    "\nPre-model Fix H complete."
)