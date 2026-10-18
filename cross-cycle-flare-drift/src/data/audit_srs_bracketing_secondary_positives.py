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
    "pre_model_srs_bracketing_event_audit.csv"
)

ROW_OUTPUT = (
    MANIFEST_DIR
    /
    "pre_model_srs_bracketing_row_audit.csv"
)

SUMMARY_OUTPUT = (
    MANIFEST_DIR
    /
    "pre_model_srs_bracketing_summary.csv"
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


def full_noaa_to_srs_display(
    full_noaa
):

    # IMPORTANT:
    #
    # Modulo is used ONLY for matching the
    # four-digit SRS textual display.
    #
    # This is NOT the GOES normalization rule.

    return int(
        full_noaa
    ) % 10000


def parse_srs_regions(text):

    regions = set()

    # Matches both:
    #
    # Part I — sunspot regions
    # Part IA — H-alpha plages
    #
    # Typical line:
    #
    # 4114 N18E39 034 0080 Dai ...
    #
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

            "path":
                str(
                    path
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

            "path":
                None,

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

            "path":
                None,
        }


    text = response.text


    # Basic sanity check.
    #
    # Prevent an HTML error page from being
    # silently treated as an SRS report.

    if (
        "Solar Region Summary" not in text
        and
        "SOLAR REGION SUMMARY" not in text
        and
        "Joint USAF/NOAA" not in text
    ):

        return {
            "status":
                "INVALID_CONTENT",

            "text":
                None,

            "url":
                url,

            "path":
                None,
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

        "path":
            str(
                path
            ),
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
# LOAD FIX C EVENT AUDIT
# ============================================================

print(
    "===================================="
)

print(
    "PRE-MODEL FIX G — SRS BRACKETING AUDIT"
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
    "observation_time_utc",
    "ENTITY_GROUP_ID",
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
        "Missing required columns: "
        f"{sorted(missing)}"
    )


# ============================================================
# KEEP ONLY SECONDARY-ONLY POSITIVE ROW EVENTS
# ============================================================

secondary = detail[
    detail[
        "positive_source"
    ]
    ==
    "SECONDARY_ONLY"
].copy()


if len(
    secondary
) == 0:

    raise RuntimeError(
        "No SECONDARY_ONLY positive rows found."
    )


secondary[
    "observation_time_utc"
] = pd.to_datetime(
    secondary[
        "observation_time_utc"
    ],
    utc=True,
    errors="coerce",
)


secondary[
    "event_time_utc"
] = pd.to_datetime(
    secondary[
        "event_time_utc"
    ],
    utc=True,
    errors="coerce",
)


secondary[
    "event_noaa_region"
] = pd.to_numeric(
    secondary[
        "event_noaa_region"
    ],
    errors="coerce",
)


secondary = secondary[
    secondary[
        "observation_time_utc"
    ]
    .notna()
    &
    secondary[
        "event_noaa_region"
    ]
    .notna()
].copy()


secondary[
    "event_noaa_region"
] = secondary[
    "event_noaa_region"
].astype(int)


print(
    "Secondary-only event records:",
    len(
        secondary
    )
)


print(
    "Unique secondary-only sample rows:",
    secondary[
        [
            "cycle",
            "HARPNUM",
            "T_REC",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


print(
    "Unique NOAA event regions:",
    secondary[
        "event_noaa_region"
    ]
    .nunique()
)


# ============================================================
# DETERMINE REQUIRED SRS REPORT DATES
# ============================================================
#
# For an observation on UTC date D:
#
# report D
#   → previous day locations valid at D 00:00 UTC
#
# report D+1
#   → D-day locations valid at D+1 00:00 UTC
#
# These two snapshots bracket the observation day.
# ============================================================

secondary[
    "_observation_date"
] = (
    secondary[
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


for observation_date in secondary[
    "_observation_date"
].unique():

    observation_date = pd.Timestamp(
        observation_date
    )


    required_report_dates.add(
        observation_date
    )


    required_report_dates.add(
        observation_date
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
# DOWNLOAD / CACHE REQUIRED REPORTS
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


    if (
        status
        in {
            "OK_CACHED",
            "OK_DOWNLOADED",
        }
    ):

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
        index % 25 == 0
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
# REPORT COVERAGE
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
    "\nSRS report status:"
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
# EVENT-LEVEL BRACKETING
# ============================================================

event_rows = []


for _, row in secondary.iterrows():

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


    if (
        "BRACKETED_PRESENT"
        in
        statuses
    ):

        row_decision = (
            "SUPPORTED_SECONDARY_POSITIVE"
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
            "STALE_SECONDARY_ASSOCIATION"
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

            "matched_secondary_events":
                len(
                    group
                ),

            "unique_secondary_regions":
                group[
                    "event_noaa_region"
                ]
                .nunique(),

            "bracketed_present_events":
                int(
                    (
                        group[
                            "bracket_status"
                        ]
                        ==
                        "BRACKETED_PRESENT"
                    )
                    .sum()
                ),

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


for cycle in [
    "cycle24",
    "cycle25",
]:

    subset = row_audit[
        row_audit[
            "cycle"
        ]
        ==
        cycle
    ]


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

            "secondary_only_rows":
                len(
                    subset
                ),

            "supported_secondary_positive":
                int(
                    counts.get(
                        "SUPPORTED_SECONDARY_POSITIVE",
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

            "stale_secondary_association":
                int(
                    counts.get(
                        "STALE_SECONDARY_ASSOCIATION",
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
# PRINT
# ============================================================

print(
    "\n===================================="
)

print(
    "SRS BRACKETING SUMMARY"
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
    "GLOBAL DECISION COUNTS"
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
    "\nTotal secondary-only rows:",
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


if int(
    global_counts.sum()
) != len(
    row_audit
):

    raise RuntimeError(
        "Row accounting mismatch."
    )


print(
    "\nPre-model Fix G complete."
)