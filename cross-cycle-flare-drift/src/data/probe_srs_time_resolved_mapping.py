from pathlib import Path
import re

import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

RAW_OUTPUT_DIR = Path(
    "data/raw/srs/probes"
)

MANIFEST_DIR = Path(
    "data_manifests"
)

OUTPUT_FILE = (
    MANIFEST_DIR
    /
    "pre_model_srs_probe.csv"
)


BASE_URL = (
    "https://www.ngdc.noaa.gov/"
    "stp/space-weather/swpc-products/"
    "daily_reports/solar_region_summaries"
)


# ============================================================
# PROBE CASES
#
# SRS issued on D+1 reports regions observed during day D.
# ============================================================

PROBES = [

    {
        "case":
            "HARP_1834",

        "observation_date":
            "2012-07-08",

        "report_date":
            "2012-07-09",

        "target_full_noaa":
            [
                11519,
                11520,
                11521,
            ],
    },

    {
        "case":
            "HARP_13354",

        "observation_date":
            "2025-06-13",

        "report_date":
            "2025-06-14",

        "target_full_noaa":
            [
                11114,
                14113,
                14114,
                14115,
            ],
    },
]


# ============================================================
# HELPERS
# ============================================================

def srs_display_number(
    full_noaa
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # SRS historically displays NOAA region IDs using
    # four digits after region numbering exceeded 10000.
    #
    # This modulo operation is ONLY for matching the textual
    # SRS display format.
    #
    # It must NOT be reused as the GOES normalization rule.
    # --------------------------------------------------------

    return int(
        full_noaa
    ) % 10000


def build_url(
    report_date
):

    timestamp = pd.Timestamp(
        report_date
    )


    year = timestamp.strftime(
        "%Y"
    )

    month = timestamp.strftime(
        "%m"
    )

    yyyymmdd = timestamp.strftime(
        "%Y%m%d"
    )


    return (
        f"{BASE_URL}/"
        f"{year}/"
        f"{month}/"
        f"{yyyymmdd}SRS.txt"
    )


def parse_region_lines(
    text
):

    records = []


    # Example:
    #
    # 4114 N15W20 123 0450 Eki ...
    #
    # or plage:
    #
    # 4114 N15W20 123
    #
    pattern = re.compile(
        r"^\s*"
        r"(?P<number>\d{4})"
        r"\s+"
        r"(?P<location>[NS]\d{2}[EW]\d{2,3})"
        r"\s+"
        r"(?P<carrington>\d{3})"
        r"(?:\s+.*)?$"
    )


    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):

        match = pattern.match(
            line
        )


        if not match:
            continue


        records.append(
            {
                "line_number":
                    line_number,

                "display_region":
                    int(
                        match.group(
                            "number"
                        )
                    ),

                "location":
                    match.group(
                        "location"
                    ),

                "carrington_longitude":
                    int(
                        match.group(
                            "carrington"
                        )
                    ),

                "raw_line":
                    line.strip(),
            }
        )


    return records


# ============================================================
# MAIN
# ============================================================

print(
    "===================================="
)

print(
    "PRE-MODEL FIX F — SRS SOURCE PROBE"
)

print(
    "===================================="
)


RAW_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


output_rows = []


for probe in PROBES:

    report_date = pd.Timestamp(
        probe[
            "report_date"
        ]
    )


    url = build_url(
        report_date
    )


    filename = (
        report_date.strftime(
            "%Y%m%d"
        )
        +
        "SRS.txt"
    )


    raw_path = (
        RAW_OUTPUT_DIR
        /
        filename
    )


    print(
        "\n===================================="
    )

    print(
        probe[
            "case"
        ]
    )

    print(
        "===================================="
    )


    print(
        "Observation date:",
        probe[
            "observation_date"
        ]
    )


    print(
        "SRS report date:",
        probe[
            "report_date"
        ]
    )


    print(
        "URL:",
        url
    )


    response = requests.get(
        url,
        timeout=30,
    )


    print(
        "HTTP status:",
        response.status_code
    )


    response.raise_for_status()


    text = response.text


    raw_path.write_text(
        text,
        encoding="utf-8"
    )


    print(
        "Saved raw report:",
        raw_path
    )


    parsed = parse_region_lines(
        text
    )


    print(
        "Parsed region-position lines:",
        len(
            parsed
        )
    )


    parsed_by_number = {}


    for record in parsed:

        parsed_by_number.setdefault(
            record[
                "display_region"
            ],
            []
        ).append(
            record
        )


    print(
        "\nTarget region check:"
    )


    for full_noaa in probe[
        "target_full_noaa"
    ]:

        display_noaa = srs_display_number(
            full_noaa
        )


        matches = parsed_by_number.get(
            display_noaa,
            []
        )


        print(
            "\nFull NOAA:",
            full_noaa
        )


        print(
            "SRS display number:",
            f"{display_noaa:04d}"
        )


        print(
            "Matches:",
            len(
                matches
            )
        )


        if matches:

            for match in matches:

                print(
                    " ",
                    match[
                        "raw_line"
                    ]
                )


                output_rows.append(
                    {
                        "case":
                            probe[
                                "case"
                            ],

                        "observation_date":
                            probe[
                                "observation_date"
                            ],

                        "report_date":
                            probe[
                                "report_date"
                            ],

                        "full_noaa_region":
                            full_noaa,

                        "srs_display_region":
                            f"{display_noaa:04d}",

                        "found":
                            True,

                        "location":
                            match[
                                "location"
                            ],

                        "carrington_longitude":
                            match[
                                "carrington_longitude"
                            ],

                        "line_number":
                            match[
                                "line_number"
                            ],

                        "raw_line":
                            match[
                                "raw_line"
                            ],

                        "source_url":
                            url,
                    }
                )


        else:

            print(
                "  NOT FOUND"
            )


            output_rows.append(
                {
                    "case":
                        probe[
                            "case"
                        ],

                    "observation_date":
                        probe[
                            "observation_date"
                        ],

                    "report_date":
                        probe[
                            "report_date"
                        ],

                    "full_noaa_region":
                        full_noaa,

                    "srs_display_region":
                        f"{display_noaa:04d}",

                    "found":
                        False,

                    "location":
                        None,

                    "carrington_longitude":
                        None,

                    "line_number":
                        None,

                    "raw_line":
                        None,

                    "source_url":
                        url,
                }
            )


# ============================================================
# SAVE MANIFEST
# ============================================================

result = pd.DataFrame(
    output_rows
)


result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "SRS PROBE SUMMARY"
)

print(
    "===================================="
)


summary = (
    result
    .groupby(
        "case"
    )
    .agg(
        targets=(
            "full_noaa_region",
            "count",
        ),

        found=(
            "found",
            "sum",
        ),
    )
    .reset_index()
)


print(
    summary.to_string(
        index=False
    )
)


print(
    "\n------------------------------------"
)

print(
    "FULL RESULT"
)

print(
    "------------------------------------"
)


print(
    result[
        [
            "case",
            "observation_date",
            "full_noaa_region",
            "srs_display_region",
            "found",
            "location",
            "carrington_longitude",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\nPre-model Fix F complete."
)