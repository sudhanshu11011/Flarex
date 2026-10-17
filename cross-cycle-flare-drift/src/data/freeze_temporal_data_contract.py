from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

import pandas as pd
from astropy.time import Time


# ============================================================
# PATHS
# ============================================================

RAW_GOES_DIR = Path(
    "data/raw/goes"
)

MANIFEST_DIR = Path(
    "data_manifests"
)

STRIDE_CONTRACT_FILE = (
    MANIFEST_DIR
    / "sampling_stride_contract.json"
)

GOES_MANIFEST_FILE = (
    MANIFEST_DIR
    / "goes_catalog_manifest_2010_2025.csv"
)

TEMPORAL_CONTRACT_FILE = (
    MANIFEST_DIR
    / "temporal_data_contract.json"
)


RAW_GOES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATA SOURCES
# ============================================================

SHARP_SERIES = (
    "hmi.sharp_cea_720s"
)

LORENTZ_SERIES = (
    "cgem.lorentz"
)

GOES_PRODUCT_VERSION = (
    "v1-0-1"
)


GOES_BASE_URL = (
    "https://data.ngdc.noaa.gov/"
    "platforms/solar-space-observing-satellites/"
    "goes/multi/l2/data/"
    "xrsf-l2-flrpt_science/csv"
)


# ============================================================
# TEMPORAL CONTRACT
# ============================================================

C24_START_UTC = pd.Timestamp(
    "2010-05-01T00:00:00Z"
)

C24_END_EXCLUSIVE_UTC = pd.Timestamp(
    "2020-01-01T00:00:00Z"
)


C25_START_UTC = pd.Timestamp(
    "2020-01-01T00:00:00Z"
)

C25_END_EXCLUSIVE_UTC = pd.Timestamp(
    "2026-01-01T00:00:00Z"
)


FORECAST_HORIZON = pd.Timedelta(
    hours=24
)


EXPECTED_STRIDE_MINUTES = 120


# ============================================================
# GOES YEARS
# ============================================================

GOES_YEARS = list(
    range(
        2010,
        2026
    )
)


REQUIRED_GOES_COLUMNS = {
    "start_time",
    "flare_class",
    "active_region",
    "xrsb_irrad",
}


# ============================================================
# HELPER: SHA256
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# HELPER: DOWNLOAD
# ============================================================

def download_with_curl(
    url: str,
    output_file: Path
) -> bool:

    print(
        "Downloading:"
    )

    print(
        url
    )


    try:

        subprocess.run(
            [
                "curl.exe",
                "-L",
                "--fail",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "30",
                "--max-time",
                "180",
                url,
                "-o",
                str(output_file),
            ],
            check=True,
        )

    except subprocess.CalledProcessError:

        return False


    return (
        output_file.exists()
        and
        output_file.stat().st_size > 0
    )


# ============================================================
# VERIFY SAMPLING STRIDE CONTRACT
# ============================================================

print(
    "===================================="
)

print(
    "VERIFY SAMPLING STRIDE CONTRACT"
)

print(
    "===================================="
)


if not STRIDE_CONTRACT_FILE.exists():

    raise FileNotFoundError(
        "Missing sampling stride contract:\n"
        f"{STRIDE_CONTRACT_FILE}"
    )


with STRIDE_CONTRACT_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    stride_contract = json.load(
        file
    )


stride_minutes = int(
    stride_contract[
        "primary_sampling_stride_minutes"
    ]
)


stride_locked = bool(
    stride_contract[
        "primary_stride_locked"
    ]
)


print(
    "Stride:",
    stride_minutes,
    "minutes"
)


print(
    "Locked:",
    stride_locked
)


if (
    stride_minutes
    !=
    EXPECTED_STRIDE_MINUTES
):

    raise RuntimeError(
        "Expected frozen stride = 120 minutes."
    )


if not stride_locked:

    raise RuntimeError(
        "Sampling stride contract is not frozen."
    )


print(
    "PASS: sampling stride contract"
)


# ============================================================
# GOES CATALOG ACQUISITION + AUDIT
# ============================================================

manifest_rows = []


print(
    "\n===================================="
)

print(
    "GOES CATALOG AUDIT 2010-2025"
)

print(
    "===================================="
)


for year in GOES_YEARS:

    print(
        "\n------------------------------------"
    )

    print(
        "Year:",
        year
    )

    print(
        "------------------------------------"
    )


    source_filename = (
        f"sci_xrsf-l2-flrpt_geo_"
        f"y{year}_"
        f"{GOES_PRODUCT_VERSION}.csv"
    )


    source_url = (
        f"{GOES_BASE_URL}/"
        f"{source_filename}"
    )


    local_file = (
        RAW_GOES_DIR
        / f"goes_flare_{year}.csv"
    )


    # --------------------------------------------------------
    # Download only when missing.
    # --------------------------------------------------------

    if (
        not local_file.exists()
        or
        local_file.stat().st_size == 0
    ):

        success = download_with_curl(
            source_url,
            local_file
        )


        if not success:

            raise RuntimeError(
                f"GOES download failed for {year}"
            )

    else:

        print(
            "Using existing local file:"
        )

        print(
            local_file
        )


    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            local_file
        )

    except Exception as error:

        raise RuntimeError(
            f"Failed reading GOES {year}: {error}"
        )


    missing_columns = (
        REQUIRED_GOES_COLUMNS
        -
        set(
            df.columns
        )
    )


    if missing_columns:

        raise RuntimeError(
            f"{year} missing required columns: "
            f"{sorted(missing_columns)}"
        )


    # --------------------------------------------------------
    # Parse start time
    # --------------------------------------------------------

    parsed_times = pd.to_datetime(
        df[
            "start_time"
        ],
        utc=True,
        errors="coerce"
    )


    invalid_times = int(
        parsed_times
        .isna()
        .sum()
    )


    valid_times = (
        parsed_times
        .dropna()
    )


    if len(
        valid_times
    ) > 0:

        first_event = (
            valid_times
            .min()
        )

        last_event = (
            valid_times
            .max()
        )

    else:

        first_event = pd.NaT
        last_event = pd.NaT


    checksum = sha256_file(
        local_file
    )


    print(
        "Rows:",
        len(df)
    )


    print(
        "Invalid start_time:",
        invalid_times
    )


    print(
        "SHA256:",
        checksum
    )


    manifest_rows.append(
        {
            "year":
                year,

            "cycle":
                (
                    "Cycle24"
                    if year <= 2019
                    else "Cycle25"
                ),

            "product_version":
                GOES_PRODUCT_VERSION,

            "source_filename":
                source_filename,

            "source_url":
                source_url,

            "local_file":
                str(
                    local_file
                ),

            "rows":
                len(df),

            "invalid_start_time":
                invalid_times,

            "first_event_utc":
                first_event,

            "last_event_utc":
                last_event,

            "file_size_bytes":
                local_file.stat().st_size,

            "sha256":
                checksum,

            "required_schema_ok":
                len(
                    missing_columns
                ) == 0,
        }
    )


manifest = pd.DataFrame(
    manifest_rows
)


manifest.to_csv(
    GOES_MANIFEST_FILE,
    index=False
)


# ============================================================
# GOES MANIFEST VALIDATION
# ============================================================

print(
    "\n===================================="
)

print(
    "GOES MANIFEST VALIDATION"
)

print(
    "===================================="
)


expected_years = set(
    GOES_YEARS
)


found_years = set(
    manifest[
        "year"
    ]
    .astype(int)
)


missing_years = sorted(
    expected_years
    -
    found_years
)


print(
    "Expected years:",
    len(expected_years)
)


print(
    "Manifest years:",
    len(found_years)
)


print(
    "Missing years:",
    missing_years
)


schema_failures = int(
    (
        ~manifest[
            "required_schema_ok"
        ]
    )
    .sum()
)


invalid_time_total = int(
    manifest[
        "invalid_start_time"
    ]
    .sum()
)


print(
    "Schema failures:",
    schema_failures
)


print(
    "Invalid start_time rows:",
    invalid_time_total
)


goes_manifest_pass = (
    len(
        missing_years
    ) == 0

    and

    schema_failures == 0

    and

    invalid_time_total == 0
)


print(
    "GOES catalog validation:",
    (
        "PASS"
        if goes_manifest_pass
        else "FAIL"
    )
)


if not goes_manifest_pass:

    raise RuntimeError(
        "GOES catalog validation failed."
    )


# ============================================================
# TAI -> UTC HELPER
# ============================================================

def tai_string_to_utc(
    value: str
) -> pd.Timestamp:

    cleaned = (
        value
        .replace(
            "_TAI",
            ""
        )
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


    tai = Time(
        cleaned,
        format="isot",
        scale="tai"
    )


    utc_datetime = (
        tai
        .utc
        .to_datetime(
            timezone=timezone.utc
        )
    )


    return pd.Timestamp(
        utc_datetime
    )


# ============================================================
# GENERATE 2-HOUR TAI GRID FOR ONE DATE
# ============================================================

def tai_grid_for_date(
    date_string: str
):

    values = []


    for hour in range(
        0,
        24,
        2
    ):

        tai_string = (
            f"{date_string}_"
            f"{hour:02d}:00:00_TAI"
        )


        utc_time = tai_string_to_utc(
            tai_string
        )


        values.append(
            {
                "tai":
                    tai_string,

                "utc":
                    utc_time,
            }
        )


    return values


# ============================================================
# FIND FIRST VALID GRID POINT
# ============================================================

def first_valid_sample(
    tai_date: str,
    domain_start_utc: pd.Timestamp
):

    candidates = tai_grid_for_date(
        tai_date
    )


    valid = [
        item

        for item in candidates

        if item[
            "utc"
        ]
        >=
        domain_start_utc
    ]


    if not valid:

        return None


    return valid[0]


# ============================================================
# FIND LAST VALID GRID POINT
# ============================================================

def last_valid_sample(
    tai_date: str,
    domain_end_exclusive_utc: pd.Timestamp
):

    candidates = tai_grid_for_date(
        tai_date
    )


    valid = [
        item

        for item in candidates

        if (
            item[
                "utc"
            ]
            +
            FORECAST_HORIZON
        )
        <
        domain_end_exclusive_utc
    ]


    if not valid:

        return None


    return valid[-1]


# ============================================================
# BOUNDARY AUDIT
# ============================================================

print(
    "\n===================================="
)

print(
    "TEMPORAL BOUNDARY AUDIT"
)

print(
    "===================================="
)


c24_first = first_valid_sample(
    "2010.05.01",
    C24_START_UTC
)


c24_last = last_valid_sample(
    "2019.12.31",
    C24_END_EXCLUSIVE_UTC
)


c25_first = first_valid_sample(
    "2020.01.01",
    C25_START_UTC
)


c25_last = last_valid_sample(
    "2025.12.31",
    C25_END_EXCLUSIVE_UTC
)


if (
    c24_first is None
    or
    c24_last is None
    or
    c25_first is None
    or
    c25_last is None
):

    raise RuntimeError(
        "Could not resolve temporal sampling boundaries."
    )


print(
    "Cycle24 start UTC:",
    C24_START_UTC
)


print(
    "Cycle24 end exclusive UTC:",
    C24_END_EXCLUSIVE_UTC
)


print(
    "Cycle24 first valid TAI sample:",
    c24_first[
        "tai"
    ]
)


print(
    "Cycle24 first valid UTC:",
    c24_first[
        "utc"
    ]
)


print(
    "Cycle24 last valid TAI sample:",
    c24_last[
        "tai"
    ]
)


print(
    "Cycle24 last valid UTC:",
    c24_last[
        "utc"
    ]
)


print()


print(
    "Cycle25 start UTC:",
    C25_START_UTC
)


print(
    "Cycle25 end exclusive UTC:",
    C25_END_EXCLUSIVE_UTC
)


print(
    "Cycle25 first valid TAI sample:",
    c25_first[
        "tai"
    ]
)


print(
    "Cycle25 first valid UTC:",
    c25_first[
        "utc"
    ]
)


print(
    "Cycle25 last valid TAI sample:",
    c25_last[
        "tai"
    ]
)


print(
    "Cycle25 last valid UTC:",
    c25_last[
        "utc"
    ]
)


# ============================================================
# BOUNDARY VALIDATION
# ============================================================

c24_start_ok = (
    c24_first[
        "utc"
    ]
    >=
    C24_START_UTC
)


c24_end_ok = (
    c24_last[
        "utc"
    ]
    +
    FORECAST_HORIZON
    <
    C24_END_EXCLUSIVE_UTC
)


c25_start_ok = (
    c25_first[
        "utc"
    ]
    >=
    C25_START_UTC
)


c25_end_ok = (
    c25_last[
        "utc"
    ]
    +
    FORECAST_HORIZON
    <
    C25_END_EXCLUSIVE_UTC
)


print(
    "\n------------------------------------"
)

print(
    "BOUNDARY VALIDATION"
)

print(
    "------------------------------------"
)


print(
    "Cycle24 first sample:",
    "PASS"
    if c24_start_ok
    else "FAIL"
)


print(
    "Cycle24 final 24h horizon:",
    "PASS"
    if c24_end_ok
    else "FAIL"
)


print(
    "Cycle25 first sample:",
    "PASS"
    if c25_start_ok
    else "FAIL"
)


print(
    "Cycle25 final 24h horizon:",
    "PASS"
    if c25_end_ok
    else "FAIL"
)


boundary_pass = (
    c24_start_ok
    and
    c24_end_ok
    and
    c25_start_ok
    and
    c25_end_ok
)


if not boundary_pass:

    raise RuntimeError(
        "Temporal boundary validation failed."
    )


# ============================================================
# FINAL TEMPORAL / DATA CONTRACT
# ============================================================

contract = {

    "contract_version":
        "1.0",

    "status":
        "FROZEN",

    "phase":
        "Phase 8 - Temporal and Data Contract Freeze",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "data_sources": {

        "sharp_series":
            SHARP_SERIES,

        "lorentz_series":
            LORENTZ_SERIES,

        "goes_product":
            "GOES XRS L2 science flare report",

        "goes_product_version":
            GOES_PRODUCT_VERSION,

        "goes_years":
            GOES_YEARS,
    },

    "sampling": {

        "native_cadence_minutes":
            12,

        "primary_stride_minutes":
            120,

        "grid_anchor":
            "00:00 TAI",

        "grid_rule":
            "fixed absolute 2-hour TAI grid",

        "sample_by_available_row_index":
            False,
    },

    "forecast_target": {

        "flare_threshold":
            "M1.0+",

        "forecast_horizon_hours":
            24,

        "future_interval":
            "(t, t+24h]",

        "event_time_field":
            "GOES start_time",
    },

    "cycle24": {

        "domain_start_utc":
            C24_START_UTC.isoformat(),

        "domain_end_exclusive_utc":
            C24_END_EXCLUSIVE_UTC.isoformat(),

        "first_valid_sample_tai":
            c24_first[
                "tai"
            ],

        "first_valid_sample_utc":
            c24_first[
                "utc"
            ].isoformat(),

        "last_valid_sample_tai":
            c24_last[
                "tai"
            ],

        "last_valid_sample_utc":
            c24_last[
                "utc"
            ].isoformat(),

        "role":
            "model development, validation and independent holdout",
    },

    "cycle25": {

        "domain_start_utc":
            C25_START_UTC.isoformat(),

        "domain_end_exclusive_utc":
            C25_END_EXCLUSIVE_UTC.isoformat(),

        "calendar_years":
            list(
                range(
                    2020,
                    2026
                )
            ),

        "first_valid_sample_tai":
            c25_first[
                "tai"
            ],

        "first_valid_sample_utc":
            c25_first[
                "utc"
            ].isoformat(),

        "last_valid_sample_tai":
            c25_last[
                "tai"
            ],

        "last_valid_sample_utc":
            c25_last[
                "utc"
            ].isoformat(),

        "role":
            "frozen-model transfer evaluation only",
    },

    "leakage_rules": {

        "cycle25_model_tuning":
            False,

        "cycle25_preprocessing_fit":
            False,

        "cycle25_threshold_selection":
            False,

        "entity_group_split_required":
            True,

        "label_horizon_must_remain_inside_cycle_domain":
            True,
    },

    "manifest_files": {

        "sampling_stride_contract":
            str(
                STRIDE_CONTRACT_FILE
            ),

        "goes_catalog_manifest":
            str(
                GOES_MANIFEST_FILE
            ),
    }
}


with TEMPORAL_CONTRACT_FILE.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        contract,
        file,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "PHASE 8 — FINAL CONTRACT"
)

print(
    "===================================="
)


print(
    "Sampling stride:",
    "120 minutes"
)


print(
    "Cycle24:"
)

print(
    c24_first[
        "tai"
    ],
    "→",
    c24_last[
        "tai"
    ]
)


print(
    "Cycle25:"
)

print(
    c25_first[
        "tai"
    ],
    "→",
    c25_last[
        "tai"
    ]
)


print(
    "GOES years:",
    "2010 → 2025"
)


print(
    "GOES catalog validation:",
    "PASS"
)


print(
    "Temporal boundary validation:",
    "PASS"
)


print(
    "Contract status:",
    "FROZEN"
)


print(
    "\nSaved:"
)

print(
    GOES_MANIFEST_FILE
)

print(
    TEMPORAL_CONTRACT_FILE
)


print(
    "\nPhase 8 temporal/data contract audit finished."
)