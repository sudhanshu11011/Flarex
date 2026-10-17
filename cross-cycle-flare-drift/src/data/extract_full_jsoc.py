from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time

import drms
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

client = drms.Client()

SHARP_SERIES = "hmi.sharp_cea_720s"
LORENTZ_SERIES = "cgem.lorentz"

CONTRACT_FILE = Path(
    "data_manifests/temporal_data_contract.json"
)

MANIFEST_FILE = Path(
    "data_manifests/phase9_jsoc_extraction_manifest.csv"
)

RAW_ROOT = Path(
    "data/raw/jsoc"
)


STRIDE_MINUTES = 120

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10


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


SHARP_COLUMNS = [
    "T_REC",
    "HARPNUM",
    "NOAA_AR",
    "NOAA_ARS",
    "QUALITY",
    "CMASK",
    "LON_FWT",
    "LAT_FWT",
] + SHARP_FEATURES


LORENTZ_COLUMNS = [
    "T_REC",
    "HARPNUM",
] + LORENTZ_FEATURES


SHARP_KEYS = ",".join(
    SHARP_COLUMNS
)


LORENTZ_KEYS = ",".join(
    LORENTZ_COLUMNS
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def parse_tai_string(value):

    """
    Parse JSOC timestamp without converting to UTC.

    Example:
    2011.06.15_12:00:00_TAI
    ->
    pandas Timestamp on the TAI clock.

    Only used for extraction-grid filtering.
    """

    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()

    text = text.replace(
        "_TAI",
        ""
    )

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

    return pd.to_datetime(
        text,
        format="%Y-%m-%dT%H:%M:%S",
        errors="coerce"
    )


def contract_tai_to_timestamp(value):

    parsed = parse_tai_string(
        value
    )

    if pd.isna(parsed):

        raise ValueError(
            f"Invalid contract TAI timestamp: {value}"
        )

    return parsed


def jsoc_time_string(timestamp):

    return (
        f"{timestamp.year:04d}."
        f"{timestamp.month:02d}."
        f"{timestamp.day:02d}_"
        f"{timestamp.hour:02d}:"
        f"{timestamp.minute:02d}:"
        f"{timestamp.second:02d}_TAI"
    )


def next_month(timestamp):

    if timestamp.month == 12:

        return pd.Timestamp(
            year=timestamp.year + 1,
            month=1,
            day=1,
        )

    return pd.Timestamp(
        year=timestamp.year,
        month=timestamp.month + 1,
        day=1,
    )


def month_iterator(
    first_timestamp,
    last_timestamp
):

    current = pd.Timestamp(
        year=first_timestamp.year,
        month=first_timestamp.month,
        day=1,
    )

    final_month = pd.Timestamp(
        year=last_timestamp.year,
        month=last_timestamp.month,
        day=1,
    )

    while current <= final_month:

        yield current

        current = next_month(
            current
        )


def query_with_retry(
    query,
    keys,
    label
):

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            print(
                f"{label}: query attempt "
                f"{attempt}/{MAX_RETRIES}"
            )

            dataframe = client.query(
                query,
                key=keys
            )

            return dataframe

        except Exception as error:

            last_error = error

            print(
                f"{label}: query failed:",
                error
            )

            if attempt < MAX_RETRIES:

                print(
                    f"Retrying in "
                    f"{RETRY_DELAY_SECONDS}s..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    raise RuntimeError(
        f"{label} query failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


def validate_required_columns(
    dataframe,
    required_columns,
    label
):

    missing = (
        set(required_columns)
        -
        set(dataframe.columns)
    )

    if missing:

        raise RuntimeError(
            f"{label} missing columns: "
            f"{sorted(missing)}"
        )


def filter_to_chunk_and_contract(
    dataframe,
    required_columns,
    chunk_start,
    chunk_end,
    contract_first,
    contract_last,
    label
):

    validate_required_columns(
        dataframe,
        required_columns,
        label
    )

    df = dataframe.copy()

    if len(df) == 0:

        return df.reindex(
            columns=required_columns
        )


    df["_T_REC_PARSED"] = (
        df["T_REC"]
        .apply(
            parse_tai_string
        )
    )


    invalid_trec = int(
        df["_T_REC_PARSED"]
        .isna()
        .sum()
    )


    if invalid_trec > 0:

        raise RuntimeError(
            f"{label}: "
            f"{invalid_trec} invalid T_REC values"
        )


    # --------------------------------------------------------
    # Half-open monthly chunk:
    #
    # [chunk_start, chunk_end)
    # --------------------------------------------------------

    mask = (
        (
            df["_T_REC_PARSED"]
            >=
            chunk_start
        )
        &
        (
            df["_T_REC_PARSED"]
            <
            chunk_end
        )
        &
        (
            df["_T_REC_PARSED"]
            >=
            contract_first
        )
        &
        (
            df["_T_REC_PARSED"]
            <=
            contract_last
        )
    )


    df = df[
        mask
    ].copy()


    # --------------------------------------------------------
    # Frozen global 2-hour TAI grid validation
    # --------------------------------------------------------

    off_grid = (
        (
            df["_T_REC_PARSED"].dt.minute
            !=
            0
        )
        |
        (
            df["_T_REC_PARSED"].dt.second
            !=
            0
        )
        |
        (
            df["_T_REC_PARSED"].dt.hour
            %
            2
            !=
            0
        )
    )


    off_grid_count = int(
        off_grid.sum()
    )


    if off_grid_count > 0:

        raise RuntimeError(
            f"{label}: "
            f"{off_grid_count} records violate "
            f"the frozen 120-minute TAI grid."
        )


    df = (
        df
        .drop(
            columns=[
                "_T_REC_PARSED"
            ]
        )
        .reindex(
            columns=required_columns
        )
        .reset_index(
            drop=True
        )
    )


    return df


def duplicate_key_count(
    dataframe
):

    if len(dataframe) == 0:
        return 0

    return int(
        dataframe[
            [
                "HARPNUM",
                "T_REC",
            ]
        ]
        .duplicated()
        .sum()
    )


def first_last_trec(
    dataframe
):

    if len(dataframe) == 0:

        return None, None

    values = (
        dataframe["T_REC"]
        .apply(
            parse_tai_string
        )
    )

    return (
        str(
            values.min()
        ),
        str(
            values.max()
        ),
    )


def atomic_save_parquet(
    dataframe,
    output_file
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_file = (
        output_file.parent
        /
        (
            output_file.stem
            +
            ".tmp.parquet"
        )
    )


    dataframe.to_parquet(
        temporary_file,
        index=False
    )


    temporary_file.replace(
        output_file
    )


def load_cached_parquet(
    path,
    required_columns
):

    if not path.exists():
        return None


    try:

        dataframe = pd.read_parquet(
            path
        )

    except Exception:

        return None


    missing = (
        set(required_columns)
        -
        set(dataframe.columns)
    )


    if missing:

        return None


    return dataframe.reindex(
        columns=required_columns
    )


def save_manifest_row(
    row
):

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    if MANIFEST_FILE.exists():

        manifest = pd.read_csv(
            MANIFEST_FILE
        )

    else:

        manifest = pd.DataFrame()


    new_row = pd.DataFrame(
        [row]
    )


    if len(manifest) > 0:

        same_chunk = (
            (
                manifest["cycle"]
                ==
                row["cycle"]
            )
            &
            (
                manifest["chunk_id"]
                ==
                row["chunk_id"]
            )
        )

        manifest = manifest[
            ~same_chunk
        ]


    manifest = pd.concat(
        [
            manifest,
            new_row,
        ],
        ignore_index=True
    )


    manifest = (
        manifest
        .sort_values(
            [
                "cycle",
                "chunk_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    manifest.to_csv(
        MANIFEST_FILE,
        index=False
    )


# ============================================================
# LOAD FROZEN CONTRACT
# ============================================================

if not CONTRACT_FILE.exists():

    raise FileNotFoundError(
        f"Missing temporal contract: "
        f"{CONTRACT_FILE}"
    )


with CONTRACT_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    contract = json.load(
        file
    )


if (
    contract.get("status")
    !=
    "FROZEN"
):

    raise RuntimeError(
        "Temporal contract is not frozen."
    )


contract_stride = int(
    contract[
        "sampling"
    ][
        "primary_stride_minutes"
    ]
)


if (
    contract_stride
    !=
    STRIDE_MINUTES
):

    raise RuntimeError(
        "Frozen contract stride does not "
        "match extraction stride."
    )


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Restart-safe full JSOC extraction "
        "on the frozen 120-minute TAI grid."
    )
)


parser.add_argument(
    "--cycle",
    choices=[
        "cycle24",
        "cycle25",
        "all",
    ],
    required=True,
)


args = parser.parse_args()


if args.cycle == "all":

    requested_cycles = [
        "cycle24",
        "cycle25",
    ]

else:

    requested_cycles = [
        args.cycle
    ]


# ============================================================
# MAIN EXTRACTION
# ============================================================

print(
    "===================================="
)

print(
    "PHASE 9 — FULL JSOC EXTRACTION"
)

print(
    "===================================="
)


print(
    "Frozen stride:",
    STRIDE_MINUTES,
    "minutes"
)


print(
    "Requested:",
    requested_cycles
)


for cycle_key in requested_cycles:

    cycle_contract = contract[
        cycle_key
    ]


    cycle_label = (
        "Cycle24"
        if cycle_key == "cycle24"
        else "Cycle25"
    )


    first_tai_string = (
        cycle_contract[
            "first_valid_sample_tai"
        ]
    )


    last_tai_string = (
        cycle_contract[
            "last_valid_sample_tai"
        ]
    )


    contract_first = (
        contract_tai_to_timestamp(
            first_tai_string
        )
    )


    contract_last = (
        contract_tai_to_timestamp(
            last_tai_string
        )
    )


    print(
        "\n\n===================================="
    )

    print(
        cycle_label.upper()
    )

    print(
        "===================================="
    )


    print(
        "First frozen sample:",
        first_tai_string
    )


    print(
        "Last frozen sample:",
        last_tai_string
    )


    for month_start in month_iterator(
        contract_first,
        contract_last
    ):

        month_end = next_month(
            month_start
        )


        chunk_id = (
            f"{month_start.year:04d}-"
            f"{month_start.month:02d}"
        )


        print(
            "\n------------------------------------"
        )

        print(
            cycle_label,
            chunk_id
        )

        print(
            "------------------------------------"
        )


        duration_hours = int(
            (
                month_end
                -
                month_start
            )
            .total_seconds()
            /
            3600
        )


        jsoc_start = jsoc_time_string(
            month_start
        )


        time_selector = (
            f"{jsoc_start}"
            f"/{duration_hours}h"
            f"@{STRIDE_MINUTES}m"
        )


        sharp_query = (
            f"{SHARP_SERIES}"
            f"[]"
            f"[{time_selector}]"
        )


        lorentz_query = (
            f"{LORENTZ_SERIES}"
            f"[]"
            f"[{time_selector}]"
        )


        cycle_dir = (
            RAW_ROOT
            /
            cycle_key
        )


        sharp_file = (
            cycle_dir
            /
            "sharp"
            /
            f"sharp_{chunk_id}.parquet"
        )


        lorentz_file = (
            cycle_dir
            /
            "lorentz"
            /
            f"lorentz_{chunk_id}.parquet"
        )


        chunk_status = "COMPLETE"
        error_message = None


        sharp_source = None
        lorentz_source = None


        sharp = None
        lorentz = None


        # ====================================================
        # SHARP
        # ====================================================

        try:

            sharp = load_cached_parquet(
                sharp_file,
                SHARP_COLUMNS
            )


            if sharp is not None:

                sharp_source = "CACHE"

                print(
                    "SHARP cached:",
                    len(sharp),
                    "rows"
                )


            else:

                raw_sharp = query_with_retry(
                    sharp_query,
                    SHARP_KEYS,
                    "SHARP"
                )


                sharp = (
                    filter_to_chunk_and_contract(
                        raw_sharp,
                        SHARP_COLUMNS,
                        month_start,
                        month_end,
                        contract_first,
                        contract_last,
                        "SHARP",
                    )
                )


                atomic_save_parquet(
                    sharp,
                    sharp_file
                )


                sharp_source = "JSOC"


                print(
                    "SHARP saved:",
                    len(sharp),
                    "rows"
                )


        except Exception as error:

            chunk_status = "FAILED"

            error_message = (
                f"SHARP: {error}"
            )


            print(
                "SHARP FAILURE:",
                error
            )


        # ====================================================
        # LORENTZ
        # ====================================================

        try:

            lorentz = load_cached_parquet(
                lorentz_file,
                LORENTZ_COLUMNS
            )


            if lorentz is not None:

                lorentz_source = "CACHE"

                print(
                    "Lorentz cached:",
                    len(lorentz),
                    "rows"
                )


            else:

                raw_lorentz = query_with_retry(
                    lorentz_query,
                    LORENTZ_KEYS,
                    "LORENTZ"
                )


                lorentz = (
                    filter_to_chunk_and_contract(
                        raw_lorentz,
                        LORENTZ_COLUMNS,
                        month_start,
                        month_end,
                        contract_first,
                        contract_last,
                        "LORENTZ",
                    )
                )


                atomic_save_parquet(
                    lorentz,
                    lorentz_file
                )


                lorentz_source = "JSOC"


                print(
                    "Lorentz saved:",
                    len(lorentz),
                    "rows"
                )


        except Exception as error:

            chunk_status = "FAILED"


            if error_message:

                error_message += (
                    f" | LORENTZ: {error}"
                )

            else:

                error_message = (
                    f"LORENTZ: {error}"
                )


            print(
                "LORENTZ FAILURE:",
                error
            )


        # ====================================================
        # MANIFEST METADATA
        # ====================================================

        if sharp is not None:

            sharp_rows = len(
                sharp
            )


            sharp_duplicates = (
                duplicate_key_count(
                    sharp
                )
            )


            (
                sharp_first,
                sharp_last,
            ) = first_last_trec(
                sharp
            )


            sharp_sha256 = (
                sha256_file(
                    sharp_file
                )
                if sharp_file.exists()
                else None
            )

        else:

            sharp_rows = None
            sharp_duplicates = None
            sharp_first = None
            sharp_last = None
            sharp_sha256 = None


        if lorentz is not None:

            lorentz_rows = len(
                lorentz
            )


            lorentz_duplicates = (
                duplicate_key_count(
                    lorentz
                )
            )


            (
                lorentz_first,
                lorentz_last,
            ) = first_last_trec(
                lorentz
            )


            lorentz_sha256 = (
                sha256_file(
                    lorentz_file
                )
                if lorentz_file.exists()
                else None
            )

        else:

            lorentz_rows = None
            lorentz_duplicates = None
            lorentz_first = None
            lorentz_last = None
            lorentz_sha256 = None


        manifest_row = {

            "cycle":
                cycle_key,

            "cycle_label":
                cycle_label,

            "chunk_id":
                chunk_id,

            "chunk_start_tai":
                jsoc_time_string(
                    month_start
                ),

            "chunk_end_exclusive_tai":
                jsoc_time_string(
                    month_end
                ),

            "stride_minutes":
                STRIDE_MINUTES,

            "status":
                chunk_status,

            "sharp_source":
                sharp_source,

            "lorentz_source":
                lorentz_source,

            "sharp_rows":
                sharp_rows,

            "lorentz_rows":
                lorentz_rows,

            "sharp_duplicate_keys":
                sharp_duplicates,

            "lorentz_duplicate_keys":
                lorentz_duplicates,

            "sharp_first_trec":
                sharp_first,

            "sharp_last_trec":
                sharp_last,

            "lorentz_first_trec":
                lorentz_first,

            "lorentz_last_trec":
                lorentz_last,

            "sharp_file":
                str(
                    sharp_file
                ),

            "lorentz_file":
                str(
                    lorentz_file
                ),

            "sharp_sha256":
                sharp_sha256,

            "lorentz_sha256":
                lorentz_sha256,

            "sharp_query":
                sharp_query,

            "lorentz_query":
                lorentz_query,

            "error":
                error_message,

            "recorded_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }


        save_manifest_row(
            manifest_row
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

manifest = pd.read_csv(
    MANIFEST_FILE
)


print(
    "\n\n===================================="
)

print(
    "PHASE 9 EXTRACTION SUMMARY"
)

print(
    "===================================="
)


overall_pass = True


for cycle_key in requested_cycles:

    subset = manifest[
        manifest[
            "cycle"
        ]
        ==
        cycle_key
    ].copy()


    expected_contract = (
        contract[
            cycle_key
        ]
    )


    expected_first = (
        contract_tai_to_timestamp(
            expected_contract[
                "first_valid_sample_tai"
            ]
        )
    )


    expected_last = (
        contract_tai_to_timestamp(
            expected_contract[
                "last_valid_sample_tai"
            ]
        )
    )


    expected_months = len(
        list(
            month_iterator(
                expected_first,
                expected_last
            )
        )
    )


    complete_chunks = int(
        (
            subset[
                "status"
            ]
            ==
            "COMPLETE"
        )
        .sum()
    )


    failed_chunks = int(
        (
            subset[
                "status"
            ]
            !=
            "COMPLETE"
        )
        .sum()
    )


    sharp_rows = int(
        pd.to_numeric(
            subset[
                "sharp_rows"
            ],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


    lorentz_rows = int(
        pd.to_numeric(
            subset[
                "lorentz_rows"
            ],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


    sharp_duplicates = int(
        pd.to_numeric(
            subset[
                "sharp_duplicate_keys"
            ],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


    lorentz_duplicates = int(
        pd.to_numeric(
            subset[
                "lorentz_duplicate_keys"
            ],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


    cycle_pass = (
        len(subset)
        ==
        expected_months

        and

        complete_chunks
        ==
        expected_months

        and

        failed_chunks
        ==
        0
    )


    if not cycle_pass:

        overall_pass = False


    print(
        "\n",
        cycle_key.upper(),
        sep=""
    )


    print(
        "Expected monthly chunks:",
        expected_months
    )


    print(
        "Manifest chunks:",
        len(subset)
    )


    print(
        "Complete chunks:",
        complete_chunks
    )


    print(
        "Failed chunks:",
        failed_chunks
    )


    print(
        "SHARP rows:",
        sharp_rows
    )


    print(
        "Lorentz rows:",
        lorentz_rows
    )


    print(
        "SHARP duplicate keys:",
        sharp_duplicates
    )


    print(
        "Lorentz duplicate keys:",
        lorentz_duplicates
    )


    print(
        "Extraction status:",
        (
            "PASS"
            if cycle_pass
            else "INCOMPLETE"
        )
    )


print(
    "\nOverall requested extraction:",
    (
        "PASS"
        if overall_pass
        else "INCOMPLETE"
    )
)


print(
    "\nManifest:"
)

print(
    MANIFEST_FILE
)


print(
    "\nPhase 9 extraction run finished."
)