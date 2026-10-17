from pathlib import Path
import subprocess
import hashlib
import re

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_HTTPS = (
    "https://data.ngdc.noaa.gov/platforms/"
    "solar-space-observing-satellites/goes/"
    "multi/l2/data/xrsf-l2-flrpt_science/csv"
)

BASE_HTTP = (
    "http://data.ngdc.noaa.gov/platforms/"
    "solar-space-observing-satellites/goes/"
    "multi/l2/data/xrsf-l2-flrpt_science/csv"
)

PRODUCT_VERSION = "v1-0-1"

YEARS = range(2010, 2020)

C24_START = pd.Timestamp(
    "2010-05-01T00:00:00Z"
)

C24_END = pd.Timestamp(
    "2020-01-01T00:00:00Z"
)


RAW_DIR = Path(
    "data/raw/goes"
)

MANIFEST_DIR = Path(
    "data_manifests"
)


RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MANIFEST_FILE = (
    MANIFEST_DIR
    / "goes_c24_catalog_manifest.csv"
)

AUDIT_FILE = (
    MANIFEST_DIR
    / "goes_c24_label_contract_audit.csv"
)


# ============================================================
# SHA256
# ============================================================

def sha256_file(path: Path) -> str:

    sha = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(
    url: str,
    output_file: Path
) -> bool:

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
# FLARE CLASS PARSER
# ============================================================

def parse_flare_flux(flare_class):

    if pd.isna(flare_class):
        return np.nan


    text = str(
        flare_class
    ).strip().upper()


    match = re.fullmatch(
        r"([ABCMX])\s*([0-9]+(?:\.[0-9]+)?)",
        text
    )


    if not match:
        return np.nan


    letter = match.group(1)

    magnitude = float(
        match.group(2)
    )


    multipliers = {
        "A": 1e-8,
        "B": 1e-7,
        "C": 1e-6,
        "M": 1e-5,
        "X": 1e-4,
    }


    return (
        magnitude
        * multipliers[letter]
    )


# ============================================================
# ACTIVE REGION PARSER
# ============================================================

def parse_active_region(value):

    if pd.isna(value):
        return np.nan


    text = str(
        value
    ).strip()


    if not text:
        return np.nan


    try:

        number = int(
            float(text)
        )

    except Exception:

        return np.nan


    if number <= 0:
        return np.nan


    return number


# ============================================================
# DOWNLOAD / LOAD FILES
# ============================================================

manifest_rows = []

all_frames = []


for year in YEARS:

    source_filename = (
        f"sci_xrsf-l2-flrpt_geo_"
        f"y{year}_{PRODUCT_VERSION}.csv"
    )


    local_file = (
        RAW_DIR
        / f"goes_flare_{year}.csv"
    )


    print(
        "\n===================================="
    )

    print(
        "GOES YEAR:",
        year
    )

    print(
        "===================================="
    )


    # --------------------------------------------
    # DOWNLOAD ONLY IF LOCAL FILE DOES NOT EXIST
    # --------------------------------------------

    if (
        not local_file.exists()
        or
        local_file.stat().st_size == 0
    ):

        https_url = (
            f"{BASE_HTTPS}/"
            f"{source_filename}"
        )

        print(
            "Downloading:",
            https_url
        )


        success = download_file(
            https_url,
            local_file
        )


        if not success:

            http_url = (
                f"{BASE_HTTP}/"
                f"{source_filename}"
            )

            print(
                "HTTPS failed."
            )

            print(
                "Trying HTTP:",
                http_url
            )


            success = download_file(
                http_url,
                local_file
            )


        if not success:

            print(
                "DOWNLOAD FAILED:",
                year
            )

            manifest_rows.append(
                {
                    "year":
                        year,

                    "source_filename":
                        source_filename,

                    "local_file":
                        str(local_file),

                    "download_ok":
                        False,

                    "file_size_bytes":
                        0,

                    "sha256":
                        None,

                    "rows":
                        0,
                }
            )

            continue

    else:

        print(
            "Using existing local file:",
            local_file
        )


    # --------------------------------------------
    # HASH
    # --------------------------------------------

    checksum = sha256_file(
        local_file
    )


    # --------------------------------------------
    # READ CSV
    # --------------------------------------------

    try:

        df = pd.read_csv(
            local_file
        )

    except Exception as error:

        print(
            "CSV READ ERROR:",
            error
        )

        continue


    print(
        "Rows:",
        len(df)
    )


    required = {
        "start_time",
        "flare_class",
        "active_region",
        "xrsb_irrad",
    }


    missing_columns = (
        required
        -
        set(df.columns)
    )


    if missing_columns:

        raise RuntimeError(
            f"{year}: Missing required columns: "
            f"{sorted(missing_columns)}"
        )


    df["catalog_year"] = year

    df["source_file"] = (
        source_filename
    )


    all_frames.append(
        df
    )


    manifest_rows.append(
        {
            "year":
                year,

            "source_filename":
                source_filename,

            "local_file":
                str(local_file),

            "download_ok":
                True,

            "file_size_bytes":
                local_file.stat().st_size,

            "sha256":
                checksum,

            "rows":
                len(df),
        }
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_df = pd.DataFrame(
    manifest_rows
)


manifest_df.to_csv(
    MANIFEST_FILE,
    index=False
)


# ============================================================
# COMBINE CATALOG
# ============================================================

if len(all_frames) == 0:

    raise RuntimeError(
        "No GOES flare catalog files loaded."
    )


goes = pd.concat(
    all_frames,
    ignore_index=True
)


# ============================================================
# PARSE TIMES
# ============================================================

goes["start_time_utc"] = pd.to_datetime(
    goes["start_time"],
    utc=True,
    errors="coerce"
)


# ============================================================
# FILTER TRUE CYCLE-24 WINDOW
# ============================================================

c24 = goes[
    (
        goes[
            "start_time_utc"
        ] >= C24_START
    )
    &
    (
        goes[
            "start_time_utc"
        ] < C24_END
    )
].copy()


# ============================================================
# PARSE CLASS
# ============================================================

c24["class_flux"] = (
    c24[
        "flare_class"
    ]
    .apply(
        parse_flare_flux
    )
)


c24["xrsb_irrad_numeric"] = (
    pd.to_numeric(
        c24["xrsb_irrad"],
        errors="coerce"
    )
)


M1_THRESHOLD = 1e-5


c24["is_m1plus_class"] = (
    c24[
        "class_flux"
    ] >= M1_THRESHOLD
)


c24["is_m1plus_irradiance"] = (
    c24[
        "xrsb_irrad_numeric"
    ] >= M1_THRESHOLD
)


# ============================================================
# ACTIVE REGION
# ============================================================

c24["active_region_parsed"] = (
    c24[
        "active_region"
    ]
    .apply(
        parse_active_region
    )
)


c24["has_active_region"] = (
    c24[
        "active_region_parsed"
    ]
    .notna()
)


c24["active_region_lt_10000"] = (
    c24[
        "active_region_parsed"
    ]
    .notna()
    &
    (
        c24[
            "active_region_parsed"
        ] < 10000
    )
)


c24["active_region_ge_10000"] = (
    c24[
        "active_region_parsed"
    ]
    .notna()
    &
    (
        c24[
            "active_region_parsed"
        ] >= 10000
    )
)


# ============================================================
# CLASS vs IRRADIANCE CONSISTENCY
# ============================================================

comparable = (
    c24[
        "class_flux"
    ].notna()
    &
    c24[
        "xrsb_irrad_numeric"
    ].notna()
)


c24["m1_threshold_disagreement"] = False


c24.loc[
    comparable,
    "m1_threshold_disagreement"
] = (

    c24.loc[
        comparable,
        "is_m1plus_class"
    ]
    !=
    c24.loc[
        comparable,
        "is_m1plus_irradiance"
    ]
)


# ============================================================
# M1+ SUBSET
# ============================================================

m1plus = c24[
    c24[
        "is_m1plus_class"
    ]
].copy()


# ============================================================
# DUPLICATE EVENT CHECK
# ============================================================

duplicate_columns = [
    "start_time_utc",
    "flare_class",
    "active_region_parsed",
]


duplicate_events = int(
    c24[
        duplicate_columns
    ]
    .duplicated()
    .sum()
)


m1_duplicate_events = int(
    m1plus[
        duplicate_columns
    ]
    .duplicated()
    .sum()
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n\n===================================="
)

print(
    "QC-8A CYCLE 24 GOES LABEL AUDIT"
)

print(
    "===================================="
)


print(
    "Cycle24 catalog rows:",
    len(c24)
)


print(
    "Invalid start_time rows:",
    int(
        c24[
            "start_time_utc"
        ]
        .isna()
        .sum()
    )
)


print(
    "Unparseable flare_class rows:",
    int(
        c24[
            "class_flux"
        ]
        .isna()
        .sum()
    )
)


print(
    "M1+ events:",
    len(m1plus)
)


print(
    "M1+ with active_region:",
    int(
        m1plus[
            "has_active_region"
        ].sum()
    )
)


print(
    "M1+ without active_region:",
    int(
        (
            ~m1plus[
                "has_active_region"
            ]
        ).sum()
    )
)


print(
    "M1+ active_region < 10000:",
    int(
        m1plus[
            "active_region_lt_10000"
        ].sum()
    )
)


print(
    "M1+ active_region >= 10000:",
    int(
        m1plus[
            "active_region_ge_10000"
        ].sum()
    )
)


print(
    "Class/irradiance M1 threshold disagreements:",
    int(
        c24[
            "m1_threshold_disagreement"
        ].sum()
    )
)


print(
    "Duplicate events:",
    duplicate_events
)


print(
    "Duplicate M1+ events:",
    m1_duplicate_events
)


# ============================================================
# YEAR-LEVEL SUMMARY
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "YEAR-LEVEL M1+ SUMMARY"
)

print(
    "------------------------------------"
)


year_summary = (
    c24
    .groupby(
        c24[
            "start_time_utc"
        ].dt.year
    )
    .agg(
        flare_rows=(
            "flare_class",
            "size"
        ),

        m1plus_events=(
            "is_m1plus_class",
            "sum"
        ),

        region_known=(
            "has_active_region",
            "sum"
        ),

        region_lt_10000=(
            "active_region_lt_10000",
            "sum"
        ),

        threshold_disagreements=(
            "m1_threshold_disagreement",
            "sum"
        ),
    )
    .reset_index(
        names="year"
    )
)


# Region availability specifically among M1+
m1_year = (
    m1plus
    .groupby(
        m1plus[
            "start_time_utc"
        ].dt.year
    )
    .agg(
        m1plus_region_known=(
            "has_active_region",
            "sum"
        ),

        m1plus_region_missing=(
            "has_active_region",
            lambda x:
            int(
                (~x).sum()
            )
        ),
    )
    .reset_index(
        names="year"
    )
)


year_summary = year_summary.merge(
    m1_year,
    on="year",
    how="left"
)


year_summary = year_summary.fillna(
    0
)


print(
    year_summary.to_string(
        index=False
    )
)


# ============================================================
# ACTIVE REGION FORMAT SUMMARY
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "M1+ ACTIVE REGION FORMAT"
)

print(
    "------------------------------------"
)


region_known_m1 = m1plus[
    m1plus[
        "has_active_region"
    ]
]


if len(region_known_m1) > 0:

    print(
        "Minimum active_region:",
        int(
            region_known_m1[
                "active_region_parsed"
            ].min()
        )
    )


    print(
        "Maximum active_region:",
        int(
            region_known_m1[
                "active_region_parsed"
            ].max()
        )
    )


    print(
        "\nFirst 30 M1+ region values:"
    )


    print(
        region_known_m1[
            [
                "start_time_utc",
                "flare_class",
                "active_region_parsed",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# UNASSIGNED M1+ EVENTS
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "M1+ EVENTS WITHOUT ACTIVE REGION"
)

print(
    "------------------------------------"
)


unassigned = m1plus[
    ~m1plus[
        "has_active_region"
    ]
]


print(
    "Count:",
    len(unassigned)
)


if len(unassigned) > 0:

    print(
        unassigned[
            [
                "start_time_utc",
                "flare_class",
                "xrsb_irrad_numeric",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# THRESHOLD DISAGREEMENTS
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "CLASS vs XRSB THRESHOLD DISAGREEMENTS"
)

print(
    "------------------------------------"
)


disagreements = c24[
    c24[
        "m1_threshold_disagreement"
    ]
]


print(
    "Count:",
    len(disagreements)
)


if len(disagreements) > 0:

    print(
        disagreements[
            [
                "start_time_utc",
                "flare_class",
                "class_flux",
                "xrsb_irrad_numeric",
                "active_region",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE AUDIT ROWS
# ============================================================

audit_columns = [
    "catalog_year",
    "source_file",
    "start_time",
    "start_time_utc",
    "flare_class",
    "class_flux",
    "xrsb_irrad",
    "xrsb_irrad_numeric",
    "active_region",
    "active_region_parsed",
    "has_active_region",
    "is_m1plus_class",
    "is_m1plus_irradiance",
    "m1_threshold_disagreement",
]


c24[
    audit_columns
].to_csv(
    AUDIT_FILE,
    index=False
)


print(
    "\nSaved:"
)

print(
    MANIFEST_FILE
)

print(
    AUDIT_FILE
)


print(
    "\nQC-8A GOES label audit finished."
)