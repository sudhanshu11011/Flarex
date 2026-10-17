import drms
import pandas as pd
import numpy as np
from pathlib import Path


client = drms.Client()

SERIES = "hmi.sharp_cea_720s"


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


# Cycle 24 audit grid
DATES = []

# Cycle 24 starts in 2010.
# Use available quarterly dates from June onward.
for month in [6, 9, 12]:
    DATES.append(
        f"2010.{month:02d}.15_12:00:00_TAI"
    )

for year in range(2011, 2020):

    for month in [3, 6, 9, 12]:

        DATES.append(
            f"{year}.{month:02d}.15_12:00:00_TAI"
        )


all_rows = []


for time in DATES:

    print("\n====================================")
    print(time)
    print("====================================")

    query = (
        f"{SERIES}"
        f"[]"
        f"[{time}]"
    )

    keys = ",".join(
        [
            "T_REC",
            "HARPNUM",
            "NOAA_AR",
            "QUALITY",
            "CMASK",
            "LON_FWT",
            "LAT_FWT",
        ]
        + SHARP_FEATURES
    )


    try:

        df = client.query(
            query,
            key=keys
        )

    except Exception as error:

        print("QUERY ERROR:", error)
        continue


    if len(df) == 0:

        print("No records.")
        continue


    print("Rows:", len(df))


    # ---------------------------------------
    # Missing SHARP feature flag
    # ---------------------------------------

    df["sharp_missing"] = (
        df[SHARP_FEATURES]
        .isna()
        .any(axis=1)
    )


    # ---------------------------------------
    # CMASK zero flag
    # ---------------------------------------

    df["cmask_zero"] = (
        df["CMASK"]
        .fillna(0)
        .eq(0)
    )


    # ---------------------------------------
    # Position validity
    # ---------------------------------------

    df["position_missing"] = (
        df["LON_FWT"].isna()
        |
        df["LAT_FWT"].isna()
    )


    df["outside_68"] = (
        df["LON_FWT"]
        .abs()
        .gt(68)
    )


    # ---------------------------------------
    # Convert QUALITY safely
    # ---------------------------------------

    df["QUALITY"] = pd.to_numeric(
        df["QUALITY"],
        errors="coerce"
    )


    df["quality_hex"] = (
        df["QUALITY"]
        .apply(
            lambda value:
            hex(int(value))
            if pd.notna(value)
            else "NaN"
        )
    )


    df["audit_time"] = time


    all_rows.append(
        df[
            [
                "audit_time",
                "T_REC",
                "HARPNUM",
                "NOAA_AR",
                "QUALITY",
                "quality_hex",
                "CMASK",
                "LON_FWT",
                "LAT_FWT",
                "sharp_missing",
                "cmask_zero",
                "position_missing",
                "outside_68",
            ]
        ]
    )


# ==================================================
# COMBINE
# ==================================================

audit = pd.concat(
    all_rows,
    ignore_index=True
)


print("\n\n====================================")
print("CYCLE 24 QUALITY VALUE DISTRIBUTION")
print("====================================")


quality_summary = (
    audit
    .groupby(
        [
            "QUALITY",
            "quality_hex",
        ],
        dropna=False
    )
    .agg(
        rows=(
            "HARPNUM",
            "size",
        ),

        sharp_missing_rows=(
            "sharp_missing",
            "sum",
        ),

        cmask_zero_rows=(
            "cmask_zero",
            "sum",
        ),

        position_missing_rows=(
            "position_missing",
            "sum",
        ),

        outside_68_rows=(
            "outside_68",
            "sum",
        ),
    )
    .reset_index()
    .sort_values(
        "rows",
        ascending=False
    )
)


quality_summary["percent"] = (
    quality_summary["rows"]
    / len(audit)
    * 100
)


print(
    quality_summary.to_string(
        index=False
    )
)


# ==================================================
# BASIC COUNTS
# ==================================================

print("\n====================================")
print("BASIC QUALITY COUNTS")
print("====================================")

print(
    "Total rows:",
    len(audit)
)

print(
    "QUALITY == 0:",
    int(
        (audit["QUALITY"] == 0)
        .sum()
    )
)

print(
    "QUALITY != 0:",
    int(
        (
            audit["QUALITY"].notna()
            &
            (audit["QUALITY"] != 0)
        )
        .sum()
    )
)

print(
    "SHARP missing rows:",
    int(
        audit["sharp_missing"].sum()
    )
)

print(
    "CMASK = 0 rows:",
    int(
        audit["cmask_zero"].sum()
    )
)

print(
    "|LON_FWT| > 68 rows:",
    int(
        audit["outside_68"].sum()
    )
)


# ==================================================
# SAVE
# ==================================================

output_dir = Path(
    "data_manifests"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


audit_file = (
    output_dir
    / "cycle24_quality_row_audit.csv"
)

summary_file = (
    output_dir
    / "cycle24_quality_value_summary.csv"
)


audit.to_csv(
    audit_file,
    index=False
)

quality_summary.to_csv(
    summary_file,
    index=False
)


print("\nSaved:")
print(audit_file)
print(summary_file)

print("\nQUALITY audit finished.")