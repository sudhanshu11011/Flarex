import drms
import pandas as pd


client = drms.Client()

SERIES = "hmi.sharp_cea_720s"


FEATURES = [
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


# Representative dates only.
# This is still an audit, NOT final dataset extraction.
DATES = [
    ("Cycle24_2011", "2011.03.09_12:00:00_TAI"),
    ("Cycle24_2014", "2014.10.24_12:00:00_TAI"),
    ("Cycle24_2017", "2017.09.06_12:00:00_TAI"),

    ("Cycle25_2020", "2020.09.27_12:00:00_TAI"),
    ("Cycle25_2022", "2022.03.30_12:00:00_TAI"),
    ("Cycle25_2024", "2024.12.30_12:00:00_TAI"),
]


summary = []


for name, time in DATES:

    print("\n====================================")
    print(name)
    print("====================================")

    # All HARPs existing at one timestamp
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
        ]
        + FEATURES
    )

    df = client.query(
        query,
        key=keys
    )

    print("Total HARPs:", len(df))

    if len(df) == 0:
        continue


    # Missing count for every feature
    missing = df[FEATURES].isna().sum()

    print("\nFeature missingness:")

    for feature in FEATURES:

        count = int(missing[feature])

        percent = (
            count / len(df) * 100
        )

        print(
            f"{feature:10s} "
            f"{count:3d} "
            f"({percent:.2f}%)"
        )


    # Rows missing at least one required feature
    rows_with_missing = (
        df[FEATURES]
        .isna()
        .any(axis=1)
        .sum()
    )

    missing_row_percent = (
        rows_with_missing
        / len(df)
        * 100
    )


    # CMASK zero
    cmask_zero = (
        df["CMASK"]
        .fillna(0)
        .eq(0)
        .sum()
    )


    print(
        "\nRows with >=1 missing feature:",
        rows_with_missing,
        f"({missing_row_percent:.2f}%)"
    )

    print(
        "CMASK = 0 rows:",
        cmask_zero
    )


    summary.append(
        {
            "sample": name,
            "total_harps": len(df),
            "rows_with_missing": int(
                rows_with_missing
            ),
            "missing_percent": round(
                missing_row_percent, 2
            ),
            "cmask_zero": int(cmask_zero),
        }
    )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

summary_df = pd.DataFrame(summary)

print("\n\n====================================")
print("CROSS-CYCLE MISSINGNESS SUMMARY")
print("====================================")

print(
    summary_df.to_string(
        index=False
    )
)