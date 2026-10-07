import drms
import pandas as pd


client = drms.Client()


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


TESTS = [
    {
        "name": "Cycle 24 - 2011",
        "harp": 401,
        "time": "2011.03.09_00:00:00_TAI",
    },
    {
        "name": "Cycle 25 - 2020",
        "harp": 7451,
        "time": "2020.09.27_00:00:00_TAI",
    },
    {
        "name": "Cycle 25 - 2024",
        "harp": 12519,
        "time": "2024.12.30_00:00:00_TAI",
    },
]


for test in TESTS:

    print("\n======================================")
    print(test["name"])
    print("======================================")

    query = (
        f"hmi.sharp_cea_720s"
        f"[{test['harp']}]"
        f"[{test['time']}/24h@12m]"
    )

    keys = ",".join(
        [
            "T_REC",
            "HARPNUM",
            "NOAA_AR",
            "QUALITY",
            "CMASK",
            "LAT_FWT",
            "LON_FWT",
        ]
        + SHARP_FEATURES
    )

    df = client.query(
        query,
        key=keys
    )

    # Count missing features per row
    df["missing_count"] = (
        df[SHARP_FEATURES]
        .isna()
        .sum(axis=1)
    )

    # Absolute longitude
    df["abs_lon"] = df["LON_FWT"].abs()

    clean_rows = df[
        df["missing_count"] == 0
    ]

    missing_rows = df[
        df["missing_count"] > 0
    ]


    print("Total rows:", len(df))

    print(
        "Rows with missing features:",
        len(missing_rows)
    )

    print(
        "Longitude range:",
        round(df["LON_FWT"].min(), 2),
        "to",
        round(df["LON_FWT"].max(), 2)
    )


    if len(clean_rows) > 0:

        print(
            "Average |longitude| for CLEAN rows:",
            round(
                clean_rows["abs_lon"].mean(),
                2
            )
        )


    if len(missing_rows) > 0:

        print(
            "Average |longitude| for MISSING rows:",
            round(
                missing_rows["abs_lon"].mean(),
                2
            )
        )


        print("\nFirst missing rows:")

        print(
            missing_rows[
                [
                    "T_REC",
                    "LON_FWT",
                    "LAT_FWT",
                    "CMASK",
                    "missing_count",
                ]
            ].head(15).to_string(index=False)
        )


print("\nPosition diagnosis finished.")