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


HARPS = [
    12456,
    12471,
    12511,
    12519,
]


for harp in HARPS:

    print("\n====================================")
    print("HARPNUM:", harp)
    print("====================================")

    query = (
        f"hmi.sharp_cea_720s"
        f"[{harp}]"
        f"[2024.12.30_00:00:00_TAI/24h@12m]"
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

    df = client.query(
        query,
        key=keys
    )

    df["missing_count"] = (
        df[SHARP_FEATURES]
        .isna()
        .sum(axis=1)
    )

    problem_rows = df[
        df["missing_count"] > 0
    ]

    print("Problem rows:", len(problem_rows))

    if len(problem_rows) == 0:
        print("No missing-feature rows.")
        continue


    for _, row in problem_rows.iterrows():

        missing_features = [
            feature
            for feature in SHARP_FEATURES
            if pd.isna(row[feature])
        ]

        print("\nTime:", row["T_REC"])
        print("QUALITY:", row["QUALITY"])
        print("CMASK:", row["CMASK"])
        print("LON_FWT:", row["LON_FWT"])
        print("LAT_FWT:", row["LAT_FWT"])

        print(
            "Missing features:",
            missing_features
        )


print("\nDiagnosis finished.")