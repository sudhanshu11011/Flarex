import drms


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


LORENTZ_FEATURES = [
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
]


TESTS = [
    {
        "name": "Cycle 24 - 2011",
        "harp": 401,
        "time": "2011.03.09_23:00:00_TAI",
    },
    {
        "name": "Cycle 25 - 2020",
        "harp": 7451,
        "time": "2020.09.27_00:00:00_TAI",
    },
    {
        "name": "Cycle 25 - 2024",
        "harp": 12519,
        "time": "2024.12.30_22:24:00_TAI",
    },
]


for test in TESTS:

    print("\n====================================")
    print(test["name"])
    print("====================================")

    harp = test["harp"]
    time = test["time"]

    sharp_query = (
        f"hmi.sharp_cea_720s"
        f"[{harp}]"
        f"[{time}/1h@12m]"
    )

    lorentz_query = (
        f"cgem.lorentz"
        f"[{harp}]"
        f"[{time}/1h@12m]"
    )


    sharp_keys = ",".join(
        [
            "T_REC",
            "HARPNUM",
            "NOAA_AR",
            "QUALITY",
        ]
        + SHARP_FEATURES
    )

    lorentz_keys = ",".join(
        [
            "T_REC",
            "HARPNUM",
        ]
        + LORENTZ_FEATURES
    )


    # Download both
    sharp_df = client.query(
        sharp_query,
        key=sharp_keys
    )

    lorentz_df = client.query(
        lorentz_query,
        key=lorentz_keys
    )


    print("SHARP rows:", len(sharp_df))
    print("Lorentz rows:", len(lorentz_df))


    # Merge
    merged_df = sharp_df.merge(
        lorentz_df,
        on=["HARPNUM", "T_REC"],
        how="inner",
        validate="one_to_one"
    )


    print("Merged rows:", len(merged_df))


    # Check missing values in 25 features
    all_features = SHARP_FEATURES + LORENTZ_FEATURES

    missing_count = (
        merged_df[all_features]
        .isna()
        .sum()
        .sum()
    )


    print("Missing feature values:", missing_count)


    if (
        len(sharp_df) == 5
        and len(lorentz_df) == 5
        and len(merged_df) == 5
        and missing_count == 0
    ):
        print("STATUS: PASS ✅")

    else:
        print("STATUS: CHECK REQUIRED ⚠️")


print("\nCross-cycle alignment test finished.")