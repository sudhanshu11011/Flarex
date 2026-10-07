import drms
import pandas as pd


# --------------------------------------------------
# 1. JSOC CLIENT
# --------------------------------------------------

client = drms.Client()


# --------------------------------------------------
# 2. 18 SHARP FEATURES
# --------------------------------------------------

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


# --------------------------------------------------
# 3. SAMPLE WINDOWS
# --------------------------------------------------

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


# --------------------------------------------------
# 4. RESULT STORAGE
# --------------------------------------------------

summary_rows = []


# --------------------------------------------------
# 5. CHECK EACH WINDOW
# --------------------------------------------------

for test in TESTS:

    print("\n========================================")
    print(test["name"])
    print("========================================")

    harp = test["harp"]
    time = test["time"]


    query = (
        f"hmi.sharp_cea_720s"
        f"[{harp}]"
        f"[{time}/24h@12m]"
    )


    keys = ",".join(
        [
            "T_REC",
            "HARPNUM",
            "NOAA_AR",
            "QUALITY",
            "CMASK",
        ]
        + SHARP_FEATURES
    )


    df = client.query(
        query,
        key=keys
    )


    # ----------------------------------------------
    # BASIC COUNTS
    # ----------------------------------------------

    total_rows = len(df)

    cmask_zero_rows = (
        df["CMASK"].fillna(0) == 0
    ).sum()


    # ----------------------------------------------
    # MISSING FEATURE COUNT PER ROW
    # ----------------------------------------------

    df["missing_feature_count"] = (
        df[SHARP_FEATURES]
        .isna()
        .sum(axis=1)
    )


    rows_with_missing = (
        df["missing_feature_count"] > 0
    ).sum()


    total_missing_values = (
        df[SHARP_FEATURES]
        .isna()
        .sum()
        .sum()
    )


    # ----------------------------------------------
    # PERCENTAGES
    # ----------------------------------------------

    if total_rows > 0:

        cmask_zero_percent = (
            cmask_zero_rows
            / total_rows
            * 100
        )

        missing_row_percent = (
            rows_with_missing
            / total_rows
            * 100
        )

    else:

        cmask_zero_percent = 0
        missing_row_percent = 0


    # ----------------------------------------------
    # PRINT RESULT
    # ----------------------------------------------

    print("HARPNUM:", harp)
    print("Total rows:", total_rows)

    print(
        "CMASK = 0 rows:",
        cmask_zero_rows
    )

    print(
        "CMASK = 0 percentage:",
        round(cmask_zero_percent, 2),
        "%"
    )

    print(
        "Rows with missing features:",
        rows_with_missing
    )

    print(
        "Rows with missing features percentage:",
        round(missing_row_percent, 2),
        "%"
    )

    print(
        "Total missing feature values:",
        total_missing_values
    )


    # ----------------------------------------------
    # RELATION CHECK
    # ----------------------------------------------

    problem_rows = df[
        df["missing_feature_count"] > 0
    ]


    if len(problem_rows) > 0:

        all_problem_rows_have_zero_cmask = (
            problem_rows["CMASK"]
            .fillna(0)
            .eq(0)
            .all()
        )

    else:

        all_problem_rows_have_zero_cmask = True


    print(
        "All missing rows have CMASK = 0:",
        all_problem_rows_have_zero_cmask
    )


    # ----------------------------------------------
    # SAVE SUMMARY
    # ----------------------------------------------

    summary_rows.append(
        {
            "sample": test["name"],
            "harp": harp,
            "total_rows": total_rows,
            "cmask_zero_rows": cmask_zero_rows,
            "cmask_zero_percent": round(
                cmask_zero_percent,
                2
            ),
            "rows_with_missing": rows_with_missing,
            "missing_row_percent": round(
                missing_row_percent,
                2
            ),
            "total_missing_values": int(
                total_missing_values
            ),
        }
    )


# --------------------------------------------------
# 6. FINAL SUMMARY TABLE
# --------------------------------------------------

summary_df = pd.DataFrame(summary_rows)


print("\n\n========================================")
print("FINAL QUALITY AUDIT SUMMARY")
print("========================================")

print(
    summary_df.to_string(
        index=False
    )
)


print("\nAudit finished.")