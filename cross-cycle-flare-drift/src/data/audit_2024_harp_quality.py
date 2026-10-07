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


# NOAA-linked HARPs visible on 2024-12-30
HARPS = [
    12456,
    12471,
    12477,
    12479,
    12481,
    12492,
    12506,
    12511,
    12515,
    12519,
]


summary = []


for harp in HARPS:

    print("\n===================================")
    print("HARPNUM:", harp)
    print("===================================")

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

    try:

        df = client.query(
            query,
            key=keys
        )

        total_rows = len(df)

        if total_rows == 0:
            print("No rows found.")

            summary.append(
                {
                    "HARPNUM": harp,
                    "rows": 0,
                    "cmask_zero_rows": 0,
                    "cmask_zero_percent": 0,
                    "missing_rows": 0,
                    "missing_percent": 0,
                }
            )

            continue


        # Missing feature count per row
        df["missing_count"] = (
            df[SHARP_FEATURES]
            .isna()
            .sum(axis=1)
        )


        # CMASK = 0
        cmask_zero_rows = (
            df["CMASK"]
            .fillna(0)
            .eq(0)
            .sum()
        )


        # Missing rows
        missing_rows = (
            df["missing_count"] > 0
        ).sum()


        cmask_zero_percent = (
            cmask_zero_rows
            / total_rows
            * 100
        )


        missing_percent = (
            missing_rows
            / total_rows
            * 100
        )


        noaa_region = (
            df["NOAA_AR"]
            .dropna()
            .iloc[0]
            if df["NOAA_AR"].notna().any()
            else None
        )


        print("NOAA AR:", noaa_region)
        print("Total rows:", total_rows)

        print(
            "CMASK = 0:",
            cmask_zero_rows,
            f"({cmask_zero_percent:.2f}%)"
        )

        print(
            "Rows with missing features:",
            missing_rows,
            f"({missing_percent:.2f}%)"
        )


        summary.append(
            {
                "HARPNUM": harp,
                "NOAA_AR": noaa_region,
                "rows": total_rows,
                "cmask_zero_rows": cmask_zero_rows,
                "cmask_zero_percent": round(
                    cmask_zero_percent, 2
                ),
                "missing_rows": missing_rows,
                "missing_percent": round(
                    missing_percent, 2
                ),
            }
        )

    except Exception as error:

        print("ERROR:", error)


# -----------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------

summary_df = pd.DataFrame(summary)


print("\n\n===================================")
print("2024 MULTI-HARP QUALITY SUMMARY")
print("===================================")

print(
    summary_df.to_string(
        index=False
    )
)


print("\nAudit finished.")