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


# Fixed dates every year
MONTH_DAYS = [
    (3, 15),
    (6, 15),
    (9, 15),
    (12, 15),
]


summary = []


for year in range(2011, 2026):

    cycle = "Cycle24" if year <= 2019 else "Cycle25"

    for month, day in MONTH_DAYS:

        time = (
            f"{year}.{month:02d}.{day:02d}"
            f"_12:00:00_TAI"
        )

        print("\n====================================")
        print(cycle, "-", time)
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
            ]
            + FEATURES
        )

        try:

            df = client.query(
                query,
                key=keys
            )

        except Exception as error:

            print("QUERY ERROR:", error)

            summary.append(
                {
                    "cycle": cycle,
                    "year": year,
                    "date": time,
                    "total_harps": 0,
                    "missing_rows": 0,
                    "cmask_zero_rows": 0,
                    "query_error": True,
                }
            )

            continue


        total = len(df)

        print("HARPs:", total)


        if total == 0:

            summary.append(
                {
                    "cycle": cycle,
                    "year": year,
                    "date": time,
                    "total_harps": 0,
                    "missing_rows": 0,
                    "cmask_zero_rows": 0,
                    "query_error": False,
                }
            )

            continue


        # Any required SHARP feature missing
        missing_mask = (
            df[FEATURES]
            .isna()
            .any(axis=1)
        )

        missing_rows = int(
            missing_mask.sum()
        )


        # CMASK = 0
        cmask_zero_rows = int(
            df["CMASK"]
            .fillna(0)
            .eq(0)
            .sum()
        )


        missing_percent = (
            missing_rows
            / total
            * 100
        )


        cmask_percent = (
            cmask_zero_rows
            / total
            * 100
        )


        print(
            "Rows with missing features:",
            missing_rows,
            f"({missing_percent:.2f}%)"
        )

        print(
            "CMASK = 0:",
            cmask_zero_rows,
            f"({cmask_percent:.2f}%)"
        )


        summary.append(
            {
                "cycle": cycle,
                "year": year,
                "date": time,
                "total_harps": total,
                "missing_rows": missing_rows,
                "missing_percent": round(
                    missing_percent, 2
                ),
                "cmask_zero_rows": cmask_zero_rows,
                "cmask_zero_percent": round(
                    cmask_percent, 2
                ),
                "query_error": False,
            }
        )


# ----------------------------------------------
# FULL TABLE
# ----------------------------------------------

summary_df = pd.DataFrame(summary)


print("\n\n====================================")
print("FULL QUALITY GRID")
print("====================================")

print(
    summary_df.to_string(
        index=False
    )
)


# ----------------------------------------------
# CYCLE-LEVEL SUMMARY
# ----------------------------------------------

valid_df = summary_df[
    summary_df["query_error"] == False
].copy()


cycle_summary = (
    valid_df
    .groupby("cycle")
    .agg(
        snapshots=("date", "count"),
        total_harps=("total_harps", "sum"),
        missing_rows=("missing_rows", "sum"),
        cmask_zero_rows=("cmask_zero_rows", "sum"),
    )
    .reset_index()
)


cycle_summary["missing_percent"] = (
    cycle_summary["missing_rows"]
    / cycle_summary["total_harps"]
    * 100
)


cycle_summary["cmask_zero_percent"] = (
    cycle_summary["cmask_zero_rows"]
    / cycle_summary["total_harps"]
    * 100
)


print("\n\n====================================")
print("CYCLE-LEVEL QUALITY SUMMARY")
print("====================================")

print(
    cycle_summary.to_string(
        index=False
    )
)


# ----------------------------------------------
# SAVE AUDIT
# ----------------------------------------------

output_file = (
    "data_manifests/"
    "cross_cycle_quality_grid.csv"
)

summary_df.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)