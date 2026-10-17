import drms
import pandas as pd
from pathlib import Path


client = drms.Client()


SHARP_SERIES = "hmi.sharp_cea_720s"
LORENTZ_SERIES = "cgem.lorentz"


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


ALL_FEATURES = (
    SHARP_FEATURES
    + LORENTZ_FEATURES
)


MONTH_DAYS = [
    (3, 15),
    (6, 15),
    (9, 15),
    (12, 15),
]


summary = []


for year in range(2011, 2026):

    cycle = (
        "Cycle24"
        if year <= 2019
        else "Cycle25"
    )

    for month, day in MONTH_DAYS:

        time = (
            f"{year}.{month:02d}.{day:02d}"
            "_12:00:00_TAI"
        )

        print("\n====================================")
        print(cycle, "-", time)
        print("====================================")


        sharp_query = (
            f"{SHARP_SERIES}"
            f"[]"
            f"[{time}]"
        )

        lorentz_query = (
            f"{LORENTZ_SERIES}"
            f"[]"
            f"[{time}]"
        )


        sharp_keys = ",".join(
            [
                "T_REC",
                "HARPNUM",
                "NOAA_AR",
                "QUALITY",
                "CMASK",
                "LON_FWT",
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


        try:

            sharp = client.query(
                sharp_query,
                key=sharp_keys
            )

            lorentz = client.query(
                lorentz_query,
                key=lorentz_keys
            )

        except Exception as error:

            print("QUERY ERROR:", error)

            summary.append(
                {
                    "cycle": cycle,
                    "year": year,
                    "date": time,

                    "sharp_rows": 0,
                    "lorentz_rows": 0,

                    "matched_rows": 0,

                    "sharp_only": 0,
                    "lorentz_only": 0,

                    "complete_25": 0,
                    "incomplete_25": 0,

                    "merge_rate": None,
                    "complete_25_rate": None,

                    "sharp_duplicates": 0,
                    "lorentz_duplicates": 0,

                    "query_error": True,
                }
            )

            continue


        sharp_count = len(sharp)
        lorentz_count = len(lorentz)


        print("SHARP rows:", sharp_count)
        print("Lorentz rows:", lorentz_count)


        # ==================================================
        # HANDLE EMPTY SNAPSHOTS
        # ==================================================

        if sharp_count == 0 or lorentz_count == 0:

            if sharp_count == 0 and lorentz_count == 0:

                print(
                    "No SHARP or Lorentz records "
                    "at this timestamp."
                )

            elif sharp_count == 0:

                print(
                    "No SHARP records "
                    "at this timestamp."
                )

            else:

                print(
                    "No Lorentz records "
                    "at this timestamp."
                )


            summary.append(
                {
                    "cycle": cycle,
                    "year": year,
                    "date": time,

                    "sharp_rows": sharp_count,
                    "lorentz_rows": lorentz_count,

                    "matched_rows": 0,

                    "sharp_only": (
                        sharp_count
                        if lorentz_count == 0
                        else 0
                    ),

                    "lorentz_only": (
                        lorentz_count
                        if sharp_count == 0
                        else 0
                    ),

                    "complete_25": 0,
                    "incomplete_25": 0,

                    "merge_rate": (
                        0.0
                        if sharp_count > 0
                        else None
                    ),

                    "complete_25_rate": (
                        0.0
                        if sharp_count > 0
                        else None
                    ),

                    "sharp_duplicates": 0,
                    "lorentz_duplicates": 0,

                    "query_error": False,
                }
            )

            continue


        # ==================================================
        # DUPLICATE CHECKS
        # ==================================================

        sharp_duplicates = int(
            sharp[
                [
                    "HARPNUM",
                    "T_REC",
                ]
            ]
            .duplicated()
            .sum()
        )


        lorentz_duplicates = int(
            lorentz[
                [
                    "HARPNUM",
                    "T_REC",
                ]
            ]
            .duplicated()
            .sum()
        )


        print(
            "SHARP duplicate keys:",
            sharp_duplicates
        )

        print(
            "Lorentz duplicate keys:",
            lorentz_duplicates
        )


        # ==================================================
        # MERGE SHARP + LORENTZ
        # ==================================================

        try:

            merged = sharp.merge(
                lorentz,
                on=[
                    "HARPNUM",
                    "T_REC",
                ],
                how="inner",
                validate="one_to_one",
            )

        except Exception as error:

            print("MERGE ERROR:", error)

            summary.append(
                {
                    "cycle": cycle,
                    "year": year,
                    "date": time,

                    "sharp_rows": sharp_count,
                    "lorentz_rows": lorentz_count,

                    "matched_rows": 0,

                    "sharp_only": 0,
                    "lorentz_only": 0,

                    "complete_25": 0,
                    "incomplete_25": 0,

                    "merge_rate": None,
                    "complete_25_rate": None,

                    "sharp_duplicates":
                        sharp_duplicates,

                    "lorentz_duplicates":
                        lorentz_duplicates,

                    "query_error": True,
                }
            )

            continue


        matched = len(merged)


        # ==================================================
        # FIND SHARP-ONLY / LORENTZ-ONLY HARPs
        # ==================================================

        sharp_harps = set(
            sharp["HARPNUM"]
            .dropna()
            .astype(int)
        )

        lorentz_harps = set(
            lorentz["HARPNUM"]
            .dropna()
            .astype(int)
        )


        sharp_only_harps = (
            sharp_harps
            - lorentz_harps
        )


        lorentz_only_harps = (
            lorentz_harps
            - sharp_harps
        )


        # ==================================================
        # COMPLETE 25-FEATURE ROW CHECK
        # ==================================================

        incomplete_mask = (
            merged[ALL_FEATURES]
            .isna()
            .any(axis=1)
        )


        incomplete_25 = int(
            incomplete_mask.sum()
        )


        complete_25 = (
            matched
            - incomplete_25
        )


        # ==================================================
        # COVERAGE RATES
        # ==================================================

        merge_rate = (
            matched
            / sharp_count
            * 100
        )


        complete_rate = (
            complete_25
            / sharp_count
            * 100
        )


        print(
            "Matched rows:",
            matched
        )


        print(
            "SHARP-only HARPs:",
            len(sharp_only_harps)
        )


        print(
            "Lorentz-only HARPs:",
            len(lorentz_only_harps)
        )


        print(
            "Complete 25-feature rows:",
            complete_25
        )


        print(
            "Incomplete matched rows:",
            incomplete_25
        )


        print(
            "Merge coverage:",
            f"{merge_rate:.2f}%"
        )


        print(
            "Complete-25 coverage:",
            f"{complete_rate:.2f}%"
        )


        summary.append(
            {
                "cycle": cycle,
                "year": year,
                "date": time,

                "sharp_rows": sharp_count,
                "lorentz_rows": lorentz_count,

                "matched_rows": matched,

                "sharp_only": len(
                    sharp_only_harps
                ),

                "lorentz_only": len(
                    lorentz_only_harps
                ),

                "complete_25": complete_25,

                "incomplete_25":
                    incomplete_25,

                "merge_rate": round(
                    merge_rate,
                    2
                ),

                "complete_25_rate": round(
                    complete_rate,
                    2
                ),

                "sharp_duplicates":
                    sharp_duplicates,

                "lorentz_duplicates":
                    lorentz_duplicates,

                "query_error": False,
            }
        )


# ==================================================
# FULL SUMMARY
# ==================================================

summary_df = pd.DataFrame(
    summary
)


print("\n\n====================================")
print("FULL SHARP + LORENTZ MERGE AUDIT")
print("====================================")


print(
    summary_df.to_string(
        index=False
    )
)


# ==================================================
# VALID ROWS ONLY
# ==================================================

valid = summary_df[
    summary_df["query_error"] == False
].copy()


# ==================================================
# CYCLE-LEVEL SUMMARY
# ==================================================

cycle_summary = (
    valid
    .groupby("cycle")
    .agg(
        snapshots=(
            "date",
            "count",
        ),

        sharp_rows=(
            "sharp_rows",
            "sum",
        ),

        lorentz_rows=(
            "lorentz_rows",
            "sum",
        ),

        matched_rows=(
            "matched_rows",
            "sum",
        ),

        complete_25=(
            "complete_25",
            "sum",
        ),

        incomplete_25=(
            "incomplete_25",
            "sum",
        ),

        sharp_only=(
            "sharp_only",
            "sum",
        ),

        lorentz_only=(
            "lorentz_only",
            "sum",
        ),
    )
    .reset_index()
)


cycle_summary["merge_rate"] = (
    cycle_summary["matched_rows"]
    / cycle_summary["sharp_rows"]
    * 100
)


cycle_summary["complete_25_rate"] = (
    cycle_summary["complete_25"]
    / cycle_summary["sharp_rows"]
    * 100
)


print("\n\n====================================")
print("CYCLE-LEVEL MERGE SUMMARY")
print("====================================")


print(
    cycle_summary.to_string(
        index=False
    )
)


# ==================================================
# SAVE RESULTS
# ==================================================

output_dir = Path(
    "data_manifests"
)


output_dir.mkdir(
    parents=True,
    exist_ok=True
)


output_file = (
    output_dir
    / "sharp_lorentz_merge_audit.csv"
)


summary_df.to_csv(
    output_file,
    index=False
)


cycle_output_file = (
    output_dir
    / "sharp_lorentz_merge_cycle_summary.csv"
)


cycle_summary.to_csv(
    cycle_output_file,
    index=False
)


print("\nSaved:")
print(output_file)
print(cycle_output_file)


print("\nMerge audit finished.")