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


MONTH_DAYS = [
    (3, 15),
    (6, 15),
    (9, 15),
    (12, 15),
]


summary = []


for year in range(2011, 2020):

    for month, day in MONTH_DAYS:

        time = (
            f"{year}.{month:02d}.{day:02d}"
            "_12:00:00_TAI"
        )

        print("\n====================================")
        print("Cycle24 -", time)
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
            continue


        if len(sharp) == 0:

            print("No SHARP records.")

            continue


        # ---------------------------------------
        # SHARP completeness
        # ---------------------------------------

        sharp_complete_mask = (
            ~sharp[SHARP_FEATURES]
            .isna()
            .any(axis=1)
        )


        sharp_complete = int(
            sharp_complete_mask.sum()
        )


        # ---------------------------------------
        # Merge using LEFT JOIN
        # ---------------------------------------

        if len(lorentz) > 0:

            merged = sharp.merge(
                lorentz,
                on=[
                    "HARPNUM",
                    "T_REC",
                ],
                how="left",
                indicator=True,
                validate="one_to_one",
            )

        else:

            merged = sharp.copy()

            for feature in LORENTZ_FEATURES:
                merged[feature] = pd.NA

            merged["_merge"] = "left_only"


        # ---------------------------------------
        # Determine completeness
        # ---------------------------------------

        sharp_missing = (
            merged[SHARP_FEATURES]
            .isna()
            .any(axis=1)
        )


        lorentz_missing = (
            merged[LORENTZ_FEATURES]
            .isna()
            .any(axis=1)
        )


        lorentz_record_missing = (
            merged["_merge"] == "left_only"
        )


        complete_25 = (
            (~sharp_missing)
            &
            (~lorentz_missing)
        )


        # ---------------------------------------
        # Missingness categories
        # ---------------------------------------

        sharp_only_problem = (
            sharp_missing
            &
            (~lorentz_missing)
        )


        lorentz_only_problem = (
            (~sharp_missing)
            &
            lorentz_missing
        )


        both_problem = (
            sharp_missing
            &
            lorentz_missing
        )


        total = len(merged)


        print("Total SHARP rows:", total)

        print(
            "18-SHARP complete:",
            int((~sharp_missing).sum())
        )

        print(
            "25-feature complete:",
            int(complete_25.sum())
        )

        print(
            "SHARP-only problem:",
            int(sharp_only_problem.sum())
        )

        print(
            "Lorentz-only problem:",
            int(lorentz_only_problem.sum())
        )

        print(
            "Both sources problem:",
            int(both_problem.sum())
        )

        print(
            "Missing Lorentz records:",
            int(lorentz_record_missing.sum())
        )


        summary.append(
            {
                "year": year,
                "date": time,

                "total_sharp_rows": total,

                "sharp18_complete":
                    int((~sharp_missing).sum()),

                "complete25":
                    int(complete_25.sum()),

                "sharp_only_problem":
                    int(sharp_only_problem.sum()),

                "lorentz_only_problem":
                    int(lorentz_only_problem.sum()),

                "both_problem":
                    int(both_problem.sum()),

                "missing_lorentz_record":
                    int(lorentz_record_missing.sum()),
            }
        )


# ==================================================
# SUMMARY
# ==================================================

summary_df = pd.DataFrame(summary)


totals = {
    "total_sharp_rows":
        summary_df["total_sharp_rows"].sum(),

    "sharp18_complete":
        summary_df["sharp18_complete"].sum(),

    "complete25":
        summary_df["complete25"].sum(),

    "sharp_only_problem":
        summary_df["sharp_only_problem"].sum(),

    "lorentz_only_problem":
        summary_df["lorentz_only_problem"].sum(),

    "both_problem":
        summary_df["both_problem"].sum(),

    "missing_lorentz_record":
        summary_df["missing_lorentz_record"].sum(),
}


total = totals["total_sharp_rows"]


print("\n\n====================================")
print("CYCLE 24 FEATURE CONTRACT SUMMARY")
print("====================================")


for key, value in totals.items():

    print(
        f"{key:25s}: {value}"
    )


print("\nRetention rates:")


print(
    "18 SHARP features:",
    f"{totals['sharp18_complete'] / total * 100:.2f}%"
)


print(
    "Full 25 features:",
    f"{totals['complete25'] / total * 100:.2f}%"
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


output_file = (
    output_dir
    / "cycle24_feature_contract_audit.csv"
)


summary_df.to_csv(
    output_file,
    index=False
)


print("\nSaved:")
print(output_file)

print("\nFeature contract audit finished.")