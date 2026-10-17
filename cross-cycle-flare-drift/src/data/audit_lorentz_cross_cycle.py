import drms
import pandas as pd
from pathlib import Path


client = drms.Client()

SERIES = "cgem.lorentz"


LORENTZ_FEATURES = [
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
]


# Same systematic grid used for SHARP quality audit
MONTH_DAYS = [
    (3, 15),
    (6, 15),
    (9, 15),
    (12, 15),
]


summary_rows = []


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
            ]
            + LORENTZ_FEATURES
        )

        try:

            df = client.query(
                query,
                key=keys
            )

        except Exception as error:

            print("QUERY ERROR:", error)

            summary_rows.append(
                {
                    "cycle": cycle,
                    "year": year,
                    "date": time,
                    "total_harps": 0,
                    "rows_with_missing": 0,
                    "missing_percent": None,
                    "query_error": True,
                }
            )

            continue


        total_rows = len(df)

        print("Lorentz HARPs:", total_rows)


        if total_rows == 0:

            summary_rows.append(
                {
                    "cycle": cycle,
                    "year": year,
                    "date": time,
                    "total_harps": 0,
                    "rows_with_missing": 0,
                    "missing_percent": None,
                    "query_error": False,
                }
            )

            continue


        # -----------------------------------------
        # Duplicate HARPNUM check
        # -----------------------------------------

        duplicate_harps = int(
            df["HARPNUM"]
            .duplicated()
            .sum()
        )

        print(
            "Duplicate HARPNUMs:",
            duplicate_harps
        )


        # -----------------------------------------
        # Missing values per feature
        # -----------------------------------------

        print("\nFeature missingness:")

        for feature in LORENTZ_FEATURES:

            missing_count = int(
                df[feature]
                .isna()
                .sum()
            )

            missing_percent = (
                missing_count
                / total_rows
                * 100
            )

            print(
                f"{feature:8s} "
                f"{missing_count:3d} "
                f"({missing_percent:.2f}%)"
            )


        # -----------------------------------------
        # Rows with at least one missing feature
        # -----------------------------------------

        missing_mask = (
            df[LORENTZ_FEATURES]
            .isna()
            .any(axis=1)
        )

        rows_with_missing = int(
            missing_mask.sum()
        )

        missing_percent = (
            rows_with_missing
            / total_rows
            * 100
        )


        print(
            "\nRows with >=1 missing Lorentz feature:",
            rows_with_missing,
            f"({missing_percent:.2f}%)"
        )


        summary_rows.append(
            {
                "cycle": cycle,
                "year": year,
                "date": time,
                "total_harps": total_rows,
                "rows_with_missing": rows_with_missing,
                "missing_percent": round(
                    missing_percent,
                    2
                ),
                "duplicate_harps": duplicate_harps,
                "query_error": False,
            }
        )


# =================================================
# FULL SUMMARY
# =================================================

summary_df = pd.DataFrame(
    summary_rows
)


print("\n\n====================================")
print("FULL LORENTZ CROSS-CYCLE AUDIT")
print("====================================")

print(
    summary_df.to_string(
        index=False
    )
)


# =================================================
# CYCLE LEVEL SUMMARY
# =================================================

valid_df = summary_df[
    summary_df["query_error"] == False
].copy()


cycle_summary = (
    valid_df
    .groupby("cycle")
    .agg(
        snapshots=("date", "count"),
        total_harps=("total_harps", "sum"),
        rows_with_missing=("rows_with_missing", "sum"),
    )
    .reset_index()
)


cycle_summary["missing_percent"] = (
    cycle_summary["rows_with_missing"]
    / cycle_summary["total_harps"]
    * 100
)


print("\n\n====================================")
print("LORENTZ CYCLE-LEVEL SUMMARY")
print("====================================")

print(
    cycle_summary.to_string(
        index=False
    )
)


# =================================================
# SAVE MANIFEST
# =================================================

output_dir = Path(
    "data_manifests"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_dir
    / "lorentz_cross_cycle_audit.csv"
)

summary_df.to_csv(
    output_file,
    index=False
)


print("\nSaved:")
print(output_file)

print("\nLorentz audit finished.")