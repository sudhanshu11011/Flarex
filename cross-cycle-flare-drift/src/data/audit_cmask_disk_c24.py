import drms
import pandas as pd
import numpy as np
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


# Candidate longitude cutoffs.
# We are NOT locking one yet.
CUTOFFS = [
    60,
    65,
    68,
    70,
    75,
]


# -----------------------------------------------
# Cycle-24 audit timestamps
# -----------------------------------------------

DATES = []

# Cycle 24 starts during 2010
for month in [6, 9, 12]:

    DATES.append(
        f"2010.{month:02d}.15_12:00:00_TAI"
    )


for year in range(2011, 2020):

    for month in [3, 6, 9, 12]:

        DATES.append(
            f"{year}.{month:02d}.15_12:00:00_TAI"
        )


all_rows = []


# =================================================
# QUERY + MERGE
# =================================================

for time in DATES:

    print("\n====================================")
    print(time)
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
            "LAT_FWT",
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


    print("SHARP rows:", len(sharp))
    print("Lorentz rows:", len(lorentz))


    # ---------------------------------------------
    # Convert important numeric metadata
    # ---------------------------------------------

    sharp["QUALITY"] = pd.to_numeric(
        sharp["QUALITY"],
        errors="coerce"
    )

    sharp["CMASK"] = pd.to_numeric(
        sharp["CMASK"],
        errors="coerce"
    )

    sharp["LON_FWT"] = pd.to_numeric(
        sharp["LON_FWT"],
        errors="coerce"
    )

    sharp["LAT_FWT"] = pd.to_numeric(
        sharp["LAT_FWT"],
        errors="coerce"
    )


    # ---------------------------------------------
    # LEFT JOIN so missing Lorentz records remain
    # visible
    # ---------------------------------------------

    if len(lorentz) > 0:

        merged = sharp.merge(
            lorentz,
            on=[
                "HARPNUM",
                "T_REC",
            ],
            how="left",
            validate="one_to_one",
            indicator=True,
        )

    else:

        merged = sharp.copy()

        for feature in LORENTZ_FEATURES:
            merged[feature] = np.nan

        merged["_merge"] = "left_only"


    # ---------------------------------------------
    # Quality flags
    # ---------------------------------------------

    merged["quality_zero"] = (
        merged["QUALITY"] == 0
    )


    merged["cmask_zero"] = (
        merged["CMASK"]
        .fillna(0)
        .eq(0)
    )


    merged["position_missing"] = (
        merged["LON_FWT"].isna()
        |
        merged["LAT_FWT"].isna()
    )


    merged["lorentz_record_missing"] = (
        merged["_merge"] == "left_only"
    )


    # ---------------------------------------------
    # Feature validity
    # ---------------------------------------------

    merged["sharp18_complete"] = (
        ~merged[SHARP_FEATURES]
        .isna()
        .any(axis=1)
    )


    merged["lorentz7_complete"] = (
        ~merged[LORENTZ_FEATURES]
        .isna()
        .any(axis=1)
    )


    merged["all25_complete"] = (
        merged["sharp18_complete"]
        &
        merged["lorentz7_complete"]
    )


    # ---------------------------------------------
    # Non-finite value check
    # ---------------------------------------------

    numeric_features = merged[
        ALL_FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )


    merged["all25_finite"] = (
        np.isfinite(
            numeric_features
        )
        .all(axis=1)
    )


    merged["abs_lon"] = (
        merged["LON_FWT"]
        .abs()
    )


    merged["audit_time"] = time


    all_rows.append(
        merged
    )


# =================================================
# COMBINE
# =================================================

audit = pd.concat(
    all_rows,
    ignore_index=True
)


print("\n\n====================================")
print("CYCLE 24 CMASK + DISK QC AUDIT")
print("====================================")


# =================================================
# PRIMARY QUALITY == 0 SUBSET
# =================================================

q0 = audit[
    audit["quality_zero"]
].copy()


print("\n------------------------------------")
print("QUALITY == 0 SUBSET")
print("------------------------------------")

print(
    "All records:",
    len(audit)
)

print(
    "QUALITY == 0:",
    len(q0)
)


print(
    "CMASK = 0:",
    int(
        q0["cmask_zero"].sum()
    )
)


print(
    "CMASK > 0:",
    int(
        (~q0["cmask_zero"]).sum()
    )
)


print(
    "Position missing:",
    int(
        q0["position_missing"].sum()
    )
)


print(
    "Missing Lorentz record:",
    int(
        q0[
            "lorentz_record_missing"
        ].sum()
    )
)


print(
    "18 SHARP complete:",
    int(
        q0[
            "sharp18_complete"
        ].sum()
    )
)


print(
    "7 Lorentz complete:",
    int(
        q0[
            "lorentz7_complete"
        ].sum()
    )
)


print(
    "All 25 complete:",
    int(
        q0[
            "all25_complete"
        ].sum()
    )
)


print(
    "All 25 finite:",
    int(
        q0[
            "all25_finite"
        ].sum()
    )
)


# =================================================
# CMASK INTERACTION
# =================================================

print("\n------------------------------------")
print("CMASK INTERACTION")
print("------------------------------------")


cmask_summary = (
    q0
    .groupby(
        "cmask_zero",
        dropna=False
    )
    .agg(
        rows=(
            "HARPNUM",
            "size",
        ),

        position_missing=(
            "position_missing",
            "sum",
        ),

        sharp18_complete=(
            "sharp18_complete",
            "sum",
        ),

        lorentz7_complete=(
            "lorentz7_complete",
            "sum",
        ),

        all25_complete=(
            "all25_complete",
            "sum",
        ),

        all25_finite=(
            "all25_finite",
            "sum",
        ),
    )
    .reset_index()
)


print(
    cmask_summary.to_string(
        index=False
    )
)


# =================================================
# POSITION / LONGITUDE AUDIT
# =================================================

position_valid = q0[
    ~q0["position_missing"]
].copy()


print("\n------------------------------------")
print("POSITION SUMMARY")
print("------------------------------------")


print(
    "Rows with valid position:",
    len(position_valid)
)


if len(position_valid) > 0:

    print(
        "Minimum LON_FWT:",
        round(
            position_valid[
                "LON_FWT"
            ].min(),
            2
        )
    )

    print(
        "Maximum LON_FWT:",
        round(
            position_valid[
                "LON_FWT"
            ].max(),
            2
        )
    )

    print(
        "Maximum |LON_FWT|:",
        round(
            position_valid[
                "abs_lon"
            ].max(),
            2
        )
    )


# =================================================
# CANDIDATE DISK CUTOFFS
# =================================================

print("\n------------------------------------")
print("DISK CUTOFF COMPARISON")
print("------------------------------------")


cutoff_rows = []


for cutoff in CUTOFFS:

    inside = q0[
        (~q0["position_missing"])
        &
        (q0["abs_lon"] <= cutoff)
    ].copy()


    total_inside = len(inside)


    if total_inside == 0:

        continue


    cmask_positive = (
        ~inside["cmask_zero"]
    )


    finite25 = (
        inside["all25_finite"]
    )


    pass_cmask_and_25 = (
        cmask_positive
        &
        finite25
    )


    cutoff_rows.append(
        {
            "cutoff_deg": cutoff,

            "rows_inside":
                total_inside,

            "retention_vs_quality0_pct":
                round(
                    total_inside
                    / len(q0)
                    * 100,
                    2
                ),

            "cmask_zero_inside":
                int(
                    inside[
                        "cmask_zero"
                    ].sum()
                ),

            "all25_finite_inside":
                int(
                    finite25.sum()
                ),

            "final_pass_candidate":
                int(
                    pass_cmask_and_25.sum()
                ),

            "final_pass_vs_quality0_pct":
                round(
                    pass_cmask_and_25.sum()
                    / len(q0)
                    * 100,
                    2
                ),

            "final_pass_vs_inside_pct":
                round(
                    pass_cmask_and_25.sum()
                    / total_inside
                    * 100,
                    2
                ),
        }
    )


cutoff_summary = pd.DataFrame(
    cutoff_rows
)


print(
    cutoff_summary.to_string(
        index=False
    )
)


# =================================================
# CHECK WHETHER CMASK > 0 IS SUFFICIENT
# =================================================

cmask_positive = q0[
    ~q0["cmask_zero"]
].copy()


cmask_positive_incomplete = cmask_positive[
    ~cmask_positive["all25_finite"]
]


print("\n------------------------------------")
print("CMASK > 0 BUT 25 FEATURES INVALID")
print("------------------------------------")


print(
    "CMASK > 0 rows:",
    len(cmask_positive)
)


print(
    "CMASK > 0 but all25 not finite:",
    len(cmask_positive_incomplete)
)


if len(cmask_positive_incomplete) > 0:

    print(
        cmask_positive_incomplete[
            [
                "T_REC",
                "HARPNUM",
                "CMASK",
                "LON_FWT",
                "LAT_FWT",
                "sharp18_complete",
                "lorentz7_complete",
                "all25_finite",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# =================================================
# SAVE
# =================================================

output_dir = Path(
    "data_manifests"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


row_file = (
    output_dir
    / "cycle24_cmask_disk_row_audit.csv"
)


cutoff_file = (
    output_dir
    / "cycle24_disk_cutoff_summary.csv"
)


cmask_file = (
    output_dir
    / "cycle24_cmask_summary.csv"
)


audit.to_csv(
    row_file,
    index=False
)


cutoff_summary.to_csv(
    cutoff_file,
    index=False
)


cmask_summary.to_csv(
    cmask_file,
    index=False
)


print("\nSaved:")
print(row_file)
print(cutoff_file)
print(cmask_file)


print("\nCMASK + disk-position audit finished.")