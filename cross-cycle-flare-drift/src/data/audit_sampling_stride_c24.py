from pathlib import Path

import drms
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

client = drms.Client()

SHARP_SERIES = "hmi.sharp_cea_720s"
LORENTZ_SERIES = "cgem.lorentz"


WINDOW_FILE = Path(
    "data_manifests/cycle24_sampling_stride_windows.csv"
)

AUDIT_FILE = Path(
    "data_manifests/cycle24_sampling_stride_audit.csv"
)

SUMMARY_FILE = Path(
    "data_manifests/cycle24_sampling_stride_summary.csv"
)


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


# ============================================================
# CANDIDATE STRIDES
# ============================================================

STRIDES_MINUTES = [
    12,
    60,
    120,
    180,
    360,
]


# ============================================================
# REPRESENTATIVE CYCLE-24 SNAPSHOTS
#
# Label-blind.
# We simply inspect availability/QC and then choose up to
# 3 QC-valid HARPs per snapshot in deterministic HARPNUM order.
# ============================================================

SNAPSHOTS = [
    "2010.09.15_12:00:00_TAI",
    "2011.06.15_12:00:00_TAI",
    "2012.06.15_12:00:00_TAI",
    "2013.06.15_12:00:00_TAI",
    "2014.06.15_12:00:00_TAI",
    "2015.06.15_12:00:00_TAI",
    "2016.03.15_12:00:00_TAI",
    "2017.06.15_12:00:00_TAI",
    "2018.06.15_12:00:00_TAI",
]


MAX_HARPS_PER_SNAPSHOT = 3


# 48h provides enough native samples for lag/redundancy audit.
WINDOW_HOURS = 48


# ============================================================
# HELPER
# ============================================================

def numeric_frame(df, columns):

    return (
        df[
            columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


# ============================================================
# FIND REPRESENTATIVE HARPs
# ============================================================

selected_windows = []


print(
    "===================================="
)

print(
    "SELECTING CYCLE-24 AUDIT WINDOWS"
)

print(
    "===================================="
)


snapshot_keys = ",".join(
    [
        "T_REC",
        "HARPNUM",
        "QUALITY",
        "CMASK",
        "LON_FWT",
        "LAT_FWT",
    ]
    + SHARP_FEATURES
)


for snapshot in SNAPSHOTS:

    print(
        "\nSnapshot:",
        snapshot
    )


    query = (
        f"{SHARP_SERIES}"
        f"[]"
        f"[{snapshot}]"
    )


    try:

        sharp = client.query(
            query,
            key=snapshot_keys
        )

    except Exception as error:

        print(
            "QUERY ERROR:",
            error
        )

        continue


    if len(sharp) == 0:

        print(
            "No SHARP rows."
        )

        continue


    for column in [
        "HARPNUM",
        "QUALITY",
        "CMASK",
        "LON_FWT",
        "LAT_FWT",
    ]:

        sharp[column] = pd.to_numeric(
            sharp[column],
            errors="coerce"
        )


    feature_values = numeric_frame(
        sharp,
        SHARP_FEATURES
    )


    sharp[
        "sharp18_finite"
    ] = (
        np.isfinite(
            feature_values
        )
        .all(axis=1)
    )


    # Preliminary selection only.
    #
    # Lorentz availability is checked in actual time-series
    # query below.
    eligible = sharp[
        sharp["HARPNUM"].notna()
        &
        (sharp["QUALITY"] == 0)
        &
        (sharp["CMASK"] > 0)
        &
        sharp["LON_FWT"].notna()
        &
        sharp["LAT_FWT"].notna()
        &
        (sharp["LON_FWT"].abs() <= 68)
        &
        sharp["sharp18_finite"]
    ].copy()


    eligible = (
        eligible
        .sort_values(
            "HARPNUM"
        )
        .head(
            MAX_HARPS_PER_SNAPSHOT
        )
    )


    print(
        "Eligible HARPs selected:",
        len(eligible)
    )


    for _, row in eligible.iterrows():

        selected_windows.append(
            {
                "snapshot":
                    snapshot,

                "HARPNUM":
                    int(
                        row["HARPNUM"]
                    ),
            }
        )


windows_df = pd.DataFrame(
    selected_windows
)


if len(windows_df) == 0:

    raise RuntimeError(
        "No representative HARP windows selected."
    )


windows_df = (
    windows_df
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)


windows_df.to_csv(
    WINDOW_FILE,
    index=False
)


print(
    "\nRepresentative windows:",
    len(windows_df)
)


# ============================================================
# QUERY 48-HOUR NATIVE-CADENCE SERIES
# ============================================================

audit_rows = []


sharp_keys = ",".join(
    [
        "T_REC",
        "HARPNUM",
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


for window_index, selection in windows_df.iterrows():

    harpnum = int(
        selection["HARPNUM"]
    )

    snapshot = (
        selection["snapshot"]
    )


    print(
        "\n===================================="
    )

    print(
        f"Window {window_index + 1}/{len(windows_df)}"
    )

    print(
        "HARPNUM:",
        harpnum
    )

    print(
        "Start:",
        snapshot
    )

    print(
        "===================================="
    )


    # Query 48h from representative timestamp
    # using native 12-minute cadence.
    time_selector = (
        f"{snapshot}"
        f"/{WINDOW_HOURS}h"
        f"@12m"
    )


    sharp_query = (
        f"{SHARP_SERIES}"
        f"[{harpnum}]"
        f"[{time_selector}]"
    )


    lorentz_query = (
        f"{LORENTZ_SERIES}"
        f"[{harpnum}]"
        f"[{time_selector}]"
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

        print(
            "QUERY ERROR:",
            error
        )

        continue


    if len(sharp) == 0:

        print(
            "No SHARP time-series rows."
        )

        continue


    # --------------------------------------------------------
    # Normalize metadata
    # --------------------------------------------------------

    for column in [
        "QUALITY",
        "CMASK",
        "LON_FWT",
        "LAT_FWT",
    ]:

        sharp[column] = pd.to_numeric(
            sharp[column],
            errors="coerce"
        )


    # --------------------------------------------------------
    # Merge Lorentz
    # --------------------------------------------------------

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

            merged[
                feature
            ] = np.nan


        merged[
            "_merge"
        ] = "left_only"


    feature_values = numeric_frame(
        merged,
        ALL_FEATURES
    )


    merged[
        "all25_finite"
    ] = (
        np.isfinite(
            feature_values
        )
        .all(axis=1)
    )


    # --------------------------------------------------------
    # Locked QC-1 → QC-6
    # --------------------------------------------------------

    merged[
        "qc_pass"
    ] = (

        (
            merged[
                "_merge"
            ]
            ==
            "both"
        )

        &

        (
            merged[
                "QUALITY"
            ]
            ==
            0
        )

        &

        (
            merged[
                "CMASK"
            ]
            >
            0
        )

        &

        merged[
            "LON_FWT"
        ].notna()

        &

        merged[
            "LAT_FWT"
        ].notna()

        &

        (
            merged[
                "LON_FWT"
            ]
            .abs()
            <=
            68
        )

        &

        merged[
            "all25_finite"
        ]
    )


    merged = (
        merged
        .reset_index(
            drop=True
        )
    )


    native_total = len(
        merged
    )


    native_qc = int(
        merged[
            "qc_pass"
        ].sum()
    )


    print(
        "Native rows:",
        native_total
    )

    print(
        "Native QC-pass:",
        native_qc
    )


    # ========================================================
    # STRIDE AUDIT
    #
    # Native cadence is exactly 12 min.
    #
    # Candidate stride N corresponds to taking every:
    #
    # N / 12 native samples.
    # ========================================================

    for stride_minutes in STRIDES_MINUTES:

        step = (
            stride_minutes
            //
            12
        )


        if (
            stride_minutes
            %
            12
            !=
            0
        ):

            raise RuntimeError(
                "Candidate stride must be "
                "multiple of 12 minutes."
            )


        sampled = merged.iloc[
            ::step
        ].copy()


        sampled_total = len(
            sampled
        )


        sampled_qc = int(
            sampled[
                "qc_pass"
            ].sum()
        )


        if sampled_total > 0:

            retention_pct = (
                sampled_qc
                /
                sampled_total
                *
                100
            )

        else:

            retention_pct = np.nan


        # ----------------------------------------------------
        # REDUNDANCY / AUTOCORRELATION
        #
        # Compare QC-valid feature vectors separated by one
        # candidate stride.
        # ----------------------------------------------------

        valid = (
            sampled[
                sampled[
                    "qc_pass"
                ]
            ]
            .copy()
        )


        feature_correlations = []


        if len(valid) >= 3:

            numeric_valid = numeric_frame(
                valid,
                ALL_FEATURES
            )


            for feature in ALL_FEATURES:

                series = (
                    numeric_valid[
                        feature
                    ]
                    .astype(float)
                )


                if (
                    len(series) < 3
                    or
                    series.nunique(
                        dropna=True
                    ) < 2
                ):

                    continue


                correlation = (
                    series
                    .autocorr(
                        lag=1
                    )
                )


                if pd.notna(
                    correlation
                ):

                    feature_correlations.append(
                        float(
                            correlation
                        )
                    )


        if feature_correlations:

            median_autocorr = float(
                np.median(
                    feature_correlations
                )
            )

            mean_autocorr = float(
                np.mean(
                    feature_correlations
                )
            )

        else:

            median_autocorr = np.nan
            mean_autocorr = np.nan


        audit_rows.append(
            {
                "snapshot":
                    snapshot,

                "HARPNUM":
                    harpnum,

                "window_hours":
                    WINDOW_HOURS,

                "stride_minutes":
                    stride_minutes,

                "native_rows":
                    native_total,

                "native_qc_rows":
                    native_qc,

                "sampled_rows":
                    sampled_total,

                "sampled_qc_rows":
                    sampled_qc,

                "retention_pct":
                    retention_pct,

                "features_with_autocorr":
                    len(
                        feature_correlations
                    ),

                "median_feature_autocorr":
                    median_autocorr,

                "mean_feature_autocorr":
                    mean_autocorr,
            }
        )


# ============================================================
# BUILD AUDIT DATAFRAME
# ============================================================

audit = pd.DataFrame(
    audit_rows
)


if len(audit) == 0:

    raise RuntimeError(
        "No sampling-stride audit rows generated."
    )


audit.to_csv(
    AUDIT_FILE,
    index=False
)


# ============================================================
# SUMMARY BY STRIDE
# ============================================================

summary = (
    audit
    .groupby(
        "stride_minutes"
    )
    .agg(
        windows=(
            "HARPNUM",
            "size"
        ),

        total_sampled_rows=(
            "sampled_rows",
            "sum"
        ),

        total_qc_rows=(
            "sampled_qc_rows",
            "sum"
        ),

        median_retention_pct=(
            "retention_pct",
            "median"
        ),

        median_autocorr=(
            "median_feature_autocorr",
            "median"
        ),

        mean_autocorr=(
            "median_feature_autocorr",
            "mean"
        ),
    )
    .reset_index()
)


summary[
    "overall_qc_retention_pct"
] = (
    summary[
        "total_qc_rows"
    ]
    /
    summary[
        "total_sampled_rows"
    ]
    *
    100
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n\n===================================="
)

print(
    "CYCLE-24 SAMPLING STRIDE AUDIT"
)

print(
    "===================================="
)


print(
    "Representative windows:",
    len(
        windows_df
    )
)


print(
    "\nCandidate-stride summary:"
)


print(
    summary[
        [
            "stride_minutes",
            "windows",
            "total_sampled_rows",
            "total_qc_rows",
            "overall_qc_retention_pct",
            "median_autocorr",
            "mean_autocorr",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# RELATIVE SAMPLE REDUCTION
# ============================================================

native_summary = summary[
    summary[
        "stride_minutes"
    ]
    ==
    12
]


print(
    "\n------------------------------------"
)

print(
    "RELATIVE SAMPLE DENSITY"
)

print(
    "------------------------------------"
)


if len(
    native_summary
) == 1:

    native_rows = int(
        native_summary.iloc[0][
            "total_sampled_rows"
        ]
    )


    for _, row in summary.iterrows():

        stride = int(
            row[
                "stride_minutes"
            ]
        )


        rows = int(
            row[
                "total_sampled_rows"
            ]
        )


        if native_rows > 0:

            fraction = (
                rows
                /
                native_rows
                *
                100
            )

        else:

            fraction = np.nan


        print(
            f"{stride:>3} min:"
            f" {rows} rows"
            f" ({fraction:.2f}% of native)"
        )


# ============================================================
# IMPORTANT NOTE
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "INTERPRETATION RULE"
)

print(
    "------------------------------------"
)


print(
    "Do NOT choose stride from model performance."
)

print(
    "Do NOT use Cycle25 outcomes."
)

print(
    "This audit measures only Cycle24 input-series "
    "redundancy, QC retention, and sample density."
)


print(
    "\nSaved:"
)

print(
    WINDOW_FILE
)

print(
    AUDIT_FILE
)

print(
    SUMMARY_FILE
)


print(
    "\nSampling-stride audit finished."
)