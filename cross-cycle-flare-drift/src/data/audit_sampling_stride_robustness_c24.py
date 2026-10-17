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

OUTPUT_FILE = Path(
    "data_manifests/cycle24_sampling_stride_robustness.csv"
)

SUMMARY_FILE = Path(
    "data_manifests/cycle24_sampling_stride_robustness_summary.csv"
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


NATIVE_CADENCE_MINUTES = 12

STRIDES_MINUTES = [
    12,
    60,
    120,
    180,
    360,
]

WINDOW_HOURS = 48


# ============================================================
# HELPERS
# ============================================================

def numeric_frame(
    dataframe,
    columns
):

    return (
        dataframe[
            columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


def parse_trec(value):

    """
    Convert:

    2011.06.15_12:00:00_TAI

    to a pandas timestamp for time-difference calculations.

    We stay on the TAI clock here because only relative
    sampling intervals are required.
    """

    text = str(
        value
    ).strip()


    text = text.replace(
        "_TAI",
        ""
    )


    text = (
        text
        .replace(
            ".",
            "-",
            2
        )
        .replace(
            "_",
            "T",
            1
        )
    )


    return pd.to_datetime(
        text,
        format="%Y-%m-%dT%H:%M:%S",
        errors="coerce"
    )


# ============================================================
# LOAD REPRESENTATIVE WINDOWS FROM PHASE 7A
# ============================================================

if not WINDOW_FILE.exists():

    raise FileNotFoundError(
        f"Missing file: {WINDOW_FILE}"
    )


windows = pd.read_csv(
    WINDOW_FILE
)


required_columns = {
    "snapshot",
    "HARPNUM",
}


if not required_columns.issubset(
    windows.columns
):

    raise RuntimeError(
        "Window file must contain "
        "snapshot and HARPNUM."
    )


print(
    "Representative windows:",
    len(windows)
)


# ============================================================
# QUERY KEYS
# ============================================================

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


audit_rows = []


# ============================================================
# PROCESS WINDOWS
# ============================================================

for window_index, row in windows.iterrows():

    harpnum = int(
        row["HARPNUM"]
    )

    snapshot = str(
        row["snapshot"]
    )


    print(
        "\n===================================="
    )

    print(
        f"Window {window_index + 1}/{len(windows)}"
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
            "No SHARP rows."
        )

        continue


    # ========================================================
    # NUMERIC METADATA
    # ========================================================

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


    # ========================================================
    # SHARP ↔ LORENTZ MERGE
    # ========================================================

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


    # ========================================================
    # PARSE TIME
    # ========================================================

    merged[
        "T_REC_PARSED"
    ] = (
        merged[
            "T_REC"
        ]
        .apply(
            parse_trec
        )
    )


    merged = (
        merged
        .sort_values(
            "T_REC_PARSED"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # FEATURE FINITE CHECK
    # ========================================================

    numeric_features = numeric_frame(
        merged,
        ALL_FEATURES
    )


    merged[
        "all25_finite"
    ] = (
        np.isfinite(
            numeric_features
        )
        .all(axis=1)
    )


    # ========================================================
    # LOCKED QC-1 → QC-6
    # ========================================================

    merged[
        "qc_pass"
    ] = (

        # exact SHARP-Lorentz merge
        (
            merged[
                "_merge"
            ]
            ==
            "both"
        )

        &

        # QUALITY
        (
            merged[
                "QUALITY"
            ]
            ==
            0
        )

        &

        # valid CMASK
        (
            merged[
                "CMASK"
            ]
            >
            0
        )

        &

        # position available
        merged[
            "LON_FWT"
        ].notna()

        &

        merged[
            "LAT_FWT"
        ].notna()

        &

        # disk-position cut
        (
            merged[
                "LON_FWT"
            ]
            .abs()
            <=
            68
        )

        &

        # all model inputs valid
        merged[
            "all25_finite"
        ]
    )


    print(
        "Native rows:",
        len(merged)
    )


    print(
        "Native QC rows:",
        int(
            merged[
                "qc_pass"
            ].sum()
        )
    )


    # ========================================================
    # TEST EVERY STRIDE AND EVERY POSSIBLE PHASE
    # ========================================================

    for stride_minutes in STRIDES_MINUTES:

        if (
            stride_minutes
            %
            NATIVE_CADENCE_MINUTES
            !=
            0
        ):

            raise RuntimeError(
                "Stride must be a multiple "
                "of 12 minutes."
            )


        step = (
            stride_minutes
            //
            NATIVE_CADENCE_MINUTES
        )


        # Example:
        #
        # 120 min / 12 min = 10 phases
        #
        # phase 0 -> row 0,10,20...
        # phase 1 -> row 1,11,21...
        # ...
        # phase 9 -> row 9,19,29...
        #
        for phase in range(
            step
        ):

            sampled = (
                merged.iloc[
                    phase::step
                ]
                .copy()
            )


            sampled_rows = len(
                sampled
            )


            sampled_qc_rows = int(
                sampled[
                    "qc_pass"
                ].sum()
            )


            if sampled_rows > 0:

                retention_pct = (
                    sampled_qc_rows
                    /
                    sampled_rows
                    *
                    100
                )

            else:

                retention_pct = np.nan


            # =================================================
            # QC-VALID SAMPLES ONLY FOR AUTOCORRELATION
            # =================================================

            valid = (
                sampled[
                    sampled[
                        "qc_pass"
                    ]
                ]
                .copy()
            )


            exact_delta = pd.Timedelta(
                minutes=stride_minutes
            )


            current = valid[
                [
                    "T_REC_PARSED",
                ]
                +
                ALL_FEATURES
            ].copy()


            previous = current.copy()


            # Shift previous observation timestamp forward
            # so merging finds t vs t-stride pairs.
            previous[
                "T_REC_PARSED"
            ] = (
                previous[
                    "T_REC_PARSED"
                ]
                +
                exact_delta
            )


            pairs = current.merge(
                previous,
                on="T_REC_PARSED",
                suffixes=(
                    "_current",
                    "_previous",
                ),
                how="inner",
            )


            correlations = []


            if len(pairs) >= 3:

                for feature in ALL_FEATURES:

                    x = pd.to_numeric(
                        pairs[
                            f"{feature}_current"
                        ],
                        errors="coerce"
                    )


                    y = pd.to_numeric(
                        pairs[
                            f"{feature}_previous"
                        ],
                        errors="coerce"
                    )


                    valid_mask = (
                        np.isfinite(x)
                        &
                        np.isfinite(y)
                    )


                    x = x[
                        valid_mask
                    ]

                    y = y[
                        valid_mask
                    ]


                    if len(x) < 3:
                        continue


                    if (
                        x.nunique() < 2
                        or
                        y.nunique() < 2
                    ):

                        continue


                    corr = x.corr(
                        y
                    )


                    if pd.notna(
                        corr
                    ):

                        correlations.append(
                            float(
                                corr
                            )
                        )


            if correlations:

                median_corr = float(
                    np.median(
                        correlations
                    )
                )


                mean_corr = float(
                    np.mean(
                        correlations
                    )
                )

            else:

                median_corr = np.nan
                mean_corr = np.nan


            audit_rows.append(
                {
                    "snapshot":
                        snapshot,

                    "HARPNUM":
                        harpnum,

                    "stride_minutes":
                        stride_minutes,

                    "phase":
                        phase,

                    "phase_count":
                        step,

                    "sampled_rows":
                        sampled_rows,

                    "sampled_qc_rows":
                        sampled_qc_rows,

                    "retention_pct":
                        retention_pct,

                    "exact_lag_pairs":
                        len(pairs),

                    "features_with_exact_lag_corr":
                        len(
                            correlations
                        ),

                    "median_exact_lag_autocorr":
                        median_corr,

                    "mean_exact_lag_autocorr":
                        mean_corr,
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
        "No stride robustness rows generated."
    )


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


audit.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY BY STRIDE
# ============================================================

summary_rows = []


for stride in STRIDES_MINUTES:

    subset = audit[
        audit[
            "stride_minutes"
        ]
        ==
        stride
    ].copy()


    retention = (
        subset[
            "retention_pct"
        ]
        .dropna()
    )


    correlations = (
        subset[
            "median_exact_lag_autocorr"
        ]
        .dropna()
    )


    total_sampled = int(
        subset[
            "sampled_rows"
        ]
        .sum()
    )


    total_qc = int(
        subset[
            "sampled_qc_rows"
        ]
        .sum()
    )


    if total_sampled > 0:

        pooled_retention = (
            total_qc
            /
            total_sampled
            *
            100
        )

    else:

        pooled_retention = np.nan


    summary_rows.append(
        {
            "stride_minutes":
                stride,

            "phase_window_evaluations":
                len(subset),

            "windows":
                subset[
                    [
                        "snapshot",
                        "HARPNUM",
                    ]
                ]
                .drop_duplicates()
                .shape[0],

            "native_density_pct":
                (
                    NATIVE_CADENCE_MINUTES
                    /
                    stride
                    *
                    100
                ),

            "pooled_qc_retention_pct":
                pooled_retention,

            "median_phase_retention_pct":
                (
                    float(
                        retention.median()
                    )
                    if len(retention)
                    else np.nan
                ),

            "retention_p10":
                (
                    float(
                        retention.quantile(
                            0.10
                        )
                    )
                    if len(retention)
                    else np.nan
                ),

            "retention_p90":
                (
                    float(
                        retention.quantile(
                            0.90
                        )
                    )
                    if len(retention)
                    else np.nan
                ),

            "median_exact_lag_autocorr":
                (
                    float(
                        correlations.median()
                    )
                    if len(correlations)
                    else np.nan
                ),

            "autocorr_p10":
                (
                    float(
                        correlations.quantile(
                            0.10
                        )
                    )
                    if len(correlations)
                    else np.nan
                ),

            "autocorr_p90":
                (
                    float(
                        correlations.quantile(
                            0.90
                        )
                    )
                    if len(correlations)
                    else np.nan
                ),

            "total_exact_lag_pairs":
                int(
                    subset[
                        "exact_lag_pairs"
                    ]
                    .sum()
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n\n===================================="
)

print(
    "PHASE 7B — SAMPLING STRIDE ROBUSTNESS"
)

print(
    "===================================="
)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# PHASE SENSITIVITY
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "PHASE SENSITIVITY"
)

print(
    "------------------------------------"
)


for _, row in summary.iterrows():

    print(
        f"\nStride: "
        f"{int(row['stride_minutes'])} min"
    )


    print(
        "Retention P10-P90:",
        f"{row['retention_p10']:.2f}%"
        " -> "
        f"{row['retention_p90']:.2f}%"
    )


    print(
        "Exact-lag autocorr P10-P90:",
        f"{row['autocorr_p10']:.3f}"
        " -> "
        f"{row['autocorr_p90']:.3f}"
    )


# ============================================================
# MAIN CANDIDATE COMPARISON
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "CURRENT STRIDE CANDIDATES"
)

print(
    "------------------------------------"
)


comparison = summary[
    summary[
        "stride_minutes"
    ]
    .isin(
        [
            60,
            120,
            180,
            360,
        ]
    )
][
    [
        "stride_minutes",
        "native_density_pct",
        "pooled_qc_retention_pct",
        "median_exact_lag_autocorr",
        "autocorr_p10",
        "autocorr_p90",
    ]
]


print(
    comparison.to_string(
        index=False
    )
)


print(
    "\nIMPORTANT:"
)

print(
    "Stride selection remains label-blind."
)

print(
    "No model performance was used."
)

print(
    "No Cycle25 outcome information was used."
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)


print(
    "\nPhase 7B robustness audit finished."
)