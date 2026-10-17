from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RAW_ROOT = Path(
    "data/raw/jsoc"
)

INTERIM_DIR = Path(
    "data/interim"
)

SUMMARY_FILE = Path(
    "data_manifests/phase10_merge_qc_summary.csv"
)


INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

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
    +
    LORENTZ_FEATURES
)


# ============================================================
# LOAD MONTHLY FILES
# ============================================================

def load_parquet_directory(
    directory: Path,
    label: str
):

    files = sorted(
        directory.glob(
            "*.parquet"
        )
    )


    if not files:

        raise FileNotFoundError(
            f"No parquet files found in "
            f"{directory}"
        )


    print(
        f"{label} files:",
        len(files)
    )


    frames = []


    for index, file in enumerate(
        files,
        start=1
    ):

        df = pd.read_parquet(
            file
        )


        frames.append(
            df
        )


        if (
            index % 20 == 0
            or
            index == len(files)
        ):

            print(
                f"{label}: loaded "
                f"{index}/{len(files)} files"
            )


    combined = pd.concat(
        frames,
        ignore_index=True
    )


    return combined


# ============================================================
# STANDARDIZE KEYS
# ============================================================

def standardize_keys(
    df,
    label
):

    if "HARPNUM" not in df.columns:

        raise RuntimeError(
            f"{label}: HARPNUM missing"
        )


    if "T_REC" not in df.columns:

        raise RuntimeError(
            f"{label}: T_REC missing"
        )


    df = df.copy()


    df[
        "HARPNUM"
    ] = pd.to_numeric(
        df[
            "HARPNUM"
        ],
        errors="coerce"
    ).astype(
        "Int64"
    )


    df[
        "T_REC"
    ] = (
        df[
            "T_REC"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )


    return df


# ============================================================
# REJECTION REASON
# ============================================================

def build_rejection_reason(
    row
):

    reasons = []


    if not row[
        "qc1_valid_key"
    ]:

        reasons.append(
            "QC1_INVALID_KEY"
        )


    if not row[
        "qc2_exact_merge"
    ]:

        if row[
            "_merge"
        ] == "left_only":

            reasons.append(
                "QC2_MISSING_LORENTZ"
            )

        elif row[
            "_merge"
        ] == "right_only":

            reasons.append(
                "QC2_MISSING_SHARP"
            )

        else:

            reasons.append(
                "QC2_MERGE_FAIL"
            )


    if not row[
        "qc3_quality0"
    ]:

        reasons.append(
            "QC3_QUALITY_NONZERO"
        )


    if not row[
        "qc4_cmask_positive"
    ]:

        reasons.append(
            "QC4_CMASK_INVALID"
        )


    if not row[
        "qc5_disk_position"
    ]:

        reasons.append(
            "QC5_POSITION_OR_LIMB"
        )


    if not row[
        "qc6_all25_finite"
    ]:

        reasons.append(
            "QC6_NONFINITE_FEATURE"
        )


    if not reasons:

        return "PASS"


    return ";".join(
        reasons
    )


# ============================================================
# PROCESS ONE CYCLE
# ============================================================

def process_cycle(
    cycle_key
):

    print(
        "\n===================================="
    )

    print(
        cycle_key.upper()
    )

    print(
        "===================================="
    )


    sharp_dir = (
        RAW_ROOT
        /
        cycle_key
        /
        "sharp"
    )


    lorentz_dir = (
        RAW_ROOT
        /
        cycle_key
        /
        "lorentz"
    )


    sharp = load_parquet_directory(
        sharp_dir,
        "SHARP"
    )


    lorentz = load_parquet_directory(
        lorentz_dir,
        "LORENTZ"
    )


    sharp = standardize_keys(
        sharp,
        "SHARP"
    )


    lorentz = standardize_keys(
        lorentz,
        "LORENTZ"
    )


    # ========================================================
    # SOURCE DUPLICATES
    # ========================================================

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


    if (
        sharp_duplicates > 0
        or
        lorentz_duplicates > 0
    ):

        raise RuntimeError(
            f"{cycle_key}: duplicate source keys found"
        )


    # ========================================================
    # EXACT OUTER MERGE
    # ========================================================

    merged = sharp.merge(
        lorentz,
        on=[
            "HARPNUM",
            "T_REC",
        ],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )


    matched = int(
        (
            merged[
                "_merge"
            ]
            ==
            "both"
        )
        .sum()
    )


    sharp_only = int(
        (
            merged[
                "_merge"
            ]
            ==
            "left_only"
        )
        .sum()
    )


    lorentz_only = int(
        (
            merged[
                "_merge"
            ]
            ==
            "right_only"
        )
        .sum()
    )


    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    metadata_numeric = [
        "QUALITY",
        "CMASK",
        "LON_FWT",
        "LAT_FWT",
    ]


    for column in metadata_numeric:

        if column not in merged.columns:

            merged[
                column
            ] = np.nan


        merged[
            column
        ] = pd.to_numeric(
            merged[
                column
            ],
            errors="coerce"
        )


    for feature in ALL_FEATURES:

        if feature not in merged.columns:

            merged[
                feature
            ] = np.nan


        merged[
            feature
        ] = pd.to_numeric(
            merged[
                feature
            ],
            errors="coerce"
        )


    # ========================================================
    # QC-1
    # ========================================================

    merged[
        "qc1_valid_key"
    ] = (
        merged[
            "HARPNUM"
        ].notna()
        &
        merged[
            "T_REC"
        ].notna()
        &
        (
            merged[
                "T_REC"
            ].str.len()
            >
            0
        )
    )


    # ========================================================
    # QC-2
    # ========================================================

    merged[
        "qc2_exact_merge"
    ] = (
        merged[
            "_merge"
        ]
        ==
        "both"
    )


    # ========================================================
    # QC-3
    # ========================================================

    merged[
        "qc3_quality0"
    ] = (
        merged[
            "QUALITY"
        ]
        ==
        0
    )


    # ========================================================
    # QC-4
    # ========================================================

    merged[
        "qc4_cmask_positive"
    ] = (
        merged[
            "CMASK"
        ]
        >
        0
    )


    # ========================================================
    # QC-5
    # ========================================================

    merged[
        "qc5_disk_position"
    ] = (
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
            ].abs()
            <=
            68
        )
    )


    # ========================================================
    # QC-6
    # ========================================================

    feature_array = (
        merged[
            ALL_FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    )


    merged[
        "qc6_all25_finite"
    ] = np.isfinite(
        feature_array
    ).all(
        axis=1
    )


    # ========================================================
    # FINAL QC-1 → QC-6
    # ========================================================

    merged[
        "qc_pass_1_6"
    ] = (
        merged[
            "qc1_valid_key"
        ]
        &
        merged[
            "qc2_exact_merge"
        ]
        &
        merged[
            "qc3_quality0"
        ]
        &
        merged[
            "qc4_cmask_positive"
        ]
        &
        merged[
            "qc5_disk_position"
        ]
        &
        merged[
            "qc6_all25_finite"
        ]
    )


    # ========================================================
    # REJECTION REASONS
    # ========================================================

    print(
        "Building rejection reasons..."
    )


    merged[
        "rejection_reason"
    ] = merged.apply(
        build_rejection_reason,
        axis=1
    )


    # ========================================================
    # CYCLE TAG
    # ========================================================

    merged[
        "cycle"
    ] = (
        "Cycle24"
        if cycle_key == "cycle24"
        else "Cycle25"
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    output_file = (
        INTERIM_DIR
        /
        f"{cycle_key}_merged_qc.parquet"
    )


    merged.to_parquet(
        output_file,
        index=False
    )


    qc_valid = int(
        merged[
            "qc_pass_1_6"
        ]
        .sum()
    )


    total_rows = len(
        merged
    )


    retention_pct = (
        qc_valid
        /
        total_rows
        *
        100
        if total_rows
        else 0
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "cycle":
            cycle_key,

        "sharp_rows":
            len(
                sharp
            ),

        "lorentz_rows":
            len(
                lorentz
            ),

        "outer_merged_rows":
            total_rows,

        "exact_matches":
            matched,

        "sharp_only":
            sharp_only,

        "lorentz_only":
            lorentz_only,

        "merge_rate_from_sharp_pct":
            (
                matched
                /
                len(
                    sharp
                )
                *
                100
                if len(
                    sharp
                )
                else 0
            ),

        "qc1_valid_key":
            int(
                merged[
                    "qc1_valid_key"
                ]
                .sum()
            ),

        "qc2_exact_merge":
            int(
                merged[
                    "qc2_exact_merge"
                ]
                .sum()
            ),

        "qc3_quality0":
            int(
                merged[
                    "qc3_quality0"
                ]
                .sum()
            ),

        "qc4_cmask_positive":
            int(
                merged[
                    "qc4_cmask_positive"
                ]
                .sum()
            ),

        "qc5_disk_position":
            int(
                merged[
                    "qc5_disk_position"
                ]
                .sum()
            ),

        "qc6_all25_finite":
            int(
                merged[
                    "qc6_all25_finite"
                ]
                .sum()
            ),

        "qc_pass_1_6":
            qc_valid,

        "qc_retention_pct":
            retention_pct,

        "sharp_duplicate_keys":
            sharp_duplicates,

        "lorentz_duplicate_keys":
            lorentz_duplicates,

        "output_file":
            str(
                output_file
            ),
    }


    print(
        "\n------------------------------------"
    )

    print(
        "MERGE + QC SUMMARY"
    )

    print(
        "------------------------------------"
    )


    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )


    return summary


# ============================================================
# MAIN
# ============================================================

summary_rows = []


for cycle in [
    "cycle24",
    "cycle25",
]:

    summary_rows.append(
        process_cycle(
            cycle
        )
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


print(
    "\n===================================="
)

print(
    "PHASE 10B/10C FINAL SUMMARY"
)

print(
    "===================================="
)


print(
    summary_df[
        [
            "cycle",
            "sharp_rows",
            "lorentz_rows",
            "exact_matches",
            "sharp_only",
            "lorentz_only",
            "merge_rate_from_sharp_pct",
            "qc_pass_1_6",
            "qc_retention_pct",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    SUMMARY_FILE
)


print(
    "\nPhase 10B/10C complete."
)