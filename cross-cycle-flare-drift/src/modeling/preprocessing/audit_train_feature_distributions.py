from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    "artifacts/splitting/data/cycle24_train.parquet"
)

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/cycle24_split_contract.json"
)

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/model_feature_contract.json"
)

PREPROCESSING_DESIGN_FILE = Path(
    "artifacts/preprocessing/contracts/preprocessing_design.json"
)


OUTPUT_DIR = Path(
    "artifacts/preprocessing/manifests"
)

FEATURE_OUTPUT = (
    OUTPUT_DIR
    /
    "model_ch2_train_feature_distribution.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    /
    "model_ch2_train_distribution_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# VERIFY INPUT STATE
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 2.3 — TRAIN FEATURE DISTRIBUTION AUDIT"
)

print(
    "===================================="
)


required_files = [
    TRAIN_FILE,
    SPLIT_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_DESIGN_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


with SPLIT_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    split_contract = json.load(file)


with FEATURE_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    feature_contract = json.load(file)


with PREPROCESSING_DESIGN_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    preprocessing_design = json.load(file)


if split_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 split contract is not frozen."
    )


if feature_contract.get(
    "status"
) != "VERIFIED":

    raise RuntimeError(
        "Feature contract is not verified."
    )


if preprocessing_design.get(
    "status"
) != "DESIGN_FROZEN":

    raise RuntimeError(
        "Preprocessing design is not frozen."
    )


# ============================================================
# VERIFY TRAIN HASH
# ============================================================

expected_train_hash = (
    split_contract[
        "train"
    ][
        "sha256"
    ]
)


actual_train_hash = sha256_file(
    TRAIN_FILE
)


if (
    actual_train_hash
    !=
    expected_train_hash
):

    raise RuntimeError(
        "TRAIN parquet hash differs from "
        "frozen split contract."
    )


print(
    "Cycle24 split: FROZEN"
)

print(
    "Feature contract: VERIFIED"
)

print(
    "Preprocessing design: FROZEN"
)

print(
    "TRAIN hash verification: PASS"
)


# ============================================================
# LOAD TRAIN ONLY
# ============================================================

train = pd.read_parquet(
    TRAIN_FILE
)


features = feature_contract.get(
    "feature_order",
    []
)


if len(features) != 25:

    raise RuntimeError(
        "Expected exactly 25 model features."
    )


X = (
    train[
        features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
)


array = X.to_numpy(
    dtype=float
)


if X.isna().any().any():

    raise RuntimeError(
        "Missing/non-numeric TRAIN feature values."
    )


if not np.isfinite(
    array
).all():

    raise RuntimeError(
        "Non-finite TRAIN feature values."
    )


print(
    "TRAIN rows:",
    len(train)
)

print(
    "Features:",
    len(features)
)

print(
    "Validation read:",
    False
)

print(
    "Holdout read:",
    False
)

print(
    "Cycle25 read:",
    False
)


# ============================================================
# DISTRIBUTION AUDIT
# ============================================================

feature_rows = []


for feature in features:

    series = X[
        feature
    ].astype(float)


    q01 = float(
        series.quantile(0.01)
    )

    q05 = float(
        series.quantile(0.05)
    )

    q25 = float(
        series.quantile(0.25)
    )

    median = float(
        series.quantile(0.50)
    )

    q75 = float(
        series.quantile(0.75)
    )

    q95 = float(
        series.quantile(0.95)
    )

    q99 = float(
        series.quantile(0.99)
    )


    iqr = (
        q75
        -
        q25
    )


    lower_fence = (
        q25
        -
        1.5
        *
        iqr
    )


    upper_fence = (
        q75
        +
        1.5
        *
        iqr
    )


    iqr_outliers = int(
        (
            (
                series
                <
                lower_fence
            )
            |
            (
                series
                >
                upper_fence
            )
        )
        .sum()
    )


    mean = float(
        series.mean()
    )


    std = float(
        series.std(
            ddof=0
        )
    )


    minimum = float(
        series.min()
    )


    maximum = float(
        series.max()
    )


    max_abs = float(
        np.max(
            np.abs(
                series.to_numpy()
            )
        )
    )


    skewness = float(
        series.skew()
    )


    zero_count = int(
        (
            series
            ==
            0
        )
        .sum()
    )


    negative_count = int(
        (
            series
            <
            0
        )
        .sum()
    )


    positive_count = int(
        (
            series
            >
            0
        )
        .sum()
    )


    unique_values = int(
        series.nunique()
    )


    near_constant = (
        std
        <=
        1e-12
    )


    high_skew = (
        abs(
            skewness
        )
        >=
        2.0
    )


    high_iqr_outlier_fraction = (
        iqr_outliers
        /
        len(
            series
        )
        >=
        0.10
    )


    feature_rows.append(
        {
            "feature":
                feature,

            "count":
                len(
                    series
                ),

            "unique_values":
                unique_values,

            "min":
                minimum,

            "q01":
                q01,

            "q05":
                q05,

            "q25":
                q25,

            "median":
                median,

            "q75":
                q75,

            "q95":
                q95,

            "q99":
                q99,

            "max":
                maximum,

            "mean":
                mean,

            "std":
                std,

            "iqr":
                iqr,

            "skewness":
                skewness,

            "max_abs":
                max_abs,

            "zero_count":
                zero_count,

            "zero_pct":
                (
                    zero_count
                    /
                    len(
                        series
                    )
                    *
                    100
                ),

            "negative_count":
                negative_count,

            "negative_pct":
                (
                    negative_count
                    /
                    len(
                        series
                    )
                    *
                    100
                ),

            "positive_count":
                positive_count,

            "positive_pct":
                (
                    positive_count
                    /
                    len(
                        series
                    )
                    *
                    100
                ),

            "iqr_outlier_count":
                iqr_outliers,

            "iqr_outlier_pct":
                (
                    iqr_outliers
                    /
                    len(
                        series
                    )
                    *
                    100
                ),

            "near_constant":
                near_constant,

            "high_skew_abs_ge_2":
                high_skew,

            "iqr_outlier_pct_ge_10":
                high_iqr_outlier_fraction,
        }
    )


feature_df = pd.DataFrame(
    feature_rows
)


# ============================================================
# NUMERICAL SCALE AUDIT
# ============================================================

positive_std = feature_df[
    feature_df[
        "std"
    ]
    >
    0
][
    "std"
]


if len(
    positive_std
) != 25:

    raise RuntimeError(
        "One or more features have zero variance."
    )


largest_std = float(
    positive_std.max()
)

smallest_std = float(
    positive_std.min()
)


std_scale_ratio = (
    largest_std
    /
    smallest_std
)


largest_max_abs = float(
    feature_df[
        "max_abs"
    ].max()
)


smallest_nonzero_max_abs = float(
    feature_df.loc[
        feature_df[
            "max_abs"
        ]
        >
        0,
        "max_abs",
    ].min()
)


magnitude_scale_ratio = (
    largest_max_abs
    /
    smallest_nonzero_max_abs
)


high_skew_count = int(
    feature_df[
        "high_skew_abs_ge_2"
    ].sum()
)


high_outlier_count = int(
    feature_df[
        "iqr_outlier_pct_ge_10"
    ].sum()
)


near_constant_count = int(
    feature_df[
        "near_constant"
    ].sum()
)


# ============================================================
# AUDIT DECISION
# ============================================================

# Heavy skew and IQR outliers are diagnostic only.
#
# They do NOT fail the audit because we pre-specified
# no clipping/log transformation.
#
# Actual failures are numerical/data-integrity problems.

numerical_integrity_pass = (
    near_constant_count
    ==
    0
    and
    np.isfinite(
        std_scale_ratio
    )
    and
    np.isfinite(
        magnitude_scale_ratio
    )
)


standard_scaler_supported = (
    numerical_integrity_pass
    and
    std_scale_ratio
    >
    1.0
)


summary = pd.DataFrame(
    [
        {
            "train_rows":
                len(
                    train
                ),

            "feature_count":
                len(
                    features
                ),

            "near_constant_features":
                near_constant_count,

            "high_skew_features_abs_ge_2":
                high_skew_count,

            "features_with_iqr_outlier_pct_ge_10":
                high_outlier_count,

            "smallest_feature_std":
                smallest_std,

            "largest_feature_std":
                largest_std,

            "std_scale_ratio":
                std_scale_ratio,

            "smallest_nonzero_max_abs":
                smallest_nonzero_max_abs,

            "largest_max_abs":
                largest_max_abs,

            "magnitude_scale_ratio":
                magnitude_scale_ratio,

            "numerical_integrity_pass":
                numerical_integrity_pass,

            "standard_scaler_supported_for_lr":
                standard_scaler_supported,

            "imputation_required":
                False,

            "clipping_applied":
                False,

            "log_transform_applied":
                False,

            "validation_used":
                False,

            "holdout_used":
                False,

            "cycle25_used":
                False,
        }
    ]
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


feature_df.to_csv(
    FEATURE_OUTPUT,
    index=False,
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "FEATURE DISTRIBUTION SUMMARY"
)

print(
    "===================================="
)


display_columns = [
    "feature",
    "mean",
    "std",
    "median",
    "q01",
    "q99",
    "skewness",
    "iqr_outlier_pct",
]


print(
    feature_df[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\n===================================="
)

print(
    "DIAGNOSTIC FLAGS"
)

print(
    "===================================="
)


print(
    "Near-constant features:",
    near_constant_count
)

print(
    "Highly skewed features |skew| >= 2:",
    high_skew_count
)

print(
    "Features with >=10% IQR outliers:",
    high_outlier_count
)


print(
    "\n===================================="
)

print(
    "NUMERICAL SCALE AUDIT"
)

print(
    "===================================="
)


print(
    "Smallest feature std:",
    smallest_std
)

print(
    "Largest feature std:",
    largest_std
)

print(
    "STD scale ratio:",
    std_scale_ratio
)

print(
    "Magnitude scale ratio:",
    magnitude_scale_ratio
)

print(
    "Numerical integrity:",
    (
        "PASS"
        if numerical_integrity_pass
        else
        "FAIL"
    )
)

print(
    "StandardScaler justified for LR:",
    standard_scaler_supported
)


print(
    "\n===================================="
)

print(
    "DATA ISOLATION"
)

print(
    "===================================="
)


print(
    "TRAIN used: True"
)

print(
    "VALIDATION used: False"
)

print(
    "HOLDOUT used: False"
)

print(
    "Cycle25 used: False"
)


print(
    "\nSaved:"
)

print(
    FEATURE_OUTPUT
)

print(
    SUMMARY_OUTPUT
)


if not numerical_integrity_pass:

    raise RuntimeError(
        "TRAIN feature numerical-integrity audit failed."
    )


print(
    "\nMODEL CHAPTER 2.3 COMPLETE."
)