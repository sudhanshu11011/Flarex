from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

VALIDATION_FILE = Path(
    "artifacts/splitting/data/cycle24_validation.parquet"
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

SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/logistic_standard_scaler.joblib"
)

SCALER_FIT_SUMMARY_FILE = Path(
    "artifacts/preprocessing/manifests/"
    "model_ch2_logistic_scaler_fit_summary.csv"
)


OUTPUT_DIR = Path(
    "artifacts/preprocessing/manifests"
)

AUDIT_FILE = (
    OUTPUT_DIR
    /
    "model_ch2_validation_transform_audit.csv"
)

FEATURE_STATS_FILE = (
    OUTPUT_DIR
    /
    "model_ch2_validation_feature_stats.csv"
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
# VERIFY INPUTS
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 2.5 — VALIDATE PREPROCESSING"
)

print(
    "===================================="
)


required_files = [
    VALIDATION_FILE,
    SPLIT_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_DESIGN_FILE,
    SCALER_FILE,
    SCALER_FIT_SUMMARY_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

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
        "Cycle24 split contract is not FROZEN."
    )


if feature_contract.get(
    "status"
) != "VERIFIED":

    raise RuntimeError(
        "Feature contract is not VERIFIED."
    )


if preprocessing_design.get(
    "status"
) != "DESIGN_FROZEN":

    raise RuntimeError(
        "Preprocessing design is not frozen."
    )


lr_design = preprocessing_design.get(
    "logistic_regression",
    {}
)


if lr_design.get(
    "validation_policy"
) != "TRANSFORM_ONLY":

    raise RuntimeError(
        "Frozen design does not permit "
        "VALIDATION transform-only policy."
    )


# ============================================================
# VALIDATION HASH CHECK
# ============================================================

expected_validation_hash = (
    split_contract[
        "validation"
    ][
        "sha256"
    ]
)


actual_validation_hash = sha256_file(
    VALIDATION_FILE
)


if (
    actual_validation_hash
    !=
    expected_validation_hash
):

    raise RuntimeError(
        "VALIDATION parquet differs from "
        "frozen split contract."
    )


# ============================================================
# SCALER HASH CHECK
# ============================================================

fit_summary = pd.read_csv(
    SCALER_FIT_SUMMARY_FILE
)


if len(
    fit_summary
) != 1:

    raise RuntimeError(
        "Unexpected scaler fit summary."
    )


fit_row = fit_summary.iloc[0]


expected_scaler_hash = str(
    fit_row[
        "scaler_sha256"
    ]
)


actual_scaler_hash = sha256_file(
    SCALER_FILE
)


if (
    actual_scaler_hash
    !=
    expected_scaler_hash
):

    raise RuntimeError(
        "Scaler artifact hash differs "
        "from Chapter 2.4 fit record."
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
    "VALIDATION hash: PASS"
)

print(
    "Scaler hash: PASS"
)


# ============================================================
# LOAD SAVED SCALER
# ============================================================

bundle = joblib.load(
    SCALER_FILE
)


if "scaler" not in bundle:

    raise RuntimeError(
        "Scaler bundle missing scaler object."
    )


if "feature_order" not in bundle:

    raise RuntimeError(
        "Scaler bundle missing feature order."
    )


scaler = bundle[
    "scaler"
]


scaler_features = bundle[
    "feature_order"
]


features = feature_contract.get(
    "feature_order",
    []
)


if len(
    features
) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


if scaler_features != features:

    raise RuntimeError(
        "Saved scaler feature order does not "
        "match feature contract."
    )


if bundle.get(
    "fit_scope"
) != "CYCLE24_TRAIN_ONLY":

    raise RuntimeError(
        "Scaler was not recorded as TRAIN-only fit."
    )


if int(
    bundle.get(
        "train_rows"
    )
) != 62064:

    raise RuntimeError(
        "Unexpected scaler training-row count."
    )


print(
    "Scaler fit scope:",
    bundle[
        "fit_scope"
    ]
)

print(
    "Scaler TRAIN rows:",
    bundle[
        "train_rows"
    ]
)


# ============================================================
# LOAD VALIDATION ONLY
# ============================================================

validation = pd.read_parquet(
    VALIDATION_FILE
)


X_validation = (
    validation[
        features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
)


if X_validation.shape != (
    13299,
    25,
):

    raise RuntimeError(
        "Unexpected VALIDATION feature shape."
    )


if X_validation.isna().any().any():

    raise RuntimeError(
        "VALIDATION contains missing/non-numeric values."
    )


X_raw = X_validation.to_numpy(
    dtype=np.float64
)


if not np.isfinite(
    X_raw
).all():

    raise RuntimeError(
        "VALIDATION contains non-finite raw values."
    )


print(
    "VALIDATION rows:",
    len(
        X_validation
    )
)

print(
    "Feature count:",
    X_validation.shape[1]
)

print(
    "TRAIN read:",
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
# TRANSFORM ONLY
# ============================================================

#
# IMPORTANT:
# No scaler.fit() or fit_transform() occurs here.
#

X_scaled = scaler.transform(
    X_raw
)


if X_scaled.shape != (
    len(
        X_validation
    ),
    25,
):

    raise RuntimeError(
        "Unexpected transformed VALIDATION shape."
    )


finite_transformed = bool(
    np.isfinite(
        X_scaled
    ).all()
)


if not finite_transformed:

    raise RuntimeError(
        "Transformed VALIDATION contains "
        "non-finite values."
    )


# ============================================================
# FEATURE-LEVEL DIAGNOSTICS
# ============================================================

feature_rows = []


for index, feature in enumerate(
    features
):

    raw_values = X_raw[
        :,
        index
    ]


    scaled_values = X_scaled[
        :,
        index
    ]


    feature_rows.append(
        {
            "feature_index":
                index,

            "feature":
                feature,

            "raw_mean":
                float(
                    raw_values.mean()
                ),

            "raw_std":
                float(
                    raw_values.std(
                        ddof=0
                    )
                ),

            "scaled_mean":
                float(
                    scaled_values.mean()
                ),

            "scaled_std":
                float(
                    scaled_values.std(
                        ddof=0
                    )
                ),

            "scaled_min":
                float(
                    scaled_values.min()
                ),

            "scaled_max":
                float(
                    scaled_values.max()
                ),

            "scaled_abs_max":
                float(
                    np.max(
                        np.abs(
                            scaled_values
                        )
                    )
                ),

            "finite":
                bool(
                    np.isfinite(
                        scaled_values
                    ).all()
                ),
        }
    )


feature_stats = pd.DataFrame(
    feature_rows
)


# ============================================================
# DIAGNOSTIC SUMMARY
# ============================================================

max_abs_scaled_value = float(
    np.max(
        np.abs(
            X_scaled
        )
    )
)


all_features_finite = bool(
    feature_stats[
        "finite"
    ].all()
)


feature_order_pass = (
    list(
        X_validation.columns
    )
    ==
    features
    ==
    scaler_features
)


transform_pass = (
    finite_transformed
    and
    all_features_finite
    and
    feature_order_pass
)


audit = pd.DataFrame(
    [
        {
            "dataset":
                "CYCLE24_VALIDATION",

            "rows":
                len(
                    X_validation
                ),

            "feature_count":
                25,

            "operation":
                "TRANSFORM_ONLY",

            "scaler_fit_scope":
                bundle[
                    "fit_scope"
                ],

            "feature_order_pass":
                feature_order_pass,

            "raw_values_finite":
                bool(
                    np.isfinite(
                        X_raw
                    ).all()
                ),

            "transformed_values_finite":
                finite_transformed,

            "max_abs_scaled_value":
                max_abs_scaled_value,

            "train_used":
                False,

            "validation_used":
                True,

            "holdout_used":
                False,

            "cycle25_used":
                False,

            "scaler_refit":
                False,

            "validation_transform_pass":
                transform_pass,

            "validation_sha256":
                actual_validation_hash,

            "scaler_sha256":
                actual_scaler_hash,
        }
    ]
)


# ============================================================
# SAVE AUDIT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


feature_stats.to_csv(
    FEATURE_STATS_FILE,
    index=False,
)


audit.to_csv(
    AUDIT_FILE,
    index=False,
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "VALIDATION TRANSFORMATION"
)

print(
    "===================================="
)


print(
    "Operation: transform() only"
)

print(
    "Scaler refit:",
    False
)

print(
    "Input shape:",
    X_raw.shape
)

print(
    "Output shape:",
    X_scaled.shape
)

print(
    "Finite transformed values:",
    finite_transformed
)

print(
    "Feature order:",
    (
        "PASS"
        if feature_order_pass
        else
        "FAIL"
    )
)

print(
    "Maximum absolute scaled value:",
    max_abs_scaled_value
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
    "TRAIN read:",
    False
)

print(
    "VALIDATION read:",
    True
)

print(
    "HOLDOUT read:",
    False
)

print(
    "Cycle25 read:",
    False
)

print(
    "Scaler refit:",
    False
)


print(
    "\n===================================="
)

print(
    "CHAPTER 2.5 RESULT"
)

print(
    "===================================="
)


print(
    "Validation transform:",
    (
        "PASS"
        if transform_pass
        else
        "FAIL"
    )
)


print(
    "\nSaved:"
)

print(
    FEATURE_STATS_FILE
)

print(
    AUDIT_FILE
)


if not transform_pass:

    raise RuntimeError(
        "VALIDATION preprocessing audit failed."
    )


print(
    "\nMODEL CHAPTER 2.5 COMPLETE."
)