from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

try:
    import sklearn
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise RuntimeError(
        "scikit-learn is required. Install with: "
        "pip install scikit-learn"
    ) from exc


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

DISTRIBUTION_SUMMARY_FILE = Path(
    "artifacts/preprocessing/manifests/model_ch2_train_distribution_summary.csv"
)


SCALER_DIR = Path(
    "artifacts/preprocessing/scalers"
)

MANIFEST_DIR = Path(
    "artifacts/preprocessing/manifests"
)


SCALER_FILE = (
    SCALER_DIR
    /
    "logistic_standard_scaler.joblib"
)

PARAMETERS_FILE = (
    MANIFEST_DIR
    /
    "model_ch2_logistic_scaler_parameters.csv"
)

FIT_SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "model_ch2_logistic_scaler_fit_summary.csv"
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
    "MODEL CHAPTER 2.4 — FIT LR STANDARD SCALER"
)

print(
    "===================================="
)


required_files = [
    TRAIN_FILE,
    SPLIT_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_DESIGN_FILE,
    DISTRIBUTION_SUMMARY_FILE,
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


if split_contract.get("status") != "FROZEN":

    raise RuntimeError(
        "Cycle24 split contract is not FROZEN."
    )


if feature_contract.get("status") != "VERIFIED":

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
    "preprocessor"
) != "StandardScaler":

    raise RuntimeError(
        "Frozen preprocessing design does not "
        "specify StandardScaler for Logistic Regression."
    )


if lr_design.get(
    "fit_scope"
) != "TRAIN_ONLY":

    raise RuntimeError(
        "Logistic scaler fit scope is not TRAIN_ONLY."
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


if actual_train_hash != expected_train_hash:

    raise RuntimeError(
        "TRAIN file hash differs from frozen split contract."
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
# VERIFY 2.3 AUDIT
# ============================================================

distribution_summary = pd.read_csv(
    DISTRIBUTION_SUMMARY_FILE
)


if len(distribution_summary) != 1:

    raise RuntimeError(
        "Unexpected TRAIN distribution summary."
    )


distribution_row = distribution_summary.iloc[0]


if not bool(
    distribution_row[
        "numerical_integrity_pass"
    ]
):

    raise RuntimeError(
        "TRAIN numerical integrity audit did not PASS."
    )


if not bool(
    distribution_row[
        "standard_scaler_supported_for_lr"
    ]
):

    raise RuntimeError(
        "Chapter 2.3 did not support StandardScaler."
    )


print(
    "Chapter 2.3 numerical audit: PASS"
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
        "Expected exactly 25 frozen features."
    )


X_train = (
    train[
        features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
)


if X_train.isna().any().any():

    raise RuntimeError(
        "TRAIN contains missing/non-numeric feature values."
    )


X_array = X_train.to_numpy(
    dtype=np.float64
)


if not np.isfinite(
    X_array
).all():

    raise RuntimeError(
        "TRAIN contains non-finite feature values."
    )


print(
    "TRAIN rows:",
    len(X_train)
)

print(
    "Feature count:",
    X_train.shape[1]
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
# FIT STANDARD SCALER — TRAIN ONLY
# ============================================================

scaler = StandardScaler(
    with_mean=True,
    with_std=True,
)


scaler.fit(
    X_array
)


if int(
    scaler.n_features_in_
) != 25:

    raise RuntimeError(
        "Scaler fitted unexpected feature count."
    )


if int(
    scaler.n_samples_seen_
) != len(
    X_train
):

    raise RuntimeError(
        "Scaler sample-count mismatch."
    )


if not np.isfinite(
    scaler.mean_
).all():

    raise RuntimeError(
        "Scaler mean contains non-finite values."
    )


if not np.isfinite(
    scaler.scale_
).all():

    raise RuntimeError(
        "Scaler scale contains non-finite values."
    )


if (
    scaler.scale_
    <=
    0
).any():

    raise RuntimeError(
        "Scaler contains non-positive scale values."
    )


# ============================================================
# TRAIN TRANSFORMATION VERIFICATION
# ============================================================

X_scaled = scaler.transform(
    X_array
)


if not np.isfinite(
    X_scaled
).all():

    raise RuntimeError(
        "Scaled TRAIN contains non-finite values."
    )


scaled_means = X_scaled.mean(
    axis=0
)


scaled_stds = X_scaled.std(
    axis=0,
    ddof=0,
)


max_abs_scaled_mean = float(
    np.max(
        np.abs(
            scaled_means
        )
    )
)


max_abs_std_error = float(
    np.max(
        np.abs(
            scaled_stds
            -
            1.0
        )
    )
)


# Numerical tolerances only.
MEAN_TOLERANCE = 1e-10
STD_TOLERANCE = 1e-10


mean_check = (
    max_abs_scaled_mean
    <=
    MEAN_TOLERANCE
)


std_check = (
    max_abs_std_error
    <=
    STD_TOLERANCE
)


if not mean_check:

    raise RuntimeError(
        "Scaled TRAIN means are not sufficiently close to zero."
    )


if not std_check:

    raise RuntimeError(
        "Scaled TRAIN stds are not sufficiently close to one."
    )


# ============================================================
# SAVE SCALER
# ============================================================

SCALER_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


scaler_bundle = {

    "scaler":
        scaler,

    "feature_order":
        features,

    "feature_count":
        25,

    "fit_scope":
        "CYCLE24_TRAIN_ONLY",

    "train_rows":
        len(
            X_train
        ),

    "train_sha256":
        actual_train_hash,

    "sklearn_version":
        sklearn.__version__,
}


joblib.dump(
    scaler_bundle,
    SCALER_FILE,
)


# Reload artifact independently.
loaded_bundle = joblib.load(
    SCALER_FILE
)


loaded_scaler = loaded_bundle[
    "scaler"
]


loaded_features = loaded_bundle[
    "feature_order"
]


if loaded_features != features:

    raise RuntimeError(
        "Saved scaler feature order mismatch."
    )


loaded_transform = loaded_scaler.transform(
    X_array
)


if not np.allclose(
    loaded_transform,
    X_scaled,
    rtol=0.0,
    atol=1e-12,
):

    raise RuntimeError(
        "Reloaded scaler does not reproduce transformation."
    )


scaler_hash = sha256_file(
    SCALER_FILE
)


# ============================================================
# PARAMETER MANIFEST
# ============================================================

parameter_rows = []


for index, feature in enumerate(
    features
):

    parameter_rows.append(
        {
            "feature_index":
                index,

            "feature":
                feature,

            "train_mean":
                float(
                    scaler.mean_[
                        index
                    ]
                ),

            "train_variance":
                float(
                    scaler.var_[
                        index
                    ]
                ),

            "train_scale":
                float(
                    scaler.scale_[
                        index
                    ]
                ),

            "scaled_train_mean":
                float(
                    scaled_means[
                        index
                    ]
                ),

            "scaled_train_std":
                float(
                    scaled_stds[
                        index
                    ]
                ),
        }
    )


parameters = pd.DataFrame(
    parameter_rows
)


parameters.to_csv(
    PARAMETERS_FILE,
    index=False,
)


# ============================================================
# FIT SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "preprocessor":
                "StandardScaler",

            "model":
                "LogisticRegression",

            "fit_scope":
                "CYCLE24_TRAIN_ONLY",

            "train_rows":
                len(
                    X_train
                ),

            "feature_count":
                25,

            "with_mean":
                True,

            "with_std":
                True,

            "n_samples_seen":
                int(
                    scaler.n_samples_seen_
                ),

            "max_abs_scaled_train_mean":
                max_abs_scaled_mean,

            "max_abs_scaled_train_std_error":
                max_abs_std_error,

            "scaled_train_mean_check":
                mean_check,

            "scaled_train_std_check":
                std_check,

            "finite_scaled_values":
                bool(
                    np.isfinite(
                        X_scaled
                    ).all()
                ),

            "validation_used":
                False,

            "holdout_used":
                False,

            "cycle25_used":
                False,

            "train_sha256":
                actual_train_hash,

            "scaler_sha256":
                scaler_hash,

            "sklearn_version":
                sklearn.__version__,
        }
    ]
)


summary.to_csv(
    FIT_SUMMARY_FILE,
    index=False,
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "SCALER FIT"
)

print(
    "===================================="
)


print(
    "Preprocessor: StandardScaler"
)

print(
    "Fit scope: CYCLE24 TRAIN ONLY"
)

print(
    "Samples seen:",
    scaler.n_samples_seen_
)

print(
    "Features:",
    scaler.n_features_in_
)

print(
    "with_mean:",
    scaler.with_mean
)

print(
    "with_std:",
    scaler.with_std
)


print(
    "\n===================================="
)

print(
    "TRAIN TRANSFORMATION CHECK"
)

print(
    "===================================="
)


print(
    "Finite transformed values:",
    np.isfinite(
        X_scaled
    ).all()
)

print(
    "Maximum |scaled mean|:",
    max_abs_scaled_mean
)

print(
    "Maximum |scaled std - 1|:",
    max_abs_std_error
)

print(
    "Mean check:",
    (
        "PASS"
        if mean_check
        else
        "FAIL"
    )
)

print(
    "STD check:",
    (
        "PASS"
        if std_check
        else
        "FAIL"
    )
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
    "TRAIN used:",
    True
)

print(
    "VALIDATION used:",
    False
)

print(
    "HOLDOUT used:",
    False
)

print(
    "Cycle25 used:",
    False
)


print(
    "\n===================================="
)

print(
    "SCALER ARTIFACT"
)

print(
    "===================================="
)


print(
    "File:",
    SCALER_FILE
)

print(
    "SHA256:",
    scaler_hash
)

print(
    "scikit-learn:",
    sklearn.__version__
)


print(
    "\nSaved:"
)

print(
    SCALER_FILE
)

print(
    PARAMETERS_FILE
)

print(
    FIT_SUMMARY_FILE
)


print(
    "\nMODEL CHAPTER 2.4 COMPLETE."
)