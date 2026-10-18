from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/model_feature_contract.json"
)

DESIGN_FILE = Path(
    "artifacts/preprocessing/contracts/preprocessing_design.json"
)

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/cycle24_split_contract.json"
)

SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/logistic_standard_scaler.joblib"
)

TRAIN_DISTRIBUTION_FILE = Path(
    "artifacts/preprocessing/manifests/"
    "model_ch2_train_feature_distribution.csv"
)

TRAIN_DISTRIBUTION_SUMMARY_FILE = Path(
    "artifacts/preprocessing/manifests/"
    "model_ch2_train_distribution_summary.csv"
)

SCALER_PARAMETERS_FILE = Path(
    "artifacts/preprocessing/manifests/"
    "model_ch2_logistic_scaler_parameters.csv"
)

SCALER_FIT_SUMMARY_FILE = Path(
    "artifacts/preprocessing/manifests/"
    "model_ch2_logistic_scaler_fit_summary.csv"
)

VALIDATION_STATS_FILE = Path(
    "artifacts/preprocessing/manifests/"
    "model_ch2_validation_feature_stats.csv"
)

VALIDATION_AUDIT_FILE = Path(
    "artifacts/preprocessing/manifests/"
    "model_ch2_validation_transform_audit.csv"
)


CONTRACT_OUTPUT = Path(
    "artifacts/preprocessing/contracts/"
    "preprocessing_contract.json"
)

FILES_OUTPUT = Path(
    "artifacts/preprocessing/manifests/"
    "model_ch2_preprocessing_files.csv"
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
# INPUT CHECK
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 2.6 — FREEZE PREPROCESSING CONTRACT"
)

print(
    "===================================="
)


required_files = [
    FEATURE_CONTRACT_FILE,
    DESIGN_FILE,
    SPLIT_CONTRACT_FILE,
    SCALER_FILE,
    TRAIN_DISTRIBUTION_FILE,
    TRAIN_DISTRIBUTION_SUMMARY_FILE,
    SCALER_PARAMETERS_FILE,
    SCALER_FIT_SUMMARY_FILE,
    VALIDATION_STATS_FILE,
    VALIDATION_AUDIT_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

with FEATURE_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    feature_contract = json.load(file)


with DESIGN_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    design = json.load(file)


with SPLIT_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    split_contract = json.load(file)


if feature_contract.get(
    "status"
) != "VERIFIED":

    raise RuntimeError(
        "Feature contract is not VERIFIED."
    )


if design.get(
    "status"
) != "DESIGN_FROZEN":

    raise RuntimeError(
        "Preprocessing design is not frozen."
    )


if split_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 split contract is not frozen."
    )


features = feature_contract.get(
    "feature_order",
    []
)


if len(features) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


print(
    "Feature contract: VERIFIED"
)

print(
    "Preprocessing design: FROZEN"
)

print(
    "Cycle24 split: FROZEN"
)


# ============================================================
# VERIFY TRAIN DISTRIBUTION AUDIT
# ============================================================

distribution_summary = pd.read_csv(
    TRAIN_DISTRIBUTION_SUMMARY_FILE
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
        "TRAIN numerical-integrity audit failed."
    )


if bool(
    distribution_row[
        "validation_used"
    ]
):

    raise RuntimeError(
        "Validation was used during TRAIN distribution audit."
    )


if bool(
    distribution_row[
        "holdout_used"
    ]
):

    raise RuntimeError(
        "Holdout was used during TRAIN distribution audit."
    )


if bool(
    distribution_row[
        "cycle25_used"
    ]
):

    raise RuntimeError(
        "Cycle25 was used during TRAIN distribution audit."
    )


print(
    "TRAIN distribution audit: PASS"
)


# ============================================================
# VERIFY SCALER FIT
# ============================================================

fit_summary = pd.read_csv(
    SCALER_FIT_SUMMARY_FILE
)


if len(fit_summary) != 1:

    raise RuntimeError(
        "Unexpected scaler fit summary."
    )


fit_row = fit_summary.iloc[0]


if fit_row[
    "fit_scope"
] != "CYCLE24_TRAIN_ONLY":

    raise RuntimeError(
        "Scaler fit scope is not TRAIN only."
    )


if int(
    fit_row[
        "train_rows"
    ]
) != 62064:

    raise RuntimeError(
        "Unexpected scaler TRAIN row count."
    )


if bool(
    fit_row[
        "validation_used"
    ]
):

    raise RuntimeError(
        "Validation used during scaler fitting."
    )


if bool(
    fit_row[
        "holdout_used"
    ]
):

    raise RuntimeError(
        "Holdout used during scaler fitting."
    )


if bool(
    fit_row[
        "cycle25_used"
    ]
):

    raise RuntimeError(
        "Cycle25 used during scaler fitting."
    )


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
        "Scaler artifact hash mismatch."
    )


print(
    "TRAIN-only scaler fit: PASS"
)

print(
    "Scaler hash verification: PASS"
)


# ============================================================
# VERIFY SAVED SCALER BUNDLE
# ============================================================

bundle = joblib.load(
    SCALER_FILE
)


if bundle.get(
    "feature_order"
) != features:

    raise RuntimeError(
        "Scaler feature order mismatch."
    )


if bundle.get(
    "fit_scope"
) != "CYCLE24_TRAIN_ONLY":

    raise RuntimeError(
        "Saved scaler fit scope mismatch."
    )


if int(
    bundle.get(
        "train_rows"
    )
) != 62064:

    raise RuntimeError(
        "Saved scaler TRAIN count mismatch."
    )


scaler = bundle.get(
    "scaler"
)


if scaler is None:

    raise RuntimeError(
        "Scaler object missing from bundle."
    )


if int(
    scaler.n_features_in_
) != 25:

    raise RuntimeError(
        "Scaler feature count mismatch."
    )


# ============================================================
# VERIFY VALIDATION AUDIT
# ============================================================

validation_audit = pd.read_csv(
    VALIDATION_AUDIT_FILE
)


if len(
    validation_audit
) != 1:

    raise RuntimeError(
        "Unexpected validation audit."
    )


validation_row = validation_audit.iloc[0]


if validation_row[
    "operation"
] != "TRANSFORM_ONLY":

    raise RuntimeError(
        "Validation operation was not transform-only."
    )


if bool(
    validation_row[
        "scaler_refit"
    ]
):

    raise RuntimeError(
        "Scaler was refitted on validation."
    )


if not bool(
    validation_row[
        "validation_transform_pass"
    ]
):

    raise RuntimeError(
        "Validation transform audit failed."
    )


if bool(
    validation_row[
        "holdout_used"
    ]
):

    raise RuntimeError(
        "Holdout used during validation audit."
    )


if bool(
    validation_row[
        "cycle25_used"
    ]
):

    raise RuntimeError(
        "Cycle25 used during validation audit."
    )


print(
    "Validation transform-only audit: PASS"
)


# ============================================================
# FILE MANIFEST
# ============================================================

tracked_files = {

    "feature_contract":
        FEATURE_CONTRACT_FILE,

    "preprocessing_design":
        DESIGN_FILE,

    "cycle24_split_contract":
        SPLIT_CONTRACT_FILE,

    "logistic_standard_scaler":
        SCALER_FILE,

    "train_feature_distribution":
        TRAIN_DISTRIBUTION_FILE,

    "train_distribution_summary":
        TRAIN_DISTRIBUTION_SUMMARY_FILE,

    "logistic_scaler_parameters":
        SCALER_PARAMETERS_FILE,

    "logistic_scaler_fit_summary":
        SCALER_FIT_SUMMARY_FILE,

    "validation_feature_stats":
        VALIDATION_STATS_FILE,

    "validation_transform_audit":
        VALIDATION_AUDIT_FILE,
}


file_rows = []


for name, path in tracked_files.items():

    file_rows.append(
        {
            "artifact":
                name,

            "file":
                str(path),

            "sha256":
                sha256_file(
                    path
                ),

            "size_bytes":
                path.stat().st_size,
        }
    )


files_df = pd.DataFrame(
    file_rows
)


FILES_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


files_df.to_csv(
    FILES_OUTPUT,
    index=False,
)


# ============================================================
# FINAL PREPROCESSING CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Frozen Model Preprocessing Contract",

    "version":
        "1.0",

    "status":
        "FROZEN",

    "frozen_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source_split_version":
        split_contract.get(
            "split_version"
        ),

    "feature_contract_version":
        feature_contract.get(
            "version"
        ),

    "feature_count":
        25,

    "feature_order":
        features,

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    "logistic_regression": {

        "preprocessor":
            "StandardScaler",

        "artifact":
            str(
                SCALER_FILE
            ),

        "artifact_sha256":
            actual_scaler_hash,

        "fit_scope":
            "CYCLE24_TRAIN_ONLY",

        "fit_rows":
            62064,

        "with_mean":
            True,

        "with_std":
            True,

        "validation_operation":
            "TRANSFORM_ONLY",

        "holdout_operation":
            (
                "TRANSFORM_ONLY_AFTER_"
                "MODEL_DEVELOPMENT_FREEZE"
            ),

        "cycle25_operation":
            (
                "TRANSFORM_ONLY_AFTER_"
                "FULL_CYCLE24_PIPELINE_FREEZE"
            ),

        "refit_allowed":
            False,

        "feature_order_locked":
            True,
    },

    # --------------------------------------------------------
    # TREE MODELS
    # --------------------------------------------------------

    "random_forest": {

        "preprocessor":
            "IDENTITY",

        "scaling":
            False,

        "feature_order_locked":
            True,
    },

    "xgboost": {

        "preprocessor":
            "IDENTITY",

        "scaling":
            False,

        "feature_order_locked":
            True,
    },

    # --------------------------------------------------------
    # GLOBAL RULES
    # --------------------------------------------------------

    "global_rules": {

        "imputation":
            False,

        "pca":
            False,

        "feature_selection":
            False,

        "outlier_clipping":
            False,

        "winsorization":
            False,

        "log_transform":
            False,

        "polynomial_features":
            False,

        "interaction_feature_engineering":
            False,

        "preserve_original_feature_identity":
            True,
    },

    # --------------------------------------------------------
    # DATA ISOLATION
    # --------------------------------------------------------

    "data_isolation": {

        "train_used_for_fit":
            True,

        "validation_used_for_fit":
            False,

        "validation_transform_verified":
            True,

        "holdout_used":
            False,

        "cycle25_used":
            False,
    },

    # --------------------------------------------------------
    # IMMUTABILITY
    # --------------------------------------------------------

    "immutability_rule":
        (
            "The preprocessing design, feature order, and "
            "Logistic Regression StandardScaler parameters "
            "are frozen. They must not be changed or refitted "
            "in response to validation, holdout, or Cycle25 "
            "model performance. Any scientifically necessary "
            "change requires a new preprocessing contract "
            "version and explicit justification."
        ),

    # --------------------------------------------------------
    # SOFTWARE
    # --------------------------------------------------------

    "software": {

        "scikit_learn_version":
            bundle.get(
                "sklearn_version"
            ),
    },

    # --------------------------------------------------------
    # PROVENANCE
    # --------------------------------------------------------

    "provenance": {

        name: {
            "file":
                str(path),

            "sha256":
                sha256_file(
                    path
                ),
        }

        for name, path
        in tracked_files.items()
    },
}


# ============================================================
# SAFE WRITE
# ============================================================

CONTRACT_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


if CONTRACT_OUTPUT.exists():

    with CONTRACT_OUTPUT.open(
        "r",
        encoding="utf-8",
    ) as file:

        existing = json.load(
            file
        )


    existing_hash = (
        existing
        .get(
            "logistic_regression",
            {}
        )
        .get(
            "artifact_sha256"
        )
    )


    if (
        existing_hash
        !=
        actual_scaler_hash
    ):

        raise RuntimeError(
            "Existing preprocessing contract "
            "references a different scaler. "
            "Frozen contract will NOT be overwritten."
        )


    print(
        "Existing preprocessing contract "
        "matches current scaler."
    )


else:

    with CONTRACT_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            contract,
            file,
            indent=2,
        )


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "FROZEN PREPROCESSING"
)

print(
    "===================================="
)


print(
    "Feature count: 25"
)

print(
    "Feature order: FROZEN"
)


print(
    "\nLogistic Regression:"
)

print(
    " StandardScaler"
)

print(
    " Fit scope: CYCLE24 TRAIN ONLY"
)

print(
    " Fit rows: 62064"
)

print(
    " Refit allowed: False"
)

print(
    " Scaler SHA256:",
    actual_scaler_hash
)


print(
    "\nRandom Forest:"
)

print(
    " Identity preprocessing"
)


print(
    "\nXGBoost:"
)

print(
    " Identity preprocessing"
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
    "TRAIN used for scaler fit: True"
)

print(
    "VALIDATION used for scaler fit: False"
)

print(
    "Validation transform verified: True"
)

print(
    "HOLDOUT used: False"
)

print(
    "Cycle25 used: False"
)


print(
    "\n===================================="
)

print(
    "PREPROCESSING CONTRACT"
)

print(
    "===================================="
)


print(
    "Version: 1.0"
)

print(
    "Status: FROZEN"
)

print(
    "Scaler refitting after this point: FORBIDDEN"
)

print(
    "Performance-driven preprocessing changes: FORBIDDEN"
)


print(
    "\nSaved:"
)

print(
    FILES_OUTPUT
)

print(
    CONTRACT_OUTPUT
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 2.6 COMPLETE — PREPROCESSING FROZEN"
)

print(
    "===================================="
)