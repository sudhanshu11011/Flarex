from pathlib import Path
from datetime import datetime, timezone
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

PREPROCESSING_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/preprocessing_contract.json"
)

LOGISTIC_SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/logistic_standard_scaler.joblib"
)

SELECTION_PROTOCOL_FILE = Path(
    "artifacts/models/selection/contracts/"
    "model_selection_protocol.json"
)

CANDIDATE_REGISTRY_FILE = Path(
    "artifacts/models/selection/manifests/"
    "frozen_candidate_registry.csv"
)

PREDICTION_FILE = Path(
    "artifacts/models/selection/predictions/"
    "cycle24_validation_predictions.parquet"
)

PREDICTION_MANIFEST_FILE = Path(
    "artifacts/models/selection/manifests/"
    "validation_prediction_manifest.csv"
)

PREDICTION_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "validation_prediction_contract.json"
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


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def json_contains_value(obj, target):

    if isinstance(obj, dict):

        return any(
            json_contains_value(
                value,
                target,
            )
            for value in obj.values()
        )

    if isinstance(obj, list):

        return any(
            json_contains_value(
                value,
                target,
            )
            for value in obj
        )

    return str(obj) == str(target)


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 6.2 — GENERATE VALIDATION PREDICTIONS"
)

print(
    "===================================="
)


# ============================================================
# VERIFY REQUIRED INPUTS
# ============================================================

required_files = [
    VALIDATION_FILE,
    SPLIT_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
    LOGISTIC_SCALER_FILE,
    SELECTION_PROTOCOL_FILE,
    CANDIDATE_REGISTRY_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# PROTECT EXISTING OUTPUTS
# ============================================================

for output_path in [
    PREDICTION_FILE,
    PREDICTION_MANIFEST_FILE,
    PREDICTION_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Output already exists: {output_path}\n"
            "Refusing to overwrite validation "
            "prediction provenance."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

split_contract = load_json(
    SPLIT_CONTRACT_FILE
)

feature_contract = load_json(
    FEATURE_CONTRACT_FILE
)

preprocessing_contract = load_json(
    PREPROCESSING_CONTRACT_FILE
)

selection_protocol = load_json(
    SELECTION_PROTOCOL_FILE
)


# ============================================================
# VERIFY CONTRACT STATES
# ============================================================

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


if preprocessing_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Preprocessing contract is not FROZEN."
    )


if selection_protocol.get(
    "status"
) != "FROZEN_BEFORE_VALIDATION":

    raise RuntimeError(
        "Selection protocol is not "
        "FROZEN_BEFORE_VALIDATION."
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
    "Cycle24 split: FROZEN"
)

print(
    "Feature contract: VERIFIED"
)

print(
    "Preprocessing contract: FROZEN"
)

print(
    "Selection protocol: FROZEN_BEFORE_VALIDATION"
)


# ============================================================
# VERIFY VALIDATION HASH BEFORE READING
# ============================================================

validation_contract = split_contract.get(
    "validation"
)


if not isinstance(
    validation_contract,
    dict,
):

    raise RuntimeError(
        "Validation section missing from "
        "split contract."
    )


expected_validation_hash = validation_contract.get(
    "sha256"
)


if not expected_validation_hash:

    raise RuntimeError(
        "Validation SHA256 missing from "
        "split contract."
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
        "Cycle24 validation dataset does not "
        "match frozen split contract."
    )


print(
    "Validation hash: PASS"
)


# ============================================================
# LOAD VALIDATION
#
# THIS IS THE FIRST MODEL-EVALUATION READ OF VALIDATION.
# ============================================================

validation = pd.read_parquet(
    VALIDATION_FILE
)


missing_features = [
    feature

    for feature in features

    if feature not in validation.columns
]


if missing_features:

    raise RuntimeError(
        f"Validation missing frozen features: "
        f"{missing_features}"
    )


if "target" not in validation.columns:

    raise RuntimeError(
        "Validation target column missing."
    )


# ============================================================
# PREPARE VALIDATION ARRAYS
# ============================================================

X_validation_df = (
    validation[
        features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
)


if X_validation_df.isna().any().any():

    raise RuntimeError(
        "Validation contains missing or "
        "non-numeric feature values."
    )


X_validation_float64 = X_validation_df.to_numpy(
    dtype=np.float64
)


X_validation_float32 = X_validation_df.to_numpy(
    dtype=np.float32
)


if not np.isfinite(
    X_validation_float64
).all():

    raise RuntimeError(
        "Validation contains non-finite values."
    )


y_validation = pd.to_numeric(
    validation[
        "target"
    ],
    errors="raise",
).to_numpy(
    dtype=np.int64
)


# ============================================================
# VALIDATION SPLIT INTEGRITY
# ============================================================

if X_validation_float64.shape != (
    13299,
    25,
):

    raise RuntimeError(
        f"Unexpected validation shape: "
        f"{X_validation_float64.shape}"
    )


if len(
    y_validation
) != 13299:

    raise RuntimeError(
        "Unexpected validation target count."
    )


unique_targets = set(
    np.unique(
        y_validation
    ).tolist()
)


if unique_targets != {
    0,
    1,
}:

    raise RuntimeError(
        "Validation target is not binary {0,1}."
    )


validation_positive = int(
    y_validation.sum()
)


validation_negative = int(
    (
        y_validation
        ==
        0
    ).sum()
)


if validation_positive != 515:

    raise RuntimeError(
        "Unexpected validation positive count."
    )


if validation_negative != 12784:

    raise RuntimeError(
        "Unexpected validation negative count."
    )


print(
    "Validation rows:",
    len(
        y_validation
    )
)

print(
    "Validation positives:",
    validation_positive
)

print(
    "Validation negatives:",
    validation_negative
)

print(
    "Features:",
    25
)


# ============================================================
# LOAD FROZEN CANDIDATE REGISTRY
# ============================================================

registry = pd.read_csv(
    CANDIDATE_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    registry
) != 34:

    raise RuntimeError(
        "Expected exactly 34 frozen candidates."
    )


if registry[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Frozen candidate registry contains "
        "duplicate candidate IDs."
    )


family_counts = (
    registry[
        "family"
    ]
    .value_counts()
    .to_dict()
)


if family_counts.get(
    "logistic"
) != 10:

    raise RuntimeError(
        "Expected 10 Logistic candidates."
    )


if family_counts.get(
    "random_forest"
) != 12:

    raise RuntimeError(
        "Expected 12 Random Forest candidates."
    )


if family_counts.get(
    "xgboost"
) != 12:

    raise RuntimeError(
        "Expected 12 XGBoost candidates."
    )


print(
    "Frozen candidate registry: PASS"
)

print(
    "Total candidates:",
    len(
        registry
    )
)


# ============================================================
# VERIFY + LOAD FROZEN LOGISTIC SCALER
# ============================================================

logistic_scaler_hash = sha256_file(
    LOGISTIC_SCALER_FILE
)


# Ensure this exact scaler hash exists somewhere in the
# frozen preprocessing contract, without depending on a
# particular JSON nesting layout.
if not json_contains_value(
    preprocessing_contract,
    logistic_scaler_hash,
):

    raise RuntimeError(
        "Logistic scaler hash is not referenced "
        "by the frozen preprocessing contract."
    )


# ============================================================
# LOAD FROZEN LOGISTIC SCALER BUNDLE
# ============================================================

logistic_scaler_bundle = joblib.load(
    LOGISTIC_SCALER_FILE
)


if not isinstance(
    logistic_scaler_bundle,
    dict,
):

    raise RuntimeError(
        "Frozen Logistic scaler artifact "
        "is expected to be a dictionary bundle."
    )


required_scaler_bundle_keys = {
    "scaler",
    "feature_order",
    "feature_count",
    "fit_scope",
    "train_rows",
    "train_sha256",
    "sklearn_version",
}


missing_scaler_keys = (
    required_scaler_bundle_keys
    -
    set(
        logistic_scaler_bundle.keys()
    )
)


if missing_scaler_keys:

    raise RuntimeError(
        "Frozen Logistic scaler bundle "
        f"is missing keys: {sorted(missing_scaler_keys)}"
    )


logistic_scaler = logistic_scaler_bundle[
    "scaler"
]


if not hasattr(
    logistic_scaler,
    "transform"
):

    raise RuntimeError(
        "Scaler object inside frozen Logistic "
        "bundle does not support transform()."
    )


if int(
    logistic_scaler_bundle[
        "feature_count"
    ]
) != 25:

    raise RuntimeError(
        "Frozen Logistic scaler bundle "
        "feature_count is not 25."
    )


if logistic_scaler_bundle[
    "feature_order"
] != features:

    raise RuntimeError(
        "Frozen Logistic scaler feature order "
        "does not match model feature contract."
    )


if int(
    logistic_scaler.n_features_in_
) != 25:

    raise RuntimeError(
        "Frozen Logistic scaler fitted "
        "feature count is not 25."
    )


if logistic_scaler_bundle[
    "fit_scope"
] != "CYCLE24_TRAIN_ONLY":

    raise RuntimeError(
        "Frozen Logistic scaler was not "
        "fit on Cycle24 TRAIN only."
    )


if logistic_scaler_bundle[
    "train_sha256"
] != split_contract[
    "train"
][
    "sha256"
]:

    raise RuntimeError(
        "Frozen Logistic scaler TRAIN hash "
        "does not match frozen split contract."
    )

X_validation_logistic = logistic_scaler.transform(
    X_validation_float64
)


if X_validation_logistic.shape != (
    13299,
    25,
):

    raise RuntimeError(
        "Unexpected transformed Logistic "
        "validation shape."
    )


if not np.isfinite(
    X_validation_logistic
).all():

    raise RuntimeError(
        "Logistic-transformed validation "
        "contains non-finite values."
    )


print(
    "Frozen Logistic scaler: VERIFIED"
)

print(
    "Logistic scaler refit:",
    False
)


# ============================================================
# CREATE BASE PREDICTION TABLE
# ============================================================

prediction_df = pd.DataFrame(
    {
        "validation_row_id":
            np.arange(
                len(
                    y_validation
                ),
                dtype=np.int64,
            ),

        "target":
            y_validation,
    }
)


manifest_rows = []


print(
    "\n===================================="
)

print(
    "GENERATING VALIDATION PROBABILITIES"
)

print(
    "===================================="
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

for _, registry_row in registry.iterrows():

    family = str(
        registry_row[
            "family"
        ]
    ).strip()


    candidate_id = str(
        registry_row[
            "candidate_id"
        ]
    ).strip()


    model_file = Path(
        str(
            registry_row[
                "model_file"
            ]
        )
    )


    expected_model_hash = str(
        registry_row[
            "model_sha256"
        ]
    ).strip()


    expected_model_family = str(
        registry_row[
            "model_family"
        ]
    ).strip()


    # --------------------------------------------------------
    # VERIFY MODEL ARTIFACT
    # --------------------------------------------------------

    if not model_file.exists():

        raise FileNotFoundError(
            f"{candidate_id}: frozen model "
            f"artifact missing: {model_file}"
        )


    actual_model_hash = sha256_file(
        model_file
    )


    if (
        actual_model_hash
        !=
        expected_model_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: model hash differs "
            "from frozen candidate registry."
        )


    # --------------------------------------------------------
    # LOAD BUNDLE
    # --------------------------------------------------------

    bundle = joblib.load(
        model_file
    )


    model = bundle.get(
        "model"
    )


    if model is None:

        raise RuntimeError(
            f"{candidate_id}: model object missing "
            "inside frozen artifact."
        )


    if bundle.get(
        "candidate_id"
    ) != candidate_id:

        raise RuntimeError(
            f"{candidate_id}: bundle candidate ID "
            "does not match registry."
        )


    if bundle.get(
        "model_family"
    ) != expected_model_family:

        raise RuntimeError(
            f"{candidate_id}: model-family mismatch."
        )


    if bundle.get(
        "feature_order"
    ) != features:

        raise RuntimeError(
            f"{candidate_id}: frozen feature "
            "order mismatch."
        )


    if int(
        bundle.get(
            "feature_count"
        )
    ) != 25:

        raise RuntimeError(
            f"{candidate_id}: feature count mismatch."
        )


    # ========================================================
    # FAMILY-SPECIFIC TRANSFORM
    # ========================================================

    if family == "logistic":

        X_model = X_validation_logistic

        preprocessing_used = (
            "FROZEN_STANDARD_SCALER_TRANSFORM_ONLY"
        )


    elif family == "random_forest":

        X_model = X_validation_float64

        preprocessing_used = (
            "IDENTITY_RAW_FLOAT64"
        )


    elif family == "xgboost":

        X_model = X_validation_float32

        preprocessing_used = (
            "IDENTITY_RAW_FLOAT32"
        )


    else:

        raise RuntimeError(
            f"{candidate_id}: unknown family "
            f"{family}"
        )


    # ========================================================
    # PROBABILITY PREDICTION
    # ========================================================

    probabilities = model.predict_proba(
        X_model
    )


    if probabilities.shape != (
        13299,
        2,
    ):

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            f"predict_proba shape "
            f"{probabilities.shape}"
        )


    positive_probabilities = np.asarray(
        probabilities[
            :,
            1
        ],
        dtype=np.float64,
    )


    if positive_probabilities.shape != (
        13299,
    ):

        raise RuntimeError(
            f"{candidate_id}: invalid positive "
            "probability vector shape."
        )


    if not np.isfinite(
        positive_probabilities
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "validation probabilities."
        )


    probability_min = float(
        positive_probabilities.min()
    )


    probability_max = float(
        positive_probabilities.max()
    )


    probability_mean = float(
        positive_probabilities.mean()
    )


    probability_std = float(
        positive_probabilities.std()
    )


    if (
        probability_min < 0.0
        or
        probability_max > 1.0
    ):

        raise RuntimeError(
            f"{candidate_id}: probabilities "
            "outside [0,1]."
        )


    # --------------------------------------------------------
    # STORE WITHOUT THRESHOLDING
    # --------------------------------------------------------

    prediction_df[
        candidate_id
    ] = positive_probabilities


    manifest_rows.append(
        {
            "family":
                family,

            "model_family":
                expected_model_family,

            "candidate_id":
                candidate_id,

            "validation_rows":
                len(
                    positive_probabilities
                ),

            "validation_positive":
                validation_positive,

            "validation_negative":
                validation_negative,

            "probability_min":
                probability_min,

            "probability_max":
                probability_max,

            "probability_mean":
                probability_mean,

            "probability_std":
                probability_std,

            "probabilities_finite":
                True,

            "probabilities_within_0_1":
                True,

            "preprocessing_used":
                preprocessing_used,

            "model_file":
                str(
                    model_file
                ),

            "model_sha256":
                actual_model_hash,

            "threshold_applied":
                False,

            "metrics_computed":
                False,

            "model_selection_performed":
                False,
        }
    )


    print(
        f"{candidate_id}: "
        f"family={family}, "
        f"rows={len(positive_probabilities)}, "
        f"p_min={probability_min:.6f}, "
        f"p_max={probability_max:.6f}, "
        f"p_mean={probability_mean:.6f}, "
        f"PASS=True"
    )


# ============================================================
# VERIFY PREDICTION TABLE
# ============================================================

expected_columns = (
    2
    +
    34
)


if prediction_df.shape != (
    13299,
    expected_columns,
):

    raise RuntimeError(
        f"Unexpected prediction table shape: "
        f"{prediction_df.shape}"
    )


candidate_columns = [
    column

    for column in prediction_df.columns

    if column not in {
        "validation_row_id",
        "target",
    }
]


if set(
    candidate_columns
) != set(
    registry[
        "candidate_id"
    ].astype(str)
):

    raise RuntimeError(
        "Prediction columns do not exactly match "
        "frozen candidate registry."
    )


manifest_df = pd.DataFrame(
    manifest_rows
)


if len(
    manifest_df
) != 34:

    raise RuntimeError(
        "Validation prediction manifest does "
        "not contain 34 candidates."
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

PREDICTION_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


PREDICTION_MANIFEST_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


PREDICTION_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# WRITE PREDICTION FILE
# ============================================================

prediction_df.to_parquet(
    PREDICTION_FILE,
    index=False,
)


prediction_file_hash = sha256_file(
    PREDICTION_FILE
)


# ============================================================
# VERIFY ROUND-TRIP
# ============================================================

roundtrip = pd.read_parquet(
    PREDICTION_FILE
)


if roundtrip.shape != prediction_df.shape:

    raise RuntimeError(
        "Prediction parquet round-trip "
        "shape mismatch."
    )


if roundtrip[
    "validation_row_id"
].tolist() != prediction_df[
    "validation_row_id"
].tolist():

    raise RuntimeError(
        "Prediction parquet row-order mismatch."
    )


if not np.array_equal(
    roundtrip[
        "target"
    ].to_numpy(
        dtype=np.int64
    ),
    y_validation,
):

    raise RuntimeError(
        "Prediction parquet target mismatch."
    )


for candidate_id in candidate_columns:

    if not np.array_equal(
        roundtrip[
            candidate_id
        ].to_numpy(
            dtype=np.float64
        ),
        prediction_df[
            candidate_id
        ].to_numpy(
            dtype=np.float64
        ),
    ):

        raise RuntimeError(
            f"{candidate_id}: prediction parquet "
            "round-trip mismatch."
        )


print(
    "\nPrediction parquet round-trip: PASS"
)


# ============================================================
# ADD PREDICTION ARTIFACT HASH TO MANIFEST
# ============================================================

manifest_df[
    "validation_prediction_file"
] = str(
    PREDICTION_FILE
)


manifest_df[
    "validation_prediction_sha256"
] = prediction_file_hash


manifest_df.to_csv(
    PREDICTION_MANIFEST_FILE,
    index=False,
)


# ============================================================
# BUILD PREDICTION CONTRACT
# ============================================================

prediction_contract = {

    "contract_name":
        "Cycle24 Validation Prediction Artifact",

    "version":
        "1.0",

    "status":
        "FROZEN_VALIDATION_PREDICTIONS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "validation_dataset":
        {
            "file":
                str(
                    VALIDATION_FILE
                ),

            "sha256":
                actual_validation_hash,

            "rows":
                13299,

            "positive":
                validation_positive,

            "negative":
                validation_negative,

            "feature_count":
                25,
        },

    "candidate_counts":
        {
            "logistic":
                10,

            "random_forest":
                12,

            "xgboost":
                12,

            "total":
                34,
        },

    "prediction_artifact":
        {
            "file":
                str(
                    PREDICTION_FILE
                ),

            "sha256":
                prediction_file_hash,

            "rows":
                13299,

            "columns":
                int(
                    prediction_df.shape[
                        1
                    ]
                ),

            "candidate_probability_columns":
                34,

            "decision_threshold_applied":
                False,

            "classification_labels_generated":
                False,

            "metrics_computed":
                False,
        },

    "preprocessing":
        {
            "logistic":
                "Frozen StandardScaler transform only",

            "random_forest":
                "Identity raw features",

            "xgboost":
                "Identity raw features",

            "scaler_refit":
                False,

            "logistic_scaler_file":
                str(
                    LOGISTIC_SCALER_FILE
                ),

            "logistic_scaler_sha256":
                logistic_scaler_hash,
        },

    "selection_state":
        {
            "thresholds_selected":
                False,

            "candidate_metrics_computed":
                False,

            "family_winners_selected":
                False,

            "cross_family_winner_selected":
                False,
        },

    "data_isolation":
        {
            "train_used_for_new_fitting":
                False,

            "validation_read":
                True,

            "holdout_read":
                False,

            "cycle25_read":
                False,
        },

    "immutability_rule":
        (
            "These validation probabilities were generated "
            "from the already-frozen TRAIN-only model "
            "artifacts using the frozen preprocessing rules. "
            "The prediction artifact must not be regenerated "
            "with changed models, features, preprocessing, "
            "or candidate definitions after validation "
            "outcomes are inspected."
        ),

    "provenance":
        {
            "selection_protocol":
                {
                    "file":
                        str(
                            SELECTION_PROTOCOL_FILE
                        ),

                    "sha256":
                        sha256_file(
                            SELECTION_PROTOCOL_FILE
                        ),
                },

            "candidate_registry":
                {
                    "file":
                        str(
                            CANDIDATE_REGISTRY_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CANDIDATE_REGISTRY_FILE
                        ),
                },

            "split_contract":
                {
                    "file":
                        str(
                            SPLIT_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            SPLIT_CONTRACT_FILE
                        ),
                },

            "feature_contract":
                {
                    "file":
                        str(
                            FEATURE_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            FEATURE_CONTRACT_FILE
                        ),
                },

            "preprocessing_contract":
                {
                    "file":
                        str(
                            PREPROCESSING_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            PREPROCESSING_CONTRACT_FILE
                        ),
                },
        },
}


with PREDICTION_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        prediction_contract,
        file,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "VALIDATION PREDICTION SUMMARY"
)

print(
    "===================================="
)


print(
    "Validation rows:",
    13299
)

print(
    "Validation positives:",
    validation_positive
)

print(
    "Validation negatives:",
    validation_negative
)

print(
    "Logistic candidates:",
    family_counts[
        "logistic"
    ]
)

print(
    "Random Forest candidates:",
    family_counts[
        "random_forest"
    ]
)

print(
    "XGBoost candidates:",
    family_counts[
        "xgboost"
    ]
)

print(
    "Total candidate probability vectors:",
    len(
        candidate_columns
    )
)

print(
    "Prediction table shape:",
    prediction_df.shape
)

print(
    "Prediction artifact SHA256:",
    prediction_file_hash
)


# ============================================================
# SELECTION STATE
# ============================================================

print(
    "\n===================================="
)

print(
    "SELECTION STATE"
)

print(
    "===================================="
)


print(
    "Threshold sweep performed:",
    False
)

print(
    "Validation metrics computed:",
    False
)

print(
    "Threshold selected:",
    False
)

print(
    "Family winners selected:",
    False
)

print(
    "Models retrained:",
    False
)


# ============================================================
# DATA ISOLATION
# ============================================================

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
    "Cycle24 VALIDATION read:",
    True
)

print(
    "Cycle24 HOLDOUT read:",
    False
)

print(
    "Cycle25 read:",
    False
)

print(
    "TRAIN used for new fitting:",
    False
)

print(
    "Logistic scaler refit:",
    False
)


# ============================================================
# OUTPUTS
# ============================================================

print(
    "\nSaved:"
)

print(
    PREDICTION_FILE
)

print(
    PREDICTION_MANIFEST_FILE
)

print(
    PREDICTION_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 6.2 COMPLETE — VALIDATION PREDICTIONS FROZEN"
)

print(
    "===================================="
)