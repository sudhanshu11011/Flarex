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

HOLDOUT_FILE = Path(
    "artifacts/splitting/data/"
    "cycle24_holdout.parquet"
)

HOLDOUT_PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_evaluation_protocol.json"
)

HOLDOUT_MODEL_REGISTRY_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "holdout_frozen_model_registry.csv"
)

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/"
    "cycle24_split_contract.json"
)

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "model_feature_contract.json"
)

PREPROCESSING_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "preprocessing_contract.json"
)

LOGISTIC_SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/"
    "logistic_standard_scaler.joblib"
)

OUTPUT_PREDICTIONS_FILE = Path(
    "artifacts/models/evaluation/predictions/"
    "cycle24_holdout_predictions.parquet"
)

OUTPUT_MANIFEST_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "holdout_prediction_manifest.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_prediction_contract.json"
)


# ============================================================
# FROZEN EXPECTATIONS
# ============================================================

EXPECTED_HOLDOUT_SHA256 = (
    "8abb3929febcde7d778c4781ac890ff5e"
    "658933c2ef2a06aa19da085e7e255fc"
)

EXPECTED_ROWS = 13072
EXPECTED_POSITIVES = 527
EXPECTED_NEGATIVES = 12545
EXPECTED_ENTITY_GROUPS = 176

ENTITY_GROUP_SOURCE_COLUMN = "ENTITY_GROUP_ID"


# ============================================================
# FROZEN FEATURE ORDER
# ============================================================

FEATURES = [
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
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
]


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

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


def json_contains_exact_list(
    obj,
    expected_list,
):

    if isinstance(
        obj,
        list,
    ):

        if obj == expected_list:
            return True

        return any(
            json_contains_exact_list(
                item,
                expected_list,
            )
            for item in obj
        )

    if isinstance(
        obj,
        dict,
    ):

        return any(
            json_contains_exact_list(
                value,
                expected_list,
            )
            for value in obj.values()
        )

    return False


def extract_estimator(
    loaded_object,
    candidate_id,
):

    # --------------------------------------------------------
    # Direct estimator artifact
    # --------------------------------------------------------

    if hasattr(
        loaded_object,
        "predict_proba",
    ):

        return loaded_object


    # --------------------------------------------------------
    # Dictionary model bundle
    # --------------------------------------------------------

    if isinstance(
        loaded_object,
        dict,
    ):

        preferred_keys = [
            "model",
            "estimator",
            "classifier",
        ]


        for key in preferred_keys:

            if (
                key in loaded_object
                and
                hasattr(
                    loaded_object[key],
                    "predict_proba",
                )
            ):

                return loaded_object[
                    key
                ]


        estimator_candidates = [
            value
            for value in loaded_object.values()
            if hasattr(
                value,
                "predict_proba",
            )
        ]


        if len(
            estimator_candidates
        ) == 1:

            return estimator_candidates[
                0
            ]


    raise RuntimeError(
        f"{candidate_id}: could not locate "
        "a unique estimator supporting "
        "predict_proba inside frozen artifact."
    )


def get_model_feature_order(
    loaded_object,
):

    if not isinstance(
        loaded_object,
        dict,
    ):

        return None


    for key in [
        "feature_order",
        "features",
        "feature_names",
    ]:

        if key in loaded_object:

            value = loaded_object[
                key
            ]

            if isinstance(
                value,
                (
                    list,
                    tuple,
                ),
            ):

                return list(
                    value
                )


    return None


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 7.2 — GENERATE CYCLE24 HOLDOUT PREDICTIONS"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    HOLDOUT_FILE,
    HOLDOUT_PROTOCOL_FILE,
    HOLDOUT_MODEL_REGISTRY_FILE,
    SPLIT_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
    LOGISTIC_SCALER_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# PROTECT OUTPUTS
# ============================================================

for output_path in [
    OUTPUT_PREDICTIONS_FILE,
    OUTPUT_MANIFEST_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Output already exists: {output_path}\n"
            "Refusing to overwrite holdout "
            "prediction provenance."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

holdout_protocol = load_json(
    HOLDOUT_PROTOCOL_FILE
)

split_contract = load_json(
    SPLIT_CONTRACT_FILE
)

feature_contract = load_json(
    FEATURE_CONTRACT_FILE
)

preprocessing_contract = load_json(
    PREPROCESSING_CONTRACT_FILE
)


# ============================================================
# VERIFY CONTRACT STATES
# ============================================================

if holdout_protocol.get(
    "status"
) != "FROZEN_BEFORE_HOLDOUT_EVALUATION":

    raise RuntimeError(
        "Holdout evaluation protocol "
        "is not frozen."
    )


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
        "Feature contract is not VERIFIED."
    )


if preprocessing_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Preprocessing contract is not FROZEN."
    )


print(
    "Holdout evaluation protocol:",
    "FROZEN"
)

print(
    "Cycle24 split:",
    "FROZEN"
)

print(
    "Feature contract:",
    "VERIFIED"
)

print(
    "Preprocessing contract:",
    "FROZEN"
)


# ============================================================
# VERIFY FEATURE CONTRACT
# ============================================================

if len(
    FEATURES
) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


if len(
    set(
        FEATURES
    )
) != 25:

    raise RuntimeError(
        "Duplicate frozen features detected."
    )


if not json_contains_exact_list(
    feature_contract,
    FEATURES,
):

    raise RuntimeError(
        "Exact 25-feature ordering was not "
        "found in frozen feature contract."
    )


print(
    "Frozen feature order:",
    "PASS"
)

print(
    "Feature count:",
    len(
        FEATURES
    )
)


# ============================================================
# VERIFY HOLDOUT HASH BEFORE READING
# ============================================================

actual_holdout_hash = sha256_file(
    HOLDOUT_FILE
)


if (
    actual_holdout_hash
    !=
    EXPECTED_HOLDOUT_SHA256
):

    raise RuntimeError(
        "Cycle24 HOLDOUT SHA256 differs "
        "from frozen split artifact."
    )


protocol_expected_hash = (
    holdout_protocol[
        "holdout_dataset"
    ][
        "expected_sha256"
    ]
)


if (
    actual_holdout_hash
    !=
    protocol_expected_hash
):

    raise RuntimeError(
        "Cycle24 HOLDOUT SHA256 differs "
        "from Chapter 7.1 protocol."
    )


print(
    "Holdout artifact hash:",
    "PASS"
)


# ============================================================
# HOLDOUT READ
# ============================================================

holdout = pd.read_parquet(
    HOLDOUT_FILE
)


# ============================================================
# VERIFY HOLDOUT STRUCTURE
# ============================================================

required_columns = (
    set(
        FEATURES
    )
    |
    {
        "target",
        ENTITY_GROUP_SOURCE_COLUMN,
        "HARPNUM",
    }
)


missing_columns = (
    required_columns
    -
    set(
        holdout.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "HOLDOUT missing required columns: "
        f"{sorted(missing_columns)}"
    )


if len(
    holdout
) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Unexpected holdout row count: "
        f"{len(holdout)}"
    )


# ============================================================
# TARGET CHECK
# ============================================================

target = holdout[
    "target"
].to_numpy(
    dtype=np.int64
)


unique_target_values = set(
    np.unique(
        target
    ).tolist()
)


if not unique_target_values.issubset(
    {
        0,
        1,
    }
):

    raise RuntimeError(
        "Holdout target is not binary."
    )


positive_count = int(
    target.sum()
)


negative_count = int(
    len(
        target
    )
    -
    positive_count
)


if (
    positive_count
    !=
    EXPECTED_POSITIVES
):

    raise RuntimeError(
        "Unexpected holdout positive count."
    )


if (
    negative_count
    !=
    EXPECTED_NEGATIVES
):

    raise RuntimeError(
        "Unexpected holdout negative count."
    )


# ============================================================
# ENTITY GROUP CHECK
# ============================================================

if holdout[
    ENTITY_GROUP_SOURCE_COLUMN
].isna().any():

    raise RuntimeError(
        "Missing ENTITY_GROUP_ID values "
        "detected in holdout."
    )


entity_group_count = int(
    holdout[
        ENTITY_GROUP_SOURCE_COLUMN
    ].nunique(
        dropna=False
    )
)


if (
    entity_group_count
    !=
    EXPECTED_ENTITY_GROUPS
):

    raise RuntimeError(
        "Unexpected holdout ENTITY_GROUP count: "
        f"{entity_group_count}"
    )


# ============================================================
# HARP CHECK
# ============================================================

if holdout[
    "HARPNUM"
].isna().any():

    raise RuntimeError(
        "Missing HARPNUM values "
        "detected in holdout."
    )


# ============================================================
# FEATURE MATRIX
# ============================================================

X_raw_float64 = holdout[
    FEATURES
].to_numpy(
    dtype=np.float64
)


if X_raw_float64.shape != (
    EXPECTED_ROWS,
    25,
):

    raise RuntimeError(
        "Unexpected raw holdout feature "
        "matrix shape."
    )


if not np.isfinite(
    X_raw_float64
).all():

    raise RuntimeError(
        "Non-finite frozen feature values "
        "detected in holdout."
    )


print(
    "Holdout rows:",
    len(
        holdout
    )
)

print(
    "Holdout positives:",
    positive_count
)

print(
    "Holdout negatives:",
    negative_count
)

print(
    "Holdout ENTITY_GROUPs:",
    entity_group_count
)

print(
    "ENTITY_GROUP source column:",
    ENTITY_GROUP_SOURCE_COLUMN
)

print(
    "Holdout feature integrity:",
    "PASS"
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
        "must be a dictionary bundle."
    )


required_scaler_keys = {
    "scaler",
    "feature_order",
    "feature_count",
    "fit_scope",
    "train_rows",
    "train_sha256",
    "sklearn_version",
}


missing_scaler_keys = (
    required_scaler_keys
    -
    set(
        logistic_scaler_bundle.keys()
    )
)


if missing_scaler_keys:

    raise RuntimeError(
        "Frozen Logistic scaler bundle "
        f"missing keys: "
        f"{sorted(missing_scaler_keys)}"
    )


logistic_scaler = (
    logistic_scaler_bundle[
        "scaler"
    ]
)


if not hasattr(
    logistic_scaler,
    "transform",
):

    raise RuntimeError(
        "Frozen Logistic scaler does "
        "not support transform()."
    )


if list(
    logistic_scaler_bundle[
        "feature_order"
    ]
) != FEATURES:

    raise RuntimeError(
        "Logistic scaler feature order "
        "differs from frozen feature order."
    )


if int(
    logistic_scaler_bundle[
        "feature_count"
    ]
) != 25:

    raise RuntimeError(
        "Logistic scaler feature count "
        "is not 25."
    )


if (
    logistic_scaler_bundle[
        "fit_scope"
    ]
    !=
    "CYCLE24_TRAIN_ONLY"
):

    raise RuntimeError(
        "Logistic scaler was not fit "
        "on Cycle24 TRAIN only."
    )


if int(
    logistic_scaler.n_features_in_
) != 25:

    raise RuntimeError(
        "Frozen Logistic scaler fitted "
        "feature count is not 25."
    )


# ============================================================
# TRANSFORM HOLDOUT — NO REFIT
# ============================================================

X_logistic = logistic_scaler.transform(
    X_raw_float64
)


if X_logistic.shape != (
    EXPECTED_ROWS,
    25,
):

    raise RuntimeError(
        "Unexpected Logistic-scaled "
        "holdout matrix shape."
    )


if not np.isfinite(
    X_logistic
).all():

    raise RuntimeError(
        "Non-finite Logistic-scaled "
        "holdout values detected."
    )


print(
    "Frozen Logistic scaler:",
    "VERIFIED"
)

print(
    "Logistic scaler refit:",
    False
)


# ============================================================
# LOAD FROZEN HOLDOUT MODEL REGISTRY
# ============================================================

registry = pd.read_csv(
    HOLDOUT_MODEL_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    registry
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen "
        "holdout models."
    )


expected_families = {
    "logistic",
    "random_forest",
    "xgboost",
}


if set(
    registry[
        "family"
    ].astype(str)
) != expected_families:

    raise RuntimeError(
        "Holdout model registry family "
        "set is incorrect."
    )


if registry[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate model candidate in "
        "holdout registry."
    )


print(
    "Frozen holdout model registry:",
    "PASS"
)


# ============================================================
# BUILD PREDICTION TABLE
# ============================================================

prediction_data = {

    "holdout_row_id":
        np.arange(
            EXPECTED_ROWS,
            dtype=np.int64,
        ),

    # --------------------------------------------------------
    # Standardized downstream bootstrap column
    # --------------------------------------------------------
    "ENTITY_GROUP":
        holdout[
            ENTITY_GROUP_SOURCE_COLUMN
        ].to_numpy(),

    # --------------------------------------------------------
    # Preserve original source column too
    # --------------------------------------------------------
    "ENTITY_GROUP_ID":
        holdout[
            ENTITY_GROUP_SOURCE_COLUMN
        ].to_numpy(),

    "HARPNUM":
        holdout[
            "HARPNUM"
        ].to_numpy(),

    "target":
        target,
}


# ============================================================
# PRESERVE USEFUL METADATA WHEN AVAILABLE
# ============================================================

for optional_column in [
    "T_REC",
    "observation_time_utc",
    "NOAA_AR",
    "NOAA_ARS",
    "associated_noaa_list",
    "associated_noaa_count",
    "entity_group_source",
]:

    if optional_column in holdout.columns:

        prediction_data[
            optional_column
        ] = holdout[
            optional_column
        ].to_numpy()


prediction_df = pd.DataFrame(
    prediction_data
)


# ============================================================
# GENERATE FROZEN-MODEL PREDICTIONS
# ============================================================

manifest_rows = []


print(
    "\n===================================="
)

print(
    "GENERATING HOLDOUT PROBABILITIES"
)

print(
    "===================================="
)


for _, registry_row in registry.iterrows():

    family = str(
        registry_row[
            "family"
        ]
    )


    candidate_id = str(
        registry_row[
            "candidate_id"
        ]
    )


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
    )


    frozen_threshold = float(
        registry_row[
            "frozen_threshold"
        ]
    )


    # ========================================================
    # VERIFY MODEL FILE
    # ========================================================

    if not model_file.exists():

        raise FileNotFoundError(
            f"{candidate_id}: frozen model "
            f"not found: {model_file}"
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
            f"{candidate_id}: model hash mismatch."
        )


    # ========================================================
    # LOAD FROZEN MODEL
    # ========================================================

    loaded_model = joblib.load(
        model_file
    )


    estimator = extract_estimator(
        loaded_model,
        candidate_id,
    )


    bundle_feature_order = (
        get_model_feature_order(
            loaded_model
        )
    )


    if (
        bundle_feature_order
        is not None
        and
        bundle_feature_order
        !=
        FEATURES
    ):

        raise RuntimeError(
            f"{candidate_id}: model bundle "
            "feature order mismatch."
        )


    # ========================================================
    # SAME REPRESENTATION AS TRAINING / VALIDATION
    # ========================================================

    if family == "logistic":

        model_input = (
            X_logistic
        )

        preprocessing_used = (
            "FROZEN_STANDARD_SCALER_TRANSFORM_ONLY"
        )


    elif family == "random_forest":

        model_input = (
            X_raw_float64
        )

        preprocessing_used = (
            "IDENTITY_RAW_FLOAT64"
        )


    elif family == "xgboost":

        model_input = (
            X_raw_float64.astype(
                np.float32
            )
        )

        preprocessing_used = (
            "IDENTITY_RAW_FLOAT32"
        )


    else:

        raise RuntimeError(
            f"Unknown model family: {family}"
        )


    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    probabilities = estimator.predict_proba(
        model_input
    )[
        :,
        1
    ].astype(
        np.float64
    )


    if len(
        probabilities
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            "probability-vector length."
        )


    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "probabilities detected."
        )


    if (
        probabilities.min() < 0.0
        or
        probabilities.max() > 1.0
    ):

        raise RuntimeError(
            f"{candidate_id}: probability "
            "outside [0,1]."
        )


    # ========================================================
    # APPLY FROZEN VALIDATION-DERIVED THRESHOLD
    # ========================================================

    predicted_class = (
        probabilities
        >=
        frozen_threshold
    ).astype(
        np.int8
    )


    probability_column = (
        f"{candidate_id}"
        "__probability"
    )


    prediction_column = (
        f"{candidate_id}"
        "__prediction"
    )


    prediction_df[
        probability_column
    ] = probabilities


    prediction_df[
        prediction_column
    ] = predicted_class


    # ========================================================
    # MANIFEST ROW
    # ========================================================

    manifest_rows.append(
        {
            "family":
                family,

            "candidate_id":
                candidate_id,

            "model_file":
                str(
                    model_file
                ),

            "model_sha256":
                actual_model_hash,

            "frozen_threshold":
                frozen_threshold,

            "threshold_source":
                "CYCLE24_VALIDATION",

            "probability_column":
                probability_column,

            "prediction_column":
                prediction_column,

            "rows":
                len(
                    probabilities
                ),

            "probability_min":
                float(
                    probabilities.min()
                ),

            "probability_max":
                float(
                    probabilities.max()
                ),

            "probability_mean":
                float(
                    probabilities.mean()
                ),

            "probability_std":
                float(
                    probabilities.std()
                ),

            "predicted_positive_count":
                int(
                    predicted_class.sum()
                ),

            "preprocessing":
                preprocessing_used,

            "model_retrained":
                False,

            "scaler_refit":
                False,

            "threshold_reoptimized":
                False,

            "holdout_metrics_computed":
                False,

            "cycle25_used":
                False,

            "pass":
                True,
        }
    )


    print(
        f"{candidate_id}: "
        f"family={family}, "
        f"rows={len(probabilities)}, "
        f"p_min={probabilities.min():.6f}, "
        f"p_max={probabilities.max():.6f}, "
        f"p_mean={probabilities.mean():.6f}, "
        f"threshold={frozen_threshold:.10g}, "
        f"predicted_positive="
        f"{int(predicted_class.sum())}, "
        f"PASS=True"
    )


# ============================================================
# FINAL TABLE INTEGRITY
# ============================================================

manifest_df = pd.DataFrame(
    manifest_rows
)


if len(
    manifest_df
) != 3:

    raise RuntimeError(
        "Expected 3 holdout prediction "
        "manifest rows."
    )


if manifest_df[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate prediction manifest candidate."
    )


expected_probability_columns = {
    (
        f"{candidate_id}"
        "__probability"
    )
    for candidate_id
    in registry[
        "candidate_id"
    ].astype(str)
}


expected_prediction_columns = {
    (
        f"{candidate_id}"
        "__prediction"
    )
    for candidate_id
    in registry[
        "candidate_id"
    ].astype(str)
}


if not expected_probability_columns.issubset(
    prediction_df.columns
):

    raise RuntimeError(
        "Missing holdout probability columns."
    )


if not expected_prediction_columns.issubset(
    prediction_df.columns
):

    raise RuntimeError(
        "Missing holdout thresholded "
        "prediction columns."
    )


if len(
    prediction_df
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Final holdout prediction table "
        "row count changed."
    )


if prediction_df[
    "ENTITY_GROUP"
].isna().any():

    raise RuntimeError(
        "Standardized ENTITY_GROUP column "
        "contains missing values."
    )


if int(
    prediction_df[
        "ENTITY_GROUP"
    ].nunique()
) != EXPECTED_ENTITY_GROUPS:

    raise RuntimeError(
        "Standardized ENTITY_GROUP count "
        "changed during prediction-table creation."
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_PREDICTIONS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_MANIFEST_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


prediction_df.to_parquet(
    OUTPUT_PREDICTIONS_FILE,
    index=False,
)


manifest_df.to_csv(
    OUTPUT_MANIFEST_FILE,
    index=False,
)


prediction_hash = sha256_file(
    OUTPUT_PREDICTIONS_FILE
)


manifest_hash = sha256_file(
    OUTPUT_MANIFEST_FILE
)


# ============================================================
# ROUND-TRIP VERIFICATION
# ============================================================

roundtrip = pd.read_parquet(
    OUTPUT_PREDICTIONS_FILE
)


if len(
    roundtrip
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Prediction parquet round-trip "
        "row mismatch."
    )


if not np.array_equal(
    roundtrip[
        "target"
    ].to_numpy(
        dtype=np.int64
    ),
    target,
):

    raise RuntimeError(
        "Target changed during parquet "
        "round-trip."
    )


if not np.array_equal(
    roundtrip[
        "ENTITY_GROUP"
    ].to_numpy(),
    prediction_df[
        "ENTITY_GROUP"
    ].to_numpy(),
):

    raise RuntimeError(
        "ENTITY_GROUP changed during "
        "parquet round-trip."
    )


if not np.array_equal(
    roundtrip[
        "ENTITY_GROUP_ID"
    ].to_numpy(),
    prediction_df[
        "ENTITY_GROUP_ID"
    ].to_numpy(),
):

    raise RuntimeError(
        "ENTITY_GROUP_ID changed during "
        "parquet round-trip."
    )


for candidate_id in registry[
    "candidate_id"
].astype(str):

    probability_column = (
        f"{candidate_id}"
        "__probability"
    )

    prediction_column = (
        f"{candidate_id}"
        "__prediction"
    )


    if not np.array_equal(
        roundtrip[
            probability_column
        ].to_numpy(
            dtype=np.float64
        ),
        prediction_df[
            probability_column
        ].to_numpy(
            dtype=np.float64
        ),
    ):

        raise RuntimeError(
            f"{candidate_id}: probability "
            "round-trip mismatch."
        )


    if not np.array_equal(
        roundtrip[
            prediction_column
        ].to_numpy(),
        prediction_df[
            prediction_column
        ].to_numpy(),
    ):

        raise RuntimeError(
            f"{candidate_id}: class prediction "
            "round-trip mismatch."
        )


print(
    "\nPrediction parquet round-trip:",
    "PASS"
)


# ============================================================
# BUILD CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Cycle24 Holdout Frozen-Model Predictions",

    "version":
        "1.0",

    "status":
        "FROZEN_HOLDOUT_PREDICTIONS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source_dataset":
        {
            "file":
                str(
                    HOLDOUT_FILE
                ),

            "sha256":
                actual_holdout_hash,

            "rows":
                EXPECTED_ROWS,

            "positive":
                positive_count,

            "negative":
                negative_count,

            "entity_groups":
                entity_group_count,

            "entity_group_source_column":
                ENTITY_GROUP_SOURCE_COLUMN,
        },

    "feature_count":
        25,

    "feature_order":
        FEATURES,

    "model_count":
        3,

    "models":
        [
            {
                "family":
                    row[
                        "family"
                    ],

                "candidate_id":
                    row[
                        "candidate_id"
                    ],

                "model_sha256":
                    row[
                        "model_sha256"
                    ],

                "frozen_threshold":
                    float(
                        row[
                            "frozen_threshold"
                        ]
                    ),

                "probability_column":
                    row[
                        "probability_column"
                    ],

                "prediction_column":
                    row[
                        "prediction_column"
                    ],
            }

            for _, row
            in manifest_df.iterrows()
        ],

    "prediction_rule":
        "probability >= frozen_threshold",

    "prediction_artifact":
        {
            "file":
                str(
                    OUTPUT_PREDICTIONS_FILE
                ),

            "sha256":
                prediction_hash,

            "rows":
                len(
                    prediction_df
                ),

            "columns":
                len(
                    prediction_df.columns
                ),
        },

    "prediction_manifest":
        {
            "file":
                str(
                    OUTPUT_MANIFEST_FILE
                ),

            "sha256":
                manifest_hash,
        },

    "preserved_metadata":
        {
            "holdout_row_id":
                True,

            "ENTITY_GROUP":
                True,

            "ENTITY_GROUP_ID":
                True,

            "HARPNUM":
                True,

            "target":
                True,

            "T_REC":
                (
                    "T_REC"
                    in
                    prediction_df.columns
                ),

            "observation_time_utc":
                (
                    "observation_time_utc"
                    in
                    prediction_df.columns
                ),

            "NOAA_AR":
                (
                    "NOAA_AR"
                    in
                    prediction_df.columns
                ),

            "NOAA_ARS":
                (
                    "NOAA_ARS"
                    in
                    prediction_df.columns
                ),

            "associated_noaa_list":
                (
                    "associated_noaa_list"
                    in
                    prediction_df.columns
                ),

            "associated_noaa_count":
                (
                    "associated_noaa_count"
                    in
                    prediction_df.columns
                ),

            "entity_group_source":
                (
                    "entity_group_source"
                    in
                    prediction_df.columns
                ),
        },

    "evaluation_state":
        {
            "holdout_opened":
                True,

            "probabilities_generated":
                True,

            "frozen_thresholds_applied":
                True,

            "holdout_metrics_computed":
                False,

            "calibration_metrics_computed":
                False,

            "bootstrap_confidence_intervals_computed":
                False,

            "models_retrained":
                False,

            "logistic_scaler_refit":
                False,

            "thresholds_reoptimized":
                False,

            "family_selection_changed":
                False,
        },

    "data_isolation":
        {
            "cycle24_holdout_read":
                True,

            "cycle25_read":
                False,

            "train_used_for_new_fitting":
                False,

            "validation_used_for_new_tuning":
                False,
        },

    "provenance":
        {
            "holdout_evaluation_protocol":
                {
                    "file":
                        str(
                            HOLDOUT_PROTOCOL_FILE
                        ),

                    "sha256":
                        sha256_file(
                            HOLDOUT_PROTOCOL_FILE
                        ),
                },

            "holdout_model_registry":
                {
                    "file":
                        str(
                            HOLDOUT_MODEL_REGISTRY_FILE
                        ),

                    "sha256":
                        sha256_file(
                            HOLDOUT_MODEL_REGISTRY_FILE
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

            "logistic_scaler":
                {
                    "file":
                        str(
                            LOGISTIC_SCALER_FILE
                        ),

                    "sha256":
                        sha256_file(
                            LOGISTIC_SCALER_FILE
                        ),
                },
        },

    "immutability_rule":
        (
            "These Cycle24 holdout probability predictions "
            "were generated using the already-frozen "
            "TRAIN-only models, preprocessing, and "
            "validation-derived thresholds. Holdout outcomes "
            "must not be used to retrain models, refit "
            "preprocessing, alter features, change candidates, "
            "or reoptimize thresholds."
        ),
}


# ============================================================
# SAVE CONTRACT
# ============================================================

with OUTPUT_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


# ============================================================
# VERIFY SAVED CONTRACT
# ============================================================

saved_contract = load_json(
    OUTPUT_CONTRACT_FILE
)


if saved_contract.get(
    "status"
) != "FROZEN_HOLDOUT_PREDICTIONS":

    raise RuntimeError(
        "Holdout prediction contract "
        "was not frozen."
    )


if int(
    saved_contract[
        "source_dataset"
    ][
        "entity_groups"
    ]
) != EXPECTED_ENTITY_GROUPS:

    raise RuntimeError(
        "Frozen prediction contract "
        "ENTITY_GROUP count mismatch."
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "HOLDOUT PREDICTION SUMMARY"
)

print(
    "===================================="
)


print(
    "Holdout rows:",
    EXPECTED_ROWS
)

print(
    "Holdout positives:",
    positive_count
)

print(
    "Holdout negatives:",
    negative_count
)

print(
    "ENTITY_GROUP source:",
    ENTITY_GROUP_SOURCE_COLUMN
)

print(
    "ENTITY_GROUPs:",
    entity_group_count
)

print(
    "Models evaluated:",
    3
)

print(
    "Probability vectors:",
    3
)

print(
    "Frozen thresholds applied:",
    True
)


# ============================================================
# EVALUATION STATE
# ============================================================

print(
    "\n===================================="
)

print(
    "EVALUATION STATE"
)

print(
    "===================================="
)


print(
    "Holdout opened:",
    True
)

print(
    "Holdout metrics computed:",
    False
)

print(
    "Calibration metrics computed:",
    False
)

print(
    "Bootstrap CIs computed:",
    False
)

print(
    "Models retrained:",
    False
)

print(
    "Thresholds reoptimized:",
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
    "Cycle24 HOLDOUT read:",
    True
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
    "VALIDATION used for new tuning:",
    False
)


# ============================================================
# SAVED OUTPUTS
# ============================================================

print(
    "\nSaved:"
)

print(
    OUTPUT_PREDICTIONS_FILE
)

print(
    OUTPUT_MANIFEST_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 7.2 COMPLETE — HOLDOUT PREDICTIONS FROZEN"
)

print(
    "===================================="
)