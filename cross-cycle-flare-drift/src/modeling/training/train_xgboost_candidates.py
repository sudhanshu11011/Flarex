from pathlib import Path
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import sklearn
import xgboost
from xgboost import XGBClassifier


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

PREPROCESSING_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/preprocessing_contract.json"
)

PROTOCOL_FILE = Path(
    "artifacts/models/xgboost/contracts/"
    "xgboost_training_protocol.json"
)

GRID_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_candidate_grid.csv"
)

MODEL_DIR = Path(
    "artifacts/models/xgboost/models"
)

SUMMARY_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_training_summary.csv"
)


# ============================================================
# SERIALIZATION VERIFICATION
# ============================================================

SERIALIZATION_PROBABILITY_ATOL = 1e-12


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
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 5.2 — TRAIN XGBOOST CANDIDATES"
)

print(
    "===================================="
)


# ============================================================
# VERIFY REQUIRED INPUTS
# ============================================================

required_files = [
    TRAIN_FILE,
    SPLIT_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
    PROTOCOL_FILE,
    GRID_FILE,
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


with PREPROCESSING_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    preprocessing_contract = json.load(file)


with PROTOCOL_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    protocol = json.load(file)


# ============================================================
# CONTRACT STATES
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


if protocol.get(
    "status"
) != "FROZEN_BEFORE_TRAINING":

    raise RuntimeError(
        "XGBoost training protocol is not "
        "FROZEN_BEFORE_TRAINING."
    )


if protocol.get(
    "model_family"
) != "XGBClassifier":

    raise RuntimeError(
        "Unexpected model family in "
        "XGBoost protocol."
    )


features = feature_contract.get(
    "feature_order",
    []
)


if len(features) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


if protocol.get(
    "feature_order"
) != features:

    raise RuntimeError(
        "XGBoost protocol feature order "
        "does not match frozen feature contract."
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
    "XGBoost training protocol: FROZEN"
)


# ============================================================
# PREPROCESSING DESIGN
# ============================================================

xgb_preprocessing = protocol.get(
    "preprocessing",
    {}
)


if xgb_preprocessing.get(
    "method"
) != "IDENTITY_RAW_FEATURES":

    raise RuntimeError(
        "XGBoost protocol does not specify "
        "IDENTITY_RAW_FEATURES."
    )


if bool(
    xgb_preprocessing.get(
        "scaling"
    )
):

    raise RuntimeError(
        "XGBoost protocol unexpectedly "
        "requests scaling."
    )


print(
    "XGBoost preprocessing: RAW / IDENTITY"
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
        "TRAIN dataset differs from "
        "frozen split contract."
    )


print(
    "TRAIN hash: PASS"
)


# ============================================================
# LOAD TRAIN ONLY
# ============================================================

train = pd.read_parquet(
    TRAIN_FILE
)


missing_features = [

    feature

    for feature in features

    if feature not in train.columns
]


if missing_features:

    raise RuntimeError(
        f"TRAIN missing frozen features: "
        f"{missing_features}"
    )


if "target" not in train.columns:

    raise RuntimeError(
        "TRAIN dataset missing target column."
    )


X_train_df = (
    train[
        features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
)


if X_train_df.isna().any().any():

    raise RuntimeError(
        "TRAIN contains missing/non-numeric "
        "feature values."
    )


X_train = X_train_df.to_numpy(
    dtype=np.float32
)


y_train = pd.to_numeric(
    train[
        "target"
    ],
    errors="raise",
).to_numpy(
    dtype=np.int32
)


# ============================================================
# VERIFY TRAIN DATA
# ============================================================

if X_train.shape != (
    62064,
    25,
):

    raise RuntimeError(
        f"Unexpected TRAIN shape: "
        f"{X_train.shape}"
    )


if len(y_train) != 62064:

    raise RuntimeError(
        "Unexpected TRAIN target length."
    )


if not np.isfinite(
    X_train
).all():

    raise RuntimeError(
        "TRAIN contains non-finite features."
    )


unique_targets = set(
    np.unique(
        y_train
    ).tolist()
)


if unique_targets != {
    0,
    1,
}:

    raise RuntimeError(
        f"TRAIN target is not binary {{0,1}}. "
        f"Found: {unique_targets}"
    )


train_positive = int(
    y_train.sum()
)


train_negative = int(
    (
        y_train
        ==
        0
    ).sum()
)


if train_positive != 2420:

    raise RuntimeError(
        "Unexpected TRAIN positive count."
    )


if train_negative != 59644:

    raise RuntimeError(
        "Unexpected TRAIN negative count."
    )


train_rows = len(
    y_train
)


balanced_scale_pos_weight = (
    train_negative
    /
    train_positive
)


protocol_ratio = float(
    protocol[
        "train_class_distribution"
    ][
        "negative_to_positive_ratio"
    ]
)


if not np.isclose(
    balanced_scale_pos_weight,
    protocol_ratio,
    rtol=0.0,
    atol=1e-12,
):

    raise RuntimeError(
        "TRAIN class ratio differs from "
        "frozen XGBoost protocol."
    )


print(
    "TRAIN rows:",
    train_rows
)

print(
    "TRAIN positives:",
    train_positive
)

print(
    "TRAIN negatives:",
    train_negative
)

print(
    "Features:",
    X_train.shape[1]
)

print(
    "Balanced scale_pos_weight:",
    balanced_scale_pos_weight
)

print(
    "Scaling applied:",
    False
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
# LOAD FROZEN GRID
# ============================================================

grid = pd.read_csv(
    GRID_FILE,
    keep_default_na=False,
)


if len(grid) != 12:

    raise RuntimeError(
        "Expected exactly 12 XGBoost candidates."
    )


if grid[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate XGBoost candidate IDs."
    )


expected_candidate_count = int(
    protocol[
        "candidate_search_space"
    ][
        "candidate_count"
    ]
)


if expected_candidate_count != 12:

    raise RuntimeError(
        "Frozen protocol candidate count "
        "is not 12."
    )


# ============================================================
# FIXED PROTOCOL PARAMETERS
# ============================================================

fixed = protocol[
    "model_fixed_parameters"
]


EXPECTED_N_ESTIMATORS = int(
    fixed[
        "n_estimators"
    ]
)

EXPECTED_BOOSTER = str(
    fixed[
        "booster"
    ]
)

EXPECTED_OBJECTIVE = str(
    fixed[
        "objective"
    ]
)

EXPECTED_EVAL_METRIC = str(
    fixed[
        "eval_metric"
    ]
)

EXPECTED_TREE_METHOD = str(
    fixed[
        "tree_method"
    ]
)

EXPECTED_MIN_CHILD_WEIGHT = float(
    fixed[
        "min_child_weight"
    ]
)

EXPECTED_GAMMA = float(
    fixed[
        "gamma"
    ]
)

EXPECTED_SUBSAMPLE = float(
    fixed[
        "subsample"
    ]
)

EXPECTED_COLSAMPLE_BYTREE = float(
    fixed[
        "colsample_bytree"
    ]
)

EXPECTED_REG_ALPHA = float(
    fixed[
        "reg_alpha"
    ]
)

EXPECTED_REG_LAMBDA = float(
    fixed[
        "reg_lambda"
    ]
)

EXPECTED_MAX_DELTA_STEP = float(
    fixed[
        "max_delta_step"
    ]
)

EXPECTED_RANDOM_STATE = int(
    fixed[
        "random_state"
    ]
)

EXPECTED_N_JOBS = int(
    fixed[
        "n_jobs"
    ]
)

EXPECTED_VERBOSITY = int(
    fixed[
        "verbosity"
    ]
)


# ============================================================
# SEARCH SPACE
# ============================================================

EXPECTED_LEARNING_RATES = {
    0.03,
    0.10,
}


EXPECTED_MAX_DEPTHS = {
    3,
    6,
    9,
}


EXPECTED_SCALE_POS_WEIGHTS = {
    1.0,
    balanced_scale_pos_weight,
}


# ============================================================
# HASHES
# ============================================================

protocol_hash = sha256_file(
    PROTOCOL_FILE
)


candidate_grid_hash = sha256_file(
    GRID_FILE
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


training_rows = []


print(
    "\n===================================="
)

print(
    "TRAINING XGBOOST CANDIDATES"
)

print(
    "===================================="
)


# ============================================================
# TRAIN CANDIDATES
# ============================================================

for _, row in grid.iterrows():

    candidate_id = str(
        row[
            "candidate_id"
        ]
    ).strip()


    # --------------------------------------------------------
    # READ PARAMETERS
    # --------------------------------------------------------

    n_estimators = int(
        row[
            "n_estimators"
        ]
    )


    learning_rate = float(
        row[
            "learning_rate"
        ]
    )


    max_depth = int(
        row[
            "max_depth"
        ]
    )


    min_child_weight = float(
        row[
            "min_child_weight"
        ]
    )


    gamma = float(
        row[
            "gamma"
        ]
    )


    subsample = float(
        row[
            "subsample"
        ]
    )


    colsample_bytree = float(
        row[
            "colsample_bytree"
        ]
    )


    reg_alpha = float(
        row[
            "reg_alpha"
        ]
    )


    reg_lambda = float(
        row[
            "reg_lambda"
        ]
    )


    scale_pos_weight = float(
        row[
            "scale_pos_weight"
        ]
    )


    max_delta_step = float(
        row[
            "max_delta_step"
        ]
    )


    booster = str(
        row[
            "booster"
        ]
    ).strip()


    objective = str(
        row[
            "objective"
        ]
    ).strip()


    eval_metric = str(
        row[
            "eval_metric"
        ]
    ).strip()


    tree_method = str(
        row[
            "tree_method"
        ]
    ).strip()


    random_state = int(
        row[
            "random_state"
        ]
    )


    n_jobs = int(
        row[
            "n_jobs"
        ]
    )


    verbosity = int(
        row[
            "verbosity"
        ]
    )


    # ========================================================
    # FIXED PARAMETER CONFORMANCE
    # ========================================================

    if n_estimators != EXPECTED_N_ESTIMATORS:

        raise RuntimeError(
            f"{candidate_id}: "
            "n_estimators mismatch."
        )


    if booster != EXPECTED_BOOSTER:

        raise RuntimeError(
            f"{candidate_id}: booster mismatch."
        )


    if objective != EXPECTED_OBJECTIVE:

        raise RuntimeError(
            f"{candidate_id}: objective mismatch."
        )


    if eval_metric != EXPECTED_EVAL_METRIC:

        raise RuntimeError(
            f"{candidate_id}: eval_metric mismatch."
        )


    if tree_method != EXPECTED_TREE_METHOD:

        raise RuntimeError(
            f"{candidate_id}: tree_method mismatch."
        )


    if not np.isclose(
        min_child_weight,
        EXPECTED_MIN_CHILD_WEIGHT,
        rtol=0.0,
        atol=0.0,
    ):

        raise RuntimeError(
            f"{candidate_id}: "
            "min_child_weight mismatch."
        )


    if not np.isclose(
        gamma,
        EXPECTED_GAMMA,
        rtol=0.0,
        atol=0.0,
    ):

        raise RuntimeError(
            f"{candidate_id}: gamma mismatch."
        )


    if not np.isclose(
        subsample,
        EXPECTED_SUBSAMPLE,
        rtol=0.0,
        atol=0.0,
    ):

        raise RuntimeError(
            f"{candidate_id}: subsample mismatch."
        )


    if not np.isclose(
        colsample_bytree,
        EXPECTED_COLSAMPLE_BYTREE,
        rtol=0.0,
        atol=0.0,
    ):

        raise RuntimeError(
            f"{candidate_id}: "
            "colsample_bytree mismatch."
        )


    if not np.isclose(
        reg_alpha,
        EXPECTED_REG_ALPHA,
        rtol=0.0,
        atol=0.0,
    ):

        raise RuntimeError(
            f"{candidate_id}: reg_alpha mismatch."
        )


    if not np.isclose(
        reg_lambda,
        EXPECTED_REG_LAMBDA,
        rtol=0.0,
        atol=0.0,
    ):

        raise RuntimeError(
            f"{candidate_id}: reg_lambda mismatch."
        )


    if not np.isclose(
        max_delta_step,
        EXPECTED_MAX_DELTA_STEP,
        rtol=0.0,
        atol=0.0,
    ):

        raise RuntimeError(
            f"{candidate_id}: "
            "max_delta_step mismatch."
        )


    if random_state != EXPECTED_RANDOM_STATE:

        raise RuntimeError(
            f"{candidate_id}: random_state mismatch."
        )


    if n_jobs != EXPECTED_N_JOBS:

        raise RuntimeError(
            f"{candidate_id}: n_jobs mismatch."
        )


    if verbosity != EXPECTED_VERBOSITY:

        raise RuntimeError(
            f"{candidate_id}: verbosity mismatch."
        )


    # ========================================================
    # SEARCH-SPACE CONFORMANCE
    # ========================================================

    if learning_rate not in EXPECTED_LEARNING_RATES:

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            f"learning_rate={learning_rate}"
        )


    if max_depth not in EXPECTED_MAX_DEPTHS:

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            f"max_depth={max_depth}"
        )


    if not any(
        np.isclose(
            scale_pos_weight,
            expected,
            rtol=0.0,
            atol=1e-12,
        )

        for expected
        in EXPECTED_SCALE_POS_WEIGHTS
    ):

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            f"scale_pos_weight="
            f"{scale_pos_weight}"
        )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        max_delta_step=max_delta_step,
        booster=booster,
        objective=objective,
        eval_metric=eval_metric,
        tree_method=tree_method,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=verbosity,
    )


    # ========================================================
    # TRAIN
    #
    # IMPORTANT:
    # No eval_set.
    # No early stopping.
    # No validation data.
    # ========================================================

    with warnings.catch_warnings(
        record=True
    ) as captured_warnings:

        warnings.simplefilter(
            "always"
        )


        model.fit(
            X_train,
            y_train,
        )


    warning_count = len(
        captured_warnings
    )


    warning_text = " | ".join(
        [
            (
                f"{warning.category.__name__}: "
                f"{warning.message}"
            )

            for warning
            in captured_warnings
        ]
    )


    # ========================================================
    # BASIC FITTED-MODEL INTEGRITY
    # ========================================================

    if not np.array_equal(
        model.classes_,
        np.array(
            [0, 1]
        ),
    ):

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            f"classes {model.classes_}"
        )


    if int(
        model.n_features_in_
    ) != 25:

        raise RuntimeError(
            f"{candidate_id}: fitted "
            "feature count is not 25."
        )


    booster_object = model.get_booster()


    if booster_object is None:

        raise RuntimeError(
            f"{candidate_id}: XGBoost booster "
            "object missing."
        )


    boosted_rounds = int(
        booster_object.num_boosted_rounds()
    )


    if boosted_rounds != n_estimators:

        raise RuntimeError(
            f"{candidate_id}: expected "
            f"{n_estimators} boosted rounds "
            f"but found {boosted_rounds}."
        )


    # ========================================================
    # FEATURE IMPORTANCE SANITY
    #
    # Diagnostic only.
    # No feature selection.
    # ========================================================

    feature_importances = np.asarray(
        model.feature_importances_,
        dtype=np.float64,
    )


    if feature_importances.shape != (
        25,
    ):

        raise RuntimeError(
            f"{candidate_id}: expected "
            "25 feature importances."
        )


    if not np.isfinite(
        feature_importances
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "feature importances."
        )


    importance_sum = float(
        feature_importances.sum()
    )


    # XGBoost normally normalizes sklearn feature_importances_
    # to approximately 1.0 when splits exist.
    importance_sum_valid = bool(
        importance_sum > 0.0
        and
        np.isfinite(
            importance_sum
        )
    )


    if not importance_sum_valid:

        raise RuntimeError(
            f"{candidate_id}: invalid feature "
            f"importance sum={importance_sum}"
        )


    # ========================================================
    # TRAIN PROBABILITY SANITY
    #
    # Diagnostic only.
    # No TSS.
    # No threshold.
    # No model selection.
    # ========================================================

    train_probabilities = (
        model.predict_proba(
            X_train
        )[
            :,
            1
        ]
    )


    if not np.isfinite(
        train_probabilities
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "TRAIN probabilities."
        )


    probability_min = float(
        train_probabilities.min()
    )


    probability_max = float(
        train_probabilities.max()
    )


    if (
        probability_min < 0.0
        or
        probability_max > 1.0
    ):

        raise RuntimeError(
            f"{candidate_id}: probability "
            "outside [0,1]."
        )


    # ========================================================
    # SAVE MODEL BUNDLE
    # ========================================================

    model_file = (
        MODEL_DIR
        /
        f"{candidate_id}.joblib"
    )


    model_bundle = {

        "candidate_id":
            candidate_id,

        "model_family":
            "XGBClassifier",

        "model":
            model,

        "feature_order":
            features,

        "feature_count":
            25,

        "preprocessing":
            "IDENTITY_RAW_FEATURES",

        "training_scope":
            "CYCLE24_TRAIN_ONLY",

        "train_sha256":
            actual_train_hash,

        "protocol_sha256":
            protocol_hash,

        "candidate_grid_sha256":
            candidate_grid_hash,

        "n_estimators":
            n_estimators,

        "learning_rate":
            learning_rate,

        "max_depth":
            max_depth,

        "min_child_weight":
            min_child_weight,

        "gamma":
            gamma,

        "subsample":
            subsample,

        "colsample_bytree":
            colsample_bytree,

        "reg_alpha":
            reg_alpha,

        "reg_lambda":
            reg_lambda,

        "scale_pos_weight":
            scale_pos_weight,

        "max_delta_step":
            max_delta_step,

        "booster":
            booster,

        "objective":
            objective,

        "eval_metric":
            eval_metric,

        "tree_method":
            tree_method,

        "random_state":
            random_state,

        "n_jobs":
            n_jobs,

        "verbosity":
            verbosity,

        "boosted_rounds":
            boosted_rounds,

        "early_stopping_used":
            False,

        "validation_used":
            False,

        "sklearn_version":
            sklearn.__version__,

        "xgboost_version":
            xgboost.__version__,
    }


    joblib.dump(
        model_bundle,
        model_file,
    )


    # ========================================================
    # RELOAD ARTIFACT
    # ========================================================

    reloaded_bundle = joblib.load(
        model_file
    )


    if (
        reloaded_bundle.get(
            "candidate_id"
        )
        !=
        candidate_id
    ):

        raise RuntimeError(
            f"{candidate_id}: reloaded "
            "candidate ID mismatch."
        )


    if (
        reloaded_bundle.get(
            "feature_order"
        )
        !=
        features
    ):

        raise RuntimeError(
            f"{candidate_id}: reloaded "
            "feature order mismatch."
        )


    if (
        reloaded_bundle.get(
            "train_sha256"
        )
        !=
        actual_train_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: reloaded "
            "TRAIN hash mismatch."
        )


    if (
        reloaded_bundle.get(
            "protocol_sha256"
        )
        !=
        protocol_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: protocol "
            "hash mismatch after reload."
        )


    if (
        reloaded_bundle.get(
            "candidate_grid_sha256"
        )
        !=
        candidate_grid_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: candidate-grid "
            "hash mismatch after reload."
        )


    reloaded_model = reloaded_bundle.get(
        "model"
    )


    if reloaded_model is None:

        raise RuntimeError(
            f"{candidate_id}: reloaded model missing."
        )


    reloaded_boosted_rounds = int(
        reloaded_model
        .get_booster()
        .num_boosted_rounds()
    )


    if (
        reloaded_boosted_rounds
        !=
        n_estimators
    ):

        raise RuntimeError(
            f"{candidate_id}: reloaded boosted "
            "round count mismatch."
        )


    # ========================================================
    # SERIALIZATION PROBABILITY CHECK
    # ========================================================

    reloaded_probabilities = (
        reloaded_model.predict_proba(
            X_train
        )[
            :,
            1
        ]
    )


    if not np.isfinite(
        reloaded_probabilities
    ).all():

        raise RuntimeError(
            f"{candidate_id}: reloaded model "
            "produced non-finite probabilities."
        )


    serialization_probability_max_abs_diff = float(
        np.max(
            np.abs(
                train_probabilities
                -
                reloaded_probabilities
            )
        )
    )


    serialization_probability_match = bool(
        np.allclose(
            train_probabilities,
            reloaded_probabilities,
            rtol=0.0,
            atol=SERIALIZATION_PROBABILITY_ATOL,
        )
    )


    if not serialization_probability_match:

        raise RuntimeError(
            f"{candidate_id}: serialized model "
            "probabilities differ beyond tolerance. "
            f"max_abs_diff="
            f"{serialization_probability_max_abs_diff:.3e}, "
            f"atol="
            f"{SERIALIZATION_PROBABILITY_ATOL:.1e}"
        )


    # ========================================================
    # ARTIFACT HASH
    # ========================================================

    model_hash = sha256_file(
        model_file
    )


    # ========================================================
    # RECORD
    # ========================================================

    training_rows.append(
        {
            "candidate_id":
                candidate_id,

            "n_estimators":
                n_estimators,

            "learning_rate":
                learning_rate,

            "max_depth":
                max_depth,

            "min_child_weight":
                min_child_weight,

            "gamma":
                gamma,

            "subsample":
                subsample,

            "colsample_bytree":
                colsample_bytree,

            "reg_alpha":
                reg_alpha,

            "reg_lambda":
                reg_lambda,

            "scale_pos_weight":
                scale_pos_weight,

            "max_delta_step":
                max_delta_step,

            "booster":
                booster,

            "objective":
                objective,

            "eval_metric":
                eval_metric,

            "tree_method":
                tree_method,

            "random_state":
                random_state,

            "n_jobs":
                n_jobs,

            "verbosity":
                verbosity,

            "train_rows":
                train_rows,

            "train_positive":
                train_positive,

            "train_negative":
                train_negative,

            "feature_count":
                25,

            "boosted_rounds":
                boosted_rounds,

            "warning_count":
                warning_count,

            "warning_text":
                warning_text,

            "feature_importance_sum":
                importance_sum,

            "feature_importances_finite":
                bool(
                    np.isfinite(
                        feature_importances
                    ).all()
                ),

            "probability_min":
                probability_min,

            "probability_max":
                probability_max,

            "probabilities_finite":
                bool(
                    np.isfinite(
                        train_probabilities
                    ).all()
                ),

            "serialization_probability_atol":
                SERIALIZATION_PROBABILITY_ATOL,

            "serialization_probability_max_abs_diff":
                serialization_probability_max_abs_diff,

            "serialization_probability_match":
                serialization_probability_match,

            "early_stopping_used":
                False,

            "validation_used":
                False,

            "holdout_used":
                False,

            "cycle25_used":
                False,

            "model_selection_performed":
                False,

            "threshold_selection_performed":
                False,

            "model_file":
                str(
                    model_file
                ),

            "model_sha256":
                model_hash,

            "train_sha256":
                actual_train_hash,

            "protocol_sha256":
                protocol_hash,

            "candidate_grid_sha256":
                candidate_grid_hash,

            "sklearn_version":
                sklearn.__version__,

            "xgboost_version":
                xgboost.__version__,
        }
    )


    print(
        f"{candidate_id}: "
        f"rounds={boosted_rounds}, "
        f"LR={learning_rate}, "
        f"depth={max_depth}, "
        f"SPW={scale_pos_weight:.6f}, "
        f"reload_prob_diff="
        f"{serialization_probability_max_abs_diff:.3e}, "
        f"warnings={warning_count}, "
        f"saved=YES"
    )


# ============================================================
# BUILD SUMMARY
# ============================================================

summary = pd.DataFrame(
    training_rows
)


if len(summary) != 12:

    raise RuntimeError(
        "XGBoost training summary does "
        "not contain 12 candidates."
    )


if summary[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "XGBoost training summary contains "
        "duplicate candidate IDs."
    )


if not summary[
    "boosted_rounds"
].eq(
        EXPECTED_N_ESTIMATORS
    ).all():

    raise RuntimeError(
        "One or more XGBoost candidates "
        "do not contain 500 boosted rounds."
    )


if not summary[
    "feature_importances_finite"
].all():

    raise RuntimeError(
        "One or more XGBoost candidates "
        "contain non-finite feature importances."
    )


if not summary[
    "probabilities_finite"
].all():

    raise RuntimeError(
        "One or more XGBoost candidates "
        "contain non-finite probabilities."
    )


if not summary[
    "serialization_probability_match"
].all():

    raise RuntimeError(
        "One or more XGBoost serialized "
        "artifacts failed probability verification."
    )


# ============================================================
# FINAL COUNTS
# ============================================================

candidate_count = len(
    summary
)


successful_candidates = int(
    summary[
        "boosted_rounds"
    ].eq(
        EXPECTED_N_ESTIMATORS
    ).sum()
)


total_boosted_rounds = int(
    summary[
        "boosted_rounds"
    ].sum()
)


total_warnings = int(
    summary[
        "warning_count"
    ].sum()
)


serialization_match_count = int(
    summary[
        "serialization_probability_match"
    ].sum()
)


max_serialization_probability_difference = float(
    summary[
        "serialization_probability_max_abs_diff"
    ].max()
)


# ============================================================
# PROTOCOL CONFORMANCE
# ============================================================

protocol_conformance = bool(

    candidate_count == 12

    and
    summary[
        "n_estimators"
    ].eq(
        EXPECTED_N_ESTIMATORS
    ).all()

    and
    summary[
        "booster"
    ].eq(
        EXPECTED_BOOSTER
    ).all()

    and
    summary[
        "objective"
    ].eq(
        EXPECTED_OBJECTIVE
    ).all()

    and
    summary[
        "eval_metric"
    ].eq(
        EXPECTED_EVAL_METRIC
    ).all()

    and
    summary[
        "tree_method"
    ].eq(
        EXPECTED_TREE_METHOD
    ).all()

    and
    summary[
        "random_state"
    ].eq(
        EXPECTED_RANDOM_STATE
    ).all()

    and
    summary[
        "n_jobs"
    ].eq(
        EXPECTED_N_JOBS
    ).all()

    and
    summary[
        "early_stopping_used"
    ].eq(
        False
    ).all()

    and
    summary[
        "boosted_rounds"
    ].eq(
        EXPECTED_N_ESTIMATORS
    ).all()

    and
    summary[
        "serialization_probability_match"
    ].all()
)


# ============================================================
# SAVE
# ============================================================

SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "TRAINING SUMMARY"
)

print(
    "===================================="
)


print(
    summary[
        [
            "candidate_id",
            "learning_rate",
            "max_depth",
            "scale_pos_weight",
            "boosted_rounds",
            "serialization_probability_max_abs_diff",
            "warning_count",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nCandidates trained:",
    candidate_count
)

print(
    "Candidates with 500 boosted rounds:",
    successful_candidates
)

print(
    "Total boosted rounds:",
    total_boosted_rounds
)

print(
    "Total training warnings:",
    total_warnings
)

print(
    "Serialization probability matches:",
    serialization_match_count,
    "/ 12"
)

print(
    "Maximum serialization probability difference:",
    max_serialization_probability_difference
)

print(
    "Serialization probability tolerance:",
    SERIALIZATION_PROBABILITY_ATOL
)


# ============================================================
# PROTOCOL CONFORMANCE
# ============================================================

print(
    "\n===================================="
)

print(
    "PROTOCOL CONFORMANCE"
)

print(
    "===================================="
)


print(
    "Frozen candidate count:",
    12
)

print(
    "n_estimators:",
    EXPECTED_N_ESTIMATORS
)

print(
    "booster:",
    EXPECTED_BOOSTER
)

print(
    "objective:",
    EXPECTED_OBJECTIVE
)

print(
    "tree_method:",
    EXPECTED_TREE_METHOD
)

print(
    "random_state:",
    EXPECTED_RANDOM_STATE
)

print(
    "Early stopping used:",
    False
)

print(
    "All candidates match frozen protocol:",
    protocol_conformance
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
    "Model selection performed:",
    False
)

print(
    "Threshold selection performed:",
    False
)


# ============================================================
# SOFTWARE PROVENANCE
# ============================================================

print(
    "\n===================================="
)

print(
    "SOFTWARE PROVENANCE"
)

print(
    "===================================="
)


print(
    "scikit-learn:",
    sklearn.__version__
)

print(
    "xgboost:",
    xgboost.__version__
)


# ============================================================
# FINAL GATE
# ============================================================

chapter_pass = bool(

    candidate_count == 12

    and
    successful_candidates == 12

    and
    total_warnings == 0

    and
    serialization_match_count == 12

    and
    protocol_conformance
)


print(
    "\n===================================="
)

print(
    "CHAPTER 5.2 RESULT"
)

print(
    "===================================="
)


print(
    "12 candidates trained:",
    candidate_count == 12
)

print(
    "12 candidates contain 500 boosted rounds:",
    successful_candidates == 12
)

print(
    "Training warnings = 0:",
    total_warnings == 0
)

print(
    "Serialization verification:",
    serialization_match_count == 12
)

print(
    "Protocol conformance:",
    protocol_conformance
)

print(
    "Chapter 5.2:",
    (
        "PASS"
        if chapter_pass
        else
        "FAIL"
    )
)


print(
    "\nSaved:"
)

print(
    SUMMARY_FILE
)

print(
    MODEL_DIR
)


if not chapter_pass:

    raise RuntimeError(
        "Chapter 5.2 XGBoost "
        "candidate training failed."
    )


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 5.2 COMPLETE"
)

print(
    "===================================="
)