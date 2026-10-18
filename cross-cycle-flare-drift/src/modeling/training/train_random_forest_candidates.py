from pathlib import Path
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestClassifier


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
    "artifacts/models/random_forest/contracts/"
    "random_forest_training_protocol.json"
)

GRID_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_candidate_grid.csv"
)

MODEL_DIR = Path(
    "artifacts/models/random_forest/models"
)

SUMMARY_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_training_summary.csv"
)


# ============================================================
# SERIALIZATION VERIFICATION TOLERANCE
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


def parse_bool(value):

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text == "true":
        return True

    if text == "false":
        return False

    raise RuntimeError(
        f"Cannot parse boolean value: {value}"
    )


def parse_optional_int(value):

    text = str(value).strip()

    if text.lower() == "none":
        return None

    return int(
        float(text)
    )


def parse_optional_float(value):

    text = str(value).strip()

    if text.lower() == "none":
        return None

    return float(text)


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 4.2 — TRAIN RANDOM FOREST CANDIDATES"
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


if protocol.get(
    "status"
) != "FROZEN_BEFORE_TRAINING":

    raise RuntimeError(
        "Random Forest training protocol "
        "is not FROZEN_BEFORE_TRAINING."
    )


if protocol.get(
    "model_family"
) != "RandomForestClassifier":

    raise RuntimeError(
        "Unexpected model family in RF protocol."
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
        "RF protocol feature order differs "
        "from frozen feature contract."
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
    "RF training protocol: FROZEN"
)


# ============================================================
# VERIFY RF PREPROCESSING DESIGN
# ============================================================

rf_preprocessing = protocol.get(
    "preprocessing",
    {}
)


if rf_preprocessing.get(
    "method"
) != "IDENTITY_RAW_FEATURES":

    raise RuntimeError(
        "RF protocol does not specify "
        "raw / identity preprocessing."
    )


if bool(
    rf_preprocessing.get(
        "scaling"
    )
):

    raise RuntimeError(
        "RF protocol unexpectedly requests scaling."
    )


print(
    "RF preprocessing: RAW / IDENTITY"
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
    dtype=np.float64
)


y_train = pd.to_numeric(
    train[
        "target"
    ],
    errors="raise",
).to_numpy(
    dtype=np.int64
)


# ============================================================
# VERIFY TRAIN STATE
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


print(
    "TRAIN rows:",
    len(y_train)
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
# LOAD FROZEN RF GRID
# ============================================================

grid = pd.read_csv(
    GRID_FILE,
    keep_default_na=False,
)


if len(grid) != 12:

    raise RuntimeError(
        "Expected exactly 12 RF candidates."
    )


if grid[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate RF candidate IDs."
    )


expected_candidate_count = int(
    protocol[
        "candidate_search_space"
    ][
        "candidate_count"
    ]
)


if (
    expected_candidate_count
    !=
    len(grid)
):

    raise RuntimeError(
        "RF candidate grid count differs "
        "from frozen protocol."
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

EXPECTED_CRITERION = str(
    fixed[
        "criterion"
    ]
)

EXPECTED_MIN_SAMPLES_SPLIT = int(
    fixed[
        "min_samples_split"
    ]
)

EXPECTED_MAX_FEATURES = str(
    fixed[
        "max_features"
    ]
)

EXPECTED_BOOTSTRAP = bool(
    fixed[
        "bootstrap"
    ]
)

EXPECTED_MAX_SAMPLES = fixed[
    "max_samples"
]

EXPECTED_OOB_SCORE = bool(
    fixed[
        "oob_score"
    ]
)

EXPECTED_WARM_START = bool(
    fixed[
        "warm_start"
    ]
)

EXPECTED_CCP_ALPHA = float(
    fixed[
        "ccp_alpha"
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


# ============================================================
# EXPECTED SEARCH SPACE
# ============================================================

EXPECTED_MAX_DEPTH_VALUES = {
    12,
    24,
    None,
}

EXPECTED_MIN_SAMPLES_LEAF_VALUES = {
    1,
    5,
}

EXPECTED_CLASS_WEIGHT_VALUES = {
    None,
    "balanced_subsample",
}


# ============================================================
# VERIFY FROZEN SEARCH SPACE
# ============================================================

protocol_max_depth = set(
    protocol[
        "candidate_search_space"
    ][
        "max_depth"
    ]
)


protocol_min_samples_leaf = set(
    protocol[
        "candidate_search_space"
    ][
        "min_samples_leaf"
    ]
)


protocol_class_weight = set(
    protocol[
        "candidate_search_space"
    ][
        "class_weight"
    ]
)


if (
    protocol_max_depth
    !=
    EXPECTED_MAX_DEPTH_VALUES
):

    raise RuntimeError(
        "Frozen max_depth search space differs "
        "from expected RF design."
    )


if (
    protocol_min_samples_leaf
    !=
    EXPECTED_MIN_SAMPLES_LEAF_VALUES
):

    raise RuntimeError(
        "Frozen min_samples_leaf search space differs "
        "from expected RF design."
    )


if (
    protocol_class_weight
    !=
    EXPECTED_CLASS_WEIGHT_VALUES
):

    raise RuntimeError(
        "Frozen class_weight search space differs "
        "from expected RF design."
    )


# ============================================================
# HASHES USED FOR ALL CANDIDATES
# ============================================================

protocol_hash = sha256_file(
    PROTOCOL_FILE
)

candidate_grid_hash = sha256_file(
    GRID_FILE
)


# ============================================================
# MODEL OUTPUT DIRECTORY
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
    "TRAINING RANDOM FOREST CANDIDATES"
)

print(
    "===================================="
)


# ============================================================
# TRAIN EACH CANDIDATE
# ============================================================

for _, row in grid.iterrows():

    candidate_id = str(
        row[
            "candidate_id"
        ]
    ).strip()


    # --------------------------------------------------------
    # READ CANDIDATE PARAMETERS
    # --------------------------------------------------------

    n_estimators = int(
        row[
            "n_estimators"
        ]
    )


    criterion = str(
        row[
            "criterion"
        ]
    ).strip()


    max_depth = parse_optional_int(
        row[
            "max_depth"
        ]
    )


    min_samples_split = int(
        row[
            "min_samples_split"
        ]
    )


    min_samples_leaf = int(
        row[
            "min_samples_leaf"
        ]
    )


    max_features = str(
        row[
            "max_features"
        ]
    ).strip()


    bootstrap = parse_bool(
        row[
            "bootstrap"
        ]
    )


    max_samples = parse_optional_float(
        row[
            "max_samples"
        ]
    )


    class_weight_text = str(
        row[
            "class_weight"
        ]
    ).strip().lower()


    if class_weight_text == "none":

        class_weight = None

    elif class_weight_text == "balanced_subsample":

        class_weight = "balanced_subsample"

    else:

        raise RuntimeError(
            f"{candidate_id}: unsupported "
            f"class_weight {class_weight_text}"
        )


    oob_score = parse_bool(
        row[
            "oob_score"
        ]
    )


    warm_start = parse_bool(
        row[
            "warm_start"
        ]
    )


    ccp_alpha = float(
        row[
            "ccp_alpha"
        ]
    )


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


    # ========================================================
    # VERIFY FIXED PARAMETERS
    # ========================================================

    if n_estimators != EXPECTED_N_ESTIMATORS:

        raise RuntimeError(
            f"{candidate_id}: n_estimators mismatch."
        )


    if criterion != EXPECTED_CRITERION:

        raise RuntimeError(
            f"{candidate_id}: criterion mismatch."
        )


    if (
        min_samples_split
        !=
        EXPECTED_MIN_SAMPLES_SPLIT
    ):

        raise RuntimeError(
            f"{candidate_id}: "
            "min_samples_split mismatch."
        )


    if max_features != EXPECTED_MAX_FEATURES:

        raise RuntimeError(
            f"{candidate_id}: max_features mismatch."
        )


    if bootstrap != EXPECTED_BOOTSTRAP:

        raise RuntimeError(
            f"{candidate_id}: bootstrap mismatch."
        )


    if max_samples != EXPECTED_MAX_SAMPLES:

        raise RuntimeError(
            f"{candidate_id}: max_samples mismatch."
        )


    if oob_score != EXPECTED_OOB_SCORE:

        raise RuntimeError(
            f"{candidate_id}: oob_score mismatch."
        )


    if warm_start != EXPECTED_WARM_START:

        raise RuntimeError(
            f"{candidate_id}: warm_start mismatch."
        )


    if not np.isclose(
        ccp_alpha,
        EXPECTED_CCP_ALPHA,
        rtol=0.0,
        atol=0.0,
    ):

        raise RuntimeError(
            f"{candidate_id}: ccp_alpha mismatch."
        )


    if random_state != EXPECTED_RANDOM_STATE:

        raise RuntimeError(
            f"{candidate_id}: random_state mismatch."
        )


    if n_jobs != EXPECTED_N_JOBS:

        raise RuntimeError(
            f"{candidate_id}: n_jobs mismatch."
        )


    # ========================================================
    # VERIFY CANDIDATE SEARCH PARAMETERS
    # ========================================================

    if max_depth not in EXPECTED_MAX_DEPTH_VALUES:

        raise RuntimeError(
            f"{candidate_id}: unexpected max_depth "
            f"{max_depth}"
        )


    if (
        min_samples_leaf
        not in
        EXPECTED_MIN_SAMPLES_LEAF_VALUES
    ):

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            f"min_samples_leaf {min_samples_leaf}"
        )


    if (
        class_weight
        not in
        EXPECTED_CLASS_WEIGHT_VALUES
    ):

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            f"class_weight {class_weight}"
        )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        max_samples=max_samples,
        class_weight=class_weight,
        oob_score=oob_score,
        warm_start=warm_start,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
        n_jobs=n_jobs,
    )


    # ========================================================
    # TRAIN + CAPTURE WARNINGS
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
    # FITTED MODEL INTEGRITY
    # ========================================================

    fitted_tree_count = len(
        model.estimators_
    )


    if (
        fitted_tree_count
        !=
        n_estimators
    ):

        raise RuntimeError(
            f"{candidate_id}: expected "
            f"{n_estimators} fitted trees but found "
            f"{fitted_tree_count}."
        )


    if not np.array_equal(
        model.classes_,
        np.array(
            [0, 1]
        ),
    ):

        raise RuntimeError(
            f"{candidate_id}: unexpected classes "
            f"{model.classes_}"
        )


    if int(
        model.n_features_in_
    ) != 25:

        raise RuntimeError(
            f"{candidate_id}: fitted feature count "
            "is not 25."
        )


    feature_importances = np.asarray(
        model.feature_importances_,
        dtype=np.float64,
    )


    if feature_importances.shape != (
        25,
    ):

        raise RuntimeError(
            f"{candidate_id}: feature-importance "
            "shape is not 25."
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


    if not np.isclose(
        importance_sum,
        1.0,
        rtol=1e-10,
        atol=1e-10,
    ):

        raise RuntimeError(
            f"{candidate_id}: feature importances "
            f"sum to {importance_sum}, not 1."
        )


    # ========================================================
    # TREE STRUCTURE DIAGNOSTICS
    # ========================================================

    tree_depths = np.asarray(
        [
            estimator.tree_.max_depth

            for estimator
            in model.estimators_
        ],
        dtype=np.int64,
    )


    tree_leaves = np.asarray(
        [
            estimator.tree_.n_leaves

            for estimator
            in model.estimators_
        ],
        dtype=np.int64,
    )


    mean_tree_depth = float(
        tree_depths.mean()
    )


    maximum_tree_depth = int(
        tree_depths.max()
    )


    minimum_tree_depth = int(
        tree_depths.min()
    )


    mean_tree_leaves = float(
        tree_leaves.mean()
    )


    maximum_tree_leaves = int(
        tree_leaves.max()
    )


    minimum_tree_leaves = int(
        tree_leaves.min()
    )


    # ========================================================
    # TRAIN-PROBABILITY SANITY CHECK
    #
    # Diagnostic only.
    # No threshold / TSS / model selection.
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
            f"{candidate_id}: invalid "
            "probability range."
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
            "RandomForestClassifier",

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

        "criterion":
            criterion,

        "max_depth":
            max_depth,

        "min_samples_split":
            min_samples_split,

        "min_samples_leaf":
            min_samples_leaf,

        "max_features":
            max_features,

        "bootstrap":
            bootstrap,

        "max_samples":
            max_samples,

        "class_weight":
            class_weight,

        "oob_score":
            oob_score,

        "warm_start":
            warm_start,

        "ccp_alpha":
            ccp_alpha,

        "random_state":
            random_state,

        "n_jobs":
            n_jobs,

        "fitted_tree_count":
            fitted_tree_count,

        "sklearn_version":
            sklearn.__version__,
    }


    joblib.dump(
        model_bundle,
        model_file,
    )


    # ========================================================
    # RELOAD VERIFICATION
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
            f"{candidate_id}: reloaded candidate "
            "ID mismatch."
        )


    if (
        reloaded_bundle.get(
            "feature_order"
        )
        !=
        features
    ):

        raise RuntimeError(
            f"{candidate_id}: reloaded feature "
            "order mismatch."
        )


    if (
        reloaded_bundle.get(
            "train_sha256"
        )
        !=
        actual_train_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: reloaded TRAIN "
            "hash mismatch."
        )


    if (
        reloaded_bundle.get(
            "protocol_sha256"
        )
        !=
        protocol_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: reloaded protocol "
            "hash mismatch."
        )


    if (
        reloaded_bundle.get(
            "candidate_grid_sha256"
        )
        !=
        candidate_grid_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: reloaded candidate-grid "
            "hash mismatch."
        )


    reloaded_model = reloaded_bundle.get(
        "model"
    )


    if reloaded_model is None:

        raise RuntimeError(
            f"{candidate_id}: reloaded model missing."
        )


    if len(
        reloaded_model.estimators_
    ) != n_estimators:

        raise RuntimeError(
            f"{candidate_id}: reloaded tree count "
            "mismatch."
        )


    # ========================================================
    # SERIALIZED PROBABILITY REPRODUCTION
    #
    # Random Forest predict_proba can accumulate tree outputs
    # in parallel. Tiny floating-point summation differences
    # may therefore appear after serialization/reload.
    #
    # We use a strict absolute tolerance of 1e-12 rather than
    # requiring bit-for-bit equality.
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
            "does not reproduce probabilities "
            f"within tolerance. "
            f"max_abs_diff="
            f"{serialization_probability_max_abs_diff:.3e}, "
            f"atol="
            f"{SERIALIZATION_PROBABILITY_ATOL:.1e}"
        )


    # ========================================================
    # MODEL HASH
    # ========================================================

    model_hash = sha256_file(
        model_file
    )


    # ========================================================
    # RECORD SUMMARY
    # ========================================================

    training_rows.append(
        {
            "candidate_id":
                candidate_id,

            "n_estimators":
                n_estimators,

            "criterion":
                criterion,

            "max_depth":
                (
                    "None"
                    if max_depth is None
                    else max_depth
                ),

            "min_samples_split":
                min_samples_split,

            "min_samples_leaf":
                min_samples_leaf,

            "max_features":
                max_features,

            "bootstrap":
                bootstrap,

            "max_samples":
                (
                    "None"
                    if max_samples is None
                    else max_samples
                ),

            "class_weight":
                (
                    "None"
                    if class_weight is None
                    else class_weight
                ),

            "oob_score":
                oob_score,

            "warm_start":
                warm_start,

            "ccp_alpha":
                ccp_alpha,

            "random_state":
                random_state,

            "n_jobs":
                n_jobs,

            "train_rows":
                len(
                    y_train
                ),

            "train_positive":
                train_positive,

            "train_negative":
                train_negative,

            "feature_count":
                25,

            "fitted_tree_count":
                fitted_tree_count,

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

            "mean_tree_depth":
                mean_tree_depth,

            "min_tree_depth":
                minimum_tree_depth,

            "max_tree_depth_observed":
                maximum_tree_depth,

            "mean_tree_leaves":
                mean_tree_leaves,

            "min_tree_leaves":
                minimum_tree_leaves,

            "max_tree_leaves":
                maximum_tree_leaves,

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
        }
    )


    print(
        f"{candidate_id}: "
        f"trees={fitted_tree_count}, "
        f"mean_depth={mean_tree_depth:.2f}, "
        f"max_depth={maximum_tree_depth}, "
        f"reload_prob_diff="
        f"{serialization_probability_max_abs_diff:.3e}, "
        f"warnings={warning_count}, "
        f"saved=YES"
    )


# ============================================================
# BUILD TRAINING SUMMARY
# ============================================================

summary = pd.DataFrame(
    training_rows
)


if len(summary) != 12:

    raise RuntimeError(
        "RF training summary does not "
        "contain 12 candidates."
    )


if summary[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "RF training summary contains "
        "duplicate candidate IDs."
    )


if not summary[
    "fitted_tree_count"
].eq(
        EXPECTED_N_ESTIMATORS
    ).all():

    raise RuntimeError(
        "One or more RF candidates do not "
        "contain exactly 500 trees."
    )


if not summary[
    "feature_importances_finite"
].all():

    raise RuntimeError(
        "One or more RF candidates contain "
        "non-finite feature importances."
    )


if not summary[
    "probabilities_finite"
].all():

    raise RuntimeError(
        "One or more RF candidates contain "
        "non-finite probabilities."
    )


if not summary[
    "serialization_probability_match"
].all():

    raise RuntimeError(
        "One or more serialized RF candidates "
        "failed probability reproduction tolerance."
    )


# ============================================================
# FINAL COUNTS
# ============================================================

total_warnings = int(
    summary[
        "warning_count"
    ].sum()
)


candidate_count = len(
    summary
)


successful_candidates = int(
    summary[
        "fitted_tree_count"
    ].eq(
        EXPECTED_N_ESTIMATORS
    ).sum()
)


serialization_match_count = int(
    summary[
        "serialization_probability_match"
    ].sum()
)


maximum_serialization_probability_difference = float(
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
        "criterion"
    ].eq(
        EXPECTED_CRITERION
    ).all()

    and
    summary[
        "min_samples_split"
    ].eq(
        EXPECTED_MIN_SAMPLES_SPLIT
    ).all()

    and
    summary[
        "max_features"
    ].eq(
        EXPECTED_MAX_FEATURES
    ).all()

    and
    summary[
        "bootstrap"
    ].eq(
        EXPECTED_BOOTSTRAP
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
        "fitted_tree_count"
    ].eq(
        EXPECTED_N_ESTIMATORS
    ).all()

    and
    summary[
        "serialization_probability_match"
    ].all()
)


# ============================================================
# SAVE SUMMARY
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
# OUTPUT — TRAINING SUMMARY
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
            "max_depth",
            "min_samples_leaf",
            "class_weight",
            "fitted_tree_count",
            "mean_tree_depth",
            "max_tree_depth_observed",
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
    "Candidates with 500 trees:",
    successful_candidates
)

print(
    "Total fitted trees:",
    int(
        summary[
            "fitted_tree_count"
        ].sum()
    )
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
    maximum_serialization_probability_difference
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
    "criterion:",
    EXPECTED_CRITERION
)

print(
    "max_features:",
    EXPECTED_MAX_FEATURES
)

print(
    "bootstrap:",
    EXPECTED_BOOTSTRAP
)

print(
    "random_state:",
    EXPECTED_RANDOM_STATE
)

print(
    "n_jobs:",
    EXPECTED_N_JOBS
)

print(
    "All candidates match frozen protocol:",
    protocol_conformance
)


# ============================================================
# SERIALIZATION VERIFICATION
# ============================================================

print(
    "\n===================================="
)

print(
    "SERIALIZATION VERIFICATION"
)

print(
    "===================================="
)


print(
    "Candidates checked:",
    candidate_count
)

print(
    "Candidates within tolerance:",
    serialization_match_count
)

print(
    "Tolerance:",
    SERIALIZATION_PROBABILITY_ATOL
)

print(
    "Maximum observed probability difference:",
    maximum_serialization_probability_difference
)

print(
    "Serialization verification:",
    (
        "PASS"
        if serialization_match_count == 12
        else
        "FAIL"
    )
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
    "CHAPTER 4.2 RESULT"
)

print(
    "===================================="
)


print(
    "12 candidates trained:",
    candidate_count == 12
)

print(
    "12 candidates contain 500 trees:",
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
    "Chapter 4.2:",
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
        "Chapter 4.2 Random Forest "
        "candidate training failed."
    )


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 4.2 COMPLETE"
)

print(
    "===================================="
)