from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

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

PROTOCOL_FILE = Path(
    "artifacts/models/xgboost/contracts/"
    "xgboost_training_protocol.json"
)

GRID_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_candidate_grid.csv"
)

TRAINING_SUMMARY_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_training_summary.csv"
)

MODEL_DIR = Path(
    "artifacts/models/xgboost/models"
)

AUDIT_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_reproducibility_audit.csv"
)

SUMMARY_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_reproducibility_summary.csv"
)


# ============================================================
# REPRODUCIBILITY TOLERANCES
# ============================================================

PROBABILITY_ATOL = 1e-12

IMPORTANCE_ATOL = 1e-12


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 5.5 — XGBOOST REPRODUCIBILITY"
)

print(
    "===================================="
)


# ============================================================
# VERIFY INPUTS
# ============================================================

required_files = [
    TRAIN_FILE,
    SPLIT_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    PROTOCOL_FILE,
    GRID_FILE,
    TRAINING_SUMMARY_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


if not MODEL_DIR.exists():

    raise FileNotFoundError(
        f"Missing XGBoost model directory: {MODEL_DIR}"
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


if protocol.get(
    "status"
) != "FROZEN_BEFORE_TRAINING":

    raise RuntimeError(
        "XGBoost protocol is not frozen."
    )


if protocol.get(
    "model_family"
) != "XGBClassifier":

    raise RuntimeError(
        "Unexpected model family."
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
    "XGBoost training protocol: FROZEN"
)


# ============================================================
# LOAD TRAIN ONLY
# ============================================================

train = pd.read_parquet(
    TRAIN_FILE
)


X_train = (
    train[
        features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .to_numpy(
        dtype=np.float32
    )
)


y_train = pd.to_numeric(
    train[
        "target"
    ],
    errors="raise",
).to_numpy(
    dtype=np.int32
)


if X_train.shape != (
    62064,
    25,
):

    raise RuntimeError(
        f"Unexpected TRAIN shape: {X_train.shape}"
    )


if not np.isfinite(
    X_train
).all():

    raise RuntimeError(
        "TRAIN contains non-finite features."
    )


if int(
    y_train.sum()
) != 2420:

    raise RuntimeError(
        "Unexpected TRAIN positive count."
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
    "Preprocessing: RAW / IDENTITY"
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
# LOAD GRID + TRAINING SUMMARY
# ============================================================

grid = pd.read_csv(
    GRID_FILE,
    keep_default_na=False,
)


training_summary = pd.read_csv(
    TRAINING_SUMMARY_FILE,
    keep_default_na=False,
)


if len(grid) != 12:

    raise RuntimeError(
        "Expected exactly 12 XGBoost candidates."
    )


if len(training_summary) != 12:

    raise RuntimeError(
        "Expected exactly 12 training-summary rows."
    )


if set(
    grid[
        "candidate_id"
    ].astype(str)
) != set(
    training_summary[
        "candidate_id"
    ].astype(str)
):

    raise RuntimeError(
        "Candidate IDs differ between "
        "grid and training summary."
    )


# ============================================================
# FIXED PARAMETERS
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
# REPRODUCIBILITY AUDIT
# ============================================================

audit_rows = []


print(
    "\n===================================="
)

print(
    "RETRAIN / REPRODUCE XGBOOST CANDIDATES"
)

print(
    "===================================="
)


for _, row in grid.iterrows():

    candidate_id = str(
        row[
            "candidate_id"
        ]
    ).strip()


    model_file = (
        MODEL_DIR
        /
        f"{candidate_id}.joblib"
    )


    if not model_file.exists():

        raise FileNotFoundError(
            f"{candidate_id}: saved model missing."
        )


    # ========================================================
    # LOAD SAVED MODEL
    # ========================================================

    saved_bundle = joblib.load(
        model_file
    )


    saved_model = saved_bundle.get(
        "model"
    )


    if saved_model is None:

        raise RuntimeError(
            f"{candidate_id}: saved model missing "
            "inside bundle."
        )


    saved_booster = saved_model.get_booster()


    # ========================================================
    # CANDIDATE PARAMETERS
    # ========================================================

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
    # VERIFY FROZEN FIXED PARAMETERS
    # ========================================================

    if n_estimators != EXPECTED_N_ESTIMATORS:

        raise RuntimeError(
            f"{candidate_id}: n_estimators mismatch."
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
            f"{candidate_id}: min_child_weight mismatch."
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
            f"{candidate_id}: colsample_bytree mismatch."
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
            f"{candidate_id}: max_delta_step mismatch."
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
    # FRESH IN-MEMORY MODEL
    # ========================================================

    fresh_model = XGBClassifier(
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
    # RETRAIN
    #
    # No eval_set.
    # No early stopping.
    # TRAIN only.
    # ========================================================

    with warnings.catch_warnings(
        record=True
    ) as captured_warnings:

        warnings.simplefilter(
            "always"
        )


        fresh_model.fit(
            X_train,
            y_train,
        )


    fresh_warning_count = len(
        captured_warnings
    )


    fresh_booster = fresh_model.get_booster()


    # ========================================================
    # BOOSTING ROUND REPRODUCIBILITY
    # ========================================================

    saved_rounds = int(
        saved_booster.num_boosted_rounds()
    )


    fresh_rounds = int(
        fresh_booster.num_boosted_rounds()
    )


    boosted_round_match = bool(
        saved_rounds
        ==
        fresh_rounds
        ==
        EXPECTED_N_ESTIMATORS
    )


    # ========================================================
    # CLASS / FEATURE REPRODUCIBILITY
    # ========================================================

    classes_match = bool(
        np.array_equal(
            saved_model.classes_,
            fresh_model.classes_,
        )
    )


    feature_count_match = bool(
        int(
            saved_model.n_features_in_
        )
        ==
        int(
            fresh_model.n_features_in_
        )
        ==
        25
    )


    booster_feature_count_match = bool(
        int(
            saved_booster.num_features()
        )
        ==
        int(
            fresh_booster.num_features()
        )
        ==
        25
    )


    # ========================================================
    # FEATURE-IMPORTANCE REPRODUCIBILITY
    # ========================================================

    saved_importances = np.asarray(
        saved_model.feature_importances_,
        dtype=np.float64,
    )


    fresh_importances = np.asarray(
        fresh_model.feature_importances_,
        dtype=np.float64,
    )


    if (
        saved_importances.shape
        !=
        (
            25,
        )
        or
        fresh_importances.shape
        !=
        (
            25,
        )
    ):

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            "feature-importance shape."
        )


    importance_max_abs_diff = float(
        np.max(
            np.abs(
                saved_importances
                -
                fresh_importances
            )
        )
    )


    importance_match = bool(
        np.allclose(
            saved_importances,
            fresh_importances,
            rtol=0.0,
            atol=IMPORTANCE_ATOL,
        )
    )


    # ========================================================
    # TREE STRUCTURE REPRODUCIBILITY
    #
    # Compare complete JSON dumps of all 500 trees.
    # ========================================================

    saved_tree_dump = saved_booster.get_dump(
        dump_format="json"
    )


    fresh_tree_dump = fresh_booster.get_dump(
        dump_format="json"
    )


    saved_tree_count = len(
        saved_tree_dump
    )


    fresh_tree_count = len(
        fresh_tree_dump
    )


    dumped_tree_count_match = bool(
        saved_tree_count
        ==
        fresh_tree_count
        ==
        EXPECTED_N_ESTIMATORS
    )


    exact_tree_matches = int(
        sum(
            saved_tree == fresh_tree

            for saved_tree, fresh_tree
            in zip(
                saved_tree_dump,
                fresh_tree_dump,
            )
        )
    )


    tree_dump_exact_match = bool(
        dumped_tree_count_match

        and
        exact_tree_matches
        ==
        EXPECTED_N_ESTIMATORS
    )


    # ========================================================
    # TRAIN PROBABILITY REPRODUCIBILITY
    # ========================================================

    saved_probabilities = (
        saved_model.predict_proba(
            X_train
        )[
            :,
            1
        ]
    )


    fresh_probabilities = (
        fresh_model.predict_proba(
            X_train
        )[
            :,
            1
        ]
    )


    if not np.isfinite(
        saved_probabilities
    ).all():

        raise RuntimeError(
            f"{candidate_id}: saved probabilities "
            "contain non-finite values."
        )


    if not np.isfinite(
        fresh_probabilities
    ).all():

        raise RuntimeError(
            f"{candidate_id}: fresh probabilities "
            "contain non-finite values."
        )


    probability_max_abs_diff = float(
        np.max(
            np.abs(
                saved_probabilities
                -
                fresh_probabilities
            )
        )
    )


    probability_match = bool(
        np.allclose(
            saved_probabilities,
            fresh_probabilities,
            rtol=0.0,
            atol=PROBABILITY_ATOL,
        )
    )


    # ========================================================
    # SOFTWARE VERSION
    # ========================================================

    saved_xgboost_version = str(
        saved_bundle.get(
            "xgboost_version"
        )
    )


    current_xgboost_version = str(
        xgboost.__version__
    )


    xgboost_version_match = bool(
        saved_xgboost_version
        ==
        current_xgboost_version
    )


    # ========================================================
    # FINAL REPRODUCIBILITY RESULT
    # ========================================================

    reproducibility_pass = bool(

        fresh_warning_count == 0

        and
        boosted_round_match

        and
        classes_match

        and
        feature_count_match

        and
        booster_feature_count_match

        and
        importance_match

        and
        dumped_tree_count_match

        and
        tree_dump_exact_match

        and
        probability_match

        and
        xgboost_version_match
    )


    audit_rows.append(
        {
            "candidate_id":
                candidate_id,

            "saved_boosted_rounds":
                saved_rounds,

            "fresh_boosted_rounds":
                fresh_rounds,

            "boosted_round_match":
                boosted_round_match,

            "classes_match":
                classes_match,

            "feature_count_match":
                feature_count_match,

            "booster_feature_count_match":
                booster_feature_count_match,

            "feature_importance_max_abs_diff":
                importance_max_abs_diff,

            "feature_importance_match":
                importance_match,

            "saved_tree_count":
                saved_tree_count,

            "fresh_tree_count":
                fresh_tree_count,

            "dumped_tree_count_match":
                dumped_tree_count_match,

            "exact_tree_matches":
                exact_tree_matches,

            "tree_dump_exact_match":
                tree_dump_exact_match,

            "probability_max_abs_diff":
                probability_max_abs_diff,

            "probability_match":
                probability_match,

            "fresh_warning_count":
                fresh_warning_count,

            "saved_xgboost_version":
                saved_xgboost_version,

            "current_xgboost_version":
                current_xgboost_version,

            "xgboost_version_match":
                xgboost_version_match,

            "reproducibility_pass":
                reproducibility_pass,
        }
    )


    print(
        f"{candidate_id}: "
        f"rounds={boosted_round_match}, "
        f"trees_exact="
        f"{exact_tree_matches}/500, "
        f"importance_diff="
        f"{importance_max_abs_diff:.3e}, "
        f"prob_diff="
        f"{probability_max_abs_diff:.3e}, "
        f"warnings="
        f"{fresh_warning_count}, "
        f"PASS={reproducibility_pass}"
    )


# ============================================================
# BUILD AUDIT
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


if len(
    audit_df
) != 12:

    raise RuntimeError(
        "XGBoost reproducibility audit "
        "does not contain 12 candidates."
    )


if audit_df[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate IDs in "
        "reproducibility audit."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

pass_count = int(
    audit_df[
        "reproducibility_pass"
    ].sum()
)


round_match_count = int(
    audit_df[
        "boosted_round_match"
    ].sum()
)


tree_exact_match_count = int(
    audit_df[
        "tree_dump_exact_match"
    ].sum()
)


importance_match_count = int(
    audit_df[
        "feature_importance_match"
    ].sum()
)


probability_match_count = int(
    audit_df[
        "probability_match"
    ].sum()
)


max_importance_difference = float(
    audit_df[
        "feature_importance_max_abs_diff"
    ].max()
)


max_probability_difference = float(
    audit_df[
        "probability_max_abs_diff"
    ].max()
)


total_fresh_warnings = int(
    audit_df[
        "fresh_warning_count"
    ].sum()
)


all_reproducible = bool(
    pass_count
    ==
    12
)


summary_df = pd.DataFrame(
    [
        {
            "candidates_checked":
                12,

            "candidates_reproducible":
                pass_count,

            "boosted_round_match_count":
                round_match_count,

            "exact_tree_dump_match_count":
                tree_exact_match_count,

            "feature_importance_match_count":
                importance_match_count,

            "probability_match_count":
                probability_match_count,

            "max_feature_importance_abs_difference":
                max_importance_difference,

            "feature_importance_atol":
                IMPORTANCE_ATOL,

            "max_probability_abs_difference":
                max_probability_difference,

            "probability_atol":
                PROBABILITY_ATOL,

            "fresh_warning_count":
                total_fresh_warnings,

            "xgboost_version":
                str(
                    xgboost.__version__
                ),

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

            "fresh_models_saved":
                False,

            "reproducibility_pass":
                all_reproducible,
        }
    ]
)


# ============================================================
# SAVE
# ============================================================

AUDIT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


audit_df.to_csv(
    AUDIT_FILE,
    index=False,
)


summary_df.to_csv(
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
    "REPRODUCIBILITY SUMMARY"
)

print(
    "===================================="
)


print(
    "Candidates checked:",
    12
)

print(
    "Candidates reproducible:",
    pass_count
)

print(
    "Maximum feature-importance difference:",
    max_importance_difference
)

print(
    "Maximum probability difference:",
    max_probability_difference
)

print(
    "Fresh training warnings:",
    total_fresh_warnings
)


print(
    "\n===================================="
)

print(
    "STRUCTURAL REPRODUCIBILITY"
)

print(
    "===================================="
)


print(
    "Boosted-round matches:",
    round_match_count,
    "/ 12"
)

print(
    "Exact 500-tree dump matches:",
    tree_exact_match_count,
    "/ 12"
)

print(
    "Feature-importance vectors match:",
    importance_match_count,
    "/ 12"
)

print(
    "Probability vectors match:",
    probability_match_count,
    "/ 12"
)


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
    "Current XGBoost version:",
    xgboost.__version__
)

print(
    "Version matches saved artifacts:",
    bool(
        audit_df[
            "xgboost_version_match"
        ].all()
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
    "Model selection performed:",
    False
)

print(
    "Threshold selection performed:",
    False
)

print(
    "Fresh models saved:",
    False
)


print(
    "\n===================================="
)

print(
    "CHAPTER 5.5 RESULT"
)

print(
    "===================================="
)


print(
    "XGBoost reproducibility:",
    (
        "PASS"
        if all_reproducible
        else
        "FAIL"
    )
)


print(
    "\nSaved:"
)

print(
    AUDIT_FILE
)

print(
    SUMMARY_FILE
)


if not all_reproducible:

    raise RuntimeError(
        "Chapter 5.5 XGBoost "
        "reproducibility verification failed."
    )


print(
    "\nMODEL CHAPTER 5.5 COMPLETE."
)