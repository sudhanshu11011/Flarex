from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

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

PROTOCOL_FILE = Path(
    "artifacts/models/xgboost/contracts/"
    "xgboost_training_protocol.json"
)

GRID_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_candidate_grid.csv"
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
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 5.1 — FREEZE XGBOOST PROTOCOL"
)

print(
    "===================================="
)


# ============================================================
# VERIFY REQUIRED INPUTS
# ============================================================

required_files = [
    SPLIT_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# LOAD FROZEN CONTRACTS
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


features = feature_contract.get(
    "feature_order",
    []
)


if len(features) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


train_hash = split_contract[
    "train"
][
    "sha256"
]


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
    "Features:",
    len(features)
)


# ============================================================
# FROZEN TRAIN CLASS COUNTS
# ============================================================

TRAIN_ROWS = 62064

TRAIN_POSITIVE = 2420

TRAIN_NEGATIVE = 59644


if (
    TRAIN_POSITIVE
    +
    TRAIN_NEGATIVE
    !=
    TRAIN_ROWS
):

    raise RuntimeError(
        "Frozen TRAIN class counts do not "
        "sum to expected TRAIN rows."
    )


BALANCED_SCALE_POS_WEIGHT = (
    TRAIN_NEGATIVE
    /
    TRAIN_POSITIVE
)


print(
    "TRAIN rows:",
    TRAIN_ROWS
)

print(
    "TRAIN positives:",
    TRAIN_POSITIVE
)

print(
    "TRAIN negatives:",
    TRAIN_NEGATIVE
)

print(
    "Balanced scale_pos_weight:",
    BALANCED_SCALE_POS_WEIGHT
)


# ============================================================
# FIXED XGBOOST PARAMETERS
# ============================================================

N_ESTIMATORS = 500

BOOSTER = "gbtree"

OBJECTIVE = "binary:logistic"

EVAL_METRIC = "logloss"

TREE_METHOD = "hist"

SUBSAMPLE = 0.8

COLSAMPLE_BYTREE = 0.8

MIN_CHILD_WEIGHT = 1.0

GAMMA = 0.0

REG_ALPHA = 0.0

REG_LAMBDA = 1.0

MAX_DELTA_STEP = 0.0

RANDOM_STATE = 24025

N_JOBS = -1

VERBOSITY = 0


# ============================================================
# CANDIDATE SEARCH SPACE
# ============================================================

LEARNING_RATE_VALUES = [
    0.03,
    0.10,
]


MAX_DEPTH_VALUES = [
    3,
    6,
    9,
]


SCALE_POS_WEIGHT_VALUES = [
    1.0,
    BALANCED_SCALE_POS_WEIGHT,
]


# ============================================================
# BUILD CANDIDATE GRID
# ============================================================

candidate_rows = []

candidate_number = 1


for learning_rate in LEARNING_RATE_VALUES:

    for max_depth in MAX_DEPTH_VALUES:

        for scale_pos_weight in SCALE_POS_WEIGHT_VALUES:

            weight_label = (
                "NONE"
                if scale_pos_weight == 1.0
                else "BAL"
            )


            learning_rate_label = (
                str(
                    learning_rate
                )
                .replace(
                    ".",
                    "p"
                )
            )


            candidate_id = (
                f"XGB_{candidate_number:02d}"
                f"_LR{learning_rate_label}"
                f"_D{max_depth}"
                f"_{weight_label}"
            )


            candidate_rows.append(
                {
                    "candidate_id":
                        candidate_id,

                    "n_estimators":
                        N_ESTIMATORS,

                    "learning_rate":
                        learning_rate,

                    "max_depth":
                        max_depth,

                    "min_child_weight":
                        MIN_CHILD_WEIGHT,

                    "gamma":
                        GAMMA,

                    "subsample":
                        SUBSAMPLE,

                    "colsample_bytree":
                        COLSAMPLE_BYTREE,

                    "reg_alpha":
                        REG_ALPHA,

                    "reg_lambda":
                        REG_LAMBDA,

                    "scale_pos_weight":
                        scale_pos_weight,

                    "max_delta_step":
                        MAX_DELTA_STEP,

                    "booster":
                        BOOSTER,

                    "objective":
                        OBJECTIVE,

                    "eval_metric":
                        EVAL_METRIC,

                    "tree_method":
                        TREE_METHOD,

                    "random_state":
                        RANDOM_STATE,

                    "n_jobs":
                        N_JOBS,

                    "verbosity":
                        VERBOSITY,
                }
            )


            candidate_number += 1


grid = pd.DataFrame(
    candidate_rows
)


# ============================================================
# VERIFY GRID
# ============================================================

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


expected_ids = [
    "XGB_01_LR0p03_D3_NONE",
    "XGB_02_LR0p03_D3_BAL",
    "XGB_03_LR0p03_D6_NONE",
    "XGB_04_LR0p03_D6_BAL",
    "XGB_05_LR0p03_D9_NONE",
    "XGB_06_LR0p03_D9_BAL",
    "XGB_07_LR0p1_D3_NONE",
    "XGB_08_LR0p1_D3_BAL",
    "XGB_09_LR0p1_D6_NONE",
    "XGB_10_LR0p1_D6_BAL",
    "XGB_11_LR0p1_D9_NONE",
    "XGB_12_LR0p1_D9_BAL",
]


if grid[
    "candidate_id"
].tolist() != expected_ids:

    raise RuntimeError(
        "XGBoost candidate ordering differs "
        "from frozen design."
    )


# ============================================================
# BUILD PROTOCOL
# ============================================================

protocol = {

    "protocol_name":
        "XGBoost Candidate Training Protocol",

    "version":
        "1.0",

    "status":
        "FROZEN_BEFORE_TRAINING",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "model_family":
        "XGBClassifier",

    "scientific_role":
        (
            "Gradient-boosted tree model used as a "
            "nonlinear primary model for cross-cycle "
            "predictive and SHAP explanation-stability "
            "analysis."
        ),

    "training_scope":
        "CYCLE24_TRAIN_ONLY",

    "feature_count":
        25,

    "feature_order":
        features,

    "preprocessing":
        {
            "method":
                "IDENTITY_RAW_FEATURES",

            "scaling":
                False,

            "imputation":
                False,

            "feature_selection":
                False,

            "pca":
                False,

            "clipping":
                False,

            "winsorization":
                False,

            "log_transform":
                False,
        },

    "train_class_distribution":
        {
            "rows":
                TRAIN_ROWS,

            "positive":
                TRAIN_POSITIVE,

            "negative":
                TRAIN_NEGATIVE,

            "positive_rate":
                (
                    TRAIN_POSITIVE
                    /
                    TRAIN_ROWS
                ),

            "negative_to_positive_ratio":
                BALANCED_SCALE_POS_WEIGHT,
        },

    "model_fixed_parameters":
        {
            "n_estimators":
                N_ESTIMATORS,

            "booster":
                BOOSTER,

            "objective":
                OBJECTIVE,

            "eval_metric":
                EVAL_METRIC,

            "tree_method":
                TREE_METHOD,

            "min_child_weight":
                MIN_CHILD_WEIGHT,

            "gamma":
                GAMMA,

            "subsample":
                SUBSAMPLE,

            "colsample_bytree":
                COLSAMPLE_BYTREE,

            "reg_alpha":
                REG_ALPHA,

            "reg_lambda":
                REG_LAMBDA,

            "max_delta_step":
                MAX_DELTA_STEP,

            "random_state":
                RANDOM_STATE,

            "n_jobs":
                N_JOBS,

            "verbosity":
                VERBOSITY,
        },

    "candidate_search_space":
        {
            "learning_rate":
                LEARNING_RATE_VALUES,

            "max_depth":
                MAX_DEPTH_VALUES,

            "scale_pos_weight":
                SCALE_POS_WEIGHT_VALUES,

            "candidate_count":
                12,
        },

    "candidate_design_rationale":
        {
            "learning_rate":
                (
                    "Compares conservative boosting "
                    "with a moderate learning rate while "
                    "keeping the tree count fixed before "
                    "validation."
                ),

            "max_depth":
                (
                    "Tests shallow, moderate, and deeper "
                    "interaction structures without using "
                    "validation information."
                ),

            "scale_pos_weight":
                (
                    "Compares unweighted boosting against "
                    "a TRAIN-only class-imbalance correction. "
                    "The balanced value is frozen as "
                    "TRAIN negatives / TRAIN positives."
                ),

            "n_estimators":
                (
                    "Fixed at 500 to avoid turning the "
                    "number of boosting rounds into an "
                    "additional validation-tuned axis."
                ),

            "subsample_and_colsample":
                (
                    "Fixed at 0.8 to introduce controlled "
                    "stochastic regularization without "
                    "expanding the candidate grid."
                ),
        },

    "early_stopping":
        {
            "enabled":
                False,

            "reason":
                (
                    "Early stopping would require an "
                    "evaluation dataset. Cycle24 validation "
                    "is intentionally isolated until "
                    "Chapter 6 model selection."
                ),
        },

    "training_provenance":
        {
            "train_split":
                "CYCLE24_TRAIN",

            "train_sha256":
                train_hash,

            "split_contract":
                str(
                    SPLIT_CONTRACT_FILE
                ),

            "split_contract_sha256":
                sha256_file(
                    SPLIT_CONTRACT_FILE
                ),

            "feature_contract":
                str(
                    FEATURE_CONTRACT_FILE
                ),

            "feature_contract_sha256":
                sha256_file(
                    FEATURE_CONTRACT_FILE
                ),

            "preprocessing_contract":
                str(
                    PREPROCESSING_CONTRACT_FILE
                ),

            "preprocessing_contract_sha256":
                sha256_file(
                    PREPROCESSING_CONTRACT_FILE
                ),
        },

    "data_isolation":
        {
            "train_design_information_used":
                True,

            "validation_used":
                False,

            "holdout_used":
                False,

            "cycle25_used":
                False,
        },

    "selection_protocol":
        {
            "selection_stage":
                "MODEL_CHAPTER_6",

            "selection_dataset":
                "CYCLE24_VALIDATION",

            "primary_metric":
                "TSS",

            "threshold_strategy":
                (
                    "For each frozen XGBoost candidate, "
                    "choose the probability threshold that "
                    "maximizes TSS on Cycle24 validation only."
                ),

            "winner_rule":
                (
                    "Select the frozen XGBoost candidate "
                    "with highest Cycle24 validation TSS "
                    "after candidate-specific threshold "
                    "selection."
                ),

            "tie_break_order":
                [
                    "higher_validation_TSS",
                    "higher_validation_PR_AUC",
                    "smaller_max_depth",
                    "smaller_learning_rate",
                    "scale_pos_weight_1_before_balanced",
                    "lexicographically_smaller_candidate_id",
                ],

            "holdout_for_selection":
                False,

            "cycle25_for_selection":
                False,
        },

    "explainability_note":
        (
            "Built-in XGBoost feature importance is not "
            "the primary explanation method. Final "
            "explanation analysis will use the frozen "
            "SHAP protocol. Feature importance from "
            "training is diagnostic only."
        ),

    "immutability_rule":
        (
            "After this protocol is frozen, XGBoost "
            "candidate hyperparameters, candidate count, "
            "feature set, TRAIN class-weight ratio, "
            "training split, preprocessing, and candidate "
            "definitions may not be changed in response "
            "to validation, holdout, or Cycle25 results."
        ),
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

PROTOCOL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


GRID_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAFE FREEZE
# ============================================================

if PROTOCOL_FILE.exists():

    raise RuntimeError(
        "XGBoost training protocol already exists. "
        "Refusing to overwrite a frozen protocol."
    )


if GRID_FILE.exists():

    raise RuntimeError(
        "XGBoost candidate grid already exists. "
        "Refusing to overwrite candidate definitions."
    )


# ============================================================
# SAVE
# ============================================================

with PROTOCOL_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        protocol,
        file,
        indent=2,
    )


grid.to_csv(
    GRID_FILE,
    index=False,
)


# ============================================================
# DISPLAY CANDIDATE GRID
# ============================================================

print(
    "\n===================================="
)

print(
    "XGBOOST CANDIDATE GRID"
)

print(
    "===================================="
)


print(
    grid[
        [
            "candidate_id",
            "learning_rate",
            "max_depth",
            "scale_pos_weight",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# PROTOCOL SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "FROZEN TRAINING PROTOCOL"
)

print(
    "===================================="
)


print(
    "Model: XGBClassifier"
)

print(
    "Candidates:",
    len(
        grid
    )
)

print(
    "n_estimators:",
    N_ESTIMATORS
)

print(
    "booster:",
    BOOSTER
)

print(
    "objective:",
    OBJECTIVE
)

print(
    "eval_metric:",
    EVAL_METRIC
)

print(
    "tree_method:",
    TREE_METHOD
)

print(
    "subsample:",
    SUBSAMPLE
)

print(
    "colsample_bytree:",
    COLSAMPLE_BYTREE
)

print(
    "random_state:",
    RANDOM_STATE
)

print(
    "Preprocessing: RAW / IDENTITY"
)

print(
    "Early stopping:",
    False
)


# ============================================================
# IMBALANCE HANDLING
# ============================================================

print(
    "\n===================================="
)

print(
    "CLASS IMBALANCE DESIGN"
)

print(
    "===================================="
)


print(
    "Unweighted scale_pos_weight:",
    1.0
)

print(
    "Balanced scale_pos_weight:",
    BALANCED_SCALE_POS_WEIGHT
)

print(
    "Balanced weight source:",
    "CYCLE24 TRAIN ONLY"
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
    "TRAIN design information used:",
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
    "Model training performed:",
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
    "\nSaved:"
)

print(
    PROTOCOL_FILE
)

print(
    GRID_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 5.1 COMPLETE — XGBOOST PROTOCOL FROZEN"
)

print(
    "===================================="
)