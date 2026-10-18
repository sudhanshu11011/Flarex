from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/cycle24_split_contract.json"
)

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/model_feature_contract.json"
)

PREPROCESSING_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/preprocessing_contract.json"
)

OUTPUT_DIR = Path(
    "artifacts/models/logistic"
)

CONTRACT_FILE = (
    OUTPUT_DIR
    / "contracts"
    / "logistic_training_protocol.json"
)

GRID_FILE = (
    OUTPUT_DIR
    / "manifests"
    / "logistic_candidate_grid.csv"
)


# ============================================================
# FIXED MODEL DESIGN
# ============================================================

PROTOCOL_VERSION = "1.0"

SOLVER = "lbfgs"
PENALTY = "l2"

C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

CLASS_WEIGHTS = [
    None,
    "balanced",
]

MAX_ITER = 5000
TOL = 1e-6

FIT_INTERCEPT = True

RANDOM_STATE = 24025


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
# VERIFY INPUT CONTRACTS
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 3.1 — LOGISTIC TRAINING PROTOCOL"
)

print(
    "===================================="
)


for path in [
    SPLIT_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required contract: {path}"
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


with PREPROCESSING_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    preprocessing_contract = json.load(file)


if split_contract.get("status") != "FROZEN":

    raise RuntimeError(
        "Cycle24 split contract is not frozen."
    )


if feature_contract.get("status") != "VERIFIED":

    raise RuntimeError(
        "Feature contract is not verified."
    )


if preprocessing_contract.get("status") != "FROZEN":

    raise RuntimeError(
        "Preprocessing contract is not frozen."
    )


features = feature_contract.get(
    "feature_order",
    []
)


if len(features) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


lr_preprocessing = preprocessing_contract.get(
    "logistic_regression",
    {}
)


if lr_preprocessing.get(
    "preprocessor"
) != "StandardScaler":

    raise RuntimeError(
        "LR preprocessing must be StandardScaler."
    )


if lr_preprocessing.get(
    "fit_scope"
) != "CYCLE24_TRAIN_ONLY":

    raise RuntimeError(
        "LR scaler was not fit on Cycle24 TRAIN only."
    )


if lr_preprocessing.get(
    "refit_allowed"
) is not False:

    raise RuntimeError(
        "Frozen scaler unexpectedly permits refitting."
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
    "Feature count:",
    len(features)
)


# ============================================================
# BUILD CANDIDATE GRID
# ============================================================

candidate_rows = []


candidate_number = 1


for c_value in C_VALUES:

    for class_weight in CLASS_WEIGHTS:

        weight_name = (
            "NONE"
            if class_weight is None
            else "BALANCED"
        )


        candidate_id = (
            f"LR_{candidate_number:02d}_"
            f"C{str(c_value).replace('.', 'p')}_"
            f"{weight_name}"
        )


        candidate_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "C":
                    c_value,

                "class_weight":
                    (
                        "None"
                        if class_weight is None
                        else class_weight
                    ),

                "penalty":
                    PENALTY,

                "solver":
                    SOLVER,

                "fit_intercept":
                    FIT_INTERCEPT,

                "max_iter":
                    MAX_ITER,

                "tol":
                    TOL,

                "random_state":
                    RANDOM_STATE,

                "train_allowed":
                    True,

                "validation_used_in_chapter3":
                    False,

                "holdout_used":
                    False,

                "cycle25_used":
                    False,
            }
        )


        candidate_number += 1


grid = pd.DataFrame(
    candidate_rows
)


if len(grid) != 10:

    raise RuntimeError(
        "Expected exactly 10 LR candidates."
    )


if grid[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate IDs."
    )


# ============================================================
# FREEZE SELECTION RULES NOW
# ============================================================

selection_protocol = {

    "selection_stage":
        "MODEL_CHAPTER_6",

    "development_dataset":
        "CYCLE24_VALIDATION",

    "primary_selection_metric":
        "TSS",

    "threshold_selection":
        (
            "For each candidate, choose the decision threshold "
            "that maximizes TSS on Cycle24 VALIDATION only."
        ),

    "hyperparameter_selection":
        (
            "Choose the pre-registered Logistic Regression "
            "candidate with the highest validation TSS after "
            "candidate-specific validation threshold selection."
        ),

    "tie_breaking_order": [
        "higher_validation_TSS",
        "higher_validation_PR_AUC",
        "smaller_C_stronger_regularization",
        "class_weight_None_before_balanced",
        "lexicographically_smaller_candidate_id",
    ],

    "holdout_used_for_selection":
        False,

    "cycle25_used_for_selection":
        False,

    "grid_change_after_validation":
        False,
}


# ============================================================
# CONTRACT
# ============================================================

protocol = {

    "contract_name":
        "Logistic Regression Training Protocol",

    "version":
        PROTOCOL_VERSION,

    "status":
        "FROZEN_BEFORE_TRAINING",

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "model_family":
        "LogisticRegression",

    "task":
        "binary_classification",

    "positive_class":
        1,

    "target_definition":
        (
            "At least one associated GOES M1.0+ flare "
            "within (t, t+24h]."
        ),

    "training_scope":
        "CYCLE24_TRAIN_ONLY",

    "feature_count":
        25,

    "feature_order":
        features,

    "preprocessing": {

        "method":
            "StandardScaler",

        "artifact":
            lr_preprocessing.get(
                "artifact"
            ),

        "artifact_sha256":
            lr_preprocessing.get(
                "artifact_sha256"
            ),

        "fit_scope":
            "CYCLE24_TRAIN_ONLY",

        "refit_allowed":
            False,
    },

    "model_fixed_parameters": {

        "penalty":
            PENALTY,

        "solver":
            SOLVER,

        "fit_intercept":
            FIT_INTERCEPT,

        "max_iter":
            MAX_ITER,

        "tol":
            TOL,

        "random_state":
            RANDOM_STATE,
    },

    "candidate_search_space": {

        "C":
            C_VALUES,

        "class_weight": [
            None,
            "balanced",
        ],

        "candidate_count":
            len(grid),
    },

    "design_rationale": {

        "l2_only":
            (
                "L2 regularization preserves the full frozen "
                "25-feature representation. L1/ElasticNet are "
                "excluded because they perform embedded feature "
                "selection, which is outside the primary "
                "feature contract."
            ),

        "class_weight_candidates":
            (
                "Both unweighted and balanced training are "
                "pre-registered because the positive class is "
                "rare. Selection will occur only on Cycle24 "
                "validation data in Chapter 6."
            ),

        "standardization":
            (
                "The frozen TRAIN-fitted StandardScaler is used "
                "because the 25 physical features span very "
                "different numerical scales."
            ),
    },

    "probability_output":
        "predict_proba(X)[:, 1]",

    "chapter3_rules": {

        "train_candidates":
            True,

        "training_data":
            "CYCLE24_TRAIN",

        "validation_metrics_allowed":
            False,

        "holdout_access_allowed":
            False,

        "cycle25_access_allowed":
            False,

        "candidate_grid_change_allowed":
            False,
    },

    "selection_protocol":
        selection_protocol,

    "future_holdout_rule":
        (
            "Cycle24 HOLDOUT remains untouched until model "
            "development and threshold selection are complete "
            "and the complete Cycle24 pipeline is frozen."
        ),

    "future_cycle25_rule":
        (
            "Cycle25 remains untouched for model development. "
            "The frozen pipeline will later be transferred "
            "unchanged to Cycle25."
        ),

    "provenance": {

        "split_contract": {
            "file":
                str(
                    SPLIT_CONTRACT_FILE
                ),

            "sha256":
                sha256_file(
                    SPLIT_CONTRACT_FILE
                ),
        },

        "feature_contract": {
            "file":
                str(
                    FEATURE_CONTRACT_FILE
                ),

            "sha256":
                sha256_file(
                    FEATURE_CONTRACT_FILE
                ),
        },

        "preprocessing_contract": {
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


# ============================================================
# SAVE
# ============================================================

CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


GRID_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


if CONTRACT_FILE.exists():

    with CONTRACT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        existing = json.load(file)


    if (
        existing.get(
            "candidate_search_space"
        )
        !=
        protocol[
            "candidate_search_space"
        ]
    ):

        raise RuntimeError(
            "Existing frozen LR protocol uses a "
            "different candidate grid."
        )


    if (
        existing.get(
            "feature_order"
        )
        !=
        features
    ):

        raise RuntimeError(
            "Existing LR protocol uses a "
            "different feature order."
        )


    print(
        "Existing Logistic Regression protocol "
        "matches current frozen design."
    )


else:

    with CONTRACT_FILE.open(
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
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "LOGISTIC REGRESSION DESIGN"
)

print(
    "===================================="
)


print(
    "Training data: CYCLE24 TRAIN ONLY"
)

print(
    "Features: 25"
)

print(
    "Preprocessing: frozen StandardScaler"
)

print(
    "Penalty: L2"
)

print(
    "Solver:",
    SOLVER
)

print(
    "C values:",
    C_VALUES
)

print(
    "Class weights:",
    CLASS_WEIGHTS
)

print(
    "Candidates:",
    len(grid)
)

print(
    "Max iterations:",
    MAX_ITER
)

print(
    "Tolerance:",
    TOL
)


print(
    "\n===================================="
)

print(
    "CANDIDATE GRID"
)

print(
    "===================================="
)


print(
    grid[
        [
            "candidate_id",
            "C",
            "class_weight",
            "penalty",
            "solver",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n===================================="
)

print(
    "SELECTION POLICY"
)

print(
    "===================================="
)


print(
    "Selection stage: Chapter 6"
)

print(
    "Primary metric: TSS"
)

print(
    "Threshold source: Cycle24 VALIDATION only"
)

print(
    "Hyperparameter source: Cycle24 VALIDATION only"
)

print(
    "Holdout used for selection: False"
)

print(
    "Cycle25 used for selection: False"
)

print(
    "Grid changes after validation: Forbidden"
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
    "TRAIN allowed: True"
)

print(
    "VALIDATION metrics in Chapter 3: False"
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
    CONTRACT_FILE
)

print(
    GRID_FILE
)


print(
    "\nMODEL CHAPTER 3.1 COMPLETE."
)