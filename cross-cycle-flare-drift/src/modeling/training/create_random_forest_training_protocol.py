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
    "artifacts/models/random_forest/contracts/"
    "random_forest_training_protocol.json"
)

GRID_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_candidate_grid.csv"
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
    "MODEL CHAPTER 4.1 — FREEZE RANDOM FOREST PROTOCOL"
)

print(
    "===================================="
)


# ============================================================
# VERIFY INPUT FILES
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
# VERIFY STATES
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
# FIXED RANDOM FOREST PARAMETERS
# ============================================================

N_ESTIMATORS = 500

CRITERION = "gini"

MAX_FEATURES = "sqrt"

MIN_SAMPLES_SPLIT = 2

BOOTSTRAP = True

MAX_SAMPLES = None

OOB_SCORE = False

WARM_START = False

CCP_ALPHA = 0.0

RANDOM_STATE = 24025

N_JOBS = -1


# ============================================================
# CANDIDATE SEARCH SPACE
# ============================================================

MAX_DEPTH_VALUES = [
    12,
    24,
    None,
]


MIN_SAMPLES_LEAF_VALUES = [
    1,
    5,
]


CLASS_WEIGHT_VALUES = [
    None,
    "balanced_subsample",
]


# ============================================================
# BUILD CANDIDATE GRID
# ============================================================

candidate_rows = []

candidate_number = 1


for max_depth in MAX_DEPTH_VALUES:

    for min_samples_leaf in MIN_SAMPLES_LEAF_VALUES:

        for class_weight in CLASS_WEIGHT_VALUES:

            depth_label = (
                "NONE"
                if max_depth is None
                else str(max_depth)
            )


            class_weight_label = (
                "NONE"
                if class_weight is None
                else "BALSUB"
            )


            candidate_id = (
                f"RF_{candidate_number:02d}"
                f"_D{depth_label}"
                f"_L{min_samples_leaf}"
                f"_{class_weight_label}"
            )


            candidate_rows.append(
                {
                    "candidate_id":
                        candidate_id,

                    "n_estimators":
                        N_ESTIMATORS,

                    "criterion":
                        CRITERION,

                    "max_depth":
                        (
                            "None"
                            if max_depth is None
                            else max_depth
                        ),

                    "min_samples_split":
                        MIN_SAMPLES_SPLIT,

                    "min_samples_leaf":
                        min_samples_leaf,

                    "max_features":
                        MAX_FEATURES,

                    "bootstrap":
                        BOOTSTRAP,

                    "max_samples":
                        "None",

                    "class_weight":
                        (
                            "None"
                            if class_weight is None
                            else class_weight
                        ),

                    "oob_score":
                        OOB_SCORE,

                    "warm_start":
                        WARM_START,

                    "ccp_alpha":
                        CCP_ALPHA,

                    "random_state":
                        RANDOM_STATE,

                    "n_jobs":
                        N_JOBS,
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
        "Expected exactly 12 RF candidates."
    )


if grid[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate RF candidate IDs."
    )


expected_ids = [
    "RF_01_D12_L1_NONE",
    "RF_02_D12_L1_BALSUB",
    "RF_03_D12_L5_NONE",
    "RF_04_D12_L5_BALSUB",
    "RF_05_D24_L1_NONE",
    "RF_06_D24_L1_BALSUB",
    "RF_07_D24_L5_NONE",
    "RF_08_D24_L5_BALSUB",
    "RF_09_DNONE_L1_NONE",
    "RF_10_DNONE_L1_BALSUB",
    "RF_11_DNONE_L5_NONE",
    "RF_12_DNONE_L5_BALSUB",
]


if grid[
    "candidate_id"
].tolist() != expected_ids:

    raise RuntimeError(
        "RF candidate ordering differs "
        "from frozen design."
    )


# ============================================================
# BUILD PROTOCOL
# ============================================================

protocol = {

    "protocol_name":
        "Random Forest Candidate Training Protocol",

    "version":
        "1.0",

    "status":
        "FROZEN_BEFORE_TRAINING",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "model_family":
        "RandomForestClassifier",

    "scientific_role":
        (
            "Nonlinear tree-ensemble baseline for "
            "cross-cycle predictive and explanation "
            "stability analysis."
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

            "log_transform":
                False,
        },

    "model_fixed_parameters":
        {
            "n_estimators":
                N_ESTIMATORS,

            "criterion":
                CRITERION,

            "min_samples_split":
                MIN_SAMPLES_SPLIT,

            "max_features":
                MAX_FEATURES,

            "bootstrap":
                BOOTSTRAP,

            "max_samples":
                MAX_SAMPLES,

            "oob_score":
                OOB_SCORE,

            "warm_start":
                WARM_START,

            "ccp_alpha":
                CCP_ALPHA,

            "random_state":
                RANDOM_STATE,

            "n_jobs":
                N_JOBS,
        },

    "candidate_search_space":
        {
            "max_depth":
                [
                    12,
                    24,
                    None,
                ],

            "min_samples_leaf":
                [
                    1,
                    5,
                ],

            "class_weight":
                [
                    None,
                    "balanced_subsample",
                ],

            "candidate_count":
                12,
        },

    "candidate_design_rationale":
        {
            "max_depth":
                (
                    "Tests moderate depth, deeper trees, "
                    "and unrestricted depth while controlling "
                    "the search space before validation."
                ),

            "min_samples_leaf":
                (
                    "Tests fine-grained leaves versus "
                    "regularized leaves to reduce possible "
                    "overfitting."
                ),

            "class_weight":
                (
                    "Compares the unweighted model against "
                    "balanced_subsample because M/X flare "
                    "positives are rare in Cycle24 TRAIN."
                ),

            "n_estimators":
                (
                    "Fixed at 500 to provide a sufficiently "
                    "large ensemble while avoiding an "
                    "unnecessary validation-tuned axis."
                ),

            "max_features":
                (
                    "sqrt is fixed as a standard RF "
                    "decorrelation mechanism and is not "
                    "tuned using validation data."
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
            "train_used":
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
                    "For each frozen RF candidate, choose "
                    "the decision threshold that maximizes "
                    "TSS on Cycle24 validation only."
                ),

            "winner_rule":
                (
                    "Select the RF candidate with highest "
                    "Cycle24 validation TSS after its "
                    "candidate-specific threshold selection."
                ),

            "tie_break_order":
                [
                    "higher_validation_TSS",
                    "higher_validation_PR_AUC",
                    "smaller_max_depth",
                    "larger_min_samples_leaf",
                    "class_weight_None_before_balanced_subsample",
                    "lexicographically_smaller_candidate_id",
                ],

            "holdout_for_selection":
                False,

            "cycle25_for_selection":
                False,
        },

    "immutability_rule":
        (
            "After this protocol is frozen, RF candidate "
            "hyperparameters, candidate count, feature set, "
            "training split, and preprocessing may not be "
            "changed in response to validation, holdout, "
            "or Cycle25 results."
        ),
}


# ============================================================
# OUTPUT DIRECTORIES
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
        "Random Forest training protocol already exists. "
        "Refusing to overwrite a frozen protocol."
    )


if GRID_FILE.exists():

    raise RuntimeError(
        "Random Forest candidate grid already exists. "
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
# DISPLAY GRID
# ============================================================

print(
    "\n===================================="
)

print(
    "RANDOM FOREST CANDIDATE GRID"
)

print(
    "===================================="
)


print(
    grid[
        [
            "candidate_id",
            "max_depth",
            "min_samples_leaf",
            "class_weight",
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
    "Model: RandomForestClassifier"
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
    "criterion:",
    CRITERION
)

print(
    "max_features:",
    MAX_FEATURES
)

print(
    "bootstrap:",
    BOOTSTRAP
)

print(
    "random_state:",
    RANDOM_STATE
)

print(
    "Preprocessing: RAW / IDENTITY"
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
    "TRAIN designated:",
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
    "MODEL CHAPTER 4.1 COMPLETE — RF PROTOCOL FROZEN"
)

print(
    "===================================="
)