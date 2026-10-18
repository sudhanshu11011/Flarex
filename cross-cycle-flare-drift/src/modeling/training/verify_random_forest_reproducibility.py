from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

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

PROTOCOL_FILE = Path(
    "artifacts/models/random_forest/contracts/"
    "random_forest_training_protocol.json"
)

GRID_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_candidate_grid.csv"
)

TRAINING_SUMMARY_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_training_summary.csv"
)

MODEL_DIR = Path(
    "artifacts/models/random_forest/models"
)

AUDIT_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_reproducibility_audit.csv"
)

SUMMARY_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_reproducibility_summary.csv"
)


# ============================================================
# REPRODUCIBILITY TOLERANCES
# ============================================================

PROBABILITY_ATOL = 1e-12

IMPORTANCE_ATOL = 1e-12

THRESHOLD_ATOL = 1e-12


# ============================================================
# HELPERS
# ============================================================

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


def parse_class_weight(value):

    text = str(value).strip().lower()

    if text == "none":
        return None

    if text == "balanced_subsample":
        return "balanced_subsample"

    raise RuntimeError(
        f"Unsupported class_weight: {value}"
    )


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 4.5 — RANDOM FOREST REPRODUCIBILITY"
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
        "RF training protocol is not frozen."
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
    "RF training protocol: FROZEN"
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
        dtype=np.float64
    )
)


y_train = pd.to_numeric(
    train[
        "target"
    ],
    errors="raise",
).to_numpy(
    dtype=np.int64
)


if X_train.shape != (
    62064,
    25,
):

    raise RuntimeError(
        f"Unexpected TRAIN shape: {X_train.shape}"
    )


if int(
    y_train.sum()
) != 2420:

    raise RuntimeError(
        "Unexpected TRAIN positive count."
    )


if not np.isfinite(
    X_train
).all():

    raise RuntimeError(
        "TRAIN contains non-finite features."
    )


print(
    "TRAIN rows:",
    len(y_train)
)

print(
    "TRAIN positives:",
    int(
        y_train.sum()
    )
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
# LOAD GRID + SUMMARY
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
        "Expected 12 RF candidates."
    )


if len(training_summary) != 12:

    raise RuntimeError(
        "Expected 12 RF training-summary rows."
    )


# ============================================================
# EXPECTED FIXED PARAMETERS
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
# AUDIT
# ============================================================

audit_rows = []


print(
    "\n===================================="
)

print(
    "RETRAIN / REPRODUCE RF CANDIDATES"
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


    saved_bundle = joblib.load(
        model_file
    )


    saved_model = saved_bundle.get(
        "model"
    )


    if saved_model is None:

        raise RuntimeError(
            f"{candidate_id}: saved model object missing."
        )


    # --------------------------------------------------------
    # CANDIDATE PARAMETERS
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


    class_weight = parse_class_weight(
        row[
            "class_weight"
        ]
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


    # --------------------------------------------------------
    # VERIFY FROZEN FIXED PARAMETERS
    # --------------------------------------------------------

    if n_estimators != EXPECTED_N_ESTIMATORS:

        raise RuntimeError(
            f"{candidate_id}: n_estimators mismatch."
        )


    if criterion != EXPECTED_CRITERION:

        raise RuntimeError(
            f"{candidate_id}: criterion mismatch."
        )


    if min_samples_split != EXPECTED_MIN_SAMPLES_SPLIT:

        raise RuntimeError(
            f"{candidate_id}: min_samples_split mismatch."
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


    # --------------------------------------------------------
    # FRESH IN-MEMORY MODEL
    # --------------------------------------------------------

    fresh_model = RandomForestClassifier(
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


    warning_count = len(
        captured_warnings
    )


    # ========================================================
    # TREE COUNT
    # ========================================================

    saved_tree_count = len(
        saved_model.estimators_
    )


    fresh_tree_count = len(
        fresh_model.estimators_
    )


    tree_count_match = bool(
        saved_tree_count
        ==
        fresh_tree_count
        ==
        EXPECTED_N_ESTIMATORS
    )


    # ========================================================
    # CLASSES
    # ========================================================

    classes_match = bool(
        np.array_equal(
            saved_model.classes_,
            fresh_model.classes_,
        )
    )


    # ========================================================
    # FEATURE IMPORTANCES
    # ========================================================

    saved_importances = np.asarray(
        saved_model.feature_importances_,
        dtype=np.float64,
    )


    fresh_importances = np.asarray(
        fresh_model.feature_importances_,
        dtype=np.float64,
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
    # TREE STRUCTURE COMPARISON
    # ========================================================

    node_count_match = True

    child_structure_match = True

    split_feature_match = True

    threshold_match = True

    maximum_threshold_difference = 0.0


    for saved_tree, fresh_tree in zip(
        saved_model.estimators_,
        fresh_model.estimators_,
    ):

        saved_structure = saved_tree.tree_

        fresh_structure = fresh_tree.tree_


        if (
            saved_structure.node_count
            !=
            fresh_structure.node_count
        ):

            node_count_match = False

            child_structure_match = False

            split_feature_match = False

            threshold_match = False

            continue


        if not np.array_equal(
            saved_structure.children_left,
            fresh_structure.children_left,
        ):

            child_structure_match = False


        if not np.array_equal(
            saved_structure.children_right,
            fresh_structure.children_right,
        ):

            child_structure_match = False


        if not np.array_equal(
            saved_structure.feature,
            fresh_structure.feature,
        ):

            split_feature_match = False


        threshold_difference = float(
            np.max(
                np.abs(
                    saved_structure.threshold
                    -
                    fresh_structure.threshold
                )
            )
        )


        maximum_threshold_difference = max(
            maximum_threshold_difference,
            threshold_difference,
        )


        if not np.allclose(
            saved_structure.threshold,
            fresh_structure.threshold,
            rtol=0.0,
            atol=THRESHOLD_ATOL,
        ):

            threshold_match = False


    tree_structure_match = bool(

        node_count_match

        and
        child_structure_match

        and
        split_feature_match

        and
        threshold_match
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
    # TREE DEPTH REPRODUCIBILITY
    # ========================================================

    saved_depths = np.asarray(
        [
            tree.tree_.max_depth

            for tree
            in saved_model.estimators_
        ],
        dtype=np.int64,
    )


    fresh_depths = np.asarray(
        [
            tree.tree_.max_depth

            for tree
            in fresh_model.estimators_
        ],
        dtype=np.int64,
    )


    depth_vector_match = bool(
        np.array_equal(
            saved_depths,
            fresh_depths,
        )
    )


    # ========================================================
    # TREE LEAF REPRODUCIBILITY
    # ========================================================

    saved_leaves = np.asarray(
        [
            tree.tree_.n_leaves

            for tree
            in saved_model.estimators_
        ],
        dtype=np.int64,
    )


    fresh_leaves = np.asarray(
        [
            tree.tree_.n_leaves

            for tree
            in fresh_model.estimators_
        ],
        dtype=np.int64,
    )


    leaf_vector_match = bool(
        np.array_equal(
            saved_leaves,
            fresh_leaves,
        )
    )


    # ========================================================
    # FINAL CANDIDATE RESULT
    # ========================================================

    reproducibility_pass = bool(

        warning_count == 0

        and
        tree_count_match

        and
        classes_match

        and
        importance_match

        and
        tree_structure_match

        and
        depth_vector_match

        and
        leaf_vector_match

        and
        probability_match
    )


    audit_rows.append(
        {
            "candidate_id":
                candidate_id,

            "saved_tree_count":
                saved_tree_count,

            "fresh_tree_count":
                fresh_tree_count,

            "tree_count_match":
                tree_count_match,

            "classes_match":
                classes_match,

            "feature_importance_max_abs_diff":
                importance_max_abs_diff,

            "feature_importance_match":
                importance_match,

            "node_count_match":
                node_count_match,

            "child_structure_match":
                child_structure_match,

            "split_feature_match":
                split_feature_match,

            "threshold_max_abs_diff":
                maximum_threshold_difference,

            "threshold_match":
                threshold_match,

            "tree_structure_match":
                tree_structure_match,

            "depth_vector_match":
                depth_vector_match,

            "leaf_vector_match":
                leaf_vector_match,

            "probability_max_abs_diff":
                probability_max_abs_diff,

            "probability_match":
                probability_match,

            "fresh_warning_count":
                warning_count,

            "reproducibility_pass":
                reproducibility_pass,
        }
    )


    print(
        f"{candidate_id}: "
        f"trees={tree_count_match}, "
        f"structure={tree_structure_match}, "
        f"importance_diff="
        f"{importance_max_abs_diff:.3e}, "
        f"prob_diff="
        f"{probability_max_abs_diff:.3e}, "
        f"PASS={reproducibility_pass}"
    )


# ============================================================
# BUILD AUDIT
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


if len(audit_df) != 12:

    raise RuntimeError(
        "RF reproducibility audit does not "
        "contain 12 candidates."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

pass_count = int(
    audit_df[
        "reproducibility_pass"
    ].sum()
)


max_importance_difference = float(
    audit_df[
        "feature_importance_max_abs_diff"
    ].max()
)


max_threshold_difference = float(
    audit_df[
        "threshold_max_abs_diff"
    ].max()
)


max_probability_difference = float(
    audit_df[
        "probability_max_abs_diff"
    ].max()
)


total_warnings = int(
    audit_df[
        "fresh_warning_count"
    ].sum()
)


all_reproducible = bool(
    pass_count == 12
)


summary_df = pd.DataFrame(
    [
        {
            "candidates_checked":
                12,

            "candidates_reproducible":
                pass_count,

            "max_feature_importance_abs_difference":
                max_importance_difference,

            "max_tree_threshold_abs_difference":
                max_threshold_difference,

            "max_probability_abs_difference":
                max_probability_difference,

            "feature_importance_atol":
                IMPORTANCE_ATOL,

            "tree_threshold_atol":
                THRESHOLD_ATOL,

            "probability_atol":
                PROBABILITY_ATOL,

            "fresh_warning_count":
                total_warnings,

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
    "Maximum tree-threshold difference:",
    max_threshold_difference
)

print(
    "Maximum probability difference:",
    max_probability_difference
)

print(
    "Fresh training warnings:",
    total_warnings
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
    "Tree-count matches:",
    int(
        audit_df[
            "tree_count_match"
        ].sum()
    ),
    "/ 12"
)

print(
    "Tree-structure matches:",
    int(
        audit_df[
            "tree_structure_match"
        ].sum()
    ),
    "/ 12"
)

print(
    "Depth vectors match:",
    int(
        audit_df[
            "depth_vector_match"
        ].sum()
    ),
    "/ 12"
)

print(
    "Leaf vectors match:",
    int(
        audit_df[
            "leaf_vector_match"
        ].sum()
    ),
    "/ 12"
)

print(
    "Feature-importance vectors match:",
    int(
        audit_df[
            "feature_importance_match"
        ].sum()
    ),
    "/ 12"
)

print(
    "Probability vectors match:",
    int(
        audit_df[
            "probability_match"
        ].sum()
    ),
    "/ 12"
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
    "CHAPTER 4.5 RESULT"
)

print(
    "===================================="
)


print(
    "Random Forest reproducibility:",
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
        "Chapter 4.5 Random Forest "
        "reproducibility verification failed."
    )


print(
    "\nMODEL CHAPTER 4.5 COMPLETE."
)