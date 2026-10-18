from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path(
    "artifacts/models/xgboost/models"
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

CANDIDATE_AUDIT_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_candidate_audit.csv"
)

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "model_feature_contract.json"
)

INVENTORY_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_artifact_inventory.csv"
)

SUMMARY_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_artifact_verification_summary.csv"
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


def as_bool(value):

    if isinstance(value, bool):
        return value

    return str(
        value
    ).strip().lower() == "true"


def float_match(
    left,
    right,
    atol=1e-12,
):

    return bool(
        np.isclose(
            float(left),
            float(right),
            rtol=0.0,
            atol=atol,
        )
    )


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 5.4 — XGBOOST ARTIFACT VERIFICATION"
)

print(
    "===================================="
)


# ============================================================
# VERIFY REQUIRED INPUTS
# ============================================================

required_files = [
    PROTOCOL_FILE,
    GRID_FILE,
    TRAINING_SUMMARY_FILE,
    CANDIDATE_AUDIT_FILE,
    FEATURE_CONTRACT_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


if not MODEL_DIR.exists():

    raise FileNotFoundError(
        f"XGBoost model directory missing: {MODEL_DIR}"
    )


# ============================================================
# LOAD CONTRACTS / MANIFESTS
# ============================================================

with PROTOCOL_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    protocol = json.load(file)


with FEATURE_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    feature_contract = json.load(file)


grid = pd.read_csv(
    GRID_FILE,
    keep_default_na=False,
)


training_summary = pd.read_csv(
    TRAINING_SUMMARY_FILE,
    keep_default_na=False,
)


candidate_audit = pd.read_csv(
    CANDIDATE_AUDIT_FILE,
    keep_default_na=False,
)


# ============================================================
# VERIFY CONTRACT STATES
# ============================================================

if protocol.get(
    "status"
) != "FROZEN_BEFORE_TRAINING":

    raise RuntimeError(
        "XGBoost training protocol is not frozen."
    )


if protocol.get(
    "model_family"
) != "XGBClassifier":

    raise RuntimeError(
        "Unexpected model family."
    )


if feature_contract.get(
    "status"
) != "VERIFIED":

    raise RuntimeError(
        "Feature contract is not VERIFIED."
    )


features = feature_contract.get(
    "feature_order",
    []
)


if len(features) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


# ============================================================
# EXPECTED CANDIDATE SET
# ============================================================

expected_ids = list(
    grid[
        "candidate_id"
    ].astype(str)
)


if len(expected_ids) != 12:

    raise RuntimeError(
        "Expected exactly 12 XGBoost candidates."
    )


if len(
    set(
        expected_ids
    )
) != 12:

    raise RuntimeError(
        "Duplicate candidate IDs in frozen grid."
    )


if set(
    training_summary[
        "candidate_id"
    ].astype(str)
) != set(
    expected_ids
):

    raise RuntimeError(
        "Training summary candidate set differs "
        "from frozen XGBoost grid."
    )


if set(
    candidate_audit[
        "candidate_id"
    ].astype(str)
) != set(
    expected_ids
):

    raise RuntimeError(
        "Chapter 5.3 candidate audit differs "
        "from frozen XGBoost grid."
    )


# ============================================================
# EXPECTED FIXED PROTOCOL VALUES
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


protocol_hash = sha256_file(
    PROTOCOL_FILE
)


grid_hash = sha256_file(
    GRID_FILE
)


# ============================================================
# DISK INVENTORY
# ============================================================

disk_files = sorted(
    MODEL_DIR.glob(
        "*.joblib"
    )
)


disk_names = {
    path.stem

    for path in disk_files
}


expected_names = set(
    expected_ids
)


missing_artifacts = (
    expected_names
    -
    disk_names
)


extra_artifacts = (
    disk_names
    -
    expected_names
)


print(
    "Expected artifacts:",
    len(
        expected_names
    )
)

print(
    "Joblib files on disk:",
    len(
        disk_files
    )
)

print(
    "Missing artifacts:",
    len(
        missing_artifacts
    )
)

print(
    "Unexpected artifacts:",
    len(
        extra_artifacts
    )
)


if missing_artifacts:

    raise RuntimeError(
        "Missing XGBoost artifacts: "
        f"{sorted(missing_artifacts)}"
    )


if extra_artifacts:

    raise RuntimeError(
        "Unexpected XGBoost artifacts: "
        f"{sorted(extra_artifacts)}"
    )


# ============================================================
# VERIFY EACH ARTIFACT
# ============================================================

inventory_rows = []


print(
    "\n===================================="
)

print(
    "ARTIFACT INVENTORY"
)

print(
    "===================================="
)


for candidate_id in expected_ids:

    model_file = (
        MODEL_DIR
        /
        f"{candidate_id}.joblib"
    )


    grid_row = grid[
        grid[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    training_row = training_summary[
        training_summary[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    audit_row = candidate_audit[
        candidate_audit[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(grid_row) != 1:

        raise RuntimeError(
            f"{candidate_id}: grid row not unique."
        )


    if len(training_row) != 1:

        raise RuntimeError(
            f"{candidate_id}: training row not unique."
        )


    if len(audit_row) != 1:

        raise RuntimeError(
            f"{candidate_id}: audit row not unique."
        )


    grid_row = grid_row.iloc[
        0
    ]

    training_row = training_row.iloc[
        0
    ]

    audit_row = audit_row.iloc[
        0
    ]


    # ========================================================
    # HASH
    # ========================================================

    actual_hash = sha256_file(
        model_file
    )


    expected_hash = str(
        training_row[
            "model_sha256"
        ]
    )


    hash_match = (
        actual_hash
        ==
        expected_hash
    )


    if not hash_match:

        raise RuntimeError(
            f"{candidate_id}: model hash mismatch."
        )


    # ========================================================
    # LOAD ARTIFACT
    # ========================================================

    bundle = joblib.load(
        model_file
    )


    model = bundle.get(
        "model"
    )


    if model is None:

        raise RuntimeError(
            f"{candidate_id}: model object missing."
        )


    booster_object = model.get_booster()


    if booster_object is None:

        raise RuntimeError(
            f"{candidate_id}: booster object missing."
        )


    # ========================================================
    # EXPECTED CANDIDATE PARAMETERS
    # ========================================================

    expected_learning_rate = float(
        grid_row[
            "learning_rate"
        ]
    )


    expected_max_depth = int(
        grid_row[
            "max_depth"
        ]
    )


    expected_scale_pos_weight = float(
        grid_row[
            "scale_pos_weight"
        ]
    )


    # ========================================================
    # BUNDLE METADATA
    # ========================================================

    candidate_id_match = (
        bundle.get(
            "candidate_id"
        )
        ==
        candidate_id
    )


    model_family_match = (
        bundle.get(
            "model_family"
        )
        ==
        "XGBClassifier"
    )


    preprocessing_match = (
        bundle.get(
            "preprocessing"
        )
        ==
        "IDENTITY_RAW_FEATURES"
    )


    training_scope_match = (
        bundle.get(
            "training_scope"
        )
        ==
        "CYCLE24_TRAIN_ONLY"
    )


    feature_order_match = (
        bundle.get(
            "feature_order"
        )
        ==
        features
    )


    feature_count_match = (
        int(
            bundle.get(
                "feature_count"
            )
        )
        ==
        25
    )


    protocol_hash_match = (
        bundle.get(
            "protocol_sha256"
        )
        ==
        protocol_hash
    )


    candidate_grid_hash_match = (
        bundle.get(
            "candidate_grid_sha256"
        )
        ==
        grid_hash
    )


    # ========================================================
    # FIXED PARAMETER MATCH
    # ========================================================

    n_estimators_match = (
        int(
            bundle.get(
                "n_estimators"
            )
        )
        ==
        EXPECTED_N_ESTIMATORS
    )


    booster_match = (
        str(
            bundle.get(
                "booster"
            )
        )
        ==
        EXPECTED_BOOSTER
    )


    objective_match = (
        str(
            bundle.get(
                "objective"
            )
        )
        ==
        EXPECTED_OBJECTIVE
    )


    eval_metric_match = (
        str(
            bundle.get(
                "eval_metric"
            )
        )
        ==
        EXPECTED_EVAL_METRIC
    )


    tree_method_match = (
        str(
            bundle.get(
                "tree_method"
            )
        )
        ==
        EXPECTED_TREE_METHOD
    )


    min_child_weight_match = float_match(
        bundle.get(
            "min_child_weight"
        ),
        EXPECTED_MIN_CHILD_WEIGHT,
    )


    gamma_match = float_match(
        bundle.get(
            "gamma"
        ),
        EXPECTED_GAMMA,
    )


    subsample_match = float_match(
        bundle.get(
            "subsample"
        ),
        EXPECTED_SUBSAMPLE,
    )


    colsample_bytree_match = float_match(
        bundle.get(
            "colsample_bytree"
        ),
        EXPECTED_COLSAMPLE_BYTREE,
    )


    reg_alpha_match = float_match(
        bundle.get(
            "reg_alpha"
        ),
        EXPECTED_REG_ALPHA,
    )


    reg_lambda_match = float_match(
        bundle.get(
            "reg_lambda"
        ),
        EXPECTED_REG_LAMBDA,
    )


    max_delta_step_match = float_match(
        bundle.get(
            "max_delta_step"
        ),
        EXPECTED_MAX_DELTA_STEP,
    )


    random_state_match = (
        int(
            bundle.get(
                "random_state"
            )
        )
        ==
        EXPECTED_RANDOM_STATE
    )


    n_jobs_match = (
        int(
            bundle.get(
                "n_jobs"
            )
        )
        ==
        EXPECTED_N_JOBS
    )


    verbosity_match = (
        int(
            bundle.get(
                "verbosity"
            )
        )
        ==
        EXPECTED_VERBOSITY
    )


    # ========================================================
    # CANDIDATE PARAMETER MATCH
    # ========================================================

    learning_rate_match = float_match(
        bundle.get(
            "learning_rate"
        ),
        expected_learning_rate,
    )


    max_depth_match = (
        int(
            bundle.get(
                "max_depth"
            )
        )
        ==
        expected_max_depth
    )


    scale_pos_weight_match = float_match(
        bundle.get(
            "scale_pos_weight"
        ),
        expected_scale_pos_weight,
    )


    # ========================================================
    # FITTED MODEL STRUCTURE
    # ========================================================

    boosted_rounds = int(
        booster_object.num_boosted_rounds()
    )


    boosted_rounds_match = (
        boosted_rounds
        ==
        EXPECTED_N_ESTIMATORS
    )


    dumped_tree_count = len(
        booster_object.get_dump(
            dump_format="json"
        )
    )


    dumped_tree_count_match = (
        dumped_tree_count
        ==
        EXPECTED_N_ESTIMATORS
    )


    classes_match = bool(
        np.array_equal(
            model.classes_,
            np.array(
                [
                    0,
                    1,
                ]
            ),
        )
    )


    fitted_feature_count_match = (
        int(
            model.n_features_in_
        )
        ==
        25
    )


    booster_feature_count_match = (
        int(
            booster_object.num_features()
        )
        ==
        25
    )


    # ========================================================
    # FEATURE IMPORTANCE INTEGRITY
    # ========================================================

    importances = np.asarray(
        model.feature_importances_,
        dtype=np.float64,
    )


    importance_shape_match = (
        importances.shape
        ==
        (
            25,
        )
    )


    importances_finite = bool(
        np.isfinite(
            importances
        ).all()
    )


    importance_sum = float(
        importances.sum()
    )


    importance_sum_valid = bool(
        np.isfinite(
            importance_sum
        )
        and
        importance_sum > 0.0
    )


    nonnegative_importances = bool(
        (
            importances
            >=
            0.0
        ).all()
    )


    # ========================================================
    # TRAINING / SERIALIZATION HISTORY
    # ========================================================

    training_warning_clean = (
        int(
            training_row[
                "warning_count"
            ]
        )
        ==
        0
    )


    serialization_pass = as_bool(
        training_row[
            "serialization_probability_match"
        ]
    )


    chapter53_pass = as_bool(
        audit_row[
            "candidate_audit_pass"
        ]
    )


    # ========================================================
    # EARLY STOPPING / VALIDATION CLEANLINESS
    # ========================================================

    early_stopping_clean = (
        bundle.get(
            "early_stopping_used"
        )
        is False
    )


    validation_clean = (
        bundle.get(
            "validation_used"
        )
        is False
    )


    # ========================================================
    # NO PREMATURE MODEL-SELECTION METADATA
    # ========================================================

    forbidden_selection_keys = {
        "selected_threshold",
        "threshold",
        "validation_tss",
        "validation_pr_auc",
        "validation_roc_auc",
        "validation_f1",
        "validation_hss",
        "holdout_tss",
        "cycle25_tss",
        "selected_model",
        "selected_candidate",
        "is_winner",
        "winner",
        "best_candidate",
        "best_score",
        "selection_rank",
    }


    stored_keys = set(
        bundle.keys()
    )


    forbidden_keys_present = sorted(
        forbidden_selection_keys
        &
        stored_keys
    )


    selection_metadata_clean = (
        len(
            forbidden_keys_present
        )
        ==
        0
    )


    # ========================================================
    # FINAL ARTIFACT PASS
    # ========================================================

    artifact_pass = bool(

        hash_match

        and
        candidate_id_match

        and
        model_family_match

        and
        preprocessing_match

        and
        training_scope_match

        and
        feature_order_match

        and
        feature_count_match

        and
        protocol_hash_match

        and
        candidate_grid_hash_match

        and
        n_estimators_match

        and
        booster_match

        and
        objective_match

        and
        eval_metric_match

        and
        tree_method_match

        and
        min_child_weight_match

        and
        gamma_match

        and
        subsample_match

        and
        colsample_bytree_match

        and
        reg_alpha_match

        and
        reg_lambda_match

        and
        max_delta_step_match

        and
        random_state_match

        and
        n_jobs_match

        and
        verbosity_match

        and
        learning_rate_match

        and
        max_depth_match

        and
        scale_pos_weight_match

        and
        boosted_rounds_match

        and
        dumped_tree_count_match

        and
        classes_match

        and
        fitted_feature_count_match

        and
        booster_feature_count_match

        and
        importance_shape_match

        and
        importances_finite

        and
        importance_sum_valid

        and
        nonnegative_importances

        and
        training_warning_clean

        and
        serialization_pass

        and
        chapter53_pass

        and
        early_stopping_clean

        and
        validation_clean

        and
        selection_metadata_clean
    )


    # ========================================================
    # INVENTORY ROW
    # ========================================================

    inventory_rows.append(
        {
            "candidate_id":
                candidate_id,

            "file":
                str(
                    model_file
                ),

            "size_bytes":
                model_file.stat().st_size,

            "sha256":
                actual_hash,

            "sha256_match":
                hash_match,

            "candidate_id_match":
                candidate_id_match,

            "model_family_match":
                model_family_match,

            "preprocessing_match":
                preprocessing_match,

            "training_scope_match":
                training_scope_match,

            "feature_order_match":
                feature_order_match,

            "feature_count_match":
                feature_count_match,

            "protocol_sha256_match":
                protocol_hash_match,

            "candidate_grid_sha256_match":
                candidate_grid_hash_match,

            "n_estimators_match":
                n_estimators_match,

            "booster_match":
                booster_match,

            "objective_match":
                objective_match,

            "eval_metric_match":
                eval_metric_match,

            "tree_method_match":
                tree_method_match,

            "min_child_weight_match":
                min_child_weight_match,

            "gamma_match":
                gamma_match,

            "subsample_match":
                subsample_match,

            "colsample_bytree_match":
                colsample_bytree_match,

            "reg_alpha_match":
                reg_alpha_match,

            "reg_lambda_match":
                reg_lambda_match,

            "max_delta_step_match":
                max_delta_step_match,

            "random_state_match":
                random_state_match,

            "n_jobs_match":
                n_jobs_match,

            "verbosity_match":
                verbosity_match,

            "learning_rate_match":
                learning_rate_match,

            "max_depth_match":
                max_depth_match,

            "scale_pos_weight_match":
                scale_pos_weight_match,

            "boosted_rounds":
                boosted_rounds,

            "boosted_rounds_match":
                boosted_rounds_match,

            "dumped_tree_count":
                dumped_tree_count,

            "dumped_tree_count_match":
                dumped_tree_count_match,

            "classes_0_1":
                classes_match,

            "fitted_feature_count_match":
                fitted_feature_count_match,

            "booster_feature_count_match":
                booster_feature_count_match,

            "importance_shape_25":
                importance_shape_match,

            "feature_importances_finite":
                importances_finite,

            "feature_importance_sum":
                importance_sum,

            "feature_importance_sum_valid":
                importance_sum_valid,

            "feature_importances_nonnegative":
                nonnegative_importances,

            "training_warning_clean":
                training_warning_clean,

            "serialization_probability_match":
                serialization_pass,

            "chapter_5_3_pass":
                chapter53_pass,

            "early_stopping_clean":
                early_stopping_clean,

            "validation_clean":
                validation_clean,

            "selection_metadata_clean":
                selection_metadata_clean,

            "forbidden_selection_keys":
                "|".join(
                    forbidden_keys_present
                ),

            "artifact_verification_pass":
                artifact_pass,

            "sklearn_version":
                str(
                    bundle.get(
                        "sklearn_version"
                    )
                ),

            "xgboost_version":
                str(
                    bundle.get(
                        "xgboost_version"
                    )
                ),
        }
    )


    print(
        f"{candidate_id}: "
        f"hash={hash_match}, "
        f"rounds={boosted_rounds}, "
        f"trees={dumped_tree_count}, "
        f"features={feature_order_match}, "
        f"selection_clean="
        f"{selection_metadata_clean}, "
        f"PASS={artifact_pass}"
    )


# ============================================================
# BUILD INVENTORY
# ============================================================

inventory = pd.DataFrame(
    inventory_rows
)


if len(
    inventory
) != 12:

    raise RuntimeError(
        "XGBoost artifact inventory does "
        "not contain 12 candidates."
    )


if inventory[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "XGBoost artifact inventory contains "
        "duplicate candidate IDs."
    )


# ============================================================
# SOFTWARE CONSISTENCY
# ============================================================

sklearn_versions = sorted(
    inventory[
        "sklearn_version"
    ].unique()
)


xgboost_versions = sorted(
    inventory[
        "xgboost_version"
    ].unique()
)


sklearn_version_consistent = (
    len(
        sklearn_versions
    )
    ==
    1
)


xgboost_version_consistent = (
    len(
        xgboost_versions
    )
    ==
    1
)


# ============================================================
# FINAL COUNTS
# ============================================================

exact_disk_inventory = bool(

    len(
        disk_files
    )
    ==
    12

    and
    len(
        missing_artifacts
    )
    ==
    0

    and
    len(
        extra_artifacts
    )
    ==
    0
)


artifact_pass_count = int(
    inventory[
        "artifact_verification_pass"
    ].sum()
)


hash_match_count = int(
    inventory[
        "sha256_match"
    ].sum()
)


feature_order_match_count = int(
    inventory[
        "feature_order_match"
    ].sum()
)


boosted_round_match_count = int(
    inventory[
        "boosted_rounds_match"
    ].sum()
)


dumped_tree_match_count = int(
    inventory[
        "dumped_tree_count_match"
    ].sum()
)


selection_clean_count = int(
    inventory[
        "selection_metadata_clean"
    ].sum()
)


early_stopping_clean_count = int(
    inventory[
        "early_stopping_clean"
    ].sum()
)


validation_clean_count = int(
    inventory[
        "validation_clean"
    ].sum()
)


all_artifacts_pass = bool(
    artifact_pass_count
    ==
    12
)


overall_pass = bool(

    exact_disk_inventory

    and
    all_artifacts_pass

    and
    sklearn_version_consistent

    and
    xgboost_version_consistent
)


# ============================================================
# VERIFICATION SUMMARY
# ============================================================

verification_summary = pd.DataFrame(
    [
        {
            "expected_artifacts":
                12,

            "disk_artifacts":
                len(
                    disk_files
                ),

            "missing_artifacts":
                len(
                    missing_artifacts
                ),

            "unexpected_artifacts":
                len(
                    extra_artifacts
                ),

            "artifact_pass_count":
                artifact_pass_count,

            "hash_match_count":
                hash_match_count,

            "feature_order_match_count":
                feature_order_match_count,

            "boosted_round_match_count":
                boosted_round_match_count,

            "dumped_tree_match_count":
                dumped_tree_match_count,

            "selection_metadata_clean_count":
                selection_clean_count,

            "early_stopping_clean_count":
                early_stopping_clean_count,

            "validation_clean_count":
                validation_clean_count,

            "sklearn_version_count":
                len(
                    sklearn_versions
                ),

            "sklearn_versions":
                "|".join(
                    sklearn_versions
                ),

            "xgboost_version_count":
                len(
                    xgboost_versions
                ),

            "xgboost_versions":
                "|".join(
                    xgboost_versions
                ),

            "dataset_files_read":
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

            "overall_artifact_verification_pass":
                overall_pass,
        }
    ]
)


# ============================================================
# SAVE
# ============================================================

INVENTORY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


inventory.to_csv(
    INVENTORY_FILE,
    index=False,
)


verification_summary.to_csv(
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
    "ARTIFACT COMPLETENESS"
)

print(
    "===================================="
)


print(
    "Expected XGBoost model files:",
    12
)

print(
    "Actual XGBoost model files:",
    len(
        disk_files
    )
)

print(
    "Missing:",
    len(
        missing_artifacts
    )
)

print(
    "Unexpected:",
    len(
        extra_artifacts
    )
)


print(
    "\n===================================="
)

print(
    "ARTIFACT INTEGRITY"
)

print(
    "===================================="
)


print(
    "Hash matches:",
    hash_match_count,
    "/ 12"
)

print(
    "Feature-order matches:",
    feature_order_match_count,
    "/ 12"
)

print(
    "500-round artifacts:",
    boosted_round_match_count,
    "/ 12"
)

print(
    "500-tree artifacts:",
    dumped_tree_match_count,
    "/ 12"
)

print(
    "Early stopping clean:",
    early_stopping_clean_count,
    "/ 12"
)

print(
    "Validation metadata clean:",
    validation_clean_count,
    "/ 12"
)

print(
    "Selection metadata clean:",
    selection_clean_count,
    "/ 12"
)

print(
    "Artifact verification PASS:",
    artifact_pass_count,
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
    "scikit-learn versions:",
    sklearn_versions
)

print(
    "scikit-learn version consistent:",
    sklearn_version_consistent
)

print(
    "XGBoost versions:",
    xgboost_versions
)

print(
    "XGBoost version consistent:",
    xgboost_version_consistent
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
    "Dataset files read:",
    False
)

print(
    "VALIDATION read:",
    False
)

print(
    "HOLDOUT read:",
    False
)

print(
    "Cycle25 read:",
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
    "\n===================================="
)

print(
    "CHAPTER 5.4 RESULT"
)

print(
    "===================================="
)


print(
    "Overall XGBoost artifact verification:",
    (
        "PASS"
        if overall_pass
        else
        "FAIL"
    )
)


print(
    "\nSaved:"
)

print(
    INVENTORY_FILE
)

print(
    SUMMARY_FILE
)


if not overall_pass:

    raise RuntimeError(
        "Chapter 5.4 XGBoost "
        "artifact verification failed."
    )


print(
    "\nMODEL CHAPTER 5.4 COMPLETE."
)