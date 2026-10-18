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
    "artifacts/models/random_forest/models"
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

CANDIDATE_AUDIT_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_candidate_audit.csv"
)

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "model_feature_contract.json"
)

INVENTORY_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_artifact_inventory.csv"
)

SUMMARY_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_artifact_verification_summary.csv"
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


def parse_optional_int(value):

    text = str(value).strip()

    if text.lower() == "none":
        return None

    return int(
        float(text)
    )


def parse_expected_class_weight(value):

    text = str(value).strip().lower()

    if text == "none":
        return None

    if text == "balanced_subsample":
        return "balanced_subsample"

    raise RuntimeError(
        f"Unsupported class_weight value: {value}"
    )


def as_bool(value):

    if isinstance(value, bool):
        return value

    return str(
        value
    ).strip().lower() == "true"


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 4.4 — RF ARTIFACT VERIFICATION"
)

print(
    "===================================="
)


# ============================================================
# VERIFY REQUIRED FILES
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
        f"Model directory missing: {MODEL_DIR}"
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
        "RF training protocol is not frozen."
    )


if feature_contract.get(
    "status"
) != "VERIFIED":

    raise RuntimeError(
        "Feature contract is not VERIFIED."
    )


if protocol.get(
    "model_family"
) != "RandomForestClassifier":

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
        "Expected exactly 12 RF candidates."
    )


if len(
    set(
        expected_ids
    )
) != 12:

    raise RuntimeError(
        "Duplicate RF candidate IDs."
    )


if set(
    training_summary[
        "candidate_id"
    ].astype(str)
) != set(expected_ids):

    raise RuntimeError(
        "Training summary candidate set differs "
        "from frozen RF grid."
    )


if set(
    candidate_audit[
        "candidate_id"
    ].astype(str)
) != set(expected_ids):

    raise RuntimeError(
        "Chapter 4.3 candidate audit differs "
        "from frozen RF grid."
    )


# ============================================================
# EXPECTED PROTOCOL VALUES
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
        "Missing RF artifacts: "
        f"{sorted(missing_artifacts)}"
    )


if extra_artifacts:

    raise RuntimeError(
        "Unexpected RF artifacts: "
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


    grid_row = grid_row.iloc[0]
    training_row = training_row.iloc[0]
    audit_row = audit_row.iloc[0]


    # --------------------------------------------------------
    # HASH
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # LOAD ARTIFACT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # EXPECTED CANDIDATE-SPECIFIC VALUES
    # --------------------------------------------------------

    expected_max_depth = parse_optional_int(
        grid_row[
            "max_depth"
        ]
    )


    expected_min_samples_leaf = int(
        grid_row[
            "min_samples_leaf"
        ]
    )


    expected_class_weight = parse_expected_class_weight(
        grid_row[
            "class_weight"
        ]
    )


    # --------------------------------------------------------
    # BUNDLE METADATA
    # --------------------------------------------------------

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
        "RandomForestClassifier"
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


    # --------------------------------------------------------
    # FIXED PARAMETER MATCH
    # --------------------------------------------------------

    n_estimators_match = (
        int(
            bundle.get(
                "n_estimators"
            )
        )
        ==
        EXPECTED_N_ESTIMATORS
    )


    criterion_match = (
        str(
            bundle.get(
                "criterion"
            )
        )
        ==
        EXPECTED_CRITERION
    )


    min_samples_split_match = (
        int(
            bundle.get(
                "min_samples_split"
            )
        )
        ==
        EXPECTED_MIN_SAMPLES_SPLIT
    )


    max_features_match = (
        str(
            bundle.get(
                "max_features"
            )
        )
        ==
        EXPECTED_MAX_FEATURES
    )


    bootstrap_match = (
        bool(
            bundle.get(
                "bootstrap"
            )
        )
        ==
        EXPECTED_BOOTSTRAP
    )


    oob_score_match = (
        bool(
            bundle.get(
                "oob_score"
            )
        )
        ==
        EXPECTED_OOB_SCORE
    )


    warm_start_match = (
        bool(
            bundle.get(
                "warm_start"
            )
        )
        ==
        EXPECTED_WARM_START
    )


    ccp_alpha_match = bool(
        np.isclose(
            float(
                bundle.get(
                    "ccp_alpha"
                )
            ),
            EXPECTED_CCP_ALPHA,
            rtol=0.0,
            atol=0.0,
        )
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


    # --------------------------------------------------------
    # CANDIDATE-SPECIFIC PARAMETER MATCH
    # --------------------------------------------------------

    max_depth_match = (
        bundle.get(
            "max_depth"
        )
        ==
        expected_max_depth
    )


    min_samples_leaf_match = (
        int(
            bundle.get(
                "min_samples_leaf"
            )
        )
        ==
        expected_min_samples_leaf
    )


    class_weight_match = (
        bundle.get(
            "class_weight"
        )
        ==
        expected_class_weight
    )


    # --------------------------------------------------------
    # FITTED MODEL STRUCTURE
    # --------------------------------------------------------

    fitted_tree_count = len(
        model.estimators_
    )


    tree_count_match = (
        fitted_tree_count
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


    importance_sum_match = bool(
        np.isclose(
            importances.sum(),
            1.0,
            rtol=1e-10,
            atol=1e-10,
        )
    )


    # --------------------------------------------------------
    # CHAPTER 4.2 / 4.3 VERIFICATION
    # --------------------------------------------------------

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


    chapter43_pass = as_bool(
        audit_row[
            "candidate_audit_pass"
        ]
    )


    # --------------------------------------------------------
    # NO PREMATURE SELECTION METADATA
    # --------------------------------------------------------

    forbidden_selection_keys = {
        "selected_threshold",
        "validation_tss",
        "validation_pr_auc",
        "holdout_tss",
        "cycle25_tss",
        "selected_model",
        "is_winner",
        "winner",
        "best_candidate",
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


    # --------------------------------------------------------
    # FINAL CANDIDATE PASS
    # --------------------------------------------------------

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
        criterion_match

        and
        min_samples_split_match

        and
        max_features_match

        and
        bootstrap_match

        and
        oob_score_match

        and
        warm_start_match

        and
        ccp_alpha_match

        and
        random_state_match

        and
        n_jobs_match

        and
        max_depth_match

        and
        min_samples_leaf_match

        and
        class_weight_match

        and
        tree_count_match

        and
        classes_match

        and
        fitted_feature_count_match

        and
        importance_shape_match

        and
        importances_finite

        and
        importance_sum_match

        and
        training_warning_clean

        and
        serialization_pass

        and
        chapter43_pass

        and
        selection_metadata_clean
    )


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

            "criterion_match":
                criterion_match,

            "min_samples_split_match":
                min_samples_split_match,

            "max_features_match":
                max_features_match,

            "bootstrap_match":
                bootstrap_match,

            "oob_score_match":
                oob_score_match,

            "warm_start_match":
                warm_start_match,

            "ccp_alpha_match":
                ccp_alpha_match,

            "random_state_match":
                random_state_match,

            "n_jobs_match":
                n_jobs_match,

            "max_depth_match":
                max_depth_match,

            "min_samples_leaf_match":
                min_samples_leaf_match,

            "class_weight_match":
                class_weight_match,

            "fitted_tree_count":
                fitted_tree_count,

            "tree_count_match":
                tree_count_match,

            "classes_0_1":
                classes_match,

            "fitted_feature_count_match":
                fitted_feature_count_match,

            "importance_shape_25":
                importance_shape_match,

            "feature_importances_finite":
                importances_finite,

            "feature_importance_sum_1":
                importance_sum_match,

            "training_warning_clean":
                training_warning_clean,

            "serialization_probability_match":
                serialization_pass,

            "chapter_4_3_pass":
                chapter43_pass,

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
        }
    )


    print(
        f"{candidate_id}: "
        f"hash={hash_match}, "
        f"trees={fitted_tree_count}, "
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
        "RF artifact inventory does not "
        "contain 12 candidates."
    )


if inventory[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "RF artifact inventory contains "
        "duplicate candidate IDs."
    )


# ============================================================
# SOFTWARE CONSISTENCY
# ============================================================

software_versions = sorted(
    inventory[
        "sklearn_version"
    ].unique()
)


software_version_consistent = (
    len(
        software_versions
    )
    ==
    1
)


# ============================================================
# FINAL GATE
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


all_artifacts_pass = bool(
    inventory[
        "artifact_verification_pass"
    ].all()
)


overall_pass = bool(

    exact_disk_inventory

    and
    all_artifacts_pass

    and
    software_version_consistent
)


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
                int(
                    inventory[
                        "artifact_verification_pass"
                    ].sum()
                ),

            "hash_match_count":
                int(
                    inventory[
                        "sha256_match"
                    ].sum()
                ),

            "feature_order_match_count":
                int(
                    inventory[
                        "feature_order_match"
                    ].sum()
                ),

            "tree_count_match_count":
                int(
                    inventory[
                        "tree_count_match"
                    ].sum()
                ),

            "selection_metadata_clean_count":
                int(
                    inventory[
                        "selection_metadata_clean"
                    ].sum()
                ),

            "software_version_count":
                len(
                    software_versions
                ),

            "software_versions":
                "|".join(
                    software_versions
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
# SAVE OUTPUTS
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
    "Expected RF model files:",
    12
)

print(
    "Actual RF model files:",
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
    int(
        inventory[
            "sha256_match"
        ].sum()
    ),
    "/ 12"
)

print(
    "Feature-order matches:",
    int(
        inventory[
            "feature_order_match"
        ].sum()
    ),
    "/ 12"
)

print(
    "500-tree artifacts:",
    int(
        inventory[
            "tree_count_match"
        ].sum()
    ),
    "/ 12"
)

print(
    "Selection metadata clean:",
    int(
        inventory[
            "selection_metadata_clean"
        ].sum()
    ),
    "/ 12"
)

print(
    "Artifact verification PASS:",
    int(
        inventory[
            "artifact_verification_pass"
        ].sum()
    ),
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
    software_versions
)

print(
    "Version consistent:",
    software_version_consistent
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
    "CHAPTER 4.4 RESULT"
)

print(
    "===================================="
)


print(
    "Overall RF artifact verification:",
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
        "Chapter 4.4 Random Forest "
        "artifact verification failed."
    )


print(
    "\nMODEL CHAPTER 4.4 COMPLETE."
)