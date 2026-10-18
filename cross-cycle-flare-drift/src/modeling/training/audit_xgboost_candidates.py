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

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "model_feature_contract.json"
)

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/"
    "cycle24_split_contract.json"
)

AUDIT_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_candidate_audit.csv"
)

IMPORTANCE_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_feature_importances_long.csv"
)

SIMILARITY_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_importance_similarity.csv"
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


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:

        return np.nan

    return float(
        np.dot(a, b)
        /
        denominator
    )


def audit_tree_json(tree_json):

    tree = json.loads(
        tree_json
    )

    node_count = 0
    leaf_count = 0
    max_depth = 0


    def walk(node, depth):

        nonlocal node_count
        nonlocal leaf_count
        nonlocal max_depth

        node_count += 1

        max_depth = max(
            max_depth,
            depth,
        )

        children = node.get(
            "children"
        )

        if not children:

            leaf_count += 1
            return

        for child in children:

            walk(
                child,
                depth + 1,
            )


    walk(
        tree,
        0,
    )

    return (
        node_count,
        leaf_count,
        max_depth,
    )


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 5.3 — XGBOOST CANDIDATE AUDIT"
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
    FEATURE_CONTRACT_FILE,
    SPLIT_CONTRACT_FILE,
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


with SPLIT_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    split_contract = json.load(file)


# ============================================================
# VERIFY CONTRACT STATES
# ============================================================

if protocol.get(
    "status"
) != "FROZEN_BEFORE_TRAINING":

    raise RuntimeError(
        "XGBoost training protocol is not frozen."
    )


if feature_contract.get(
    "status"
) != "VERIFIED":

    raise RuntimeError(
        "Feature contract is not VERIFIED."
    )


if split_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 split contract is not FROZEN."
    )


if protocol.get(
    "model_family"
) != "XGBClassifier":

    raise RuntimeError(
        "Unexpected XGBoost model family."
    )


features = feature_contract.get(
    "feature_order",
    []
)


if len(features) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


expected_train_hash = (
    split_contract[
        "train"
    ][
        "sha256"
    ]
)


protocol_hash = sha256_file(
    PROTOCOL_FILE
)


grid_hash = sha256_file(
    GRID_FILE
)


print(
    "XGBoost training protocol: FROZEN"
)

print(
    "Feature contract: VERIFIED"
)

print(
    "Cycle24 split: FROZEN"
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


if grid[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate IDs in XGBoost grid."
    )


if training_summary[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate IDs in training summary."
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

EXPECTED_RANDOM_STATE = int(
    fixed[
        "random_state"
    ]
)


# ============================================================
# AUDIT
# ============================================================

audit_rows = []

importance_rows = []

importance_vectors = {}


print(
    "\n===================================="
)

print(
    "CANDIDATE TREE / IMPORTANCE AUDIT"
)

print(
    "===================================="
)


for _, grid_row in grid.iterrows():

    candidate_id = str(
        grid_row[
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
            f"{candidate_id}: model artifact missing."
        )


    summary_row = training_summary[
        training_summary[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(summary_row) != 1:

        raise RuntimeError(
            f"{candidate_id}: training-summary "
            "row is not unique."
        )


    summary_row = summary_row.iloc[
        0
    ]


    # ========================================================
    # HASH VERIFICATION
    # ========================================================

    actual_hash = sha256_file(
        model_file
    )


    expected_hash = str(
        summary_row[
            "model_sha256"
        ]
    )


    model_hash_match = (
        actual_hash
        ==
        expected_hash
    )


    if not model_hash_match:

        raise RuntimeError(
            f"{candidate_id}: model hash mismatch."
        )


    # ========================================================
    # LOAD MODEL BUNDLE
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


    booster = model.get_booster()


    if booster is None:

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


    train_hash_match = (
        bundle.get(
            "train_sha256"
        )
        ==
        expected_train_hash
    )


    protocol_hash_match = (
        bundle.get(
            "protocol_sha256"
        )
        ==
        protocol_hash
    )


    grid_hash_match = (
        bundle.get(
            "candidate_grid_sha256"
        )
        ==
        grid_hash
    )


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


    random_state_match = (
        int(
            bundle.get(
                "random_state"
            )
        )
        ==
        EXPECTED_RANDOM_STATE
    )


    learning_rate_match = bool(
        np.isclose(
            float(
                bundle.get(
                    "learning_rate"
                )
            ),
            expected_learning_rate,
            rtol=0.0,
            atol=0.0,
        )
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


    scale_pos_weight_match = bool(
        np.isclose(
            float(
                bundle.get(
                    "scale_pos_weight"
                )
            ),
            expected_scale_pos_weight,
            rtol=0.0,
            atol=1e-12,
        )
    )


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
    # BOOSTING ROUND AUDIT
    # ========================================================

    boosted_rounds = int(
        booster.num_boosted_rounds()
    )


    boosted_rounds_match = (
        boosted_rounds
        ==
        EXPECTED_N_ESTIMATORS
    )


    tree_dump = booster.get_dump(
        dump_format="json"
    )


    dumped_tree_count = len(
        tree_dump
    )


    dumped_tree_count_match = (
        dumped_tree_count
        ==
        EXPECTED_N_ESTIMATORS
    )


    # ========================================================
    # TREE STRUCTURE AUDIT
    # ========================================================

    tree_node_counts = []

    tree_leaf_counts = []

    tree_depths = []


    for tree_json in tree_dump:

        (
            node_count,
            leaf_count,
            observed_depth,
        ) = audit_tree_json(
            tree_json
        )


        tree_node_counts.append(
            node_count
        )

        tree_leaf_counts.append(
            leaf_count
        )

        tree_depths.append(
            observed_depth
        )


    tree_node_counts = np.asarray(
        tree_node_counts,
        dtype=np.int64,
    )


    tree_leaf_counts = np.asarray(
        tree_leaf_counts,
        dtype=np.int64,
    )


    tree_depths = np.asarray(
        tree_depths,
        dtype=np.int64,
    )


    mean_tree_depth = float(
        tree_depths.mean()
    )


    min_tree_depth = int(
        tree_depths.min()
    )


    max_tree_depth_observed = int(
        tree_depths.max()
    )


    mean_tree_nodes = float(
        tree_node_counts.mean()
    )


    min_tree_nodes = int(
        tree_node_counts.min()
    )


    max_tree_nodes = int(
        tree_node_counts.max()
    )


    mean_tree_leaves = float(
        tree_leaf_counts.mean()
    )


    min_tree_leaves = int(
        tree_leaf_counts.min()
    )


    max_tree_leaves = int(
        tree_leaf_counts.max()
    )


    depth_constraint_pass = bool(
        max_tree_depth_observed
        <=
        expected_max_depth
    )


    # ========================================================
    # FEATURE IMPORTANCE
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


    if not importance_shape_match:

        raise RuntimeError(
            f"{candidate_id}: feature importance "
            "vector length is not 25."
        )


    importances_finite = bool(
        np.isfinite(
            importances
        ).all()
    )


    if not importances_finite:

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "feature importances."
        )


    negative_importances = int(
        np.sum(
            importances
            <
            0.0
        )
    )


    zero_importances = int(
        np.sum(
            importances
            ==
            0.0
        )
    )


    positive_importances = int(
        np.sum(
            importances
            >
            0.0
        )
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


    if not importance_sum_valid:

        raise RuntimeError(
            f"{candidate_id}: invalid importance "
            f"sum {importance_sum}"
        )


    importance_vectors[
        candidate_id
    ] = importances.copy()


    # ========================================================
    # LONG IMPORTANCE TABLE
    # ========================================================

    descending_indices = np.argsort(
        -importances
    )


    rank_map = {
        int(feature_index):
            int(rank + 1)

        for rank, feature_index
        in enumerate(
            descending_indices
        )
    }


    for index, feature in enumerate(
        features
    ):

        importance_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "feature_index":
                    index,

                "feature":
                    feature,

                "xgboost_builtin_importance":
                    float(
                        importances[
                            index
                        ]
                    ),

                "importance_rank":
                    rank_map[
                        index
                    ],

                "learning_rate":
                    expected_learning_rate,

                "max_depth":
                    expected_max_depth,

                "scale_pos_weight":
                    expected_scale_pos_weight,
            }
        )


    # ========================================================
    # FITTED MODEL BASIC INTEGRITY
    # ========================================================

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
            booster.num_features()
        )
        ==
        25
    )


    warning_count = int(
        summary_row[
            "warning_count"
        ]
    )


    warnings_clean = (
        warning_count
        ==
        0
    )


    serialization_match = (
        str(
            summary_row[
                "serialization_probability_match"
            ]
        )
        .strip()
        .lower()
        ==
        "true"
    )


    # ========================================================
    # FINAL CANDIDATE PASS
    # ========================================================

    candidate_pass = bool(

        model_hash_match

        and
        candidate_id_match

        and
        model_family_match

        and
        feature_order_match

        and
        feature_count_match

        and
        preprocessing_match

        and
        training_scope_match

        and
        train_hash_match

        and
        protocol_hash_match

        and
        grid_hash_match

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
        random_state_match

        and
        learning_rate_match

        and
        max_depth_match

        and
        scale_pos_weight_match

        and
        early_stopping_clean

        and
        validation_clean

        and
        boosted_rounds_match

        and
        dumped_tree_count_match

        and
        depth_constraint_pass

        and
        importance_shape_match

        and
        importances_finite

        and
        importance_sum_valid

        and
        negative_importances
        ==
        0

        and
        classes_match

        and
        fitted_feature_count_match

        and
        booster_feature_count_match

        and
        warnings_clean

        and
        serialization_match
    )


    audit_rows.append(
        {
            "candidate_id":
                candidate_id,

            "model_sha256_match":
                model_hash_match,

            "candidate_id_match":
                candidate_id_match,

            "model_family_match":
                model_family_match,

            "feature_order_match":
                feature_order_match,

            "feature_count_match":
                feature_count_match,

            "preprocessing_match":
                preprocessing_match,

            "training_scope_match":
                training_scope_match,

            "train_sha256_match":
                train_hash_match,

            "protocol_sha256_match":
                protocol_hash_match,

            "candidate_grid_sha256_match":
                grid_hash_match,

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

            "random_state_match":
                random_state_match,

            "learning_rate_match":
                learning_rate_match,

            "max_depth_match":
                max_depth_match,

            "scale_pos_weight_match":
                scale_pos_weight_match,

            "early_stopping_clean":
                early_stopping_clean,

            "validation_clean":
                validation_clean,

            "boosted_rounds":
                boosted_rounds,

            "boosted_rounds_match":
                boosted_rounds_match,

            "dumped_tree_count":
                dumped_tree_count,

            "dumped_tree_count_match":
                dumped_tree_count_match,

            "configured_max_depth":
                expected_max_depth,

            "mean_tree_depth":
                mean_tree_depth,

            "min_tree_depth":
                min_tree_depth,

            "max_tree_depth_observed":
                max_tree_depth_observed,

            "depth_constraint_pass":
                depth_constraint_pass,

            "mean_tree_nodes":
                mean_tree_nodes,

            "min_tree_nodes":
                min_tree_nodes,

            "max_tree_nodes":
                max_tree_nodes,

            "mean_tree_leaves":
                mean_tree_leaves,

            "min_tree_leaves":
                min_tree_leaves,

            "max_tree_leaves":
                max_tree_leaves,

            "feature_importance_sum":
                importance_sum,

            "feature_importance_sum_valid":
                importance_sum_valid,

            "feature_importances_finite":
                importances_finite,

            "negative_feature_importances":
                negative_importances,

            "zero_feature_importances":
                zero_importances,

            "positive_feature_importances":
                positive_importances,

            "max_feature_importance":
                float(
                    importances.max()
                ),

            "classes_match":
                classes_match,

            "fitted_feature_count_match":
                fitted_feature_count_match,

            "booster_feature_count_match":
                booster_feature_count_match,

            "warning_count":
                warning_count,

            "warnings_clean":
                warnings_clean,

            "serialization_probability_match":
                serialization_match,

            "candidate_audit_pass":
                candidate_pass,
        }
    )


    print(
        f"{candidate_id}: "
        f"rounds={boosted_rounds}, "
        f"trees={dumped_tree_count}, "
        f"depth_mean={mean_tree_depth:.2f}, "
        f"depth_max={max_tree_depth_observed}, "
        f"importance_sum={importance_sum:.6f}, "
        f"PASS={candidate_pass}"
    )


# ============================================================
# BUILD OUTPUT TABLES
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


importance_df = pd.DataFrame(
    importance_rows
)


if len(audit_df) != 12:

    raise RuntimeError(
        "XGBoost candidate audit does "
        "not contain 12 rows."
    )


if len(importance_df) != (
    12
    *
    25
):

    raise RuntimeError(
        "XGBoost importance table does "
        "not contain 300 rows."
    )


# ============================================================
# PAIRWISE IMPORTANCE SIMILARITY
# ============================================================

similarity_rows = []


candidate_ids = list(
    importance_vectors.keys()
)


for i in range(
    len(
        candidate_ids
    )
):

    for j in range(
        i + 1,
        len(
            candidate_ids
        ),
    ):

        left_id = candidate_ids[
            i
        ]

        right_id = candidate_ids[
            j
        ]


        left = importance_vectors[
            left_id
        ]

        right = importance_vectors[
            right_id
        ]


        cosine = cosine_similarity(
            left,
            right,
        )


        left_rank = pd.Series(
            left
        ).rank(
            method="average",
            ascending=False,
        )


        right_rank = pd.Series(
            right
        ).rank(
            method="average",
            ascending=False,
        )


        rank_correlation = float(
            left_rank.corr(
                right_rank,
                method="pearson",
            )
        )


        similarity_rows.append(
            {
                "candidate_a":
                    left_id,

                "candidate_b":
                    right_id,

                "cosine_similarity":
                    cosine,

                "importance_rank_spearman":
                    rank_correlation,
            }
        )


similarity_df = pd.DataFrame(
    similarity_rows
)


# ============================================================
# FINAL SUMMARY
# ============================================================

pass_count = int(
    audit_df[
        "candidate_audit_pass"
    ].sum()
)


round_count_pass = int(
    audit_df[
        "boosted_rounds_match"
    ].sum()
)


tree_count_pass = int(
    audit_df[
        "dumped_tree_count_match"
    ].sum()
)


depth_constraint_pass = int(
    audit_df[
        "depth_constraint_pass"
    ].sum()
)


importance_pass = int(
    (
        audit_df[
            "feature_importances_finite"
        ]
        &
        audit_df[
            "feature_importance_sum_valid"
        ]
    ).sum()
)


total_negative_importances = int(
    audit_df[
        "negative_feature_importances"
    ].sum()
)


total_zero_importances = int(
    audit_df[
        "zero_feature_importances"
    ].sum()
)


all_candidates_pass = (
    pass_count
    ==
    12
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


importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False,
)


similarity_df.to_csv(
    SIMILARITY_FILE,
    index=False,
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "TREE STRUCTURE AUDIT"
)

print(
    "===================================="
)


print(
    "Candidates audited:",
    len(
        audit_df
    )
)

print(
    "Candidates with 500 boosted rounds:",
    round_count_pass,
    "/ 12"
)

print(
    "Candidates with 500 dumped trees:",
    tree_count_pass,
    "/ 12"
)

print(
    "Candidates respecting max_depth:",
    depth_constraint_pass,
    "/ 12"
)

print(
    "Maximum observed tree depth:",
    int(
        audit_df[
            "max_tree_depth_observed"
        ].max()
    )
)


print(
    "\n===================================="
)

print(
    "FEATURE IMPORTANCE AUDIT"
)

print(
    "===================================="
)


print(
    "Features per candidate:",
    25
)

print(
    "Candidates with valid importance vectors:",
    importance_pass,
    "/ 12"
)

print(
    "Total negative importances:",
    total_negative_importances
)

print(
    "Total exact-zero importances:",
    total_zero_importances
)

print(
    "Importance interpretation:",
    "DIAGNOSTIC_ONLY"
)

print(
    "Feature selection performed:",
    False
)


print(
    "\n===================================="
)

print(
    "ARTIFACT / PROTOCOL AUDIT"
)

print(
    "===================================="
)


print(
    "Model hashes match:",
    bool(
        audit_df[
            "model_sha256_match"
        ].all()
    )
)

print(
    "TRAIN hashes match:",
    bool(
        audit_df[
            "train_sha256_match"
        ].all()
    )
)

print(
    "Protocol hashes match:",
    bool(
        audit_df[
            "protocol_sha256_match"
        ].all()
    )
)

print(
    "Candidate-grid hashes match:",
    bool(
        audit_df[
            "candidate_grid_sha256_match"
        ].all()
    )
)

print(
    "Feature order match:",
    bool(
        audit_df[
            "feature_order_match"
        ].all()
    )
)

print(
    "Early stopping clean:",
    bool(
        audit_df[
            "early_stopping_clean"
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
    "CHAPTER 5.3 RESULT"
)

print(
    "===================================="
)


print(
    "Candidates passing audit:",
    pass_count,
    "/ 12"
)

print(
    "Overall XGBoost candidate audit:",
    (
        "PASS"
        if all_candidates_pass
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
    IMPORTANCE_FILE
)

print(
    SIMILARITY_FILE
)


if not all_candidates_pass:

    raise RuntimeError(
        "Chapter 5.3 XGBoost "
        "candidate audit failed."
    )


print(
    "\nMODEL CHAPTER 5.3 COMPLETE."
)