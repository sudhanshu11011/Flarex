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

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "model_feature_contract.json"
)

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/"
    "cycle24_split_contract.json"
)

AUDIT_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_candidate_audit.csv"
)

IMPORTANCE_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_feature_importances_long.csv"
)

SIMILARITY_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_importance_similarity.csv"
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


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 4.3 — RANDOM FOREST CANDIDATE AUDIT"
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
        f"Missing RF model directory: {MODEL_DIR}"
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


if split_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 split contract is not FROZEN."
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
    "RF training protocol: FROZEN"
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
        "Expected exactly 12 RF candidates."
    )


if len(training_summary) != 12:

    raise RuntimeError(
        "Expected exactly 12 RF training-summary rows."
    )


if grid[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate IDs in RF grid."
    )


if training_summary[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate IDs in RF summary."
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
        "RF grid and training summary."
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
# AUDIT EACH RF MODEL
# ============================================================

audit_rows = []

importance_rows = []

importance_vectors = {}


print(
    "\n===================================="
)

print(
    "CANDIDATE STRUCTURE AUDIT"
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
            f"{candidate_id}: training-summary row "
            "not unique."
        )


    summary_row = summary_row.iloc[0]


    # --------------------------------------------------------
    # HASH
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # LOAD BUNDLE
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
    # CANDIDATE VALUES
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


    class_weight_text = str(
        grid_row[
            "class_weight"
        ]
    ).strip().lower()


    if class_weight_text == "none":

        expected_class_weight = None

    elif class_weight_text == "balanced_subsample":

        expected_class_weight = "balanced_subsample"

    else:

        raise RuntimeError(
            f"{candidate_id}: unsupported class_weight."
        )


    # --------------------------------------------------------
    # BUNDLE CONFORMANCE
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


    candidate_grid_hash_match = (
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


    criterion_match = (
        str(
            bundle.get(
                "criterion"
            )
        )
        ==
        EXPECTED_CRITERION
    )


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


    class_weight_match = (
        bundle.get(
            "class_weight"
        )
        ==
        expected_class_weight
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
    # TREE COUNT
    # --------------------------------------------------------

    fitted_tree_count = len(
        model.estimators_
    )


    tree_count_match = (
        fitted_tree_count
        ==
        EXPECTED_N_ESTIMATORS
    )


    # --------------------------------------------------------
    # TREE STRUCTURE
    # --------------------------------------------------------

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


    tree_nodes = np.asarray(
        [
            estimator.tree_.node_count

            for estimator
            in model.estimators_
        ],
        dtype=np.int64,
    )


    mean_depth = float(
        tree_depths.mean()
    )


    min_depth = int(
        tree_depths.min()
    )


    max_depth_observed = int(
        tree_depths.max()
    )


    mean_leaves = float(
        tree_leaves.mean()
    )


    min_leaves = int(
        tree_leaves.min()
    )


    max_leaves = int(
        tree_leaves.max()
    )


    mean_nodes = float(
        tree_nodes.mean()
    )


    min_nodes = int(
        tree_nodes.min()
    )


    max_nodes = int(
        tree_nodes.max()
    )


    if expected_max_depth is None:

        max_depth_constraint_pass = True

    else:

        max_depth_constraint_pass = bool(
            max_depth_observed
            <=
            expected_max_depth
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE AUDIT
    # --------------------------------------------------------

    importances = np.asarray(
        model.feature_importances_,
        dtype=np.float64,
    )


    if importances.shape != (
        25,
    ):

        raise RuntimeError(
            f"{candidate_id}: expected "
            "25 feature importances."
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


    importance_sum = float(
        importances.sum()
    )


    importance_sum_pass = bool(
        np.isclose(
            importance_sum,
            1.0,
            rtol=1e-10,
            atol=1e-10,
        )
    )


    if not importance_sum_pass:

        raise RuntimeError(
            f"{candidate_id}: feature "
            f"importance sum={importance_sum}"
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


    max_importance = float(
        np.max(
            importances
        )
    )


    mean_importance = float(
        np.mean(
            importances
        )
    )


    importance_vectors[
        candidate_id
    ] = importances.copy()


    # --------------------------------------------------------
    # LONG IMPORTANCE TABLE
    # --------------------------------------------------------

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

                "gini_importance":
                    float(
                        importances[
                            index
                        ]
                    ),

                "importance_rank":
                    rank_map[
                        index
                    ],

                "max_depth":
                    (
                        "None"
                        if expected_max_depth is None
                        else expected_max_depth
                    ),

                "min_samples_leaf":
                    expected_min_samples_leaf,

                "class_weight":
                    (
                        "None"
                        if expected_class_weight is None
                        else expected_class_weight
                    ),
            }
        )


    # --------------------------------------------------------
    # CLASSES + FEATURES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # WARNING PROVENANCE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FINAL CANDIDATE AUDIT
    # --------------------------------------------------------

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
        training_scope_match

        and
        train_hash_match

        and
        protocol_hash_match

        and
        candidate_grid_hash_match

        and
        n_estimators_match

        and
        criterion_match

        and
        max_depth_match

        and
        min_samples_leaf_match

        and
        max_features_match

        and
        bootstrap_match

        and
        class_weight_match

        and
        random_state_match

        and
        n_jobs_match

        and
        tree_count_match

        and
        max_depth_constraint_pass

        and
        importances_finite

        and
        importance_sum_pass

        and
        negative_importances
        ==
        0

        and
        classes_match

        and
        fitted_feature_count_match

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

            "training_scope_match":
                training_scope_match,

            "train_sha256_match":
                train_hash_match,

            "protocol_sha256_match":
                protocol_hash_match,

            "candidate_grid_sha256_match":
                candidate_grid_hash_match,

            "n_estimators_match":
                n_estimators_match,

            "criterion_match":
                criterion_match,

            "max_depth_match":
                max_depth_match,

            "min_samples_leaf_match":
                min_samples_leaf_match,

            "max_features_match":
                max_features_match,

            "bootstrap_match":
                bootstrap_match,

            "class_weight_match":
                class_weight_match,

            "random_state_match":
                random_state_match,

            "n_jobs_match":
                n_jobs_match,

            "fitted_tree_count":
                fitted_tree_count,

            "tree_count_match":
                tree_count_match,

            "expected_max_depth":
                (
                    "None"
                    if expected_max_depth is None
                    else expected_max_depth
                ),

            "mean_tree_depth":
                mean_depth,

            "min_tree_depth":
                min_depth,

            "max_tree_depth_observed":
                max_depth_observed,

            "max_depth_constraint_pass":
                max_depth_constraint_pass,

            "mean_tree_leaves":
                mean_leaves,

            "min_tree_leaves":
                min_leaves,

            "max_tree_leaves":
                max_leaves,

            "mean_tree_nodes":
                mean_nodes,

            "min_tree_nodes":
                min_nodes,

            "max_tree_nodes":
                max_nodes,

            "feature_importance_sum":
                importance_sum,

            "feature_importance_sum_pass":
                importance_sum_pass,

            "feature_importances_finite":
                importances_finite,

            "negative_feature_importances":
                negative_importances,

            "zero_feature_importances":
                zero_importances,

            "positive_feature_importances":
                positive_importances,

            "mean_feature_importance":
                mean_importance,

            "max_feature_importance":
                max_importance,

            "classes_match":
                classes_match,

            "fitted_feature_count_match":
                fitted_feature_count_match,

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
        f"trees={fitted_tree_count}, "
        f"depth_mean={mean_depth:.2f}, "
        f"depth_max={max_depth_observed}, "
        f"importance_sum={importance_sum:.6f}, "
        f"PASS={candidate_pass}"
    )


# ============================================================
# BUILD DATAFRAMES
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


importance_df = pd.DataFrame(
    importance_rows
)


if len(
    audit_df
) != 12:

    raise RuntimeError(
        "RF candidate audit does not "
        "contain 12 rows."
    )


if len(
    importance_df
) != (
    12
    *
    25
):

    raise RuntimeError(
        "RF importance table does not "
        "contain 300 rows."
    )


# ============================================================
# PAIRWISE FEATURE-IMPORTANCE SIMILARITY
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


        spearman_rank_correlation = float(
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
                    spearman_rank_correlation,
            }
        )


similarity_df = pd.DataFrame(
    similarity_rows
)


# ============================================================
# FINAL AUDIT SUMMARY
# ============================================================

pass_count = int(
    audit_df[
        "candidate_audit_pass"
    ].sum()
)


tree_count_pass = int(
    audit_df[
        "tree_count_match"
    ].sum()
)


depth_constraint_pass = int(
    audit_df[
        "max_depth_constraint_pass"
    ].sum()
)


importance_pass = int(
    (
        audit_df[
            "feature_importance_sum_pass"
        ]
        &
        audit_df[
            "feature_importances_finite"
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
    "Candidates with 500 trees:",
    tree_count_pass,
    "/ 12"
)

print(
    "Candidates respecting depth constraint:",
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
    "Maximum mean tree depth:",
    float(
        audit_df[
            "mean_tree_depth"
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
    "Random state match:",
    bool(
        audit_df[
            "random_state_match"
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
    "CHAPTER 4.3 RESULT"
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
    "Overall RF candidate audit:",
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
        "Chapter 4.3 Random Forest "
        "candidate audit failed."
    )


print(
    "\nMODEL CHAPTER 4.3 COMPLETE."
)