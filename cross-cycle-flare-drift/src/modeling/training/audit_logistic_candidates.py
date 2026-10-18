from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROTOCOL_FILE = Path(
    "artifacts/models/logistic/contracts/"
    "logistic_training_protocol.json"
)

GRID_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_candidate_grid.csv"
)

TRAINING_SUMMARY_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_training_summary.csv"
)

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "model_feature_contract.json"
)

PREPROCESSING_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "preprocessing_contract.json"
)

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/"
    "cycle24_split_contract.json"
)

MODEL_DIR = Path(
    "artifacts/models/logistic/models"
)

OUTPUT_DIR = Path(
    "artifacts/models/logistic/manifests"
)

AUDIT_FILE = (
    OUTPUT_DIR
    /
    "logistic_candidate_audit.csv"
)

COEFFICIENT_FILE = (
    OUTPUT_DIR
    /
    "logistic_coefficients_long.csv"
)

SIMILARITY_FILE = (
    OUTPUT_DIR
    /
    "logistic_coefficient_similarity.csv"
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


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 3.3 — LOGISTIC CANDIDATE AUDIT"
)

print(
    "===================================="
)


required_files = [
    PROTOCOL_FILE,
    GRID_FILE,
    TRAINING_SUMMARY_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
    SPLIT_CONTRACT_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
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


with PREPROCESSING_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    preprocessing_contract = json.load(file)


with SPLIT_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    split_contract = json.load(file)


if protocol.get(
    "status"
) != "FROZEN_BEFORE_TRAINING":

    raise RuntimeError(
        "LR protocol is not frozen."
    )


if feature_contract.get(
    "status"
) != "VERIFIED":

    raise RuntimeError(
        "Feature contract is not verified."
    )


if preprocessing_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Preprocessing contract is not frozen."
    )


if split_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Split contract is not frozen."
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
    "LR protocol: FROZEN"
)

print(
    "Feature contract: VERIFIED"
)

print(
    "Preprocessing contract: FROZEN"
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


summary = pd.read_csv(
    TRAINING_SUMMARY_FILE,
    keep_default_na=False,
)


if len(grid) != 10:

    raise RuntimeError(
        "Candidate grid does not contain 10 candidates."
    )


if len(summary) != 10:

    raise RuntimeError(
        "Training summary does not contain 10 candidates."
    )


if grid[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate IDs in candidate grid."
    )


if summary[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate IDs in training summary."
    )


if set(
    grid[
        "candidate_id"
    ]
) != set(
    summary[
        "candidate_id"
    ]
):

    raise RuntimeError(
        "Grid and training-summary candidate IDs differ."
    )


# ============================================================
# EXPECTED FROZEN VALUES
# ============================================================

fixed_parameters = protocol[
    "model_fixed_parameters"
]


expected_penalty = str(
    fixed_parameters[
        "penalty"
    ]
)


expected_solver = str(
    fixed_parameters[
        "solver"
    ]
)


expected_max_iter = int(
    fixed_parameters[
        "max_iter"
    ]
)


expected_tol = float(
    fixed_parameters[
        "tol"
    ]
)


expected_random_state = int(
    fixed_parameters[
        "random_state"
    ]
)


expected_train_hash = (
    split_contract[
        "train"
    ][
        "sha256"
    ]
)


expected_scaler_hash = (
    preprocessing_contract[
        "logistic_regression"
    ][
        "artifact_sha256"
    ]
)


# ============================================================
# AUDIT EACH MODEL
# ============================================================

audit_rows = []
coefficient_rows = []

coefficient_vectors = {}


print(
    "\n===================================="
)

print(
    "CANDIDATE ARTIFACT AUDIT"
)

print(
    "===================================="
)


for _, summary_row in summary.iterrows():

    candidate_id = str(
        summary_row[
            "candidate_id"
        ]
    )


    model_file = (
        MODEL_DIR
        /
        f"{candidate_id}.joblib"
    )


    if not model_file.exists():

        raise FileNotFoundError(
            f"{candidate_id}: model file missing."
        )


    actual_model_hash = sha256_file(
        model_file
    )


    expected_model_hash = str(
        summary_row[
            "model_sha256"
        ]
    )


    hash_match = (
        actual_model_hash
        ==
        expected_model_hash
    )


    if not hash_match:

        raise RuntimeError(
            f"{candidate_id}: model hash mismatch."
        )


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
    # BUNDLE / PROTOCOL CONFORMANCE
    # --------------------------------------------------------

    candidate_match = (
        bundle.get(
            "candidate_id"
        )
        ==
        candidate_id
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


    train_hash_match = (
        bundle.get(
            "train_sha256"
        )
        ==
        expected_train_hash
    )


    scaler_hash_match = (
        bundle.get(
            "scaler_sha256"
        )
        ==
        expected_scaler_hash
    )


    training_scope_match = (
        bundle.get(
            "training_scope"
        )
        ==
        "CYCLE24_TRAIN_ONLY"
    )


    penalty_match = (
        str(
            bundle.get(
                "penalty"
            )
        )
        ==
        expected_penalty
    )


    solver_match = (
        str(
            bundle.get(
                "solver"
            )
        )
        ==
        expected_solver
    )


    max_iter_match = (
        int(
            bundle.get(
                "max_iter"
            )
        )
        ==
        expected_max_iter
    )


    tol_match = bool(
        np.isclose(
            float(
                bundle.get(
                    "tol"
                )
            ),
            expected_tol,
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
        expected_random_state
    )


    # --------------------------------------------------------
    # CONVERGENCE
    # --------------------------------------------------------

    n_iter = int(
        np.max(
            model.n_iter_
        )
    )


    convergence_warning_count = int(
        summary_row[
            "convergence_warning_count"
        ]
    )


    future_warning_count = int(
        summary_row[
            "future_warning_count"
        ]
    )


    other_warning_count = int(
        summary_row[
            "other_warning_count"
        ]
    )


    converged = bool(
        convergence_warning_count
        ==
        0
        and
        n_iter
        <
        expected_max_iter
        and
        bool(
            summary_row[
                "converged"
            ]
        )
    )


    # --------------------------------------------------------
    # COEFFICIENT AUDIT
    # --------------------------------------------------------

    coefficients = (
        np.asarray(
            model.coef_,
            dtype=np.float64,
        )
        .reshape(-1)
    )


    intercept = (
        np.asarray(
            model.intercept_,
            dtype=np.float64,
        )
        .reshape(-1)
    )


    if coefficients.size != 25:

        raise RuntimeError(
            f"{candidate_id}: expected 25 coefficients."
        )


    coefficients_finite = bool(
        np.isfinite(
            coefficients
        ).all()
    )


    intercept_finite = bool(
        np.isfinite(
            intercept
        ).all()
    )


    if not coefficients_finite:

        raise RuntimeError(
            f"{candidate_id}: non-finite coefficients."
        )


    if not intercept_finite:

        raise RuntimeError(
            f"{candidate_id}: non-finite intercept."
        )


    l1_norm = float(
        np.linalg.norm(
            coefficients,
            ord=1,
        )
    )


    l2_norm = float(
        np.linalg.norm(
            coefficients,
            ord=2,
        )
    )


    max_abs_coefficient = float(
        np.max(
            np.abs(
                coefficients
            )
        )
    )


    mean_abs_coefficient = float(
        np.mean(
            np.abs(
                coefficients
            )
        )
    )


    exact_zero_coefficients = int(
        np.sum(
            coefficients
            ==
            0.0
        )
    )


    near_zero_coefficients = int(
        np.sum(
            np.abs(
                coefficients
            )
            <
            1e-12
        )
    )


    positive_coefficients = int(
        np.sum(
            coefficients
            >
            0
        )
    )


    negative_coefficients = int(
        np.sum(
            coefficients
            <
            0
        )
    )


    coefficient_vectors[
        candidate_id
    ] = coefficients.copy()


    # --------------------------------------------------------
    # LONG COEFFICIENT TABLE
    # --------------------------------------------------------

    for index, feature in enumerate(
        features
    ):

        coefficient_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "feature_index":
                    index,

                "feature":
                    feature,

                "coefficient":
                    float(
                        coefficients[
                            index
                        ]
                    ),

                "abs_coefficient":
                    float(
                        abs(
                            coefficients[
                                index
                            ]
                        )
                    ),

                "sign":
                    (
                        1
                        if coefficients[index] > 0
                        else
                        -1
                        if coefficients[index] < 0
                        else
                        0
                    ),

                "C":
                    float(
                        bundle.get(
                            "C"
                        )
                    ),

                "class_weight":
                    (
                        "None"
                        if bundle.get(
                            "class_weight"
                        )
                        is None
                        else str(
                            bundle.get(
                                "class_weight"
                            )
                        )
                    ),
            }
        )


    # --------------------------------------------------------
    # FINAL CANDIDATE PASS
    # --------------------------------------------------------

    candidate_pass = bool(

        hash_match

        and
        candidate_match

        and
        feature_order_match

        and
        feature_count_match

        and
        train_hash_match

        and
        scaler_hash_match

        and
        training_scope_match

        and
        penalty_match

        and
        solver_match

        and
        max_iter_match

        and
        tol_match

        and
        random_state_match

        and
        converged

        and
        coefficients_finite

        and
        intercept_finite

        and
        other_warning_count
        ==
        0
    )


    audit_rows.append(
        {
            "candidate_id":
                candidate_id,

            "model_sha256_match":
                hash_match,

            "feature_order_match":
                feature_order_match,

            "train_sha256_match":
                train_hash_match,

            "scaler_sha256_match":
                scaler_hash_match,

            "training_scope_match":
                training_scope_match,

            "penalty_match":
                penalty_match,

            "solver_match":
                solver_match,

            "max_iter_match":
                max_iter_match,

            "tol_match":
                tol_match,

            "random_state_match":
                random_state_match,

            "n_iter":
                n_iter,

            "convergence_warning_count":
                convergence_warning_count,

            "future_warning_count":
                future_warning_count,

            "other_warning_count":
                other_warning_count,

            "converged":
                converged,

            "coefficient_count":
                coefficients.size,

            "exact_zero_coefficients":
                exact_zero_coefficients,

            "near_zero_coefficients_lt_1e-12":
                near_zero_coefficients,

            "positive_coefficients":
                positive_coefficients,

            "negative_coefficients":
                negative_coefficients,

            "coefficient_l1_norm":
                l1_norm,

            "coefficient_l2_norm":
                l2_norm,

            "mean_abs_coefficient":
                mean_abs_coefficient,

            "max_abs_coefficient":
                max_abs_coefficient,

            "intercept":
                float(
                    intercept[
                        0
                    ]
                ),

            "coefficients_finite":
                coefficients_finite,

            "intercept_finite":
                intercept_finite,

            "candidate_audit_pass":
                candidate_pass,
        }
    )


    print(
        f"{candidate_id}: "
        f"converged={converged}, "
        f"coef_count={coefficients.size}, "
        f"L2={l2_norm:.6f}, "
        f"max|coef|={max_abs_coefficient:.6f}, "
        f"PASS={candidate_pass}"
    )


# ============================================================
# BUILD AUDIT TABLES
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


coefficient_df = pd.DataFrame(
    coefficient_rows
)


if len(
    audit_df
) != 10:

    raise RuntimeError(
        "Candidate audit does not contain 10 rows."
    )


if len(
    coefficient_df
) != 250:

    raise RuntimeError(
        "Coefficient table does not contain "
        "10 × 25 = 250 rows."
    )


# ============================================================
# PAIRWISE COEFFICIENT SIMILARITY
# ============================================================

similarity_rows = []


candidate_ids = list(
    coefficient_vectors.keys()
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


        left = coefficient_vectors[
            left_id
        ]

        right = coefficient_vectors[
            right_id
        ]


        cosine = cosine_similarity(
            left,
            right,
        )


        sign_agreement = float(
            np.mean(
                np.sign(
                    left
                )
                ==
                np.sign(
                    right
                )
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

                "sign_agreement_fraction":
                    sign_agreement,
            }
        )


similarity_df = pd.DataFrame(
    similarity_rows
)


# ============================================================
# FINAL AUDIT GATE
# ============================================================

all_candidates_pass = bool(
    audit_df[
        "candidate_audit_pass"
    ].all()
)


all_converged = bool(
    audit_df[
        "converged"
    ].all()
)


total_convergence_warnings = int(
    audit_df[
        "convergence_warning_count"
    ].sum()
)


total_future_warnings = int(
    audit_df[
        "future_warning_count"
    ].sum()
)


total_unexpected_warnings = int(
    audit_df[
        "other_warning_count"
    ].sum()
)


all_coefficients_finite = bool(
    audit_df[
        "coefficients_finite"
    ].all()
)


all_intercepts_finite = bool(
    audit_df[
        "intercept_finite"
    ].all()
)


overall_pass = bool(

    all_candidates_pass

    and
    all_converged

    and
    total_convergence_warnings
    ==
    0

    and
    total_unexpected_warnings
    ==
    0

    and
    all_coefficients_finite

    and
    all_intercepts_finite
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


audit_df.to_csv(
    AUDIT_FILE,
    index=False,
)


coefficient_df.to_csv(
    COEFFICIENT_FILE,
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
    "CONVERGENCE AUDIT"
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
    "Candidates converged:",
    int(
        audit_df[
            "converged"
        ].sum()
    )
)

print(
    "Convergence warnings:",
    total_convergence_warnings
)

print(
    "FutureWarnings:",
    total_future_warnings
)

print(
    "Unexpected warnings:",
    total_unexpected_warnings
)


print(
    "\n===================================="
)

print(
    "COEFFICIENT AUDIT"
)

print(
    "===================================="
)


print(
    "Expected coefficients/model: 25"
)

print(
    "Models with 25 coefficients:",
    int(
        audit_df[
            "coefficient_count"
        ]
        .eq(25)
        .sum()
    ),
    "/ 10"
)

print(
    "All coefficients finite:",
    all_coefficients_finite
)

print(
    "All intercepts finite:",
    all_intercepts_finite
)

print(
    "Total exact-zero coefficients:",
    int(
        audit_df[
            "exact_zero_coefficients"
        ].sum()
    )
)


print(
    "\nCoefficient L2 norm range:"
)

print(
    float(
        audit_df[
            "coefficient_l2_norm"
        ].min()
    ),
    "to",
    float(
        audit_df[
            "coefficient_l2_norm"
        ].max()
    )
)


print(
    "Maximum absolute coefficient:"
)

print(
    float(
        audit_df[
            "max_abs_coefficient"
        ].max()
    )
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
    "Feature order match:",
    bool(
        audit_df[
            "feature_order_match"
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
    "Scaler hashes match:",
    bool(
        audit_df[
            "scaler_sha256_match"
        ].all()
    )
)

print(
    "random_state match:",
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
    "Training artifacts inspected: True"
)

print(
    "VALIDATION read: False"
)

print(
    "HOLDOUT read: False"
)

print(
    "Cycle25 read: False"
)

print(
    "Model selection performed: False"
)


print(
    "\n===================================="
)

print(
    "CHAPTER 3.3 RESULT"
)

print(
    "===================================="
)


print(
    "Overall candidate audit:",
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
    AUDIT_FILE
)

print(
    COEFFICIENT_FILE
)

print(
    SIMILARITY_FILE
)


if not overall_pass:

    raise RuntimeError(
        "Chapter 3.3 Logistic Regression "
        "candidate audit failed."
    )


print(
    "\nMODEL CHAPTER 3.3 COMPLETE."
)