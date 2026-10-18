from pathlib import Path
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


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

PREPROCESSING_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/preprocessing_contract.json"
)

SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/logistic_standard_scaler.joblib"
)

PROTOCOL_FILE = Path(
    "artifacts/models/logistic/contracts/"
    "logistic_training_protocol.json"
)

GRID_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_candidate_grid.csv"
)

MODEL_DIR = Path(
    "artifacts/models/logistic/models"
)

SUMMARY_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_training_summary.csv"
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


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 3.2 — TRAIN LOGISTIC CANDIDATES"
)

print(
    "===================================="
)


# ============================================================
# VERIFY REQUIRED INPUTS
# ============================================================

required_files = [
    TRAIN_FILE,
    SPLIT_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
    SCALER_FILE,
    PROTOCOL_FILE,
    GRID_FILE,
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


with PREPROCESSING_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    preprocessing_contract = json.load(file)


with PROTOCOL_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    protocol = json.load(file)


# ============================================================
# VERIFY CONTRACT STATES
# ============================================================

if split_contract.get("status") != "FROZEN":

    raise RuntimeError(
        "Cycle24 split contract is not FROZEN."
    )


if feature_contract.get("status") != "VERIFIED":

    raise RuntimeError(
        "Feature contract is not VERIFIED."
    )


if preprocessing_contract.get("status") != "FROZEN":

    raise RuntimeError(
        "Preprocessing contract is not FROZEN."
    )


if protocol.get(
    "status"
) != "FROZEN_BEFORE_TRAINING":

    raise RuntimeError(
        "Logistic training protocol is not "
        "FROZEN_BEFORE_TRAINING."
    )


features = feature_contract.get(
    "feature_order",
    []
)


if len(features) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


if protocol.get(
    "feature_order"
) != features:

    raise RuntimeError(
        "Training protocol feature order differs "
        "from frozen feature contract."
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
    "LR training protocol: FROZEN"
)


# ============================================================
# VERIFY TRAIN FILE HASH
# ============================================================

expected_train_hash = (
    split_contract[
        "train"
    ][
        "sha256"
    ]
)


actual_train_hash = sha256_file(
    TRAIN_FILE
)


if (
    actual_train_hash
    !=
    expected_train_hash
):

    raise RuntimeError(
        "TRAIN dataset differs from "
        "frozen split contract."
    )


# ============================================================
# VERIFY SCALER HASH
# ============================================================

expected_scaler_hash = (
    preprocessing_contract[
        "logistic_regression"
    ][
        "artifact_sha256"
    ]
)


actual_scaler_hash = sha256_file(
    SCALER_FILE
)


if (
    actual_scaler_hash
    !=
    expected_scaler_hash
):

    raise RuntimeError(
        "StandardScaler artifact differs from "
        "frozen preprocessing contract."
    )


print(
    "TRAIN hash: PASS"
)

print(
    "Scaler hash: PASS"
)


# ============================================================
# LOAD TRAIN ONLY
# ============================================================

train = pd.read_parquet(
    TRAIN_FILE
)


missing_features = [

    feature

    for feature in features

    if feature not in train.columns
]


if missing_features:

    raise RuntimeError(
        f"TRAIN missing frozen features: "
        f"{missing_features}"
    )


if "target" not in train.columns:

    raise RuntimeError(
        "TRAIN dataset missing target column."
    )


X_train_df = (
    train[
        features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
)


if X_train_df.isna().any().any():

    raise RuntimeError(
        "TRAIN contains missing/non-numeric "
        "feature values."
    )


X_train = X_train_df.to_numpy(
    dtype=np.float64
)


y_train = pd.to_numeric(
    train[
        "target"
    ],
    errors="raise",
).to_numpy(
    dtype=np.int64
)


# ============================================================
# VERIFY TRAIN STATE
# ============================================================

if X_train.shape != (
    62064,
    25,
):

    raise RuntimeError(
        f"Unexpected TRAIN feature shape: "
        f"{X_train.shape}"
    )


if len(y_train) != 62064:

    raise RuntimeError(
        "Unexpected TRAIN target length."
    )


if not np.isfinite(
    X_train
).all():

    raise RuntimeError(
        "TRAIN contains non-finite features."
    )


unique_targets = set(
    np.unique(
        y_train
    ).tolist()
)


if unique_targets != {
    0,
    1,
}:

    raise RuntimeError(
        f"TRAIN target is not binary {{0,1}}. "
        f"Found: {unique_targets}"
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


if train_positive != 2420:

    raise RuntimeError(
        "Unexpected TRAIN positive count."
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
# LOAD FROZEN SCALER
# ============================================================

scaler_bundle = joblib.load(
    SCALER_FILE
)


if (
    scaler_bundle.get(
        "feature_order"
    )
    !=
    features
):

    raise RuntimeError(
        "Scaler feature order mismatch."
    )


if scaler_bundle.get(
    "fit_scope"
) != "CYCLE24_TRAIN_ONLY":

    raise RuntimeError(
        "Scaler was not fit on "
        "Cycle24 TRAIN only."
    )


if int(
    scaler_bundle.get(
        "train_rows"
    )
) != 62064:

    raise RuntimeError(
        "Scaler TRAIN-row count mismatch."
    )


if (
    scaler_bundle.get(
        "train_sha256"
    )
    !=
    actual_train_hash
):

    raise RuntimeError(
        "Scaler was fitted from a different "
        "TRAIN dataset hash."
    )


scaler = scaler_bundle.get(
    "scaler"
)


if scaler is None:

    raise RuntimeError(
        "Scaler object missing."
    )


if int(
    scaler.n_features_in_
) != 25:

    raise RuntimeError(
        "Scaler feature count mismatch."
    )


# ============================================================
# TRANSFORM TRAIN USING FROZEN SCALER
# ============================================================

X_scaled = scaler.transform(
    X_train
)


if X_scaled.shape != (
    62064,
    25,
):

    raise RuntimeError(
        "Scaled TRAIN shape mismatch."
    )


if not np.isfinite(
    X_scaled
).all():

    raise RuntimeError(
        "Scaled TRAIN contains "
        "non-finite values."
    )


print(
    "Frozen scaler transform: PASS"
)


# ============================================================
# LOAD CANDIDATE GRID
# ============================================================

#
# IMPORTANT:
# keep_default_na=False prevents the literal text
# "None" from becoming NaN.
#

grid = pd.read_csv(
    GRID_FILE,
    keep_default_na=False,
)


if len(grid) != 10:

    raise RuntimeError(
        "Expected exactly 10 LR candidates."
    )


if grid[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate LR candidate IDs."
    )


expected_candidates = int(
    protocol[
        "candidate_search_space"
    ][
        "candidate_count"
    ]
)


if (
    expected_candidates
    !=
    len(grid)
):

    raise RuntimeError(
        "Candidate grid count differs "
        "from frozen training protocol."
    )


# ============================================================
# READ FIXED PROTOCOL VALUES
# ============================================================

protocol_fixed = protocol[
    "model_fixed_parameters"
]


expected_penalty = str(
    protocol_fixed[
        "penalty"
    ]
)


expected_solver = str(
    protocol_fixed[
        "solver"
    ]
)


expected_max_iter = int(
    protocol_fixed[
        "max_iter"
    ]
)


expected_tol = float(
    protocol_fixed[
        "tol"
    ]
)


expected_random_state = int(
    protocol_fixed[
        "random_state"
    ]
)


expected_fit_intercept = bool(
    protocol_fixed[
        "fit_intercept"
    ]
)


protocol_c_values = sorted(
    [
        float(value)

        for value
        in protocol[
            "candidate_search_space"
        ][
            "C"
        ]
    ]
)


grid_c_values = sorted(
    grid[
        "C"
    ]
    .astype(float)
    .unique()
    .tolist()
)


if grid_c_values != protocol_c_values:

    raise RuntimeError(
        "Candidate C values differ "
        "from frozen protocol."
    )


# ============================================================
# TRAIN ALL CANDIDATES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


training_rows = []


print(
    "\n===================================="
)

print(
    "TRAINING CANDIDATES"
)

print(
    "===================================="
)


for _, row in grid.iterrows():

    # --------------------------------------------------------
    # CANDIDATE ID
    # --------------------------------------------------------

    candidate_id = str(
        row[
            "candidate_id"
        ]
    ).strip()


    # --------------------------------------------------------
    # READ CANDIDATE PARAMETERS
    # --------------------------------------------------------

    c_value = float(
        row[
            "C"
        ]
    )


    class_weight_raw = str(
        row[
            "class_weight"
        ]
    ).strip()


    if (
        class_weight_raw.lower()
        ==
        "none"
    ):

        class_weight = None

    elif (
        class_weight_raw.lower()
        ==
        "balanced"
    ):

        class_weight = "balanced"

    else:

        raise RuntimeError(
            f"{candidate_id}: unsupported "
            f"class_weight "
            f"{class_weight_raw}"
        )


    penalty = str(
        row[
            "penalty"
        ]
    ).strip()


    solver = str(
        row[
            "solver"
        ]
    ).strip()


    max_iter = int(
        row[
            "max_iter"
        ]
    )


    tolerance = float(
        row[
            "tol"
        ]
    )


    fit_intercept = parse_bool(
        row[
            "fit_intercept"
        ]
    )


    random_state = int(
        row[
            "random_state"
        ]
    )


    # ========================================================
    # VERIFY AGAINST FROZEN PROTOCOL
    # ========================================================

    if penalty != expected_penalty:

        raise RuntimeError(
            f"{candidate_id}: penalty differs "
            "from frozen protocol."
        )


    if solver != expected_solver:

        raise RuntimeError(
            f"{candidate_id}: solver differs "
            "from frozen protocol."
        )


    if max_iter != expected_max_iter:

        raise RuntimeError(
            f"{candidate_id}: max_iter differs "
            "from frozen protocol."
        )


    if not np.isclose(
        tolerance,
        expected_tol,
        rtol=0.0,
        atol=0.0,
    ):

        raise RuntimeError(
            f"{candidate_id}: tolerance differs "
            "from frozen protocol."
        )


    if (
        fit_intercept
        !=
        expected_fit_intercept
    ):

        raise RuntimeError(
            f"{candidate_id}: fit_intercept "
            "differs from frozen protocol."
        )


    if (
        random_state
        !=
        expected_random_state
    ):

        raise RuntimeError(
            f"{candidate_id}: random_state "
            "differs from frozen protocol."
        )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        penalty=penalty,
        solver=solver,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tolerance,
        random_state=random_state,
    )


    # ========================================================
    # FIT MODEL + CAPTURE WARNINGS
    # ========================================================

    with warnings.catch_warnings(
        record=True
    ) as captured_warnings:

        warnings.simplefilter(
            "always"
        )


        model.fit(
            X_scaled,
            y_train,
        )


    # ========================================================
    # CLASSIFY WARNINGS
    # ========================================================

    convergence_warnings = [

        warning

        for warning
        in captured_warnings

        if issubclass(
            warning.category,
            ConvergenceWarning,
        )
    ]


    future_warnings = [

        warning

        for warning
        in captured_warnings

        if issubclass(
            warning.category,
            FutureWarning,
        )
    ]


    other_warnings = [

        warning

        for warning
        in captured_warnings

        if (
            not issubclass(
                warning.category,
                ConvergenceWarning,
            )
            and
            not issubclass(
                warning.category,
                FutureWarning,
            )
        )
    ]


    # ========================================================
    # CONVERGENCE / MODEL INTEGRITY
    # ========================================================

    n_iter = int(
        np.max(
            model.n_iter_
        )
    )


    coefficient_count = int(
        model.coef_.size
    )


    if coefficient_count != 25:

        raise RuntimeError(
            f"{candidate_id}: coefficient "
            "count is not 25."
        )


    if model.coef_.shape != (
        1,
        25,
    ):

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            f"coefficient shape "
            f"{model.coef_.shape}"
        )


    if not np.isfinite(
        model.coef_
    ).all():

        raise RuntimeError(
            f"{candidate_id}: "
            "non-finite coefficients."
        )


    if not np.isfinite(
        model.intercept_
    ).all():

        raise RuntimeError(
            f"{candidate_id}: "
            "non-finite intercept."
        )


    if not np.array_equal(
        model.classes_,
        np.array(
            [0, 1]
        ),
    ):

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            f"classes {model.classes_}"
        )


    # ========================================================
    # TRAIN-PROBABILITY SANITY CHECK
    #
    # This is NOT model selection.
    # It only verifies valid output from fitted candidate.
    # ========================================================

    train_probabilities = (
        model.predict_proba(
            X_scaled
        )[
            :,
            1
        ]
    )


    if not np.isfinite(
        train_probabilities
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "training probabilities."
        )


    probability_min = float(
        train_probabilities.min()
    )


    probability_max = float(
        train_probabilities.max()
    )


    if (
        probability_min
        <
        0.0
        or
        probability_max
        >
        1.0
    ):

        raise RuntimeError(
            f"{candidate_id}: probability "
            "outside [0,1]."
        )


    # ========================================================
    # CONVERGENCE DECISION
    # ========================================================

    converged = (
        len(
            convergence_warnings
        )
        ==
        0
        and
        n_iter
        <
        max_iter
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_file = (
        MODEL_DIR
        /
        f"{candidate_id}.joblib"
    )


    model_bundle = {

        "candidate_id":
            candidate_id,

        "model_family":
            "LogisticRegression",

        "model":
            model,

        "feature_order":
            features,

        "feature_count":
            25,

        "preprocessor":
            "StandardScaler",

        "scaler_sha256":
            actual_scaler_hash,

        "train_sha256":
            actual_train_hash,

        "training_scope":
            "CYCLE24_TRAIN_ONLY",

        "C":
            c_value,

        "class_weight":
            class_weight,

        "penalty":
            penalty,

        "solver":
            solver,

        "fit_intercept":
            fit_intercept,

        "max_iter":
            max_iter,

        "tol":
            tolerance,

        "random_state":
            random_state,

        "n_iter":
            n_iter,

        "converged":
            converged,

        "sklearn_version":
            sklearn.__version__,
    }


    joblib.dump(
        model_bundle,
        model_file,
    )


    # ========================================================
    # RELOAD ARTIFACT
    # ========================================================

    reloaded = joblib.load(
        model_file
    )


    if (
        reloaded.get(
            "candidate_id"
        )
        !=
        candidate_id
    ):

        raise RuntimeError(
            f"{candidate_id}: saved "
            "artifact ID mismatch."
        )


    if (
        reloaded.get(
            "random_state"
        )
        !=
        random_state
    ):

        raise RuntimeError(
            f"{candidate_id}: saved "
            "random_state mismatch."
        )


    if (
        reloaded.get(
            "feature_order"
        )
        !=
        features
    ):

        raise RuntimeError(
            f"{candidate_id}: saved "
            "feature-order mismatch."
        )


    if (
        reloaded.get(
            "scaler_sha256"
        )
        !=
        actual_scaler_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: saved "
            "scaler-hash mismatch."
        )


    if (
        reloaded.get(
            "train_sha256"
        )
        !=
        actual_train_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: saved "
            "TRAIN-hash mismatch."
        )


    reloaded_model = reloaded.get(
        "model"
    )


    if reloaded_model is None:

        raise RuntimeError(
            f"{candidate_id}: "
            "reloaded model missing."
        )


    # ========================================================
    # RELOAD REPRODUCIBILITY CHECK
    # ========================================================

    reloaded_probabilities = (
        reloaded_model.predict_proba(
            X_scaled
        )[
            :,
            1
        ]
    )


    if not np.allclose(
        train_probabilities,
        reloaded_probabilities,
        rtol=0.0,
        atol=1e-15,
    ):

        raise RuntimeError(
            f"{candidate_id}: saved model "
            "does not reproduce predictions."
        )


    model_hash = sha256_file(
        model_file
    )


    # ========================================================
    # WARNING TEXT
    # ========================================================

    future_warning_text = ""


    if future_warnings:

        future_warning_text = " | ".join(
            [
                str(
                    warning.message
                )

                for warning
                in future_warnings
            ]
        )


    other_warning_text = ""


    if other_warnings:

        other_warning_text = " | ".join(
            [
                (
                    f"{warning.category.__name__}: "
                    f"{warning.message}"
                )

                for warning
                in other_warnings
            ]
        )


    # ========================================================
    # RECORD CANDIDATE
    # ========================================================

    training_rows.append(
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
                penalty,

            "solver":
                solver,

            "fit_intercept":
                fit_intercept,

            "max_iter":
                max_iter,

            "tol":
                tolerance,

            "random_state":
                random_state,

            "train_rows":
                len(
                    y_train
                ),

            "train_positive":
                train_positive,

            "train_negative":
                train_negative,

            "feature_count":
                25,

            "n_iter":
                n_iter,

            "convergence_warning_count":
                len(
                    convergence_warnings
                ),

            "future_warning_count":
                len(
                    future_warnings
                ),

            "other_warning_count":
                len(
                    other_warnings
                ),

            "future_warning_text":
                future_warning_text,

            "other_warning_text":
                other_warning_text,

            "converged":
                converged,

            "coefficient_count":
                coefficient_count,

            "finite_coefficients":
                bool(
                    np.isfinite(
                        model.coef_
                    ).all()
                ),

            "finite_intercept":
                bool(
                    np.isfinite(
                        model.intercept_
                    ).all()
                ),

            "probability_min":
                probability_min,

            "probability_max":
                probability_max,

            "probabilities_finite":
                bool(
                    np.isfinite(
                        train_probabilities
                    ).all()
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

            "model_file":
                str(
                    model_file
                ),

            "model_sha256":
                model_hash,

            "train_sha256":
                actual_train_hash,

            "scaler_sha256":
                actual_scaler_hash,

            "sklearn_version":
                sklearn.__version__,
        }
    )


    print(
        f"{candidate_id}: "
        f"n_iter={n_iter}, "
        f"random_state={random_state}, "
        f"convergence_warnings="
        f"{len(convergence_warnings)}, "
        f"future_warnings="
        f"{len(future_warnings)}, "
        f"saved=YES"
    )


# ============================================================
# BUILD TRAINING SUMMARY
# ============================================================

summary = pd.DataFrame(
    training_rows
)


if len(summary) != 10:

    raise RuntimeError(
        "Training summary does not contain "
        "10 candidates."
    )


if summary[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Training summary contains "
        "duplicate candidate IDs."
    )


if summary[
    "model_sha256"
].isna().any():

    raise RuntimeError(
        "Missing model hashes."
    )


if not summary[
    "finite_coefficients"
].all():

    raise RuntimeError(
        "One or more candidates contain "
        "non-finite coefficients."
    )


if not summary[
    "finite_intercept"
].all():

    raise RuntimeError(
        "One or more candidates contain "
        "non-finite intercepts."
    )


if not summary[
    "probabilities_finite"
].all():

    raise RuntimeError(
        "One or more candidates contain "
        "non-finite probabilities."
    )


SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# SUMMARY COUNTS
# ============================================================

converged_count = int(
    summary[
        "converged"
    ].sum()
)


convergence_warning_count = int(
    summary[
        "convergence_warning_count"
    ].sum()
)


future_warning_count = int(
    summary[
        "future_warning_count"
    ].sum()
)


other_warning_count = int(
    summary[
        "other_warning_count"
    ].sum()
)


# ============================================================
# OUTPUT — TRAINING SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "TRAINING SUMMARY"
)

print(
    "===================================="
)


print(
    summary[
        [
            "candidate_id",
            "C",
            "class_weight",
            "random_state",
            "n_iter",
            "convergence_warning_count",
            "future_warning_count",
            "converged",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nCandidates trained:",
    len(
        summary
    )
)

print(
    "Candidates converged:",
    converged_count
)

print(
    "Total convergence warnings:",
    convergence_warning_count
)

print(
    "Total FutureWarnings:",
    future_warning_count
)

print(
    "Total unexpected other warnings:",
    other_warning_count
)


# ============================================================
# PROTOCOL CONFORMANCE
# ============================================================

protocol_conformance = bool(

    len(
        summary
    )
    ==
    10

    and
    summary[
        "random_state"
    ].eq(
        expected_random_state
    ).all()

    and
    summary[
        "penalty"
    ].eq(
        expected_penalty
    ).all()

    and
    summary[
        "solver"
    ].eq(
        expected_solver
    ).all()

    and
    summary[
        "max_iter"
    ].eq(
        expected_max_iter
    ).all()

    and
    summary[
        "converged"
    ].all()
)


print(
    "\n===================================="
)

print(
    "PROTOCOL CONFORMANCE"
)

print(
    "===================================="
)


print(
    "Frozen candidate count:",
    10
)

print(
    "random_state:",
    expected_random_state
)

print(
    "Penalty:",
    expected_penalty
)

print(
    "Solver:",
    expected_solver
)

print(
    "Max iterations:",
    expected_max_iter
)

print(
    "All candidates match frozen protocol:",
    protocol_conformance
)


# ============================================================
# WARNING CLASSIFICATION
# ============================================================

print(
    "\n===================================="
)

print(
    "WARNING CLASSIFICATION"
)

print(
    "===================================="
)


print(
    "Convergence warnings:",
    convergence_warning_count
)

print(
    "FutureWarnings:",
    future_warning_count
)

print(
    "Unexpected warnings:",
    other_warning_count
)


if future_warning_count > 0:

    print(
        "FutureWarning status: DOCUMENTED / NON-BLOCKING"
    )

    print(
        "Reason: scikit-learn API deprecation warning "
        "for explicit penalty parameter."
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


# ============================================================
# FINAL GATE
# ============================================================

chapter_pass = bool(

    protocol_conformance

    and
    converged_count
    ==
    10

    and
    convergence_warning_count
    ==
    0

    and
    other_warning_count
    ==
    0
)


print(
    "\n===================================="
)

print(
    "CHAPTER 3.2 RESULT"
)

print(
    "===================================="
)


print(
    "10 candidates trained:",
    len(
        summary
    )
    ==
    10
)

print(
    "10 candidates converged:",
    converged_count
    ==
    10
)

print(
    "Convergence warnings = 0:",
    convergence_warning_count
    ==
    0
)

print(
    "Unexpected warnings = 0:",
    other_warning_count
    ==
    0
)

print(
    "Protocol conformance:",
    protocol_conformance
)

print(
    "Chapter 3.2:",
    (
        "PASS"
        if chapter_pass
        else
        "FAIL"
    )
)


print(
    "\nSaved:"
)

print(
    SUMMARY_FILE
)

print(
    MODEL_DIR
)


if not chapter_pass:

    raise RuntimeError(
        "Chapter 3.2 Logistic Regression "
        "candidate training audit failed."
    )


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 3.2 COMPLETE"
)

print(
    "===================================="
)