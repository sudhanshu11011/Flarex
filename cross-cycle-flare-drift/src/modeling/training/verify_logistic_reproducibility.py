from pathlib import Path
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    "artifacts/splitting/data/cycle24_train.parquet"
)

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "model_feature_contract.json"
)

PREPROCESSING_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "preprocessing_contract.json"
)

SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/"
    "logistic_standard_scaler.joblib"
)

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

MODEL_DIR = Path(
    "artifacts/models/logistic/models"
)

AUDIT_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_reproducibility_audit.csv"
)

SUMMARY_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_reproducibility_summary.csv"
)


# ============================================================
# REPRODUCIBILITY TOLERANCES
# ============================================================

COEFFICIENT_ATOL = 1e-10
COEFFICIENT_RTOL = 1e-10

INTERCEPT_ATOL = 1e-10
INTERCEPT_RTOL = 1e-10

PROBABILITY_ATOL = 1e-10
PROBABILITY_RTOL = 1e-10


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

    if isinstance(
        value,
        bool,
    ):
        return value

    text = str(
        value
    ).strip().lower()

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
    "MODEL CHAPTER 3.5 — LOGISTIC REPRODUCIBILITY"
)

print(
    "===================================="
)


# ============================================================
# VERIFY INPUTS
# ============================================================

required_files = [
    TRAIN_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
    SCALER_FILE,
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


if protocol.get(
    "status"
) != "FROZEN_BEFORE_TRAINING":

    raise RuntimeError(
        "LR training protocol is not frozen."
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
    "Feature contract: VERIFIED"
)

print(
    "Preprocessing contract: FROZEN"
)

print(
    "LR training protocol: FROZEN"
)


# ============================================================
# VERIFY SCALER
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
        "Scaler hash differs from frozen contract."
    )


scaler_bundle = joblib.load(
    SCALER_FILE
)


if scaler_bundle.get(
    "feature_order"
) != features:

    raise RuntimeError(
        "Scaler feature order mismatch."
    )


if scaler_bundle.get(
    "fit_scope"
) != "CYCLE24_TRAIN_ONLY":

    raise RuntimeError(
        "Scaler was not fitted on TRAIN only."
    )


scaler = scaler_bundle.get(
    "scaler"
)


if scaler is None:

    raise RuntimeError(
        "Scaler object missing."
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
        "Non-finite TRAIN features."
    )


X_scaled = scaler.transform(
    X_train
)


if not np.isfinite(
    X_scaled
).all():

    raise RuntimeError(
        "Non-finite scaled TRAIN features."
    )


print(
    "TRAIN rows:",
    len(
        y_train
    )
)

print(
    "TRAIN positives:",
    int(
        y_train.sum()
    )
)

print(
    "Features:",
    X_scaled.shape[1]
)

print(
    "Frozen scaler transform: PASS"
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


if len(grid) != 10:

    raise RuntimeError(
        "Expected 10 frozen candidates."
    )


if len(training_summary) != 10:

    raise RuntimeError(
        "Expected 10 training-summary rows."
    )


# ============================================================
# FROZEN PROTOCOL
# ============================================================

fixed = protocol[
    "model_fixed_parameters"
]


expected_penalty = str(
    fixed[
        "penalty"
    ]
)


expected_solver = str(
    fixed[
        "solver"
    ]
)


expected_max_iter = int(
    fixed[
        "max_iter"
    ]
)


expected_tol = float(
    fixed[
        "tol"
    ]
)


expected_random_state = int(
    fixed[
        "random_state"
    ]
)


expected_fit_intercept = bool(
    fixed[
        "fit_intercept"
    ]
)


# ============================================================
# REPRODUCIBILITY AUDIT
# ============================================================

audit_rows = []


print(
    "\n===================================="
)

print(
    "RETRAIN / REPRODUCE CANDIDATES"
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


    # --------------------------------------------------------
    # LOAD SAVED MODEL
    # --------------------------------------------------------

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

    c_value = float(
        row[
            "C"
        ]
    )


    class_weight_text = str(
        row[
            "class_weight"
        ]
    ).strip().lower()


    if class_weight_text == "none":

        class_weight = None

    elif class_weight_text == "balanced":

        class_weight = "balanced"

    else:

        raise RuntimeError(
            f"{candidate_id}: invalid class_weight."
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


    # --------------------------------------------------------
    # VERIFY PROTOCOL FIRST
    # --------------------------------------------------------

    if penalty != expected_penalty:

        raise RuntimeError(
            f"{candidate_id}: penalty mismatch."
        )


    if solver != expected_solver:

        raise RuntimeError(
            f"{candidate_id}: solver mismatch."
        )


    if max_iter != expected_max_iter:

        raise RuntimeError(
            f"{candidate_id}: max_iter mismatch."
        )


    if not np.isclose(
        tolerance,
        expected_tol,
        rtol=0.0,
        atol=0.0,
    ):

        raise RuntimeError(
            f"{candidate_id}: tolerance mismatch."
        )


    if fit_intercept != expected_fit_intercept:

        raise RuntimeError(
            f"{candidate_id}: fit_intercept mismatch."
        )


    if random_state != expected_random_state:

        raise RuntimeError(
            f"{candidate_id}: random_state mismatch."
        )


    # --------------------------------------------------------
    # FRESH IN-MEMORY MODEL
    # --------------------------------------------------------

    fresh_model = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        penalty=penalty,
        solver=solver,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tolerance,
        random_state=random_state,
    )


    with warnings.catch_warnings(
        record=True
    ) as captured_warnings:

        warnings.simplefilter(
            "always"
        )


        fresh_model.fit(
            X_scaled,
            y_train,
        )


    convergence_warnings = [

        warning

        for warning
        in captured_warnings

        if issubclass(
            warning.category,
            ConvergenceWarning,
        )
    ]


    unexpected_warnings = [

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


    # --------------------------------------------------------
    # COMPARE ITERATIONS
    # --------------------------------------------------------

    saved_n_iter = int(
        np.max(
            saved_model.n_iter_
        )
    )


    fresh_n_iter = int(
        np.max(
            fresh_model.n_iter_
        )
    )


    n_iter_match = (
        saved_n_iter
        ==
        fresh_n_iter
    )


    # --------------------------------------------------------
    # COMPARE COEFFICIENTS
    # --------------------------------------------------------

    saved_coef = np.asarray(
        saved_model.coef_,
        dtype=np.float64,
    )


    fresh_coef = np.asarray(
        fresh_model.coef_,
        dtype=np.float64,
    )


    coefficient_max_abs_diff = float(
        np.max(
            np.abs(
                saved_coef
                -
                fresh_coef
            )
        )
    )


    coefficient_match = bool(
        np.allclose(
            saved_coef,
            fresh_coef,
            rtol=COEFFICIENT_RTOL,
            atol=COEFFICIENT_ATOL,
        )
    )


    # --------------------------------------------------------
    # COMPARE INTERCEPT
    # --------------------------------------------------------

    saved_intercept = np.asarray(
        saved_model.intercept_,
        dtype=np.float64,
    )


    fresh_intercept = np.asarray(
        fresh_model.intercept_,
        dtype=np.float64,
    )


    intercept_max_abs_diff = float(
        np.max(
            np.abs(
                saved_intercept
                -
                fresh_intercept
            )
        )
    )


    intercept_match = bool(
        np.allclose(
            saved_intercept,
            fresh_intercept,
            rtol=INTERCEPT_RTOL,
            atol=INTERCEPT_ATOL,
        )
    )


    # --------------------------------------------------------
    # COMPARE TRAIN PROBABILITIES
    #
    # Diagnostic reproducibility check only.
    # No threshold or performance metric is used.
    # --------------------------------------------------------

    saved_probabilities = (
        saved_model.predict_proba(
            X_scaled
        )[
            :,
            1
        ]
    )


    fresh_probabilities = (
        fresh_model.predict_proba(
            X_scaled
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
            rtol=PROBABILITY_RTOL,
            atol=PROBABILITY_ATOL,
        )
    )


    # --------------------------------------------------------
    # CLASSES
    # --------------------------------------------------------

    classes_match = bool(
        np.array_equal(
            saved_model.classes_,
            fresh_model.classes_,
        )
    )


    # --------------------------------------------------------
    # FINAL CANDIDATE RESULT
    # --------------------------------------------------------

    reproducible = bool(

        len(
            convergence_warnings
        )
        ==
        0

        and
        len(
            unexpected_warnings
        )
        ==
        0

        and
        n_iter_match

        and
        coefficient_match

        and
        intercept_match

        and
        probability_match

        and
        classes_match
    )


    audit_rows.append(
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

            "saved_n_iter":
                saved_n_iter,

            "fresh_n_iter":
                fresh_n_iter,

            "n_iter_match":
                n_iter_match,

            "coefficient_max_abs_diff":
                coefficient_max_abs_diff,

            "coefficient_match":
                coefficient_match,

            "intercept_max_abs_diff":
                intercept_max_abs_diff,

            "intercept_match":
                intercept_match,

            "probability_max_abs_diff":
                probability_max_abs_diff,

            "probability_match":
                probability_match,

            "classes_match":
                classes_match,

            "fresh_convergence_warning_count":
                len(
                    convergence_warnings
                ),

            "fresh_unexpected_warning_count":
                len(
                    unexpected_warnings
                ),

            "reproducibility_pass":
                reproducible,
        }
    )


    print(
        f"{candidate_id}: "
        f"n_iter_match={n_iter_match}, "
        f"coef_diff={coefficient_max_abs_diff:.3e}, "
        f"prob_diff={probability_max_abs_diff:.3e}, "
        f"PASS={reproducible}"
    )


# ============================================================
# BUILD AUDIT
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


if len(
    audit_df
) != 10:

    raise RuntimeError(
        "Reproducibility audit does not contain "
        "10 candidates."
    )


# ============================================================
# FINAL GATE
# ============================================================

pass_count = int(
    audit_df[
        "reproducibility_pass"
    ].sum()
)


all_reproducible = bool(
    pass_count
    ==
    10
)


max_coefficient_difference = float(
    audit_df[
        "coefficient_max_abs_diff"
    ].max()
)


max_intercept_difference = float(
    audit_df[
        "intercept_max_abs_diff"
    ].max()
)


max_probability_difference = float(
    audit_df[
        "probability_max_abs_diff"
    ].max()
)


total_convergence_warnings = int(
    audit_df[
        "fresh_convergence_warning_count"
    ].sum()
)


total_unexpected_warnings = int(
    audit_df[
        "fresh_unexpected_warning_count"
    ].sum()
)


summary = pd.DataFrame(
    [
        {
            "candidates_checked":
                10,

            "candidates_reproducible":
                pass_count,

            "max_coefficient_abs_difference":
                max_coefficient_difference,

            "max_intercept_abs_difference":
                max_intercept_difference,

            "max_probability_abs_difference":
                max_probability_difference,

            "coefficient_atol":
                COEFFICIENT_ATOL,

            "coefficient_rtol":
                COEFFICIENT_RTOL,

            "intercept_atol":
                INTERCEPT_ATOL,

            "intercept_rtol":
                INTERCEPT_RTOL,

            "probability_atol":
                PROBABILITY_ATOL,

            "probability_rtol":
                PROBABILITY_RTOL,

            "fresh_convergence_warnings":
                total_convergence_warnings,

            "fresh_unexpected_warnings":
                total_unexpected_warnings,

            "validation_used":
                False,

            "holdout_used":
                False,

            "cycle25_used":
                False,

            "model_selection_performed":
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


summary.to_csv(
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
    10
)

print(
    "Candidates reproducible:",
    pass_count
)

print(
    "Maximum coefficient difference:",
    max_coefficient_difference
)

print(
    "Maximum intercept difference:",
    max_intercept_difference
)

print(
    "Maximum probability difference:",
    max_probability_difference
)

print(
    "Fresh convergence warnings:",
    total_convergence_warnings
)

print(
    "Fresh unexpected warnings:",
    total_unexpected_warnings
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
    "TRAIN used: True"
)

print(
    "VALIDATION used: False"
)

print(
    "HOLDOUT used: False"
)

print(
    "Cycle25 used: False"
)

print(
    "Model selection performed: False"
)

print(
    "Threshold selection performed: False"
)

print(
    "Fresh models saved: False"
)


print(
    "\n===================================="
)

print(
    "CHAPTER 3.5 RESULT"
)

print(
    "===================================="
)


print(
    "Reproducibility:",
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
        "Chapter 3.5 Logistic Regression "
        "reproducibility verification failed."
    )


print(
    "\nMODEL CHAPTER 3.5 COMPLETE."
)