# ============================================================
# MODEL CHAPTER 9.6B
# PARALLEL XGBOOST CYCLE24 HOLDOUT SHAP
#
# Scientific method:
#   - Frozen XGB model
#   - Full Cycle24 HOLDOUT: 13,072 rows
#   - Same 25 SHARP features
#   - Same frozen 512-row Cycle24 TRAIN SHAP background
#   - TreeExplainer
#   - interventional feature perturbation
#   - model_output="raw"
#
# Compatibility correction:
#   SHAP 0.52 rejects interventional TreeExplainer when the
#   sklearn XGBClassifier wrapper has enable_categorical=True,
#   even though the fitted Booster contains no categorical
#   feature types.
#
#   Therefore TreeExplainer receives the EXACT fitted Booster
#   extracted from the frozen XGBClassifier bundle.
#
#   Diagnostic verification established:
#       wrapper raw margin == booster raw margin
#       mean abs difference = 0.0
#       max abs difference  = 0.0
#
# Execution optimization only:
#   - 6 independent workers
#   - 128-row chunks
#   - checkpoint after every chunk
#   - safe resume
#
# NO Cycle25
# NO retraining
# NO approximation
# NO sampling reduction
# ============================================================


# ------------------------------------------------------------
# LIMIT INTERNAL NUMERICAL THREADING
# MUST BE SET BEFORE NUMPY / XGBOOST / SHAP IMPORT
# ------------------------------------------------------------

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_ID = "XGB_02_LR0p03_D3_BAL"
MODEL_FAMILY = "xgboost"

CHUNK_SIZE = 128

EXPECTED_HOLDOUT_ROWS = 13072
EXPECTED_HOLDOUT_GROUPS = 176
EXPECTED_POSITIVES = 527
EXPECTED_NEGATIVES = 12545

BACKGROUND_SIZE = 512

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TARGET_COLUMN = "target"
TIME_COLUMN = "observation_time_utc"


# ============================================================
# PATHS
# ============================================================

MODEL_FILE = Path(
    "artifacts/models/xgboost/models/"
    "XGB_02_LR0p03_D3_BAL.joblib"
)

HOLDOUT_FILE = Path(
    "artifacts/splitting/data/"
    "cycle24_holdout.parquet"
)

BACKGROUND_FILE = Path(
    "artifacts/models/shift_controls/shap/background/"
    "cycle24_train_shap_background.parquet"
)

BACKGROUND_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "shap_background_contract.json"
)

CHECKPOINT_DIR = Path(
    "artifacts/models/shift_controls/shap/checkpoints/"
    "xgb_cycle24_holdout"
)

FINAL_OUTPUT_FILE = Path(
    "artifacts/models/shift_controls/shap/cycle24_holdout/"
    "XGB_02_LR0p03_D3_BAL.parquet"
)

XGB_SUMMARY_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "xgb_cycle24_holdout_shap_summary.csv"
)

XGB_CHUNK_MANIFEST_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "xgb_cycle24_holdout_shap_chunk_manifest.csv"
)

XGB_AUDIT_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "xgb_cycle24_holdout_shap_parallel_audit.json"
)


# ============================================================
# EXPECTED HASHES
# ============================================================

EXPECTED_MODEL_HASH = (
    "5839ac44333265b1b771008a82f6fc5a"
    "2f722e992386cde93fe8c04d48443d8b"
)

EXPECTED_HOLDOUT_HASH = (
    "8abb3929febcde7d778c4781ac890ff5e"
    "658933c2ef2a06aa19da085e7e255fc"
)

EXPECTED_TRAIN_HASH = (
    "4801a6e3b11d39cef8d4b0ef0eea4ae1"
    "67c27862431662c3f43ff55e4c433def"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "TOTUSJH",
    "TOTPOT",
    "TOTUSJZ",
    "ABSNJZH",
    "SAVNCPP",
    "USFLUX",
    "AREA_ACR",
    "MEANPOT",
    "R_VALUE",
    "SHRGT45",
    "MEANSHR",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "MEANJZH",
    "MEANJZD",
    "MEANALP",
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
]

SHAP_COLUMNS = [
    f"SHAP_{feature}"
    for feature in FEATURES
]


# ============================================================
# BASIC HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

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


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def checkpoint_path(
    chunk_index: int,
    start: int,
    end: int,
) -> Path:

    return (
        CHECKPOINT_DIR
        /
        (
            f"chunk_{chunk_index:03d}_"
            f"{start:05d}_{end - 1:05d}.parquet"
        )
    )


# ============================================================
# XGB SHAP OUTPUT EXTRACTION
# ============================================================

def extract_xgb_shap(
    values,
    row_count,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    # Standard binary-XGBoost output:
    # rows x features
    if values.shape == (
        row_count,
        len(FEATURES),
    ):

        return values


    # Defensive:
    # rows x features x outputs
    if (
        values.ndim == 3
        and
        values.shape[0] == row_count
        and
        values.shape[1] == len(FEATURES)
    ):

        if values.shape[2] == 1:

            return values[:, :, 0]

        if values.shape[2] >= 2:

            return values[:, :, 1]


    # Defensive alternate layout:
    # rows x outputs x features
    if (
        values.ndim == 3
        and
        values.shape[0] == row_count
        and
        values.shape[2] == len(FEATURES)
    ):

        if values.shape[1] == 1:

            return values[:, 0, :]

        if values.shape[1] >= 2:

            return values[:, 1, :]


    raise RuntimeError(
        f"Unexpected XGB SHAP shape: {values.shape}"
    )


def extract_xgb_base(
    base_values,
    row_count,
):

    base_values = np.asarray(
        base_values,
        dtype=np.float64,
    )


    if base_values.ndim == 0:

        return np.full(
            row_count,
            float(base_values),
            dtype=np.float64,
        )


    if base_values.shape == (
        row_count,
    ):

        return base_values.astype(
            np.float64
        )


    if base_values.shape == (
        row_count,
        1,
    ):

        return base_values[
            :,
            0
        ].astype(
            np.float64
        )


    if (
        base_values.ndim == 1
        and
        len(base_values) == 1
    ):

        return np.full(
            row_count,
            float(base_values[0]),
            dtype=np.float64,
        )


    if (
        base_values.ndim == 2
        and
        base_values.shape[0] == row_count
        and
        base_values.shape[1] >= 2
    ):

        return base_values[
            :,
            1
        ].astype(
            np.float64
        )


    raise RuntimeError(
        "Unexpected XGB SHAP base-value shape: "
        f"{base_values.shape}"
    )


# ============================================================
# XGBOOST RAW-MARGIN PREDICTION
# ============================================================

def predict_raw_margin(
    booster,
    X,
):

    output = booster.inplace_predict(
        X,
        predict_type="margin",
    )


    output = np.asarray(
        output,
        dtype=np.float64,
    )


    if output.ndim == 2:

        if output.shape[1] == 1:

            output = output[:, 0]

        elif output.shape[1] >= 2:

            output = output[:, 1]


    if output.ndim != 1:

        raise RuntimeError(
            "Unexpected XGB raw-margin shape: "
            f"{output.shape}"
        )


    return output


# ============================================================
# LOAD + VALIDATE FROZEN INPUTS
# ============================================================

def load_common_inputs():

    required_files = [
        MODEL_FILE,
        HOLDOUT_FILE,
        BACKGROUND_FILE,
        BACKGROUND_CONTRACT_FILE,
    ]


    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required input: {path}"
            )


    # --------------------------------------------------------
    # HASH VALIDATION
    # --------------------------------------------------------

    model_hash = sha256_file(
        MODEL_FILE
    )

    holdout_hash = sha256_file(
        HOLDOUT_FILE
    )

    background_hash = sha256_file(
        BACKGROUND_FILE
    )


    if model_hash != EXPECTED_MODEL_HASH:

        raise RuntimeError(
            "XGB model hash mismatch."
        )


    if holdout_hash != EXPECTED_HOLDOUT_HASH:

        raise RuntimeError(
            "Cycle24 HOLDOUT hash mismatch."
        )


    # --------------------------------------------------------
    # BACKGROUND CONTRACT
    # --------------------------------------------------------

    background_contract = load_json(
        BACKGROUND_CONTRACT_FILE
    )


    if (
        background_contract.get("status")
        !=
        "FROZEN_SHAP_BACKGROUND"
    ):

        raise RuntimeError(
            "SHAP background is not frozen."
        )


    contract_background_hash = (
        background_contract[
            "artifacts"
        ][
            "background"
        ][
            "sha256"
        ]
    )


    if (
        background_hash
        !=
        contract_background_hash
    ):

        raise RuntimeError(
            "Frozen SHAP background "
            "hash mismatch."
        )


    # --------------------------------------------------------
    # FROZEN MODEL BUNDLE
    # --------------------------------------------------------

    bundle = joblib.load(
        MODEL_FILE
    )


    if not isinstance(
        bundle,
        dict,
    ):

        raise RuntimeError(
            "Expected XGB model bundle dictionary."
        )


    required_bundle_keys = {
        "candidate_id",
        "model_family",
        "model",
        "feature_order",
        "feature_count",
        "training_scope",
        "train_sha256",
    }


    missing_keys = (
        required_bundle_keys
        -
        set(bundle.keys())
    )


    if missing_keys:

        raise RuntimeError(
            "XGB bundle missing keys: "
            f"{sorted(missing_keys)}"
        )


    if (
        bundle["candidate_id"]
        !=
        CANDIDATE_ID
    ):

        raise RuntimeError(
            "XGB candidate ID mismatch."
        )


    if (
        bundle["training_scope"]
        !=
        "CYCLE24_TRAIN_ONLY"
    ):

        raise RuntimeError(
            "XGB training scope mismatch."
        )


    if (
        bundle["train_sha256"]
        !=
        EXPECTED_TRAIN_HASH
    ):

        raise RuntimeError(
            "XGB TRAIN provenance mismatch."
        )


    if (
        bundle["feature_order"]
        !=
        FEATURES
    ):

        raise RuntimeError(
            "XGB feature-order mismatch."
        )


    if int(
        bundle["feature_count"]
    ) != 25:

        raise RuntimeError(
            "XGB feature-count mismatch."
        )


    model = bundle[
        "model"
    ]


    # --------------------------------------------------------
    # EXACT FITTED BOOSTER
    # --------------------------------------------------------

    booster = model.get_booster()


    # Execution-only thread limits.
    # No tree/model parameter is retrained or modified.
    try:

        model.set_params(
            n_jobs=1
        )

    except Exception:

        pass


    try:

        booster.set_param(
            {
                "nthread": 1
            }
        )

    except Exception:

        pass


    # --------------------------------------------------------
    # HOLDOUT
    # --------------------------------------------------------

    holdout = pd.read_parquet(
        HOLDOUT_FILE
    )


    if len(
        holdout
    ) != EXPECTED_HOLDOUT_ROWS:

        raise RuntimeError(
            "Cycle24 HOLDOUT row-count mismatch."
        )


    if int(
        holdout[
            ENTITY_GROUP_COLUMN
        ].nunique()
    ) != EXPECTED_HOLDOUT_GROUPS:

        raise RuntimeError(
            "Cycle24 HOLDOUT "
            "ENTITY_GROUP count mismatch."
        )


    positives = int(
        holdout[
            TARGET_COLUMN
        ].sum()
    )


    negatives = int(
        (
            holdout[
                TARGET_COLUMN
            ]
            ==
            0
        ).sum()
    )


    if positives != EXPECTED_POSITIVES:

        raise RuntimeError(
            "Cycle24 HOLDOUT "
            "positive-count mismatch."
        )


    if negatives != EXPECTED_NEGATIVES:

        raise RuntimeError(
            "Cycle24 HOLDOUT "
            "negative-count mismatch."
        )


    X_holdout = holdout[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )


    if X_holdout.shape != (
        EXPECTED_HOLDOUT_ROWS,
        25,
    ):

        raise RuntimeError(
            "Cycle24 HOLDOUT "
            "feature matrix mismatch."
        )


    if not np.isfinite(
        X_holdout
    ).all():

        raise RuntimeError(
            "Cycle24 HOLDOUT contains "
            "non-finite feature values."
        )


    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    background_df = pd.read_parquet(
        BACKGROUND_FILE
    )


    if len(
        background_df
    ) != BACKGROUND_SIZE:

        raise RuntimeError(
            "SHAP background row-count mismatch."
        )


    if int(
        background_df[
            ENTITY_GROUP_COLUMN
        ].nunique()
    ) != BACKGROUND_SIZE:

        raise RuntimeError(
            "SHAP background must contain "
            "512 unique ENTITY_GROUPs."
        )


    X_background = background_df[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )


    if X_background.shape != (
        BACKGROUND_SIZE,
        25,
    ):

        raise RuntimeError(
            "SHAP background "
            "feature matrix mismatch."
        )


    if not np.isfinite(
        X_background
    ).all():

        raise RuntimeError(
            "SHAP background contains "
            "non-finite feature values."
        )


    # --------------------------------------------------------
    # WRAPPER ↔ BOOSTER EQUIVALENCE AUDIT
    #
    # Cheap verification over the COMPLETE HOLDOUT.
    # This proves the fitted Booster used by SHAP produces
    # the same raw margins as the frozen sklearn wrapper.
    # --------------------------------------------------------

    wrapper_margin = model.predict(
        X_holdout,
        output_margin=True,
    )


    wrapper_margin = np.asarray(
        wrapper_margin,
        dtype=np.float64,
    ).reshape(-1)


    booster_margin = predict_raw_margin(
        booster,
        X_holdout,
    )


    wrapper_booster_error = np.abs(
        wrapper_margin
        -
        booster_margin
    )


    wrapper_booster_mean_abs_error = float(
        wrapper_booster_error.mean()
    )


    wrapper_booster_max_abs_error = float(
        wrapper_booster_error.max()
    )


    wrapper_booster_equivalent = bool(
        np.allclose(
            wrapper_margin,
            booster_margin,
            atol=1e-12,
            rtol=1e-12,
        )
    )


    if not wrapper_booster_equivalent:

        raise RuntimeError(
            "Frozen XGBClassifier wrapper "
            "and its fitted Booster do not "
            "produce equivalent raw margins. "
            f"max_abs_error="
            f"{wrapper_booster_max_abs_error}"
        )


    return {
        "model":
            model,

        "booster":
            booster,

        "bundle":
            bundle,

        "holdout":
            holdout,

        "X_holdout":
            X_holdout,

        "X_background":
            X_background,

        "model_hash":
            model_hash,

        "holdout_hash":
            holdout_hash,

        "background_hash":
            background_hash,

        "wrapper_booster_equivalent":
            wrapper_booster_equivalent,

        "wrapper_booster_mean_abs_error":
            wrapper_booster_mean_abs_error,

        "wrapper_booster_max_abs_error":
            wrapper_booster_max_abs_error,
    }


# ============================================================
# CREATE INTERVENTIONAL TREE EXPLAINER
#
# IMPORTANT:
#   Pass exact fitted Booster, NOT sklearn XGBClassifier.
# ============================================================

def create_explainer(
    booster,
    X_background,
):

    masker = shap.maskers.Independent(
        X_background,
        max_samples=BACKGROUND_SIZE,
    )


    masker_rows = int(
        masker.data.shape[0]
    )


    if masker_rows != BACKGROUND_SIZE:

        raise RuntimeError(
            "XGB masker failed to retain "
            "all 512 frozen background rows."
        )


    explainer = shap.TreeExplainer(
        booster,
        data=masker,
        feature_perturbation="interventional",
        model_output="raw",
    )


    if not hasattr(
        explainer,
        "masker",
    ):

        raise RuntimeError(
            "XGB TreeExplainer masker missing."
        )


    if not hasattr(
        explainer.masker,
        "data",
    ):

        raise RuntimeError(
            "XGB TreeExplainer masker "
            "does not expose background data."
        )


    actual_background_rows = int(
        explainer.masker.data.shape[0]
    )


    if (
        actual_background_rows
        !=
        BACKGROUND_SIZE
    ):

        raise RuntimeError(
            "XGB TreeExplainer did not retain "
            "the exact frozen 512-row background."
        )


    return explainer


# ============================================================
# CHECKPOINT VALIDATION
# ============================================================

def validate_checkpoint(
    path: Path,
    start: int,
    end: int,
):

    frame = pd.read_parquet(
        path
    )


    expected_rows = (
        end
        -
        start
    )


    if len(frame) != expected_rows:

        raise RuntimeError(
            f"Invalid checkpoint row count: {path}"
        )


    required_columns = {
        "source_row_number",
        "candidate_id",
        "model_family",
        "ENTITY_GROUP",
        ENTITY_GROUP_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN,
        "shap_base_value",
        *SHAP_COLUMNS,
    }


    missing_columns = (
        required_columns
        -
        set(frame.columns)
    )


    if missing_columns:

        raise RuntimeError(
            "Checkpoint missing columns "
            f"{sorted(missing_columns)}: {path}"
        )


    if not (
        frame[
            "candidate_id"
        ].astype(str)
        ==
        CANDIDATE_ID
    ).all():

        raise RuntimeError(
            f"Checkpoint candidate mismatch: {path}"
        )


    if not (
        frame[
            "model_family"
        ].astype(str)
        ==
        MODEL_FAMILY
    ).all():

        raise RuntimeError(
            f"Checkpoint model-family mismatch: {path}"
        )


    expected_indices = np.arange(
        start,
        end,
        dtype=np.int64,
    )


    actual_indices = frame[
        "source_row_number"
    ].to_numpy(
        dtype=np.int64
    )


    if not np.array_equal(
        actual_indices,
        expected_indices,
    ):

        raise RuntimeError(
            f"Checkpoint row-index mismatch: {path}"
        )


    shap_values = frame[
        SHAP_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )


    base_values = frame[
        "shap_base_value"
    ].to_numpy(
        dtype=np.float64
    )


    if shap_values.shape != (
        expected_rows,
        25,
    ):

        raise RuntimeError(
            f"Checkpoint SHAP shape mismatch: {path}"
        )


    if not np.isfinite(
        shap_values
    ).all():

        raise RuntimeError(
            f"Non-finite SHAP checkpoint: {path}"
        )


    if not np.isfinite(
        base_values
    ).all():

        raise RuntimeError(
            f"Non-finite SHAP base checkpoint: {path}"
        )


    return True


# ============================================================
# WORKER EXECUTION
# ============================================================

def run_worker(
    worker_id: int,
    total_workers: int,
):

    if total_workers < 1:

        raise RuntimeError(
            "workers must be >= 1"
        )


    if (
        worker_id < 0
        or
        worker_id >= total_workers
    ):

        raise RuntimeError(
            f"worker-id must be between "
            f"0 and {total_workers - 1}"
        )


    print("=" * 72)

    print(
        "CHAPTER 9.6B — "
        "XGB PARALLEL SHAP WORKER"
    )

    print("=" * 72)


    print(
        "Worker:",
        f"{worker_id}/{total_workers - 1}"
    )

    print(
        "XGBoost runtime:",
        xgboost.__version__
    )

    print(
        "SHAP runtime:",
        shap.__version__
    )

    print(
        "Internal numerical threads:",
        1
    )


    common = load_common_inputs()


    booster = common[
        "booster"
    ]

    holdout = common[
        "holdout"
    ]

    X_holdout = common[
        "X_holdout"
    ]

    X_background = common[
        "X_background"
    ]


    print(
        "Input validation:",
        "PASS"
    )

    print(
        "XGB model hash:",
        "PASS"
    )

    print(
        "Cycle24 HOLDOUT hash:",
        "PASS"
    )

    print(
        "Frozen background hash:",
        "PASS"
    )

    print(
        "Training scope:",
        "CYCLE24_TRAIN_ONLY"
    )

    print(
        "Wrapper → Booster raw-margin equivalence:",
        "PASS"
    )

    print(
        "Wrapper/Booster mean abs error:",
        f"{common['wrapper_booster_mean_abs_error']:.16e}"
    )

    print(
        "Wrapper/Booster max abs error:",
        f"{common['wrapper_booster_max_abs_error']:.16e}"
    )

    print(
        "HOLDOUT rows:",
        len(holdout)
    )

    print(
        "Background rows:",
        len(X_background)
    )


    # --------------------------------------------------------
    # CREATE EXPLAINER FROM EXACT FITTED BOOSTER
    # --------------------------------------------------------

    explainer = create_explainer(
        booster,
        X_background,
    )


    print(
        "Explainer model input:",
        "FROZEN_XGBOOST_BOOSTER"
    )

    print(
        "TreeExplainer background rows:",
        explainer.masker.data.shape[0]
    )

    print(
        "TreeExplainer background audit:",
        "PASS"
    )


    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    total_chunks = math.ceil(
        EXPECTED_HOLDOUT_ROWS
        /
        CHUNK_SIZE
    )


    assigned_chunks = [
        chunk_index
        for chunk_index in range(
            total_chunks
        )
        if (
            chunk_index
            %
            total_workers
        )
        ==
        worker_id
    ]


    print(
        "Total global chunks:",
        total_chunks
    )

    print(
        "Chunks assigned:",
        len(assigned_chunks)
    )

    print(
        "Assigned chunk IDs:",
        assigned_chunks
    )


    completed_now = 0

    skipped_existing = 0


    # --------------------------------------------------------
    # PROCESS ASSIGNED CHUNKS
    # --------------------------------------------------------

    for position, chunk_index in enumerate(
        assigned_chunks,
        start=1,
    ):

        start = (
            chunk_index
            *
            CHUNK_SIZE
        )


        end = min(
            start
            +
            CHUNK_SIZE,
            EXPECTED_HOLDOUT_ROWS,
        )


        path = checkpoint_path(
            chunk_index,
            start,
            end,
        )


        print()

        print(
            f"[Worker {worker_id}] "
            f"Chunk {chunk_index} "
            f"({position}/{len(assigned_chunks)}) "
            f"rows {start}–{end - 1}"
        )


        # ----------------------------------------------------
        # SAFE RESUME
        # ----------------------------------------------------

        if path.exists():

            validate_checkpoint(
                path,
                start,
                end,
            )

            print(
                "Existing checkpoint valid — SKIP"
            )

            skipped_existing += 1

            continue


        # ----------------------------------------------------
        # COMPUTE SHAP
        # ----------------------------------------------------

        X_chunk = X_holdout[
            start:end
        ]


        explanation = explainer(
            X_chunk,
            check_additivity=True,
        )


        shap_matrix = extract_xgb_shap(
            explanation.values,
            len(X_chunk),
        )


        base_vector = extract_xgb_base(
            explanation.base_values,
            len(X_chunk),
        )


        if shap_matrix.shape != (
            len(X_chunk),
            25,
        ):

            raise RuntimeError(
                "XGB SHAP matrix shape mismatch."
            )


        if base_vector.shape != (
            len(X_chunk),
        ):

            raise RuntimeError(
                "XGB SHAP base-vector "
                "shape mismatch."
            )


        if not np.isfinite(
            shap_matrix
        ).all():

            raise RuntimeError(
                "Non-finite XGB SHAP values."
            )


        if not np.isfinite(
            base_vector
        ).all():

            raise RuntimeError(
                "Non-finite XGB SHAP "
                "base values."
            )


        # ----------------------------------------------------
        # RAW-MARGIN ADDITIVITY AUDIT
        # ----------------------------------------------------

        target_output = predict_raw_margin(
            booster,
            X_chunk,
        )


        reconstructed_output = (
            base_vector
            +
            shap_matrix.sum(
                axis=1
            )
        )


        abs_error = np.abs(
            reconstructed_output
            -
            target_output
        )


        native_style_pass = bool(
            np.allclose(
                reconstructed_output,
                target_output,
                atol=1e-2,
                rtol=1e-2,
            )
        )


        if not native_style_pass:

            raise RuntimeError(
                "XGB chunk additivity "
                "validation failed. "
                f"max_abs_error="
                f"{float(abs_error.max())}"
            )


        print(
            "Chunk additivity:",
            "PASS"
        )

        print(
            "Chunk max abs error:",
            f"{float(abs_error.max()):.10f}"
        )


        # ----------------------------------------------------
        # BUILD CHECKPOINT
        # ----------------------------------------------------

        holdout_chunk = holdout.iloc[
            start:end
        ]


        output = pd.DataFrame(
            {
                "source_row_number":
                    np.arange(
                        start,
                        end,
                        dtype=np.int64,
                    ),

                "candidate_id":
                    CANDIDATE_ID,

                "model_family":
                    MODEL_FAMILY,

                "ENTITY_GROUP":
                    holdout_chunk[
                        ENTITY_GROUP_COLUMN
                    ].astype(
                        str
                    ).to_numpy(),

                ENTITY_GROUP_COLUMN:
                    holdout_chunk[
                        ENTITY_GROUP_COLUMN
                    ].astype(
                        str
                    ).to_numpy(),

                TARGET_COLUMN:
                    holdout_chunk[
                        TARGET_COLUMN
                    ].astype(
                        np.int8
                    ).to_numpy(),

                TIME_COLUMN:
                    holdout_chunk[
                        TIME_COLUMN
                    ].to_numpy(),

                "shap_base_value":
                    base_vector,
            }
        )


        for feature_index, feature in enumerate(
            FEATURES
        ):

            output[
                f"SHAP_{feature}"
            ] = shap_matrix[
                :,
                feature_index
            ]


        # ----------------------------------------------------
        # ATOMIC CHECKPOINT WRITE
        # ----------------------------------------------------

        temp_path = path.with_suffix(
            ".tmp.parquet"
        )


        output.to_parquet(
            temp_path,
            index=False,
        )


        validate_checkpoint(
            temp_path,
            start,
            end,
        )


        os.replace(
            temp_path,
            path,
        )


        validate_checkpoint(
            path,
            start,
            end,
        )


        completed_now += 1


        print(
            "Checkpoint saved:",
            path
        )


    print()

    print("=" * 72)

    print(
        f"XGB WORKER {worker_id} COMPLETE"
    )

    print("=" * 72)


    print(
        "New chunks computed:",
        completed_now
    )

    print(
        "Existing chunks reused:",
        skipped_existing
    )

    print(
        "Cycle25 read:",
        False
    )

    print(
        "Model retrained:",
        False
    )


# ============================================================
# FINALIZE ALL XGB CHECKPOINTS
# ============================================================

def finalize(
    total_workers: int,
):

    print("=" * 72)

    print(
        "CHAPTER 9.6B — "
        "XGB PARALLEL SHAP FINALIZATION"
    )

    print("=" * 72)


    common = load_common_inputs()


    booster = common[
        "booster"
    ]

    holdout = common[
        "holdout"
    ]

    X_holdout = common[
        "X_holdout"
    ]


    FINAL_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    XGB_SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    XGB_CHUNK_MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    XGB_AUDIT_CONTRACT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    if FINAL_OUTPUT_FILE.exists():

        raise RuntimeError(
            "Final XGB SHAP artifact "
            "already exists:\n"
            f"{FINAL_OUTPUT_FILE}\n"
            "Refusing to overwrite it."
        )


    total_chunks = math.ceil(
        EXPECTED_HOLDOUT_ROWS
        /
        CHUNK_SIZE
    )


    frames = []

    manifest_rows = []


    # --------------------------------------------------------
    # VERIFY EVERY EXPECTED CHECKPOINT
    # --------------------------------------------------------

    for chunk_index in range(
        total_chunks
    ):

        start = (
            chunk_index
            *
            CHUNK_SIZE
        )


        end = min(
            start
            +
            CHUNK_SIZE,
            EXPECTED_HOLDOUT_ROWS,
        )


        path = checkpoint_path(
            chunk_index,
            start,
            end,
        )


        if not path.exists():

            raise RuntimeError(
                "Missing XGB checkpoint:\n"
                f"{path}"
            )


        validate_checkpoint(
            path,
            start,
            end,
        )


        frame = pd.read_parquet(
            path
        )


        frames.append(
            frame
        )


        manifest_rows.append(
            {
                "chunk_index":
                    chunk_index,

                "assigned_worker_mod":
                    (
                        chunk_index
                        %
                        total_workers
                    ),

                "start_row":
                    start,

                "end_row_inclusive":
                    end - 1,

                "rows":
                    end - start,

                "file":
                    str(path),

                "sha256":
                    sha256_file(path),
            }
        )


    print(
        "All expected checkpoints:",
        "PASS"
    )

    print(
        "Checkpoint count:",
        len(frames)
    )


    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    merged = pd.concat(
        frames,
        ignore_index=True,
    )


    merged = merged.sort_values(
        "source_row_number",
        kind="stable",
    ).reset_index(
        drop=True
    )


    if len(
        merged
    ) != EXPECTED_HOLDOUT_ROWS:

        raise RuntimeError(
            "Merged XGB row-count mismatch."
        )


    expected_indices = np.arange(
        EXPECTED_HOLDOUT_ROWS,
        dtype=np.int64,
    )


    actual_indices = merged[
        "source_row_number"
    ].to_numpy(
        dtype=np.int64
    )


    if not np.array_equal(
        actual_indices,
        expected_indices,
    ):

        raise RuntimeError(
            "Merged XGB row sequence mismatch."
        )


    if merged[
        "source_row_number"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate XGB source rows detected."
        )


    print(
        "Merged row-order audit:",
        "PASS"
    )


    # --------------------------------------------------------
    # FINAL SHAP MATRIX
    # --------------------------------------------------------

    shap_matrix = merged[
        SHAP_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )


    base_vector = merged[
        "shap_base_value"
    ].to_numpy(
        dtype=np.float64
    )


    if shap_matrix.shape != (
        EXPECTED_HOLDOUT_ROWS,
        25,
    ):

        raise RuntimeError(
            "Merged XGB SHAP shape mismatch."
        )


    if base_vector.shape != (
        EXPECTED_HOLDOUT_ROWS,
    ):

        raise RuntimeError(
            "Merged XGB base-value "
            "shape mismatch."
        )


    if not np.isfinite(
        shap_matrix
    ).all():

        raise RuntimeError(
            "Merged XGB SHAP contains "
            "non-finite values."
        )


    if not np.isfinite(
        base_vector
    ).all():

        raise RuntimeError(
            "Merged XGB base values contain "
            "non-finite values."
        )


    # --------------------------------------------------------
    # FINAL ADDITIVITY AUDIT
    # --------------------------------------------------------

    target_output = predict_raw_margin(
        booster,
        X_holdout,
    )


    reconstructed_output = (
        base_vector
        +
        shap_matrix.sum(
            axis=1
        )
    )


    abs_error = np.abs(
        reconstructed_output
        -
        target_output
    )


    native_additivity_pass = bool(
        np.allclose(
            reconstructed_output,
            target_output,
            atol=1e-2,
            rtol=1e-2,
        )
    )


    if not native_additivity_pass:

        raise RuntimeError(
            "Final XGB native-style "
            "additivity audit failed. "
            f"max_abs_error="
            f"{float(abs_error.max())}"
        )


    relative_error_diagnostic = (
        abs_error
        /
        (
            np.abs(
                reconstructed_output
            )
            +
            1e-2
        )
    )


    print(
        "Final native-style additivity:",
        "PASS"
    )

    print(
        "Mean absolute error:",
        f"{float(abs_error.mean()):.10f}"
    )

    print(
        "P95 absolute error:",
        f"{float(np.quantile(abs_error, 0.95)):.10f}"
    )

    print(
        "P99 absolute error:",
        f"{float(np.quantile(abs_error, 0.99)):.10f}"
    )

    print(
        "Max absolute error:",
        f"{float(abs_error.max()):.10f}"
    )

    print(
        "Max relative diagnostic:",
        f"{float(relative_error_diagnostic.max()):.6%}"
    )

    print(
        "Rows >1% relative diagnostic:",
        int(
            np.sum(
                relative_error_diagnostic
                >
                0.01
            )
        )
    )


    # --------------------------------------------------------
    # SAVE FINAL SHAP ARTIFACT
    # --------------------------------------------------------

    merged.to_parquet(
        FINAL_OUTPUT_FILE,
        index=False,
    )


    roundtrip = pd.read_parquet(
        FINAL_OUTPUT_FILE
    )


    if len(
        roundtrip
    ) != EXPECTED_HOLDOUT_ROWS:

        raise RuntimeError(
            "Final XGB parquet "
            "round-trip failed."
        )


    roundtrip_shap = roundtrip[
        SHAP_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )


    if roundtrip_shap.shape != (
        EXPECTED_HOLDOUT_ROWS,
        25,
    ):

        raise RuntimeError(
            "Final XGB parquet "
            "SHAP shape mismatch."
        )


    if not np.isfinite(
        roundtrip_shap
    ).all():

        raise RuntimeError(
            "Final XGB parquet contains "
            "non-finite SHAP values."
        )


    final_hash = sha256_file(
        FINAL_OUTPUT_FILE
    )


    print(
        "Final XGB parquet:",
        "PASS"
    )

    print(
        "XGB artifact SHA256:",
        final_hash
    )


    # --------------------------------------------------------
    # GLOBAL + CLASS-CONDITIONAL SUMMARY
    # --------------------------------------------------------

    summary_rows = []


    strata = {
        "OVERALL":
            np.ones(
                EXPECTED_HOLDOUT_ROWS,
                dtype=bool,
            ),

        "TARGET_0":
            (
                holdout[
                    TARGET_COLUMN
                ].to_numpy(
                    dtype=np.int8
                )
                ==
                0
            ),

        "TARGET_1":
            (
                holdout[
                    TARGET_COLUMN
                ].to_numpy(
                    dtype=np.int8
                )
                ==
                1
            ),
    }


    for stratum, mask in strata.items():

        subset = shap_matrix[
            mask
        ]


        if len(subset) == 0:

            raise RuntimeError(
                f"Empty XGB SHAP stratum: {stratum}"
            )


        mean_abs = np.mean(
            np.abs(subset),
            axis=0,
        )


        mean_signed = np.mean(
            subset,
            axis=0,
        )


        ranks = (
            pd.Series(
                mean_abs,
                index=FEATURES,
            )
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )


        for feature_index, feature in enumerate(
            FEATURES,
            start=1,
        ):

            summary_rows.append(
                {
                    "candidate_id":
                        CANDIDATE_ID,

                    "model_family":
                        MODEL_FAMILY,

                    "stratum":
                        stratum,

                    "rows":
                        int(mask.sum()),

                    "entity_groups":
                        int(
                            holdout.loc[
                                mask,
                                ENTITY_GROUP_COLUMN,
                            ].nunique()
                        ),

                    "feature_index":
                        feature_index,

                    "feature":
                        feature,

                    "mean_absolute_shap":
                        float(
                            mean_abs[
                                feature_index - 1
                            ]
                        ),

                    "mean_signed_shap":
                        float(
                            mean_signed[
                                feature_index - 1
                            ]
                        ),

                    "mean_abs_rank":
                        int(
                            ranks[
                                feature
                            ]
                        ),
                }
            )


    summary_df = pd.DataFrame(
        summary_rows
    )


    if len(
        summary_df
    ) != 75:

        raise RuntimeError(
            "Expected exactly 75 "
            "XGB summary rows."
        )


    summary_df.to_csv(
        XGB_SUMMARY_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # CHUNK MANIFEST
    # --------------------------------------------------------

    manifest_df = pd.DataFrame(
        manifest_rows
    )


    if len(
        manifest_df
    ) != total_chunks:

        raise RuntimeError(
            "XGB manifest chunk-count mismatch."
        )


    manifest_df.to_csv(
        XGB_CHUNK_MANIFEST_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # AUDIT CONTRACT
    # --------------------------------------------------------

    audit_contract = {

        "contract_name":
            (
                "Cycle24 HOLDOUT XGB SHAP "
                "Parallel Execution Audit"
            ),

        "version":
            "1.1",

        "status":
            (
                "VALIDATED_XGB_COMPONENT_"
                "FOR_CHAPTER_9_6B"
            ),

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "candidate_id":
            CANDIDATE_ID,

        "model_family":
            MODEL_FAMILY,

        "software":
            {
                "shap_version":
                    shap.__version__,

                "xgboost_version":
                    xgboost.__version__,
            },

        "execution":
            {
                "execution_type":
                    "PARALLEL_CHECKPOINTED",

                "requested_workers":
                    total_workers,

                "chunk_size":
                    CHUNK_SIZE,

                "total_chunks":
                    total_chunks,

                "internal_threads_per_worker":
                    1,

                "checkpoint_resume_enabled":
                    True,
            },

        "scientific_method":
            {
                "evaluation_population":
                    "CYCLE24_HOLDOUT",

                "evaluation_rows":
                    EXPECTED_HOLDOUT_ROWS,

                "entity_groups":
                    EXPECTED_HOLDOUT_GROUPS,

                "feature_count":
                    25,

                "background_source":
                    "FROZEN_CYCLE24_TRAIN",

                "background_rows":
                    BACKGROUND_SIZE,

                "explainer":
                    "TreeExplainer",

                "explainer_model_input":
                    "FROZEN_XGBOOST_BOOSTER",

                "booster_extracted_from":
                    "FROZEN_XGBCLASSIFIER_BUNDLE",

                "feature_perturbation":
                    "interventional",

                "model_output":
                    "raw",

                "raw_output_interpretation":
                    "XGBOOST_RAW_MARGIN",

                "approximation":
                    False,

                "sampling_reduction":
                    False,

                "cycle25_read":
                    False,

                "model_retrained":
                    False,

                "trees_modified":
                    False,

                "feature_contract_changed":
                    False,

                "background_changed":
                    False,

                "threading_change_only":
                    True,
            },

        "wrapper_booster_equivalence":
            {
                "population":
                    "FULL_CYCLE24_HOLDOUT",

                "rows":
                    EXPECTED_HOLDOUT_ROWS,

                "comparison":
                    (
                        "XGBClassifier output_margin=True "
                        "vs fitted Booster inplace_predict "
                        "predict_type=margin"
                    ),

                "equivalent_at_1e_12":
                    common[
                        "wrapper_booster_equivalent"
                    ],

                "mean_absolute_error":
                    common[
                        "wrapper_booster_mean_abs_error"
                    ],

                "max_absolute_error":
                    common[
                        "wrapper_booster_max_abs_error"
                    ],
            },

        "provenance":
            {
                "model_file":
                    str(
                        MODEL_FILE
                    ),

                "model_sha256":
                    common[
                        "model_hash"
                    ],

                "holdout_file":
                    str(
                        HOLDOUT_FILE
                    ),

                "holdout_sha256":
                    common[
                        "holdout_hash"
                    ],

                "background_file":
                    str(
                        BACKGROUND_FILE
                    ),

                "background_sha256":
                    common[
                        "background_hash"
                    ],
            },

        "additivity":
            {
                "target_output":
                    "XGBOOST_RAW_MARGIN",

                "native_style_rule":
                    (
                        "numpy.allclose("
                        "reconstructed_output,"
                        "raw_margin,"
                        "atol=1e-2,"
                        "rtol=1e-2)"
                    ),

                "native_style_pass":
                    native_additivity_pass,

                "mean_absolute_error":
                    float(
                        abs_error.mean()
                    ),

                "p95_absolute_error":
                    float(
                        np.quantile(
                            abs_error,
                            0.95,
                        )
                    ),

                "p99_absolute_error":
                    float(
                        np.quantile(
                            abs_error,
                            0.99,
                        )
                    ),

                "max_absolute_error":
                    float(
                        abs_error.max()
                    ),

                "relative_error_is_diagnostic_only":
                    True,

                "max_relative_diagnostic":
                    float(
                        relative_error_diagnostic.max()
                    ),

                "rows_over_1pct_relative_diagnostic":
                    int(
                        np.sum(
                            relative_error_diagnostic
                            >
                            0.01
                        )
                    ),
            },

        "artifacts":
            {
                "xgb_shap":
                    {
                        "file":
                            str(
                                FINAL_OUTPUT_FILE
                            ),

                        "sha256":
                            final_hash,

                        "rows":
                            EXPECTED_HOLDOUT_ROWS,

                        "features":
                            25,
                    },

                "xgb_summary":
                    {
                        "file":
                            str(
                                XGB_SUMMARY_FILE
                            ),

                        "sha256":
                            sha256_file(
                                XGB_SUMMARY_FILE
                            ),

                        "rows":
                            75,
                    },

                "chunk_manifest":
                    {
                        "file":
                            str(
                                XGB_CHUNK_MANIFEST_FILE
                            ),

                        "sha256":
                            sha256_file(
                                XGB_CHUNK_MANIFEST_FILE
                            ),

                        "chunks":
                            total_chunks,
                    },
            },

        "compatibility_note":
            (
                "SHAP 0.52 rejected interventional "
                "TreeExplainer when supplied with the frozen "
                "XGBClassifier wrapper because the wrapper "
                "reported enable_categorical=True, although "
                "the fitted Booster had no categorical "
                "feature types. TreeExplainer was therefore "
                "supplied with the exact fitted Booster "
                "extracted from the frozen XGBClassifier "
                "bundle. Wrapper and Booster raw margins "
                "were verified over the complete Cycle24 "
                "HOLDOUT before SHAP computation. No model "
                "retraining, tree modification, background "
                "change, feature change, approximation, "
                "sampling reduction, or scientific-analysis "
                "change occurred."
            ),
    }


    with XGB_AUDIT_CONTRACT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit_contract,
            file,
            indent=2,
        )


    print()

    print("=" * 72)

    print(
        "XGB CYCLE24 HOLDOUT SHAP — COMPLETE"
    )

    print("=" * 72)


    print(
        "Rows:",
        EXPECTED_HOLDOUT_ROWS
    )

    print(
        "Features:",
        25
    )

    print(
        "Background:",
        BACKGROUND_SIZE
    )

    print(
        "Explainer model input:",
        "FROZEN_XGBOOST_BOOSTER"
    )

    print(
        "Explanation output:",
        "RAW_MARGIN"
    )

    print(
        "Wrapper → Booster equivalence:",
        "PASS"
    )

    print(
        "Native additivity:",
        "PASS"
    )

    print(
        "Cycle25 read:",
        False
    )

    print(
        "Model retrained:",
        False
    )


    print()

    print(
        "Saved:",
        FINAL_OUTPUT_FILE
    )

    print(
        "Saved:",
        XGB_SUMMARY_FILE
    )

    print(
        "Saved:",
        XGB_CHUNK_MANIFEST_FILE
    )

    print(
        "Saved:",
        XGB_AUDIT_CONTRACT_FILE
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Parallel checkpointed XGB SHAP "
            "generation for Chapter 9.6B."
        )
    )


    parser.add_argument(
        "--worker-id",
        type=int,
        default=None,
        help=(
            "Worker ID starting from 0."
        ),
    )


    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help=(
            "Total number of parallel workers."
        ),
    )


    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Merge all completed XGB checkpoints "
            "and create final XGB artifact."
        ),
    )


    args = parser.parse_args()


    if args.finalize:

        finalize(
            total_workers=args.workers
        )

        return


    if args.worker_id is None:

        parser.error(
            "--worker-id is required "
            "unless --finalize is used."
        )


    run_worker(
        worker_id=args.worker_id,
        total_workers=args.workers,
    )


if __name__ == "__main__":

    main()