# ============================================================
# MODEL CHAPTER 9.6B
# PARALLEL RANDOM-FOREST CYCLE24 HOLDOUT SHAP
#
# Purpose:
#   - Compute RF SHAP over all 13,072 Cycle24 HOLDOUT rows
#   - Same frozen RF
#   - Same 25 features
#   - Same frozen 512-row C24 TRAIN background
#   - TreeExplainer(interventional, model_output="raw")
#   - 6 independent workers
#   - 128-row checkpoints
#   - Crash-safe resume
#   - Native SHAP additivity checking
#
# IMPORTANT:
#   This script does NOT read Cycle25.
#   This script does NOT retrain any model.
# ============================================================


# ------------------------------------------------------------
# LIMIT INTERNAL THREADING BEFORE NUMPY / SHAP IMPORT
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


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_ID = "RF_04_D12_L5_BALSUB"
MODEL_FAMILY = "random_forest"

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
    "artifacts/models/random_forest/models/"
    "RF_04_D12_L5_BALSUB.joblib"
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
    "rf_cycle24_holdout"
)


FINAL_OUTPUT_FILE = Path(
    "artifacts/models/shift_controls/shap/cycle24_holdout/"
    "RF_04_D12_L5_BALSUB.parquet"
)


RF_SUMMARY_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "rf_cycle24_holdout_shap_summary.csv"
)


RF_CHUNK_MANIFEST_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "rf_cycle24_holdout_shap_chunk_manifest.csv"
)


RF_AUDIT_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "rf_cycle24_holdout_shap_parallel_audit.json"
)


# ============================================================
# EXPECTED HASHES
# ============================================================

EXPECTED_MODEL_HASH = (
    "519afcdf06e28f1da9b36d112bee9070"
    "bc51748e1c415fe2f3c244bd476bbd31"
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
# HELPERS
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
# SHAP OUTPUT EXTRACTION
# ============================================================

def extract_positive_class_shap(
    values,
    row_count,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    # rows x features
    if values.shape == (
        row_count,
        len(FEATURES),
    ):

        return (
            values,
            "SINGLE_OUTPUT",
        )


    # rows x features x classes
    if (
        values.ndim == 3
        and
        values.shape[0] == row_count
        and
        values.shape[1] == len(FEATURES)
        and
        values.shape[2] >= 2
    ):

        return (
            values[:, :, 1],
            "CLASS_1_LAST_AXIS",
        )


    # rows x classes x features
    if (
        values.ndim == 3
        and
        values.shape[0] == row_count
        and
        values.shape[1] >= 2
        and
        values.shape[2] == len(FEATURES)
    ):

        return (
            values[:, 1, :],
            "CLASS_1_MIDDLE_AXIS",
        )


    raise RuntimeError(
        f"Unexpected SHAP shape: {values.shape}"
    )


def extract_positive_class_base(
    base_values,
    row_count,
    selection_mode,
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

        return base_values


    if base_values.shape == (
        row_count,
        1,
    ):

        return base_values[:, 0]


    if (
        base_values.ndim == 2
        and
        base_values.shape[0] == row_count
        and
        base_values.shape[1] >= 2
    ):

        return base_values[:, 1]


    if (
        base_values.ndim == 1
        and
        len(base_values) >= 2
        and
        selection_mode
        in {
            "CLASS_1_LAST_AXIS",
            "CLASS_1_MIDDLE_AXIS",
        }
    ):

        return np.full(
            row_count,
            float(base_values[1]),
            dtype=np.float64,
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


    raise RuntimeError(
        "Unexpected SHAP base-value shape: "
        f"{base_values.shape}"
    )


# ============================================================
# LOAD + VALIDATE COMMON INPUTS
# ============================================================

def load_common_inputs():

    required = [
        MODEL_FILE,
        HOLDOUT_FILE,
        BACKGROUND_FILE,
        BACKGROUND_CONTRACT_FILE,
    ]


    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required input: {path}"
            )


    # --------------------------------------------------------
    # HASHES
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
            "RF model hash mismatch."
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


    expected_background_hash = (
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
        expected_background_hash
    ):

        raise RuntimeError(
            "Frozen background hash mismatch."
        )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    bundle = joblib.load(
        MODEL_FILE
    )


    if not isinstance(
        bundle,
        dict,
    ):

        raise RuntimeError(
            "Expected RF dictionary bundle."
        )


    if (
        bundle["candidate_id"]
        !=
        CANDIDATE_ID
    ):

        raise RuntimeError(
            "RF candidate ID mismatch."
        )


    if (
        bundle["training_scope"]
        !=
        "CYCLE24_TRAIN_ONLY"
    ):

        raise RuntimeError(
            "RF training scope mismatch."
        )


    if (
        bundle["train_sha256"]
        !=
        EXPECTED_TRAIN_HASH
    ):

        raise RuntimeError(
            "RF TRAIN provenance mismatch."
        )


    if (
        bundle["feature_order"]
        !=
        FEATURES
    ):

        raise RuntimeError(
            "RF feature order mismatch."
        )


    if int(
        bundle["feature_count"]
    ) != 25:

        raise RuntimeError(
            "RF feature count mismatch."
        )


    model = bundle["model"]


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
            "HOLDOUT row count mismatch."
        )


    if int(
        holdout[
            ENTITY_GROUP_COLUMN
        ].nunique()
    ) != EXPECTED_HOLDOUT_GROUPS:

        raise RuntimeError(
            "HOLDOUT ENTITY_GROUP count mismatch."
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
            "HOLDOUT positive count mismatch."
        )


    if negatives != EXPECTED_NEGATIVES:

        raise RuntimeError(
            "HOLDOUT negative count mismatch."
        )


    X_holdout = holdout[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        X_holdout
    ).all():

        raise RuntimeError(
            "Non-finite HOLDOUT features."
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
            "Background row count mismatch."
        )


    if int(
        background_df[
            ENTITY_GROUP_COLUMN
        ].nunique()
    ) != BACKGROUND_SIZE:

        raise RuntimeError(
            "Background must contain "
            "512 unique groups."
        )


    X_background = background_df[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        X_background
    ).all():

        raise RuntimeError(
            "Non-finite background features."
        )


    return {
        "model":
            model,

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
    }


# ============================================================
# CREATE FROZEN EXPLAINER
# ============================================================

def create_explainer(
    model,
    X_background,
):

    masker = shap.maskers.Independent(
        X_background,
        max_samples=BACKGROUND_SIZE,
    )


    if int(
        masker.data.shape[0]
    ) != BACKGROUND_SIZE:

        raise RuntimeError(
            "Masker failed to retain "
            "512 background rows."
        )


    explainer = shap.TreeExplainer(
        model,
        data=masker,
        feature_perturbation="interventional",
        model_output="raw",
    )


    if not hasattr(
        explainer,
        "masker",
    ):

        raise RuntimeError(
            "TreeExplainer masker missing."
        )


    if not hasattr(
        explainer.masker,
        "data",
    ):

        raise RuntimeError(
            "TreeExplainer masker data missing."
        )


    actual_rows = int(
        explainer.masker.data.shape[0]
    )


    if actual_rows != BACKGROUND_SIZE:

        raise RuntimeError(
            "TreeExplainer did not retain "
            "all 512 background rows."
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


    if (
        "source_row_number"
        not in frame.columns
    ):

        raise RuntimeError(
            f"Missing source_row_number: {path}"
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


    for column in SHAP_COLUMNS:

        if column not in frame.columns:

            raise RuntimeError(
                f"Missing {column}: {path}"
            )


    if (
        "shap_base_value"
        not in frame.columns
    ):

        raise RuntimeError(
            f"Missing base value: {path}"
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
            f"Non-finite base checkpoint: {path}"
        )


    return True


# ============================================================
# WORKER
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


    print("=" * 70)

    print(
        "CHAPTER 9.6B — "
        "RF PARALLEL SHAP WORKER"
    )

    print("=" * 70)

    print(
        "Worker:",
        f"{worker_id}/{total_workers - 1}"
    )

    print(
        "Internal numerical threads:",
        1
    )


    common = load_common_inputs()


    model = common["model"]

    holdout = common["holdout"]

    X_holdout = common["X_holdout"]

    X_background = common["X_background"]


    print("Input validation: PASS")
    print("RF model hash: PASS")
    print("Cycle24 HOLDOUT hash: PASS")
    print("Frozen background hash: PASS")
    print("HOLDOUT rows:", len(holdout))
    print("Background rows:", len(X_background))


    explainer = create_explainer(
        model,
        X_background,
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
        "Chunks assigned to this worker:",
        len(assigned_chunks)
    )

    print(
        "Assigned chunk IDs:",
        assigned_chunks
    )


    completed_now = 0
    skipped_existing = 0


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
        # RESUME SUPPORT
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


        # IMPORTANT:
        # Native SHAP additivity validation remains enabled.
        explanation = explainer(
            X_chunk,
            check_additivity=True,
        )


        shap_matrix, selection_mode = (
            extract_positive_class_shap(
                explanation.values,
                len(X_chunk),
            )
        )


        base_vector = (
            extract_positive_class_base(
                explanation.base_values,
                len(X_chunk),
                selection_mode,
            )
        )


        if shap_matrix.shape != (
            len(X_chunk),
            25,
        ):

            raise RuntimeError(
                "SHAP matrix shape mismatch."
            )


        if base_vector.shape != (
            len(X_chunk),
        ):

            raise RuntimeError(
                "SHAP base shape mismatch."
            )


        if not np.isfinite(
            shap_matrix
        ).all():

            raise RuntimeError(
                "Non-finite SHAP values."
            )


        if not np.isfinite(
            base_vector
        ).all():

            raise RuntimeError(
                "Non-finite SHAP base values."
            )


        # ----------------------------------------------------
        # MODEL OUTPUT
        # ----------------------------------------------------

        target_output = model.predict_proba(
            X_chunk
        )[:, 1]


        reconstructed_output = (
            base_vector
            +
            shap_matrix.sum(
                axis=1
            )
        )


        # Same tolerance structure used by native SHAP's
        # TreeExplainer additivity validation.
        if not np.allclose(
            reconstructed_output,
            target_output,
            atol=1e-2,
            rtol=1e-2,
        ):

            max_abs_error = float(
                np.max(
                    np.abs(
                        reconstructed_output
                        -
                        target_output
                    )
                )
            )

            raise RuntimeError(
                "RF chunk additivity validation "
                f"failed. max_abs_error="
                f"{max_abs_error}"
            )


        abs_error = np.abs(
            reconstructed_output
            -
            target_output
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


        # Validate temp artifact BEFORE promotion.
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

    print("=" * 70)

    print(
        f"WORKER {worker_id} COMPLETE"
    )

    print("=" * 70)

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
# FINALIZE ALL RF CHECKPOINTS
# ============================================================

def finalize(
    total_workers: int,
):

    print("=" * 70)

    print(
        "CHAPTER 9.6B — "
        "RF PARALLEL SHAP FINALIZATION"
    )

    print("=" * 70)


    common = load_common_inputs()


    model = common["model"]

    holdout = common["holdout"]

    X_holdout = common["X_holdout"]


    FINAL_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RF_SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RF_CHUNK_MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RF_AUDIT_CONTRACT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    if FINAL_OUTPUT_FILE.exists():

        raise RuntimeError(
            "Final RF SHAP artifact already exists:\n"
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
    # VERIFY EVERY EXPECTED CHUNK
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
                "Missing checkpoint:\n"
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
            "Merged row count mismatch."
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
            "Merged row sequence mismatch."
        )


    if merged[
        "source_row_number"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate rows detected."
        )


    print(
        "Merged row-order audit:",
        "PASS"
    )


    # --------------------------------------------------------
    # SHAP MATRIX
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
            "Merged SHAP shape mismatch."
        )


    if not np.isfinite(
        shap_matrix
    ).all():

        raise RuntimeError(
            "Merged SHAP contains "
            "non-finite values."
        )


    if not np.isfinite(
        base_vector
    ).all():

        raise RuntimeError(
            "Merged base values contain "
            "non-finite values."
        )


    # --------------------------------------------------------
    # FINAL ADDITIVITY AUDIT
    # --------------------------------------------------------

    target_output = model.predict_proba(
        X_holdout
    )[:, 1]


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
            "Final RF native-style "
            "additivity audit failed."
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
    # SAVE FINAL RF ARTIFACT
    # --------------------------------------------------------

    merged.to_parquet(
        FINAL_OUTPUT_FILE,
        index=False,
    )


    # Round trip
    roundtrip = pd.read_parquet(
        FINAL_OUTPUT_FILE
    )


    if len(
        roundtrip
    ) != EXPECTED_HOLDOUT_ROWS:

        raise RuntimeError(
            "Final RF parquet "
            "round-trip failed."
        )


    final_hash = sha256_file(
        FINAL_OUTPUT_FILE
    )


    print(
        "Final RF parquet:",
        "PASS"
    )

    print(
        "RF artifact SHA256:",
        final_hash
    )


    # --------------------------------------------------------
    # RF SUMMARY
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
                ].to_numpy()
                ==
                0
            ),

        "TARGET_1":
            (
                holdout[
                    TARGET_COLUMN
                ].to_numpy()
                ==
                1
            ),
    }


    for stratum, mask in strata.items():

        subset = shap_matrix[
            mask
        ]


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
            "Expected 75 RF summary rows."
        )


    summary_df.to_csv(
        RF_SUMMARY_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # CHUNK MANIFEST
    # --------------------------------------------------------

    manifest_df = pd.DataFrame(
        manifest_rows
    )


    manifest_df.to_csv(
        RF_CHUNK_MANIFEST_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # AUDIT CONTRACT
    # --------------------------------------------------------

    audit_contract = {

        "contract_name":
            (
                "Cycle24 HOLDOUT RF SHAP "
                "Parallel Execution Audit"
            ),

        "version":
            "1.0",

        "status":
            (
                "VALIDATED_RF_COMPONENT_"
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

                "feature_count":
                    25,

                "background_source":
                    "FROZEN_CYCLE24_TRAIN",

                "background_rows":
                    512,

                "explainer":
                    "TreeExplainer",

                "feature_perturbation":
                    "interventional",

                "model_output":
                    "raw",

                "positive_class":
                    1,

                "approximation":
                    False,

                "sampling_reduction":
                    False,

                "cycle25_read":
                    False,

                "model_retrained":
                    False,
            },

        "provenance":
            {
                "model_file":
                    str(MODEL_FILE),

                "model_sha256":
                    common["model_hash"],

                "holdout_file":
                    str(HOLDOUT_FILE),

                "holdout_sha256":
                    common["holdout_hash"],

                "background_file":
                    str(BACKGROUND_FILE),

                "background_sha256":
                    common["background_hash"],
            },

        "additivity":
            {
                "native_style_rule":
                    (
                        "numpy.allclose("
                        "reconstructed_output,"
                        "model_output,"
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

        "previous_diagnostic":
            {
                "rf_internal_vs_sklearn_max_abs_error":
                    2.1094237467877974e-15,

                "rows_over_1e_12":
                    0,

                "interpretation":
                    (
                        "SHAP internal RF model "
                        "representation reproduces "
                        "sklearn RF probability output "
                        "to machine precision."
                    ),
            },

        "artifacts":
            {
                "rf_shap":
                    {
                        "file":
                            str(
                                FINAL_OUTPUT_FILE
                            ),

                        "sha256":
                            final_hash,

                        "rows":
                            EXPECTED_HOLDOUT_ROWS,
                    },

                "rf_summary":
                    {
                        "file":
                            str(
                                RF_SUMMARY_FILE
                            ),

                        "sha256":
                            sha256_file(
                                RF_SUMMARY_FILE
                            ),

                        "rows":
                            75,
                    },

                "chunk_manifest":
                    {
                        "file":
                            str(
                                RF_CHUNK_MANIFEST_FILE
                            ),

                        "sha256":
                            sha256_file(
                                RF_CHUNK_MANIFEST_FILE
                            ),

                        "chunks":
                            total_chunks,
                    },
            },

        "note":
            (
                "Parallelization changes only "
                "execution scheduling. It does not "
                "change evaluation rows, model, "
                "features, SHAP background, explainer "
                "configuration, or scientific analysis."
            ),
    }


    with RF_AUDIT_CONTRACT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit_contract,
            file,
            indent=2,
        )


    print()

    print("=" * 70)

    print(
        "RF CYCLE24 HOLDOUT SHAP — COMPLETE"
    )

    print("=" * 70)

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
        512
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
        RF_SUMMARY_FILE
    )

    print(
        "Saved:",
        RF_CHUNK_MANIFEST_FILE
    )

    print(
        "Saved:",
        RF_AUDIT_CONTRACT_FILE
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Parallel checkpointed RF SHAP "
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
            "Merge all completed checkpoints "
            "and create final RF artifact."
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