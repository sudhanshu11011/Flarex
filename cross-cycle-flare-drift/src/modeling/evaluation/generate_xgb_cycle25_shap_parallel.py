# ============================================================
# MODEL CHAPTER 10.2.3
# XGBOOST — CYCLE25 PARALLEL SHAP
#
# Frozen transfer experiment:
#
# Candidate:
#   XGB_02_LR0p03_D3_BAL
#
# Model:
#   exact fitted Booster extracted from frozen XGB wrapper
#
# Training:
#   Cycle24 TRAIN only
#
# Evaluation:
#   full immutable Cycle25 dataset_v2
#
# SHAP:
#   TreeExplainer
#   interventional
#   frozen 512-row Cycle24 TRAIN background
#   raw XGBoost margin output
#
# NO:
#   retraining
#   retuning
#   Cycle25 fitting
#   approximation
#   sampling reduction
#   background change
#   drift conclusion
# ============================================================


# ============================================================
# LIMIT INTERNAL NUMERICAL THREADING
# ============================================================

import os

for variable in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:

    os.environ[variable] = "1"


# ============================================================
# IMPORTS
# ============================================================

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
# FROZEN CONSTANTS
# ============================================================

CANDIDATE_ID = "XGB_02_LR0p03_D3_BAL"

MODEL_FAMILY = "xgboost"

EXPECTED_ROWS = 79361

EXPECTED_FEATURES = 25

EXPECTED_BACKGROUND_ROWS = 512


# Execution-only setting.
# Does not change scientific computation.
CHUNK_SIZE = 256


EXPECTED_MODEL_HASH = (
    "5839ac44333265b1b771008a82f6fc5a"
    "2f722e992386cde93fe8c04d48443d8b"
)


EXPECTED_C25_HASH = (
    "0bd6a364e152685d13c7eb955da074a4"
    "b9f019f1851c6b5dafebf295c406cd4d"
)


EXPECTED_C24_TRAIN_HASH = (
    "4801a6e3b11d39cef8d4b0ef0eea4ae1"
    "67c27862431662c3f43ff55e4c433def"
)


# ============================================================
# ADDITIVITY
#
# Native SHAP check remains enabled.
#
# Secondary audit tolerance is intentionally tighter than the
# RF probability-space audit, while still safely above the
# ~1e-5 numerical residual observed in frozen Cycle24 XGB SHAP.
# ============================================================

ADDITIVITY_ATOL = 1e-4

ADDITIVITY_RTOL = 1e-4

RELATIVE_DIAGNOSTIC_STABILIZER = 1e-2


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


TARGET_COLUMN = "target"

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"

TIME_COLUMN = "observation_time_utc"


# ============================================================
# INPUT PATHS
# ============================================================

EXECUTION_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "cycle25_shap_execution_inputs_contract.json"
)


MODEL_MANIFEST_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "cycle25_shap_model_manifest.csv"
)


POPULATION_MANIFEST_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "cycle25_shap_population_manifest.csv"
)


C25_DATASET = Path(
    "data/final/dataset_v2/"
    "cycle25_dataset_v2.parquet"
)


MODEL_FILE = Path(
    "artifacts/models/xgboost/models/"
    "XGB_02_LR0p03_D3_BAL.joblib"
)


BACKGROUND_FILE = Path(
    "artifacts/models/shift_controls/shap/background/"
    "cycle24_train_shap_background.parquet"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

CHECKPOINT_DIR = Path(
    "artifacts/models/cross_cycle_shap/shap/"
    "checkpoints/xgb_cycle25"
)


FINAL_FILE = Path(
    "artifacts/models/cross_cycle_shap/shap/cycle25/"
    "XGB_02_LR0p03_D3_BAL.parquet"
)


SUMMARY_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "xgb_cycle25_shap_summary.csv"
)


CHUNK_MANIFEST_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "xgb_cycle25_shap_chunk_manifest.csv"
)


AUDIT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "xgb_cycle25_shap_parallel_audit.json"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def verify_exists(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


# ============================================================
# CYCLE25 BLOCK LABELS
# ============================================================

def build_block_labels(
    frame: pd.DataFrame,
) -> pd.Series:

    timestamps = pd.to_datetime(
        frame[
            TIME_COLUMN
        ],
        utc=True,
        errors="raise",
    )


    years = timestamps.dt.year


    labels = pd.Series(
        index=frame.index,
        dtype="object",
    )


    labels.loc[
        years.isin(
            [2020, 2021]
        )
    ] = "B1"


    labels.loc[
        years.isin(
            [2022, 2023]
        )
    ] = "B2"


    labels.loc[
        years.isin(
            [2024, 2025]
        )
    ] = "B3"


    if labels.isna().any():

        raise RuntimeError(
            "Cycle25 temporal block assignment failed."
        )


    return labels


# ============================================================
# EXACT 10.1 POPULATION FINGERPRINT
# ============================================================

def ordered_population_fingerprint(
    frame: pd.DataFrame,
    block_labels: pd.Series,
) -> str:

    digest = hashlib.sha256()


    timestamps = pd.to_datetime(
        frame[
            TIME_COLUMN
        ],
        utc=True,
    )


    groups = frame[
        ENTITY_GROUP_COLUMN
    ].astype(str)


    targets = frame[
        TARGET_COLUMN
    ].astype(int)


    for (
        row_number,
        timestamp,
        group,
        target,
        block,
    ) in zip(
        range(
            len(frame)
        ),
        timestamps,
        groups,
        targets,
        block_labels,
    ):

        line = (
            f"{row_number}|"
            f"{timestamp.isoformat()}|"
            f"{group}|"
            f"{target}|"
            f"{block}\n"
        )


        digest.update(
            line.encode(
                "utf-8"
            )
        )


    return digest.hexdigest()


# ============================================================
# BASE VALUE NORMALIZATION
# ============================================================

def normalize_base_values(
    values,
    n_rows: int,
) -> np.ndarray:

    base = np.asarray(
        values,
        dtype=np.float64,
    )


    if base.ndim == 0:

        result = np.full(
            n_rows,
            float(base),
            dtype=np.float64,
        )


    else:

        base = np.squeeze(
            base
        )


        if base.ndim == 0:

            result = np.full(
                n_rows,
                float(base),
                dtype=np.float64,
            )


        elif base.size == 1:

            result = np.full(
                n_rows,
                float(
                    base.reshape(-1)[0]
                ),
                dtype=np.float64,
            )


        elif base.shape == (
            n_rows,
        ):

            result = base.astype(
                np.float64,
                copy=False,
            )


        else:

            raise RuntimeError(
                "Unexpected XGB base-value shape: "
                f"{base.shape}"
            )


    if result.shape != (
        n_rows,
    ):

        raise RuntimeError(
            "XGB base-value final shape mismatch."
        )


    if not np.isfinite(
        result
    ).all():

        raise RuntimeError(
            "XGB base values contain non-finite values."
        )


    return result


# ============================================================
# CHECKPOINT PATH
# ============================================================

def checkpoint_path(
    chunk_id: int,
    start: int,
    stop: int,
) -> Path:

    return (
        CHECKPOINT_DIR
        /
        (
            f"chunk_{chunk_id:03d}_"
            f"{start:05d}_"
            f"{stop - 1:05d}.parquet"
        )
    )


# ============================================================
# CHUNK HELPERS
# ============================================================

def total_chunks() -> int:

    return math.ceil(
        EXPECTED_ROWS
        /
        CHUNK_SIZE
    )


def chunk_bounds(
    chunk_id: int,
):

    start = (
        chunk_id
        *
        CHUNK_SIZE
    )


    stop = min(
        start
        +
        CHUNK_SIZE,
        EXPECTED_ROWS,
    )


    return (
        start,
        stop,
    )


# ============================================================
# LOAD + VERIFY ALL FROZEN INPUTS
# ============================================================

def load_common_inputs():

    required_files = [
        EXECUTION_CONTRACT_FILE,
        MODEL_MANIFEST_FILE,
        POPULATION_MANIFEST_FILE,
        C25_DATASET,
        MODEL_FILE,
        BACKGROUND_FILE,
    ]


    for path in required_files:

        verify_exists(
            path
        )


    # --------------------------------------------------------
    # CHAPTER 10.1 CONTRACT
    # --------------------------------------------------------

    contract = load_json(
        EXECUTION_CONTRACT_FILE
    )


    if (
        contract.get(
            "status"
        )
        !=
        "FROZEN_CYCLE25_SHAP_EXECUTION_INPUTS"
    ):

        raise RuntimeError(
            "Chapter 10.1 execution contract "
            "is not frozen."
        )


    # --------------------------------------------------------
    # POPULATION MANIFEST
    # --------------------------------------------------------

    population_manifest_hash = sha256_file(
        POPULATION_MANIFEST_FILE
    )


    if (
        population_manifest_hash
        !=
        contract[
            "temporal_blocks"
        ][
            "population_manifest"
        ][
            "sha256"
        ]
    ):

        raise RuntimeError(
            "Cycle25 population manifest changed."
        )


    # --------------------------------------------------------
    # MODEL MANIFEST
    # --------------------------------------------------------

    model_manifest_hash = sha256_file(
        MODEL_MANIFEST_FILE
    )


    if (
        model_manifest_hash
        !=
        contract[
            "frozen_models"
        ][
            "model_manifest"
        ][
            "sha256"
        ]
    ):

        raise RuntimeError(
            "Frozen model manifest changed."
        )


    model_manifest = pd.read_csv(
        MODEL_MANIFEST_FILE,
        keep_default_na=False,
    )


    row = model_manifest[
        model_manifest[
            "candidate_id"
        ].astype(str)
        ==
        CANDIDATE_ID
    ]


    if len(
        row
    ) != 1:

        raise RuntimeError(
            "Frozen XGB model manifest lookup failed."
        )


    row = row.iloc[
        0
    ]


    if (
        str(
            row[
                "model_family"
            ]
        )
        !=
        MODEL_FAMILY
    ):

        raise RuntimeError(
            "XGB model-family contract changed."
        )


    if (
        str(
            row[
                "explainer"
            ]
        )
        !=
        "TreeExplainer"
    ):

        raise RuntimeError(
            "XGB explainer contract changed."
        )


    if (
        str(
            row[
                "model_input_representation"
            ]
        )
        !=
        "FROZEN_XGBOOST_BOOSTER"
    ):

        raise RuntimeError(
            "XGB model representation changed."
        )


    if (
        str(
            row[
                "feature_input_space"
            ]
        )
        !=
        "RAW"
    ):

        raise RuntimeError(
            "XGB input-space contract changed."
        )


    if (
        str(
            row[
                "output_space"
            ]
        )
        !=
        "XGBOOST_RAW_MARGIN"
    ):

        raise RuntimeError(
            "XGB output-space contract changed."
        )


    if (
        str(
            row[
                "feature_perturbation"
            ]
        )
        !=
        "INTERVENTIONAL"
    ):

        raise RuntimeError(
            "XGB perturbation contract changed."
        )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset_hash = sha256_file(
        C25_DATASET
    )


    if dataset_hash != EXPECTED_C25_HASH:

        raise RuntimeError(
            "Cycle25 hard-frozen hash mismatch."
        )


    if (
        dataset_hash
        !=
        contract[
            "cycle25_population"
        ][
            "dataset_sha256"
        ]
    ):

        raise RuntimeError(
            "Cycle25 differs from 10.1 contract."
        )


    c25 = pd.read_parquet(
        C25_DATASET
    )


    if len(
        c25
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            "Cycle25 row-count mismatch."
        )


    required_columns = {
        TARGET_COLUMN,
        ENTITY_GROUP_COLUMN,
        TIME_COLUMN,
        *FEATURES,
    }


    missing = (
        required_columns
        -
        set(
            c25.columns
        )
    )


    if missing:

        raise RuntimeError(
            "Cycle25 missing columns: "
            f"{sorted(missing)}"
        )


    X = c25[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )


    if X.shape != (
        EXPECTED_ROWS,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            "Cycle25 XGB input shape mismatch."
        )


    if not np.isfinite(
        X
    ).all():

        raise RuntimeError(
            "Cycle25 contains non-finite XGB inputs."
        )


    # --------------------------------------------------------
    # BLOCKS + ORDER
    # --------------------------------------------------------

    block_labels = build_block_labels(
        c25
    )


    population_hash = (
        ordered_population_fingerprint(
            frame=c25,
            block_labels=block_labels,
        )
    )


    if (
        population_hash
        !=
        contract[
            "cycle25_population"
        ][
            "ordered_population_sha256"
        ]
    ):

        raise RuntimeError(
            "Cycle25 row order changed after 10.1."
        )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model_hash = sha256_file(
        MODEL_FILE
    )


    if model_hash != EXPECTED_MODEL_HASH:

        raise RuntimeError(
            "XGB hard-frozen model hash mismatch."
        )


    if (
        model_hash
        !=
        str(
            row[
                "model_sha256"
            ]
        )
    ):

        raise RuntimeError(
            "XGB model differs from 10.1 manifest."
        )


    bundle = joblib.load(
        MODEL_FILE
    )


    if (
        bundle[
            "candidate_id"
        ]
        !=
        CANDIDATE_ID
    ):

        raise RuntimeError(
            "XGB candidate ID mismatch."
        )


    if (
        bundle[
            "training_scope"
        ]
        !=
        "CYCLE24_TRAIN_ONLY"
    ):

        raise RuntimeError(
            "XGB training scope changed."
        )


    if (
        bundle[
            "train_sha256"
        ]
        !=
        EXPECTED_C24_TRAIN_HASH
    ):

        raise RuntimeError(
            "XGB TRAIN provenance mismatch."
        )


    if (
        bundle[
            "feature_order"
        ]
        !=
        FEATURES
    ):

        raise RuntimeError(
            "XGB feature order changed."
        )


    wrapper = bundle[
        "model"
    ]


    booster = wrapper.get_booster()


    # --------------------------------------------------------
    # WRAPPER <-> BOOSTER EQUIVALENCE
    #
    # Full Cycle25 validation.
    # No fitting.
    # --------------------------------------------------------

    wrapper_margin = np.asarray(
        wrapper.predict(
            X,
            output_margin=True,
        ),
        dtype=np.float64,
    ).reshape(-1)


    booster_margin = np.asarray(
        booster.inplace_predict(
            X,
            predict_type="margin",
        ),
        dtype=np.float64,
    ).reshape(-1)


    wrapper_booster_difference = np.abs(
        wrapper_margin
        -
        booster_margin
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
            "Frozen XGB wrapper -> Booster "
            "equivalence failed."
        )


    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    background_hash = sha256_file(
        BACKGROUND_FILE
    )


    if (
        background_hash
        !=
        contract[
            "shap_background"
        ][
            "sha256"
        ]
    ):

        raise RuntimeError(
            "Frozen SHAP background changed."
        )


    if (
        background_hash
        !=
        str(
            row[
                "background_sha256"
            ]
        )
    ):

        raise RuntimeError(
            "XGB manifest background hash mismatch."
        )


    background = pd.read_parquet(
        BACKGROUND_FILE
    )


    if len(
        background
    ) != EXPECTED_BACKGROUND_ROWS:

        raise RuntimeError(
            "Expected exact 512-row background."
        )


    X_background = background[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )


    if X_background.shape != (
        EXPECTED_BACKGROUND_ROWS,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            "XGB background shape mismatch."
        )


    if not np.isfinite(
        X_background
    ).all():

        raise RuntimeError(
            "XGB background contains non-finite values."
        )


    return {
        "contract":
            contract,

        "manifest_row":
            row,

        "c25":
            c25,

        "X":
            X,

        "block_labels":
            block_labels,

        "wrapper":
            wrapper,

        "booster":
            booster,

        "dataset_hash":
            dataset_hash,

        "model_hash":
            model_hash,

        "background_hash":
            background_hash,

        "X_background":
            X_background,

        "population_hash":
            population_hash,

        "wrapper_booster_mean_abs_error":
            float(
                wrapper_booster_difference.mean()
            ),

        "wrapper_booster_max_abs_error":
            float(
                wrapper_booster_difference.max()
            ),
    }


# ============================================================
# CREATE FROZEN XGB TREE EXPLAINER
# ============================================================

def create_explainer(
    booster,
    X_background: np.ndarray,
):

    masker = shap.maskers.Independent(
        X_background,
        max_samples=EXPECTED_BACKGROUND_ROWS,
    )


    if (
        masker.data.shape[
            0
        ]
        !=
        EXPECTED_BACKGROUND_ROWS
    ):

        raise RuntimeError(
            "XGB masker failed exact "
            "512-row background requirement."
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
            "XGB TreeExplainer does not expose masker."
        )


    if (
        int(
            explainer.masker.data.shape[
                0
            ]
        )
        !=
        EXPECTED_BACKGROUND_ROWS
    ):

        raise RuntimeError(
            "XGB TreeExplainer changed "
            "the frozen 512-row background."
        )


    return explainer


# ============================================================
# CHECKPOINT VALIDATION
# ============================================================

def validate_checkpoint(
    path: Path,
    start: int,
    stop: int,
    c25: pd.DataFrame,
    block_labels: pd.Series,
):

    frame = pd.read_parquet(
        path
    )


    expected_rows = (
        stop
        -
        start
    )


    if len(
        frame
    ) != expected_rows:

        raise RuntimeError(
            f"Checkpoint row-count mismatch: {path}"
        )


    expected_source_rows = np.arange(
        start,
        stop,
        dtype=np.int64,
    )


    if not np.array_equal(
        frame[
            "source_row_number"
        ].to_numpy(
            dtype=np.int64
        ),
        expected_source_rows,
    ):

        raise RuntimeError(
            f"Checkpoint row-order mismatch: {path}"
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


    expected_groups = c25.iloc[
        start:stop
    ][
        ENTITY_GROUP_COLUMN
    ].astype(str).to_numpy()


    if not np.array_equal(
        frame[
            ENTITY_GROUP_COLUMN
        ].astype(str).to_numpy(),
        expected_groups,
    ):

        raise RuntimeError(
            f"Checkpoint entity-group mismatch: {path}"
        )


    expected_blocks = block_labels.iloc[
        start:stop
    ].astype(str).to_numpy()


    if not np.array_equal(
        frame[
            "cycle25_block"
        ].astype(str).to_numpy(),
        expected_blocks,
    ):

        raise RuntimeError(
            f"Checkpoint block mismatch: {path}"
        )


    shap_matrix = frame[
        SHAP_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )


    if shap_matrix.shape != (
        expected_rows,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            f"Checkpoint SHAP shape mismatch: {path}"
        )


    if not np.isfinite(
        shap_matrix
    ).all():

        raise RuntimeError(
            f"Checkpoint contains non-finite SHAP: {path}"
        )


    base_values = frame[
        "shap_base_value"
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        base_values
    ).all():

        raise RuntimeError(
            f"Checkpoint contains non-finite base values: "
            f"{path}"
        )


    return frame


# ============================================================
# WORKER
# ============================================================

def run_worker(
    workers: int,
    worker_id: int,
    max_new_chunks=None,
):

    if workers < 1:

        raise RuntimeError(
            "--workers must be >= 1."
        )


    if (
        worker_id < 0
        or
        worker_id >= workers
    ):

        raise RuntimeError(
            "worker-id must satisfy "
            "0 <= worker-id < workers."
        )


    if (
        max_new_chunks is not None
        and
        max_new_chunks < 1
    ):

        raise RuntimeError(
            "--max-new-chunks must be >= 1."
        )


    print("=" * 78)

    print(
        "MODEL CHAPTER 10.2.3 — "
        "XGB CYCLE25 PARALLEL SHAP WORKER"
    )

    print("=" * 78)


    print(
        "Worker:",
        f"{worker_id}/{workers - 1}"
    )

    print(
        "SHAP runtime:",
        shap.__version__
    )

    print(
        "Internal numerical threads:",
        1
    )


    inputs = load_common_inputs()


    c25 = inputs[
        "c25"
    ]


    X = inputs[
        "X"
    ]


    block_labels = inputs[
        "block_labels"
    ]


    booster = inputs[
        "booster"
    ]


    X_background = inputs[
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
        "Cycle25 dataset hash:",
        "PASS"
    )

    print(
        "Cycle25 ordered population:",
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
        "Wrapper -> Booster equivalence:",
        "PASS"
    )

    print(
        "Wrapper -> Booster max abs error:",
        f"{inputs['wrapper_booster_max_abs_error']:.12f}"
    )

    print(
        "Cycle25 rows:",
        EXPECTED_ROWS
    )

    print(
        "Background rows:",
        EXPECTED_BACKGROUND_ROWS
    )


    # --------------------------------------------------------
    # EXPLAINER
    # --------------------------------------------------------

    explainer = create_explainer(
        booster,
        X_background,
    )


    print(
        "TreeExplainer background rows:",
        int(
            explainer.masker.data.shape[
                0
            ]
        )
    )

    print(
        "TreeExplainer background audit:",
        "PASS"
    )


    # --------------------------------------------------------
    # CHUNK ASSIGNMENT
    # --------------------------------------------------------

    n_chunks = total_chunks()


    assigned_chunks = [
        chunk_id
        for chunk_id
        in range(
            n_chunks
        )
        if (
            chunk_id
            %
            workers
            ==
            worker_id
        )
    ]


    print(
        "Chunk size:",
        CHUNK_SIZE
    )

    print(
        "Total global chunks:",
        n_chunks
    )

    print(
        "Chunks assigned:",
        len(
            assigned_chunks
        )
    )

    print(
        "Assigned chunk IDs:",
        assigned_chunks
    )


    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    new_chunks = 0

    reused_chunks = 0


    for position, chunk_id in enumerate(
        assigned_chunks,
        start=1,
    ):

        start, stop = chunk_bounds(
            chunk_id
        )


        path = checkpoint_path(
            chunk_id,
            start,
            stop,
        )


        print()

        print(
            f"[Worker {worker_id}] "
            f"Chunk {chunk_id} "
            f"({position}/{len(assigned_chunks)}) "
            f"rows {start}–{stop - 1}"
        )


        # ----------------------------------------------------
        # REUSE CHECKPOINT
        # ----------------------------------------------------

        if path.exists():

            validate_checkpoint(
                path=path,
                start=start,
                stop=stop,
                c25=c25,
                block_labels=block_labels,
            )


            reused_chunks += 1


            print(
                "Existing checkpoint:",
                "PASS — reused"
            )


            continue


        # ----------------------------------------------------
        # PREFLIGHT LIMIT
        # ----------------------------------------------------

        if (
            max_new_chunks is not None
            and
            new_chunks >= max_new_chunks
        ):

            print()

            print(
                "Requested max-new-chunks reached."
            )

            break


        X_chunk = X[
            start:stop
        ]


        # ----------------------------------------------------
        # EXACT SHAP
        # ----------------------------------------------------

        explanation = explainer(
            X_chunk,
            check_additivity=True,
        )


        shap_values = np.asarray(
            explanation.values,
            dtype=np.float64,
        )


        if shap_values.shape != (
            stop - start,
            EXPECTED_FEATURES,
        ):

            raise RuntimeError(
                "Unexpected XGB SHAP shape: "
                f"{shap_values.shape}"
            )


        if not np.isfinite(
            shap_values
        ).all():

            raise RuntimeError(
                "XGB SHAP contains non-finite values."
            )


        base_values = normalize_base_values(
            explanation.base_values,
            stop - start,
        )


        # ----------------------------------------------------
        # RAW-MARGIN ADDITIVITY
        # ----------------------------------------------------

        target_margin = np.asarray(
            booster.inplace_predict(
                X_chunk,
                predict_type="margin",
            ),
            dtype=np.float64,
        ).reshape(-1)


        reconstructed_margin = (
            base_values
            +
            shap_values.sum(
                axis=1
            )
        )


        absolute_error = np.abs(
            reconstructed_margin
            -
            target_margin
        )


        additivity_pass = bool(
            np.allclose(
                reconstructed_margin,
                target_margin,
                atol=ADDITIVITY_ATOL,
                rtol=ADDITIVITY_RTOL,
            )
        )


        if not additivity_pass:

            raise RuntimeError(
                f"Chunk {chunk_id}: XGB raw-margin "
                "additivity failed. "
                f"Mean abs="
                f"{float(absolute_error.mean())}, "
                f"max abs="
                f"{float(absolute_error.max())}"
            )


        print(
            "Chunk additivity:",
            "PASS"
        )

        print(
            "Chunk max abs error:",
            f"{float(absolute_error.max()):.10f}"
        )


        # ----------------------------------------------------
        # BUILD CHECKPOINT
        # ----------------------------------------------------

        source = c25.iloc[
            start:stop
        ]


        if (
            "ENTITY_GROUP"
            in
            source.columns
        ):

            entity_group_alias = (
                source[
                    "ENTITY_GROUP"
                ]
                .astype(str)
                .to_numpy()
            )

        else:

            entity_group_alias = (
                source[
                    ENTITY_GROUP_COLUMN
                ]
                .astype(str)
                .to_numpy()
            )


        checkpoint = pd.DataFrame(
            {
                "source_row_number":
                    np.arange(
                        start,
                        stop,
                        dtype=np.int64,
                    ),

                "candidate_id":
                    CANDIDATE_ID,

                "model_family":
                    MODEL_FAMILY,

                "cycle25_block":
                    block_labels.iloc[
                        start:stop
                    ].astype(str).to_numpy(),

                "ENTITY_GROUP":
                    entity_group_alias,

                ENTITY_GROUP_COLUMN:
                    source[
                        ENTITY_GROUP_COLUMN
                    ].astype(str).to_numpy(),

                TARGET_COLUMN:
                    source[
                        TARGET_COLUMN
                    ].to_numpy(
                        dtype=np.int8
                    ),

                TIME_COLUMN:
                    pd.to_datetime(
                        source[
                            TIME_COLUMN
                        ],
                        utc=True,
                    ).to_numpy(),

                "shap_base_value":
                    base_values,
            }
        )


        for (
            feature_index,
            shap_column,
        ) in enumerate(
            SHAP_COLUMNS
        ):

            checkpoint[
                shap_column
            ] = shap_values[
                :,
                feature_index
            ]


        # ----------------------------------------------------
        # ATOMIC WRITE
        # ----------------------------------------------------

        temp_path = path.with_name(
            path.stem
            +
            ".tmp.parquet"
        )


        if temp_path.exists():

            temp_path.unlink()


        checkpoint.to_parquet(
            temp_path,
            index=False,
        )


        validate_checkpoint(
            path=temp_path,
            start=start,
            stop=stop,
            c25=c25,
            block_labels=block_labels,
        )


        temp_path.replace(
            path
        )


        new_chunks += 1


        print(
            "Checkpoint saved:",
            path
        )


    print()

    print("=" * 78)

    print(
        f"XGB CYCLE25 WORKER "
        f"{worker_id} COMPLETE"
    )

    print("=" * 78)


    print(
        "New chunks computed:",
        new_chunks
    )

    print(
        "Existing chunks reused:",
        reused_chunks
    )

    print(
        "Cycle25 read:",
        True
    )

    print(
        "Cycle25 used for fitting:",
        False
    )

    print(
        "Model retrained:",
        False
    )

    print(
        "Booster is exact fitted model:",
        True
    )

    print(
        "SHAP approximation:",
        False
    )

    print(
        "Sampling reduction:",
        False
    )


# ============================================================
# FINALIZE
# ============================================================

def finalize(
    workers: int,
):

    print("=" * 78)

    print(
        "MODEL CHAPTER 10.2.3 — "
        "XGB CYCLE25 SHAP FINALIZATION"
    )

    print("=" * 78)


    for path in [
        FINAL_FILE,
        SUMMARY_FILE,
        CHUNK_MANIFEST_FILE,
        AUDIT_FILE,
    ]:

        if path.exists():

            raise RuntimeError(
                f"Final output already exists: {path}\n"
                "Refusing to overwrite completed "
                "XGB Cycle25 SHAP output."
            )


    inputs = load_common_inputs()


    c25 = inputs[
        "c25"
    ]


    X = inputs[
        "X"
    ]


    block_labels = inputs[
        "block_labels"
    ]


    booster = inputs[
        "booster"
    ]


    n_chunks = total_chunks()


    frames = []

    manifest_rows = []


    # --------------------------------------------------------
    # REQUIRE ALL CHECKPOINTS
    # --------------------------------------------------------

    for chunk_id in range(
        n_chunks
    ):

        start, stop = chunk_bounds(
            chunk_id
        )


        path = checkpoint_path(
            chunk_id,
            start,
            stop,
        )


        if not path.exists():

            raise RuntimeError(
                "Missing required XGB checkpoint:\n"
                f"{path}"
            )


        frame = validate_checkpoint(
            path=path,
            start=start,
            stop=stop,
            c25=c25,
            block_labels=block_labels,
        )


        frames.append(
            frame
        )


        manifest_rows.append(
            {
                "chunk_id":
                    chunk_id,

                "worker_id":
                    (
                        chunk_id
                        %
                        workers
                    ),

                "start_row":
                    start,

                "stop_row_exclusive":
                    stop,

                "rows":
                    stop - start,

                "file":
                    path.as_posix(),

                "sha256":
                    sha256_file(
                        path
                    ),

                "integrity_pass":
                    True,
            }
        )


    print(
        "All expected checkpoints:",
        "PASS"
    )

    print(
        "Checkpoint count:",
        len(
            frames
        )
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
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            "Merged XGB row-count mismatch."
        )


    expected_rows = np.arange(
        EXPECTED_ROWS,
        dtype=np.int64,
    )


    if not np.array_equal(
        merged[
            "source_row_number"
        ].to_numpy(
            dtype=np.int64
        ),
        expected_rows,
    ):

        raise RuntimeError(
            "Merged XGB row-order audit failed."
        )


    if merged[
        "source_row_number"
    ].duplicated().any():

        raise RuntimeError(
            "Merged XGB contains duplicate rows."
        )


    print(
        "Merged row-order audit:",
        "PASS"
    )


    # --------------------------------------------------------
    # FULL ADDITIVITY
    # --------------------------------------------------------

    shap_matrix = merged[
        SHAP_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )


    base_values = merged[
        "shap_base_value"
    ].to_numpy(
        dtype=np.float64
    )


    if shap_matrix.shape != (
        EXPECTED_ROWS,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            "Final XGB SHAP matrix shape mismatch."
        )


    target_margin = np.asarray(
        booster.inplace_predict(
            X,
            predict_type="margin",
        ),
        dtype=np.float64,
    ).reshape(-1)


    reconstructed_margin = (
        base_values
        +
        shap_matrix.sum(
            axis=1
        )
    )


    absolute_error = np.abs(
        reconstructed_margin
        -
        target_margin
    )


    additivity_pass = bool(
        np.allclose(
            reconstructed_margin,
            target_margin,
            atol=ADDITIVITY_ATOL,
            rtol=ADDITIVITY_RTOL,
        )
    )


    if not additivity_pass:

        raise RuntimeError(
            "Final XGB Cycle25 raw-margin "
            "additivity failed."
        )


    relative_error = (
        absolute_error
        /
        (
            np.abs(
                reconstructed_margin
            )
            +
            RELATIVE_DIAGNOSTIC_STABILIZER
        )
    )


    print(
        "Final raw-margin additivity:",
        "PASS"
    )

    print(
        "Mean absolute error:",
        f"{float(absolute_error.mean()):.10f}"
    )

    print(
        "P95 absolute error:",
        f"{float(np.quantile(absolute_error, 0.95)):.10f}"
    )

    print(
        "P99 absolute error:",
        f"{float(np.quantile(absolute_error, 0.99)):.10f}"
    )

    print(
        "Max absolute error:",
        f"{float(absolute_error.max()):.10f}"
    )

    print(
        "Max relative diagnostic:",
        f"{float(relative_error.max()):.6%}"
    )

    print(
        "Rows >1% relative diagnostic:",
        int(
            np.sum(
                relative_error > 0.01
            )
        )
    )


    # --------------------------------------------------------
    # SAVE FINAL PARQUET
    # --------------------------------------------------------

    FINAL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temp_final = FINAL_FILE.with_name(
        FINAL_FILE.stem
        +
        ".tmp.parquet"
    )


    if temp_final.exists():

        temp_final.unlink()


    merged.to_parquet(
        temp_final,
        index=False,
    )


    roundtrip = pd.read_parquet(
        temp_final
    )


    if len(
        roundtrip
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            "XGB final parquet round-trip failed."
        )


    temp_final.replace(
        FINAL_FILE
    )


    final_hash = sha256_file(
        FINAL_FILE
    )


    print(
        "Final XGB Cycle25 parquet:",
        "PASS"
    )

    print(
        "XGB artifact SHA256:",
        final_hash
    )


    # --------------------------------------------------------
    # POOLED SUMMARY
    # --------------------------------------------------------

    target = c25[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.int8
    )


    strata = {
        "OVERALL":
            np.ones(
                EXPECTED_ROWS,
                dtype=bool,
            ),

        "TARGET_0":
            target == 0,

        "TARGET_1":
            target == 1,
    }


    summary_rows = []


    for stratum, mask in strata.items():

        subset = shap_matrix[
            mask
        ]


        mean_abs = np.mean(
            np.abs(
                subset
            ),
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


        for (
            feature_index,
            feature,
        ) in enumerate(
            FEATURES,
            start=1,
        ):

            summary_rows.append(
                {
                    "candidate_id":
                        CANDIDATE_ID,

                    "model_family":
                        MODEL_FAMILY,

                    "population":
                        "CYCLE25_POOLED",

                    "stratum":
                        stratum,

                    "rows":
                        int(
                            mask.sum()
                        ),

                    "entity_groups":
                        int(
                            c25.loc[
                                mask,
                                ENTITY_GROUP_COLUMN,
                            ]
                            .astype(str)
                            .nunique()
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


    summary = pd.DataFrame(
        summary_rows
    )


    if len(
        summary
    ) != 75:

        raise RuntimeError(
            "Expected 75 XGB summary rows."
        )


    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # CHUNK MANIFEST
    # --------------------------------------------------------

    chunk_manifest = pd.DataFrame(
        manifest_rows
    )


    if len(
        chunk_manifest
    ) != n_chunks:

        raise RuntimeError(
            "XGB chunk-manifest count mismatch."
        )


    CHUNK_MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    chunk_manifest.to_csv(
        CHUNK_MANIFEST_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    audit = {

        "audit_name":
            "Cycle25 XGBoost Parallel SHAP",

        "version":
            "1.0",

        "status":
            "COMPLETE_XGB_CYCLE25_SHAP",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "chapter":
            "MODEL_CHAPTER_10_2_3",

        "source_execution_contract":
            {
                "file":
                    EXECUTION_CONTRACT_FILE.as_posix(),

                "sha256":
                    sha256_file(
                        EXECUTION_CONTRACT_FILE
                    ),

                "required_status":
                    "FROZEN_CYCLE25_SHAP_EXECUTION_INPUTS",
            },

        "candidate":
            {
                "candidate_id":
                    CANDIDATE_ID,

                "model_family":
                    MODEL_FAMILY,

                "model_file":
                    MODEL_FILE.as_posix(),

                "model_sha256":
                    inputs[
                        "model_hash"
                    ],

                "training_scope":
                    "CYCLE24_TRAIN_ONLY",

                "shap_model_representation":
                    "EXACT_FITTED_XGBOOST_BOOSTER",

                "wrapper_booster_equivalence":
                    True,

                "wrapper_booster_mean_abs_error":
                    inputs[
                        "wrapper_booster_mean_abs_error"
                    ],

                "wrapper_booster_max_abs_error":
                    inputs[
                        "wrapper_booster_max_abs_error"
                    ],
            },

        "evaluation_population":
            {
                "dataset":
                    C25_DATASET.as_posix(),

                "dataset_sha256":
                    inputs[
                        "dataset_hash"
                    ],

                "ordered_population_sha256":
                    inputs[
                        "population_hash"
                    ],

                "rows":
                    EXPECTED_ROWS,
            },

        "shap":
            {
                "explainer":
                    "TreeExplainer",

                "feature_perturbation":
                    "interventional",

                "feature_input_space":
                    "RAW",

                "background_input_space":
                    "RAW",

                "background_file":
                    BACKGROUND_FILE.as_posix(),

                "background_sha256":
                    inputs[
                        "background_hash"
                    ],

                "background_rows":
                    EXPECTED_BACKGROUND_ROWS,

                "output_space":
                    "XGBOOST_RAW_MARGIN",

                "approximate":
                    False,

                "sampling_reduction":
                    False,
            },

        "parallel_execution":
            {
                "workers":
                    workers,

                "internal_threads_per_worker":
                    1,

                "chunk_size":
                    CHUNK_SIZE,

                "global_chunks":
                    n_chunks,

                "checkpointed":
                    True,
            },

        "additivity":
            {
                "native_shap_check_enabled":
                    True,

                "secondary_check":
                    "NUMPY_ALLCLOSE",

                "pass":
                    True,

                "atol":
                    ADDITIVITY_ATOL,

                "rtol":
                    ADDITIVITY_RTOL,

                "mean_abs_error":
                    float(
                        absolute_error.mean()
                    ),

                "p95_abs_error":
                    float(
                        np.quantile(
                            absolute_error,
                            0.95,
                        )
                    ),

                "p99_abs_error":
                    float(
                        np.quantile(
                            absolute_error,
                            0.99,
                        )
                    ),

                "max_abs_error":
                    float(
                        absolute_error.max()
                    ),

                "relative_error_is_diagnostic_only":
                    True,

                "max_relative_error_diagnostic":
                    float(
                        relative_error.max()
                    ),

                "rows_over_1pct_relative_diagnostic":
                    int(
                        np.sum(
                            relative_error > 0.01
                        )
                    ),
            },

        "artifact":
            {
                "file":
                    FINAL_FILE.as_posix(),

                "sha256":
                    final_hash,

                "rows":
                    EXPECTED_ROWS,

                "features":
                    EXPECTED_FEATURES,
            },

        "summary":
            {
                "file":
                    SUMMARY_FILE.as_posix(),

                "sha256":
                    sha256_file(
                        SUMMARY_FILE
                    ),

                "rows":
                    75,
            },

        "chunk_manifest":
            {
                "file":
                    CHUNK_MANIFEST_FILE.as_posix(),

                "sha256":
                    sha256_file(
                        CHUNK_MANIFEST_FILE
                    ),

                "rows":
                    n_chunks,
            },

        "scientific_guardrails":
            {
                "cycle25_used_for_training":
                    False,

                "cycle25_used_for_fitting":
                    False,

                "model_retrained":
                    False,

                "model_retuned":
                    False,

                "background_changed":
                    False,

                "cycle24_null_changed":
                    False,

                "shap_approximation":
                    False,

                "sampling_reduction":
                    False,

                "drift_conclusion_made":
                    False,
            },

        "next_stage":
            "MODEL_CHAPTER_10_2_4_FINALIZE_CYCLE25_SHAP",
    }


    with AUDIT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit,
            file,
            indent=2,
        )


    audit_rt = load_json(
        AUDIT_FILE
    )


    if (
        audit_rt.get(
            "status"
        )
        !=
        "COMPLETE_XGB_CYCLE25_SHAP"
    ):

        raise RuntimeError(
            "XGB audit round-trip failed."
        )


    print()

    print("=" * 78)

    print(
        "XGB CYCLE25 SHAP — COMPLETE"
    )

    print("=" * 78)


    print(
        "Rows:",
        EXPECTED_ROWS
    )

    print(
        "Features:",
        EXPECTED_FEATURES
    )

    print(
        "Background:",
        EXPECTED_BACKGROUND_ROWS
    )

    print(
        "Global chunks:",
        n_chunks
    )

    print(
        "Explanation output:",
        "XGBOOST_RAW_MARGIN"
    )

    print(
        "Wrapper -> Booster equivalence:",
        "PASS"
    )

    print(
        "Raw-margin additivity:",
        "PASS"
    )

    print(
        "Cycle25 used for training:",
        False
    )

    print(
        "Cycle25 used for fitting:",
        False
    )

    print(
        "Model retrained:",
        False
    )

    print(
        "SHAP approximation:",
        False
    )

    print(
        "Sampling reduction:",
        False
    )

    print(
        "Drift conclusion made:",
        False
    )


    print()

    print(
        "Saved:",
        FINAL_FILE
    )

    print(
        "Saved:",
        SUMMARY_FILE
    )

    print(
        "Saved:",
        CHUNK_MANIFEST_FILE
    )

    print(
        "Saved:",
        AUDIT_FILE
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen XGB Cycle25 SHAP "
            "with deterministic checkpointed workers."
        )
    )


    parser.add_argument(
        "--workers",
        type=int,
        default=6,
    )


    parser.add_argument(
        "--worker-id",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--finalize",
        action="store_true",
    )


    parser.add_argument(
        "--max-new-chunks",
        type=int,
        default=None,
    )


    return parser.parse_args()


def main():

    args = parse_args()


    if args.finalize:

        if args.worker_id is not None:

            raise RuntimeError(
                "Do not use --worker-id "
                "with --finalize."
            )


        if args.max_new_chunks is not None:

            raise RuntimeError(
                "Do not use --max-new-chunks "
                "with --finalize."
            )


        finalize(
            workers=args.workers,
        )


    else:

        if args.worker_id is None:

            raise RuntimeError(
                "Worker mode requires --worker-id."
            )


        run_worker(
            workers=args.workers,
            worker_id=args.worker_id,
            max_new_chunks=args.max_new_chunks,
        )


if __name__ == "__main__":

    main()