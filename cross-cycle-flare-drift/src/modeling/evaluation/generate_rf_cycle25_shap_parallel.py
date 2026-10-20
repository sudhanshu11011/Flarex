# ============================================================
# MODEL CHAPTER 10.2.2
# RANDOM FOREST — CYCLE25 PARALLEL SHAP
#
# Frozen experiment:
#
#   Candidate:
#       RF_04_D12_L5_BALSUB
#
#   Training:
#       Cycle24 TRAIN only
#
#   Evaluation:
#       FULL Cycle25 dataset_v2
#
#   SHAP:
#       TreeExplainer
#       interventional
#       exact frozen 512-row Cycle24 TRAIN background
#       positive-class probability output
#
# Execution:
#       checkpointed deterministic global chunks
#       safe multi-process execution
#
# IMPORTANT:
#   - NO retraining
#   - NO retuning
#   - NO feature changes
#   - NO background changes
#   - NO Cycle25 fitting
#   - NO approximation
#   - NO sampling reduction
#   - NO drift conclusion
# ============================================================


# ============================================================
# THREAD LIMITS — MUST BE SET BEFORE NUMERIC IMPORTS
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

CANDIDATE_ID = "RF_04_D12_L5_BALSUB"

MODEL_FAMILY = "random_forest"

EXPECTED_ROWS = 79361

EXPECTED_FEATURES = 25

EXPECTED_BACKGROUND_ROWS = 512

CHUNK_SIZE = 128


EXPECTED_MODEL_HASH = (
    "519afcdf06e28f1da9b36d112bee9070"
    "bc51748e1c415fe2f3c244bd476bbd31"
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
# Match the corrected Cycle24 RF validation:
#
# np.allclose(
#     reconstructed,
#     target,
#     atol=1e-2,
#     rtol=1e-2
# )
#
# Relative error is diagnostic ONLY.
# ============================================================

ADDITIVITY_ATOL = 1e-2
ADDITIVITY_RTOL = 1e-2

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
    "artifacts/models/random_forest/models/"
    "RF_04_D12_L5_BALSUB.joblib"
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
    "checkpoints/rf_cycle25"
)


FINAL_FILE = Path(
    "artifacts/models/cross_cycle_shap/shap/cycle25/"
    "RF_04_D12_L5_BALSUB.parquet"
)


SUMMARY_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "rf_cycle25_shap_summary.csv"
)


CHUNK_MANIFEST_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "rf_cycle25_shap_chunk_manifest.csv"
)


AUDIT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "rf_cycle25_shap_parallel_audit.json"
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
# BLOCK LABELS
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
            [
                2020,
                2021,
            ]
        )
    ] = "B1"


    labels.loc[
        years.isin(
            [
                2022,
                2023,
            ]
        )
    ] = "B2"


    labels.loc[
        years.isin(
            [
                2024,
                2025,
            ]
        )
    ] = "B3"


    if labels.isna().any():

        raise RuntimeError(
            "Cycle25 block assignment failed."
        )


    return labels


# ============================================================
# POPULATION FINGERPRINT
#
# Must reproduce Chapter 10.1 exactly.
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
# RF SHAP EXTRACTION
#
# SHAP versions may expose binary RF outputs as:
#
#   (rows, features, classes)
#
# or occasionally already as:
#
#   (rows, features)
#
# We always retain positive class = class index 1.
# ============================================================

def extract_positive_class_values(
    explanation,
    n_rows: int,
) -> np.ndarray:

    values = np.asarray(
        explanation.values,
        dtype=np.float64,
    )


    if values.ndim == 3:

        if values.shape[0] != n_rows:

            raise RuntimeError(
                "RF SHAP row dimension mismatch."
            )


        if values.shape[1] != EXPECTED_FEATURES:

            raise RuntimeError(
                "RF SHAP feature dimension mismatch."
            )


        if values.shape[2] < 2:

            raise RuntimeError(
                "RF SHAP output does not contain "
                "binary class dimension."
            )


        values = values[
            :,
            :,
            1
        ]


    elif values.ndim == 2:

        if values.shape != (
            n_rows,
            EXPECTED_FEATURES,
        ):

            raise RuntimeError(
                "Unexpected RF SHAP matrix shape: "
                f"{values.shape}"
            )


    else:

        raise RuntimeError(
            "Unsupported RF SHAP value shape: "
            f"{values.shape}"
        )


    if values.shape != (
        n_rows,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            "Final RF positive-class SHAP "
            "shape mismatch."
        )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            "RF SHAP contains non-finite values."
        )


    return values


# ============================================================
# BASE VALUE EXTRACTION
# ============================================================

def extract_positive_class_base_values(
    explanation,
    n_rows: int,
) -> np.ndarray:

    base = np.asarray(
        explanation.base_values,
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # Scalar base
    # --------------------------------------------------------

    if base.ndim == 0:

        result = np.full(
            n_rows,
            float(base),
            dtype=np.float64,
        )


    # --------------------------------------------------------
    # 1D
    # --------------------------------------------------------

    elif base.ndim == 1:

        if base.size == 1:

            result = np.full(
                n_rows,
                float(
                    base[0]
                ),
                dtype=np.float64,
            )


        elif base.size == 2:

            # Binary class base vector.
            result = np.full(
                n_rows,
                float(
                    base[1]
                ),
                dtype=np.float64,
            )


        elif base.size == n_rows:

            result = base.astype(
                np.float64,
                copy=False,
            )


        else:

            raise RuntimeError(
                "Unsupported RF base-value "
                f"shape: {base.shape}"
            )


    # --------------------------------------------------------
    # 2D
    # --------------------------------------------------------

    elif base.ndim == 2:

        if (
            base.shape[0] == n_rows
            and
            base.shape[1] >= 2
        ):

            result = base[
                :,
                1
            ]


        elif base.shape == (
            n_rows,
            1,
        ):

            result = base[
                :,
                0
            ]


        elif (
            base.shape[0] == 1
            and
            base.shape[1] >= 2
        ):

            result = np.full(
                n_rows,
                float(
                    base[
                        0,
                        1
                    ]
                ),
                dtype=np.float64,
            )


        else:

            raise RuntimeError(
                "Unsupported RF base-value "
                f"shape: {base.shape}"
            )


    else:

        raise RuntimeError(
            "Unsupported RF base-value "
            f"dimension: {base.shape}"
        )


    result = np.asarray(
        result,
        dtype=np.float64,
    ).reshape(-1)


    if result.shape != (
        n_rows,
    ):

        raise RuntimeError(
            "RF base-value final shape mismatch."
        )


    if not np.isfinite(
        result
    ).all():

        raise RuntimeError(
            "RF base values contain "
            "non-finite values."
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
# COMMON INPUT LOADER
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
    # 10.1 CONTRACT
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
            "Chapter 10.1 execution inputs "
            "are not frozen."
        )


    # --------------------------------------------------------
    # MODEL MANIFEST HASH
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
            "Frozen model manifest hash mismatch."
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
            "Frozen RF model manifest "
            "lookup failed."
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
            "RF model-family mismatch."
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
            "RF explainer contract changed."
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
            "RF input-space contract changed."
        )


    if (
        str(
            row[
                "output_space"
            ]
        )
        !=
        "POSITIVE_CLASS_PROBABILITY"
    ):

        raise RuntimeError(
            "RF output-space contract changed."
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
            "RF feature-perturbation "
            "contract changed."
        )


    # --------------------------------------------------------
    # DATASET HASH
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
            "Cycle25 dataset differs "
            "from Chapter 10.1."
        )


    # --------------------------------------------------------
    # LOAD CYCLE25
    # --------------------------------------------------------

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
            "Cycle25 RF matrix shape mismatch."
        )


    if not np.isfinite(
        X
    ).all():

        raise RuntimeError(
            "Cycle25 RF input contains "
            "non-finite values."
        )


    # --------------------------------------------------------
    # BLOCK LABELS + ROW ORDER
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
            "Cycle25 row order / block "
            "assignment differs from 10.1."
        )


    # --------------------------------------------------------
    # MODEL HASH
    # --------------------------------------------------------

    model_hash = sha256_file(
        MODEL_FILE
    )


    if model_hash != EXPECTED_MODEL_HASH:

        raise RuntimeError(
            "RF hard-frozen model hash mismatch."
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
            "RF model differs from "
            "Chapter 10.1 manifest."
        )


    # --------------------------------------------------------
    # LOAD FROZEN MODEL
    # --------------------------------------------------------

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
            "RF model bundle candidate mismatch."
        )


    if (
        bundle[
            "training_scope"
        ]
        !=
        "CYCLE24_TRAIN_ONLY"
    ):

        raise RuntimeError(
            "RF training scope changed."
        )


    if (
        bundle[
            "train_sha256"
        ]
        !=
        EXPECTED_C24_TRAIN_HASH
    ):

        raise RuntimeError(
            "RF TRAIN provenance mismatch."
        )


    if (
        bundle[
            "feature_order"
        ]
        !=
        FEATURES
    ):

        raise RuntimeError(
            "RF feature order changed."
        )


    model = bundle[
        "model"
    ]


    # Runtime scheduling only.
    # Does not change fitted tree structure.
    try:

        model.set_params(
            n_jobs=1
        )

    except Exception:

        pass


    # --------------------------------------------------------
    # BACKGROUND HASH
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
            "RF manifest background hash mismatch."
        )


    # --------------------------------------------------------
    # LOAD BACKGROUND
    # --------------------------------------------------------

    background = pd.read_parquet(
        BACKGROUND_FILE
    )


    if len(
        background
    ) != EXPECTED_BACKGROUND_ROWS:

        raise RuntimeError(
            "Expected exact 512-row "
            "SHAP background."
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
            "RF background shape mismatch."
        )


    if not np.isfinite(
        X_background
    ).all():

        raise RuntimeError(
            "RF background contains "
            "non-finite values."
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

        "model":
            model,

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
    }


# ============================================================
# CREATE EXACT TREE EXPLAINER
# ============================================================

def create_explainer(
    model,
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
            "RF masker failed exact "
            "512-background requirement."
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
            "RF TreeExplainer does not "
            "expose masker."
        )


    if not hasattr(
        explainer.masker,
        "data",
    ):

        raise RuntimeError(
            "RF TreeExplainer masker does "
            "not expose background data."
        )


    actual_background_rows = int(
        explainer.masker.data.shape[
            0
        ]
    )


    if (
        actual_background_rows
        !=
        EXPECTED_BACKGROUND_ROWS
    ):

        raise RuntimeError(
            "RF TreeExplainer changed "
            "background from 512 to "
            f"{actual_background_rows}."
        )


    return explainer


# ============================================================
# EXPECTED CHUNKS
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
# CHECKPOINT VALIDATION
# ============================================================

def validate_checkpoint(
    path: Path,
    chunk_id: int,
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
            f"Checkpoint row-count mismatch: "
            f"{path}"
        )


    required_columns = {
        "source_row_number",
        "candidate_id",
        "model_family",
        "cycle25_block",
        "ENTITY_GROUP",
        ENTITY_GROUP_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN,
        "shap_base_value",
        *SHAP_COLUMNS,
    }


    missing = (
        required_columns
        -
        set(
            frame.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"Checkpoint missing columns "
            f"{sorted(missing)}: {path}"
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
            f"Checkpoint row order mismatch: "
            f"{path}"
        )


    if not (
        frame[
            "candidate_id"
        ].astype(str)
        ==
        CANDIDATE_ID
    ).all():

        raise RuntimeError(
            f"Checkpoint candidate mismatch: "
            f"{path}"
        )


    if not (
        frame[
            "model_family"
        ].astype(str)
        ==
        MODEL_FAMILY
    ).all():

        raise RuntimeError(
            f"Checkpoint family mismatch: "
            f"{path}"
        )


    expected_target = c25.iloc[
        start:stop
    ][
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.int8
    )


    if not np.array_equal(
        frame[
            TARGET_COLUMN
        ].to_numpy(
            dtype=np.int8
        ),
        expected_target,
    ):

        raise RuntimeError(
            f"Checkpoint target mismatch: "
            f"{path}"
        )


    expected_groups = c25.iloc[
        start:stop
    ][
        ENTITY_GROUP_COLUMN
    ].astype(
        str
    ).to_numpy()


    if not np.array_equal(
        frame[
            ENTITY_GROUP_COLUMN
        ].astype(
            str
        ).to_numpy(),
        expected_groups,
    ):

        raise RuntimeError(
            f"Checkpoint group mismatch: "
            f"{path}"
        )


    expected_blocks = block_labels.iloc[
        start:stop
    ].astype(
        str
    ).to_numpy()


    if not np.array_equal(
        frame[
            "cycle25_block"
        ].astype(
            str
        ).to_numpy(),
        expected_blocks,
    ):

        raise RuntimeError(
            f"Checkpoint block mismatch: "
            f"{path}"
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
            f"Checkpoint SHAP shape mismatch: "
            f"{path}"
        )


    if not np.isfinite(
        shap_matrix
    ).all():

        raise RuntimeError(
            f"Checkpoint contains non-finite "
            f"SHAP values: {path}"
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
            f"Checkpoint contains non-finite "
            f"base values: {path}"
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
            "--worker-id must satisfy "
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
        "MODEL CHAPTER 10.2.2 — "
        "RF CYCLE25 PARALLEL SHAP WORKER"
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


    model = inputs[
        "model"
    ]


    X_background = inputs[
        "X_background"
    ]


    print(
        "Input validation:",
        "PASS"
    )

    print(
        "RF model hash:",
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
        "Cycle25 rows:",
        len(
            c25
        )
    )

    print(
        "Background rows:",
        len(
            X_background
        )
    )


    # --------------------------------------------------------
    # EXPLAINER
    # --------------------------------------------------------

    explainer = create_explainer(
        model,
        X_background,
    )


    actual_background_rows = int(
        explainer.masker.data.shape[
            0
        ]
    )


    print(
        "TreeExplainer background rows:",
        actual_background_rows
    )

    print(
        "TreeExplainer background audit:",
        "PASS"
    )


    # --------------------------------------------------------
    # GLOBAL CHUNKS
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
        # REUSE VALID CHECKPOINT
        # ----------------------------------------------------

        if path.exists():

            validate_checkpoint(
                path=path,
                chunk_id=chunk_id,
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
        # OPTIONAL ONE/Few-CHUNK PREFLIGHT LIMIT
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


        # ----------------------------------------------------
        # CHUNK INPUT
        # ----------------------------------------------------

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


        shap_values = (
            extract_positive_class_values(
                explanation=explanation,
                n_rows=(
                    stop
                    -
                    start
                ),
            )
        )


        base_values = (
            extract_positive_class_base_values(
                explanation=explanation,
                n_rows=(
                    stop
                    -
                    start
                ),
            )
        )


        # ----------------------------------------------------
        # SECONDARY POSITIVE-CLASS ADDITIVITY AUDIT
        # ----------------------------------------------------

        target_output = np.asarray(
            model.predict_proba(
                X_chunk
            )[
                :,
                1
            ],
            dtype=np.float64,
        )


        reconstructed_output = (
            base_values
            +
            shap_values.sum(
                axis=1
            )
        )


        absolute_error = np.abs(
            reconstructed_output
            -
            target_output
        )


        additivity_pass = bool(
            np.allclose(
                reconstructed_output,
                target_output,
                atol=ADDITIVITY_ATOL,
                rtol=ADDITIVITY_RTOL,
            )
        )


        if not additivity_pass:

            raise RuntimeError(
                f"Chunk {chunk_id}: RF additivity "
                "failed. "
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
                    ].astype(
                        str
                    ).to_numpy(),

                "ENTITY_GROUP":
                    entity_group_alias,

                ENTITY_GROUP_COLUMN:
                    source[
                        ENTITY_GROUP_COLUMN
                    ]
                    .astype(str)
                    .to_numpy(),

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
        # ATOMIC SAVE
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


        # Validate temp file before rename.
        validate_checkpoint(
            path=temp_path,
            chunk_id=chunk_id,
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
        f"RF CYCLE25 WORKER "
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
        "SHAP approximation:",
        False
    )

    print(
        "Sampling reduction:",
        False
    )


# ============================================================
# FINALIZATION
# ============================================================

def finalize(
    workers: int,
):

    print("=" * 78)

    print(
        "MODEL CHAPTER 10.2.2 — "
        "RF CYCLE25 SHAP FINALIZATION"
    )

    print("=" * 78)


    # --------------------------------------------------------
    # OUTPUT PROTECTION
    # --------------------------------------------------------

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
                "RF Cycle25 SHAP output."
            )


    inputs = load_common_inputs()


    contract = inputs[
        "contract"
    ]


    c25 = inputs[
        "c25"
    ]


    X = inputs[
        "X"
    ]


    block_labels = inputs[
        "block_labels"
    ]


    model = inputs[
        "model"
    ]


    n_chunks = total_chunks()


    frames = []

    manifest_rows = []


    # --------------------------------------------------------
    # REQUIRE ALL EXPECTED CHECKPOINTS
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
                "Missing required RF checkpoint:\n"
                f"{path}"
            )


        frame = validate_checkpoint(
            path=path,
            chunk_id=chunk_id,
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
                    (
                        stop
                        -
                        start
                    ),

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


    if len(
        frames
    ) != n_chunks:

        raise RuntimeError(
            "RF checkpoint-count mismatch."
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
            "Merged RF row-count mismatch."
        )


    expected_rows = np.arange(
        EXPECTED_ROWS,
        dtype=np.int64,
    )


    actual_rows = merged[
        "source_row_number"
    ].to_numpy(
        dtype=np.int64
    )


    if not np.array_equal(
        actual_rows,
        expected_rows,
    ):

        raise RuntimeError(
            "Merged RF row-order audit failed."
        )


    if merged[
        "source_row_number"
    ].duplicated().any():

        raise RuntimeError(
            "Merged RF contains duplicate rows."
        )


    print(
        "Merged row-order audit:",
        "PASS"
    )


    # --------------------------------------------------------
    # FULL SHAP MATRIX
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
            "Final RF SHAP matrix shape mismatch."
        )


    if not np.isfinite(
        shap_matrix
    ).all():

        raise RuntimeError(
            "Final RF SHAP matrix contains "
            "non-finite values."
        )


    if not np.isfinite(
        base_values
    ).all():

        raise RuntimeError(
            "Final RF base values contain "
            "non-finite values."
        )


    # --------------------------------------------------------
    # FULL-POPULATION ADDITIVITY
    # --------------------------------------------------------

    target_output = np.asarray(
        model.predict_proba(
            X
        )[
            :,
            1
        ],
        dtype=np.float64,
    )


    reconstructed_output = (
        base_values
        +
        shap_matrix.sum(
            axis=1
        )
    )


    absolute_error = np.abs(
        reconstructed_output
        -
        target_output
    )


    additivity_pass = bool(
        np.allclose(
            reconstructed_output,
            target_output,
            atol=ADDITIVITY_ATOL,
            rtol=ADDITIVITY_RTOL,
        )
    )


    if not additivity_pass:

        raise RuntimeError(
            "Final RF Cycle25 native-style "
            "additivity failed."
        )


    relative_error = (
        absolute_error
        /
        (
            np.abs(
                reconstructed_output
            )
            +
            RELATIVE_DIAGNOSTIC_STABILIZER
        )
    )


    print(
        "Final native-style additivity:",
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
                relative_error
                >
                0.01
            )
        )
    )


    # --------------------------------------------------------
    # FINAL PARQUET
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


    final_roundtrip = pd.read_parquet(
        temp_final
    )


    if len(
        final_roundtrip
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            "Final RF parquet round-trip failed."
        )


    if not np.array_equal(
        final_roundtrip[
            "source_row_number"
        ].to_numpy(
            dtype=np.int64
        ),
        expected_rows,
    ):

        raise RuntimeError(
            "Final RF parquet row-order "
            "round-trip failed."
        )


    temp_final.replace(
        FINAL_FILE
    )


    final_hash = sha256_file(
        FINAL_FILE
    )


    print(
        "Final RF Cycle25 parquet:",
        "PASS"
    )

    print(
        "RF artifact SHA256:",
        final_hash
    )


    # --------------------------------------------------------
    # SUMMARY
    #
    # 3 strata:
    #   OVERALL
    #   TARGET_0
    #   TARGET_1
    #
    # 3 * 25 = 75 rows
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
            (
                target
                ==
                0
            ),

        "TARGET_1":
            (
                target
                ==
                1
            ),
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
                                feature_index
                                -
                                1
                            ]
                        ),

                    "mean_signed_shap":
                        float(
                            mean_signed[
                                feature_index
                                -
                                1
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
            "Expected exactly 75 RF "
            "summary rows."
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
            "RF chunk-manifest row count mismatch."
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
    # ROUND TRIP
    # --------------------------------------------------------

    summary_rt = pd.read_csv(
        SUMMARY_FILE,
        keep_default_na=False,
    )


    manifest_rt = pd.read_csv(
        CHUNK_MANIFEST_FILE,
        keep_default_na=False,
    )


    if len(
        summary_rt
    ) != 75:

        raise RuntimeError(
            "RF summary round-trip failed."
        )


    if len(
        manifest_rt
    ) != n_chunks:

        raise RuntimeError(
            "RF chunk manifest round-trip failed."
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
            (
                "Cycle25 Random Forest "
                "Parallel SHAP"
            ),

        "version":
            "1.0",

        "status":
            "COMPLETE_RF_CYCLE25_SHAP",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "chapter":
            "MODEL_CHAPTER_10_2_2",

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
            },

        "evaluation_population":
            {
                "dataset_file":
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

                "blocks":
                    {
                        "B1":
                            10411,

                        "B2":
                            33132,

                        "B3":
                            35818,
                    },
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
                    "POSITIVE_CLASS_PROBABILITY",

                "positive_class_index":
                    1,

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

                "secondary_validation":
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
                            relative_error
                            >
                            0.01
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
                "cycle25_read":
                    True,

                "cycle25_used_for_training":
                    False,

                "cycle25_used_for_fitting":
                    False,

                "model_retrained":
                    False,

                "model_retuned":
                    False,

                "feature_selection_changed":
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
            "MODEL_CHAPTER_10_2_3_XGB_CYCLE25_SHAP",
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


    # --------------------------------------------------------
    # AUDIT ROUNDTRIP
    # --------------------------------------------------------

    audit_rt = load_json(
        AUDIT_FILE
    )


    if (
        audit_rt.get(
            "status"
        )
        !=
        "COMPLETE_RF_CYCLE25_SHAP"
    ):

        raise RuntimeError(
            "RF Cycle25 audit round-trip failed."
        )


    if (
        audit_rt[
            "artifact"
        ][
            "sha256"
        ]
        !=
        final_hash
    ):

        raise RuntimeError(
            "RF Cycle25 artifact hash "
            "round-trip failed."
        )


    # --------------------------------------------------------
    # FINAL PRINT
    # --------------------------------------------------------

    print()

    print("=" * 78)

    print(
        "RF CYCLE25 SHAP — COMPLETE"
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
        "POSITIVE_CLASS_PROBABILITY"
    )

    print(
        "Native-style additivity:",
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
            "Generate frozen RF Cycle25 SHAP "
            "using deterministic checkpointed "
            "parallel workers."
        )
    )


    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Total number of workers.",
    )


    parser.add_argument(
        "--worker-id",
        type=int,
        default=None,
        help=(
            "Worker ID in range "
            "[0, workers-1]."
        ),
    )


    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Validate all checkpoints, merge, "
            "audit and create final artifact."
        ),
    )


    parser.add_argument(
        "--max-new-chunks",
        type=int,
        default=None,
        help=(
            "Optional preflight/debug limit on "
            "new chunks computed by this worker. "
            "Existing checkpoints do not count."
        ),
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
                "Worker mode requires "
                "--worker-id."
            )


        run_worker(
            workers=args.workers,
            worker_id=args.worker_id,
            max_new_chunks=args.max_new_chunks,
        )


if __name__ == "__main__":

    main()