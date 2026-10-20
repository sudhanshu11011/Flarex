# ============================================================
# MODEL CHAPTER 10.2.1
# GENERATE LOGISTIC REGRESSION CYCLE25 SHAP
#
# Frozen transfer experiment:
#
#   Model:
#       LR_02_C0p01_BALANCED
#
#   Training:
#       Cycle24 TRAIN only
#
#   Evaluation:
#       FULL Cycle25 dataset_v2
#
#   Preprocessing:
#       frozen Cycle24-TRAIN StandardScaler
#
#   SHAP background:
#       frozen 512-row Cycle24 TRAIN background
#
#   SHAP input space:
#       STANDARDIZED
#
#   Explanation output:
#       LOG-ODDS / decision_function
#
# IMPORTANT:
#   - NO Cycle25 fitting
#   - NO model retraining
#   - NO model retuning
#   - NO SHAP background change
#   - NO Cycle24 null modification
#   - NO drift conclusion
# ============================================================


from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# CONSTANTS
# ============================================================

CANDIDATE_ID = "LR_02_C0p01_BALANCED"

MODEL_FAMILY = "logistic"

EXPECTED_ROWS = 79361

EXPECTED_FEATURES = 25

EXPECTED_BACKGROUND_ROWS = 512


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


POPULATION_MANIFEST_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "cycle25_shap_population_manifest.csv"
)


MODEL_MANIFEST_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "cycle25_shap_model_manifest.csv"
)


C25_DATASET = Path(
    "data/final/dataset_v2/"
    "cycle25_dataset_v2.parquet"
)


MODEL_FILE = Path(
    "artifacts/models/logistic/models/"
    "LR_02_C0p01_BALANCED.joblib"
)


SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/"
    "logistic_standard_scaler.joblib"
)


BACKGROUND_FILE = Path(
    "artifacts/models/shift_controls/shap/background/"
    "cycle24_train_shap_background.parquet"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/shap/cycle25/"
    "LR_02_C0p01_BALANCED.parquet"
)


AUDIT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "lr_cycle25_shap_audit.json"
)


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

            digest.update(
                chunk
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
            "Some Cycle25 rows could not "
            "be assigned to B1/B2/B3."
        )


    return labels


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


def normalize_base_values(
    values,
    n_rows: int,
) -> np.ndarray:

    base = np.asarray(
        values,
        dtype=np.float64,
    )


    if base.ndim == 0:

        return np.full(
            n_rows,
            float(base),
            dtype=np.float64,
        )


    base = np.squeeze(
        base
    )


    if base.ndim == 0:

        return np.full(
            n_rows,
            float(base),
            dtype=np.float64,
        )


    if base.size == 1:

        return np.full(
            n_rows,
            float(
                base.reshape(-1)[0]
            ),
            dtype=np.float64,
        )


    if base.shape == (
        n_rows,
    ):

        return base.astype(
            np.float64,
            copy=False,
        )


    raise RuntimeError(
        "Unexpected SHAP base-value shape: "
        f"{base.shape}"
    )


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.2.1 — "
    "LR CYCLE25 SHAP"
)

print("=" * 78)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    EXECUTION_CONTRACT_FILE,
    POPULATION_MANIFEST_FILE,
    MODEL_MANIFEST_FILE,
    C25_DATASET,
    MODEL_FILE,
    SCALER_FILE,
    BACKGROUND_FILE,
]


for path in required_files:

    verify_exists(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for output in [
    OUTPUT_FILE,
    AUDIT_FILE,
]:

    if output.exists():

        raise RuntimeError(
            f"Output already exists: {output}\n"
            "Refusing to overwrite existing "
            "Cycle25 LR SHAP output."
        )


# ============================================================
# LOAD + VERIFY 10.1 CONTRACT
# ============================================================

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


if (
    contract.get(
        "next_stage"
    )
    !=
    "MODEL_CHAPTER_10_2_CYCLE25_SHAP_GENERATION"
):

    raise RuntimeError(
        "Chapter 10.1 contract does not "
        "point to Chapter 10.2."
    )


print(
    "Chapter 10.1 frozen contract:",
    "PASS"
)


# ============================================================
# VERIFY POPULATION MANIFEST
# ============================================================

population_manifest_hash = sha256_file(
    POPULATION_MANIFEST_FILE
)


expected_population_manifest_hash = (
    contract[
        "temporal_blocks"
    ][
        "population_manifest"
    ][
        "sha256"
    ]
)


if (
    population_manifest_hash
    !=
    expected_population_manifest_hash
):

    raise RuntimeError(
        "Cycle25 population manifest "
        "hash mismatch."
    )


population_manifest = pd.read_csv(
    POPULATION_MANIFEST_FILE,
    keep_default_na=False,
)


if len(
    population_manifest
) != 4:

    raise RuntimeError(
        "Expected POOLED+B1+B2+B3 "
        "population manifest."
    )


print(
    "Cycle25 population manifest:",
    "PASS"
)


# ============================================================
# VERIFY MODEL MANIFEST
# ============================================================

model_manifest_hash = sha256_file(
    MODEL_MANIFEST_FILE
)


expected_model_manifest_hash = (
    contract[
        "frozen_models"
    ][
        "model_manifest"
    ][
        "sha256"
    ]
)


if (
    model_manifest_hash
    !=
    expected_model_manifest_hash
):

    raise RuntimeError(
        "Cycle25 model manifest hash mismatch."
    )


model_manifest = pd.read_csv(
    MODEL_MANIFEST_FILE,
    keep_default_na=False,
)


model_row = model_manifest[
    model_manifest[
        "candidate_id"
    ].astype(str)
    ==
    CANDIDATE_ID
]


if len(
    model_row
) != 1:

    raise RuntimeError(
        "Frozen LR model manifest "
        "lookup failed."
    )


model_row = model_row.iloc[
    0
]


if (
    str(
        model_row[
            "model_family"
        ]
    )
    !=
    MODEL_FAMILY
):

    raise RuntimeError(
        "Frozen LR model family mismatch."
    )


if (
    str(
        model_row[
            "explainer"
        ]
    )
    !=
    "LinearExplainer"
):

    raise RuntimeError(
        "Frozen LR explainer changed."
    )


if (
    str(
        model_row[
            "feature_input_space"
        ]
    )
    !=
    "FROZEN_STANDARDIZED"
):

    raise RuntimeError(
        "Frozen LR input space changed."
    )


if (
    str(
        model_row[
            "output_space"
        ]
    )
    !=
    "LOG_ODDS_DECISION_FUNCTION"
):

    raise RuntimeError(
        "Frozen LR output space changed."
    )


print(
    "Frozen LR model manifest:",
    "PASS"
)


# ============================================================
# VERIFY CYCLE25 DATASET
# ============================================================

dataset_hash = sha256_file(
    C25_DATASET
)


expected_dataset_hash = (
    contract[
        "cycle25_population"
    ][
        "dataset_sha256"
    ]
)


if dataset_hash != expected_dataset_hash:

    raise RuntimeError(
        "Cycle25 dataset hash mismatch."
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


missing_columns = (
    required_columns
    -
    set(
        c25.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "Cycle25 missing columns: "
        f"{sorted(missing_columns)}"
    )


X_raw = c25[
    FEATURES
].to_numpy(
    dtype=np.float64
)


if X_raw.shape != (
    EXPECTED_ROWS,
    EXPECTED_FEATURES,
):

    raise RuntimeError(
        "Cycle25 matrix shape mismatch."
    )


if not np.isfinite(
    X_raw
).all():

    raise RuntimeError(
        "Cycle25 contains non-finite "
        "feature values."
    )


block_labels = build_block_labels(
    c25
)


current_population_hash = (
    ordered_population_fingerprint(
        frame=c25,
        block_labels=block_labels,
    )
)


expected_population_hash = (
    contract[
        "cycle25_population"
    ][
        "ordered_population_sha256"
    ]
)


if (
    current_population_hash
    !=
    expected_population_hash
):

    raise RuntimeError(
        "Cycle25 row order / block "
        "assignment changed after 10.1."
    )


print(
    "Cycle25 dataset + row order:",
    "PASS"
)


# ============================================================
# VERIFY EXACT BLOCK COUNTS
# ============================================================

expected_block_rows = {
    "B1": 10411,
    "B2": 33132,
    "B3": 35818,
}


for block_name, expected_rows in (
    expected_block_rows.items()
):

    actual_rows = int(
        np.sum(
            block_labels.to_numpy()
            ==
            block_name
        )
    )


    if actual_rows != expected_rows:

        raise RuntimeError(
            f"{block_name}: expected "
            f"{expected_rows}, got "
            f"{actual_rows}."
        )


print(
    "Cycle25 B1/B2/B3 assignment:",
    "PASS"
)


# ============================================================
# VERIFY MODEL
# ============================================================

model_hash = sha256_file(
    MODEL_FILE
)


if (
    model_hash
    !=
    str(
        model_row[
            "model_sha256"
        ]
    )
):

    raise RuntimeError(
        "Frozen LR model hash mismatch."
    )


model_bundle = joblib.load(
    MODEL_FILE
)


if (
    model_bundle[
        "candidate_id"
    ]
    !=
    CANDIDATE_ID
):

    raise RuntimeError(
        "LR model bundle candidate mismatch."
    )


if (
    model_bundle[
        "training_scope"
    ]
    !=
    "CYCLE24_TRAIN_ONLY"
):

    raise RuntimeError(
        "LR model was not trained "
        "under frozen C24-only scope."
    )


if (
    model_bundle[
        "feature_order"
    ]
    !=
    FEATURES
):

    raise RuntimeError(
        "LR model feature order changed."
    )


model = model_bundle[
    "model"
]


print(
    "Frozen LR model:",
    "PASS"
)


# ============================================================
# VERIFY FROZEN SCALER
# ============================================================

scaler_hash = sha256_file(
    SCALER_FILE
)


expected_scaler_hash = (
    contract[
        "preprocessing"
    ][
        "lr_scaler_sha256"
    ]
)


if (
    scaler_hash
    !=
    expected_scaler_hash
):

    raise RuntimeError(
        "Frozen LR scaler hash mismatch."
    )


scaler_bundle = joblib.load(
    SCALER_FILE
)


if (
    scaler_bundle[
        "fit_scope"
    ]
    !=
    "CYCLE24_TRAIN_ONLY"
):

    raise RuntimeError(
        "LR scaler scope changed."
    )


if (
    scaler_bundle[
        "feature_order"
    ]
    !=
    FEATURES
):

    raise RuntimeError(
        "LR scaler feature order changed."
    )


scaler = scaler_bundle[
    "scaler"
]


X_scaled = scaler.transform(
    X_raw
)


if not np.isfinite(
    X_scaled
).all():

    raise RuntimeError(
        "Scaled Cycle25 matrix contains "
        "non-finite values."
    )


print(
    "Frozen LR scaler:",
    "PASS"
)


# ============================================================
# VERIFY FROZEN BACKGROUND
# ============================================================

background_hash = sha256_file(
    BACKGROUND_FILE
)


expected_background_hash = (
    contract[
        "shap_background"
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
        "Frozen SHAP background hash mismatch."
    )


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


background_raw = background[
    FEATURES
].to_numpy(
    dtype=np.float64
)


if not np.isfinite(
    background_raw
).all():

    raise RuntimeError(
        "Frozen background contains "
        "non-finite values."
    )


background_scaled = scaler.transform(
    background_raw
)


if not np.isfinite(
    background_scaled
).all():

    raise RuntimeError(
        "Scaled SHAP background contains "
        "non-finite values."
    )


print(
    "Frozen 512-row background:",
    "PASS"
)


# ============================================================
# CREATE EXACT BACKGROUND MASKER
# ============================================================

masker = shap.maskers.Independent(
    background_scaled,
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
        "LinearExplainer masker does not "
        "contain exact 512-row background."
    )


print(
    "LinearExplainer masker background:",
    "512 PASS"
)


# ============================================================
# CREATE FROZEN LR EXPLAINER
# ============================================================

explainer = shap.LinearExplainer(
    model,
    masker,
)


if not hasattr(
    explainer,
    "masker",
):

    raise RuntimeError(
        "LinearExplainer does not expose masker."
    )


if (
    explainer.masker.data.shape[
        0
    ]
    !=
    EXPECTED_BACKGROUND_ROWS
):

    raise RuntimeError(
        "LinearExplainer changed "
        "background sample count."
    )


print(
    "LinearExplainer configuration:",
    "PASS"
)


# ============================================================
# GENERATE CYCLE25 SHAP
# ============================================================

print()

print(
    "Generating Cycle25 LR SHAP..."
)


explanation = explainer(
    X_scaled
)


shap_values = np.asarray(
    explanation.values,
    dtype=np.float64,
)


if shap_values.shape != (
    EXPECTED_ROWS,
    EXPECTED_FEATURES,
):

    raise RuntimeError(
        "Unexpected LR SHAP matrix shape: "
        f"{shap_values.shape}"
    )


if not np.isfinite(
    shap_values
).all():

    raise RuntimeError(
        "LR SHAP contains non-finite values."
    )


base_values = normalize_base_values(
    explanation.base_values,
    EXPECTED_ROWS,
)


if not np.isfinite(
    base_values
).all():

    raise RuntimeError(
        "LR SHAP base values contain "
        "non-finite values."
    )


print(
    "SHAP matrix:",
    shap_values.shape
)

print(
    "SHAP numeric integrity:",
    "PASS"
)


# ============================================================
# FINAL ADDITIVITY AUDIT
# ============================================================

target_output = np.asarray(
    model.decision_function(
        X_scaled
    ),
    dtype=np.float64,
).reshape(-1)


if target_output.shape != (
    EXPECTED_ROWS,
):

    raise RuntimeError(
        "Unexpected LR decision-function shape."
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


ADDITIVITY_ATOL = 1e-8
ADDITIVITY_RTOL = 1e-8


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
        "LR Cycle25 SHAP additivity failed.\n"
        f"Mean abs error: "
        f"{float(absolute_error.mean())}\n"
        f"Max abs error: "
        f"{float(absolute_error.max())}"
    )


print(
    "LR additivity:",
    "PASS"
)

print(
    "Mean absolute error:",
    f"{float(absolute_error.mean()):.12f}"
)

print(
    "P95 absolute error:",
    f"{float(np.quantile(absolute_error, 0.95)):.12f}"
)

print(
    "P99 absolute error:",
    f"{float(np.quantile(absolute_error, 0.99)):.12f}"
)

print(
    "Max absolute error:",
    f"{float(absolute_error.max()):.12f}"
)


# ============================================================
# BUILD FINAL ROW-LEVEL ARTIFACT
# ============================================================

entity_group_alias = (
    c25[
        "ENTITY_GROUP"
    ].astype(str)
    if
    "ENTITY_GROUP"
    in
    c25.columns
    else
    c25[
        ENTITY_GROUP_COLUMN
    ].astype(str)
)


output = pd.DataFrame(
    {
        "source_row_number":
            np.arange(
                EXPECTED_ROWS,
                dtype=np.int64,
            ),

        "candidate_id":
            CANDIDATE_ID,

        "model_family":
            MODEL_FAMILY,

        "cycle25_block":
            block_labels.to_numpy(),

        "ENTITY_GROUP":
            entity_group_alias.to_numpy(),

        ENTITY_GROUP_COLUMN:
            c25[
                ENTITY_GROUP_COLUMN
            ].astype(str).to_numpy(),

        TARGET_COLUMN:
            c25[
                TARGET_COLUMN
            ].to_numpy(
                dtype=np.int8
            ),

        TIME_COLUMN:
            pd.to_datetime(
                c25[
                    TIME_COLUMN
                ],
                utc=True,
            ).to_numpy(),

        "shap_base_value":
            base_values,
    }
)


for feature_index, shap_column in enumerate(
    SHAP_COLUMNS
):

    output[
        shap_column
    ] = shap_values[
        :,
        feature_index
    ]


if len(
    output
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Final LR SHAP row-count mismatch."
    )


if output[
    "source_row_number"
].duplicated().any():

    raise RuntimeError(
        "Duplicate source-row numbers."
    )


# ============================================================
# SAVE ATOMICALLY
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


AUDIT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


temporary_file = OUTPUT_FILE.with_name(
    OUTPUT_FILE.stem
    +
    ".tmp.parquet"
)


if temporary_file.exists():

    temporary_file.unlink()


output.to_parquet(
    temporary_file,
    index=False,
)


# ============================================================
# TEMP ROUND-TRIP BEFORE FINAL RENAME
# ============================================================

roundtrip = pd.read_parquet(
    temporary_file
)


if len(
    roundtrip
) != EXPECTED_ROWS:

    raise RuntimeError(
        "LR SHAP parquet round-trip "
        "row-count failed."
    )


if not np.array_equal(
    roundtrip[
        "source_row_number"
    ].to_numpy(
        dtype=np.int64
    ),
    np.arange(
        EXPECTED_ROWS,
        dtype=np.int64,
    ),
):

    raise RuntimeError(
        "LR SHAP parquet round-trip "
        "row order failed."
    )


roundtrip_shap = roundtrip[
    SHAP_COLUMNS
].to_numpy(
    dtype=np.float64
)


if not np.allclose(
    roundtrip_shap,
    shap_values,
    atol=0.0,
    rtol=0.0,
):

    raise RuntimeError(
        "LR SHAP values changed during "
        "parquet round-trip."
    )


temporary_file.replace(
    OUTPUT_FILE
)


# ============================================================
# FINAL FILE HASH
# ============================================================

output_hash = sha256_file(
    OUTPUT_FILE
)


print(
    "Final LR Cycle25 parquet:",
    "PASS"
)


# ============================================================
# AUDIT CONTRACT
# ============================================================

audit = {

    "audit_name":
        "Cycle25 Logistic Regression SHAP",

    "version":
        "1.0",

    "status":
        "COMPLETE_LR_CYCLE25_SHAP",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_2_1",

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
                model_hash,

            "training_scope":
                "CYCLE24_TRAIN_ONLY",
        },

    "evaluation_population":
        {
            "dataset":
                C25_DATASET.as_posix(),

            "dataset_sha256":
                dataset_hash,

            "ordered_population_sha256":
                current_population_hash,

            "rows":
                EXPECTED_ROWS,

            "blocks":
                {
                    "B1": 10411,
                    "B2": 33132,
                    "B3": 35818,
                },
        },

    "preprocessing":
        {
            "type":
                "StandardScaler",

            "file":
                SCALER_FILE.as_posix(),

            "sha256":
                scaler_hash,

            "fit_scope":
                "CYCLE24_TRAIN_ONLY",

            "cycle25_fit":
                False,
        },

    "shap":
        {
            "explainer":
                "LinearExplainer",

            "feature_input_space":
                "FROZEN_STANDARDIZED",

            "background_input_space":
                "FROZEN_STANDARDIZED",

            "background_file":
                BACKGROUND_FILE.as_posix(),

            "background_sha256":
                background_hash,

            "background_rows":
                EXPECTED_BACKGROUND_ROWS,

            "output_space":
                "LOG_ODDS_DECISION_FUNCTION",

            "rows":
                EXPECTED_ROWS,

            "features":
                EXPECTED_FEATURES,
        },

    "additivity":
        {
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
        },

    "artifact":
        {
            "file":
                OUTPUT_FILE.as_posix(),

            "sha256":
                output_hash,

            "rows":
                EXPECTED_ROWS,

            "features":
                EXPECTED_FEATURES,
        },

    "scientific_guardrails":
        {
            "cycle25_read":
                True,

            "cycle25_used_for_training":
                False,

            "cycle25_used_for_scaler_fit":
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

            "drift_conclusion_made":
                False,
        },

    "next_stage":
        "MODEL_CHAPTER_10_2_2_RF_CYCLE25_SHAP",
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


# ============================================================
# AUDIT ROUND-TRIP
# ============================================================

audit_roundtrip = load_json(
    AUDIT_FILE
)


if (
    audit_roundtrip.get(
        "status"
    )
    !=
    "COMPLETE_LR_CYCLE25_SHAP"
):

    raise RuntimeError(
        "LR Cycle25 SHAP audit "
        "round-trip failed."
    )


if (
    audit_roundtrip[
        "artifact"
    ][
        "sha256"
    ]
    !=
    output_hash
):

    raise RuntimeError(
        "LR Cycle25 SHAP audit hash mismatch."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 10.2.1 — COMPLETE"
)

print("=" * 78)


print(
    "Candidate:",
    CANDIDATE_ID
)

print(
    "Cycle25 rows:",
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
    "Explanation output:",
    "LOG_ODDS_DECISION_FUNCTION"
)

print(
    "Native/final additivity:",
    "PASS"
)

print(
    "Cycle25 used for training:",
    False
)

print(
    "Cycle25 used for scaler fit:",
    False
)

print(
    "Model retrained:",
    False
)

print(
    "Drift conclusion made:",
    False
)


print()

print(
    "Artifact SHA256:",
    output_hash
)

print(
    "Saved:",
    OUTPUT_FILE
)

print(
    "Saved:",
    AUDIT_FILE
)


print()

print(
    "NEXT:",
    "Chapter 10.2.2 — RF Cycle25 SHAP"
)