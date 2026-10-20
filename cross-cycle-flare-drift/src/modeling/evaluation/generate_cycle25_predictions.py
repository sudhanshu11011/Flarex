from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import joblib

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CYCLE25_DATASET_FILE = Path(
    "data/final/dataset_v2/"
    "cycle25_dataset_v2.parquet"
)

CYCLE25_PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_evaluation_protocol.json"
)

CYCLE25_MODEL_REGISTRY_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_frozen_model_registry.csv"
)

CYCLE25_BLOCKS_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_temporal_blocks.csv"
)

LOGISTIC_SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/"
    "logistic_standard_scaler.joblib"
)

OUTPUT_PREDICTION_FILE = Path(
    "artifacts/models/evaluation/predictions/"
    "cycle25_predictions.parquet"
)

OUTPUT_MANIFEST_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_prediction_manifest.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_prediction_contract.json"
)


# ============================================================
# FROZEN EXPECTATIONS
# ============================================================

EXPECTED_DATASET_SHA256 = (
    "0bd6a364e152685d13c7eb955da074a4"
    "b9f019f1851c6b5dafebf295c406cd4d"
)

EXPECTED_ROWS = 79361
EXPECTED_POSITIVES = 4836
EXPECTED_NEGATIVES = 74525

ENTITY_GROUP_SOURCE_COLUMN = "ENTITY_GROUP_ID"
TIME_COLUMN = "observation_time_utc"


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


EXPECTED_MODELS = {
    "logistic": {
        "candidate_id":
            "LR_02_C0p01_BALANCED",

        "threshold":
            0.4548081408,
    },

    "random_forest": {
        "candidate_id":
            "RF_04_D12_L5_BALSUB",

        "threshold":
            0.1196946373,
    },

    "xgboost": {
        "candidate_id":
            "XGB_02_LR0p03_D3_BAL",

        "threshold":
            0.2638225555,
    },
}


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


def load_estimator(model_file: Path):

    bundle = joblib.load(
        model_file
    )

    # --------------------------------------------------------
    # Direct estimator
    # --------------------------------------------------------

    if hasattr(
        bundle,
        "predict_proba",
    ):

        return (
            bundle,
            bundle,
        )

    # --------------------------------------------------------
    # Dictionary model bundle
    # --------------------------------------------------------

    if isinstance(
        bundle,
        dict,
    ):

        preferred_keys = [
            "model",
            "estimator",
            "classifier",
            "pipeline",
        ]

        for key in preferred_keys:

            candidate = bundle.get(
                key
            )

            if hasattr(
                candidate,
                "predict_proba",
            ):

                return (
                    bundle,
                    candidate,
                )

        for value in bundle.values():

            if hasattr(
                value,
                "predict_proba",
            ):

                return (
                    bundle,
                    value,
                )

    raise RuntimeError(
        f"Could not locate estimator with "
        f"predict_proba in {model_file}"
    )


def verify_bundle_feature_order(
    bundle,
    candidate_id,
):

    if not isinstance(
        bundle,
        dict,
    ):
        return

    for key in [
        "feature_order",
        "features",
        "feature_names",
    ]:

        if key in bundle:

            stored = list(
                bundle[
                    key
                ]
            )

            if stored != FEATURES:

                raise RuntimeError(
                    f"{candidate_id}: model "
                    "bundle feature order mismatch."
                )

            return


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 8.2 — GENERATE CYCLE25 FROZEN-MODEL PREDICTIONS"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    CYCLE25_DATASET_FILE,
    CYCLE25_PROTOCOL_FILE,
    CYCLE25_MODEL_REGISTRY_FILE,
    CYCLE25_BLOCKS_FILE,
    LOGISTIC_SCALER_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# PROTECT OUTPUTS
# ============================================================

for path in [
    OUTPUT_PREDICTION_FILE,
    OUTPUT_MANIFEST_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Cycle25 prediction outputs."
        )


# ============================================================
# LOAD / VERIFY PROTOCOL
# ============================================================

protocol = load_json(
    CYCLE25_PROTOCOL_FILE
)


if protocol.get(
    "status"
) != "FROZEN_BEFORE_CYCLE25_EVALUATION":

    raise RuntimeError(
        "Cycle25 evaluation protocol "
        "is not frozen."
    )


if int(
    protocol[
        "model_count"
    ]
) != 3:

    raise RuntimeError(
        "Cycle25 protocol does not "
        "contain exactly 3 models."
    )


if protocol[
    "feature_contract"
][
    "feature_order"
] != FEATURES:

    raise RuntimeError(
        "Protocol feature order mismatch."
    )


print(
    "Cycle25 evaluation protocol:",
    "FROZEN"
)

print(
    "Frozen feature order:",
    "PASS"
)


# ============================================================
# VERIFY FROZEN MANIFEST HASHES
# ============================================================

registry_hash = sha256_file(
    CYCLE25_MODEL_REGISTRY_FILE
)


expected_registry_hash = protocol[
    "artifacts"
][
    "cycle25_frozen_model_registry"
][
    "sha256"
]


if (
    registry_hash
    !=
    expected_registry_hash
):

    raise RuntimeError(
        "Cycle25 frozen model registry "
        "hash mismatch."
    )


blocks_hash = sha256_file(
    CYCLE25_BLOCKS_FILE
)


expected_blocks_hash = protocol[
    "artifacts"
][
    "cycle25_temporal_blocks"
][
    "sha256"
]


if (
    blocks_hash
    !=
    expected_blocks_hash
):

    raise RuntimeError(
        "Cycle25 temporal-block manifest "
        "hash mismatch."
    )


print(
    "Frozen model registry hash:",
    "PASS"
)

print(
    "Temporal-block manifest hash:",
    "PASS"
)


# ============================================================
# VERIFY DATASET HASH BEFORE READING CONTENT
# ============================================================

actual_dataset_hash = sha256_file(
    CYCLE25_DATASET_FILE
)


if (
    actual_dataset_hash
    !=
    EXPECTED_DATASET_SHA256
):

    raise RuntimeError(
        "Cycle25 Dataset v2 hash differs "
        "from frozen expected hash."
    )


protocol_dataset_hash = protocol[
    "cycle25_dataset"
][
    "expected_sha256"
]


if (
    actual_dataset_hash
    !=
    protocol_dataset_hash
):

    raise RuntimeError(
        "Cycle25 Dataset v2 hash differs "
        "from Chapter 8.1 protocol."
    )


print(
    "Cycle25 Dataset v2 hash:",
    "PASS"
)


# ============================================================
# FIRST CONTENT READ OF CYCLE25
# ============================================================

cycle25 = pd.read_parquet(
    CYCLE25_DATASET_FILE
)


print(
    "Cycle25 dataset opened:",
    True
)


# ============================================================
# SCHEMA CHECK
# ============================================================

required_columns = set(
    FEATURES
    +
    [
        TIME_COLUMN,
        ENTITY_GROUP_SOURCE_COLUMN,
        "HARPNUM",
        "target",
    ]
)


missing_columns = (
    required_columns
    -
    set(
        cycle25.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "Cycle25 dataset missing required "
        f"columns: {sorted(missing_columns)}"
    )


if len(
    cycle25
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Unexpected Cycle25 row count."
    )


# ============================================================
# TARGET INTEGRITY
# ============================================================

target = cycle25[
    "target"
].to_numpy(
    dtype=np.int64
)


target_values = set(
    np.unique(
        target
    ).tolist()
)


if target_values != {
    0,
    1,
}:

    raise RuntimeError(
        "Cycle25 target must contain "
        "exactly classes 0 and 1."
    )


positive_count = int(
    target.sum()
)


negative_count = int(
    len(
        target
    )
    -
    positive_count
)


if positive_count != EXPECTED_POSITIVES:

    raise RuntimeError(
        "Unexpected Cycle25 positive count."
    )


if negative_count != EXPECTED_NEGATIVES:

    raise RuntimeError(
        "Unexpected Cycle25 negative count."
    )


# ============================================================
# ENTITY GROUP INTEGRITY
# ============================================================

if cycle25[
    ENTITY_GROUP_SOURCE_COLUMN
].isna().any():

    raise RuntimeError(
        "Cycle25 contains missing "
        "ENTITY_GROUP_ID values."
    )


entity_group_count = int(
    cycle25[
        ENTITY_GROUP_SOURCE_COLUMN
    ].nunique(
        dropna=False
    )
)


# ============================================================
# FEATURE INTEGRITY
# ============================================================

feature_frame = cycle25[
    FEATURES
].copy()


if feature_frame.isna().any().any():

    raise RuntimeError(
        "Cycle25 frozen feature matrix "
        "contains missing values."
    )


X_raw64 = feature_frame.to_numpy(
    dtype=np.float64
)


if not np.isfinite(
    X_raw64
).all():

    raise RuntimeError(
        "Cycle25 frozen feature matrix "
        "contains non-finite values."
    )


if X_raw64.shape != (
    EXPECTED_ROWS,
    25,
):

    raise RuntimeError(
        "Cycle25 feature matrix shape "
        "is incorrect."
    )


print(
    "Cycle25 rows:",
    len(
        cycle25
    )
)

print(
    "Cycle25 positives:",
    positive_count
)

print(
    "Cycle25 negatives:",
    negative_count
)

print(
    "Cycle25 ENTITY_GROUPs:",
    entity_group_count
)

print(
    "Cycle25 feature integrity:",
    "PASS"
)


# ============================================================
# TEMPORAL BLOCK ASSIGNMENT
# ============================================================

blocks = pd.read_csv(
    CYCLE25_BLOCKS_FILE,
    keep_default_na=False,
)


primary_blocks = blocks[
    blocks[
        "primary_block"
    ].astype(str).str.lower()
    ==
    "true"
].copy()


if len(
    primary_blocks
) != 3:

    raise RuntimeError(
        "Expected exactly 3 primary "
        "Cycle25 temporal blocks."
    )


observation_time = pd.to_datetime(
    cycle25[
        TIME_COLUMN
    ],
    utc=True,
    errors="raise",
)


primary_block_id = pd.Series(
    pd.NA,
    index=cycle25.index,
    dtype="string",
)


block_summary = []


for _, block in primary_blocks.iterrows():

    block_id = str(
        block[
            "block_id"
        ]
    )

    start = pd.Timestamp(
        block[
            "start_utc"
        ]
    )

    end_exclusive = pd.Timestamp(
        block[
            "end_exclusive_utc"
        ]
    )


    mask = (
        observation_time
        >=
        start
    ) & (
        observation_time
        <
        end_exclusive
    )


    if primary_block_id[
        mask
    ].notna().any():

        raise RuntimeError(
            f"Temporal block overlap "
            f"detected for {block_id}."
        )


    primary_block_id.loc[
        mask
    ] = block_id


    block_rows = int(
        mask.sum()
    )


    block_groups = int(
        cycle25.loc[
            mask,
            ENTITY_GROUP_SOURCE_COLUMN,
        ].nunique(
            dropna=False
        )
    )


    block_summary.append(
        {
            "block_id":
                block_id,

            "rows":
                block_rows,

            "entity_groups":
                block_groups,
        }
    )


if primary_block_id.isna().any():

    unassigned_count = int(
        primary_block_id.isna().sum()
    )

    raise RuntimeError(
        f"{unassigned_count} Cycle25 rows "
        "were not assigned to a frozen "
        "primary temporal block."
    )


if len(
    primary_block_id
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Temporal assignment length mismatch."
    )


print(
    "\n===================================="
)

print(
    "FROZEN TEMPORAL BLOCK ASSIGNMENT"
)

print(
    "===================================="
)


for item in block_summary:

    print(
        f"{item['block_id']}: "
        f"rows={item['rows']}, "
        f"ENTITY_GROUPs={item['entity_groups']}"
    )


print(
    "All Cycle25 rows assigned exactly once:",
    "PASS"
)


# ============================================================
# LOAD AND VERIFY LOGISTIC SCALER
# ============================================================

actual_scaler_hash = sha256_file(
    LOGISTIC_SCALER_FILE
)


expected_scaler_hash = protocol[
    "provenance"
][
    "logistic_scaler"
][
    "sha256"
]


if (
    actual_scaler_hash
    !=
    expected_scaler_hash
):

    raise RuntimeError(
        "Frozen Logistic scaler "
        "hash mismatch."
    )


scaler_bundle = joblib.load(
    LOGISTIC_SCALER_FILE
)


if not isinstance(
    scaler_bundle,
    dict,
):

    raise RuntimeError(
        "Expected Logistic scaler "
        "artifact to be a dictionary bundle."
    )


required_scaler_keys = {
    "scaler",
    "feature_order",
    "feature_count",
    "fit_scope",
}


missing_scaler_keys = (
    required_scaler_keys
    -
    set(
        scaler_bundle.keys()
    )
)


if missing_scaler_keys:

    raise RuntimeError(
        "Frozen scaler bundle missing "
        f"keys: {sorted(missing_scaler_keys)}"
    )


if list(
    scaler_bundle[
        "feature_order"
    ]
) != FEATURES:

    raise RuntimeError(
        "Frozen Logistic scaler feature "
        "order mismatch."
    )


if int(
    scaler_bundle[
        "feature_count"
    ]
) != 25:

    raise RuntimeError(
        "Frozen Logistic scaler feature "
        "count mismatch."
    )


if scaler_bundle[
    "fit_scope"
] != "CYCLE24_TRAIN_ONLY":

    raise RuntimeError(
        "Logistic scaler was not fitted "
        "on Cycle24 TRAIN only."
    )


scaler = scaler_bundle[
    "scaler"
]


if int(
    scaler.n_features_in_
) != 25:

    raise RuntimeError(
        "Frozen Logistic scaler "
        "n_features_in_ mismatch."
    )


# TRANSFORM ONLY — NO FIT
X_logistic = scaler.transform(
    X_raw64
)


if not np.isfinite(
    X_logistic
).all():

    raise RuntimeError(
        "Scaled Cycle25 feature matrix "
        "contains non-finite values."
    )


print(
    "\nFrozen Logistic scaler:",
    "VERIFIED"
)

print(
    "Logistic scaler refit:",
    False
)


# ============================================================
# LOAD FROZEN MODEL REGISTRY
# ============================================================

registry = pd.read_csv(
    CYCLE25_MODEL_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    registry
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen models."
    )


if set(
    registry[
        "family"
    ].astype(str)
) != set(
    EXPECTED_MODELS.keys()
):

    raise RuntimeError(
        "Cycle25 frozen model-family "
        "set mismatch."
    )


print(
    "Frozen Cycle25 model registry:",
    "PASS"
)


# ============================================================
# BUILD BASE PREDICTION TABLE
# ============================================================

prediction_table = pd.DataFrame(
    {
        "cycle25_row_id":
            np.arange(
                EXPECTED_ROWS,
                dtype=np.int64,
            ),

        "primary_block_id":
            primary_block_id.astype(
                str
            ),

        "ENTITY_GROUP":
            cycle25[
                ENTITY_GROUP_SOURCE_COLUMN
            ].copy(),

        "ENTITY_GROUP_ID":
            cycle25[
                ENTITY_GROUP_SOURCE_COLUMN
            ].copy(),

        "HARPNUM":
            cycle25[
                "HARPNUM"
            ].copy(),

        "target":
            target,
    }
)


# ============================================================
# PRESERVE SAFE SCALAR METADATA
# ============================================================

optional_columns = [
    "T_REC",
    "observation_time_utc",
    "NOAA_AR",
    "cycle",
]


for column in optional_columns:

    if column in cycle25.columns:

        prediction_table[
            column
        ] = cycle25[
            column
        ].copy()


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

manifest_rows = []


print(
    "\n===================================="
)

print(
    "GENERATING CYCLE25 PROBABILITIES"
)

print(
    "===================================="
)


for _, row in registry.iterrows():

    family = str(
        row[
            "family"
        ]
    )


    candidate_id = str(
        row[
            "candidate_id"
        ]
    )


    model_file = Path(
        str(
            row[
                "model_file"
            ]
        )
    )


    frozen_threshold = float(
        row[
            "frozen_threshold"
        ]
    )


    expected = EXPECTED_MODELS[
        family
    ]


    if (
        candidate_id
        !=
        expected[
            "candidate_id"
        ]
    ):

        raise RuntimeError(
            f"{family}: candidate identity "
            "differs from frozen expectation."
        )


    if abs(
        frozen_threshold
        -
        expected[
            "threshold"
        ]
    ) > 1e-9:

        raise RuntimeError(
            f"{candidate_id}: frozen "
            "threshold mismatch."
        )


    if not model_file.exists():

        raise FileNotFoundError(
            f"{candidate_id}: model file "
            f"missing: {model_file}"
        )


    actual_model_hash = sha256_file(
        model_file
    )


    expected_model_hash = str(
        row[
            "model_sha256"
        ]
    )


    if (
        actual_model_hash
        !=
        expected_model_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: model "
            "artifact hash mismatch."
        )


    bundle, estimator = load_estimator(
        model_file
    )


    verify_bundle_feature_order(
        bundle,
        candidate_id,
    )


    # --------------------------------------------------------
    # MODEL-SPECIFIC INPUT
    # --------------------------------------------------------

    if family == "logistic":

        model_input = X_logistic


    elif family == "random_forest":

        model_input = X_raw64


    elif family == "xgboost":

        model_input = X_raw64.astype(
            np.float32,
            copy=False,
        )


    else:

        raise RuntimeError(
            f"Unsupported model family: "
            f"{family}"
        )


    probabilities = estimator.predict_proba(
        model_input
    )[:, 1]


    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )


    if len(
        probabilities
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            f"{candidate_id}: probability "
            "vector row mismatch."
        )


    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "probability detected."
        )


    if (
        probabilities.min() < 0.0
        or
        probabilities.max() > 1.0
    ):

        raise RuntimeError(
            f"{candidate_id}: probability "
            "outside [0,1]."
        )


    classes = (
        probabilities
        >=
        frozen_threshold
    ).astype(
        np.int8
    )


    probability_column = (
        f"{candidate_id}__probability"
    )

    prediction_column = (
        f"{candidate_id}__prediction"
    )


    prediction_table[
        probability_column
    ] = probabilities


    prediction_table[
        prediction_column
    ] = classes


    manifest_rows.append(
        {
            "family":
                family,

            "candidate_id":
                candidate_id,

            "model_file":
                str(
                    model_file
                ),

            "model_sha256":
                actual_model_hash,

            "frozen_threshold":
                frozen_threshold,

            "threshold_source":
                "CYCLE24_VALIDATION",

            "rows":
                EXPECTED_ROWS,

            "probability_min":
                float(
                    probabilities.min()
                ),

            "probability_max":
                float(
                    probabilities.max()
                ),

            "probability_mean":
                float(
                    probabilities.mean()
                ),

            "probability_std":
                float(
                    probabilities.std()
                ),

            "predicted_positive":
                int(
                    classes.sum()
                ),

            "model_retrained":
                False,

            "preprocessing_refit":
                False,

            "threshold_reoptimized":
                False,

            "metrics_computed":
                False,
        }
    )


    print(
        f"{candidate_id}: "
        f"family={family}, "
        f"rows={len(probabilities)}, "
        f"p_min={probabilities.min():.6f}, "
        f"p_max={probabilities.max():.6f}, "
        f"p_mean={probabilities.mean():.6f}, "
        f"threshold={frozen_threshold:.10g}, "
        f"predicted_positive={int(classes.sum())}, "
        f"PASS=True"
    )


manifest = pd.DataFrame(
    manifest_rows
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PREDICTION_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_MANIFEST_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


prediction_table.to_parquet(
    OUTPUT_PREDICTION_FILE,
    index=False,
)


manifest.to_csv(
    OUTPUT_MANIFEST_FILE,
    index=False,
)


# ============================================================
# ROUND-TRIP VERIFICATION
# ============================================================

roundtrip = pd.read_parquet(
    OUTPUT_PREDICTION_FILE
)


if len(
    roundtrip
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Cycle25 prediction parquet "
        "round-trip row mismatch."
    )


if not np.array_equal(
    roundtrip[
        "cycle25_row_id"
    ].to_numpy(
        dtype=np.int64
    ),
    prediction_table[
        "cycle25_row_id"
    ].to_numpy(
        dtype=np.int64
    ),
):

    raise RuntimeError(
        "Cycle25 row identity "
        "round-trip mismatch."
    )


if not np.array_equal(
    roundtrip[
        "target"
    ].to_numpy(
        dtype=np.int64
    ),
    target,
):

    raise RuntimeError(
        "Cycle25 target round-trip mismatch."
    )


if not np.array_equal(
    roundtrip[
        "primary_block_id"
    ].astype(str).to_numpy(),
    prediction_table[
        "primary_block_id"
    ].astype(str).to_numpy(),
):

    raise RuntimeError(
        "Cycle25 temporal block "
        "round-trip mismatch."
    )


for _, row in registry.iterrows():

    candidate_id = str(
        row[
            "candidate_id"
        ]
    )


    probability_column = (
        f"{candidate_id}__probability"
    )


    prediction_column = (
        f"{candidate_id}__prediction"
    )


    original_prob = prediction_table[
        probability_column
    ].to_numpy(
        dtype=np.float64
    )


    saved_prob = roundtrip[
        probability_column
    ].to_numpy(
        dtype=np.float64
    )


    if not np.array_equal(
        original_prob,
        saved_prob,
    ):

        raise RuntimeError(
            f"{candidate_id}: probability "
            "round-trip mismatch."
        )


    original_class = prediction_table[
        prediction_column
    ].to_numpy(
        dtype=np.int64
    )


    saved_class = roundtrip[
        prediction_column
    ].to_numpy(
        dtype=np.int64
    )


    if not np.array_equal(
        original_class,
        saved_class,
    ):

        raise RuntimeError(
            f"{candidate_id}: prediction "
            "round-trip mismatch."
        )


print(
    "\nPrediction parquet round-trip:",
    "PASS"
)


# ============================================================
# ARTIFACT HASHES
# ============================================================

prediction_hash = sha256_file(
    OUTPUT_PREDICTION_FILE
)


manifest_hash = sha256_file(
    OUTPUT_MANIFEST_FILE
)


# ============================================================
# BLOCK SUMMARY
# ============================================================

block_contract_summary = []


for item in block_summary:

    block_contract_summary.append(
        {
            "block_id":
                item[
                    "block_id"
                ],

            "rows":
                item[
                    "rows"
                ],

            "entity_groups":
                item[
                    "entity_groups"
                ],
        }
    )


# ============================================================
# CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Frozen Solar Cycle 25 Prediction Artifact",

    "version":
        "1.0",

    "status":
        "FROZEN_CYCLE25_PREDICTIONS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_8_2",

    "source_dataset":
        {
            "file":
                str(
                    CYCLE25_DATASET_FILE
                ),

            "sha256":
                actual_dataset_hash,

            "rows":
                EXPECTED_ROWS,

            "positive":
                EXPECTED_POSITIVES,

            "negative":
                EXPECTED_NEGATIVES,

            "entity_groups":
                entity_group_count,

            "entity_group_source_column":
                ENTITY_GROUP_SOURCE_COLUMN,

            "time_assignment_column":
                TIME_COLUMN,
        },

    "feature_contract":
        {
            "feature_count":
                25,

            "feature_order":
                FEATURES,

            "changes_made":
                False,
        },

    "temporal_assignment":
        {
            "primary_block_count":
                3,

            "assignment_rule":
                (
                    "start-inclusive and "
                    "end-exclusive using "
                    "observation_time_utc"
                ),

            "all_rows_assigned_exactly_once":
                True,

            "blocks":
                block_contract_summary,

            "pooled_summary_available":
                True,

            "pooled_scope":
                "C25_ALL_2020_2025",
        },

    "models":
        manifest[
            [
                "family",
                "candidate_id",
                "model_file",
                "model_sha256",
                "frozen_threshold",
                "threshold_source",
            ]
        ].to_dict(
            orient="records"
        ),

    "prediction_policy":
        {
            "probability_source":
                "frozen_Cycle24_TRAIN_only_models",

            "classification_rule":
                (
                    "probability >= "
                    "frozen_threshold"
                ),

            "threshold_source":
                "CYCLE24_VALIDATION",

            "threshold_reoptimization":
                False,
        },

    "preprocessing_policy":
        {
            "logistic":
                "Frozen StandardScaler transform only",

            "random_forest":
                "Identity/raw frozen features",

            "xgboost":
                "Identity/raw frozen features",

            "scaler_refit":
                False,

            "cycle25_fitted_preprocessing":
                False,
        },

    "artifacts":
        {
            "prediction_artifact":
                {
                    "file":
                        str(
                            OUTPUT_PREDICTION_FILE
                        ),

                    "sha256":
                        prediction_hash,

                    "rows":
                        EXPECTED_ROWS,
                },

            "prediction_manifest":
                {
                    "file":
                        str(
                            OUTPUT_MANIFEST_FILE
                        ),

                    "sha256":
                        manifest_hash,

                    "rows":
                        3,
                },
        },

    "evaluation_state":
        {
            "cycle25_dataset_opened":
                True,

            "cycle25_probabilities_generated":
                True,

            "frozen_thresholds_applied":
                True,

            "point_metrics_computed":
                False,

            "calibration_computed":
                False,

            "bootstrap_ci_computed":
                False,

            "cross_cycle_drift_computed":
                False,

            "model_retraining":
                False,

            "preprocessing_refit":
                False,

            "threshold_reoptimization":
                False,

            "candidate_replacement":
                False,

            "feature_changes":
                False,
        },

    "data_isolation":
        {
            "cycle25_read":
                True,

            "cycle24_holdout_raw_dataset_read":
                False,

            "cycle24_validation_dataset_read":
                False,

            "cycle24_train_dataset_read":
                False,

            "new_model_fitting":
                False,

            "new_threshold_tuning":
                False,
        },

    "provenance":
        {
            "cycle25_evaluation_protocol":
                {
                    "file":
                        str(
                            CYCLE25_PROTOCOL_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CYCLE25_PROTOCOL_FILE
                        ),
                },

            "cycle25_model_registry":
                {
                    "file":
                        str(
                            CYCLE25_MODEL_REGISTRY_FILE
                        ),

                    "sha256":
                        registry_hash,
                },

            "cycle25_temporal_blocks":
                {
                    "file":
                        str(
                            CYCLE25_BLOCKS_FILE
                        ),

                    "sha256":
                        blocks_hash,
                },

            "logistic_scaler":
                {
                    "file":
                        str(
                            LOGISTIC_SCALER_FILE
                        ),

                    "sha256":
                        actual_scaler_hash,
                },
        },

    "immutability_rule":
        (
            "These Cycle25 probabilities were produced "
            "using the exact Cycle24-frozen models, "
            "preprocessing, feature order, temporal "
            "blocks, and validation-derived thresholds. "
            "The prediction artifact must not be used "
            "to alter the models, preprocessing, "
            "features, thresholds, or temporal blocks."
        ),
}


# ============================================================
# SAVE CONTRACT
# ============================================================

with OUTPUT_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "CYCLE25 PREDICTION SUMMARY"
)

print(
    "===================================="
)


print(
    "Cycle25 rows:",
    EXPECTED_ROWS
)

print(
    "Cycle25 positives:",
    EXPECTED_POSITIVES
)

print(
    "Cycle25 negatives:",
    EXPECTED_NEGATIVES
)

print(
    "ENTITY_GROUPs:",
    entity_group_count
)

print(
    "Primary temporal blocks:",
    3
)

print(
    "Models evaluated:",
    3
)

print(
    "Probability vectors:",
    3
)

print(
    "Frozen thresholds applied:",
    True
)


print(
    "\n===================================="
)

print(
    "EVALUATION STATE"
)

print(
    "===================================="
)


print(
    "Cycle25 opened:",
    True
)

print(
    "Cycle25 probabilities generated:",
    True
)

print(
    "Cycle25 metrics computed:",
    False
)

print(
    "Calibration computed:",
    False
)

print(
    "Bootstrap CIs computed:",
    False
)

print(
    "Cross-cycle drift computed:",
    False
)

print(
    "Models retrained:",
    False
)

print(
    "Preprocessing refit:",
    False
)

print(
    "Thresholds reoptimized:",
    False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_PREDICTION_FILE
)

print(
    OUTPUT_MANIFEST_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 8.2 COMPLETE — CYCLE25 PREDICTIONS FROZEN"
)

print(
    "===================================="
)