# ============================================================
# MODEL CHAPTER 10.1
# FREEZE CYCLE25 SHAP EXECUTION INPUTS
#
# NO SHAP COMPUTATION
# NO RETRAINING
# NO RETUNING
# NO CYCLE25-BASED FITTING
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_C25_ROWS = 79361
EXPECTED_C25_POSITIVES = 4836
EXPECTED_C25_NEGATIVES = 74525
EXPECTED_C25_GROUPS = 1011

EXPECTED_C25_HASH = (
    "0bd6a364e152685d13c7eb955da074a4"
    "b9f019f1851c6b5dafebf295c406cd4d"
)

EXPECTED_C24_TRAIN_HASH = (
    "4801a6e3b11d39cef8d4b0ef0eea4ae1"
    "67c27862431662c3f43ff55e4c433def"
)

EXPECTED_SCALER_HASH = (
    "48a9c07301f30052f49bad2db048418d"
    "3db16951f0433002b831b59fa10a1af5"
)


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


TARGET_COLUMN = "target"
ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TIME_COLUMN = "observation_time_utc"


BLOCKS = {
    "B1": {
        "years": [2020, 2021],
        "rows": 10411,
        "positives": 133,
        "groups": 146,
    },
    "B2": {
        "years": [2022, 2023],
        "rows": 33132,
        "positives": 1801,
        "groups": 414,
    },
    "B3": {
        "years": [2024, 2025],
        "rows": 35818,
        "positives": 2902,
        "groups": 458,
    },
}


MODEL_HASHES = {
    "LR_02_C0p01_BALANCED":
        (
            "7ab15bfc0f770e9ec12d5389aa51bbc1"
            "eac81f4b6b4580103642032ec9d61abd"
        ),

    "RF_04_D12_L5_BALSUB":
        (
            "519afcdf06e28f1da9b36d112bee9070"
            "bc51748e1c415fe2f3c244bd476bbd31"
        ),

    "XGB_02_LR0p03_D3_BAL":
        (
            "5839ac44333265b1b771008a82f6fc5a"
            "2f722e992386cde93fe8c04d48443d8b"
        ),
}


MODEL_CONFIG = {
    "LR_02_C0p01_BALANCED": {
        "family": "logistic",
        "explainer": "LinearExplainer",
        "model_representation": "FROZEN_SKLEARN_MODEL",
        "input_space": "FROZEN_STANDARDIZED",
        "background_space": "FROZEN_STANDARDIZED",
        "output_space": "LOG_ODDS_DECISION_FUNCTION",
        "feature_perturbation": "LINEAR_EXPLAINER",
        "requires_scaler": True,
    },

    "RF_04_D12_L5_BALSUB": {
        "family": "random_forest",
        "explainer": "TreeExplainer",
        "model_representation": "FROZEN_SKLEARN_MODEL",
        "input_space": "RAW",
        "background_space": "RAW",
        "output_space": "POSITIVE_CLASS_PROBABILITY",
        "feature_perturbation": "INTERVENTIONAL",
        "requires_scaler": False,
    },

    "XGB_02_LR0p03_D3_BAL": {
        "family": "xgboost",
        "explainer": "TreeExplainer",
        "model_representation": "FROZEN_XGBOOST_BOOSTER",
        "input_space": "RAW",
        "background_space": "RAW",
        "output_space": "XGBOOST_RAW_MARGIN",
        "feature_perturbation": "INTERVENTIONAL",
        "requires_scaler": False,
    },
}


# ============================================================
# INPUT PATHS
# ============================================================

C25_DATASET = Path(
    "data/final/dataset_v2/"
    "cycle25_dataset_v2.parquet"
)


SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/"
    "logistic_standard_scaler.joblib"
)


BACKGROUND_FILE = Path(
    "artifacts/models/shift_controls/shap/background/"
    "cycle24_train_shap_background.parquet"
)


BACKGROUND_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "shap_background_contract.json"
)


C24_SHAP_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "cycle24_holdout_shap_contract.json"
)


C24_NULL_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "cycle24_shap_null_contract.json"
)


CHAPTER9_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "chapter9_shift_null_controls_contract.json"
)


CHAPTER9_INVENTORY_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "chapter9_shift_null_controls_inventory.csv"
)


MODEL_PATHS = {
    "LR_02_C0p01_BALANCED":
        Path(
            "artifacts/models/logistic/models/"
            "LR_02_C0p01_BALANCED.joblib"
        ),

    "RF_04_D12_L5_BALSUB":
        Path(
            "artifacts/models/random_forest/models/"
            "RF_04_D12_L5_BALSUB.joblib"
        ),

    "XGB_02_LR0p03_D3_BAL":
        Path(
            "artifacts/models/xgboost/models/"
            "XGB_02_LR0p03_D3_BAL.joblib"
        ),
}


# ============================================================
# CHAPTER 10 OUTPUT PATHS
# ============================================================

OUTPUT_ROOT = Path(
    "artifacts/models/cross_cycle_shap"
)


POPULATION_MANIFEST_FILE = (
    OUTPUT_ROOT
    /
    "manifests"
    /
    "cycle25_shap_population_manifest.csv"
)


MODEL_MANIFEST_FILE = (
    OUTPUT_ROOT
    /
    "manifests"
    /
    "cycle25_shap_model_manifest.csv"
)


EXECUTION_CONTRACT_FILE = (
    OUTPUT_ROOT
    /
    "contracts"
    /
    "cycle25_shap_execution_inputs_contract.json"
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


def ordered_population_fingerprint(
    frame: pd.DataFrame,
    block_labels: pd.Series,
) -> str:

    digest = hashlib.sha256()

    timestamps = pd.to_datetime(
        frame[TIME_COLUMN],
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
        range(len(frame)),
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


def subset_fingerprint(
    frame: pd.DataFrame,
    block_labels: pd.Series,
    mask: np.ndarray,
) -> str:

    digest = hashlib.sha256()

    timestamps = pd.to_datetime(
        frame[TIME_COLUMN],
        utc=True,
    )

    groups = frame[
        ENTITY_GROUP_COLUMN
    ].astype(str)

    targets = frame[
        TARGET_COLUMN
    ].astype(int)

    indices = np.flatnonzero(
        mask
    )


    for row_number in indices:

        line = (
            f"{int(row_number)}|"
            f"{timestamps.iloc[row_number].isoformat()}|"
            f"{groups.iloc[row_number]}|"
            f"{int(targets.iloc[row_number])}|"
            f"{block_labels.iloc[row_number]}\n"
        )

        digest.update(
            line.encode(
                "utf-8"
            )
        )

    return digest.hexdigest()


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.1 — "
    "FREEZE CYCLE25 SHAP EXECUTION INPUTS"
)

print("=" * 78)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    C25_DATASET,
    SCALER_FILE,
    BACKGROUND_FILE,
    BACKGROUND_CONTRACT_FILE,
    C24_SHAP_CONTRACT_FILE,
    C24_NULL_CONTRACT_FILE,
    CHAPTER9_CONTRACT_FILE,
    CHAPTER9_INVENTORY_FILE,
    *MODEL_PATHS.values(),
]


for path in required_files:

    verify_exists(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for output_path in [
    POPULATION_MANIFEST_FILE,
    MODEL_MANIFEST_FILE,
    EXECUTION_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Output already exists: {output_path}\n"
            "Refusing to overwrite frozen "
            "Chapter 10.1 outputs."
        )


# ============================================================
# VERIFY CHAPTER 9 FINAL FREEZE
# ============================================================

chapter9_contract = load_json(
    CHAPTER9_CONTRACT_FILE
)


if (
    chapter9_contract.get("status")
    !=
    "FROZEN_CHAPTER9_SHIFT_NULL_CONTROLS"
):

    raise RuntimeError(
        "Chapter 9 controls are not frozen."
    )


print(
    "Chapter 9 final freeze:",
    "PASS"
)


# ============================================================
# VERIFY CHAPTER 9 INVENTORY HASH
# ============================================================

chapter9_inventory_hash = sha256_file(
    CHAPTER9_INVENTORY_FILE
)


if (
    chapter9_inventory_hash
    !=
    chapter9_contract[
        "artifact_inventory"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Chapter 9 inventory hash mismatch."
    )


chapter9_inventory = pd.read_csv(
    CHAPTER9_INVENTORY_FILE,
    keep_default_na=False,
)


if len(
    chapter9_inventory
) != 48:

    raise RuntimeError(
        "Expected exactly 48 frozen "
        "Chapter-9 artifacts."
    )


# ============================================================
# VERIFY ALL 48 CHAPTER-9 ARTIFACT HASHES
# ============================================================

print(
    "Auditing frozen Chapter 9 artifacts..."
)


for _, row in chapter9_inventory.iterrows():

    artifact_path = Path(
        str(
            row[
                "path"
            ]
        )
    )

    verify_exists(
        artifact_path
    )

    current_hash = sha256_file(
        artifact_path
    )

    expected_hash = str(
        row[
            "sha256"
        ]
    )

    if current_hash != expected_hash:

        raise RuntimeError(
            "Frozen Chapter-9 artifact changed:\n"
            f"{artifact_path}"
        )


print(
    "Chapter 9 artifact immutability:",
    "PASS"
)


# ============================================================
# VERIFY C24 SHAP CONTRACT
# ============================================================

c24_shap_contract = load_json(
    C24_SHAP_CONTRACT_FILE
)


if (
    c24_shap_contract.get("status")
    !=
    "FROZEN_CYCLE24_HOLDOUT_SHAP"
):

    raise RuntimeError(
        "Cycle24 HOLDOUT SHAP contract "
        "is not frozen."
    )


# ============================================================
# VERIFY C24 NULL CONTRACT
# ============================================================

c24_null_contract = load_json(
    C24_NULL_CONTRACT_FILE
)


if (
    c24_null_contract.get("status")
    !=
    "FROZEN_CYCLE24_SHAP_NULL_BASELINE"
):

    raise RuntimeError(
        "Cycle24 SHAP null contract "
        "is not frozen."
    )


if (
    c24_null_contract[
        "scientific_guardrails"
    ][
        "null_thresholds_frozen_before_cycle25_shap"
    ]
    is not True
):

    raise RuntimeError(
        "Null thresholds were not frozen "
        "before Cycle25 SHAP."
    )


print(
    "Cycle24 SHAP + null baseline:",
    "PASS"
)


# ============================================================
# VERIFY CYCLE25 DATASET HASH
# ============================================================

c25_hash = sha256_file(
    C25_DATASET
)


if c25_hash != EXPECTED_C25_HASH:

    raise RuntimeError(
        "Cycle25 dataset_v2 hash mismatch."
    )


print(
    "Cycle25 dataset hash:",
    "PASS"
)


# ============================================================
# LOAD CYCLE25
# ============================================================

c25 = pd.read_parquet(
    C25_DATASET
)


if len(
    c25
) != EXPECTED_C25_ROWS:

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
        "Cycle25 missing required columns: "
        f"{sorted(missing_columns)}"
    )


if c25[
    TARGET_COLUMN
].isna().any():

    raise RuntimeError(
        "Cycle25 target contains missing values."
    )


if c25[
    ENTITY_GROUP_COLUMN
].isna().any():

    raise RuntimeError(
        "Cycle25 ENTITY_GROUP_ID contains "
        "missing values."
    )


if c25[
    TIME_COLUMN
].isna().any():

    raise RuntimeError(
        "Cycle25 timestamps contain "
        "missing values."
    )


# ============================================================
# TARGET + GROUP AUDIT
# ============================================================

target = c25[
    TARGET_COLUMN
].to_numpy(
    dtype=np.int8
)


if not np.isin(
    target,
    [0, 1],
).all():

    raise RuntimeError(
        "Cycle25 target contains values "
        "outside {0,1}."
    )


positive_rows = int(
    np.sum(
        target == 1
    )
)


negative_rows = int(
    np.sum(
        target == 0
    )
)


group_count = int(
    c25[
        ENTITY_GROUP_COLUMN
    ].astype(str).nunique()
)


if positive_rows != EXPECTED_C25_POSITIVES:

    raise RuntimeError(
        "Cycle25 positive-count mismatch."
    )


if negative_rows != EXPECTED_C25_NEGATIVES:

    raise RuntimeError(
        "Cycle25 negative-count mismatch."
    )


if group_count != EXPECTED_C25_GROUPS:

    raise RuntimeError(
        "Cycle25 ENTITY_GROUP count mismatch."
    )


# ============================================================
# FEATURE MATRIX AUDIT
# ============================================================

X_raw = c25[
    FEATURES
].to_numpy(
    dtype=np.float64
)


if X_raw.shape != (
    EXPECTED_C25_ROWS,
    25,
):

    raise RuntimeError(
        "Cycle25 feature matrix shape mismatch."
    )


if not np.isfinite(
    X_raw
).all():

    raise RuntimeError(
        "Cycle25 feature matrix contains "
        "NaN/inf values."
    )


print(
    "Cycle25 population integrity:",
    "PASS"
)


# ============================================================
# TEMPORAL BLOCK ASSIGNMENT
# ============================================================

timestamps = pd.to_datetime(
    c25[
        TIME_COLUMN
    ],
    utc=True,
    errors="raise",
)


years = timestamps.dt.year


block_labels = pd.Series(
    index=c25.index,
    dtype="object",
)


block_labels.loc[
    years.isin(
        [2020, 2021]
    )
] = "B1"


block_labels.loc[
    years.isin(
        [2022, 2023]
    )
] = "B2"


block_labels.loc[
    years.isin(
        [2024, 2025]
    )
] = "B3"


if block_labels.isna().any():

    unexpected_years = sorted(
        years[
            block_labels.isna()
        ].unique().tolist()
    )

    raise RuntimeError(
        "Rows outside frozen Cycle25 "
        "temporal blocks. Years: "
        f"{unexpected_years}"
    )


# ============================================================
# POPULATION MANIFEST
# ============================================================

population_rows = []


def add_population(
    population_name: str,
    mask: np.ndarray,
    year_definition: str,
):

    subset = c25.loc[
        mask
    ]

    subset_target = subset[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.int8
    )

    subset_times = pd.to_datetime(
        subset[
            TIME_COLUMN
        ],
        utc=True,
    )

    positives = int(
        np.sum(
            subset_target == 1
        )
    )

    negatives = int(
        np.sum(
            subset_target == 0
        )
    )

    groups = int(
        subset[
            ENTITY_GROUP_COLUMN
        ].astype(str).nunique()
    )

    positive_groups = int(
        subset.loc[
            subset[
                TARGET_COLUMN
            ]
            ==
            1,
            ENTITY_GROUP_COLUMN,
        ]
        .astype(str)
        .nunique()
    )

    fingerprint = subset_fingerprint(
        frame=c25,
        block_labels=block_labels,
        mask=mask,
    )

    population_rows.append(
        {
            "population":
                population_name,

            "year_definition":
                year_definition,

            "rows":
                int(
                    len(
                        subset
                    )
                ),

            "positive_rows":
                positives,

            "negative_rows":
                negatives,

            "prevalence":
                float(
                    positives
                    /
                    len(
                        subset
                    )
                ),

            "entity_groups":
                groups,

            "positive_entity_groups":
                positive_groups,

            "minimum_time_utc":
                subset_times.min().isoformat(),

            "maximum_time_utc":
                subset_times.max().isoformat(),

            "ordered_subset_sha256":
                fingerprint,
        }
    )


# Pooled
add_population(
    population_name="POOLED",
    mask=np.ones(
        EXPECTED_C25_ROWS,
        dtype=bool,
    ),
    year_definition="2020-2025",
)


# B1-B3
for block_name, spec in BLOCKS.items():

    mask = (
        block_labels.to_numpy()
        ==
        block_name
    )

    add_population(
        population_name=block_name,
        mask=mask,
        year_definition=(
            f"{spec['years'][0]}-"
            f"{spec['years'][-1]}"
        ),
    )


population_manifest = pd.DataFrame(
    population_rows
)


# ============================================================
# VERIFY EXACT BLOCK COUNTS
# ============================================================

for block_name, spec in BLOCKS.items():

    row = population_manifest[
        population_manifest[
            "population"
        ]
        ==
        block_name
    ].iloc[0]


    if int(
        row[
            "rows"
        ]
    ) != spec[
        "rows"
    ]:

        raise RuntimeError(
            f"{block_name}: row-count mismatch."
        )


    if int(
        row[
            "positive_rows"
        ]
    ) != spec[
        "positives"
    ]:

        raise RuntimeError(
            f"{block_name}: positive-count mismatch."
        )


    if int(
        row[
            "entity_groups"
        ]
    ) != spec[
        "groups"
    ]:

        raise RuntimeError(
            f"{block_name}: ENTITY_GROUP "
            "count mismatch."
        )


if (
    population_manifest.loc[
        population_manifest[
            "population"
        ].isin(
            ["B1", "B2", "B3"]
        ),
        "rows",
    ].sum()
    !=
    EXPECTED_C25_ROWS
):

    raise RuntimeError(
        "B1+B2+B3 row totals do not "
        "equal pooled Cycle25."
    )


if (
    population_manifest.loc[
        population_manifest[
            "population"
        ].isin(
            ["B1", "B2", "B3"]
        ),
        "positive_rows",
    ].sum()
    !=
    EXPECTED_C25_POSITIVES
):

    raise RuntimeError(
        "B1+B2+B3 positive totals do "
        "not equal pooled Cycle25."
    )


print(
    "Cycle25 B1/B2/B3 blocks:",
    "PASS"
)


# ============================================================
# GROUPS SPANNING MULTIPLE TEMPORAL BLOCKS
# ============================================================

group_block_frame = pd.DataFrame(
    {
        ENTITY_GROUP_COLUMN:
            c25[
                ENTITY_GROUP_COLUMN
            ].astype(str),

        "block":
            block_labels.astype(str),
    }
)


group_block_counts = (
    group_block_frame
    .drop_duplicates()
    .groupby(
        ENTITY_GROUP_COLUMN
    )[
        "block"
    ]
    .nunique()
)


groups_spanning_blocks = int(
    (
        group_block_counts > 1
    ).sum()
)


maximum_blocks_per_group = int(
    group_block_counts.max()
)


# ============================================================
# FREEZE FULL ROW ORDER + BLOCK ASSIGNMENT
# ============================================================

ordered_population_hash = (
    ordered_population_fingerprint(
        frame=c25,
        block_labels=block_labels,
    )
)


print(
    "Cycle25 ordered population fingerprint:",
    ordered_population_hash
)


# ============================================================
# VERIFY FROZEN BACKGROUND
# ============================================================

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


background_hash = sha256_file(
    BACKGROUND_FILE
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
        "Frozen SHAP background hash mismatch."
    )


background = pd.read_parquet(
    BACKGROUND_FILE
)


if len(
    background
) != 512:

    raise RuntimeError(
        "Expected exactly 512 "
        "background rows."
    )


missing_background_features = (
    set(FEATURES)
    -
    set(background.columns)
)


if missing_background_features:

    raise RuntimeError(
        "Background missing features: "
        f"{sorted(missing_background_features)}"
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
        "Background contains non-finite values."
    )


print(
    "Frozen 512-row SHAP background:",
    "PASS"
)


# ============================================================
# VERIFY FROZEN LR SCALER
# ============================================================

scaler_hash = sha256_file(
    SCALER_FILE
)


if (
    scaler_hash
    !=
    EXPECTED_SCALER_HASH
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
        "LR scaler fit scope changed."
    )


if (
    scaler_bundle[
        "train_sha256"
    ]
    !=
    EXPECTED_C24_TRAIN_HASH
):

    raise RuntimeError(
        "LR scaler TRAIN provenance mismatch."
    )


if (
    scaler_bundle[
        "feature_order"
    ]
    !=
    FEATURES
):

    raise RuntimeError(
        "LR scaler feature order mismatch."
    )


scaler = scaler_bundle[
    "scaler"
]


X_scaled = scaler.transform(
    X_raw
)


background_scaled = scaler.transform(
    background_raw
)


if not np.isfinite(
    X_scaled
).all():

    raise RuntimeError(
        "Scaled Cycle25 matrix contains "
        "non-finite values."
    )


if not np.isfinite(
    background_scaled
).all():

    raise RuntimeError(
        "Scaled LR background contains "
        "non-finite values."
    )


print(
    "Frozen LR preprocessing:",
    "PASS"
)


# ============================================================
# FROZEN C24 SHAP MODEL CONFIGURATION
# ============================================================

frozen_c24_models = {
    model_row[
        "candidate_id"
    ]:
        model_row
    for model_row
    in c24_shap_contract[
        "models"
    ]
}


if set(
    frozen_c24_models.keys()
) != set(
    MODEL_PATHS.keys()
):

    raise RuntimeError(
        "Frozen C24 SHAP model set mismatch."
    )


# ============================================================
# VERIFY FROZEN MODELS
# ============================================================

model_manifest_rows = []


for candidate_id in [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]:

    print()

    print(
        "VERIFY MODEL:",
        candidate_id
    )


    path = MODEL_PATHS[
        candidate_id
    ]


    model_hash = sha256_file(
        path
    )


    if (
        model_hash
        !=
        MODEL_HASHES[
            candidate_id
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: model hash mismatch."
        )


    bundle = joblib.load(
        path
    )


    if (
        bundle[
            "candidate_id"
        ]
        !=
        candidate_id
    ):

        raise RuntimeError(
            f"{candidate_id}: candidate mismatch."
        )


    if (
        bundle[
            "training_scope"
        ]
        !=
        "CYCLE24_TRAIN_ONLY"
    ):

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            "training scope."
        )


    if (
        bundle[
            "train_sha256"
        ]
        !=
        EXPECTED_C24_TRAIN_HASH
    ):

        raise RuntimeError(
            f"{candidate_id}: C24 TRAIN "
            "provenance mismatch."
        )


    if (
        bundle[
            "feature_order"
        ]
        !=
        FEATURES
    ):

        raise RuntimeError(
            f"{candidate_id}: feature-order "
            "mismatch."
        )


    config = MODEL_CONFIG[
        candidate_id
    ]


    c24_model_contract = (
        frozen_c24_models[
            candidate_id
        ]
    )


    if (
        c24_model_contract[
            "model_sha256"
        ]
        !=
        model_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: model differs "
            "from frozen Cycle24 SHAP model."
        )


    if (
        c24_model_contract[
            "explainer"
        ]
        !=
        config[
            "explainer"
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: explainer changed."
        )


    if (
        c24_model_contract[
            "input_space"
        ]
        !=
        config[
            "input_space"
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: input space changed."
        )


    if (
        c24_model_contract[
            "output_space"
        ]
        !=
        config[
            "output_space"
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: output space changed."
        )


    model = bundle[
        "model"
    ]


    # --------------------------------------------------------
    # XGB wrapper -> exact fitted Booster equivalence on
    # FULL Cycle25 population.
    #
    # This is NOT SHAP computation.
    # --------------------------------------------------------

    wrapper_booster_equivalence = None
    wrapper_booster_mean_abs_error = None
    wrapper_booster_max_abs_error = None


    if candidate_id == "XGB_02_LR0p03_D3_BAL":

        booster = model.get_booster()


        wrapper_margin = np.asarray(
            model.predict(
                X_raw,
                output_margin=True,
            ),
            dtype=np.float64,
        ).reshape(-1)


        booster_margin = np.asarray(
            booster.inplace_predict(
                X_raw,
                predict_type="margin",
            ),
            dtype=np.float64,
        ).reshape(-1)


        difference = np.abs(
            wrapper_margin
            -
            booster_margin
        )


        wrapper_booster_equivalence = bool(
            np.allclose(
                wrapper_margin,
                booster_margin,
                atol=1e-12,
                rtol=1e-12,
            )
        )


        wrapper_booster_mean_abs_error = float(
            difference.mean()
        )


        wrapper_booster_max_abs_error = float(
            difference.max()
        )


        if not wrapper_booster_equivalence:

            raise RuntimeError(
                "XGB wrapper -> Booster equivalence "
                "failed on Cycle25."
            )


        print(
            "Wrapper -> Booster equivalence:",
            "PASS"
        )


    model_manifest_rows.append(
        {
            "candidate_id":
                candidate_id,

            "model_family":
                config[
                    "family"
                ],

            "model_file":
                path.as_posix(),

            "model_sha256":
                model_hash,

            "training_scope":
                "CYCLE24_TRAIN_ONLY",

            "train_sha256":
                EXPECTED_C24_TRAIN_HASH,

            "feature_count":
                25,

            "explainer":
                config[
                    "explainer"
                ],

            "model_input_representation":
                config[
                    "model_representation"
                ],

            "feature_input_space":
                config[
                    "input_space"
                ],

            "background_input_space":
                config[
                    "background_space"
                ],

            "output_space":
                config[
                    "output_space"
                ],

            "feature_perturbation":
                config[
                    "feature_perturbation"
                ],

            "background_rows":
                512,

            "background_sha256":
                background_hash,

            "requires_scaler":
                bool(
                    config[
                        "requires_scaler"
                    ]
                ),

            "scaler_sha256":
                (
                    scaler_hash
                    if config[
                        "requires_scaler"
                    ]
                    else
                    ""
                ),

            "xgb_wrapper_booster_equivalence":
                (
                    ""
                    if wrapper_booster_equivalence is None
                    else
                    wrapper_booster_equivalence
                ),

            "xgb_wrapper_booster_mean_abs_error":
                (
                    ""
                    if wrapper_booster_mean_abs_error is None
                    else
                    wrapper_booster_mean_abs_error
                ),

            "xgb_wrapper_booster_max_abs_error":
                (
                    ""
                    if wrapper_booster_max_abs_error is None
                    else
                    wrapper_booster_max_abs_error
                ),

            "retraining_allowed":
                False,

            "retuning_allowed":
                False,

            "cycle25_fit_allowed":
                False,
        }
    )


    print(
        "Frozen model execution inputs:",
        "PASS"
    )


model_manifest = pd.DataFrame(
    model_manifest_rows
)


if len(
    model_manifest
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen models."
    )


# ============================================================
# DOWNSTREAM CONTROL REFERENCES
# ============================================================

def get_inventory_role(
    role: str,
):

    subset = chapter9_inventory[
        chapter9_inventory[
            "role"
        ].astype(str)
        ==
        role
    ]


    if len(
        subset
    ) != 1:

        raise RuntimeError(
            f"Chapter-9 inventory role "
            f"lookup failed: {role}"
        )


    row = subset.iloc[
        0
    ]


    return {
        "path":
            str(
                row[
                    "path"
                ]
            ),

        "sha256":
            str(
                row[
                    "sha256"
                ]
            ),
    }


downstream_controls = {

    "structure_null_thresholds":
        get_inventory_role(
            "SHAP_NULL_STRUCTURE_THRESHOLDS"
        ),

    "featurewise_null_thresholds":
        get_inventory_role(
            "SHAP_NULL_FEATUREWISE_THRESHOLDS"
        ),

    "common_support_summary":
        get_inventory_role(
            "COMMON_SUPPORT_SUMMARY"
        ),

    "domain_common_support":
        get_inventory_role(
            "DOMAIN_OOF_COMMON_SUPPORT"
        ),

    "correlated_feature_groups":
        get_inventory_role(
            "CORRELATED_FEATURE_GROUPS"
        ),

    "correlated_feature_groups_contract":
        get_inventory_role(
            "CORRELATED_FEATURE_GROUPS_CONTRACT"
        ),
}


print()

print(
    "Frozen downstream control references:",
    "PASS"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

POPULATION_MANIFEST_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_MANIFEST_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


EXECUTION_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAVE MANIFESTS
# ============================================================

population_manifest.to_csv(
    POPULATION_MANIFEST_FILE,
    index=False,
)


model_manifest.to_csv(
    MODEL_MANIFEST_FILE,
    index=False,
)


# ============================================================
# ROUND-TRIP
# ============================================================

population_rt = pd.read_csv(
    POPULATION_MANIFEST_FILE,
    keep_default_na=False,
)


model_rt = pd.read_csv(
    MODEL_MANIFEST_FILE,
    keep_default_na=False,
)


if len(
    population_rt
) != 4:

    raise RuntimeError(
        "Cycle25 population manifest "
        "round-trip failed."
    )


if len(
    model_rt
) != 3:

    raise RuntimeError(
        "Cycle25 model manifest "
        "round-trip failed."
    )


population_manifest_hash = sha256_file(
    POPULATION_MANIFEST_FILE
)


model_manifest_hash = sha256_file(
    MODEL_MANIFEST_FILE
)


print(
    "Manifest round-trip:",
    "PASS"
)


# ============================================================
# FINAL 10.1 CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Cycle25 SHAP "
            "Execution Inputs"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CYCLE25_SHAP_EXECUTION_INPUTS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_1",

    "cycle25_population":
        {
            "dataset_file":
                C25_DATASET.as_posix(),

            "dataset_sha256":
                c25_hash,

            "rows":
                EXPECTED_C25_ROWS,

            "positive_rows":
                EXPECTED_C25_POSITIVES,

            "negative_rows":
                EXPECTED_C25_NEGATIVES,

            "entity_groups":
                EXPECTED_C25_GROUPS,

            "features":
                25,

            "feature_order":
                FEATURES,

            "ordered_population_sha256":
                ordered_population_hash,

            "dataset_row_order_preserved":
                True,

            "timestamp_sort_applied":
                False,

            "timestamp_monotonic_in_original_order":
                bool(
                    timestamps.is_monotonic_increasing
                ),
        },

    "temporal_blocks":
        {
            "definition":
                {
                    "B1":
                        "2020-2021",

                    "B2":
                        "2022-2023",

                    "B3":
                        "2024-2025",
                },

            "assignment_basis":
                TIME_COLUMN,

            "row_membership_rule":
                (
                    "Calendar year of "
                    "observation_time_utc"
                ),

            "groups_spanning_multiple_blocks":
                groups_spanning_blocks,

            "maximum_blocks_per_group":
                maximum_blocks_per_group,

            "population_manifest":
                {
                    "file":
                        POPULATION_MANIFEST_FILE.as_posix(),

                    "sha256":
                        population_manifest_hash,

                    "rows":
                        4,
                },
        },

    "frozen_models":
        {
            "count":
                3,

            "model_manifest":
                {
                    "file":
                        MODEL_MANIFEST_FILE.as_posix(),

                    "sha256":
                        model_manifest_hash,

                    "rows":
                        3,
                },

            "training_scope":
                "CYCLE24_TRAIN_ONLY",

            "model_retraining_allowed":
                False,

            "model_retuning_allowed":
                False,

            "cycle25_training_allowed":
                False,
        },

    "preprocessing":
        {
            "lr_scaler_file":
                SCALER_FILE.as_posix(),

            "lr_scaler_sha256":
                scaler_hash,

            "fit_scope":
                "CYCLE24_TRAIN_ONLY",

            "cycle25_fit_allowed":
                False,
        },

    "shap_background":
        {
            "file":
                BACKGROUND_FILE.as_posix(),

            "sha256":
                background_hash,

            "rows":
                512,

            "source":
                "CYCLE24_TRAIN_ONLY",

            "changed_for_cycle25":
                False,
        },

    "chapter9_controls":
        {
            "contract_file":
                CHAPTER9_CONTRACT_FILE.as_posix(),

            "contract_sha256":
                sha256_file(
                    CHAPTER9_CONTRACT_FILE
                ),

            "inventory_file":
                CHAPTER9_INVENTORY_FILE.as_posix(),

            "inventory_sha256":
                chapter9_inventory_hash,

            "frozen_artifacts_verified":
                48,

            "all_verified_unchanged":
                True,

            "downstream_controls":
                downstream_controls,
        },

    "explanation_comparison_rule":
        {
            "same_model_cross_population_only":
                True,

            "cross_model_raw_shap_magnitude_comparison":
                False,

            "lr_output":
                "LOG_ODDS_DECISION_FUNCTION",

            "rf_output":
                "POSITIVE_CLASS_PROBABILITY",

            "xgb_output":
                "XGBOOST_RAW_MARGIN",

            "shap_is_causal":
                False,
        },

    "scientific_guardrails":
        {
            "cycle25_dataset_read":
                True,

            "cycle25_shap_generated":
                False,

            "cycle25_used_for_training":
                False,

            "cycle25_used_for_model_selection":
                False,

            "cycle25_used_for_feature_selection":
                False,

            "cycle25_used_for_scaler_fit":
                False,

            "cycle25_used_for_threshold_tuning":
                False,

            "models_retrained":
                False,

            "models_retuned":
                False,

            "background_changed":
                False,

            "cycle24_null_changed":
                False,

            "correlated_feature_groups_changed":
                False,

            "common_support_definition_changed":
                False,

            "drift_claim_made":
                False,
        },

    "software":
        {
            "numpy":
                np.__version__,

            "pandas":
                pd.__version__,

            "scikit_learn":
                sklearn.__version__,

            "xgboost":
                xgboost.__version__,

            "shap":
                shap.__version__,
        },

    "immutability_rule":
        (
            "After Chapter 10.1 is frozen, "
            "the Cycle25 dataset population, "
            "row order, temporal block definitions, "
            "feature order, frozen model binaries, "
            "LR preprocessing, SHAP background, "
            "model-specific explanation spaces, "
            "Chapter-9 null thresholds, common-support "
            "definition, and correlated-feature groups "
            "must not be changed in response to "
            "Cycle25 SHAP results."
        ),

    "next_stage":
        "MODEL_CHAPTER_10_2_CYCLE25_SHAP_GENERATION",
}


with EXECUTION_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


# ============================================================
# FINAL CONTRACT ROUND-TRIP
# ============================================================

contract_rt = load_json(
    EXECUTION_CONTRACT_FILE
)


if (
    contract_rt.get(
        "status"
    )
    !=
    "FROZEN_CYCLE25_SHAP_EXECUTION_INPUTS"
):

    raise RuntimeError(
        "Chapter 10.1 contract "
        "round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "cycle25_shap_generated"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected Cycle25 SHAP state."
    )


contract_hash = sha256_file(
    EXECUTION_CONTRACT_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 10.1 — COMPLETE"
)

print("=" * 78)


print(
    "Cycle25 rows:",
    EXPECTED_C25_ROWS
)

print(
    "Cycle25 positives:",
    EXPECTED_C25_POSITIVES
)

print(
    "Cycle25 groups:",
    EXPECTED_C25_GROUPS
)


print()

for block_name in [
    "B1",
    "B2",
    "B3",
]:

    row = population_manifest[
        population_manifest[
            "population"
        ]
        ==
        block_name
    ].iloc[0]


    print(
        f"{block_name}: "
        f"rows={int(row['rows'])}, "
        f"positives={int(row['positive_rows'])}, "
        f"groups={int(row['entity_groups'])}"
    )


print()

print(
    "Frozen models:",
    3
)

print(
    "Frozen background rows:",
    512
)

print(
    "Chapter-9 artifacts verified:",
    48
)

print(
    "Cycle25 SHAP generated:",
    False
)

print(
    "Models retrained:",
    False
)

print(
    "Cycle25 used for fitting:",
    False
)

print(
    "Drift conclusion made:",
    False
)


print()

print(
    "Status:",
    "FROZEN_CYCLE25_SHAP_EXECUTION_INPUTS"
)


print()

print(
    "Population manifest:",
    POPULATION_MANIFEST_FILE
)

print(
    "Model manifest:",
    MODEL_MANIFEST_FILE
)

print(
    "Execution contract:",
    EXECUTION_CONTRACT_FILE
)

print(
    "Contract SHA256:",
    contract_hash
)


print()

print(
    "NEXT:",
    "Chapter 10.2 — Generate frozen-model Cycle25 SHAP"
)