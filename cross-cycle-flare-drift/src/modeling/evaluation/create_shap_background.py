from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROTOCOL_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "shift_null_control_protocol.json"
)

CYCLE24_TRAIN_FILE = Path(
    "artifacts/splitting/data/"
    "cycle24_train.parquet"
)

OUTPUT_BACKGROUND_FILE = Path(
    "artifacts/models/shift_controls/shap/background/"
    "cycle24_train_shap_background.parquet"
)

OUTPUT_MANIFEST_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "shap_background_manifest.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "shap_background_contract.json"
)


# ============================================================
# FROZEN EXPECTATIONS
# ============================================================

EXPECTED_TRAIN_SHA256 = (
    "4801a6e3b11d39cef8d4b0ef0eea4ae1"
    "67c27862431662c3f43ff55e4c433def"
)

EXPECTED_TRAIN_ROWS = 62064
EXPECTED_TRAIN_GROUPS = 822

BACKGROUND_SIZE = 512
BACKGROUND_SEED = 29061

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
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


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 9.6A — FREEZE CYCLE24-TRAIN SHAP BACKGROUND"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

for path in [
    PROTOCOL_FILE,
    CYCLE24_TRAIN_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# PROTECT OUTPUTS
# ============================================================

for path in [
    OUTPUT_BACKGROUND_FILE,
    OUTPUT_MANIFEST_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter 9.6A artifacts."
        )


# ============================================================
# VERIFY 9.1 PROTOCOL
# ============================================================

protocol = load_json(
    PROTOCOL_FILE
)


if protocol.get(
    "status"
) != "FROZEN_BEFORE_SHIFT_AND_SHAP_ANALYSIS":

    raise RuntimeError(
        "Chapter 9.1 protocol is not frozen."
    )


shap_background_protocol = protocol[
    "shap_background"
]


if shap_background_protocol[
    "source"
] != "CYCLE24_TRAIN_ONLY":

    raise RuntimeError(
        "Frozen SHAP-background source mismatch."
    )


if int(
    shap_background_protocol[
        "size"
    ]
) != BACKGROUND_SIZE:

    raise RuntimeError(
        "Frozen SHAP-background size mismatch."
    )


if shap_background_protocol[
    "sampling_unit"
] != "ENTITY_GROUP":

    raise RuntimeError(
        "Frozen SHAP-background "
        "sampling unit mismatch."
    )


if int(
    shap_background_protocol[
        "seed"
    ]
) != BACKGROUND_SEED:

    raise RuntimeError(
        "Frozen SHAP-background seed mismatch."
    )


if shap_background_protocol[
    "cycle25_used"
]:

    raise RuntimeError(
        "Frozen protocol incorrectly "
        "allows Cycle25 background use."
    )


if protocol[
    "feature_contract"
][
    "feature_order"
] != FEATURES:

    raise RuntimeError(
        "Frozen feature order mismatch."
    )


print(
    "Chapter 9.1 protocol:",
    "FROZEN"
)

print(
    "Background source:",
    "CYCLE24_TRAIN_ONLY"
)

print(
    "Background size:",
    BACKGROUND_SIZE
)

print(
    "Sampling unit:",
    "ENTITY_GROUP"
)

print(
    "Frozen seed:",
    BACKGROUND_SEED
)


# ============================================================
# VERIFY TRAIN HASH
# ============================================================

train_hash = sha256_file(
    CYCLE24_TRAIN_FILE
)


if train_hash != EXPECTED_TRAIN_SHA256:

    raise RuntimeError(
        "Cycle24 TRAIN hash mismatch."
    )


if (
    train_hash
    !=
    protocol[
        "dataset_expectations"
    ][
        "cycle24_train"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle24 TRAIN hash differs "
        "from Chapter 9.1 protocol."
    )


print(
    "Cycle24 TRAIN hash:",
    "PASS"
)


# ============================================================
# READ CYCLE24 TRAIN
# ============================================================

train = pd.read_parquet(
    CYCLE24_TRAIN_FILE
)


if len(
    train
) != EXPECTED_TRAIN_ROWS:

    raise RuntimeError(
        "Cycle24 TRAIN row-count mismatch."
    )


required_columns = set(
    FEATURES
    +
    [
        ENTITY_GROUP_COLUMN,
        TIME_COLUMN,
    ]
)


missing_columns = (
    required_columns
    -
    set(
        train.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "Cycle24 TRAIN missing required "
        f"columns: {sorted(missing_columns)}"
    )


if train[
    ENTITY_GROUP_COLUMN
].isna().any():

    raise RuntimeError(
        "Cycle24 TRAIN contains missing "
        "ENTITY_GROUP_ID."
    )


group_count = int(
    train[
        ENTITY_GROUP_COLUMN
    ].nunique()
)


if group_count != EXPECTED_TRAIN_GROUPS:

    raise RuntimeError(
        "Cycle24 TRAIN ENTITY_GROUP "
        "count mismatch."
    )


print(
    "Cycle24 TRAIN rows:",
    len(
        train
    )
)

print(
    "Cycle24 TRAIN ENTITY_GROUPs:",
    group_count
)


# ============================================================
# FEATURE INTEGRITY
# ============================================================

feature_values = train[
    FEATURES
].to_numpy(
    dtype=np.float64
)


if not np.isfinite(
    feature_values
).all():

    raise RuntimeError(
        "Cycle24 TRAIN contains "
        "non-finite model features."
    )


observation_time = pd.to_datetime(
    train[
        TIME_COLUMN
    ],
    utc=True,
    errors="raise",
)


train = train.copy()

train[
    "_OBS_TIME_PARSED"
] = observation_time


print(
    "Frozen feature values:",
    "FINITE"
)

print(
    "Observation times:",
    "VALID"
)


# ============================================================
# SORT UNIQUE ENTITY GROUPS
# ============================================================

unique_groups = np.array(
    sorted(
        train[
            ENTITY_GROUP_COLUMN
        ].unique().tolist(),
        key=lambda value:
            str(value),
    ),
    dtype=object,
)


if len(
    unique_groups
) != EXPECTED_TRAIN_GROUPS:

    raise RuntimeError(
        "Sorted ENTITY_GROUP registry "
        "count mismatch."
    )


# ============================================================
# SAMPLE 512 GROUPS WITHOUT REPLACEMENT
# ============================================================

rng = np.random.default_rng(
    BACKGROUND_SEED
)


selected_indices = rng.choice(
    len(
        unique_groups
    ),
    size=BACKGROUND_SIZE,
    replace=False,
)


selected_groups = unique_groups[
    selected_indices
]


if len(
    np.unique(
        selected_groups
    )
) != BACKGROUND_SIZE:

    raise RuntimeError(
        "Background group sampling "
        "contains duplicates."
    )


selected_group_set = set(
    selected_groups.tolist()
)


print(
    "Selected ENTITY_GROUPs:",
    len(
        selected_group_set
    )
)

print(
    "Sampling without replacement:",
    "PASS"
)


# ============================================================
# SELECT ONE TEMPORAL-MEDIAN ROW PER GROUP
# ============================================================

background_rows = []
manifest_rows = []


for selection_order, group_id in enumerate(
    selected_groups,
    start=1,
):

    group_frame = train[
        train[
            ENTITY_GROUP_COLUMN
        ]
        ==
        group_id
    ].copy()


    if len(
        group_frame
    ) == 0:

        raise RuntimeError(
            f"Selected group not found: "
            f"{group_id}"
        )


    group_frame = (
        group_frame
        .sort_values(
            [
                "_OBS_TIME_PARSED",
            ],
            kind="mergesort",
        )
        .copy()
    )


    group_row_count = int(
        len(
            group_frame
        )
    )


    # Deterministic lower temporal median:
    # odd n  -> exact middle
    # even n -> earlier of the two middle rows
    median_position_zero_based = (
        group_row_count
        -
        1
    ) // 2


    selected_row = group_frame.iloc[
        median_position_zero_based
    ]


    source_index = selected_row.name


    background_row = {
        ENTITY_GROUP_COLUMN:
            selected_row[
                ENTITY_GROUP_COLUMN
            ],

        TIME_COLUMN:
            selected_row[
                TIME_COLUMN
            ],
    }


    for feature in FEATURES:

        background_row[
            feature
        ] = float(
            selected_row[
                feature
            ]
        )


    background_rows.append(
        background_row
    )


    manifest_rows.append(
        {
            "selection_order":
                selection_order,

            ENTITY_GROUP_COLUMN:
                selected_row[
                    ENTITY_GROUP_COLUMN
                ],

            "group_row_count":
                group_row_count,

            "median_position_zero_based":
                median_position_zero_based,

            "median_rule":
                "LOWER_TEMPORAL_MEDIAN",

            "selected_time":
                selected_row[
                    TIME_COLUMN
                ],

            "source_dataframe_index":
                source_index,
        }
    )


background_df = pd.DataFrame(
    background_rows
)


manifest_df = pd.DataFrame(
    manifest_rows
)


# ============================================================
# VERIFY BACKGROUND
# ============================================================

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
        "SHAP background does not contain "
        "512 unique ENTITY_GROUPs."
    )


if len(
    manifest_df
) != BACKGROUND_SIZE:

    raise RuntimeError(
        "SHAP background manifest "
        "row-count mismatch."
    )


if not np.isfinite(
    background_df[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )
).all():

    raise RuntimeError(
        "Background contains non-finite "
        "feature values."
    )


print(
    "Background rows:",
    len(
        background_df
    )
)

print(
    "Unique background ENTITY_GROUPs:",
    int(
        background_df[
            ENTITY_GROUP_COLUMN
        ].nunique()
    )
)

print(
    "Rows per selected group:",
    1
)

print(
    "Temporal-median rule:",
    "LOWER_TEMPORAL_MEDIAN"
)

print(
    "Background integrity:",
    "PASS"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_BACKGROUND_FILE.parent.mkdir(
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


background_df.to_parquet(
    OUTPUT_BACKGROUND_FILE,
    index=False,
)


manifest_df.to_csv(
    OUTPUT_MANIFEST_FILE,
    index=False,
)


# ============================================================
# HASHES
# ============================================================

background_hash = sha256_file(
    OUTPUT_BACKGROUND_FILE
)

manifest_hash = sha256_file(
    OUTPUT_MANIFEST_FILE
)


# ============================================================
# ROUND TRIP
# ============================================================

background_roundtrip = pd.read_parquet(
    OUTPUT_BACKGROUND_FILE
)

manifest_roundtrip = pd.read_csv(
    OUTPUT_MANIFEST_FILE,
    keep_default_na=False,
)


if len(
    background_roundtrip
) != BACKGROUND_SIZE:

    raise RuntimeError(
        "Background parquet "
        "round-trip failed."
    )


if len(
    manifest_roundtrip
) != BACKGROUND_SIZE:

    raise RuntimeError(
        "Background manifest "
        "round-trip failed."
    )


if int(
    background_roundtrip[
        ENTITY_GROUP_COLUMN
    ].nunique()
) != BACKGROUND_SIZE:

    raise RuntimeError(
        "Background round-trip unique "
        "group check failed."
    )


print(
    "Background parquet round-trip:",
    "PASS"
)

print(
    "Background manifest round-trip:",
    "PASS"
)


# ============================================================
# CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Cycle24-TRAIN Fixed SHAP Background",

    "version":
        "1.0",

    "status":
        "FROZEN_SHAP_BACKGROUND",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_9_6A",

    "source_population":
        {
            "name":
                "CYCLE24_TRAIN_ONLY",

            "file":
                str(
                    CYCLE24_TRAIN_FILE
                ),

            "sha256":
                train_hash,

            "rows":
                int(
                    len(
                        train
                    )
                ),

            "entity_groups":
                group_count,
        },

    "selection_design":
        {
            "background_rows":
                BACKGROUND_SIZE,

            "sampling_unit":
                "ENTITY_GROUP",

            "groups_sampled":
                BACKGROUND_SIZE,

            "groups_available":
                group_count,

            "replace":
                False,

            "seed":
                BACKGROUND_SEED,

            "rng":
                "numpy.random.default_rng",

            "unique_group_sort_rule":
                "STRING_ASCENDING",

            "representative_rows_per_group":
                1,

            "representative_row_rule":
                "TEMPORAL_MEDIAN",

            "even_group_tie_rule":
                "LOWER_TEMPORAL_MEDIAN",

            "time_column":
                TIME_COLUMN,
        },

    "feature_contract":
        {
            "feature_count":
                25,

            "feature_order":
                FEATURES,
        },

    "artifacts":
        {
            "background":
                {
                    "file":
                        str(
                            OUTPUT_BACKGROUND_FILE
                        ),

                    "sha256":
                        background_hash,

                    "rows":
                        len(
                            background_df
                        ),
                },

            "manifest":
                {
                    "file":
                        str(
                            OUTPUT_MANIFEST_FILE
                        ),

                    "sha256":
                        manifest_hash,

                    "rows":
                        len(
                            manifest_df
                        ),
                },
        },

    "data_access":
        {
            "cycle24_train_read":
                True,

            "cycle24_validation_read":
                False,

            "cycle24_holdout_read":
                False,

            "cycle25_read":
                False,

            "model_artifacts_loaded":
                False,

            "shap_computed":
                False,
        },

    "future_use":
        {
            "primary_cross_cycle_background":
                True,

            "within_cycle24_null_background":
                True,

            "same_background_all_models":
                True,

            "cycle25_may_modify_background":
                False,

            "background_may_be_reselected_after_shap":
                False,
        },

    "provenance":
        {
            "chapter_9_1_protocol":
                {
                    "file":
                        str(
                            PROTOCOL_FILE
                        ),

                    "sha256":
                        sha256_file(
                            PROTOCOL_FILE
                        ),
                },
        },

    "interpretation_rule":
        (
            "This background is a deterministic, "
            "Cycle24-TRAIN-derived reference for "
            "within-Cycle24 null analysis and later "
            "same-model cross-cycle SHAP analysis. "
            "It is not selected using Cycle25 or "
            "SHAP outcomes."
        ),

    "immutability_rule":
        (
            "After Chapter 9.6A, selected groups, "
            "representative rows, feature order, "
            "background size, and sampling seed "
            "must not change in response to SHAP "
            "or Cycle25 results."
        ),
}


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
    "CHAPTER 9.6A SUMMARY"
)

print(
    "===================================="
)


print(
    "Cycle24 TRAIN rows:",
    len(
        train
    )
)

print(
    "Cycle24 TRAIN ENTITY_GROUPs:",
    group_count
)

print(
    "Background groups selected:",
    BACKGROUND_SIZE
)

print(
    "Background rows:",
    len(
        background_df
    )
)

print(
    "Unique background groups:",
    int(
        background_df[
            ENTITY_GROUP_COLUMN
        ].nunique()
    )
)

print(
    "Selection seed:",
    BACKGROUND_SEED
)

print(
    "Cycle24 HOLDOUT read:",
    False
)

print(
    "Cycle25 read:",
    False
)

print(
    "Models loaded:",
    False
)

print(
    "SHAP computed:",
    False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_BACKGROUND_FILE
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
    "MODEL CHAPTER 9.6A COMPLETE — FIXED SHAP BACKGROUND FROZEN"
)

print(
    "===================================="
)