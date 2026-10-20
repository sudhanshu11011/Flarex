from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# PATHS
# ============================================================

PROTOCOL_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "shift_null_control_protocol.json"
)

CYCLE25_BLOCKS_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_temporal_blocks.csv"
)

CYCLE24_HOLDOUT_FILE = Path(
    "artifacts/splitting/data/"
    "cycle24_holdout.parquet"
)

CYCLE25_FILE = Path(
    "data/final/dataset_v2/"
    "cycle25_dataset_v2.parquet"
)

SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/"
    "logistic_standard_scaler.joblib"
)


OUTPUT_CV_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "joint_domain_shift_cv.csv"
)

OUTPUT_SUPPORT_SUMMARY_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "common_support_summary.csv"
)

OUTPUT_OOF_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "domain_oof_common_support.parquet"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "joint_shift_common_support_contract.json"
)


# ============================================================
# FROZEN EXPECTATIONS
# ============================================================

EXPECTED_C24_HASH = (
    "8abb3929febcde7d778c4781ac890ff5e"
    "658933c2ef2a06aa19da085e7e255fc"
)

EXPECTED_C25_HASH = (
    "0bd6a364e152685d13c7eb955da074a4"
    "b9f019f1851c6b5dafebf295c406cd4d"
)

EXPECTED_SCALER_HASH = (
    "48a9c07301f30052f49bad2db048418d"
    "3db16951f0433002b831b59fa10a1af5"
)

EXPECTED_TRAIN_HASH = (
    "4801a6e3b11d39cef8d4b0ef0eea4ae1"
    "67c27862431662c3f43ff55e4c433def"
)

EXPECTED_TRAIN_ROWS = 62064

EXPECTED_C24_ROWS = 13072
EXPECTED_C25_ROWS = 79361

EXPECTED_C24_GROUPS = 176
EXPECTED_C25_GROUPS = 1011

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TARGET_COLUMN = "target"
TIME_COLUMN = "observation_time_utc"

CV_SPLITS = 5
DOMAIN_SEED = 29041

PREDICTION_THRESHOLD = 0.5

SUPPORT_Q_LOW = 0.01
SUPPORT_Q_HIGH = 0.99

WEIGHT_Q_LOW = 0.01
WEIGHT_Q_HIGH = 0.99


PRIMARY_BLOCKS = [
    "C25_B1_2020_2021",
    "C25_B2_2022_2023",
    "C25_B3_2024_2025",
]

POOLED_SCOPE = "C25_ALL_2020_2025"

SUMMARY_SCOPE_ORDER = [
    "C24_HOLDOUT",
    *PRIMARY_BLOCKS,
    POOLED_SCOPE,
]


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


def quantile(values, probability):

    return float(
        np.quantile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            probability,
            method="linear",
        )
    )


def effective_sample_size(weights):

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    if len(weights) == 0:
        return 0.0

    denominator = float(
        np.sum(
            weights ** 2
        )
    )

    if denominator <= 0.0:
        return 0.0

    return float(
        (
            np.sum(weights) ** 2
        )
        /
        denominator
    )


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 9.4 — JOINT DOMAIN SHIFT + COMMON SUPPORT"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    PROTOCOL_FILE,
    CYCLE25_BLOCKS_FILE,
    CYCLE24_HOLDOUT_FILE,
    CYCLE25_FILE,
    SCALER_FILE,
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
    OUTPUT_CV_FILE,
    OUTPUT_SUPPORT_SUMMARY_FILE,
    OUTPUT_OOF_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter 9.4 results."
        )


# ============================================================
# LOAD + VERIFY 9.1 PROTOCOL
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


joint_protocol = protocol[
    "joint_distribution_shift"
]

support_protocol = protocol[
    "common_support_reweighting"
]


if joint_protocol[
    "primary_method"
] != "DOMAIN_CLASSIFIER":

    raise RuntimeError(
        "Frozen joint-shift method mismatch."
    )


classifier_protocol = joint_protocol[
    "classifier"
]


if classifier_protocol[
    "type"
] != "LogisticRegression":

    raise RuntimeError(
        "Frozen domain classifier mismatch."
    )


if float(
    classifier_protocol[
        "C"
    ]
) != 1.0:

    raise RuntimeError(
        "Frozen domain-classifier C mismatch."
    )


if classifier_protocol[
    "class_weight"
] != "balanced":

    raise RuntimeError(
        "Frozen class_weight mismatch."
    )


validation_protocol = joint_protocol[
    "validation"
]


if validation_protocol[
    "type"
] != "StratifiedGroupKFold":

    raise RuntimeError(
        "Frozen domain CV mismatch."
    )


if int(
    validation_protocol[
        "n_splits"
    ]
) != CV_SPLITS:

    raise RuntimeError(
        "Frozen CV split count mismatch."
    )


if int(
    validation_protocol[
        "random_state"
    ]
) != DOMAIN_SEED:

    raise RuntimeError(
        "Frozen CV random-state mismatch."
    )


if joint_protocol[
    "features"
] != FEATURES:

    raise RuntimeError(
        "Frozen feature order mismatch."
    )


print(
    "Chapter 9.1 protocol:",
    "FROZEN"
)

print(
    "Joint-shift classifier:",
    "LogisticRegression"
)

print(
    "Cross-validation:",
    "StratifiedGroupKFold"
)

print(
    "CV folds:",
    CV_SPLITS
)

print(
    "Frozen 25-feature order:",
    "PASS"
)


# ============================================================
# HASH VERIFICATION
# ============================================================

c24_hash = sha256_file(
    CYCLE24_HOLDOUT_FILE
)

c25_hash = sha256_file(
    CYCLE25_FILE
)

scaler_hash = sha256_file(
    SCALER_FILE
)


if c24_hash != EXPECTED_C24_HASH:

    raise RuntimeError(
        "Cycle24 holdout hash mismatch."
    )


if c25_hash != EXPECTED_C25_HASH:

    raise RuntimeError(
        "Cycle25 dataset hash mismatch."
    )


if scaler_hash != EXPECTED_SCALER_HASH:

    raise RuntimeError(
        "Frozen StandardScaler hash mismatch."
    )


if (
    c24_hash
    !=
    protocol[
        "dataset_expectations"
    ][
        "cycle24_holdout"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle24 hash differs from "
        "Chapter 9.1 protocol."
    )


if (
    c25_hash
    !=
    protocol[
        "dataset_expectations"
    ][
        "cycle25"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle25 hash differs from "
        "Chapter 9.1 protocol."
    )


print(
    "Cycle24 holdout hash:",
    "PASS"
)

print(
    "Cycle25 dataset hash:",
    "PASS"
)

print(
    "Frozen StandardScaler hash:",
    "PASS"
)


# ============================================================
# LOAD FROZEN SCALER
# ============================================================

scaler_bundle = joblib.load(
    SCALER_FILE
)


if not isinstance(
    scaler_bundle,
    dict,
):

    raise RuntimeError(
        "Expected scaler artifact to be "
        "a dictionary bundle."
    )


required_scaler_keys = {
    "scaler",
    "feature_order",
    "feature_count",
    "fit_scope",
    "train_rows",
    "train_sha256",
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
        "Scaler bundle missing keys: "
        f"{sorted(missing_scaler_keys)}"
    )


if scaler_bundle[
    "feature_order"
] != FEATURES:

    raise RuntimeError(
        "Scaler feature order mismatch."
    )


if int(
    scaler_bundle[
        "feature_count"
    ]
) != 25:

    raise RuntimeError(
        "Scaler feature count mismatch."
    )


# IMPORTANT:
# Actual frozen artifact metadata is CYCLE24_TRAIN_ONLY.
if scaler_bundle[
    "fit_scope"
] != "CYCLE24_TRAIN_ONLY":

    raise RuntimeError(
        "Scaler was not fitted only "
        "on Cycle24 TRAIN."
    )


if int(
    scaler_bundle[
        "train_rows"
    ]
) != EXPECTED_TRAIN_ROWS:

    raise RuntimeError(
        "Scaler TRAIN row-count "
        "provenance mismatch."
    )


if scaler_bundle[
    "train_sha256"
] != EXPECTED_TRAIN_HASH:

    raise RuntimeError(
        "Scaler TRAIN hash "
        "provenance mismatch."
    )


scaler = scaler_bundle[
    "scaler"
]


print(
    "Frozen scaler fit scope:",
    "CYCLE24_TRAIN_ONLY"
)

print(
    "Scaler TRAIN rows:",
    scaler_bundle[
        "train_rows"
    ]
)

print(
    "Scaler TRAIN hash:",
    "PASS"
)

print(
    "Scaler feature contract:",
    "PASS"
)


# ============================================================
# READ EVALUATION DATA
# ============================================================

c24 = pd.read_parquet(
    CYCLE24_HOLDOUT_FILE
)

c25 = pd.read_parquet(
    CYCLE25_FILE
)


required_columns = set(
    FEATURES
    +
    [
        ENTITY_GROUP_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN,
    ]
)


for name, dataframe in [
    ("C24_HOLDOUT", c24),
    ("CYCLE25", c25),
]:

    missing = (
        required_columns
        -
        set(
            dataframe.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"{name}: missing required "
            f"columns {sorted(missing)}"
        )


if len(c24) != EXPECTED_C24_ROWS:

    raise RuntimeError(
        "Cycle24 holdout row-count mismatch."
    )


if len(c25) != EXPECTED_C25_ROWS:

    raise RuntimeError(
        "Cycle25 row-count mismatch."
    )


if int(
    c24[
        ENTITY_GROUP_COLUMN
    ].nunique()
) != EXPECTED_C24_GROUPS:

    raise RuntimeError(
        "Cycle24 group-count mismatch."
    )


if int(
    c25[
        ENTITY_GROUP_COLUMN
    ].nunique()
) != EXPECTED_C25_GROUPS:

    raise RuntimeError(
        "Cycle25 group-count mismatch."
    )


print(
    "Cycle24 holdout population:",
    "PASS"
)

print(
    "Cycle25 population:",
    "PASS"
)


# ============================================================
# FEATURE FINITENESS
# ============================================================

X24_raw = c24[
    FEATURES
].to_numpy(
    dtype=np.float64
)

X25_raw = c25[
    FEATURES
].to_numpy(
    dtype=np.float64
)


if not np.isfinite(
    X24_raw
).all():

    raise RuntimeError(
        "Cycle24 contains non-finite "
        "model features."
    )


if not np.isfinite(
    X25_raw
).all():

    raise RuntimeError(
        "Cycle25 contains non-finite "
        "model features."
    )


print(
    "Frozen feature values finite:",
    "PASS"
)


# ============================================================
# ASSIGN FROZEN CYCLE25 BLOCKS
# ============================================================

blocks = pd.read_csv(
    CYCLE25_BLOCKS_FILE,
    keep_default_na=False,
)


primary_blocks = blocks[
    blocks[
        "primary_block"
    ].astype(
        str
    ).str.lower()
    ==
    "true"
].copy()


if len(
    primary_blocks
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen "
        "Cycle25 temporal blocks."
    )


c25_time = pd.to_datetime(
    c25[
        TIME_COLUMN
    ],
    utc=True,
    errors="raise",
)


c25_block = pd.Series(
    pd.NA,
    index=c25.index,
    dtype="string",
)


for _, block in primary_blocks.iterrows():

    current_id = str(
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
        c25_time
        >=
        start
    ) & (
        c25_time
        <
        end_exclusive
    )


    if c25_block.loc[
        mask
    ].notna().any():

        raise RuntimeError(
            "Cycle25 temporal-block overlap."
        )


    c25_block.loc[
        mask
    ] = current_id


if c25_block.isna().any():

    raise RuntimeError(
        "Some Cycle25 rows were not "
        "assigned to a frozen block."
    )


if set(
    c25_block.astype(
        str
    ).unique()
) != set(
    PRIMARY_BLOCKS
):

    raise RuntimeError(
        "Cycle25 block IDs mismatch."
    )


print(
    "Frozen Cycle25 block assignment:",
    "PASS"
)


# ============================================================
# FROZEN TRAIN-SCALER TRANSFORM
# ============================================================

X24 = scaler.transform(
    X24_raw
)

X25 = scaler.transform(
    X25_raw
)


if not np.isfinite(
    X24
).all():

    raise RuntimeError(
        "Scaled Cycle24 matrix "
        "contains non-finite values."
    )


if not np.isfinite(
    X25
).all():

    raise RuntimeError(
        "Scaled Cycle25 matrix "
        "contains non-finite values."
    )


X = np.vstack(
    [
        X24,
        X25,
    ]
)


domain_y = np.concatenate(
    [
        np.zeros(
            len(c24),
            dtype=np.int8,
        ),

        np.ones(
            len(c25),
            dtype=np.int8,
        ),
    ]
)


# Prefix groups so group identifiers
# can never collide across cycles.
domain_groups = np.concatenate(
    [
        (
            "C24::"
            +
            c24[
                ENTITY_GROUP_COLUMN
            ].astype(
                str
            )
        ).to_numpy(),

        (
            "C25::"
            +
            c25[
                ENTITY_GROUP_COLUMN
            ].astype(
                str
            )
        ).to_numpy(),
    ]
)


if len(
    np.unique(
        domain_groups
    )
) != (
    EXPECTED_C24_GROUPS
    +
    EXPECTED_C25_GROUPS
):

    raise RuntimeError(
        "Prefixed domain group-count mismatch."
    )


print(
    "Frozen Cycle24-TRAIN scaler applied:",
    "PASS"
)

print(
    "Combined domain rows:",
    len(
        domain_y
    )
)

print(
    "Combined domain groups:",
    len(
        np.unique(
            domain_groups
        )
    )
)


# ============================================================
# GROUP-SAFE DOMAIN CLASSIFIER CV
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=CV_SPLITS,
    shuffle=True,
    random_state=DOMAIN_SEED,
)


oof_probability = np.full(
    len(
        domain_y
    ),
    np.nan,
    dtype=np.float64,
)


fold_assignment = np.full(
    len(
        domain_y
    ),
    -1,
    dtype=np.int16,
)


cv_rows = []


print(
    "\n===================================="
)

print(
    "RUNNING GROUP-SAFE DOMAIN CLASSIFIER"
)

print(
    "===================================="
)


for fold_id, (
    train_index,
    test_index,
) in enumerate(
    cv.split(
        X,
        domain_y,
        groups=domain_groups,
    ),
    start=1,
):

    train_groups = set(
        domain_groups[
            train_index
        ]
    )

    test_groups = set(
        domain_groups[
            test_index
        ]
    )


    if train_groups & test_groups:

        raise RuntimeError(
            f"Fold {fold_id}: "
            "ENTITY_GROUP leakage detected."
        )


    y_train = domain_y[
        train_index
    ]

    y_test = domain_y[
        test_index
    ]


    if set(
        np.unique(
            y_train
        ).tolist()
    ) != {0, 1}:

        raise RuntimeError(
            f"Fold {fold_id}: training fold "
            "does not contain both domains."
        )


    if set(
        np.unique(
            y_test
        ).tolist()
    ) != {0, 1}:

        raise RuntimeError(
            f"Fold {fold_id}: test fold "
            "does not contain both domains."
        )


    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=DOMAIN_SEED,
    )


    classifier.fit(
        X[
            train_index
        ],
        y_train,
    )


    probability = classifier.predict_proba(
        X[
            test_index
        ]
    )[
        :,
        1
    ]


    if not np.isfinite(
        probability
    ).all():

        raise RuntimeError(
            f"Fold {fold_id}: non-finite "
            "domain probabilities."
        )


    predicted = (
        probability
        >=
        PREDICTION_THRESHOLD
    ).astype(
        np.int8
    )


    fold_auc = float(
        roc_auc_score(
            y_test,
            probability,
        )
    )


    fold_balanced_accuracy = float(
        balanced_accuracy_score(
            y_test,
            predicted,
        )
    )


    oof_probability[
        test_index
    ] = probability


    fold_assignment[
        test_index
    ] = fold_id


    cv_rows.append(
        {
            "evaluation_scope":
                f"FOLD_{fold_id}",

            "fold_id":
                fold_id,

            "train_rows":
                int(
                    len(
                        train_index
                    )
                ),

            "test_rows":
                int(
                    len(
                        test_index
                    )
                ),

            "test_cycle24_rows":
                int(
                    (
                        y_test
                        ==
                        0
                    ).sum()
                ),

            "test_cycle25_rows":
                int(
                    (
                        y_test
                        ==
                        1
                    ).sum()
                ),

            "train_groups":
                int(
                    len(
                        train_groups
                    )
                ),

            "test_groups":
                int(
                    len(
                        test_groups
                    )
                ),

            "roc_auc":
                fold_auc,

            "balanced_accuracy":
                fold_balanced_accuracy,

            "threshold":
                PREDICTION_THRESHOLD,

            "group_overlap":
                0,
        }
    )


    print(
        f"Fold {fold_id}: "
        f"test_rows={len(test_index)}, "
        f"AUC={fold_auc:.6f}, "
        f"balanced_accuracy="
        f"{fold_balanced_accuracy:.6f}"
    )


# ============================================================
# OOF INTEGRITY
# ============================================================

if np.isnan(
    oof_probability
).any():

    raise RuntimeError(
        "Some rows have no OOF "
        "domain probability."
    )


if (
    fold_assignment
    <=
    0
).any():

    raise RuntimeError(
        "Some rows have no CV fold assignment."
    )


if (
    oof_probability
    <=
    0.0
).any() or (
    oof_probability
    >=
    1.0
).any():

    raise RuntimeError(
        "Domain classifier produced "
        "probabilities at numerical "
        "boundary 0 or 1."
    )


oof_predicted = (
    oof_probability
    >=
    PREDICTION_THRESHOLD
).astype(
    np.int8
)


oof_auc = float(
    roc_auc_score(
        domain_y,
        oof_probability,
    )
)


oof_balanced_accuracy = float(
    balanced_accuracy_score(
        domain_y,
        oof_predicted,
    )
)


cv_rows.append(
    {
        "evaluation_scope":
            "ALL_OOF",

        "fold_id":
            0,

        "train_rows":
            0,

        "test_rows":
            int(
                len(
                    domain_y
                )
            ),

        "test_cycle24_rows":
            int(
                (
                    domain_y
                    ==
                    0
                ).sum()
            ),

        "test_cycle25_rows":
            int(
                (
                    domain_y
                    ==
                    1
                ).sum()
            ),

        "train_groups":
            0,

        "test_groups":
            int(
                len(
                    np.unique(
                        domain_groups
                    )
                )
            ),

        "roc_auc":
            oof_auc,

        "balanced_accuracy":
            oof_balanced_accuracy,

        "threshold":
            PREDICTION_THRESHOLD,

        "group_overlap":
            0,
    }
)


cv_df = pd.DataFrame(
    cv_rows
)


print(
    "\nOOF ROC-AUC:",
    f"{oof_auc:.6f}"
)

print(
    "OOF balanced accuracy:",
    f"{oof_balanced_accuracy:.6f}"
)

print(
    "OOF coverage:",
    "PASS"
)


# ============================================================
# BUILD OOF MANIFEST
# ============================================================

c24_manifest = pd.DataFrame(
    {
        "domain_name":
            "C24_HOLDOUT",

        "domain_label":
            np.zeros(
                len(c24),
                dtype=np.int8,
            ),

        "scope_id":
            "C24_HOLDOUT",

        "source_row_number":
            np.arange(
                len(c24),
                dtype=np.int64,
            ),

        ENTITY_GROUP_COLUMN:
            c24[
                ENTITY_GROUP_COLUMN
            ].astype(
                str
            ).to_numpy(),

        TARGET_COLUMN:
            c24[
                TARGET_COLUMN
            ].astype(
                np.int8
            ).to_numpy(),

        TIME_COLUMN:
            pd.to_datetime(
                c24[
                    TIME_COLUMN
                ],
                utc=True,
                errors="raise",
            ).to_numpy(),

        "fold_id":
            fold_assignment[
                :len(c24)
            ],

        "domain_probability_cycle25":
            oof_probability[
                :len(c24)
            ],
    }
)


c25_manifest = pd.DataFrame(
    {
        "domain_name":
            "CYCLE25",

        "domain_label":
            np.ones(
                len(c25),
                dtype=np.int8,
            ),

        "scope_id":
            c25_block.astype(
                str
            ).to_numpy(),

        "source_row_number":
            np.arange(
                len(c25),
                dtype=np.int64,
            ),

        ENTITY_GROUP_COLUMN:
            c25[
                ENTITY_GROUP_COLUMN
            ].astype(
                str
            ).to_numpy(),

        TARGET_COLUMN:
            c25[
                TARGET_COLUMN
            ].astype(
                np.int8
            ).to_numpy(),

        TIME_COLUMN:
            pd.to_datetime(
                c25[
                    TIME_COLUMN
                ],
                utc=True,
                errors="raise",
            ).to_numpy(),

        "fold_id":
            fold_assignment[
                len(c24):
            ],

        "domain_probability_cycle25":
            oof_probability[
                len(c24):
            ],
    }
)


oof_df = pd.concat(
    [
        c24_manifest,
        c25_manifest,
    ],
    ignore_index=True,
)


if len(
    oof_df
) != (
    EXPECTED_C24_ROWS
    +
    EXPECTED_C25_ROWS
):

    raise RuntimeError(
        "OOF manifest row-count mismatch."
    )


# ============================================================
# COMMON SUPPORT INTERVAL
# ============================================================

p24 = oof_df.loc[
    oof_df[
        "domain_label"
    ]
    ==
    0,
    "domain_probability_cycle25",
].to_numpy(
    dtype=np.float64
)


p25 = oof_df.loc[
    oof_df[
        "domain_label"
    ]
    ==
    1,
    "domain_probability_cycle25",
].to_numpy(
    dtype=np.float64
)


c24_q01 = quantile(
    p24,
    SUPPORT_Q_LOW,
)

c24_q99 = quantile(
    p24,
    SUPPORT_Q_HIGH,
)

c25_q01 = quantile(
    p25,
    SUPPORT_Q_LOW,
)

c25_q99 = quantile(
    p25,
    SUPPORT_Q_HIGH,
)


support_lower = max(
    c24_q01,
    c25_q01,
)

support_upper = min(
    c24_q99,
    c25_q99,
)


if support_lower >= support_upper:

    raise RuntimeError(
        "No non-empty common-support "
        "interval between Cycle24 and Cycle25."
    )


oof_df[
    "in_common_support"
] = (
    (
        oof_df[
            "domain_probability_cycle25"
        ]
        >=
        support_lower
    )
    &
    (
        oof_df[
            "domain_probability_cycle25"
        ]
        <=
        support_upper
    )
)


c24_support_count = int(
    (
        (
            oof_df[
                "domain_label"
            ]
            ==
            0
        )
        &
        oof_df[
            "in_common_support"
        ]
    ).sum()
)


c25_support_count = int(
    (
        (
            oof_df[
                "domain_label"
            ]
            ==
            1
        )
        &
        oof_df[
            "in_common_support"
        ]
    ).sum()
)


if c24_support_count == 0:

    raise RuntimeError(
        "No Cycle24 rows retained "
        "inside common support."
    )


if c25_support_count == 0:

    raise RuntimeError(
        "No Cycle25 rows retained "
        "inside common support."
    )


print(
    "\n===================================="
)

print(
    "COMMON SUPPORT"
)

print(
    "===================================="
)


print(
    "Cycle24 score 1st–99th percentile:",
    f"[{c24_q01:.6f}, {c24_q99:.6f}]"
)

print(
    "Cycle25 score 1st–99th percentile:",
    f"[{c25_q01:.6f}, {c25_q99:.6f}]"
)

print(
    "Common-support interval:",
    f"[{support_lower:.6f}, {support_upper:.6f}]"
)


# ============================================================
# C25 -> C24 REWEIGHTING
# ============================================================

oof_df[
    "raw_c25_to_c24_weight"
] = np.nan

oof_df[
    "clipped_c25_to_c24_weight"
] = np.nan

oof_df[
    "analysis_weight"
] = 0.0


# Reference rows retained in support receive weight 1.
reference_support_mask = (
    (
        oof_df[
            "domain_label"
        ]
        ==
        0
    )
    &
    oof_df[
        "in_common_support"
    ]
)


oof_df.loc[
    reference_support_mask,
    "analysis_weight"
] = 1.0


c25_support_mask = (
    (
        oof_df[
            "domain_label"
        ]
        ==
        1
    )
    &
    oof_df[
        "in_common_support"
    ]
)


supported_probability = oof_df.loc[
    c25_support_mask,
    "domain_probability_cycle25",
].to_numpy(
    dtype=np.float64
)


n24 = float(
    EXPECTED_C24_ROWS
)

n25 = float(
    EXPECTED_C25_ROWS
)


raw_weights = (
    (
        1.0
        -
        supported_probability
    )
    /
    supported_probability
) * (
    n25
    /
    n24
)


if not np.isfinite(
    raw_weights
).all():

    raise RuntimeError(
        "Non-finite C25 raw "
        "common-support weights."
    )


if (
    raw_weights
    <=
    0.0
).any():

    raise RuntimeError(
        "Non-positive C25 raw weights."
    )


weight_clip_lower = quantile(
    raw_weights,
    WEIGHT_Q_LOW,
)

weight_clip_upper = quantile(
    raw_weights,
    WEIGHT_Q_HIGH,
)


clipped_weights = np.clip(
    raw_weights,
    weight_clip_lower,
    weight_clip_upper,
)


clipped_mean = float(
    clipped_weights.mean()
)


if clipped_mean <= 0.0:

    raise RuntimeError(
        "Invalid clipped-weight mean."
    )


normalized_weights = (
    clipped_weights
    /
    clipped_mean
)


if not np.isfinite(
    normalized_weights
).all():

    raise RuntimeError(
        "Non-finite normalized weights."
    )


if not np.isclose(
    normalized_weights.mean(),
    1.0,
    rtol=0.0,
    atol=1e-12,
):

    raise RuntimeError(
        "Normalized C25 weights "
        "do not have mean 1."
    )


oof_df.loc[
    c25_support_mask,
    "raw_c25_to_c24_weight"
] = raw_weights


oof_df.loc[
    c25_support_mask,
    "clipped_c25_to_c24_weight"
] = clipped_weights


oof_df.loc[
    c25_support_mask,
    "analysis_weight"
] = normalized_weights


print(
    "C24 supported rows:",
    c24_support_count
)

print(
    "C25 supported rows:",
    c25_support_count
)

print(
    "C25 weight clip interval:",
    f"[{weight_clip_lower:.6f}, "
    f"{weight_clip_upper:.6f}]"
)

print(
    "C25 normalized weight mean:",
    f"{normalized_weights.mean():.6f}"
)

print(
    "C25 supported-row ESS:",
    f"{effective_sample_size(normalized_weights):.2f}"
)


# ============================================================
# SUPPORT SUMMARY
# ============================================================

summary_rows = []


for scope_id in SUMMARY_SCOPE_ORDER:

    if scope_id == "C24_HOLDOUT":

        subset = oof_df[
            oof_df[
                "domain_label"
            ]
            ==
            0
        ].copy()

    elif scope_id == POOLED_SCOPE:

        subset = oof_df[
            oof_df[
                "domain_label"
            ]
            ==
            1
        ].copy()

    else:

        subset = oof_df[
            (
                oof_df[
                    "domain_label"
                ]
                ==
                1
            )
            &
            (
                oof_df[
                    "scope_id"
                ]
                ==
                scope_id
            )
        ].copy()


    if len(
        subset
    ) == 0:

        raise RuntimeError(
            f"{scope_id}: empty "
            "support-summary scope."
        )


    probability = subset[
        "domain_probability_cycle25"
    ].to_numpy(
        dtype=np.float64
    )


    supported = subset[
        "in_common_support"
    ].to_numpy(
        dtype=bool
    )


    supported_subset = subset.loc[
        supported
    ].copy()


    supported_rows = int(
        supported.sum()
    )


    total_rows = int(
        len(
            subset
        )
    )


    supported_groups = int(
        supported_subset[
            ENTITY_GROUP_COLUMN
        ].nunique()
    )


    total_groups = int(
        subset[
            ENTITY_GROUP_COLUMN
        ].nunique()
    )


    supported_fraction = float(
        supported_rows
        /
        total_rows
    )


    analysis_weights = supported_subset[
        "analysis_weight"
    ].to_numpy(
        dtype=np.float64
    )


    if supported_rows > 0:

        analysis_weight_mean = float(
            analysis_weights.mean()
        )

        analysis_weight_median = float(
            np.median(
                analysis_weights
            )
        )

        analysis_weight_q01 = quantile(
            analysis_weights,
            0.01,
        )

        analysis_weight_q99 = quantile(
            analysis_weights,
            0.99,
        )

        ess = effective_sample_size(
            analysis_weights
        )

    else:

        analysis_weight_mean = np.nan
        analysis_weight_median = np.nan
        analysis_weight_q01 = np.nan
        analysis_weight_q99 = np.nan
        ess = 0.0


    summary_rows.append(
        {
            "scope_id":
                scope_id,

            "rows":
                total_rows,

            "entity_groups":
                total_groups,

            "domain_score_q01":
                quantile(
                    probability,
                    0.01,
                ),

            "domain_score_q25":
                quantile(
                    probability,
                    0.25,
                ),

            "domain_score_median":
                quantile(
                    probability,
                    0.50,
                ),

            "domain_score_q75":
                quantile(
                    probability,
                    0.75,
                ),

            "domain_score_q99":
                quantile(
                    probability,
                    0.99,
                ),

            "common_support_lower":
                support_lower,

            "common_support_upper":
                support_upper,

            "supported_rows":
                supported_rows,

            "supported_fraction":
                supported_fraction,

            "supported_entity_groups":
                supported_groups,

            "analysis_weight_mean_supported":
                analysis_weight_mean,

            "analysis_weight_median_supported":
                analysis_weight_median,

            "analysis_weight_q01_supported":
                analysis_weight_q01,

            "analysis_weight_q99_supported":
                analysis_weight_q99,

            "effective_sample_size_supported":
                ess,
        }
    )


    print(
        f"{scope_id}: "
        f"support={supported_rows}/"
        f"{total_rows} "
        f"({supported_fraction:.4f}), "
        f"groups={supported_groups}/"
        f"{total_groups}"
    )


support_summary_df = pd.DataFrame(
    summary_rows
)


if len(
    support_summary_df
) != 5:

    raise RuntimeError(
        "Expected exactly five "
        "common-support summary rows."
    )


if support_summary_df[
    "scope_id"
].tolist() != SUMMARY_SCOPE_ORDER:

    raise RuntimeError(
        "Common-support summary "
        "scope order mismatch."
    )


# ============================================================
# STRUCTURAL VERIFICATION
# ============================================================

if len(
    cv_df
) != 6:

    raise RuntimeError(
        "Expected 5 fold rows + 1 OOF row."
    )


if cv_df[
    "evaluation_scope"
].duplicated().any():

    raise RuntimeError(
        "Duplicate CV summary row."
    )


if oof_df.duplicated(
    subset=[
        "domain_name",
        "source_row_number",
    ]
).any():

    raise RuntimeError(
        "Duplicate OOF manifest row."
    )


if not np.isfinite(
    oof_df[
        "domain_probability_cycle25"
    ].to_numpy(
        dtype=np.float64
    )
).all():

    raise RuntimeError(
        "OOF manifest contains "
        "non-finite probabilities."
    )


print(
    "\nCV rows:",
    len(
        cv_df
    )
)

print(
    "Support summary rows:",
    len(
        support_summary_df
    )
)

print(
    "OOF manifest rows:",
    len(
        oof_df
    )
)

print(
    "9.4 structural checks:",
    "PASS"
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_CV_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_SUPPORT_SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_OOF_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


cv_df.to_csv(
    OUTPUT_CV_FILE,
    index=False,
)


support_summary_df.to_csv(
    OUTPUT_SUPPORT_SUMMARY_FILE,
    index=False,
)


oof_df.to_parquet(
    OUTPUT_OOF_FILE,
    index=False,
)


# ============================================================
# HASH OUTPUTS
# ============================================================

cv_hash = sha256_file(
    OUTPUT_CV_FILE
)

support_hash = sha256_file(
    OUTPUT_SUPPORT_SUMMARY_FILE
)

oof_hash = sha256_file(
    OUTPUT_OOF_FILE
)


# ============================================================
# ROUND-TRIP VERIFICATION
# ============================================================

cv_roundtrip = pd.read_csv(
    OUTPUT_CV_FILE,
    keep_default_na=False,
)

support_roundtrip = pd.read_csv(
    OUTPUT_SUPPORT_SUMMARY_FILE,
    keep_default_na=False,
)

oof_roundtrip = pd.read_parquet(
    OUTPUT_OOF_FILE
)


if len(
    cv_roundtrip
) != 6:

    raise RuntimeError(
        "CV CSV round-trip failed."
    )


if len(
    support_roundtrip
) != 5:

    raise RuntimeError(
        "Support-summary round-trip failed."
    )


if len(
    oof_roundtrip
) != (
    EXPECTED_C24_ROWS
    +
    EXPECTED_C25_ROWS
):

    raise RuntimeError(
        "OOF parquet round-trip failed."
    )


print(
    "CV round-trip:",
    "PASS"
)

print(
    "Support-summary round-trip:",
    "PASS"
)

print(
    "OOF manifest round-trip:",
    "PASS"
)


# ============================================================
# CONTRACT
# ============================================================

fold_auc_values = cv_df.loc[
    cv_df[
        "evaluation_scope"
    ]
    !=
    "ALL_OOF",
    "roc_auc",
].to_numpy(
    dtype=np.float64
)


fold_bacc_values = cv_df.loc[
    cv_df[
        "evaluation_scope"
    ]
    !=
    "ALL_OOF",
    "balanced_accuracy",
].to_numpy(
    dtype=np.float64
)


contract = {

    "contract_name":
        (
            "Joint Multivariate Domain Shift "
            "and Common-Support Analysis"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_JOINT_SHIFT_COMMON_SUPPORT",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_9_4",

    "reference_domain":
        "CYCLE24_HOLDOUT",

    "transfer_domain":
        "CYCLE25",

    "features":
        FEATURES,

    "feature_count":
        25,

    "domain_classifier":
        {
            "type":
                "LogisticRegression",

            "penalty":
                "l2",

            "C":
                1.0,

            "class_weight":
                "balanced",

            "solver":
                "lbfgs",

            "max_iter":
                5000,

            "random_state":
                DOMAIN_SEED,

            "input_preprocessing":
                (
                    "Frozen StandardScaler "
                    "fit only on Cycle24 TRAIN."
                ),

            "scaler_fit_scope":
                "CYCLE24_TRAIN_ONLY",

            "scaler_train_rows":
                EXPECTED_TRAIN_ROWS,

            "scaler_train_sha256":
                EXPECTED_TRAIN_HASH,

            "domain_target":
                {
                    "C24_HOLDOUT":
                        0,

                    "CYCLE25":
                        1,
                },

            "probability_interpretation":
                (
                    "Out-of-fold discriminative "
                    "Cycle25-domain score from the "
                    "frozen domain-classifier design. "
                    "It is used as the protocol-defined "
                    "propensity/common-support score."
                ),
        },

    "cross_validation":
        {
            "type":
                "StratifiedGroupKFold",

            "n_splits":
                CV_SPLITS,

            "shuffle":
                True,

            "random_state":
                DOMAIN_SEED,

            "group":
                (
                    "Cycle-prefixed "
                    "ENTITY_GROUP_ID"
                ),

            "group_leakage_detected":
                False,

            "prediction_mode":
                "OUT_OF_FOLD",
        },

    "joint_shift_metrics":
        {
            "oof_roc_auc":
                oof_auc,

            "oof_balanced_accuracy":
                oof_balanced_accuracy,

            "fold_roc_auc_mean":
                float(
                    fold_auc_values.mean()
                ),

            "fold_roc_auc_std":
                float(
                    fold_auc_values.std(
                        ddof=1
                    )
                ),

            "fold_balanced_accuracy_mean":
                float(
                    fold_bacc_values.mean()
                ),

            "fold_balanced_accuracy_std":
                float(
                    fold_bacc_values.std(
                        ddof=1
                    )
                ),

            "classification_threshold":
                PREDICTION_THRESHOLD,

            "binary_domain_shift_cutoff":
                None,
        },

    "common_support":
        {
            "score":
                "OOF_CYCLE25_DOMAIN_PROBABILITY",

            "cycle24_score_q01":
                c24_q01,

            "cycle24_score_q99":
                c24_q99,

            "cycle25_score_q01":
                c25_q01,

            "cycle25_score_q99":
                c25_q99,

            "support_interval_rule":
                (
                    "Intersection of Cycle24 "
                    "and Cycle25 OOF score "
                    "1st-to-99th percentile "
                    "intervals."
                ),

            "support_lower":
                support_lower,

            "support_upper":
                support_upper,

            "cycle24_supported_rows":
                c24_support_count,

            "cycle25_supported_rows":
                c25_support_count,
        },

    "reweighting":
        {
            "direction":
                "CYCLE25_TO_CYCLE24",

            "raw_weight_formula":
                (
                    "((1-p)/p) * "
                    "(N_C25/N_C24)"
                ),

            "applied_only_inside_common_support":
                True,

            "raw_weight_clip_quantiles":
                [
                    WEIGHT_Q_LOW,
                    WEIGHT_Q_HIGH,
                ],

            "raw_weight_clip_lower":
                weight_clip_lower,

            "raw_weight_clip_upper":
                weight_clip_upper,

            "normalization":
                (
                    "Supported Cycle25 clipped "
                    "weights normalized to mean 1."
                ),

            "normalized_weight_mean":
                float(
                    normalized_weights.mean()
                ),

            "supported_cycle25_effective_sample_size":
                effective_sample_size(
                    normalized_weights
                ),

            "primary_or_sensitivity":
                "SENSITIVITY_ONLY",
        },

    "artifacts":
        {
            "joint_domain_shift_cv":
                {
                    "file":
                        str(
                            OUTPUT_CV_FILE
                        ),

                    "sha256":
                        cv_hash,

                    "rows":
                        len(
                            cv_df
                        ),
                },

            "common_support_summary":
                {
                    "file":
                        str(
                            OUTPUT_SUPPORT_SUMMARY_FILE
                        ),

                    "sha256":
                        support_hash,

                    "rows":
                        len(
                            support_summary_df
                        ),
                },

            "domain_oof_common_support":
                {
                    "file":
                        str(
                            OUTPUT_OOF_FILE
                        ),

                    "sha256":
                        oof_hash,

                    "rows":
                        len(
                            oof_df
                        ),
                },
        },

    "data_access":
        {
            "cycle24_train_raw_data_read":
                False,

            "cycle24_holdout_read":
                True,

            "cycle25_read":
                True,

            "frozen_cycle24_train_scaler_loaded":
                True,

            "frozen_scaler_fit_scope":
                "CYCLE24_TRAIN_ONLY",

            "flare_models_loaded":
                False,

            "flare_models_retrained":
                False,

            "prediction_artifacts_read":
                False,

            "shap_computed":
                False,
        },

    "evaluation_state":
        {
            "marginal_shift":
                True,

            "prevalence_ar_population":
                True,

            "joint_domain_shift":
                True,

            "common_support":
                True,

            "common_support_weights_frozen":
                True,

            "correlated_feature_groups":
                False,

            "within_cycle24_shap_null":
                False,

            "cross_cycle_shap":
                False,

            "explanation_drift":
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

            "cycle25_temporal_blocks":
                {
                    "file":
                        str(
                            CYCLE25_BLOCKS_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CYCLE25_BLOCKS_FILE
                        ),
                },

            "cycle24_holdout":
                {
                    "file":
                        str(
                            CYCLE24_HOLDOUT_FILE
                        ),

                    "sha256":
                        c24_hash,
                },

            "cycle25_dataset":
                {
                    "file":
                        str(
                            CYCLE25_FILE
                        ),

                    "sha256":
                        c25_hash,
                },

            "frozen_standard_scaler":
                {
                    "file":
                        str(
                            SCALER_FILE
                        ),

                    "sha256":
                        scaler_hash,

                    "fit_scope":
                        "CYCLE24_TRAIN_ONLY",

                    "train_rows":
                        EXPECTED_TRAIN_ROWS,

                    "train_sha256":
                        EXPECTED_TRAIN_HASH,
                },
        },

    "interpretation_rule":
        (
            "ROC-AUC and balanced accuracy quantify "
            "how separable Cycle24-holdout and "
            "Cycle25 feature vectors are under the "
            "pre-specified multivariate domain "
            "classifier. No arbitrary AUC threshold "
            "is used to make a binary shift claim. "
            "Common-support restriction and "
            "reweighting are sensitivity controls "
            "for later SHAP analysis and do not "
            "replace the primary unweighted analysis."
        ),

    "immutability_rule":
        (
            "Observed domain-classifier performance, "
            "common-support fraction, or weights must "
            "not alter the frozen flare models, "
            "features, Cycle25 temporal blocks, "
            "correlated-feature grouping rule, "
            "SHAP background, or within-Cycle24 "
            "SHAP null-control design."
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
    "CHAPTER 9.4 SUMMARY"
)

print(
    "===================================="
)


print(
    "Domain rows:",
    len(
        oof_df
    )
)

print(
    "CV folds:",
    CV_SPLITS
)

print(
    "OOF ROC-AUC:",
    f"{oof_auc:.6f}"
)

print(
    "OOF balanced accuracy:",
    f"{oof_balanced_accuracy:.6f}"
)

print(
    "Common-support interval:",
    f"[{support_lower:.6f}, "
    f"{support_upper:.6f}]"
)

print(
    "Cycle24 supported rows:",
    c24_support_count
)

print(
    "Cycle25 supported rows:",
    c25_support_count
)

print(
    "C25 normalized-weight mean:",
    f"{normalized_weights.mean():.6f}"
)

print(
    "C25 supported ESS:",
    f"{effective_sample_size(normalized_weights):.2f}"
)

print(
    "Cycle24 TRAIN raw data read:",
    False
)

print(
    "Frozen TRAIN scaler loaded:",
    True
)

print(
    "Frozen TRAIN scaler fit scope:",
    "CYCLE24_TRAIN_ONLY"
)

print(
    "Flare models loaded/retrained:",
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
    OUTPUT_CV_FILE
)

print(
    OUTPUT_SUPPORT_SUMMARY_FILE
)

print(
    OUTPUT_OOF_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 9.4 COMPLETE — JOINT SHIFT + COMMON SUPPORT FROZEN"
)

print(
    "===================================="
)