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
# CHAPTER 9.6B — FINAL CONSOLIDATION
# ============================================================

EXPECTED_ROWS = 13072
EXPECTED_GROUPS = 176
EXPECTED_POSITIVES = 527
EXPECTED_NEGATIVES = 12545

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TARGET_COLUMN = "target"
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

SHAP_COLUMNS = [
    f"SHAP_{feature}"
    for feature in FEATURES
]


# ============================================================
# INPUT PATHS
# ============================================================

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

SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/"
    "logistic_standard_scaler.joblib"
)


MODEL_PATHS = {
    "LR_02_C0p01_BALANCED": Path(
        "artifacts/models/logistic/models/"
        "LR_02_C0p01_BALANCED.joblib"
    ),

    "RF_04_D12_L5_BALSUB": Path(
        "artifacts/models/random_forest/models/"
        "RF_04_D12_L5_BALSUB.joblib"
    ),

    "XGB_02_LR0p03_D3_BAL": Path(
        "artifacts/models/xgboost/models/"
        "XGB_02_LR0p03_D3_BAL.joblib"
    ),
}


SHAP_PATHS = {
    "LR_02_C0p01_BALANCED": Path(
        "artifacts/models/shift_controls/shap/"
        "cycle24_holdout/"
        "LR_02_C0p01_BALANCED.parquet"
    ),

    "RF_04_D12_L5_BALSUB": Path(
        "artifacts/models/shift_controls/shap/"
        "cycle24_holdout/"
        "RF_04_D12_L5_BALSUB.parquet"
    ),

    "XGB_02_LR0p03_D3_BAL": Path(
        "artifacts/models/shift_controls/shap/"
        "cycle24_holdout/"
        "XGB_02_LR0p03_D3_BAL.parquet"
    ),
}


RF_AUDIT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "rf_cycle24_holdout_shap_parallel_audit.json"
)

XGB_AUDIT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "xgb_cycle24_holdout_shap_parallel_audit.json"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

SUMMARY_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "cycle24_holdout_shap_summary.csv"
)

REGISTRY_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "cycle24_holdout_shap_registry.csv"
)

CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "cycle24_holdout_shap_contract.json"
)


# ============================================================
# EXPECTED HASHES
# ============================================================

EXPECTED_HOLDOUT_HASH = (
    "8abb3929febcde7d778c4781ac890ff5e"
    "658933c2ef2a06aa19da085e7e255fc"
)

EXPECTED_TRAIN_HASH = (
    "4801a6e3b11d39cef8d4b0ef0eea4ae1"
    "67c27862431662c3f43ff55e4c433def"
)

EXPECTED_SCALER_HASH = (
    "48a9c07301f30052f49bad2db048418d"
    "3db16951f0433002b831b59fa10a1af5"
)

EXPECTED_MODEL_HASHES = {
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


# ============================================================
# MODEL METADATA
# ============================================================

MODEL_META = {
    "LR_02_C0p01_BALANCED": {
        "family": "logistic",
        "explainer": "LinearExplainer",
        "input_space": "FROZEN_STANDARDIZED",
        "output_space": "LOG_ODDS_DECISION_FUNCTION",
    },

    "RF_04_D12_L5_BALSUB": {
        "family": "random_forest",
        "explainer": "TreeExplainer",
        "input_space": "RAW",
        "output_space": "POSITIVE_CLASS_PROBABILITY",
    },

    "XGB_02_LR0p03_D3_BAL": {
        "family": "xgboost",
        "explainer": "TreeExplainer",
        "input_space": "RAW",
        "output_space": "XGBOOST_RAW_MARGIN",
    },
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

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


def verify_exists(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


# ============================================================
# START
# ============================================================

print("=" * 72)

print(
    "MODEL CHAPTER 9.6B — "
    "FINAL CYCLE24 HOLDOUT SHAP CONSOLIDATION"
)

print("=" * 72)


# ============================================================
# INPUT EXISTENCE
# ============================================================

required_files = [
    HOLDOUT_FILE,
    BACKGROUND_FILE,
    BACKGROUND_CONTRACT_FILE,
    SCALER_FILE,
    RF_AUDIT_FILE,
    XGB_AUDIT_FILE,
    *MODEL_PATHS.values(),
    *SHAP_PATHS.values(),
]


for path in required_files:

    verify_exists(path)


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    SUMMARY_FILE,
    REGISTRY_FILE,
    CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter 9.6B output."
        )


# ============================================================
# VERIFY HOLDOUT
# ============================================================

holdout_hash = sha256_file(
    HOLDOUT_FILE
)


if holdout_hash != EXPECTED_HOLDOUT_HASH:

    raise RuntimeError(
        "Cycle24 HOLDOUT hash mismatch."
    )


holdout = pd.read_parquet(
    HOLDOUT_FILE
)


if len(holdout) != EXPECTED_ROWS:

    raise RuntimeError(
        "Cycle24 HOLDOUT row-count mismatch."
    )


if int(
    holdout[
        ENTITY_GROUP_COLUMN
    ].nunique()
) != EXPECTED_GROUPS:

    raise RuntimeError(
        "Cycle24 HOLDOUT group-count mismatch."
    )


if int(
    holdout[
        TARGET_COLUMN
    ].sum()
) != EXPECTED_POSITIVES:

    raise RuntimeError(
        "Cycle24 HOLDOUT positive-count mismatch."
    )


if int(
    (
        holdout[
            TARGET_COLUMN
        ]
        ==
        0
    ).sum()
) != EXPECTED_NEGATIVES:

    raise RuntimeError(
        "Cycle24 HOLDOUT negative-count mismatch."
    )


X_raw = holdout[
    FEATURES
].to_numpy(
    dtype=np.float64
)


if not np.isfinite(
    X_raw
).all():

    raise RuntimeError(
        "Non-finite HOLDOUT features."
    )


print(
    "Cycle24 HOLDOUT integrity:",
    "PASS"
)


# ============================================================
# VERIFY BACKGROUND CONTRACT
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
        "SHAP background contract "
        "is not frozen."
    )


background_hash = sha256_file(
    BACKGROUND_FILE
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
        "SHAP background hash mismatch."
    )


background = pd.read_parquet(
    BACKGROUND_FILE
)


if len(background) != 512:

    raise RuntimeError(
        "Expected 512-row SHAP background."
    )


print(
    "Frozen 512-row SHAP background:",
    "PASS"
)


# ============================================================
# VERIFY SCALER
# ============================================================

scaler_hash = sha256_file(
    SCALER_FILE
)


if scaler_hash != EXPECTED_SCALER_HASH:

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
        "Scaler fit scope mismatch."
    )


if (
    scaler_bundle[
        "train_sha256"
    ]
    !=
    EXPECTED_TRAIN_HASH
):

    raise RuntimeError(
        "Scaler TRAIN provenance mismatch."
    )


if (
    scaler_bundle[
        "feature_order"
    ]
    !=
    FEATURES
):

    raise RuntimeError(
        "Scaler feature order mismatch."
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
        "Non-finite scaled features."
    )


print(
    "Frozen LR preprocessing:",
    "PASS"
)


# ============================================================
# VERIFY COMPONENT AUDITS
# ============================================================

rf_audit = load_json(
    RF_AUDIT_FILE
)

xgb_audit = load_json(
    XGB_AUDIT_FILE
)


if not rf_audit[
    "additivity"
][
    "native_style_pass"
]:

    raise RuntimeError(
        "RF component audit did not pass."
    )


if not xgb_audit[
    "additivity"
][
    "native_style_pass"
]:

    raise RuntimeError(
        "XGB component audit did not pass."
    )


if not xgb_audit[
    "wrapper_booster_equivalence"
][
    "equivalent_at_1e_12"
]:

    raise RuntimeError(
        "XGB wrapper/Booster "
        "equivalence audit failed."
    )


print(
    "RF component audit:",
    "PASS"
)

print(
    "XGB component audit:",
    "PASS"
)


# ============================================================
# RESULT CONTAINERS
# ============================================================

summary_rows = []

registry_rows = []

model_contract_rows = []


# ============================================================
# PROCESS THREE MODELS
# ============================================================

for candidate_id in [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]:

    print()

    print("-" * 72)

    print(
        "VERIFY:",
        candidate_id
    )

    print("-" * 72)


    meta = MODEL_META[
        candidate_id
    ]


    family = meta[
        "family"
    ]


    model_path = MODEL_PATHS[
        candidate_id
    ]


    shap_path = SHAP_PATHS[
        candidate_id
    ]


    # ========================================================
    # MODEL HASH
    # ========================================================

    model_hash = sha256_file(
        model_path
    )


    if (
        model_hash
        !=
        EXPECTED_MODEL_HASHES[
            candidate_id
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: model hash mismatch."
        )


    bundle = joblib.load(
        model_path
    )


    if (
        bundle[
            "candidate_id"
        ]
        !=
        candidate_id
    ):

        raise RuntimeError(
            f"{candidate_id}: bundle "
            "candidate mismatch."
        )


    if (
        bundle[
            "training_scope"
        ]
        !=
        "CYCLE24_TRAIN_ONLY"
    ):

        raise RuntimeError(
            f"{candidate_id}: training "
            "scope mismatch."
        )


    if (
        bundle[
            "train_sha256"
        ]
        !=
        EXPECTED_TRAIN_HASH
    ):

        raise RuntimeError(
            f"{candidate_id}: TRAIN "
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


    model = bundle[
        "model"
    ]


    print(
        "Model integrity:",
        "PASS"
    )


    # ========================================================
    # LOAD SHAP ARTIFACT
    # ========================================================

    shap_df = pd.read_parquet(
        shap_path
    )


    if len(
        shap_df
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            f"{candidate_id}: SHAP "
            "row-count mismatch."
        )


    required_columns = {
        "source_row_number",
        "candidate_id",
        "model_family",
        ENTITY_GROUP_COLUMN,
        TARGET_COLUMN,
        "shap_base_value",
        *SHAP_COLUMNS,
    }


    missing_columns = (
        required_columns
        -
        set(
            shap_df.columns
        )
    )


    if missing_columns:

        raise RuntimeError(
            f"{candidate_id}: missing "
            f"SHAP columns "
            f"{sorted(missing_columns)}"
        )


    # ========================================================
    # ROW ORDER
    # ========================================================

    expected_row_numbers = np.arange(
        EXPECTED_ROWS,
        dtype=np.int64,
    )


    actual_row_numbers = shap_df[
        "source_row_number"
    ].to_numpy(
        dtype=np.int64
    )


    if not np.array_equal(
        expected_row_numbers,
        actual_row_numbers,
    ):

        raise RuntimeError(
            f"{candidate_id}: source row "
            "order mismatch."
        )


    # ========================================================
    # CANDIDATE / FAMILY
    # ========================================================

    if not (
        shap_df[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ).all():

        raise RuntimeError(
            f"{candidate_id}: SHAP candidate "
            "column mismatch."
        )


    if not (
        shap_df[
            "model_family"
        ].astype(str)
        ==
        family
    ).all():

        raise RuntimeError(
            f"{candidate_id}: model-family "
            "column mismatch."
        )


    # ========================================================
    # TARGET ALIGNMENT
    # ========================================================

    shap_target = shap_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.int8
    )


    holdout_target = holdout[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.int8
    )


    if not np.array_equal(
        shap_target,
        holdout_target,
    ):

        raise RuntimeError(
            f"{candidate_id}: target "
            "alignment mismatch."
        )


    # ========================================================
    # ENTITY-GROUP ALIGNMENT
    # ========================================================

    shap_groups = shap_df[
        ENTITY_GROUP_COLUMN
    ].astype(
        str
    ).to_numpy()


    holdout_groups = holdout[
        ENTITY_GROUP_COLUMN
    ].astype(
        str
    ).to_numpy()


    if not np.array_equal(
        shap_groups,
        holdout_groups,
    ):

        raise RuntimeError(
            f"{candidate_id}: ENTITY_GROUP "
            "alignment mismatch."
        )


    # ========================================================
    # TIME ALIGNMENT IF PRESENT
    # ========================================================

    if (
        TIME_COLUMN in shap_df.columns
        and
        TIME_COLUMN in holdout.columns
    ):

        shap_time = pd.to_datetime(
            shap_df[
                TIME_COLUMN
            ],
            utc=True,
        )


        holdout_time = pd.to_datetime(
            holdout[
                TIME_COLUMN
            ],
            utc=True,
        )


        if not np.array_equal(
            shap_time.to_numpy(),
            holdout_time.to_numpy(),
        ):

            raise RuntimeError(
                f"{candidate_id}: observation-time "
                "alignment mismatch."
            )


    # ========================================================
    # SHAP NUMERIC INTEGRITY
    # ========================================================

    shap_matrix = shap_df[
        SHAP_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )


    base_vector = shap_df[
        "shap_base_value"
    ].to_numpy(
        dtype=np.float64
    )


    if shap_matrix.shape != (
        EXPECTED_ROWS,
        25,
    ):

        raise RuntimeError(
            f"{candidate_id}: SHAP "
            "matrix shape mismatch."
        )


    if base_vector.shape != (
        EXPECTED_ROWS,
    ):

        raise RuntimeError(
            f"{candidate_id}: SHAP "
            "base-vector mismatch."
        )


    if not np.isfinite(
        shap_matrix
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "SHAP values."
        )


    if not np.isfinite(
        base_vector
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "SHAP base values."
        )


    print(
        "SHAP artifact integrity:",
        "PASS"
    )


    # ========================================================
    # MODEL OUTPUT
    # ========================================================

    if family == "logistic":

        target_output = np.asarray(
            model.decision_function(
                X_scaled
            ),
            dtype=np.float64,
        )

        additivity_atol = 1e-8
        additivity_rtol = 1e-8


    elif family == "random_forest":

        target_output = np.asarray(
            model.predict_proba(
                X_raw
            )[
                :,
                1
            ],
            dtype=np.float64,
        )

        additivity_atol = 1e-2
        additivity_rtol = 1e-2


    elif family == "xgboost":

        booster = model.get_booster()

        target_output = np.asarray(
            booster.inplace_predict(
                X_raw,
                predict_type="margin",
            ),
            dtype=np.float64,
        ).reshape(-1)

        additivity_atol = 1e-2
        additivity_rtol = 1e-2


    else:

        raise RuntimeError(
            f"Unknown family: {family}"
        )


    # ========================================================
    # FINAL ADDITIVITY AUDIT
    # ========================================================

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


    additivity_pass = bool(
        np.allclose(
            reconstructed_output,
            target_output,
            atol=additivity_atol,
            rtol=additivity_rtol,
        )
    )


    if not additivity_pass:

        raise RuntimeError(
            f"{candidate_id}: final "
            "additivity audit failed. "
            f"max_abs_error="
            f"{float(abs_error.max())}"
        )


    relative_error = (
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
        "Final additivity:",
        "PASS"
    )

    print(
        "Mean absolute error:",
        f"{float(abs_error.mean()):.10f}"
    )

    print(
        "Max absolute error:",
        f"{float(abs_error.max()):.10f}"
    )


    # ========================================================
    # SHAP HASH
    # ========================================================

    shap_hash = sha256_file(
        shap_path
    )


    # ========================================================
    # SUMMARY — OVERALL / TARGET_0 / TARGET_1
    # ========================================================

    strata = {
        "OVERALL":
            np.ones(
                EXPECTED_ROWS,
                dtype=bool,
            ),

        "TARGET_0":
            (
                holdout_target
                ==
                0
            ),

        "TARGET_1":
            (
                holdout_target
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
                        candidate_id,

                    "model_family":
                        family,

                    "stratum":
                        stratum,

                    "rows":
                        int(
                            mask.sum()
                        ),

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


    # ========================================================
    # REGISTRY ROW
    # ========================================================

    registry_rows.append(
        {
            "candidate_id":
                candidate_id,

            "model_family":
                family,

            "model_file":
                str(
                    model_path
                ),

            "model_sha256":
                model_hash,

            "shap_file":
                str(
                    shap_path
                ),

            "shap_sha256":
                shap_hash,

            "rows":
                EXPECTED_ROWS,

            "features":
                25,

            "entity_groups":
                EXPECTED_GROUPS,

            "explainer":
                meta[
                    "explainer"
                ],

            "input_space":
                meta[
                    "input_space"
                ],

            "output_space":
                meta[
                    "output_space"
                ],

            "background_rows":
                512,

            "additivity_pass":
                additivity_pass,

            "additivity_atol":
                additivity_atol,

            "additivity_rtol":
                additivity_rtol,

            "additivity_mean_abs_error":
                float(
                    abs_error.mean()
                ),

            "additivity_p95_abs_error":
                float(
                    np.quantile(
                        abs_error,
                        0.95,
                    )
                ),

            "additivity_p99_abs_error":
                float(
                    np.quantile(
                        abs_error,
                        0.99,
                    )
                ),

            "additivity_max_abs_error":
                float(
                    abs_error.max()
                ),

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
        }
    )


    # ========================================================
    # CONTRACT MODEL ROW
    # ========================================================

    model_contract_rows.append(
        {
            "candidate_id":
                candidate_id,

            "model_family":
                family,

            "model_sha256":
                model_hash,

            "training_scope":
                "CYCLE24_TRAIN_ONLY",

            "train_sha256":
                EXPECTED_TRAIN_HASH,

            "explainer":
                meta[
                    "explainer"
                ],

            "input_space":
                meta[
                    "input_space"
                ],

            "output_space":
                meta[
                    "output_space"
                ],

            "background_rows":
                512,

            "shap_artifact":
                {
                    "file":
                        str(
                            shap_path
                        ),

                    "sha256":
                        shap_hash,

                    "rows":
                        EXPECTED_ROWS,

                    "features":
                        25,
                },

            "additivity":
                {
                    "pass":
                        additivity_pass,

                    "atol":
                        additivity_atol,

                    "rtol":
                        additivity_rtol,

                    "mean_abs_error":
                        float(
                            abs_error.mean()
                        ),

                    "p95_abs_error":
                        float(
                            np.quantile(
                                abs_error,
                                0.95,
                            )
                        ),

                    "p99_abs_error":
                        float(
                            np.quantile(
                                abs_error,
                                0.99,
                            )
                        ),

                    "max_abs_error":
                        float(
                            abs_error.max()
                        ),

                    "relative_error_diagnostic_only":
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
        }
    )


    print(
        f"{candidate_id}: VERIFIED ✅"
    )


# ============================================================
# BUILD FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


if len(
    summary_df
) != 225:

    raise RuntimeError(
        "Expected exactly 225 "
        "combined SHAP summary rows."
    )


if summary_df.duplicated(
    subset=[
        "candidate_id",
        "stratum",
        "feature",
    ]
).any():

    raise RuntimeError(
        "Duplicate combined summary rows."
    )


# ============================================================
# BUILD FINAL REGISTRY
# ============================================================

registry_df = pd.DataFrame(
    registry_rows
)


if len(
    registry_df
) != 3:

    raise RuntimeError(
        "Expected exactly 3 "
        "registry rows."
    )


if not registry_df[
    "additivity_pass"
].all():

    raise RuntimeError(
        "At least one model failed "
        "final additivity validation."
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REGISTRY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAVE SUMMARY + REGISTRY
# ============================================================

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


registry_df.to_csv(
    REGISTRY_FILE,
    index=False,
)


# ============================================================
# ROUND-TRIP
# ============================================================

summary_rt = pd.read_csv(
    SUMMARY_FILE,
    keep_default_na=False,
)


registry_rt = pd.read_csv(
    REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    summary_rt
) != 225:

    raise RuntimeError(
        "Summary round-trip failed."
    )


if len(
    registry_rt
) != 3:

    raise RuntimeError(
        "Registry round-trip failed."
    )


summary_hash = sha256_file(
    SUMMARY_FILE
)

registry_hash = sha256_file(
    REGISTRY_FILE
)


print()

print(
    "Combined SHAP summary rows:",
    len(summary_df)
)

print(
    "Combined model registry rows:",
    len(registry_df)
)

print(
    "Combined summary round-trip:",
    "PASS"
)

print(
    "Combined registry round-trip:",
    "PASS"
)


# ============================================================
# FINAL 9.6B CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Cycle24 HOLDOUT "
            "SHAP Attribution Set"
        ),

    "version":
        "2.0",

    "status":
        "FROZEN_CYCLE24_HOLDOUT_SHAP",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_9_6B",

    "software":
        {
            "shap":
                shap.__version__,

            "scikit_learn":
                sklearn.__version__,

            "xgboost":
                xgboost.__version__,
        },

    "evaluation_population":
        {
            "name":
                "CYCLE24_HOLDOUT",

            "file":
                str(
                    HOLDOUT_FILE
                ),

            "sha256":
                holdout_hash,

            "rows":
                EXPECTED_ROWS,

            "entity_groups":
                EXPECTED_GROUPS,

            "positive_rows":
                EXPECTED_POSITIVES,

            "negative_rows":
                EXPECTED_NEGATIVES,
        },

    "feature_contract":
        {
            "count":
                25,

            "order":
                FEATURES,
        },

    "background":
        {
            "source":
                "CYCLE24_TRAIN_ONLY",

            "file":
                str(
                    BACKGROUND_FILE
                ),

            "sha256":
                background_hash,

            "rows":
                512,

            "frozen":
                True,
        },

    "models":
        model_contract_rows,

    "execution_notes":
        {
            "lr":
                (
                    "LinearExplainer using frozen "
                    "Cycle24-TRAIN StandardScaler."
                ),

            "rf":
                (
                    "TreeExplainer interventional "
                    "using frozen 512-row background. "
                    "RF was executed in checkpointed "
                    "parallel chunks. Native-style "
                    "additivity validation passed."
                ),

            "xgb":
                (
                    "TreeExplainer interventional "
                    "using the exact fitted Booster "
                    "extracted from the frozen "
                    "XGBClassifier bundle because "
                    "SHAP 0.52 rejected the wrapper "
                    "when enable_categorical=True. "
                    "Wrapper and Booster raw margins "
                    "were verified equivalent before "
                    "SHAP generation."
                ),
        },

    "scientific_guardrails":
        {
            "cycle24_train_raw_read_for_evaluation":
                False,

            "cycle24_holdout_read":
                True,

            "cycle25_read":
                False,

            "models_retrained":
                False,

            "models_retuned":
                False,

            "feature_selection_changed":
                False,

            "background_changed":
                False,

            "sampling_reduction":
                False,

            "shap_approximation":
                False,

            "cross_model_raw_shap_magnitude_comparison_allowed":
                False,
        },

    "comparison_rule":
        (
            "SHAP attribution structure is compared "
            "within the same frozen model across "
            "resampled populations or cycles. "
            "Raw SHAP magnitudes are not directly "
            "compared between Logistic Regression, "
            "Random Forest and XGBoost because their "
            "explanation output spaces differ."
        ),

    "artifacts":
        {
            "combined_summary":
                {
                    "file":
                        str(
                            SUMMARY_FILE
                        ),

                    "sha256":
                        summary_hash,

                    "rows":
                        225,
                },

            "combined_registry":
                {
                    "file":
                        str(
                            REGISTRY_FILE
                        ),

                    "sha256":
                        registry_hash,

                    "rows":
                        3,
                },

            "rf_component_audit":
                {
                    "file":
                        str(
                            RF_AUDIT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            RF_AUDIT_FILE
                        ),
                },

            "xgb_component_audit":
                {
                    "file":
                        str(
                            XGB_AUDIT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            XGB_AUDIT_FILE
                        ),
                },
        },

    "evaluation_state":
        {
            "cycle24_holdout_shap":
                True,

            "within_cycle24_shap_null":
                False,

            "cycle25_shap":
                False,

            "cross_cycle_shap_drift":
                False,
        },

    "next_stage":
        "MODEL_CHAPTER_9_6C",

    "immutability_rule":
        (
            "After Chapter 9.6B is frozen, "
            "the Cycle24 HOLDOUT SHAP values, "
            "model-specific explanation spaces, "
            "feature order, frozen background, "
            "and explainer configurations must "
            "not be changed in response to "
            "within-Cycle24 null or Cycle25 results."
        ),
}


with CONTRACT_FILE.open(
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
    CONTRACT_FILE
)


if (
    contract_rt[
        "status"
    ]
    !=
    "FROZEN_CYCLE24_HOLDOUT_SHAP"
):

    raise RuntimeError(
        "Final contract round-trip failed."
    )


if len(
    contract_rt[
        "models"
    ]
) != 3:

    raise RuntimeError(
        "Final contract model-count mismatch."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 72)

print(
    "MODEL CHAPTER 9.6B — COMPLETE"
)

print("=" * 72)

print(
    "LR SHAP:",
    "PASS"
)

print(
    "RF SHAP:",
    "PASS"
)

print(
    "XGB SHAP:",
    "PASS"
)

print(
    "Models:",
    3
)

print(
    "Rows/model:",
    EXPECTED_ROWS
)

print(
    "Features/model:",
    25
)

print(
    "Background rows:",
    512
)

print(
    "Combined summary rows:",
    225
)

print(
    "Cycle25 read:",
    False
)

print(
    "Models retrained:",
    False
)

print(
    "Status:",
    "FROZEN_CYCLE24_HOLDOUT_SHAP"
)

print()

print(
    "Saved:",
    SUMMARY_FILE
)

print(
    "Saved:",
    REGISTRY_FILE
)

print(
    "Saved:",
    CONTRACT_FILE
)

print()

print(
    "NEXT:",
    "Chapter 9.6C — "
    "2000 A-vs-B ENTITY_GROUP SHAP null"
)