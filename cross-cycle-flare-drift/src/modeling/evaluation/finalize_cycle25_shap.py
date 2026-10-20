# ============================================================
# MODEL CHAPTER 10.2.4
# FINALIZE + FREEZE ALL CYCLE25 SHAP ARTIFACTS
#
# Models:
#   LR_02_C0p01_BALANCED
#   RF_04_D12_L5_BALSUB
#   XGB_02_LR0p03_D3_BAL
#
# Purpose:
#   - verify all three completed Cycle25 SHAP artifacts
#   - verify exact same Cycle25 population / row alignment
#   - verify model-specific explanation spaces
#   - build unified pooled summary
#   - build frozen registry
#   - freeze Cycle25 SHAP before any cross-cycle drift analysis
#
# DOES NOT:
#   - generate SHAP
#   - retrain models
#   - refit preprocessing
#   - change background
#   - change Cycle24 null baseline
#   - calculate explanation drift
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_ROWS = 79361

EXPECTED_FEATURES = 25

EXPECTED_MODELS = 3

EXPECTED_SUMMARY_ROWS = (
    EXPECTED_MODELS
    *
    3
    *
    EXPECTED_FEATURES
)


TARGET_COLUMN = "target"

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


SHAP_COLUMNS = [
    f"SHAP_{feature}"
    for feature in FEATURES
]


# ============================================================
# FROZEN CHAPTER 10.1 INPUTS
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


# ============================================================
# MODEL DEFINITIONS
# ============================================================

MODEL_SPECS = {

    "LR_02_C0p01_BALANCED": {

        "family":
            "logistic",

        "shap_file":
            Path(
                "artifacts/models/cross_cycle_shap/"
                "shap/cycle25/"
                "LR_02_C0p01_BALANCED.parquet"
            ),

        "audit_file":
            Path(
                "artifacts/models/cross_cycle_shap/"
                "contracts/"
                "lr_cycle25_shap_audit.json"
            ),

        "required_audit_status":
            "COMPLETE_LR_CYCLE25_SHAP",

        "explainer":
            "LinearExplainer",

        "input_space":
            "FROZEN_STANDARDIZED",

        "output_space":
            "LOG_ODDS_DECISION_FUNCTION",
    },


    "RF_04_D12_L5_BALSUB": {

        "family":
            "random_forest",

        "shap_file":
            Path(
                "artifacts/models/cross_cycle_shap/"
                "shap/cycle25/"
                "RF_04_D12_L5_BALSUB.parquet"
            ),

        "audit_file":
            Path(
                "artifacts/models/cross_cycle_shap/"
                "contracts/"
                "rf_cycle25_shap_parallel_audit.json"
            ),

        "required_audit_status":
            "COMPLETE_RF_CYCLE25_SHAP",

        "explainer":
            "TreeExplainer",

        "input_space":
            "RAW",

        "output_space":
            "POSITIVE_CLASS_PROBABILITY",
    },


    "XGB_02_LR0p03_D3_BAL": {

        "family":
            "xgboost",

        "shap_file":
            Path(
                "artifacts/models/cross_cycle_shap/"
                "shap/cycle25/"
                "XGB_02_LR0p03_D3_BAL.parquet"
            ),

        "audit_file":
            Path(
                "artifacts/models/cross_cycle_shap/"
                "contracts/"
                "xgb_cycle25_shap_parallel_audit.json"
            ),

        "required_audit_status":
            "COMPLETE_XGB_CYCLE25_SHAP",

        "explainer":
            "TreeExplainer",

        "input_space":
            "RAW",

        "output_space":
            "XGBOOST_RAW_MARGIN",
    },
}


# ============================================================
# OUTPUTS
# ============================================================

REGISTRY_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "cycle25_shap_registry.csv"
)


SUMMARY_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "cycle25_shap_summary.csv"
)


FINAL_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "cycle25_shap_contract.json"
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
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.2.4 — "
    "FINALIZE + FREEZE CYCLE25 SHAP"
)

print("=" * 78)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    EXECUTION_CONTRACT_FILE,
    MODEL_MANIFEST_FILE,
    POPULATION_MANIFEST_FILE,
]


for spec in MODEL_SPECS.values():

    required_files.append(
        spec[
            "shap_file"
        ]
    )

    required_files.append(
        spec[
            "audit_file"
        ]
    )


for path in required_files:

    verify_exists(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for output_path in [
    REGISTRY_FILE,
    SUMMARY_FILE,
    FINAL_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Output already exists: {output_path}\n"
            "Refusing to overwrite frozen "
            "Cycle25 SHAP finalization output."
        )


# ============================================================
# VERIFY CHAPTER 10.1 CONTRACT
# ============================================================

execution_contract = load_json(
    EXECUTION_CONTRACT_FILE
)


if (
    execution_contract.get(
        "status"
    )
    !=
    "FROZEN_CYCLE25_SHAP_EXECUTION_INPUTS"
):

    raise RuntimeError(
        "Chapter 10.1 execution inputs "
        "are not frozen."
    )


execution_contract_hash = sha256_file(
    EXECUTION_CONTRACT_FILE
)


print(
    "Chapter 10.1 execution contract:",
    "PASS"
)


# ============================================================
# VERIFY FROZEN MANIFEST HASHES
# ============================================================

model_manifest_hash = sha256_file(
    MODEL_MANIFEST_FILE
)


expected_model_manifest_hash = (
    execution_contract[
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
        "Cycle25 model manifest changed "
        "after Chapter 10.1."
    )


population_manifest_hash = sha256_file(
    POPULATION_MANIFEST_FILE
)


expected_population_manifest_hash = (
    execution_contract[
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
        "Cycle25 population manifest changed "
        "after Chapter 10.1."
    )


model_manifest = pd.read_csv(
    MODEL_MANIFEST_FILE,
    keep_default_na=False,
)


population_manifest = pd.read_csv(
    POPULATION_MANIFEST_FILE,
    keep_default_na=False,
)


if len(
    model_manifest
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen models."
    )


if len(
    population_manifest
) != 4:

    raise RuntimeError(
        "Expected POOLED + B1 + B2 + B3 "
        "population manifest."
    )


print(
    "Frozen 10.1 manifests:",
    "PASS"
)


# ============================================================
# LOAD + VERIFY ALL MODEL ARTIFACTS
# ============================================================

registry_rows = []

summary_rows = []

frames = {}


reference_metadata = None


for candidate_id, spec in MODEL_SPECS.items():

    print()

    print(
        "VERIFY:",
        candidate_id
    )


    # --------------------------------------------------------
    # VERIFY MODEL MANIFEST ENTRY
    # --------------------------------------------------------

    manifest_match = model_manifest[
        model_manifest[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(
        manifest_match
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: frozen model "
            "manifest lookup failed."
        )


    manifest_row = manifest_match.iloc[
        0
    ]


    if (
        str(
            manifest_row[
                "model_family"
            ]
        )
        !=
        spec[
            "family"
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: model family mismatch."
        )


    if (
        str(
            manifest_row[
                "explainer"
            ]
        )
        !=
        spec[
            "explainer"
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: explainer mismatch."
        )


    if (
        str(
            manifest_row[
                "feature_input_space"
            ]
        )
        !=
        spec[
            "input_space"
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: input space mismatch."
        )


    if (
        str(
            manifest_row[
                "output_space"
            ]
        )
        !=
        spec[
            "output_space"
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: output space mismatch."
        )


    # --------------------------------------------------------
    # VERIFY AUDIT
    # --------------------------------------------------------

    audit = load_json(
        spec[
            "audit_file"
        ]
    )


    if (
        audit.get(
            "status"
        )
        !=
        spec[
            "required_audit_status"
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: audit status "
            "is not complete."
        )


    source_contract = audit.get(
        "source_execution_contract",
        {},
    )


    if (
        source_contract.get(
            "sha256"
        )
        !=
        execution_contract_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: audit does not "
            "reference the current frozen "
            "10.1 execution contract."
        )


    scientific_guardrails = audit.get(
        "scientific_guardrails",
        {},
    )


    required_false_guardrails = [
        "cycle25_used_for_training",
        "model_retrained",
        "model_retuned",
        "background_changed",
        "cycle24_null_changed",
        "drift_conclusion_made",
    ]


    for guardrail in required_false_guardrails:

        if (
            guardrail
            in
            scientific_guardrails
        ):

            if (
                scientific_guardrails[
                    guardrail
                ]
                is not False
            ):

                raise RuntimeError(
                    f"{candidate_id}: scientific "
                    f"guardrail failed: {guardrail}"
                )


    # --------------------------------------------------------
    # VERIFY ARTIFACT HASH
    # --------------------------------------------------------

    shap_file = spec[
        "shap_file"
    ]


    current_shap_hash = sha256_file(
        shap_file
    )


    audit_artifact = audit.get(
        "artifact",
        {},
    )


    expected_shap_hash = audit_artifact.get(
        "sha256"
    )


    if (
        expected_shap_hash
        !=
        current_shap_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: SHAP artifact "
            "hash mismatch."
        )


    # --------------------------------------------------------
    # LOAD ARTIFACT
    # --------------------------------------------------------

    frame = pd.read_parquet(
        shap_file
    )


    if len(
        frame
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            f"{candidate_id}: expected "
            f"{EXPECTED_ROWS} rows, got "
            f"{len(frame)}."
        )


    required_columns = {
        "source_row_number",
        "candidate_id",
        "model_family",
        "cycle25_block",
        ENTITY_GROUP_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN,
        "shap_base_value",
        *SHAP_COLUMNS,
    }


    missing_columns = (
        required_columns
        -
        set(
            frame.columns
        )
    )


    if missing_columns:

        raise RuntimeError(
            f"{candidate_id}: missing columns: "
            f"{sorted(missing_columns)}"
        )


    # --------------------------------------------------------
    # EXACT SOURCE ROW ORDER
    # --------------------------------------------------------

    expected_source_rows = np.arange(
        EXPECTED_ROWS,
        dtype=np.int64,
    )


    actual_source_rows = frame[
        "source_row_number"
    ].to_numpy(
        dtype=np.int64
    )


    if not np.array_equal(
        actual_source_rows,
        expected_source_rows,
    ):

        raise RuntimeError(
            f"{candidate_id}: source row "
            "order mismatch."
        )


    if frame[
        "source_row_number"
    ].duplicated().any():

        raise RuntimeError(
            f"{candidate_id}: duplicate "
            "source rows."
        )


    # --------------------------------------------------------
    # CANDIDATE / FAMILY
    # --------------------------------------------------------

    if not (
        frame[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ).all():

        raise RuntimeError(
            f"{candidate_id}: candidate column "
            "mismatch."
        )


    if not (
        frame[
            "model_family"
        ].astype(str)
        ==
        spec[
            "family"
        ]
    ).all():

        raise RuntimeError(
            f"{candidate_id}: model family "
            "column mismatch."
        )


    # --------------------------------------------------------
    # SHAP NUMERIC INTEGRITY
    # --------------------------------------------------------

    shap_matrix = frame[
        SHAP_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )


    if shap_matrix.shape != (
        EXPECTED_ROWS,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            f"{candidate_id}: SHAP matrix "
            "shape mismatch."
        )


    if not np.isfinite(
        shap_matrix
    ).all():

        raise RuntimeError(
            f"{candidate_id}: SHAP matrix "
            "contains non-finite values."
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
            f"{candidate_id}: SHAP base values "
            "contain non-finite values."
        )


    # --------------------------------------------------------
    # BLOCK AUDIT
    # --------------------------------------------------------

    block_counts = (
        frame[
            "cycle25_block"
        ]
        .astype(str)
        .value_counts()
        .to_dict()
    )


    expected_block_counts = {
        "B1": 10411,
        "B2": 33132,
        "B3": 35818,
    }


    if block_counts != expected_block_counts:

        raise RuntimeError(
            f"{candidate_id}: Cycle25 block "
            f"counts mismatch: {block_counts}"
        )


    # --------------------------------------------------------
    # METADATA ALIGNMENT ACROSS MODELS
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        {
            "source_row_number":
                frame[
                    "source_row_number"
                ].to_numpy(
                    dtype=np.int64
                ),

            ENTITY_GROUP_COLUMN:
                frame[
                    ENTITY_GROUP_COLUMN
                ].astype(str).to_numpy(),

            TARGET_COLUMN:
                frame[
                    TARGET_COLUMN
                ].to_numpy(
                    dtype=np.int8
                ),

            "cycle25_block":
                frame[
                    "cycle25_block"
                ].astype(str).to_numpy(),

            TIME_COLUMN:
                pd.to_datetime(
                    frame[
                        TIME_COLUMN
                    ],
                    utc=True,
                ).astype(
                    "int64"
                ).to_numpy(),
        }
    )


    if reference_metadata is None:

        reference_metadata = metadata


    else:

        if not np.array_equal(
            metadata[
                "source_row_number"
            ].to_numpy(),
            reference_metadata[
                "source_row_number"
            ].to_numpy(),
        ):

            raise RuntimeError(
                f"{candidate_id}: source-row "
                "alignment differs across models."
            )


        if not np.array_equal(
            metadata[
                ENTITY_GROUP_COLUMN
            ].to_numpy(),
            reference_metadata[
                ENTITY_GROUP_COLUMN
            ].to_numpy(),
        ):

            raise RuntimeError(
                f"{candidate_id}: ENTITY_GROUP_ID "
                "alignment differs across models."
            )


        if not np.array_equal(
            metadata[
                TARGET_COLUMN
            ].to_numpy(),
            reference_metadata[
                TARGET_COLUMN
            ].to_numpy(),
        ):

            raise RuntimeError(
                f"{candidate_id}: target alignment "
                "differs across models."
            )


        if not np.array_equal(
            metadata[
                "cycle25_block"
            ].to_numpy(),
            reference_metadata[
                "cycle25_block"
            ].to_numpy(),
        ):

            raise RuntimeError(
                f"{candidate_id}: temporal-block "
                "alignment differs across models."
            )


        if not np.array_equal(
            metadata[
                TIME_COLUMN
            ].to_numpy(),
            reference_metadata[
                TIME_COLUMN
            ].to_numpy(),
        ):

            raise RuntimeError(
                f"{candidate_id}: timestamp "
                "alignment differs across models."
            )


    # --------------------------------------------------------
    # UNIFIED POOLED SUMMARY
    #
    # This is descriptive only.
    # No cross-cycle comparison occurs here.
    # --------------------------------------------------------

    target = frame[
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


    for stratum_name, mask in strata.items():

        stratum_shap = shap_matrix[
            mask
        ]


        mean_abs = np.mean(
            np.abs(
                stratum_shap
            ),
            axis=0,
        )


        mean_signed = np.mean(
            stratum_shap,
            axis=0,
        )


        mean_abs_rank = (
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
                        candidate_id,

                    "model_family":
                        spec[
                            "family"
                        ],

                    "population":
                        "CYCLE25_POOLED",

                    "stratum":
                        stratum_name,

                    "rows":
                        int(
                            mask.sum()
                        ),

                    "entity_groups":
                        int(
                            frame.loc[
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
                            mean_abs_rank[
                                feature
                            ]
                        ),

                    "output_space":
                        spec[
                            "output_space"
                        ],
                }
            )


    # --------------------------------------------------------
    # REGISTRY
    # --------------------------------------------------------

    registry_rows.append(
        {
            "candidate_id":
                candidate_id,

            "model_family":
                spec[
                    "family"
                ],

            "shap_file":
                shap_file.as_posix(),

            "shap_sha256":
                current_shap_hash,

            "rows":
                EXPECTED_ROWS,

            "features":
                EXPECTED_FEATURES,

            "explainer":
                spec[
                    "explainer"
                ],

            "feature_input_space":
                spec[
                    "input_space"
                ],

            "output_space":
                spec[
                    "output_space"
                ],

            "audit_file":
                spec[
                    "audit_file"
                ].as_posix(),

            "audit_sha256":
                sha256_file(
                    spec[
                        "audit_file"
                    ]
                ),

            "audit_status":
                spec[
                    "required_audit_status"
                ],

            "cycle25_used_for_training":
                False,

            "model_retrained":
                False,

            "drift_conclusion_made":
                False,
        }
    )


    frames[
        candidate_id
    ] = frame


    print(
        "Artifact hash:",
        "PASS"
    )

    print(
        "Rows:",
        EXPECTED_ROWS
    )

    print(
        "SHAP matrix:",
        (
            EXPECTED_ROWS,
            EXPECTED_FEATURES,
        )
    )

    print(
        "Metadata alignment:",
        "PASS"
    )

    print(
        "Scientific guardrails:",
        "PASS"
    )


# ============================================================
# VERIFY THREE-MODEL ALIGNMENT
# ============================================================

if len(
    frames
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 verified "
        "Cycle25 SHAP artifacts."
    )


print()

print(
    "Cross-model Cycle25 row alignment:",
    "PASS"
)


# ============================================================
# VERIFY FINAL POPULATION COUNTS
# ============================================================

if reference_metadata is None:

    raise RuntimeError(
        "Reference metadata was not created."
    )


if len(
    reference_metadata
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Reference metadata row-count mismatch."
    )


positive_rows = int(
    (
        reference_metadata[
            TARGET_COLUMN
        ]
        ==
        1
    ).sum()
)


negative_rows = int(
    (
        reference_metadata[
            TARGET_COLUMN
        ]
        ==
        0
    ).sum()
)


entity_groups = int(
    reference_metadata[
        ENTITY_GROUP_COLUMN
    ].nunique()
)


if positive_rows != 4836:

    raise RuntimeError(
        "Cycle25 positive-count mismatch."
    )


if negative_rows != 74525:

    raise RuntimeError(
        "Cycle25 negative-count mismatch."
    )


if entity_groups != 1011:

    raise RuntimeError(
        "Cycle25 entity-group count mismatch."
    )


print(
    "Cycle25 population counts:",
    "PASS"
)


# ============================================================
# BUILD FINAL REGISTRY + SUMMARY
# ============================================================

registry = pd.DataFrame(
    registry_rows
)


summary = pd.DataFrame(
    summary_rows
)


if len(
    registry
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 registry rows."
    )


if len(
    summary
) != EXPECTED_SUMMARY_ROWS:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_SUMMARY_ROWS} summary rows, "
        f"got {len(summary)}."
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

REGISTRY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


FINAL_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ATOMIC CSV WRITE
# ============================================================

registry_temp = REGISTRY_FILE.with_name(
    REGISTRY_FILE.stem
    +
    ".tmp.csv"
)


summary_temp = SUMMARY_FILE.with_name(
    SUMMARY_FILE.stem
    +
    ".tmp.csv"
)


for path in [
    registry_temp,
    summary_temp,
]:

    if path.exists():

        path.unlink()


registry.to_csv(
    registry_temp,
    index=False,
)


summary.to_csv(
    summary_temp,
    index=False,
)


# ============================================================
# CSV ROUND-TRIP VALIDATION
# ============================================================

registry_rt = pd.read_csv(
    registry_temp,
    keep_default_na=False,
)


summary_rt = pd.read_csv(
    summary_temp,
    keep_default_na=False,
)


if len(
    registry_rt
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Registry round-trip failed."
    )


if len(
    summary_rt
) != EXPECTED_SUMMARY_ROWS:

    raise RuntimeError(
        "Summary round-trip failed."
    )


registry_temp.replace(
    REGISTRY_FILE
)


summary_temp.replace(
    SUMMARY_FILE
)


registry_hash = sha256_file(
    REGISTRY_FILE
)


summary_hash = sha256_file(
    SUMMARY_FILE
)


print(
    "Registry round-trip:",
    "PASS"
)

print(
    "Unified summary round-trip:",
    "PASS"
)


# ============================================================
# FINAL FROZEN CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Frozen Cycle25 SHAP Artifacts",

    "version":
        "1.0",

    "status":
        "FROZEN_CYCLE25_SHAP",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_2_4",

    "source_execution_contract":
        {
            "file":
                EXECUTION_CONTRACT_FILE.as_posix(),

            "sha256":
                execution_contract_hash,

            "status":
                "FROZEN_CYCLE25_SHAP_EXECUTION_INPUTS",
        },

    "population":
        {
            "rows":
                EXPECTED_ROWS,

            "positive_rows":
                positive_rows,

            "negative_rows":
                negative_rows,

            "entity_groups":
                entity_groups,

            "temporal_blocks":
                {
                    "B1":
                        10411,

                    "B2":
                        33132,

                    "B3":
                        35818,
                },

            "population_manifest_file":
                POPULATION_MANIFEST_FILE.as_posix(),

            "population_manifest_sha256":
                population_manifest_hash,
        },

    "models":
        {
            "count":
                EXPECTED_MODELS,

            "candidate_ids":
                list(
                    MODEL_SPECS.keys()
                ),

            "registry_file":
                REGISTRY_FILE.as_posix(),

            "registry_sha256":
                registry_hash,

            "registry_rows":
                EXPECTED_MODELS,
        },

    "summary":
        {
            "file":
                SUMMARY_FILE.as_posix(),

            "sha256":
                summary_hash,

            "rows":
                EXPECTED_SUMMARY_ROWS,

            "population":
                "CYCLE25_POOLED",

            "strata":
                [
                    "OVERALL",
                    "TARGET_0",
                    "TARGET_1",
                ],

            "features_per_model_per_stratum":
                EXPECTED_FEATURES,
        },

    "explanation_spaces":
        {
            "LR_02_C0p01_BALANCED":
                "LOG_ODDS_DECISION_FUNCTION",

            "RF_04_D12_L5_BALSUB":
                "POSITIVE_CLASS_PROBABILITY",

            "XGB_02_LR0p03_D3_BAL":
                "XGBOOST_RAW_MARGIN",
        },

    "comparison_rules":
        {
            "cross_cycle_comparison":
                "WITHIN_MODEL_ONLY",

            "raw_shap_magnitude_across_models":
                "PROHIBITED",

            "reason":
                (
                    "LR, RF and XGB explanations "
                    "exist in different model-output "
                    "spaces."
                ),

            "cycle25_population_same_across_models":
                True,

            "cycle25_row_order_same_across_models":
                True,

            "cycle25_targets_same_across_models":
                True,

            "cycle25_entity_groups_same_across_models":
                True,

            "cycle25_temporal_blocks_same_across_models":
                True,
        },

    "scientific_guardrails":
        {
            "cycle25_shap_generated":
                True,

            "all_three_models_complete":
                True,

            "cycle25_used_for_training":
                False,

            "cycle25_used_for_model_selection":
                False,

            "cycle25_used_for_feature_selection":
                False,

            "cycle25_used_for_preprocessing_fit":
                False,

            "models_retrained":
                False,

            "models_retuned":
                False,

            "shap_background_changed":
                False,

            "cycle24_null_changed":
                False,

            "common_support_definition_changed":
                False,

            "correlated_feature_groups_changed":
                False,

            "drift_analysis_performed":
                False,

            "drift_conclusion_made":
                False,

            "shap_is_causal":
                False,
        },

    "immutability_rule":
        (
            "After MODEL_CHAPTER_10_2_4, the three "
            "Cycle25 SHAP artifacts, their row order, "
            "model-specific explanation spaces, frozen "
            "Cycle24-TRAIN background, preprocessing, "
            "and model binaries must not be changed in "
            "response to subsequent cross-cycle drift "
            "results."
        ),

    "next_stage":
        "MODEL_CHAPTER_10_3_OVERALL_CROSS_CYCLE_SHAP",
}


# ============================================================
# ATOMIC CONTRACT WRITE
# ============================================================

contract_temp = FINAL_CONTRACT_FILE.with_name(
    FINAL_CONTRACT_FILE.stem
    +
    ".tmp.json"
)


if contract_temp.exists():

    contract_temp.unlink()


with contract_temp.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


# ============================================================
# CONTRACT ROUND-TRIP
# ============================================================

contract_rt = load_json(
    contract_temp
)


if (
    contract_rt.get(
        "status"
    )
    !=
    "FROZEN_CYCLE25_SHAP"
):

    raise RuntimeError(
        "Cycle25 SHAP final contract "
        "round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "drift_analysis_performed"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected drift-analysis state."
    )


contract_temp.replace(
    FINAL_CONTRACT_FILE
)


contract_hash = sha256_file(
    FINAL_CONTRACT_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 10.2.4 — COMPLETE"
)

print("=" * 78)


print(
    "Verified Cycle25 SHAP models:",
    EXPECTED_MODELS
)

print(
    "Rows per model:",
    EXPECTED_ROWS
)

print(
    "Features per model:",
    EXPECTED_FEATURES
)

print(
    "Positive rows:",
    positive_rows
)

print(
    "Negative rows:",
    negative_rows
)

print(
    "Entity groups:",
    entity_groups
)


print()

print(
    "LR artifact:",
    "PASS"
)

print(
    "RF artifact:",
    "PASS"
)

print(
    "XGB artifact:",
    "PASS"
)

print(
    "Cross-model row alignment:",
    "PASS"
)


print()

print(
    "Unified summary rows:",
    len(
        summary
    )
)

print(
    "Registry rows:",
    len(
        registry
    )
)


print()

print(
    "Cycle25 used for training:",
    False
)

print(
    "Models retrained:",
    False
)

print(
    "Cycle24 null changed:",
    False
)

print(
    "Drift analysis performed:",
    False
)

print(
    "Drift conclusion made:",
    False
)


print()

print(
    "Status:",
    "FROZEN_CYCLE25_SHAP"
)


print()

print(
    "Registry:",
    REGISTRY_FILE
)

print(
    "Summary:",
    SUMMARY_FILE
)

print(
    "Contract:",
    FINAL_CONTRACT_FILE
)

print(
    "Contract SHA256:",
    contract_hash
)


print()

print(
    "NEXT:",
    "Chapter 10.3 — Overall Cycle24 vs Cycle25 SHAP"
)