from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

CYCLE25_DATASET_FILE = Path(
    "data/final/dataset_v2/"
    "cycle25_dataset_v2.parquet"
)

HOLDOUT_RESULTS_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_results_contract.json"
)

HOLDOUT_RESULTS_MANIFEST_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "frozen_holdout_results.csv"
)

HOLDOUT_AUDIT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_evaluation_audit.json"
)

HOLDOUT_BOOTSTRAP_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_bootstrap_contract.json"
)

HOLDOUT_MODEL_REGISTRY_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "holdout_frozen_model_registry.csv"
)

SELECTED_MODELS_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "selected_models_contract.json"
)

PREPROCESSING_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "preprocessing_contract.json"
)

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "model_feature_contract.json"
)

TEMPORAL_DATA_CONTRACT_FILE = Path(
    "data_manifests/"
    "temporal_data_contract.json"
)

LOGISTIC_SCALER_FILE = Path(
    "artifacts/preprocessing/scalers/"
    "logistic_standard_scaler.joblib"
)

OUTPUT_PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_evaluation_protocol.json"
)

OUTPUT_MODEL_REGISTRY_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_frozen_model_registry.csv"
)

OUTPUT_BLOCKS_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_temporal_blocks.csv"
)


# ============================================================
# FROZEN CYCLE25 DATASET EXPECTATIONS
# ============================================================

EXPECTED_CYCLE25_SHA256 = (
    "0bd6a364e152685d13c7eb955da074a4"
    "b9f019f1851c6b5dafebf295c406cd4d"
)

EXPECTED_CYCLE25_ROWS = 79361
EXPECTED_CYCLE25_POSITIVES = 4836
EXPECTED_CYCLE25_NEGATIVES = 74525

CYCLE25_START_UTC = (
    "2020-01-01T00:00:00+00:00"
)

CYCLE25_END_EXCLUSIVE_UTC = (
    "2026-01-01T00:00:00+00:00"
)

TIME_ASSIGNMENT_COLUMN = (
    "observation_time_utc"
)

ENTITY_GROUP_SOURCE_COLUMN = (
    "ENTITY_GROUP_ID"
)


# ============================================================
# FROZEN FEATURE ORDER
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
# TEMPORAL BLOCKS
# ============================================================

TEMPORAL_BLOCKS = [
    {
        "block_id":
            "C25_B1_2020_2021",

        "start_utc":
            "2020-01-01T00:00:00+00:00",

        "end_exclusive_utc":
            "2022-01-01T00:00:00+00:00",

        "primary_block":
            True,

        "pooled_summary":
            False,

        "bootstrap_seed":
            25025,
    },

    {
        "block_id":
            "C25_B2_2022_2023",

        "start_utc":
            "2022-01-01T00:00:00+00:00",

        "end_exclusive_utc":
            "2024-01-01T00:00:00+00:00",

        "primary_block":
            True,

        "pooled_summary":
            False,

        "bootstrap_seed":
            25026,
    },

    {
        "block_id":
            "C25_B3_2024_2025",

        "start_utc":
            "2024-01-01T00:00:00+00:00",

        "end_exclusive_utc":
            "2026-01-01T00:00:00+00:00",

        "primary_block":
            True,

        "pooled_summary":
            False,

        "bootstrap_seed":
            25027,
    },

    {
        "block_id":
            "C25_ALL_2020_2025",

        "start_utc":
            "2020-01-01T00:00:00+00:00",

        "end_exclusive_utc":
            "2026-01-01T00:00:00+00:00",

        "primary_block":
            False,

        "pooled_summary":
            True,

        "bootstrap_seed":
            25028,
    },
]


# ============================================================
# FROZEN BOOTSTRAP DESIGN
# ============================================================

BOOTSTRAP_VALID_REPLICATES = 2000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

BOOTSTRAP_LOWER_QUANTILE = 0.025
BOOTSTRAP_UPPER_QUANTILE = 0.975


BOOTSTRAP_METRICS = [
    "TSS",
    "HSS",
    "F1",
    "sensitivity",
    "specificity",
    "precision",
    "PR_AUC",
    "ROC_AUC",
    "Brier_score",
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


def json_contains_exact_list(
    obj,
    expected_list,
):

    if isinstance(
        obj,
        list,
    ):

        if obj == expected_list:
            return True

        return any(
            json_contains_exact_list(
                item,
                expected_list,
            )
            for item in obj
        )


    if isinstance(
        obj,
        dict,
    ):

        return any(
            json_contains_exact_list(
                value,
                expected_list,
            )
            for value in obj.values()
        )


    return False


def json_contains_scalar(
    obj,
    target,
):

    if isinstance(
        obj,
        dict,
    ):

        return any(
            json_contains_scalar(
                value,
                target,
            )
            for value in obj.values()
        )


    if isinstance(
        obj,
        list,
    ):

        return any(
            json_contains_scalar(
                value,
                target,
            )
            for value in obj
        )


    return obj == target


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 8.1 — FREEZE CYCLE25 EVALUATION PROTOCOL"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED UPSTREAM FILES
# ============================================================

required_files = [
    HOLDOUT_RESULTS_CONTRACT_FILE,
    HOLDOUT_RESULTS_MANIFEST_FILE,
    HOLDOUT_AUDIT_CONTRACT_FILE,
    HOLDOUT_BOOTSTRAP_CONTRACT_FILE,
    HOLDOUT_MODEL_REGISTRY_FILE,
    SELECTED_MODELS_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    TEMPORAL_DATA_CONTRACT_FILE,
    LOGISTIC_SCALER_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required upstream "
            f"artifact: {path}"
        )


# ============================================================
# CYCLE25 FILE EXISTENCE ONLY
# ============================================================

if not CYCLE25_DATASET_FILE.exists():

    raise FileNotFoundError(
        "Frozen Cycle25 Dataset v2 "
        f"was not found: {CYCLE25_DATASET_FILE}"
    )


print(
    "Cycle25 Dataset v2 file exists:",
    True
)

print(
    "Cycle25 Dataset v2 content read:",
    False
)


# ============================================================
# PROTECT OUTPUTS
# ============================================================

for output_path in [
    OUTPUT_PROTOCOL_FILE,
    OUTPUT_MODEL_REGISTRY_FILE,
    OUTPUT_BLOCKS_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Protocol output already exists: "
            f"{output_path}\n"
            "Refusing to overwrite frozen "
            "Cycle25 evaluation design."
        )


# ============================================================
# LOAD UPSTREAM CONTRACTS
# ============================================================

holdout_results_contract = load_json(
    HOLDOUT_RESULTS_CONTRACT_FILE
)

holdout_audit_contract = load_json(
    HOLDOUT_AUDIT_CONTRACT_FILE
)

holdout_bootstrap_contract = load_json(
    HOLDOUT_BOOTSTRAP_CONTRACT_FILE
)

selected_models_contract = load_json(
    SELECTED_MODELS_CONTRACT_FILE
)

preprocessing_contract = load_json(
    PREPROCESSING_CONTRACT_FILE
)

feature_contract = load_json(
    FEATURE_CONTRACT_FILE
)

temporal_data_contract = load_json(
    TEMPORAL_DATA_CONTRACT_FILE
)


# ============================================================
# VERIFY CYCLE24 HANDOFF
# ============================================================

if holdout_results_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 holdout results are "
        "not FROZEN."
    )


if holdout_results_contract.get(
    "evaluation_stage"
) != "CYCLE24_HOLDOUT_COMPLETE":

    raise RuntimeError(
        "Cycle24 holdout evaluation "
        "is not marked complete."
    )


if holdout_audit_contract.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "Cycle24 holdout independent "
        "audit did not PASS."
    )


if not holdout_audit_contract.get(
    "overall_holdout_evaluation_audit_pass",
    False,
):

    raise RuntimeError(
        "Cycle24 holdout overall audit "
        "PASS flag is False."
    )


if selected_models_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Selected model contract "
        "is not FROZEN."
    )


if preprocessing_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Preprocessing contract "
        "is not FROZEN."
    )


if feature_contract.get(
    "status"
) != "VERIFIED":

    raise RuntimeError(
        "Feature contract "
        "is not VERIFIED."
    )


if not json_contains_scalar(
    temporal_data_contract,
    "FROZEN",
):

    raise RuntimeError(
        "Temporal data contract does not "
        "contain a FROZEN state."
    )


print(
    "Cycle24 holdout results:",
    "FROZEN"
)

print(
    "Cycle24 holdout independent audit:",
    "PASS"
)

print(
    "Selected models:",
    "FROZEN"
)

print(
    "Preprocessing:",
    "FROZEN"
)

print(
    "Feature contract:",
    "VERIFIED"
)

print(
    "Temporal data contract:",
    "FROZEN"
)


# ============================================================
# VERIFY FEATURE ORDER
# ============================================================

if len(
    FEATURES
) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


if len(
    set(
        FEATURES
    )
) != 25:

    raise RuntimeError(
        "Duplicate frozen features detected."
    )


if not json_contains_exact_list(
    feature_contract,
    FEATURES,
):

    raise RuntimeError(
        "Exact frozen 25-feature ordering "
        "was not found in feature contract."
    )


print(
    "Frozen feature order:",
    "PASS"
)


# ============================================================
# VERIFY FROZEN CYCLE24 HOLDOUT RESULT MANIFEST
# ============================================================

holdout_results = pd.read_csv(
    HOLDOUT_RESULTS_MANIFEST_FILE,
    keep_default_na=False,
)


if len(
    holdout_results
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen "
        "Cycle24 holdout result rows."
    )


if holdout_results[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate in frozen "
        "Cycle24 holdout results."
    )


expected_families = {
    "logistic",
    "random_forest",
    "xgboost",
}


if set(
    holdout_results[
        "family"
    ].astype(str)
) != expected_families:

    raise RuntimeError(
        "Frozen Cycle24 holdout family "
        "set is incorrect."
    )


expected_holdout_manifest_hash = (
    holdout_results_contract[
        "artifacts"
    ][
        "frozen_holdout_manifest"
    ][
        "sha256"
    ]
)


actual_holdout_manifest_hash = sha256_file(
    HOLDOUT_RESULTS_MANIFEST_FILE
)


if (
    actual_holdout_manifest_hash
    !=
    expected_holdout_manifest_hash
):

    raise RuntimeError(
        "Frozen Cycle24 holdout result "
        "manifest hash mismatch."
    )


print(
    "Cycle24 frozen baseline hash:",
    "PASS"
)


# ============================================================
# LOAD FROZEN MODEL REGISTRY
# ============================================================

holdout_registry = pd.read_csv(
    HOLDOUT_MODEL_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    holdout_registry
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen models."
    )


if set(
    holdout_registry[
        "family"
    ].astype(str)
) != expected_families:

    raise RuntimeError(
        "Frozen model registry family "
        "set is incorrect."
    )


if holdout_registry[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate frozen candidate."
    )


# ============================================================
# VERIFY EXACT EXPECTED MODELS
# ============================================================

expected_models = {

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


cycle25_registry_rows = []


print(
    "\n===================================="
)

print(
    "VERIFY FROZEN CYCLE24 MODELS"
)

print(
    "===================================="
)


for family, expected in expected_models.items():

    row = holdout_registry[
        holdout_registry[
            "family"
        ].astype(str)
        ==
        family
    ]


    if len(
        row
    ) != 1:

        raise RuntimeError(
            f"{family}: expected exactly "
            "one frozen model."
        )


    row = row.iloc[
        0
    ]


    candidate_id = str(
        row[
            "candidate_id"
        ]
    )


    if (
        candidate_id
        !=
        expected[
            "candidate_id"
        ]
    ):

        raise RuntimeError(
            f"{family}: frozen candidate "
            "identity mismatch."
        )


    threshold = float(
        row[
            "frozen_threshold"
        ]
    )


    if abs(
        threshold
        -
        expected[
            "threshold"
        ]
    ) > 1e-9:

        raise RuntimeError(
            f"{candidate_id}: frozen "
            "threshold mismatch."
        )


    model_file = Path(
        str(
            row[
                "model_file"
            ]
        )
    )


    if not model_file.exists():

        raise FileNotFoundError(
            f"{candidate_id}: frozen model "
            f"artifact missing: {model_file}"
        )


    expected_model_hash = str(
        row[
            "model_sha256"
        ]
    )


    actual_model_hash = sha256_file(
        model_file
    )


    if (
        actual_model_hash
        !=
        expected_model_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: frozen model "
            "artifact hash mismatch."
        )


    holdout_result_row = holdout_results[
        holdout_results[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(
        holdout_result_row
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: frozen Cycle24 "
            "baseline row missing."
        )


    holdout_result_row = (
        holdout_result_row.iloc[
            0
        ]
    )


    if abs(
        float(
            holdout_result_row[
                "frozen_threshold"
            ]
        )
        -
        threshold
    ) > 1e-9:

        raise RuntimeError(
            f"{candidate_id}: threshold "
            "differs between model registry "
            "and Cycle24 holdout baseline."
        )


    cycle25_registry_rows.append(
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
                threshold,

            "threshold_source":
                "CYCLE24_VALIDATION",

            "training_scope":
                "CYCLE24_TRAIN_ONLY",

            "cycle24_holdout_tss":
                float(
                    holdout_result_row[
                        "tss"
                    ]
                ),

            "cycle24_holdout_tss_ci_lower_95":
                float(
                    holdout_result_row[
                        "tss_ci_lower_95"
                    ]
                ),

            "cycle24_holdout_tss_ci_upper_95":
                float(
                    holdout_result_row[
                        "tss_ci_upper_95"
                    ]
                ),

            "cycle25_action":
                "PREDICT_ONLY",

            "model_retraining_allowed":
                False,

            "preprocessing_refit_allowed":
                False,

            "threshold_reoptimization_allowed":
                False,

            "feature_change_allowed":
                False,

            "candidate_replacement_allowed":
                False,
        }
    )


    print(
        f"{family}: "
        f"{candidate_id} | "
        f"threshold={threshold:.10g} | "
        f"model_hash=PASS"
    )


cycle25_registry = pd.DataFrame(
    cycle25_registry_rows
)


# ============================================================
# VERIFY LOGISTIC SCALER HASH
# ============================================================

if not LOGISTIC_SCALER_FILE.exists():

    raise FileNotFoundError(
        "Frozen Logistic scaler artifact "
        "is missing."
    )


logistic_scaler_hash = sha256_file(
    LOGISTIC_SCALER_FILE
)


print(
    "\nFrozen Logistic scaler artifact:",
    "PRESENT"
)


# ============================================================
# BUILD TEMPORAL BLOCK MANIFEST
# ============================================================

blocks_df = pd.DataFrame(
    TEMPORAL_BLOCKS
)


if len(
    blocks_df[
        blocks_df[
            "primary_block"
        ]
    ]
) != 3:

    raise RuntimeError(
        "Exactly 3 primary temporal "
        "blocks are required."
    )


if len(
    blocks_df[
        blocks_df[
            "pooled_summary"
        ]
    ]
) != 1:

    raise RuntimeError(
        "Exactly one pooled Cycle25 "
        "summary definition is required."
    )


if blocks_df[
    "block_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate temporal block ID."
    )


# ============================================================
# SAVE REGISTRIES BEFORE PROTOCOL
# ============================================================

OUTPUT_MODEL_REGISTRY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_BLOCKS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_PROTOCOL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


cycle25_registry.to_csv(
    OUTPUT_MODEL_REGISTRY_FILE,
    index=False,
)


blocks_df.to_csv(
    OUTPUT_BLOCKS_FILE,
    index=False,
)


model_registry_hash = sha256_file(
    OUTPUT_MODEL_REGISTRY_FILE
)


blocks_hash = sha256_file(
    OUTPUT_BLOCKS_FILE
)


# ============================================================
# METRIC DEFINITIONS
# ============================================================

metric_definitions = {

    "TSS":
        "TPR - FPR",

    "HSS":
        (
            "2*(TP*TN - FN*FP) / "
            "((TP+FN)*(FN+TN) + "
            "(TP+FP)*(FP+TN))"
        ),

    "F1":
        "2*TP / (2*TP + FP + FN)",

    "sensitivity":
        "TP / (TP + FN)",

    "specificity":
        "TN / (TN + FP)",

    "precision":
        "TP / (TP + FP)",

    "PR_AUC":
        (
            "sklearn.metrics."
            "average_precision_score"
        ),

    "PR_AUC_operational_name":
        (
            "Average Precision; preserved "
            "for consistency with frozen "
            "Cycle24 evaluation."
        ),

    "ROC_AUC":
        "sklearn.metrics.roc_auc_score",

    "Brier_score":
        "sklearn.metrics.brier_score_loss",
}


# ============================================================
# BUILD CYCLE25 PROTOCOL
# ============================================================

protocol = {

    "contract_name":
        "Frozen Solar Cycle 25 Predictive Evaluation Protocol",

    "version":
        "1.0",

    "status":
        "FROZEN_BEFORE_CYCLE25_EVALUATION",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_8",

    "scientific_role":
        (
            "Temporal/domain-transfer evaluation of "
            "the exact models frozen after Cycle24 "
            "development. Cycle25 is evaluation-only "
            "and is not used for adaptation."
        ),

    "cycle25_dataset":
        {
            "file":
                str(
                    CYCLE25_DATASET_FILE
                ),

            "expected_sha256":
                EXPECTED_CYCLE25_SHA256,

            "expected_rows":
                EXPECTED_CYCLE25_ROWS,

            "expected_positive":
                EXPECTED_CYCLE25_POSITIVES,

            "expected_negative":
                EXPECTED_CYCLE25_NEGATIVES,

            "start_utc":
                CYCLE25_START_UTC,

            "end_exclusive_utc":
                CYCLE25_END_EXCLUSIVE_UTC,

            "time_assignment_column":
                TIME_ASSIGNMENT_COLUMN,

            "entity_group_source_column":
                ENTITY_GROUP_SOURCE_COLUMN,

            "evaluation_only":
                True,

            "domain_language":
                (
                    "Cycle25 is treated as a temporal/"
                    "domain shift evaluation regime. "
                    "It is not automatically labelled "
                    "out-of-distribution."
                ),
        },

    "feature_contract":
        {
            "feature_count":
                25,

            "feature_order":
                FEATURES,

            "changes_allowed":
                False,
        },

    "model_count":
        3,

    "cross_family_winner_declared":
        False,

    "models":
        [
            {
                "family":
                    row[
                        "family"
                    ],

                "candidate_id":
                    row[
                        "candidate_id"
                    ],

                "model_file":
                    row[
                        "model_file"
                    ],

                "model_sha256":
                    row[
                        "model_sha256"
                    ],

                "frozen_threshold":
                    float(
                        row[
                            "frozen_threshold"
                        ]
                    ),

                "training_scope":
                    "CYCLE24_TRAIN_ONLY",

                "threshold_source":
                    "CYCLE24_VALIDATION",
            }

            for _, row
            in cycle25_registry.iterrows()
        ],

    "temporal_evaluation":
        {
            "assignment_rule":
                (
                    "Assign rows using "
                    "observation_time_utc with "
                    "start-inclusive and "
                    "end-exclusive intervals."
                ),

            "primary_block_count":
                3,

            "primary_blocks":
                [
                    block
                    for block
                    in TEMPORAL_BLOCKS
                    if block[
                        "primary_block"
                    ]
                ],

            "pooled_secondary_summary":
                [
                    block
                    for block
                    in TEMPORAL_BLOCKS
                    if block[
                        "pooled_summary"
                    ]
                ][0],

            "primary_inference":
                (
                    "Temporal-block results are the "
                    "primary representation of Cycle25 "
                    "transfer behavior."
                ),

            "pooled_role":
                (
                    "The complete 2020-2025 pool is a "
                    "secondary overall-transfer summary "
                    "and must not replace temporal-block "
                    "reporting."
                ),

            "block_boundaries_selected_before_outcomes":
                True,

            "block_design_basis":
                (
                    "Three equal-duration two-year "
                    "calendar windows spanning the "
                    "frozen 2020-2025 Cycle25 analysis "
                    "window. Boundaries are fixed before "
                    "Cycle25 predictions or metrics are "
                    "examined."
                ),
        },

    "prediction_rule":
        (
            "For each model, classify positive when "
            "predicted probability is greater than "
            "or equal to that model's frozen "
            "Cycle24-validation-derived threshold."
        ),

    "metric_definitions":
        metric_definitions,

    "primary_predictive_metric":
        "TSS",

    "complementary_metrics":
        [
            "HSS",
            "F1",
            "sensitivity",
            "specificity",
            "precision",
            "PR_AUC",
            "ROC_AUC",
            "Brier_score",
        ],

    "calibration":
        {
            "brier_score":
                True,

            "reliability_table":
                True,

            "binning":
                "fixed_probability_width",

            "bin_edges":
                [
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ],

            "same_as_cycle24":
                True,

            "recalibration_allowed":
                False,

            "diagnostic_only":
                True,
        },

    "bootstrap":
        {
            "enabled":
                True,

            "unit":
                "ENTITY_GROUP",

            "source_column":
                ENTITY_GROUP_SOURCE_COLUMN,

            "method":
                (
                    "Within each temporal block and "
                    "within the pooled Cycle25 summary, "
                    "sample all available ENTITY_GROUP "
                    "clusters with replacement. Every "
                    "row belonging to a sampled group "
                    "is included once per draw of that "
                    "group."
                ),

            "valid_replicates_per_evaluation":
                BOOTSTRAP_VALID_REPLICATES,

            "confidence_level":
                BOOTSTRAP_CONFIDENCE_LEVEL,

            "ci_method":
                "percentile",

            "lower_quantile":
                BOOTSTRAP_LOWER_QUANTILE,

            "upper_quantile":
                BOOTSTRAP_UPPER_QUANTILE,

            "invalid_replicate_rule":
                (
                    "A replicate lacking either target "
                    "class is invalid and must be redrawn "
                    "until exactly 2000 valid replicates "
                    "are obtained."
                ),

            "block_specific_seeds":
                {
                    block[
                        "block_id"
                    ]:
                        block[
                            "bootstrap_seed"
                        ]

                    for block
                    in TEMPORAL_BLOCKS
                },

            "metrics":
                BOOTSTRAP_METRICS,

            "models_frozen":
                True,

            "thresholds_frozen":
                True,
        },

    "cross_cycle_comparison":
        {
            "reference":
                "CYCLE24_HOLDOUT",

            "comparison_direction":
                (
                    "delta = Cycle25 metric "
                    "- Cycle24 holdout metric"
                ),

            "primary_delta_metric":
                "TSS",

            "cycle24_baseline_source":
                str(
                    HOLDOUT_RESULTS_MANIFEST_FILE
                ),

            "difference_uncertainty":
                {
                    "method":
                        (
                            "Combine independently generated "
                            "Cycle24 and Cycle25 ENTITY_GROUP "
                            "bootstrap replicates by matching "
                            "replicate_id. Cycle24 uses its "
                            "already-frozen bootstrap; Cycle25 "
                            "uses distinct predeclared seeds."
                        ),

                    "replicates":
                        2000,

                    "ci_method":
                        "percentile",

                    "confidence_level":
                        0.95,
                },

            "descriptive_tss_change_labels":
                {
                    "measurable_decrease":
                        (
                            "95% bootstrap delta-TSS CI "
                            "lies entirely below zero."
                        ),

                    "measurable_increase":
                        (
                            "95% bootstrap delta-TSS CI "
                            "lies entirely above zero."
                        ),

                    "no_measurable_change_at_95ci":
                        (
                            "95% bootstrap delta-TSS CI "
                            "contains zero."
                        ),
                },

            "important_interpretation_rule":
                (
                    "A delta CI containing zero is not "
                    "proof of equivalence or stability. "
                    "Report it only as no measurable "
                    "difference at the stated uncertainty "
                    "level unless a separate equivalence "
                    "criterion is predeclared."
                ),

            "marginal_ci_overlap_not_used_as_difference_test":
                True,

            "cross_family_superiority_test":
                False,
        },

    "cycle24_baseline":
        [
            {
                "family":
                    row[
                        "family"
                    ],

                "candidate_id":
                    row[
                        "candidate_id"
                    ],

                "threshold":
                    float(
                        row[
                            "frozen_threshold"
                        ]
                    ),

                "TSS":
                    float(
                        row[
                            "tss"
                        ]
                    ),

                "TSS_CI95":
                    [
                        float(
                            row[
                                "tss_ci_lower_95"
                            ]
                        ),

                        float(
                            row[
                                "tss_ci_upper_95"
                            ]
                        ),
                    ],

                "PR_AUC":
                    float(
                        row[
                            "pr_auc"
                        ]
                    ),

                "ROC_AUC":
                    float(
                        row[
                            "roc_auc"
                        ]
                    ),

                "Brier_score":
                    float(
                        row[
                            "brier_score"
                        ]
                    ),
            }

            for _, row
            in holdout_results.iterrows()
        ],

    "strict_prohibitions":
        [
            "retrain_model_on_cycle25",
            "fine_tune_model_on_cycle25",
            "replace_selected_candidate",
            "modify_hyperparameters",
            "change_feature_set",
            "change_feature_order",
            "fit_preprocessing_on_cycle25",
            "refit_logistic_scaler_on_cycle25",
            "reoptimize_threshold_on_cycle25",
            "calibrate_probabilities_using_cycle25",
            "change_temporal_blocks_after_viewing_results",
            "drop_difficult_cycle25_blocks_based_on_performance",
            "select_overall_model_using_cycle25",
        ],

    "later_analysis_boundary":
        {
            "distribution_shift_audit":
                "LATER_CHAPTER",

            "shap_null_baseline":
                "LATER_CHAPTER",

            "cross_cycle_shap_drift":
                "LATER_CHAPTER",

            "cycle25_predictive_results_do_not_define_shap_outcome":
                True,
        },

    "data_isolation_at_protocol_freeze":
        {
            "cycle25_dataset_file_existence_checked":
                True,

            "cycle25_dataset_content_read":
                False,

            "cycle25_hash_recomputed":
                False,

            "cycle25_rows_inspected":
                False,

            "cycle25_labels_inspected":
                False,

            "cycle25_temporal_block_counts_seen":
                False,

            "cycle25_predictions_generated":
                False,

            "cycle25_metrics_computed":
                False,

            "model_retraining":
                False,

            "preprocessing_refit":
                False,

            "threshold_reoptimization":
                False,
        },

    "artifacts":
        {
            "cycle25_frozen_model_registry":
                {
                    "file":
                        str(
                            OUTPUT_MODEL_REGISTRY_FILE
                        ),

                    "sha256":
                        model_registry_hash,
                },

            "cycle25_temporal_blocks":
                {
                    "file":
                        str(
                            OUTPUT_BLOCKS_FILE
                        ),

                    "sha256":
                        blocks_hash,
                },
        },

    "provenance":
        {
            "cycle24_holdout_results_contract":
                {
                    "file":
                        str(
                            HOLDOUT_RESULTS_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            HOLDOUT_RESULTS_CONTRACT_FILE
                        ),
                },

            "cycle24_holdout_results_manifest":
                {
                    "file":
                        str(
                            HOLDOUT_RESULTS_MANIFEST_FILE
                        ),

                    "sha256":
                        actual_holdout_manifest_hash,
                },

            "cycle24_holdout_audit_contract":
                {
                    "file":
                        str(
                            HOLDOUT_AUDIT_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            HOLDOUT_AUDIT_CONTRACT_FILE
                        ),
                },

            "cycle24_bootstrap_contract":
                {
                    "file":
                        str(
                            HOLDOUT_BOOTSTRAP_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            HOLDOUT_BOOTSTRAP_CONTRACT_FILE
                        ),
                },

            "selected_models_contract":
                {
                    "file":
                        str(
                            SELECTED_MODELS_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            SELECTED_MODELS_CONTRACT_FILE
                        ),
                },

            "preprocessing_contract":
                {
                    "file":
                        str(
                            PREPROCESSING_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            PREPROCESSING_CONTRACT_FILE
                        ),
                },

            "feature_contract":
                {
                    "file":
                        str(
                            FEATURE_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            FEATURE_CONTRACT_FILE
                        ),
                },

            "temporal_data_contract":
                {
                    "file":
                        str(
                            TEMPORAL_DATA_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            TEMPORAL_DATA_CONTRACT_FILE
                        ),
                },

            "logistic_scaler":
                {
                    "file":
                        str(
                            LOGISTIC_SCALER_FILE
                        ),

                    "sha256":
                        logistic_scaler_hash,
                },
        },

    "immutability_rule":
        (
            "After this Cycle25 evaluation protocol is "
            "created, the Cycle25 dataset may be opened "
            "only for frozen-pipeline evaluation. The "
            "three Cycle24 TRAIN-only models, 25-feature "
            "contract, preprocessing, validation-derived "
            "thresholds, temporal-block boundaries, "
            "metric definitions, calibration design, "
            "bootstrap design, and cross-cycle comparison "
            "rules must remain unchanged regardless of "
            "Cycle25 outcomes."
        ),
}


# ============================================================
# SAVE PROTOCOL
# ============================================================

with OUTPUT_PROTOCOL_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        protocol,
        file,
        indent=2,
    )


# ============================================================
# VERIFY SAVED PROTOCOL
# ============================================================

saved_protocol = load_json(
    OUTPUT_PROTOCOL_FILE
)


if saved_protocol.get(
    "status"
) != "FROZEN_BEFORE_CYCLE25_EVALUATION":

    raise RuntimeError(
        "Cycle25 evaluation protocol "
        "was not frozen."
    )


if int(
    saved_protocol[
        "model_count"
    ]
) != 3:

    raise RuntimeError(
        "Cycle25 protocol model count "
        "is not 3."
    )


if int(
    saved_protocol[
        "temporal_evaluation"
    ][
        "primary_block_count"
    ]
) != 3:

    raise RuntimeError(
        "Cycle25 protocol primary temporal "
        "block count is not 3."
    )


if saved_protocol[
    "data_isolation_at_protocol_freeze"
][
    "cycle25_dataset_content_read"
]:

    raise RuntimeError(
        "Cycle25 protocol incorrectly "
        "reports Cycle25 content access."
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "CYCLE25 FROZEN EVALUATION DESIGN"
)

print(
    "===================================="
)


print(
    "Frozen models:",
    3
)

print(
    "Feature count:",
    25
)

print(
    "Primary temporal blocks:",
    3
)

print(
    "Pooled secondary summary:",
    True
)

print(
    "Primary metric:",
    "TSS"
)

print(
    "PR-AUC implementation:",
    "average_precision_score"
)

print(
    "Bootstrap unit:",
    "ENTITY_GROUP"
)

print(
    "Bootstrap replicates/evaluation:",
    2000
)

print(
    "Bootstrap CI:",
    "95% percentile"
)

print(
    "Calibration bins:",
    "10 fixed-width"
)


print(
    "\nTemporal blocks:"
)


for block in TEMPORAL_BLOCKS:

    role = (
        "PRIMARY"
        if block[
            "primary_block"
        ]
        else
        "SECONDARY_POOLED"
    )


    print(
        f"{block['block_id']}: "
        f"{block['start_utc']} -> "
        f"{block['end_exclusive_utc']} "
        f"[{role}]"
    )


print(
    "\n===================================="
)

print(
    "CYCLE25 ISOLATION"
)

print(
    "===================================="
)


print(
    "Cycle25 dataset content read:",
    False
)

print(
    "Cycle25 hash recomputed:",
    False
)

print(
    "Cycle25 labels inspected:",
    False
)

print(
    "Cycle25 block counts seen:",
    False
)

print(
    "Cycle25 predictions generated:",
    False
)

print(
    "Cycle25 metrics computed:",
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
    "Thresholds changed:",
    False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_MODEL_REGISTRY_FILE
)

print(
    OUTPUT_BLOCKS_FILE
)

print(
    OUTPUT_PROTOCOL_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 8.1 COMPLETE — CYCLE25 PROTOCOL FROZEN"
)

print(
    "===================================="
)