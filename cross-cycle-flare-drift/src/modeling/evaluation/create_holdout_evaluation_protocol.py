from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

SELECTED_MODELS_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "selected_models_contract.json"
)

SELECTED_MODELS_MANIFEST_FILE = Path(
    "artifacts/models/selection/manifests/"
    "frozen_selected_models.csv"
)

SELECTION_AUDIT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "family_selection_audit.json"
)

PREPROCESSING_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "preprocessing_contract.json"
)

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "model_feature_contract.json"
)

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/"
    "cycle24_split_contract.json"
)

OUTPUT_PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_evaluation_protocol.json"
)

OUTPUT_MODEL_REGISTRY_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "holdout_frozen_model_registry.csv"
)


# ============================================================
# FROZEN HOLDOUT INFORMATION
# ============================================================

EXPECTED_HOLDOUT_ROWS = 13072
EXPECTED_HOLDOUT_POSITIVES = 527
EXPECTED_HOLDOUT_NEGATIVES = 12545
EXPECTED_HOLDOUT_GROUPS = 176

EXPECTED_HOLDOUT_SHA256 = (
    "8abb3929febcde7d778c4781ac890ff5e"
    "658933c2ef2a06aa19da085e7e255fc"
)

BOOTSTRAP_SEED = 24025
BOOTSTRAP_VALID_REPLICATES = 2000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path):

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
    "MODEL CHAPTER 7.1 — FREEZE HOLDOUT EVALUATION PROTOCOL"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    SELECTED_MODELS_CONTRACT_FILE,
    SELECTED_MODELS_MANIFEST_FILE,
    SELECTION_AUDIT_FILE,
    PREPROCESSING_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    SPLIT_CONTRACT_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# PROTECT OUTPUTS
# ============================================================

for output_path in [
    OUTPUT_PROTOCOL_FILE,
    OUTPUT_MODEL_REGISTRY_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Frozen protocol output already exists: "
            f"{output_path}\n"
            "Refusing to overwrite holdout protocol."
        )


# ============================================================
# LOAD DEVELOPMENT-FREEZE CONTRACTS
# ============================================================

selected_models_contract = load_json(
    SELECTED_MODELS_CONTRACT_FILE
)

selection_audit = load_json(
    SELECTION_AUDIT_FILE
)

preprocessing_contract = load_json(
    PREPROCESSING_CONTRACT_FILE
)

feature_contract = load_json(
    FEATURE_CONTRACT_FILE
)

split_contract = load_json(
    SPLIT_CONTRACT_FILE
)


# ============================================================
# VERIFY FROZEN DEVELOPMENT STATE
# ============================================================

if selected_models_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Selected-model contract is not FROZEN."
    )


if selected_models_contract.get(
    "development_stage"
) != "MODEL_SELECTION_COMPLETE":

    raise RuntimeError(
        "Model-selection development stage "
        "is not complete."
    )


if selection_audit.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "Independent family-selection audit "
        "did not PASS."
    )


if preprocessing_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Preprocessing contract is not FROZEN."
    )


if feature_contract.get(
    "status"
) != "VERIFIED":

    raise RuntimeError(
        "Model feature contract is not VERIFIED."
    )


if split_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 split contract is not FROZEN."
    )


print(
    "Selected models contract: FROZEN"
)

print(
    "Independent selection audit: PASS"
)

print(
    "Preprocessing contract: FROZEN"
)

print(
    "Feature contract: VERIFIED"
)

print(
    "Cycle24 split contract: FROZEN"
)


# ============================================================
# LOAD FROZEN SELECTED-MODEL MANIFEST
# ============================================================

selected_models = pd.read_csv(
    SELECTED_MODELS_MANIFEST_FILE,
    keep_default_na=False,
)


if len(
    selected_models
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen selected models."
    )


expected_families = {
    "logistic",
    "random_forest",
    "xgboost",
}


if set(
    selected_models[
        "family"
    ].astype(str)
) != expected_families:

    raise RuntimeError(
        "Frozen selected-model families "
        "are not exactly LR, RF, XGBoost."
    )


if selected_models[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate selected candidate detected."
    )


# ============================================================
# VERIFY MODEL ARTIFACT HASHES
# ============================================================

registry_rows = []


print(
    "\n===================================="
)

print(
    "VERIFY FROZEN MODEL ARTIFACTS"
)

print(
    "===================================="
)


for _, row in selected_models.iterrows():

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


    expected_hash = str(
        row[
            "model_sha256"
        ]
    )


    threshold = float(
        row[
            "selected_threshold"
        ]
    )


    if not model_file.exists():

        raise FileNotFoundError(
            f"{candidate_id}: frozen model "
            f"artifact missing: {model_file}"
        )


    actual_hash = sha256_file(
        model_file
    )


    if actual_hash != expected_hash:

        raise RuntimeError(
            f"{candidate_id}: frozen model "
            "artifact hash mismatch."
        )


    registry_rows.append(
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
                actual_hash,

            "frozen_threshold":
                threshold,

            "threshold_source":
                "CYCLE24_VALIDATION",

            "training_scope":
                "CYCLE24_TRAIN_ONLY",

            "holdout_tuning_allowed":
                False,

            "cycle25_tuning_allowed":
                False,

            "model_retraining_allowed":
                False,

            "threshold_reoptimization_allowed":
                False,
        }
    )


    print(
        f"{family}: "
        f"{candidate_id} | "
        f"threshold={threshold:.10g} | "
        f"model_hash=PASS"
    )


frozen_registry = pd.DataFrame(
    registry_rows
)


# ============================================================
# VERIFY EXPECTED SELECTED MODELS
# ============================================================

expected_selected = {

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


for family, expected in expected_selected.items():

    row = frozen_registry[
        frozen_registry[
            "family"
        ]
        ==
        family
    ]


    if len(row) != 1:

        raise RuntimeError(
            f"{family}: expected exactly one "
            "frozen selected model."
        )


    row = row.iloc[
        0
    ]


    if (
        str(
            row[
                "candidate_id"
            ]
        )
        !=
        expected[
            "candidate_id"
        ]
    ):

        raise RuntimeError(
            f"{family}: frozen candidate differs "
            "from Chapter 6 selection."
        )


    if abs(
        float(
            row[
                "frozen_threshold"
            ]
        )
        -
        expected[
            "threshold"
        ]
    ) > 1e-9:

        raise RuntimeError(
            f"{family}: frozen threshold differs "
            "from Chapter 6 selection."
        )


print(
    "\nFrozen winner identities: PASS"
)


# ============================================================
# SAVE FROZEN HOLDOUT MODEL REGISTRY
# ============================================================

OUTPUT_MODEL_REGISTRY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_PROTOCOL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


frozen_registry.to_csv(
    OUTPUT_MODEL_REGISTRY_FILE,
    index=False,
)


registry_hash = sha256_file(
    OUTPUT_MODEL_REGISTRY_FILE
)


# ============================================================
# DEFINE HOLDOUT METRICS
# ============================================================

threshold_dependent_metrics = [
    "TSS",
    "HSS",
    "F1",
    "sensitivity",
    "specificity",
    "precision",
    "TP",
    "FP",
    "TN",
    "FN",
]


threshold_independent_metrics = [
    "PR_AUC",
    "ROC_AUC",
    "Brier_score",
]


bootstrap_metrics = [
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
# BUILD HOLDOUT EVALUATION PROTOCOL
# ============================================================

protocol = {

    "contract_name":
        "Independent Cycle24 Holdout Evaluation Protocol",

    "version":
        "1.0",

    "status":
        "FROZEN_BEFORE_HOLDOUT_EVALUATION",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "scientific_role":
        (
            "Independent within-Cycle24 test set used "
            "after model-family selection and threshold "
            "selection have been fully frozen."
        ),

    "holdout_dataset":
        {
            "file":
                (
                    "artifacts/splitting/data/"
                    "cycle24_holdout.parquet"
                ),

            "expected_sha256":
                EXPECTED_HOLDOUT_SHA256,

            "expected_rows":
                EXPECTED_HOLDOUT_ROWS,

            "expected_positive":
                EXPECTED_HOLDOUT_POSITIVES,

            "expected_negative":
                EXPECTED_HOLDOUT_NEGATIVES,

            "expected_entity_groups":
                EXPECTED_HOLDOUT_GROUPS,

            "evaluation_only":
                True,
        },

    "model_count":
        3,

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
            in frozen_registry.iterrows()
        ],

    "prediction_rule":
        (
            "Classify positive when predicted probability "
            "is greater than or equal to the frozen "
            "family-specific threshold."
        ),

    "metric_definitions":
        {
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
                    "Average Precision used consistently "
                    "as the frozen PR-AUC implementation "
                    "from Chapter 6."
                ),

            "ROC_AUC":
                "sklearn.metrics.roc_auc_score",

            "Brier_score":
                "sklearn.metrics.brier_score_loss",
        },

    "threshold_dependent_metrics":
        threshold_dependent_metrics,

    "threshold_independent_metrics":
        threshold_independent_metrics,

    "bootstrap":
        {
            "enabled":
                True,

            "unit":
                "ENTITY_GROUP",

            "method":
                (
                    "Connected-entity-group cluster bootstrap. "
                    "Sample exactly G holdout ENTITY_GROUPs "
                    "with replacement, where G is the number "
                    "of unique holdout groups. Every row "
                    "belonging to each sampled group is "
                    "included once per draw of that group."
                ),

            "group_count_expected":
                EXPECTED_HOLDOUT_GROUPS,

            "random_seed":
                BOOTSTRAP_SEED,

            "valid_replicates":
                BOOTSTRAP_VALID_REPLICATES,

            "confidence_level":
                BOOTSTRAP_CONFIDENCE_LEVEL,

            "ci_method":
                "percentile",

            "lower_quantile":
                0.025,

            "upper_quantile":
                0.975,

            "invalid_replicate_rule":
                (
                    "A bootstrap replicate lacking either "
                    "positive or negative target examples "
                    "is invalid for the common metric set "
                    "and must be redrawn. Continue until "
                    "exactly 2000 valid replicates exist."
                ),

            "bootstrap_metrics":
                bootstrap_metrics,

            "thresholds_during_bootstrap":
                "FROZEN_UNCHANGED",

            "models_during_bootstrap":
                "FROZEN_UNCHANGED",
        },

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

            "per_bin_outputs":
                [
                    "row_count",
                    "positive_count",
                    "mean_predicted_probability",
                    "observed_event_rate",
                ],

            "recalibration_allowed":
                False,

            "calibration_outputs_are_diagnostic":
                True,
        },

    "prediction_artifact_requirements":
        {
            "preserve_target":
                True,

            "preserve_ENTITY_GROUP":
                True,

            "preserve_HARPNUM":
                True,

            "preserve_row_identity":
                True,

            "probability_columns":
                3,

            "thresholded_prediction_columns":
                3,

            "reason":
                (
                    "Connected ENTITY_GROUP identifiers must "
                    "remain available for group-bootstrap "
                    "confidence intervals and leakage-aware "
                    "evaluation."
                ),
        },

    "prohibited_actions":
        [
            "retrain_selected_models",
            "refit_logistic_scaler",
            "change_features",
            "change_preprocessing",
            "change_hyperparameters",
            "replace_selected_candidate",
            "reoptimize_threshold_on_holdout",
            "select_model_using_holdout",
            "calibrate_probabilities_using_holdout_for_future_tuning",
            "use_cycle25_to_modify_holdout_analysis",
        ],

    "post_holdout_policy":
        {
            "poor_holdout_performance":
                (
                    "Report as observed. Do not repair or "
                    "retune the frozen development pipeline."
                ),

            "strong_holdout_performance":
                (
                    "Report as observed. Do not alter the "
                    "pipeline before Cycle25 evaluation."
                ),

            "cycle25_next_stage":
                (
                    "Use the same frozen models, features, "
                    "preprocessing, and thresholds unchanged."
                ),
        },

    "data_isolation_at_protocol_freeze":
        {
            "holdout_data_read":
                False,

            "holdout_predictions_generated":
                False,

            "holdout_metrics_seen":
                False,

            "cycle25_read":
                False,

            "model_retraining":
                False,

            "threshold_reoptimization":
                False,
        },

    "artifacts":
        {
            "frozen_holdout_model_registry":
                {
                    "file":
                        str(
                            OUTPUT_MODEL_REGISTRY_FILE
                        ),

                    "sha256":
                        registry_hash,
                },
        },

    "provenance":
        {
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

            "selected_models_manifest":
                {
                    "file":
                        str(
                            SELECTED_MODELS_MANIFEST_FILE
                        ),

                    "sha256":
                        sha256_file(
                            SELECTED_MODELS_MANIFEST_FILE
                        ),
                },

            "selection_audit":
                {
                    "file":
                        str(
                            SELECTION_AUDIT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            SELECTION_AUDIT_FILE
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

            "cycle24_split_contract":
                {
                    "file":
                        str(
                            SPLIT_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            SPLIT_CONTRACT_FILE
                        ),
                },
        },

    "immutability_rule":
        (
            "After this protocol is created, the Cycle24 "
            "holdout may be opened only for evaluation. "
            "The three frozen TRAIN-only models, their "
            "preprocessing, feature order, and validation-"
            "derived thresholds must remain unchanged. "
            "Holdout outcomes must not trigger model, "
            "threshold, feature, preprocessing, or candidate "
            "changes."
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
# VERIFY OUTPUT
# ============================================================

saved_protocol = load_json(
    OUTPUT_PROTOCOL_FILE
)


if saved_protocol.get(
    "status"
) != "FROZEN_BEFORE_HOLDOUT_EVALUATION":

    raise RuntimeError(
        "Holdout evaluation protocol was not frozen."
    )


if int(
    saved_protocol[
        "model_count"
    ]
) != 3:

    raise RuntimeError(
        "Frozen holdout protocol model count "
        "is not 3."
    )


if saved_protocol[
    "data_isolation_at_protocol_freeze"
][
    "holdout_data_read"
]:

    raise RuntimeError(
        "Holdout protocol incorrectly reports "
        "holdout access."
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "HOLDOUT EVALUATION DESIGN"
)

print(
    "===================================="
)


print(
    "Selected models:",
    3
)

print(
    "Holdout expected rows:",
    EXPECTED_HOLDOUT_ROWS
)

print(
    "Holdout expected positives:",
    EXPECTED_HOLDOUT_POSITIVES
)

print(
    "Holdout expected negatives:",
    EXPECTED_HOLDOUT_NEGATIVES
)

print(
    "Holdout expected ENTITY_GROUPs:",
    EXPECTED_HOLDOUT_GROUPS
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
    "Bootstrap valid replicates:",
    BOOTSTRAP_VALID_REPLICATES
)

print(
    "Bootstrap seed:",
    BOOTSTRAP_SEED
)

print(
    "Bootstrap CI:",
    "95% percentile"
)

print(
    "Calibration bins:",
    "10 fixed-width bins"
)


print(
    "\n===================================="
)

print(
    "DATA ISOLATION"
)

print(
    "===================================="
)


print(
    "Cycle24 HOLDOUT read:",
    False
)

print(
    "Holdout predictions generated:",
    False
)

print(
    "Holdout metrics seen:",
    False
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
    OUTPUT_PROTOCOL_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 7.1 COMPLETE — HOLDOUT PROTOCOL FROZEN"
)

print(
    "===================================="
)