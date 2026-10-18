from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

HOLDOUT_PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_evaluation_protocol.json"
)

PREDICTION_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_prediction_contract.json"
)

POINT_METRICS_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "holdout_metrics.csv"
)

POINT_METRICS_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_metrics_contract.json"
)

CALIBRATION_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "holdout_calibration.csv"
)

BOOTSTRAP_CI_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "holdout_bootstrap_ci.csv"
)

BOOTSTRAP_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_bootstrap_contract.json"
)

AUDIT_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "holdout_evaluation_audit.csv"
)

AUDIT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_evaluation_audit.json"
)

SELECTED_MODELS_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "selected_models_contract.json"
)

OUTPUT_MANIFEST_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "frozen_holdout_results.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_results_contract.json"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_ROWS = 13072
EXPECTED_POSITIVES = 527
EXPECTED_NEGATIVES = 12545
EXPECTED_ENTITY_GROUPS = 176

EXPECTED_MODELS = 3
EXPECTED_CI_ROWS = 27


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


def get_ci(
    ci_df,
    candidate_id,
    metric,
):

    row = ci_df[
        (
            ci_df[
                "candidate_id"
            ].astype(str)
            ==
            candidate_id
        )
        &
        (
            ci_df[
                "metric"
            ].astype(str)
            ==
            metric
        )
    ]


    if len(row) != 1:

        raise RuntimeError(
            f"{candidate_id}/{metric}: "
            "expected exactly one CI row."
        )


    row = row.iloc[0]


    return (
        float(
            row[
                "ci_lower_2p5"
            ]
        ),
        float(
            row[
                "ci_upper_97p5"
            ]
        ),
    )


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 7.6 — FREEZE FINAL HOLDOUT RESULTS"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    HOLDOUT_PROTOCOL_FILE,
    PREDICTION_CONTRACT_FILE,
    POINT_METRICS_FILE,
    POINT_METRICS_CONTRACT_FILE,
    CALIBRATION_FILE,
    BOOTSTRAP_CI_FILE,
    BOOTSTRAP_CONTRACT_FILE,
    AUDIT_FILE,
    AUDIT_CONTRACT_FILE,
    SELECTED_MODELS_CONTRACT_FILE,
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
    OUTPUT_MANIFEST_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Frozen output already exists: "
            f"{output_path}\n"
            "Refusing to overwrite final "
            "holdout freeze."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

holdout_protocol = load_json(
    HOLDOUT_PROTOCOL_FILE
)

prediction_contract = load_json(
    PREDICTION_CONTRACT_FILE
)

metrics_contract = load_json(
    POINT_METRICS_CONTRACT_FILE
)

bootstrap_contract = load_json(
    BOOTSTRAP_CONTRACT_FILE
)

audit_contract = load_json(
    AUDIT_CONTRACT_FILE
)

selected_models_contract = load_json(
    SELECTED_MODELS_CONTRACT_FILE
)


# ============================================================
# VERIFY CONTRACT STATES
# ============================================================

if holdout_protocol.get(
    "status"
) != "FROZEN_BEFORE_HOLDOUT_EVALUATION":

    raise RuntimeError(
        "Holdout protocol is not frozen."
    )


if prediction_contract.get(
    "status"
) != "FROZEN_HOLDOUT_PREDICTIONS":

    raise RuntimeError(
        "Holdout predictions are not frozen."
    )


if metrics_contract.get(
    "status"
) != "FROZEN_HOLDOUT_POINT_METRICS":

    raise RuntimeError(
        "Holdout point metrics are not frozen."
    )


if bootstrap_contract.get(
    "status"
) != "FROZEN_HOLDOUT_BOOTSTRAP_CI":

    raise RuntimeError(
        "Holdout bootstrap CI is not frozen."
    )


if audit_contract.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "Independent holdout audit did not PASS."
    )


if not audit_contract.get(
    "overall_holdout_evaluation_audit_pass",
    False,
):

    raise RuntimeError(
        "Independent holdout audit overall "
        "PASS flag is False."
    )


if selected_models_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Selected-model development contract "
        "is not frozen."
    )


print(
    "Holdout protocol: VERIFIED"
)

print(
    "Holdout predictions: FROZEN"
)

print(
    "Holdout point metrics: FROZEN"
)

print(
    "Holdout bootstrap CI: FROZEN"
)

print(
    "Independent holdout audit: PASS"
)

print(
    "Selected models: FROZEN"
)


# ============================================================
# VERIFY HASH CHAIN
# ============================================================

metrics_hash = sha256_file(
    POINT_METRICS_FILE
)


if (
    metrics_hash
    !=
    metrics_contract[
        "artifacts"
    ][
        "holdout_metrics"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Holdout metrics hash mismatch."
    )


calibration_hash = sha256_file(
    CALIBRATION_FILE
)


if (
    calibration_hash
    !=
    metrics_contract[
        "artifacts"
    ][
        "holdout_calibration"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Holdout calibration hash mismatch."
    )


bootstrap_ci_hash = sha256_file(
    BOOTSTRAP_CI_FILE
)


if (
    bootstrap_ci_hash
    !=
    bootstrap_contract[
        "artifacts"
    ][
        "bootstrap_ci_summary"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Holdout bootstrap CI hash mismatch."
    )


audit_hash = sha256_file(
    AUDIT_FILE
)


if (
    audit_hash
    !=
    audit_contract[
        "audit_manifest"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Holdout audit manifest hash mismatch."
    )


print(
    "Final result hash chain:",
    "PASS"
)


# ============================================================
# LOAD FINAL RESULT TABLES
# ============================================================

metrics = pd.read_csv(
    POINT_METRICS_FILE,
    keep_default_na=False,
)

ci = pd.read_csv(
    BOOTSTRAP_CI_FILE,
    keep_default_na=False,
)

calibration = pd.read_csv(
    CALIBRATION_FILE,
    keep_default_na=False,
)

audit = pd.read_csv(
    AUDIT_FILE,
    keep_default_na=False,
)


# ============================================================
# VERIFY TABLE STRUCTURE
# ============================================================

if len(
    metrics
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 holdout result rows."
    )


if len(
    ci
) != EXPECTED_CI_ROWS:

    raise RuntimeError(
        "Expected exactly 27 bootstrap CI rows."
    )


if len(
    calibration
) != 30:

    raise RuntimeError(
        "Expected exactly 30 calibration rows."
    )


if len(
    audit
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 audit rows."
    )


if metrics[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate in holdout metrics."
    )


expected_families = {
    "logistic",
    "random_forest",
    "xgboost",
}


if set(
    metrics[
        "family"
    ].astype(str)
) != expected_families:

    raise RuntimeError(
        "Holdout result families are incorrect."
    )


# ============================================================
# VERIFY POPULATION
# ============================================================

if not (
    metrics[
        "rows"
    ].astype(int)
    ==
    EXPECTED_ROWS
).all():

    raise RuntimeError(
        "Holdout row-count mismatch."
    )


if not (
    metrics[
        "positive"
    ].astype(int)
    ==
    EXPECTED_POSITIVES
).all():

    raise RuntimeError(
        "Holdout positive-count mismatch."
    )


if not (
    metrics[
        "negative"
    ].astype(int)
    ==
    EXPECTED_NEGATIVES
).all():

    raise RuntimeError(
        "Holdout negative-count mismatch."
    )


# ============================================================
# BUILD FINAL FROZEN RESULT MANIFEST
# ============================================================

freeze_rows = []


print(
    "\n===================================="
)

print(
    "FINAL CYCLE24 HOLDOUT RESULTS"
)

print(
    "===================================="
)


for _, row in metrics.iterrows():

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


    # --------------------------------------------------------
    # TSS CI
    # --------------------------------------------------------

    tss_lower, tss_upper = get_ci(
        ci,
        candidate_id,
        "tss",
    )


    # --------------------------------------------------------
    # HSS CI
    # --------------------------------------------------------

    hss_lower, hss_upper = get_ci(
        ci,
        candidate_id,
        "hss",
    )


    # --------------------------------------------------------
    # F1 CI
    # --------------------------------------------------------

    f1_lower, f1_upper = get_ci(
        ci,
        candidate_id,
        "f1",
    )


    # --------------------------------------------------------
    # PR-AUC CI
    # --------------------------------------------------------

    pr_lower, pr_upper = get_ci(
        ci,
        candidate_id,
        "pr_auc",
    )


    # --------------------------------------------------------
    # ROC-AUC CI
    # --------------------------------------------------------

    roc_lower, roc_upper = get_ci(
        ci,
        candidate_id,
        "roc_auc",
    )


    # --------------------------------------------------------
    # Brier CI
    # --------------------------------------------------------

    brier_lower, brier_upper = get_ci(
        ci,
        candidate_id,
        "brier_score",
    )


    freeze_rows.append(
        {
            "family":
                family,

            "candidate_id":
                candidate_id,

            "frozen_threshold":
                float(
                    row[
                        "frozen_threshold"
                    ]
                ),

            "tp":
                int(
                    row[
                        "tp"
                    ]
                ),

            "fp":
                int(
                    row[
                        "fp"
                    ]
                ),

            "tn":
                int(
                    row[
                        "tn"
                    ]
                ),

            "fn":
                int(
                    row[
                        "fn"
                    ]
                ),

            "tss":
                float(
                    row[
                        "tss"
                    ]
                ),

            "tss_ci_lower_95":
                tss_lower,

            "tss_ci_upper_95":
                tss_upper,

            "hss":
                float(
                    row[
                        "hss"
                    ]
                ),

            "hss_ci_lower_95":
                hss_lower,

            "hss_ci_upper_95":
                hss_upper,

            "f1":
                float(
                    row[
                        "f1"
                    ]
                ),

            "f1_ci_lower_95":
                f1_lower,

            "f1_ci_upper_95":
                f1_upper,

            "sensitivity":
                float(
                    row[
                        "sensitivity"
                    ]
                ),

            "specificity":
                float(
                    row[
                        "specificity"
                    ]
                ),

            "precision":
                float(
                    row[
                        "precision"
                    ]
                ),

            "pr_auc":
                float(
                    row[
                        "pr_auc"
                    ]
                ),

            "pr_auc_ci_lower_95":
                pr_lower,

            "pr_auc_ci_upper_95":
                pr_upper,

            "roc_auc":
                float(
                    row[
                        "roc_auc"
                    ]
                ),

            "roc_auc_ci_lower_95":
                roc_lower,

            "roc_auc_ci_upper_95":
                roc_upper,

            "brier_score":
                float(
                    row[
                        "brier_score"
                    ]
                ),

            "brier_ci_lower_95":
                brier_lower,

            "brier_ci_upper_95":
                brier_upper,

            "evaluation_dataset":
                "CYCLE24_HOLDOUT",

            "threshold_source":
                "CYCLE24_VALIDATION",

            "bootstrap_unit":
                "ENTITY_GROUP",

            "bootstrap_valid_replicates":
                2000,

            "model_retrained":
                False,

            "threshold_reoptimized":
                False,

            "cycle25_used":
                False,

            "result_frozen":
                True,
        }
    )


    print(
        f"{candidate_id}: "
        f"TSS={float(row['tss']):.6f} "
        f"[{tss_lower:.6f}, {tss_upper:.6f}] | "
        f"PR_AUC={float(row['pr_auc']):.6f} | "
        f"ROC_AUC={float(row['roc_auc']):.6f} | "
        f"Brier={float(row['brier_score']):.6f}"
    )


frozen_results = pd.DataFrame(
    freeze_rows
)


if len(
    frozen_results
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Frozen result manifest must "
        "contain exactly 3 rows."
    )


# ============================================================
# SAVE FINAL MANIFEST
# ============================================================

OUTPUT_MANIFEST_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


frozen_results.to_csv(
    OUTPUT_MANIFEST_FILE,
    index=False,
)


manifest_hash = sha256_file(
    OUTPUT_MANIFEST_FILE
)


# ============================================================
# BUILD FINAL HOLDOUT RESULT CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Frozen Independent Cycle24 Holdout Results",

    "version":
        "1.0",

    "status":
        "FROZEN",

    "frozen_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_7",

    "evaluation_stage":
        "CYCLE24_HOLDOUT_COMPLETE",

    "scientific_role":
        (
            "Independent within-Cycle24 evaluation "
            "performed after all model-family selection, "
            "preprocessing, hyperparameters, and thresholds "
            "were frozen."
        ),

    "population":
        {
            "rows":
                EXPECTED_ROWS,

            "positive":
                EXPECTED_POSITIVES,

            "negative":
                EXPECTED_NEGATIVES,

            "entity_groups":
                EXPECTED_ENTITY_GROUPS,
        },

    "model_count":
        EXPECTED_MODELS,

    "cross_family_winner_declared":
        False,

    "models_retained_for_cycle25":
        [
            str(candidate_id)
            for candidate_id
            in frozen_results[
                "candidate_id"
            ].tolist()
        ],

    "results":
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

                "frozen_threshold":
                    float(
                        row[
                            "frozen_threshold"
                        ]
                    ),

                "TSS":
                    {
                        "point":
                            float(
                                row[
                                    "tss"
                                ]
                            ),

                        "ci95_lower":
                            float(
                                row[
                                    "tss_ci_lower_95"
                                ]
                            ),

                        "ci95_upper":
                            float(
                                row[
                                    "tss_ci_upper_95"
                                ]
                            ),
                    },

                "HSS":
                    float(
                        row[
                            "hss"
                        ]
                    ),

                "F1":
                    float(
                        row[
                            "f1"
                        ]
                    ),

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
            in frozen_results.iterrows()
        ],

    "uncertainty":
        {
            "bootstrap_unit":
                "ENTITY_GROUP",

            "valid_replicates":
                2000,

            "seed":
                24025,

            "confidence_level":
                0.95,

            "ci_method":
                "percentile",

            "paired_group_draws_across_models":
                True,
        },

    "metric_policy":
        {
            "TSS":
                "TPR - FPR",

            "PR_AUC":
                "sklearn.metrics.average_precision_score",

            "ROC_AUC":
                "sklearn.metrics.roc_auc_score",

            "Brier_score":
                "sklearn.metrics.brier_score_loss",
        },

    "development_integrity":
        {
            "selected_models_changed":
                False,

            "models_retrained":
                False,

            "train_plus_validation_refit":
                False,

            "preprocessing_refit":
                False,

            "features_changed":
                False,

            "hyperparameters_changed":
                False,

            "thresholds_reoptimized":
                False,

            "holdout_used_for_selection":
                False,
        },

    "cycle25_policy":
        {
            "cycle25_used_so_far":
                False,

            "models_must_remain_unchanged":
                True,

            "preprocessing_must_remain_unchanged":
                True,

            "features_must_remain_unchanged":
                True,

            "thresholds_must_remain_unchanged":
                True,

            "retraining_allowed":
                False,

            "threshold_tuning_allowed":
                False,

            "feature_selection_allowed":
                False,

            "preprocessing_refit_allowed":
                False,
        },

    "interpretation_constraints":
        {
            "highest_point_tss_is_not_cross_family_selection":
                True,

            "marginal_ci_overlap_does_not_test_pairwise_superiority":
                True,

            "holdout_results_must_not_trigger_pipeline_changes":
                True,

            "cycle25_is_temporal_test":
                True,
        },

    "verification":
        {
            "chapter_7_1_protocol":
                "PASS",

            "chapter_7_2_predictions":
                "PASS",

            "chapter_7_3_metrics_and_calibration":
                "PASS",

            "chapter_7_4_group_bootstrap":
                "PASS",

            "chapter_7_5_independent_audit":
                "PASS",
        },

    "artifacts":
        {
            "frozen_holdout_manifest":
                {
                    "file":
                        str(
                            OUTPUT_MANIFEST_FILE
                        ),

                    "sha256":
                        manifest_hash,
                },

            "point_metrics":
                {
                    "file":
                        str(
                            POINT_METRICS_FILE
                        ),

                    "sha256":
                        metrics_hash,
                },

            "calibration":
                {
                    "file":
                        str(
                            CALIBRATION_FILE
                        ),

                    "sha256":
                        calibration_hash,
                },

            "bootstrap_ci":
                {
                    "file":
                        str(
                            BOOTSTRAP_CI_FILE
                        ),

                    "sha256":
                        bootstrap_ci_hash,
                },

            "independent_audit":
                {
                    "file":
                        str(
                            AUDIT_FILE
                        ),

                    "sha256":
                        audit_hash,
                },
        },

    "data_access_at_freeze":
        {
            "raw_holdout_dataset_read_by_freeze_script":
                False,

            "prediction_artifact_read_by_freeze_script":
                False,

            "model_files_loaded_by_freeze_script":
                False,

            "cycle25_read":
                False,

            "new_model_fitting":
                False,

            "new_threshold_optimization":
                False,

            "new_bootstrap_sampling":
                False,
        },

    "provenance":
        {
            "holdout_protocol":
                {
                    "file":
                        str(
                            HOLDOUT_PROTOCOL_FILE
                        ),

                    "sha256":
                        sha256_file(
                            HOLDOUT_PROTOCOL_FILE
                        ),
                },

            "prediction_contract":
                {
                    "file":
                        str(
                            PREDICTION_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            PREDICTION_CONTRACT_FILE
                        ),
                },

            "point_metrics_contract":
                {
                    "file":
                        str(
                            POINT_METRICS_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            POINT_METRICS_CONTRACT_FILE
                        ),
                },

            "bootstrap_contract":
                {
                    "file":
                        str(
                            BOOTSTRAP_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            BOOTSTRAP_CONTRACT_FILE
                        ),
                },

            "audit_contract":
                {
                    "file":
                        str(
                            AUDIT_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            AUDIT_CONTRACT_FILE
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
        },

    "immutability_rule":
        (
            "The Cycle24 independent holdout results are "
            "final and frozen. No result from the holdout "
            "may be used to retrain a model, change a "
            "candidate, modify hyperparameters, alter "
            "features or preprocessing, or reoptimize a "
            "decision threshold. The same three frozen "
            "models and validation-derived thresholds must "
            "proceed unchanged to Cycle25 evaluation."
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
# VERIFY FINAL CONTRACT
# ============================================================

saved_contract = load_json(
    OUTPUT_CONTRACT_FILE
)


if saved_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Final holdout result contract "
        "was not frozen."
    )


if saved_contract.get(
    "evaluation_stage"
) != "CYCLE24_HOLDOUT_COMPLETE":

    raise RuntimeError(
        "Holdout evaluation stage "
        "was not marked complete."
    )


if int(
    saved_contract[
        "model_count"
    ]
) != 3:

    raise RuntimeError(
        "Final holdout result model "
        "count is not 3."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "FINAL HOLDOUT FREEZE"
)

print(
    "===================================="
)


print(
    "Models frozen:",
    3
)

print(
    "Point metrics frozen:",
    True
)

print(
    "Calibration diagnostics frozen:",
    True
)

print(
    "ENTITY_GROUP bootstrap CIs frozen:",
    True
)

print(
    "Independent audit:",
    "PASS"
)

print(
    "Cross-family winner declared:",
    False
)


print(
    "\n===================================="
)

print(
    "CYCLE25 HANDOFF POLICY"
)

print(
    "===================================="
)


print(
    "Cycle25 read so far:",
    False
)

print(
    "Model retraining allowed:",
    False
)

print(
    "Threshold changes allowed:",
    False
)

print(
    "Feature changes allowed:",
    False
)

print(
    "Preprocessing refit allowed:",
    False
)


print(
    "\nSaved:"
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
    "MODEL CHAPTER 7.6 COMPLETE — CYCLE24 HOLDOUT RESULTS FROZEN"
)

print(
    "===================================="
)