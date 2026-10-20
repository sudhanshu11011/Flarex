from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CYCLE25_PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_evaluation_protocol.json"
)

CYCLE25_PREDICTION_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_prediction_contract.json"
)

CYCLE25_METRICS_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "cycle25_metrics.csv"
)

CYCLE25_METRICS_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_metrics_contract.json"
)

CYCLE25_CI_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "cycle25_bootstrap_ci.csv"
)

CYCLE25_BOOTSTRAP_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_bootstrap_contract.json"
)

DRIFT_SUMMARY_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "cross_cycle_predictive_drift.csv"
)

DRIFT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cross_cycle_predictive_drift_contract.json"
)

AUDIT_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_evaluation_audit.csv"
)

AUDIT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_evaluation_audit.json"
)

CYCLE24_RESULTS_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_results_contract.json"
)

OUTPUT_MANIFEST_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "frozen_cycle25_predictive_results.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_results_contract.json"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_SCOPES = [
    "C25_B1_2020_2021",
    "C25_B2_2022_2023",
    "C25_B3_2024_2025",
    "C25_ALL_2020_2025",
]

EXPECTED_MODELS = {
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
}

EXPECTED_RESULT_ROWS = 12
EXPECTED_TSS_COMPARISONS = 12


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


def get_single_row(
    dataframe,
    **conditions,
):

    subset = dataframe.copy()

    for column, value in conditions.items():

        subset = subset[
            subset[
                column
            ].astype(str)
            ==
            str(value)
        ]

    if len(
        subset
    ) != 1:

        raise RuntimeError(
            "Expected exactly one row for "
            f"conditions: {conditions}"
        )

    return subset.iloc[
        0
    ]


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 8.7 — FREEZE FINAL CYCLE25 PREDICTIVE RESULTS"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    CYCLE25_PROTOCOL_FILE,
    CYCLE25_PREDICTION_CONTRACT_FILE,
    CYCLE25_METRICS_FILE,
    CYCLE25_METRICS_CONTRACT_FILE,
    CYCLE25_CI_FILE,
    CYCLE25_BOOTSTRAP_CONTRACT_FILE,
    DRIFT_SUMMARY_FILE,
    DRIFT_CONTRACT_FILE,
    AUDIT_FILE,
    AUDIT_CONTRACT_FILE,
    CYCLE24_RESULTS_CONTRACT_FILE,
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
    OUTPUT_MANIFEST_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Frozen output already exists: {path}\n"
            "Refusing to overwrite final "
            "Cycle25 predictive results."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

protocol = load_json(
    CYCLE25_PROTOCOL_FILE
)

prediction_contract = load_json(
    CYCLE25_PREDICTION_CONTRACT_FILE
)

metrics_contract = load_json(
    CYCLE25_METRICS_CONTRACT_FILE
)

bootstrap_contract = load_json(
    CYCLE25_BOOTSTRAP_CONTRACT_FILE
)

drift_contract = load_json(
    DRIFT_CONTRACT_FILE
)

audit_contract = load_json(
    AUDIT_CONTRACT_FILE
)

cycle24_contract = load_json(
    CYCLE24_RESULTS_CONTRACT_FILE
)


# ============================================================
# STATUS VERIFICATION
# ============================================================

if protocol.get(
    "status"
) != "FROZEN_BEFORE_CYCLE25_EVALUATION":

    raise RuntimeError(
        "Cycle25 protocol is not frozen."
    )


if prediction_contract.get(
    "status"
) != "FROZEN_CYCLE25_PREDICTIONS":

    raise RuntimeError(
        "Cycle25 predictions are not frozen."
    )


if metrics_contract.get(
    "status"
) != "FROZEN_CYCLE25_POINT_METRICS":

    raise RuntimeError(
        "Cycle25 point metrics are not frozen."
    )


if bootstrap_contract.get(
    "status"
) != "FROZEN_CYCLE25_BOOTSTRAP_CI":

    raise RuntimeError(
        "Cycle25 bootstrap CIs are not frozen."
    )


if drift_contract.get(
    "status"
) != "FROZEN_CROSS_CYCLE_PREDICTIVE_DRIFT":

    raise RuntimeError(
        "Cross-cycle predictive drift "
        "results are not frozen."
    )


if audit_contract.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "Cycle25 independent audit "
        "did not PASS."
    )


if not audit_contract.get(
    "overall_cycle25_evaluation_audit_pass",
    False,
):

    raise RuntimeError(
        "Cycle25 overall audit "
        "PASS flag is False."
    )


if cycle24_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 baseline is not frozen."
    )


print(
    "Cycle25 protocol: VERIFIED"
)

print(
    "Cycle25 predictions: FROZEN"
)

print(
    "Cycle25 point metrics: FROZEN"
)

print(
    "Cycle25 bootstrap CIs: FROZEN"
)

print(
    "Cross-cycle predictive drift: FROZEN"
)

print(
    "Independent Cycle25 audit: PASS"
)

print(
    "Cycle24 reference baseline: FROZEN"
)


# ============================================================
# HASH VERIFICATION
# ============================================================

metrics_hash = sha256_file(
    CYCLE25_METRICS_FILE
)


if (
    metrics_hash
    !=
    metrics_contract[
        "artifacts"
    ][
        "cycle25_metrics"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle25 metrics hash mismatch."
    )


ci_hash = sha256_file(
    CYCLE25_CI_FILE
)


if (
    ci_hash
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
        "Cycle25 bootstrap CI hash mismatch."
    )


drift_hash = sha256_file(
    DRIFT_SUMMARY_FILE
)


if (
    drift_hash
    !=
    drift_contract[
        "artifacts"
    ][
        "predictive_drift_summary"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cross-cycle predictive-drift "
        "summary hash mismatch."
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
        "Cycle25 audit-manifest hash mismatch."
    )


print(
    "Final predictive-result hash chain:",
    "PASS"
)


# ============================================================
# LOAD FINAL TABLES
# ============================================================

metrics = pd.read_csv(
    CYCLE25_METRICS_FILE,
    keep_default_na=False,
)

ci = pd.read_csv(
    CYCLE25_CI_FILE,
    keep_default_na=False,
)

drift = pd.read_csv(
    DRIFT_SUMMARY_FILE,
    keep_default_na=False,
)

audit = pd.read_csv(
    AUDIT_FILE,
    keep_default_na=False,
)


# ============================================================
# STRUCTURE VERIFICATION
# ============================================================

if len(
    metrics
) != EXPECTED_RESULT_ROWS:

    raise RuntimeError(
        "Expected exactly 12 Cycle25 "
        "point-result rows."
    )


if set(
    metrics[
        "scope_id"
    ].astype(str)
) != set(
    EXPECTED_SCOPES
):

    raise RuntimeError(
        "Cycle25 scope set mismatch."
    )


if set(
    metrics[
        "candidate_id"
    ].astype(str)
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Cycle25 model set mismatch."
    )


if metrics.duplicated(
    subset=[
        "scope_id",
        "candidate_id",
    ]
).any():

    raise RuntimeError(
        "Duplicate Cycle25 result row."
    )


if len(
    audit
) != EXPECTED_RESULT_ROWS:

    raise RuntimeError(
        "Expected exactly 12 independent "
        "audit rows."
    )


if not audit[
    "pass"
].astype(
    bool
).all():

    raise RuntimeError(
        "Not all Cycle25 point-result "
        "audit rows passed."
    )


# ============================================================
# BUILD FINAL RESULT MANIFEST
# ============================================================

frozen_rows = []


print(
    "\n===================================="
)

print(
    "FINAL CYCLE25 PREDICTIVE RESULTS"
)

print(
    "===================================="
)


for _, point in metrics.iterrows():

    scope_id = str(
        point[
            "scope_id"
        ]
    )

    family = str(
        point[
            "family"
        ]
    )

    candidate_id = str(
        point[
            "candidate_id"
        ]
    )


    tss_ci = get_single_row(
        ci,
        scope_id=scope_id,
        candidate_id=candidate_id,
        metric="tss",
    )


    tss_drift = get_single_row(
        drift,
        scope_id=scope_id,
        candidate_id=candidate_id,
        metric="tss",
    )


    if not np.isclose(
        float(
            point[
                "tss"
            ]
        ),
        float(
            tss_ci[
                "point_estimate"
            ]
        ),
        rtol=0.0,
        atol=1e-12,
    ):

        raise RuntimeError(
            f"{scope_id}/{candidate_id}: "
            "TSS differs between point "
            "metric and CI table."
        )


    if not np.isclose(
        float(
            point[
                "tss"
            ]
        ),
        float(
            tss_drift[
                "cycle25_point"
            ]
        ),
        rtol=0.0,
        atol=1e-12,
    ):

        raise RuntimeError(
            f"{scope_id}/{candidate_id}: "
            "TSS differs between point "
            "metric and drift table."
        )


    frozen_rows.append(
        {
            "scope_id":
                scope_id,

            "scope_type":
                str(
                    point[
                        "scope_type"
                    ]
                ),

            "family":
                family,

            "candidate_id":
                candidate_id,

            "frozen_threshold":
                float(
                    point[
                        "frozen_threshold"
                    ]
                ),

            "rows":
                int(
                    point[
                        "rows"
                    ]
                ),

            "positive":
                int(
                    point[
                        "positive"
                    ]
                ),

            "negative":
                int(
                    point[
                        "negative"
                    ]
                ),

            "positive_prevalence":
                float(
                    point[
                        "positive_prevalence"
                    ]
                ),

            "entity_groups":
                int(
                    point[
                        "entity_groups"
                    ]
                ),

            "tp":
                int(
                    point[
                        "tp"
                    ]
                ),

            "fp":
                int(
                    point[
                        "fp"
                    ]
                ),

            "tn":
                int(
                    point[
                        "tn"
                    ]
                ),

            "fn":
                int(
                    point[
                        "fn"
                    ]
                ),

            "tss":
                float(
                    point[
                        "tss"
                    ]
                ),

            "tss_ci_lower_95":
                float(
                    tss_ci[
                        "ci_lower_2p5"
                    ]
                ),

            "tss_ci_upper_95":
                float(
                    tss_ci[
                        "ci_upper_97p5"
                    ]
                ),

            "hss":
                float(
                    point[
                        "hss"
                    ]
                ),

            "f1":
                float(
                    point[
                        "f1"
                    ]
                ),

            "sensitivity":
                float(
                    point[
                        "sensitivity"
                    ]
                ),

            "specificity":
                float(
                    point[
                        "specificity"
                    ]
                ),

            "precision":
                float(
                    point[
                        "precision"
                    ]
                ),

            "pr_auc":
                float(
                    point[
                        "pr_auc"
                    ]
                ),

            "roc_auc":
                float(
                    point[
                        "roc_auc"
                    ]
                ),

            "brier_score":
                float(
                    point[
                        "brier_score"
                    ]
                ),

            "cycle24_holdout_tss":
                float(
                    tss_drift[
                        "cycle24_holdout_point"
                    ]
                ),

            "delta_tss_cycle25_minus_cycle24":
                float(
                    tss_drift[
                        "delta_cycle25_minus_cycle24"
                    ]
                ),

            "delta_tss_ci_lower_95":
                float(
                    tss_drift[
                        "delta_ci_lower_2p5"
                    ]
                ),

            "delta_tss_ci_upper_95":
                float(
                    tss_drift[
                        "delta_ci_upper_97p5"
                    ]
                ),

            "predictive_change_classification":
                str(
                    tss_drift[
                        "predictive_change_classification"
                    ]
                ),

            "equivalence_claimed":
                False,

            "model_retrained":
                False,

            "threshold_reoptimized":
                False,

            "preprocessing_refit":
                False,

            "result_frozen":
                True,
        }
    )


    print(
        f"{scope_id} | "
        f"{candidate_id}: "
        f"TSS={float(point['tss']):.6f} "
        f"[{float(tss_ci['ci_lower_2p5']):.6f}, "
        f"{float(tss_ci['ci_upper_97p5']):.6f}] | "
        f"delta={float(tss_drift['delta_cycle25_minus_cycle24']):+.6f} "
        f"[{float(tss_drift['delta_ci_lower_2p5']):.6f}, "
        f"{float(tss_drift['delta_ci_upper_97p5']):.6f}] | "
        f"{tss_drift['predictive_change_classification']}"
    )


frozen_results = pd.DataFrame(
    frozen_rows
)


if len(
    frozen_results
) != EXPECTED_RESULT_ROWS:

    raise RuntimeError(
        "Final frozen result manifest "
        "must contain exactly 12 rows."
    )


# ============================================================
# VERIFY TSS CLASSIFICATIONS
# ============================================================

classification_counts = (
    frozen_results[
        "predictive_change_classification"
    ]
    .value_counts()
    .to_dict()
)


expected_no_change = int(
    classification_counts.get(
        "NO_MEASURABLE_CHANGE_AT_95CI",
        0,
    )
)


if expected_no_change != 12:

    raise RuntimeError(
        "Expected all 12 frozen TSS "
        "comparisons to have "
        "NO_MEASURABLE_CHANGE_AT_95CI "
        "based on audited Chapter 8.5 output."
    )


print(
    "\nTSS classifications:",
    "12 / 12 NO_MEASURABLE_CHANGE_AT_95CI"
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
# CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Frozen Final Cycle25 Predictive Evaluation Results",

    "version":
        "1.0",

    "status":
        "FROZEN",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_8",

    "evaluation_stage":
        "CYCLE25_PREDICTIVE_EVALUATION_COMPLETE",

    "scientific_role":
        (
            "Frozen cross-cycle predictive evaluation "
            "of the exact Cycle24-trained models on "
            "predeclared Cycle25 temporal blocks and "
            "the secondary pooled Cycle25 summary."
        ),

    "primary_metric":
        "TSS",

    "primary_temporal_blocks":
        [
            "C25_B1_2020_2021",
            "C25_B2_2022_2023",
            "C25_B3_2024_2025",
        ],

    "secondary_pooled_scope":
        "C25_ALL_2020_2025",

    "model_count":
        3,

    "result_rows":
        12,

    "cross_family_winner_declared":
        False,

    "predictive_change_summary":
        {
            "total_tss_comparisons":
                12,

            "measurable_decrease":
                int(
                    classification_counts.get(
                        "MEASURABLE_DECREASE",
                        0,
                    )
                ),

            "measurable_increase":
                int(
                    classification_counts.get(
                        "MEASURABLE_INCREASE",
                        0,
                    )
                ),

            "no_measurable_change_at_95ci":
                expected_no_change,

            "equivalence_claimed":
                False,

            "interpretation":
                (
                    "Across all 12 predeclared "
                    "Cycle24-holdout versus Cycle25 "
                    "TSS comparisons, the 95% "
                    "bootstrap delta interval contains "
                    "zero. Therefore no statistically "
                    "measurable TSS difference is "
                    "identified at this uncertainty "
                    "level. This is not a formal "
                    "equivalence or proof-of-stability "
                    "claim."
                ),
        },

    "bootstrap":
        {
            "unit":
                "ENTITY_GROUP",

            "valid_replicates_per_scope":
                2000,

            "confidence_level":
                0.95,

            "ci_method":
                "percentile",
        },

    "result_manifest":
        {
            "file":
                str(
                    OUTPUT_MANIFEST_FILE
                ),

            "sha256":
                manifest_hash,

            "rows":
                12,
        },

    "development_integrity":
        {
            "cycle24_models_retrained":
                False,

            "cycle25_training_performed":
                False,

            "preprocessing_refit":
                False,

            "feature_set_changed":
                False,

            "thresholds_reoptimized":
                False,

            "temporal_blocks_changed":
                False,

            "probability_recalibration":
                False,

            "candidate_replacement":
                False,
        },

    "next_scientific_boundary":
        {
            "distribution_shift_analysis":
                "NOT_YET_PERFORMED",

            "within_cycle24_shap_null":
                "NOT_YET_PERFORMED",

            "cross_cycle_shap_analysis":
                "NOT_YET_PERFORMED",

            "predictive_results_must_not_determine_shap_conclusion":
                True,
        },

    "interpretation_constraints":
        {
            "no_measurable_change_not_equivalence":
                True,

            "no_measurable_change_not_proof_of_stability":
                True,

            "predictive_skill_and_explanation_stability_are_separate":
                True,

            "cycle25_treated_as_temporal_domain_shift":
                True,

            "cycle25_not_automatically_ood":
                True,

            "shap_not_causal_solar_physics":
                True,
        },

    "verification":
        {
            "chapter_8_1_protocol":
                "PASS",

            "chapter_8_2_predictions":
                "PASS",

            "chapter_8_3_point_metrics":
                "PASS",

            "chapter_8_4_bootstrap":
                "PASS",

            "chapter_8_5_predictive_drift":
                "PASS",

            "chapter_8_6_independent_audit":
                "PASS",
        },

    "artifacts":
        {
            "cycle25_metrics":
                {
                    "file":
                        str(
                            CYCLE25_METRICS_FILE
                        ),

                    "sha256":
                        metrics_hash,
                },

            "cycle25_bootstrap_ci":
                {
                    "file":
                        str(
                            CYCLE25_CI_FILE
                        ),

                    "sha256":
                        ci_hash,
                },

            "cross_cycle_predictive_drift":
                {
                    "file":
                        str(
                            DRIFT_SUMMARY_FILE
                        ),

                    "sha256":
                        drift_hash,
                },

            "independent_cycle25_audit":
                {
                    "file":
                        str(
                            AUDIT_FILE
                        ),

                    "sha256":
                        audit_hash,
                },

            "frozen_cycle25_result_manifest":
                {
                    "file":
                        str(
                            OUTPUT_MANIFEST_FILE
                        ),

                    "sha256":
                        manifest_hash,
                },
        },

    "data_access_at_freeze":
        {
            "raw_cycle25_dataset_read":
                False,

            "raw_cycle24_holdout_read":
                False,

            "prediction_parquet_read":
                False,

            "bootstrap_parquet_read":
                False,

            "model_files_loaded":
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
            "cycle25_protocol":
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

            "cycle25_prediction_contract":
                {
                    "file":
                        str(
                            CYCLE25_PREDICTION_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CYCLE25_PREDICTION_CONTRACT_FILE
                        ),
                },

            "cycle25_metrics_contract":
                {
                    "file":
                        str(
                            CYCLE25_METRICS_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CYCLE25_METRICS_CONTRACT_FILE
                        ),
                },

            "cycle25_bootstrap_contract":
                {
                    "file":
                        str(
                            CYCLE25_BOOTSTRAP_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CYCLE25_BOOTSTRAP_CONTRACT_FILE
                        ),
                },

            "cross_cycle_drift_contract":
                {
                    "file":
                        str(
                            DRIFT_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            DRIFT_CONTRACT_FILE
                        ),
                },

            "cycle25_audit_contract":
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

            "cycle24_reference_contract":
                {
                    "file":
                        str(
                            CYCLE24_RESULTS_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CYCLE24_RESULTS_CONTRACT_FILE
                        ),
                },
        },

    "immutability_rule":
        (
            "Cycle25 predictive evaluation is final "
            "and frozen. These results must not be "
            "used to retrain models, alter features, "
            "modify preprocessing, reoptimize "
            "thresholds, change temporal blocks, or "
            "select a replacement model. Subsequent "
            "distribution-shift and explanation-drift "
            "analyses must preserve this frozen "
            "predictive pipeline."
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
# VERIFY FINAL SAVE
# ============================================================

saved_contract = load_json(
    OUTPUT_CONTRACT_FILE
)


if saved_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Final Cycle25 result contract "
        "was not frozen."
    )


if saved_contract.get(
    "evaluation_stage"
) != "CYCLE25_PREDICTIVE_EVALUATION_COMPLETE":

    raise RuntimeError(
        "Cycle25 predictive evaluation "
        "was not marked complete."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "FINAL CYCLE25 FREEZE"
)

print(
    "===================================="
)


print(
    "Frozen result rows:",
    12
)

print(
    "Models:",
    3
)

print(
    "Primary temporal blocks:",
    3
)

print(
    "Secondary pooled scope:",
    1
)

print(
    "TSS comparisons:",
    12
)

print(
    "Measurable TSS decrease:",
    0
)

print(
    "Measurable TSS increase:",
    0
)

print(
    "No measurable TSS change at 95% CI:",
    12
)

print(
    "Formal equivalence claimed:",
    False
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
    "NEXT-SCIENCE BOUNDARY"
)

print(
    "===================================="
)


print(
    "Predictive evaluation:",
    "COMPLETE"
)

print(
    "Distribution-shift analysis:",
    "NOT YET"
)

print(
    "Within-Cycle24 SHAP null:",
    "NOT YET"
)

print(
    "Cross-cycle SHAP analysis:",
    "NOT YET"
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
    "MODEL CHAPTER 8.7 COMPLETE — CYCLE25 PREDICTIVE RESULTS FROZEN"
)

print(
    "===================================="
)