# ============================================================
# MODEL CHAPTER 10.10.1B
# FREEZE FINAL CROSS-CYCLE SHAP SYNTHESIS PROTOCOL
#
# Purpose:
#   Freeze how the already-computed Chapter-10 evidence
#   will be combined BEFORE producing the final synthesis.
#
# IMPORTANT:
#   This is NOT a preregistration before seeing Cycle25.
#   Cycle25 SHAP results and inferential outputs already exist.
#
#   This protocol prevents an arbitrary post-hoc binary
#   "drift/stable" rule from being invented during synthesis.
#
# Primary inferential evidence tier:
#
#   TIER_0:
#     no OVERALL frozen structure-null exceedance
#
#   TIER_1:
#     OVERALL structure-null exceedance
#     but no OVERALL featurewise BH-FDR signal
#
#   TIER_2:
#     OVERALL structure-null exceedance
#     AND >=1 OVERALL featurewise BH-FDR significant feature
#
# Class-conditional, support/reweighting, correlated-group,
# and temporal results are retained as robustness/context
# annotations rather than being forced into arbitrary binary
# thresholds that were never frozen.
#
# THIS STEP DOES NOT:
#   - compute final model evidence tiers
#   - make a final explanation-drift conclusion
#   - make a final stability conclusion
#   - modify any prior artifact
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json

import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_MODELS = 3

EXPECTED_STRUCTURE_SUMMARY_ROWS = 18
EXPECTED_SUPPORT_ROWS = 288
EXPECTED_GROUP_ROWS = 270
EXPECTED_FDR_SUMMARY_ROWS = 18


MODEL_ORDER = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


PRIMARY_EVIDENCE_TIERS = {

    "TIER_0": {
        "label":
            "NO_GLOBAL_STRUCTURE_NULL_EXCEEDANCE",

        "rule":
            (
                "OVERALL any_null_exceedance == False"
            ),
    },

    "TIER_1": {
        "label":
            "GLOBAL_STRUCTURE_ONLY_EVIDENCE",

        "rule":
            (
                "OVERALL any_null_exceedance == True "
                "AND OVERALL total FDR-significant "
                "featurewise hypotheses == 0"
            ),
    },

    "TIER_2": {
        "label":
            "GLOBAL_STRUCTURE_AND_FEATUREWISE_EVIDENCE",

        "rule":
            (
                "OVERALL any_null_exceedance == True "
                "AND OVERALL total FDR-significant "
                "featurewise hypotheses >= 1"
            ),
    },
}


# ============================================================
# INPUT CONTRACTS
# ============================================================

NULL_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "cross_cycle_shap_structure_null_comparison_contract.json"
)


SUPPORT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "common_support_reweighting_shap_sensitivity_contract.json"
)


GROUP_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "correlated_group_shap_robustness_contract.json"
)


FDR_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "featurewise_cross_cycle_inference_contract.json"
)


INFERENCE_PROTOCOL_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "featurewise_inference_protocol.json"
)


# ============================================================
# EVIDENCE TABLES
# ============================================================

STRUCTURE_SUMMARY_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "cross_cycle_shap_structure_null_summary.csv"
)


SUPPORT_STRUCTURE_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "common_support_reweighting_shap_structure_sensitivity.csv"
)


GROUP_STRUCTURE_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "correlated_group_shap_structure_robustness.csv"
)


FDR_SUMMARY_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "featurewise_cross_cycle_inference_summary.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "final_cross_cycle_shap_synthesis_protocol.json"
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

        return json.load(file)


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
    "MODEL CHAPTER 10.10.1B — "
    "FREEZE FINAL CROSS-CYCLE SHAP SYNTHESIS PROTOCOL"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    NULL_CONTRACT_FILE,
    SUPPORT_CONTRACT_FILE,
    GROUP_CONTRACT_FILE,
    FDR_CONTRACT_FILE,
    INFERENCE_PROTOCOL_FILE,
    STRUCTURE_SUMMARY_FILE,
    SUPPORT_STRUCTURE_FILE,
    GROUP_STRUCTURE_FILE,
    FDR_SUMMARY_FILE,
]


for path in required_files:

    verify_exists(path)


# ============================================================
# OUTPUT PROTECTION
# ============================================================

if OUTPUT_CONTRACT_FILE.exists():

    raise RuntimeError(
        f"Output already exists: {OUTPUT_CONTRACT_FILE}\n"
        "Refusing to overwrite frozen "
        "Chapter 10.10.1B synthesis protocol."
    )


# ============================================================
# VERIFY CONTRACT STATUSES
# ============================================================

null_contract = load_json(
    NULL_CONTRACT_FILE
)


support_contract = load_json(
    SUPPORT_CONTRACT_FILE
)


group_contract = load_json(
    GROUP_CONTRACT_FILE
)


fdr_contract = load_json(
    FDR_CONTRACT_FILE
)


inference_protocol = load_json(
    INFERENCE_PROTOCOL_FILE
)


required_statuses = [
    (
        null_contract.get("status"),
        "FROZEN_CROSS_CYCLE_SHAP_STRUCTURE_NULL_COMPARISON",
    ),
    (
        support_contract.get("status"),
        "FROZEN_COMMON_SUPPORT_REWEIGHTING_SHAP_SENSITIVITY",
    ),
    (
        group_contract.get("status"),
        "FROZEN_CORRELATED_GROUP_SHAP_ROBUSTNESS",
    ),
    (
        fdr_contract.get("status"),
        "FROZEN_FEATUREWISE_CROSS_CYCLE_INFERENCE_BH_FDR",
    ),
    (
        inference_protocol.get("status"),
        "FROZEN_FEATUREWISE_INFERENCE_PROTOCOL",
    ),
]


for actual, expected in required_statuses:

    if actual != expected:

        raise RuntimeError(
            "Required frozen contract missing: "
            f"{expected}"
        )


print(
    "Required frozen evidence contracts:",
    "PASS"
)


# ============================================================
# VERIFY TABLE HASHES AGAINST CONTRACTS
# ============================================================

hash_checks = [
    (
        STRUCTURE_SUMMARY_FILE,
        null_contract[
            "summary_output"
        ][
            "sha256"
        ],
    ),
    (
        SUPPORT_STRUCTURE_FILE,
        support_contract[
            "structure_output"
        ][
            "sha256"
        ],
    ),
    (
        GROUP_STRUCTURE_FILE,
        group_contract[
            "structure_output"
        ][
            "sha256"
        ],
    ),
    (
        FDR_SUMMARY_FILE,
        fdr_contract[
            "summary_output"
        ][
            "sha256"
        ],
    ),
]


for path, expected_hash in hash_checks:

    actual_hash = sha256_file(path)

    if actual_hash != expected_hash:

        raise RuntimeError(
            f"Frozen evidence artifact changed: {path}"
        )


print(
    "Frozen evidence artifact hashes:",
    "PASS"
)


# ============================================================
# LOAD TABLES
# ============================================================

structure_summary = pd.read_csv(
    STRUCTURE_SUMMARY_FILE,
    keep_default_na=False,
)


support_structure = pd.read_csv(
    SUPPORT_STRUCTURE_FILE,
    keep_default_na=False,
)


group_structure = pd.read_csv(
    GROUP_STRUCTURE_FILE,
    keep_default_na=False,
)


fdr_summary = pd.read_csv(
    FDR_SUMMARY_FILE,
    keep_default_na=False,
)


# ============================================================
# INVENTORY CHECKS
# ============================================================

if len(
    structure_summary
) != EXPECTED_STRUCTURE_SUMMARY_ROWS:

    raise RuntimeError(
        "Structure-null summary row-count mismatch."
    )


if len(
    support_structure
) != EXPECTED_SUPPORT_ROWS:

    raise RuntimeError(
        "Support/reweight structure row-count mismatch."
    )


if len(
    group_structure
) != EXPECTED_GROUP_ROWS:

    raise RuntimeError(
        "Correlated-group structure row-count mismatch."
    )


if len(
    fdr_summary
) != EXPECTED_FDR_SUMMARY_ROWS:

    raise RuntimeError(
        "Featurewise FDR summary row-count mismatch."
    )


print(
    "Final evidence table inventories:",
    "PASS"
)


# ============================================================
# REQUIRED SCHEMA CHECKS
# ============================================================

required_structure_columns = {
    "analysis_scope",
    "candidate_id",
    "observed_stratum",
    "metrics_evaluated",
    "metrics_exceeding_null",
    "any_null_exceedance",
    "null_screen_status",
}


required_support_columns = {
    "candidate_id",
    "analysis_scope",
    "scope_key",
    "sensitivity_variant",
    "metric",
    "primary_full_population_value",
    "sensitivity_value",
    "delta_sensitivity_minus_primary",
}


required_group_columns = {
    "candidate_id",
    "analysis_variant",
    "analysis_scope",
    "scope_key",
    "metric",
    "individual_feature_metric_value",
    "group_level_metric_value",
    "group_minus_individual_metric_value",
    "group_level_null_threshold_available",
}


required_fdr_columns = {
    "candidate_id",
    "stratum",
    "quantity",
    "features_tested",
    "fdr_significant_features",
    "significant_feature_names",
    "fdr_q",
    "family_status",
}


schema_checks = [
    (
        "structure_summary",
        structure_summary,
        required_structure_columns,
    ),
    (
        "support_structure",
        support_structure,
        required_support_columns,
    ),
    (
        "group_structure",
        group_structure,
        required_group_columns,
    ),
    (
        "fdr_summary",
        fdr_summary,
        required_fdr_columns,
    ),
]


for name, frame, required_columns in schema_checks:

    missing = (
        required_columns
        -
        set(frame.columns)
    )

    if missing:

        raise RuntimeError(
            f"{name}: missing columns "
            f"{sorted(missing)}"
        )


print(
    "Final evidence schemas:",
    "PASS"
)


# ============================================================
# VERIFY MODEL SET
# ============================================================

for name, frame in [
    (
        "structure_summary",
        structure_summary,
    ),
    (
        "support_structure",
        support_structure,
    ),
    (
        "group_structure",
        group_structure,
    ),
    (
        "fdr_summary",
        fdr_summary,
    ),
]:

    model_set = set(
        frame[
            "candidate_id"
        ].astype(str).unique()
    )


    if model_set != set(MODEL_ORDER):

        raise RuntimeError(
            f"{name}: model set mismatch."
        )


print(
    "Cross-evidence model alignment:",
    "PASS"
)


# ============================================================
# VERIFY PRIMARY OVERALL STRUCTURE UNIT
# ============================================================

overall_structure = structure_summary[
    (
        structure_summary[
            "analysis_scope"
        ].astype(str)
        ==
        "OVERALL_POOLED"
    )
    &
    (
        structure_summary[
            "observed_stratum"
        ].astype(str)
        ==
        "OVERALL"
    )
].copy()


if len(
    overall_structure
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected one OVERALL structure-null "
        "summary row per model."
    )


if (
    overall_structure[
        "candidate_id"
    ].astype(str).nunique()
    != EXPECTED_MODELS
):

    raise RuntimeError(
        "OVERALL structure model coverage failed."
    )


# ============================================================
# VERIFY PRIMARY OVERALL FDR FAMILIES
# ============================================================

overall_fdr = fdr_summary[
    fdr_summary[
        "stratum"
    ].astype(str)
    ==
    "OVERALL"
].copy()


if len(
    overall_fdr
) != (
    EXPECTED_MODELS
    *
    2
):

    raise RuntimeError(
        "Expected two OVERALL FDR families "
        "per model."
    )


if not (
    overall_fdr[
        "features_tested"
    ].astype(int)
    ==
    25
).all():

    raise RuntimeError(
        "OVERALL FDR family size changed."
    )


if not (
    overall_fdr[
        "fdr_q"
    ].astype(float)
    ==
    0.05
).all():

    raise RuntimeError(
        "OVERALL FDR q changed."
    )


print(
    "Primary synthesis evidence units:",
    "PASS"
)


# ============================================================
# ROBUSTNESS HANDLING RULES
# ============================================================

robustness_rules = {

    "class_conditional": {
        "required_to_report":
            True,

        "strata":
            [
                "TARGET_0",
                "TARGET_1",
            ],

        "uses_frozen_structure_null":
            True,

        "uses_featurewise_bh_fdr":
            True,

        "automatic_binary_gate":
            False,

        "reason":
            (
                "Class-conditioned evidence is used to "
                "assess whether pooled attribution changes "
                "remain visible within classes rather than "
                "being explained solely by prevalence mix."
            ),
    },

    "common_support_reweighting": {
        "required_to_report":
            True,

        "variants":
            [
                "COMMON_SUPPORT_UNWEIGHTED",
                "COMMON_SUPPORT_REWEIGHTED",
            ],

        "automatic_binary_gate":
            False,

        "reason":
            (
                "The sensitivity estimand uses a "
                "support-restricted reference population. "
                "The full-population Chapter-9 null is not "
                "validly reusable as a formal threshold."
            ),

        "allowed_interpretation":
            [
                "ATTENUATES_PRIMARY_PATTERN",
                "SIMILAR_TO_PRIMARY_PATTERN",
                "STRENGTHENS_PRIMARY_PATTERN",
                "MIXED_ACROSS_METRICS",
            ],
    },

    "correlated_groups": {
        "required_to_report":
            True,

        "automatic_binary_gate":
            False,

        "group_level_null_available":
            False,

        "reason":
            (
                "Group-level metrics are descriptive "
                "robustness checks because no frozen "
                "10-group Cycle24 null distribution exists."
            ),

        "allowed_interpretation":
            [
                "GROUP_AGGREGATION_ATTENUATES_PATTERN",
                "GROUP_AGGREGATION_PRESERVES_PATTERN",
                "GROUP_AGGREGATION_STRENGTHENS_PATTERN",
                "MIXED_ACROSS_METRICS",
            ],
    },

    "temporal_blocks": {
        "required_to_report":
            True,

        "blocks":
            [
                "B1",
                "B2",
                "B3",
            ],

        "automatic_primary_drift_gate":
            False,

        "reason":
            (
                "Temporal blocks were screened against "
                "the frozen OVERALL null, not against "
                "block-size-matched temporal nulls."
            ),
    },
}


# ============================================================
# CROSS-MODEL SYNTHESIS RULE
# ============================================================

cross_model_rule = {

    "unit":
        "FROZEN_MODEL_FAMILY",

    "model_count":
        EXPECTED_MODELS,

    "no_winner_selected":
        True,

    "do_not_average_raw_shap_across_models":
        True,

    "consistency_categories":
        {

            "CONSISTENT_ACROSS_ALL_MODELS":
                (
                    "All 3 models receive the same "
                    "primary inferential evidence tier."
                ),

            "MAJORITY_MODEL_PATTERN":
                (
                    "Exactly 2 of 3 models receive the "
                    "same primary evidence tier."
                ),

            "MODEL_DEPENDENT_MIXED_PATTERN":
                (
                    "No single evidence tier describes "
                    "a majority pattern cleanly."
                ),
        },

    "final_binary_drift_label_automatic":
        False,

    "reason":
        (
            "Different model families operate in different "
            "SHAP output spaces and may exhibit different "
            "attribution-drift behavior. The synthesis must "
            "preserve model-specific evidence instead of "
            "forcing a pooled raw-SHAP conclusion."
        ),
}


# ============================================================
# FINAL PROTOCOL
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Final Cross-Cycle SHAP "
            "Synthesis Decision Protocol"
        ),

    "version":
        "1.0",

    "status":
        (
            "FROZEN_FINAL_CROSS_CYCLE_"
            "SHAP_SYNTHESIS_PROTOCOL"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_10_1B",

    "timing_disclosure":
        {
            "cycle25_shap_results_already_observed":
                True,

            "structure_null_results_already_observed":
                True,

            "support_reweight_results_already_observed":
                True,

            "correlated_group_results_already_observed":
                True,

            "featurewise_fdr_results_already_observed":
                True,

            "final_synthesis_table_created_before_this_protocol":
                False,

            "final_model_evidence_tiers_assigned_before_this_protocol":
                False,

            "final_explanation_drift_label_assigned_before_this_protocol":
                False,

            "interpretation":
                (
                    "This is a pre-synthesis decision-rule "
                    "freeze, not a preregistration before "
                    "Cycle25 data were observed."
                ),
        },

    "frozen_evidence_sources":
        {
            "structure_null":
                {
                    "contract":
                        NULL_CONTRACT_FILE.as_posix(),

                    "contract_sha256":
                        sha256_file(
                            NULL_CONTRACT_FILE
                        ),

                    "summary":
                        STRUCTURE_SUMMARY_FILE.as_posix(),

                    "summary_sha256":
                        sha256_file(
                            STRUCTURE_SUMMARY_FILE
                        ),
                },

            "support_reweighting":
                {
                    "contract":
                        SUPPORT_CONTRACT_FILE.as_posix(),

                    "contract_sha256":
                        sha256_file(
                            SUPPORT_CONTRACT_FILE
                        ),

                    "structure":
                        SUPPORT_STRUCTURE_FILE.as_posix(),

                    "structure_sha256":
                        sha256_file(
                            SUPPORT_STRUCTURE_FILE
                        ),
                },

            "correlated_groups":
                {
                    "contract":
                        GROUP_CONTRACT_FILE.as_posix(),

                    "contract_sha256":
                        sha256_file(
                            GROUP_CONTRACT_FILE
                        ),

                    "structure":
                        GROUP_STRUCTURE_FILE.as_posix(),

                    "structure_sha256":
                        sha256_file(
                            GROUP_STRUCTURE_FILE
                        ),
                },

            "featurewise_fdr":
                {
                    "contract":
                        FDR_CONTRACT_FILE.as_posix(),

                    "contract_sha256":
                        sha256_file(
                            FDR_CONTRACT_FILE
                        ),

                    "summary":
                        FDR_SUMMARY_FILE.as_posix(),

                    "summary_sha256":
                        sha256_file(
                            FDR_SUMMARY_FILE
                        ),
                },
        },

    "primary_model_evidence_rule":
        {
            "evidence_tiers":
                PRIMARY_EVIDENCE_TIERS,

            "primary_scope":
                "OVERALL_POOLED",

            "primary_stratum":
                "OVERALL",

            "structure_condition":
                (
                    "Use frozen OVERALL "
                    "any_null_exceedance."
                ),

            "featurewise_condition":
                (
                    "Sum OVERALL BH-FDR significant "
                    "features across the two predefined "
                    "featurewise quantities."
                ),

            "featurewise_signal_rule":
                (
                    "TOTAL_OVERALL_FDR_SIGNIFICANT_"
                    "HYPOTHESES >= 1"
                ),

            "featurewise_zero_rule":
                (
                    "TOTAL_OVERALL_FDR_SIGNIFICANT_"
                    "HYPOTHESES == 0"
                ),

            "binary_drift_label_created":
                False,

            "binary_stability_label_created":
                False,
        },

    "robustness_rules":
        robustness_rules,

    "cross_model_rule":
        cross_model_rule,

    "prohibited_logic":
        [
            (
                "Do not classify explanation drift "
                "from featurewise FDR alone."
            ),
            (
                "Do not classify stability merely "
                "because a frozen null threshold "
                "was not exceeded."
            ),
            (
                "Do not reuse full-population null "
                "thresholds as formal thresholds for "
                "common-support-restricted estimates."
            ),
            (
                "Do not invent group-level statistical "
                "significance without a frozen group null."
            ),
            (
                "Do not treat temporal B1/B2/B3 screening "
                "as block-specific formal inference."
            ),
            (
                "Do not compare raw SHAP magnitudes "
                "across LR, RF, and XGB."
            ),
            (
                "Do not interpret SHAP differences "
                "as causal solar-physics effects."
            ),
        ],

    "required_final_synthesis_outputs":
        {
            "model_level_rows":
                EXPECTED_MODELS,

            "must_report":
                [
                    "overall_structure_null_exceedance",
                    "overall_structure_exceedance_count",
                    "overall_fdr_mean_abs_count",
                    "overall_fdr_mean_signed_count",
                    "primary_evidence_tier",
                    "target0_structure_null_exceedance",
                    "target1_structure_null_exceedance",
                    "target0_fdr_count",
                    "target1_fdr_count",
                    "support_reweighting_annotation",
                    "correlated_group_annotation",
                    "temporal_screening_annotation",
                    "sample_size_caution",
                ],
        },

    "scientific_guardrails":
        {
            "models_retrained":
                False,

            "shap_recomputed":
                False,

            "cycle24_null_recomputed":
                False,

            "support_weights_recomputed":
                False,

            "correlated_groups_recomputed":
                False,

            "fdr_protocol_changed":
                False,

            "new_statistical_threshold_invented":
                False,

            "support_binary_threshold_invented":
                False,

            "group_binary_threshold_invented":
                False,

            "final_synthesis_computed_here":
                False,

            "final_explanation_drift_conclusion_made":
                False,

            "final_explanation_stability_conclusion_made":
                False,

            "shap_is_causal":
                False,
        },

    "interpretation_rule":
        (
            "The final synthesis must preserve the "
            "distinction between formal inferential "
            "evidence and descriptive robustness evidence. "
            "Primary model-level evidence tiers are based "
            "on the frozen OVERALL structure-null screen "
            "and OVERALL featurewise BH-FDR results. "
            "Class-conditional, support/reweighting, "
            "correlated-group, and temporal analyses must "
            "be reported as predefined context and "
            "robustness evidence without inventing "
            "unfrozen significance thresholds."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_10_2_"
            "FINAL_CROSS_CYCLE_SHAP_SYNTHESIS"
        ),
}


# ============================================================
# SAVE CONTRACT
# ============================================================

OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


temp_file = OUTPUT_CONTRACT_FILE.with_name(
    OUTPUT_CONTRACT_FILE.stem
    +
    ".tmp.json"
)


if temp_file.exists():

    temp_file.unlink()


with temp_file.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


roundtrip = load_json(
    temp_file
)


if (
    roundtrip.get("status")
    !=
    "FROZEN_FINAL_CROSS_CYCLE_SHAP_SYNTHESIS_PROTOCOL"
):

    raise RuntimeError(
        "10.10.1B contract round-trip failed."
    )


if (
    roundtrip[
        "scientific_guardrails"
    ][
        "final_synthesis_computed_here"
    ]
    is not False
):

    raise RuntimeError(
        "Protocol unexpectedly computed "
        "final synthesis."
    )


if (
    roundtrip[
        "scientific_guardrails"
    ][
        "new_statistical_threshold_invented"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected new statistical threshold."
    )


temp_file.replace(
    OUTPUT_CONTRACT_FILE
)


contract_hash = sha256_file(
    OUTPUT_CONTRACT_FILE
)


# ============================================================
# FINAL
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 10.10.1B — COMPLETE"
)

print("=" * 78)


print(
    "Models:",
    EXPECTED_MODELS
)

print(
    "Primary evidence tiers:",
    len(
        PRIMARY_EVIDENCE_TIERS
    )
)


print()

print(
    "Primary structure scope:",
    "OVERALL_POOLED"
)

print(
    "Primary featurewise FDR stratum:",
    "OVERALL"
)


print()

print(
    "Class-conditional robustness required:",
    True
)

print(
    "Support/reweighting robustness required:",
    True
)

print(
    "Correlated-group robustness required:",
    True
)

print(
    "Temporal screening required:",
    True
)


print()

print(
    "New support significance threshold:",
    False
)

print(
    "New group significance threshold:",
    False
)

print(
    "Automatic binary drift label:",
    False
)


print()

print(
    "Final synthesis computed:",
    False
)

print(
    "Final explanation-drift conclusion:",
    False
)


print()

print(
    "Status:",
    (
        "FROZEN_FINAL_CROSS_CYCLE_"
        "SHAP_SYNTHESIS_PROTOCOL"
    )
)


print()

print(
    "Contract:",
    OUTPUT_CONTRACT_FILE
)

print(
    "Contract SHA256:",
    contract_hash
)


print()

print(
    "NEXT:",
    (
        "Chapter 10.10.2 — "
        "Final Cross-Cycle SHAP Synthesis"
    )
)