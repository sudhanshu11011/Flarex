# ============================================================
# MODEL CHAPTER 11.3
# EXPLANATION CROSS-CYCLE SYNTHESIS
#
# Purpose:
#   Re-express the already-frozen Chapter-10 SHAP evidence
#   inside the final Chapter-11 synthesis namespace.
#
# IMPORTANT:
#   - NO SHAP recomputation
#   - NO new null
#   - NO new FDR
#   - NO new threshold
#   - NO causal interpretation
#   - NO universal binary drift/stability label
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json

import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

MODEL_ORDER = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


EXPECTED_TIERS = {
    "LR_02_C0p01_BALANCED":
        "TIER_0",

    "RF_04_D12_L5_BALSUB":
        "TIER_2",

    "XGB_02_LR0p03_D3_BAL":
        "TIER_2",
}


# ============================================================
# INPUTS
# ============================================================

CHAPTER11_INPUT_CONTRACT = Path(
    "artifacts/models/final_synthesis/contracts/"
    "chapter11_synthesis_inputs_contract.json"
)


CHAPTER10_CONTRACT = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "chapter10_cross_cycle_shap_contract.json"
)


SHAP_SYNTHESIS_CONTRACT = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "final_cross_cycle_shap_synthesis_contract.json"
)


SHAP_MODEL_SYNTHESIS = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "final_cross_cycle_shap_model_synthesis.csv"
)


SHAP_CROSS_MODEL_SUMMARY = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "final_cross_cycle_shap_cross_model_summary.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_ROOT = Path(
    "artifacts/models/final_synthesis"
)


METRIC_DIR = (
    OUTPUT_ROOT
    /
    "metrics"
)


CONTRACT_DIR = (
    OUTPUT_ROOT
    /
    "contracts"
)


MODEL_OUTPUT = (
    METRIC_DIR
    /
    "explanation_cross_cycle_model_synthesis.csv"
)


SUMMARY_OUTPUT = (
    METRIC_DIR
    /
    "explanation_cross_cycle_summary.csv"
)


CONTRACT_OUTPUT = (
    CONTRACT_DIR
    /
    "explanation_cross_cycle_synthesis_contract.json"
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


def require_file(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def parse_bool(value) -> bool:

    if isinstance(value, bool):

        return value


    text = str(
        value
    ).strip().lower()


    if text in {
        "true",
        "1",
    }:

        return True


    if text in {
        "false",
        "0",
        "",
    }:

        return False


    raise RuntimeError(
        f"Cannot parse boolean: {value}"
    )


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 11.3 — "
    "EXPLANATION CROSS-CYCLE SYNTHESIS"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    CHAPTER11_INPUT_CONTRACT,
    CHAPTER10_CONTRACT,
    SHAP_SYNTHESIS_CONTRACT,
    SHAP_MODEL_SYNTHESIS,
    SHAP_CROSS_MODEL_SUMMARY,
]


for path in required_files:

    require_file(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    MODEL_OUTPUT,
    SUMMARY_OUTPUT,
    CONTRACT_OUTPUT,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter-11.3 outputs."
        )


# ============================================================
# VERIFY CHAPTER-11 INPUT FREEZE
# ============================================================

input_contract = load_json(
    CHAPTER11_INPUT_CONTRACT
)


if (
    input_contract.get("status")
    !=
    "FROZEN_CHAPTER11_SYNTHESIS_INPUTS"
):

    raise RuntimeError(
        "Chapter-11 synthesis inputs are not frozen."
    )


explanation_state = input_contract.get(
    "explanation_evidence_state",
    {},
)


if (
    explanation_state.get(
        "cross_model_pattern"
    )
    !=
    "MAJORITY_MODEL_PATTERN"
):

    raise RuntimeError(
        "Frozen explanation cross-model "
        "pattern changed."
    )


if (
    explanation_state.get(
        "dominant_primary_evidence_tier"
    )
    !=
    "TIER_2"
):

    raise RuntimeError(
        "Frozen dominant explanation tier changed."
    )


print(
    "Frozen Chapter-11 input contract:",
    "PASS"
)


# ============================================================
# VERIFY CHAPTER-10 FREEZE
# ============================================================

chapter10 = load_json(
    CHAPTER10_CONTRACT
)


if (
    chapter10.get("status")
    !=
    "FROZEN_CHAPTER10_CROSS_CYCLE_SHAP"
):

    raise RuntimeError(
        "Chapter-10 SHAP package is not frozen."
    )


shap_contract = load_json(
    SHAP_SYNTHESIS_CONTRACT
)


if (
    shap_contract.get("status")
    !=
    "FROZEN_CROSS_CYCLE_SHAP_SYNTHESIS"
):

    raise RuntimeError(
        "Chapter-10 final SHAP synthesis "
        "is not frozen."
    )


guardrails = shap_contract.get(
    "scientific_guardrails",
    {},
)


if (
    guardrails.get(
        "automatic_binary_explanation_drift_label_created"
    )
    is not False
):

    raise RuntimeError(
        "Unexpected automatic drift label."
    )


if (
    guardrails.get(
        "automatic_binary_explanation_stability_label_created"
    )
    is not False
):

    raise RuntimeError(
        "Unexpected automatic stability label."
    )


print(
    "Frozen Chapter-10 explanation contracts:",
    "PASS"
)


# ============================================================
# LOAD CHAPTER-10 SYNTHESIS TABLES
# ============================================================

source_models = pd.read_csv(
    SHAP_MODEL_SYNTHESIS,
    keep_default_na=False,
)


source_summary = pd.read_csv(
    SHAP_CROSS_MODEL_SUMMARY,
    keep_default_na=False,
)


if len(
    source_models
) != 3:

    raise RuntimeError(
        "Expected 3 Chapter-10 model synthesis rows."
    )


if len(
    source_summary
) != 1:

    raise RuntimeError(
        "Expected 1 Chapter-10 cross-model summary row."
    )


if set(
    source_models[
        "candidate_id"
    ].astype(str)
) != set(
    MODEL_ORDER
):

    raise RuntimeError(
        "Explanation model set mismatch."
    )


actual_tiers = dict(
    zip(
        source_models[
            "candidate_id"
        ].astype(str),

        source_models[
            "primary_evidence_tier"
        ].astype(str),
    )
)


if actual_tiers != EXPECTED_TIERS:

    raise RuntimeError(
        "Frozen explanation tiers changed."
    )


print(
    "Chapter-10 explanation tables:",
    "PASS"
)


# ============================================================
# BUILD MODEL-LEVEL EXPLANATION SYNTHESIS
# ============================================================

model_rows = []


for candidate_id in MODEL_ORDER:

    source = source_models[
        source_models[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(
        source
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: expected one "
            "SHAP synthesis row."
        )


    row = source.iloc[0]


    tier = str(
        row[
            "primary_evidence_tier"
        ]
    )


    if tier != EXPECTED_TIERS[
        candidate_id
    ]:

        raise RuntimeError(
            f"{candidate_id}: tier mismatch."
        )


    overall_structure_exceedance = parse_bool(
        row[
            "overall_structure_null_exceedance"
        ]
    )


    overall_fdr_total = int(
        row[
            "overall_fdr_total_significant_hypotheses"
        ]
    )


    # --------------------------------------------
    # Verify frozen tier logic only.
    # No new threshold is created here.
    # --------------------------------------------

    if tier == "TIER_0":

        if overall_structure_exceedance:

            raise RuntimeError(
                f"{candidate_id}: invalid TIER_0 state."
            )


    elif tier == "TIER_2":

        if not overall_structure_exceedance:

            raise RuntimeError(
                f"{candidate_id}: invalid TIER_2 "
                "structure state."
            )


        if overall_fdr_total < 1:

            raise RuntimeError(
                f"{candidate_id}: invalid TIER_2 "
                "featurewise state."
            )


    else:

        raise RuntimeError(
            f"Unexpected frozen tier: {tier}"
        )


    model_rows.append(
        {
            "candidate_id":
                candidate_id,

            "model_family":
                str(
                    row[
                        "model_family"
                    ]
                ),

            # --------------------------------------------
            # Primary OVERALL evidence
            # --------------------------------------------

            "overall_structure_null_exceedance":
                overall_structure_exceedance,

            "overall_structure_exceedance_count":
                int(
                    row[
                        "overall_structure_exceedance_count"
                    ]
                ),

            "overall_structure_exceeded_metrics":
                str(
                    row[
                        "overall_structure_exceeded_metrics"
                    ]
                ),

            "overall_fdr_mean_abs_count":
                int(
                    row[
                        "overall_fdr_mean_abs_count"
                    ]
                ),

            "overall_fdr_mean_signed_count":
                int(
                    row[
                        "overall_fdr_mean_signed_count"
                    ]
                ),

            "overall_fdr_total_significant_hypotheses":
                overall_fdr_total,

            "overall_fdr_unique_feature_count":
                int(
                    row[
                        "overall_fdr_unique_feature_count"
                    ]
                ),

            "overall_fdr_unique_features":
                str(
                    row[
                        "overall_fdr_unique_features"
                    ]
                ),

            "primary_explanation_evidence_tier":
                tier,

            "primary_explanation_evidence_label":
                str(
                    row[
                        "primary_evidence_label"
                    ]
                ),

            # --------------------------------------------
            # Class conditional
            # --------------------------------------------

            "target0_structure_null_exceedance":
                parse_bool(
                    row[
                        "target0_structure_null_exceedance"
                    ]
                ),

            "target0_fdr_total_significant_hypotheses":
                int(
                    row[
                        "target0_fdr_total_significant_hypotheses"
                    ]
                ),

            "target0_fdr_unique_feature_count":
                int(
                    row[
                        "target0_fdr_unique_feature_count"
                    ]
                ),

            "target1_structure_null_exceedance":
                parse_bool(
                    row[
                        "target1_structure_null_exceedance"
                    ]
                ),

            "target1_fdr_total_significant_hypotheses":
                int(
                    row[
                        "target1_fdr_total_significant_hypotheses"
                    ]
                ),

            "target1_fdr_unique_feature_count":
                int(
                    row[
                        "target1_fdr_unique_feature_count"
                    ]
                ),

            # --------------------------------------------
            # Robustness/context
            # --------------------------------------------

            "support_reweighting_annotation":
                str(
                    row[
                        "support_reweighting_annotation"
                    ]
                ),

            "correlated_group_annotation":
                str(
                    row[
                        "correlated_group_annotation"
                    ]
                ),

            "temporal_blocks_with_null_exceedance":
                int(
                    row[
                        "temporal_blocks_with_null_exceedance"
                    ]
                ),

            "temporal_exceedance_blocks":
                str(
                    row[
                        "temporal_exceedance_blocks"
                    ]
                ),

            "temporal_screening_annotation":
                str(
                    row[
                        "temporal_screening_annotation"
                    ]
                ),

            # --------------------------------------------
            # Cautions / restrictions
            # --------------------------------------------

            "target1_cycle24_entity_groups":
                int(
                    row[
                        "target1_cycle24_entity_groups"
                    ]
                ),

            "sample_size_caution":
                str(
                    row[
                        "sample_size_caution"
                    ]
                ),

            "support_results_formally_thresholded":
                parse_bool(
                    row[
                        "support_results_formally_thresholded"
                    ]
                ),

            "group_results_formally_thresholded":
                parse_bool(
                    row[
                        "group_results_formally_thresholded"
                    ]
                ),

            "binary_explanation_drift_label":
                "NOT_ASSIGNED",

            "binary_explanation_stability_label":
                "NOT_ASSIGNED",

            "shap_causal_interpretation":
                False,
        }
    )


model_synthesis = pd.DataFrame(
    model_rows
)


if len(
    model_synthesis
) != 3:

    raise RuntimeError(
        "Expected 3 explanation synthesis rows."
    )


# ============================================================
# CROSS-MODEL EXPLANATION SUMMARY
# ============================================================

tier0_count = int(
    (
        model_synthesis[
            "primary_explanation_evidence_tier"
        ]
        ==
        "TIER_0"
    ).sum()
)


tier1_count = int(
    (
        model_synthesis[
            "primary_explanation_evidence_tier"
        ]
        ==
        "TIER_1"
    ).sum()
)


tier2_count = int(
    (
        model_synthesis[
            "primary_explanation_evidence_tier"
        ]
        ==
        "TIER_2"
    ).sum()
)


if (
    tier0_count,
    tier1_count,
    tier2_count,
) != (
    1,
    0,
    2,
):

    raise RuntimeError(
        "Unexpected explanation tier counts."
    )


summary_source = source_summary.iloc[
    0
]


if (
    str(
        summary_source[
            "cross_model_primary_pattern"
        ]
    )
    !=
    "MAJORITY_MODEL_PATTERN"
):

    raise RuntimeError(
        "Unexpected cross-model explanation pattern."
    )


if (
    str(
        summary_source[
            "dominant_primary_evidence_tier"
        ]
    )
    !=
    "TIER_2"
):

    raise RuntimeError(
        "Unexpected dominant explanation tier."
    )


summary = pd.DataFrame(
    [
        {
            "models_evaluated":
                3,

            "tier0_models":
                tier0_count,

            "tier1_models":
                tier1_count,

            "tier2_models":
                tier2_count,

            "tier0_model_names":
                "LR_02_C0p01_BALANCED",

            "tier1_model_names":
                "",

            "tier2_model_names":
                (
                    "RF_04_D12_L5_BALSUB;"
                    "XGB_02_LR0p03_D3_BAL"
                ),

            "cross_model_explanation_pattern":
                "MAJORITY_MODEL_PATTERN",

            "dominant_primary_explanation_tier":
                "TIER_2",

            "all_models_same_explanation_tier":
                False,

            "automatic_binary_explanation_drift_label":
                "NOT_ASSIGNED",

            "automatic_binary_explanation_stability_label":
                "NOT_ASSIGNED",

            "primary_interpretation":
                (
                    "Explanation evidence is model-dependent. "
                    "RF and XGB show the frozen TIER_2 pattern "
                    "of global structure-null exceedance plus "
                    "OVERALL featurewise BH-FDR evidence. "
                    "LR is TIER_0 because its OVERALL global "
                    "structure did not exceed the frozen "
                    "Cycle24 null, although featurewise "
                    "changes were present. Therefore the "
                    "result is a majority-model TIER_2 "
                    "pattern, not a universal binary "
                    "explanation-drift or stability result."
                ),

            "support_reweighting_is_formal_null_test":
                False,

            "correlated_group_analysis_is_formal_null_test":
                False,

            "temporal_block_null_is_sample_size_matched":
                False,

            "shap_is_causal":
                False,
        }
    ]
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


model_temp = MODEL_OUTPUT.with_name(
    MODEL_OUTPUT.stem
    +
    ".tmp.csv"
)


summary_temp = SUMMARY_OUTPUT.with_name(
    SUMMARY_OUTPUT.stem
    +
    ".tmp.csv"
)


for path in [
    model_temp,
    summary_temp,
]:

    if path.exists():

        path.unlink()


model_synthesis.to_csv(
    model_temp,
    index=False,
)


summary.to_csv(
    summary_temp,
    index=False,
)


model_rt = pd.read_csv(
    model_temp,
    keep_default_na=False,
)


summary_rt = pd.read_csv(
    summary_temp,
    keep_default_na=False,
)


if len(
    model_rt
) != 3:

    raise RuntimeError(
        "Explanation model synthesis "
        "round-trip failed."
    )


if len(
    summary_rt
) != 1:

    raise RuntimeError(
        "Explanation summary round-trip failed."
    )


model_temp.replace(
    MODEL_OUTPUT
)


summary_temp.replace(
    SUMMARY_OUTPUT
)


model_hash = sha256_file(
    MODEL_OUTPUT
)


summary_hash = sha256_file(
    SUMMARY_OUTPUT
)


print(
    "Explanation synthesis CSV round-trip:",
    "PASS"
)


# ============================================================
# CONTRACT
# ============================================================

CONTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


contract = {

    "contract_name":
        (
            "Frozen Chapter-11 Explanation "
            "Cross-Cycle Synthesis"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CHAPTER11_EXPLANATION_SYNTHESIS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_11_3",

    "source_chapter":
        "MODEL_CHAPTER_10",

    "model_evidence_tiers":
        EXPECTED_TIERS,

    "cross_model_evidence":
        {
            "pattern":
                "MAJORITY_MODEL_PATTERN",

            "dominant_tier":
                "TIER_2",

            "tier0_models":
                1,

            "tier1_models":
                0,

            "tier2_models":
                2,

            "universal_binary_drift_label":
                False,

            "universal_binary_stability_label":
                False,
        },

    "model_synthesis":
        {
            "file":
                MODEL_OUTPUT.as_posix(),

            "sha256":
                model_hash,

            "rows":
                3,
        },

    "cross_model_summary":
        {
            "file":
                SUMMARY_OUTPUT.as_posix(),

            "sha256":
                summary_hash,

            "rows":
                1,
        },

    "robustness_context":
        {
            "class_conditional_required":
                True,

            "support_reweighting_required":
                True,

            "correlated_group_required":
                True,

            "temporal_screening_required":
                True,

            "support_reweighting_formal_null_applied":
                False,

            "correlated_group_formal_null_applied":
                False,

            "temporal_null_sample_size_matched":
                False,

            "target1_cycle24_small_group_caution":
                True,

            "target1_cycle24_entity_groups":
                25,
        },

    "scientific_guardrails":
        {
            "shap_recomputed":
                False,

            "new_null_distribution_created":
                False,

            "new_fdr_computation_performed":
                False,

            "new_statistical_threshold_created":
                False,

            "models_retrained":
                False,

            "preprocessing_refit":
                False,

            "thresholds_reoptimized":
                False,

            "raw_shap_pooled_across_model_families":
                False,

            "binary_explanation_drift_label_created":
                False,

            "binary_explanation_stability_label_created":
                False,

            "shap_interpreted_as_causal":
                False,

            "joint_prediction_explanation_outcome_assigned":
                False,

            "hypotheses_evaluated":
                False,
        },

    "required_interpretation":
        (
            "The frozen cross-cycle explanation evidence "
            "is model-dependent. RF and XGB exhibit TIER_2 "
            "evidence, requiring both OVERALL global "
            "structure-null exceedance and featurewise "
            "BH-FDR evidence. LR remains TIER_0 because "
            "its OVERALL structure does not exceed the "
            "frozen Cycle24 null despite featurewise "
            "changes. Therefore Chapter 11 must preserve "
            "a majority-model TIER_2 result rather than "
            "claiming universal explanation drift or "
            "universal explanation stability."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_11_4_"
            "JOINT_PREDICTION_EXPLANATION_OUTCOME"
        ),
}


contract_temp = CONTRACT_OUTPUT.with_name(
    CONTRACT_OUTPUT.stem
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


contract_rt = load_json(
    contract_temp
)


if (
    contract_rt.get("status")
    !=
    "FROZEN_CHAPTER11_EXPLANATION_SYNTHESIS"
):

    raise RuntimeError(
        "Explanation synthesis contract "
        "round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "binary_explanation_drift_label_created"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected explanation drift label."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "shap_interpreted_as_causal"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected causal SHAP interpretation."
    )


contract_temp.replace(
    CONTRACT_OUTPUT
)


contract_hash = sha256_file(
    CONTRACT_OUTPUT
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 11.3 — COMPLETE"
)

print("=" * 78)


for _, row in model_synthesis.iterrows():

    print()

    print(
        row[
            "candidate_id"
        ]
    )

    print(
        "Tier:",
        row[
            "primary_explanation_evidence_tier"
        ],
        "|",
        row[
            "primary_explanation_evidence_label"
        ],
    )

    print(
        "OVERALL structure null exceedance:",
        row[
            "overall_structure_null_exceedance"
        ],
        "| metrics:",
        row[
            "overall_structure_exceedance_count"
        ],
    )

    print(
        "OVERALL FDR significant hypotheses:",
        row[
            "overall_fdr_total_significant_hypotheses"
        ],
    )

    print(
        "TARGET_0 FDR hypotheses:",
        row[
            "target0_fdr_total_significant_hypotheses"
        ],
    )

    print(
        "TARGET_1 FDR hypotheses:",
        row[
            "target1_fdr_total_significant_hypotheses"
        ],
    )


print()

print(
    "TIER_0 models:",
    tier0_count
)

print(
    "TIER_1 models:",
    tier1_count
)

print(
    "TIER_2 models:",
    tier2_count
)


print()

print(
    "Cross-model explanation pattern:",
    "MAJORITY_MODEL_PATTERN"
)

print(
    "Dominant explanation tier:",
    "TIER_2"
)


print()

print(
    "Universal binary drift label:",
    False
)

print(
    "Universal binary stability label:",
    False
)

print(
    "SHAP causal interpretation:",
    False
)


print()

print(
    "Status:",
    "FROZEN_CHAPTER11_EXPLANATION_SYNTHESIS"
)


print()

print(
    "Model synthesis:",
    MODEL_OUTPUT
)

print(
    "Summary:",
    SUMMARY_OUTPUT
)

print(
    "Contract:",
    CONTRACT_OUTPUT
)

print(
    "Contract SHA256:",
    contract_hash
)


print()

print(
    "NEXT:",
    (
        "Chapter 11.4 — "
        "Joint Prediction × Explanation Outcome"
    )
)