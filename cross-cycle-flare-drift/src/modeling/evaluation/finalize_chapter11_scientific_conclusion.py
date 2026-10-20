# ============================================================
# MODEL CHAPTER 11.6
# FINAL SCIENTIFIC CONCLUSION
#
# Purpose:
#   Freeze the final scientific interpretation of the
#   already-completed predictive, explanation, joint,
#   and hypothesis evidence.
#
# IMPORTANT:
#   - no new statistical test
#   - no new bootstrap
#   - no model retraining
#   - no SHAP recomputation
#   - no new threshold
#   - no equivalence claim
#   - no universal explanation-drift claim
#   - no causal SHAP interpretation
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json

import pandas as pd


# ============================================================
# INPUTS
# ============================================================

PREDICTIVE_CONTRACT = Path(
    "artifacts/models/final_synthesis/contracts/"
    "predictive_cross_cycle_synthesis_contract.json"
)


EXPLANATION_CONTRACT = Path(
    "artifacts/models/final_synthesis/contracts/"
    "explanation_cross_cycle_synthesis_contract.json"
)


JOINT_CONTRACT = Path(
    "artifacts/models/final_synthesis/contracts/"
    "joint_prediction_explanation_outcome_contract.json"
)


HYPOTHESIS_CONTRACT = Path(
    "artifacts/models/final_synthesis/contracts/"
    "chapter11_hypothesis_evaluation_contract.json"
)


HYPOTHESIS_TABLE = Path(
    "artifacts/models/final_synthesis/metrics/"
    "chapter11_hypothesis_evaluation.csv"
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


CONCLUSION_OUTPUT = (
    METRIC_DIR
    /
    "chapter11_final_scientific_conclusion.csv"
)


CONTRACT_OUTPUT = (
    CONTRACT_DIR
    /
    "chapter11_final_scientific_conclusion_contract.json"
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

            digest.update(block)

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


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 11.6 — "
    "FINAL SCIENTIFIC CONCLUSION"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    PREDICTIVE_CONTRACT,
    EXPLANATION_CONTRACT,
    JOINT_CONTRACT,
    HYPOTHESIS_CONTRACT,
    HYPOTHESIS_TABLE,
]


for path in required_files:

    require_file(path)


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    CONCLUSION_OUTPUT,
    CONTRACT_OUTPUT,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter-11.6 outputs."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

predictive = load_json(
    PREDICTIVE_CONTRACT
)


explanation = load_json(
    EXPLANATION_CONTRACT
)


joint = load_json(
    JOINT_CONTRACT
)


hypotheses = load_json(
    HYPOTHESIS_CONTRACT
)


# ============================================================
# VERIFY CONTRACT STATUSES
# ============================================================

expected_statuses = [

    (
        predictive.get("status"),
        "FROZEN_CHAPTER11_PREDICTIVE_SYNTHESIS",
    ),

    (
        explanation.get("status"),
        "FROZEN_CHAPTER11_EXPLANATION_SYNTHESIS",
    ),

    (
        joint.get("status"),
        (
            "FROZEN_CHAPTER11_JOINT_"
            "PREDICTION_EXPLANATION_OUTCOME"
        ),
    ),

    (
        hypotheses.get("status"),
        "FROZEN_CHAPTER11_HYPOTHESIS_EVALUATION",
    ),
]


for actual, expected in expected_statuses:

    if actual != expected:

        raise RuntimeError(
            f"Required frozen status missing: {expected}"
        )


print(
    "Frozen Chapter-11 evidence contracts:",
    "PASS"
)


# ============================================================
# VERIFY PREDICTIVE STATE
# ============================================================

predictive_state = predictive[
    "evidence_state"
]


if (
    int(
        predictive_state[
            "primary_comparisons"
        ]
    )
    !=
    12
):

    raise RuntimeError(
        "Expected 12 predictive comparisons."
    )


if (
    int(
        predictive_state[
            "no_measurable_change_at_95ci"
        ]
    )
    !=
    12
):

    raise RuntimeError(
        "Unexpected predictive evidence state."
    )


if (
    predictive_state[
        "equivalence_claimed"
    ]
    is not False
):

    raise RuntimeError(
        "Predictive equivalence unexpectedly claimed."
    )


if (
    predictive_state[
        "stability_proven"
    ]
    is not False
):

    raise RuntimeError(
        "Strict predictive stability unexpectedly proven."
    )


print(
    "Predictive conclusion inputs:",
    "PASS"
)


# ============================================================
# VERIFY EXPLANATION STATE
# ============================================================

explanation_state = explanation[
    "cross_model_evidence"
]


if (
    explanation_state[
        "pattern"
    ]
    !=
    "MAJORITY_MODEL_PATTERN"
):

    raise RuntimeError(
        "Unexpected explanation pattern."
    )


if (
    explanation_state[
        "dominant_tier"
    ]
    !=
    "TIER_2"
):

    raise RuntimeError(
        "Unexpected dominant explanation tier."
    )


if (
    int(
        explanation_state[
            "tier0_models"
        ]
    )
    !=
    1
):

    raise RuntimeError(
        "Expected one TIER_0 model."
    )


if (
    int(
        explanation_state[
            "tier2_models"
        ]
    )
    !=
    2
):

    raise RuntimeError(
        "Expected two TIER_2 models."
    )


print(
    "Explanation conclusion inputs:",
    "PASS"
)


# ============================================================
# VERIFY JOINT STATE
# ============================================================

joint_state = joint[
    "joint_state"
]


if (
    int(
        joint_state[
            "models_with_no_measurable_tss_change_and_tier2_explanation"
        ]
    )
    !=
    2
):

    raise RuntimeError(
        "Expected two prediction/explanation "
        "separation models."
    )


if (
    joint_state[
        "cross_model_pattern"
    ]
    !=
    (
        "MAJORITY_MODEL_EXPLANATION_CHANGE_"
        "EVIDENCE_WITHOUT_MEASURABLE_PRIMARY_TSS_CHANGE"
    )
):

    raise RuntimeError(
        "Unexpected joint evidence pattern."
    )


print(
    "Joint conclusion inputs:",
    "PASS"
)


# ============================================================
# VERIFY HYPOTHESIS STATE
# ============================================================

expected_hypothesis_statuses = {

    "H0":
        "MIXED_NOT_SUFFICIENT_AS_UNIVERSAL_EXPLANATION",

    "H1":
        "NOT_SUPPORTED_BY_PRIMARY_TSS_CRITERION",

    "H2":
        "SUPPORTED_WITH_MODEL_DEPENDENCE",

    "H3":
        (
            "SUPPORTED_WITH_QUALIFICATION_"
            "BY_MAJORITY_MODEL_EVIDENCE"
        ),
}


actual_hypothesis_statuses = hypotheses[
    "final_hypothesis_statuses"
]


if (
    actual_hypothesis_statuses
    !=
    expected_hypothesis_statuses
):

    raise RuntimeError(
        "Frozen hypothesis state changed."
    )


hypothesis_table = pd.read_csv(
    HYPOTHESIS_TABLE,
    keep_default_na=False,
)


if len(
    hypothesis_table
) != 4:

    raise RuntimeError(
        "Expected four hypothesis rows."
    )


print(
    "Hypothesis conclusion inputs:",
    "PASS"
)


# ============================================================
# DEFINE FINAL SCIENTIFIC CONCLUSION
# ============================================================

final_conclusion_class = (
    "MAJORITY_MODEL_PREDICTION_EXPLANATION_"
    "SEPARATION_WITH_MODEL_DEPENDENCE"
)


primary_predictive_conclusion = (
    "Across all three frozen models and all four Cycle25 "
    "evaluation scopes, no measurable change in the primary "
    "TSS metric was detected under the predefined 95% "
    "bootstrap confidence-interval criterion. This does not "
    "establish predictive equivalence or strict stability."
)


primary_explanation_conclusion = (
    "Cross-cycle explanation evidence is model-dependent. "
    "Random Forest and XGBoost show TIER_2 evidence, combining "
    "OVERALL explanation-structure null exceedance with "
    "featurewise BH-FDR-significant attribution changes. "
    "Logistic Regression remains TIER_0 at the global "
    "structure level despite featurewise attribution changes."
)


primary_joint_conclusion = (
    "The dominant cross-model result is majority-model "
    "evidence that explanation behavior can change even when "
    "a commensurate measurable change in primary predictive "
    "TSS is not detected. This prediction-explanation "
    "separation is observed in Random Forest and XGBoost, "
    "but is not universal across all model families."
)


robustness_conclusion = (
    "The interpretation remains qualified by distribution "
    "shift and robustness analyses. Common-support and "
    "Cycle25-to-Cycle24 reweighting analyses can attenuate "
    "some explanation differences, correlated-feature-group "
    "analysis is descriptive because no frozen group-level "
    "null was available, and temporal block screening did not "
    "use block-size-matched null distributions."
)


scientific_limitations = (
    "The results do not prove predictive equivalence, strict "
    "predictive stability, universal explanation drift, "
    "causal relationships between SHARP variables and solar "
    "flares, or a 'right for the wrong reasons' mechanism."
)


paper_level_conclusion = (
    "Within the frozen Cycle24-to-Cycle25 experimental design, "
    "predictive-change detection and explanation-change evidence "
    "do not necessarily move together. Primary TSS showed no "
    "measurable cross-cycle change under the predefined criterion, "
    "while two of three model families exhibited explanation "
    "changes beyond the frozen Cycle24 attribution baseline. "
    "Therefore, cross-cycle model evaluation should examine both "
    "predictive performance and explanation behavior rather than "
    "treating predictive metrics alone as evidence that a model's "
    "internal attribution structure has remained unchanged."
)


# ============================================================
# BUILD CONCLUSION TABLE
# ============================================================

conclusion = pd.DataFrame(
    [
        {
            "conclusion_component":
                "FINAL_CONCLUSION_CLASS",

            "conclusion":
                final_conclusion_class,
        },

        {
            "conclusion_component":
                "PRIMARY_PREDICTIVE_CONCLUSION",

            "conclusion":
                primary_predictive_conclusion,
        },

        {
            "conclusion_component":
                "PRIMARY_EXPLANATION_CONCLUSION",

            "conclusion":
                primary_explanation_conclusion,
        },

        {
            "conclusion_component":
                "PRIMARY_JOINT_CONCLUSION",

            "conclusion":
                primary_joint_conclusion,
        },

        {
            "conclusion_component":
                "ROBUSTNESS_QUALIFICATION",

            "conclusion":
                robustness_conclusion,
        },

        {
            "conclusion_component":
                "SCIENTIFIC_LIMITATIONS",

            "conclusion":
                scientific_limitations,
        },

        {
            "conclusion_component":
                "PAPER_LEVEL_CONCLUSION",

            "conclusion":
                paper_level_conclusion,
        },
    ]
)


if len(
    conclusion
) != 7:

    raise RuntimeError(
        "Expected seven final conclusion components."
    )


# ============================================================
# SAVE CONCLUSION TABLE
# ============================================================

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


temp_csv = CONCLUSION_OUTPUT.with_name(
    CONCLUSION_OUTPUT.stem
    +
    ".tmp.csv"
)


if temp_csv.exists():

    temp_csv.unlink()


conclusion.to_csv(
    temp_csv,
    index=False,
)


roundtrip = pd.read_csv(
    temp_csv,
    keep_default_na=False,
)


if len(
    roundtrip
) != 7:

    raise RuntimeError(
        "Final scientific conclusion "
        "CSV round-trip failed."
    )


temp_csv.replace(
    CONCLUSION_OUTPUT
)


conclusion_hash = sha256_file(
    CONCLUSION_OUTPUT
)


print(
    "Final conclusion CSV round-trip:",
    "PASS"
)


# ============================================================
# FINAL CONTRACT
# ============================================================

CONTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


contract = {

    "contract_name":
        (
            "Frozen Chapter-11 "
            "Final Scientific Conclusion"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CHAPTER11_FINAL_SCIENTIFIC_CONCLUSION",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_11_6",

    "final_conclusion_class":
        final_conclusion_class,

    "predictive_result":
        {
            "primary_metric":
                "TSS",

            "model_scope_comparisons":
                12,

            "no_measurable_change_at_95ci":
                12,

            "equivalence_established":
                False,

            "strict_stability_established":
                False,

            "conclusion":
                primary_predictive_conclusion,
        },

    "explanation_result":
        {
            "tier0_models":
                [
                    "LR_02_C0p01_BALANCED"
                ],

            "tier2_models":
                [
                    "RF_04_D12_L5_BALSUB",
                    "XGB_02_LR0p03_D3_BAL",
                ],

            "model_dependent":
                True,

            "universal_explanation_drift":
                False,

            "universal_explanation_stability":
                False,

            "conclusion":
                primary_explanation_conclusion,
        },

    "joint_result":
        {
            "prediction_explanation_separation_models":
                2,

            "model_total":
                3,

            "majority_model_evidence":
                True,

            "universal_across_models":
                False,

            "conclusion":
                primary_joint_conclusion,
        },

    "hypothesis_results":
        expected_hypothesis_statuses,

    "robustness_qualification":
        robustness_conclusion,

    "scientific_limitations":
        scientific_limitations,

    "paper_level_conclusion":
        paper_level_conclusion,

    "output":
        {
            "file":
                CONCLUSION_OUTPUT.as_posix(),

            "sha256":
                conclusion_hash,

            "rows":
                7,
        },

    "scientific_guardrails":
        {
            "new_statistical_test_performed":
                False,

            "new_bootstrap_performed":
                False,

            "new_null_distribution_created":
                False,

            "new_significance_threshold_created":
                False,

            "models_retrained":
                False,

            "predictions_recomputed":
                False,

            "shap_recomputed":
                False,

            "predictive_equivalence_claimed":
                False,

            "strict_predictive_stability_claimed":
                False,

            "universal_explanation_drift_claimed":
                False,

            "universal_explanation_stability_claimed":
                False,

            "causal_shap_claim_made":
                False,

            "right_for_wrong_reasons_claim_made":
                False,

            "final_scientific_conclusion_assigned":
                True,
        },

    "next_stage":
        (
            "MODEL_CHAPTER_11_7_"
            "FINAL_CHAPTER11_FREEZE"
        ),
}


temp_json = CONTRACT_OUTPUT.with_name(
    CONTRACT_OUTPUT.stem
    +
    ".tmp.json"
)


if temp_json.exists():

    temp_json.unlink()


with temp_json.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


contract_rt = load_json(
    temp_json
)


if (
    contract_rt.get("status")
    !=
    "FROZEN_CHAPTER11_FINAL_SCIENTIFIC_CONCLUSION"
):

    raise RuntimeError(
        "Final scientific conclusion "
        "contract round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "final_scientific_conclusion_assigned"
    ]
    is not True
):

    raise RuntimeError(
        "Final conclusion not marked complete."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "predictive_equivalence_claimed"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected predictive equivalence claim."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "universal_explanation_drift_claimed"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected universal explanation drift claim."
    )


temp_json.replace(
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
    "MODEL CHAPTER 11.6 — COMPLETE"
)

print("=" * 78)


print()

print(
    "Final conclusion class:"
)

print(
    final_conclusion_class
)


print()

print(
    "Predictive:"
)

print(
    "12/12 NO_MEASURABLE_CHANGE_AT_95CI"
)


print(
    "Equivalence established:",
    False
)


print()

print(
    "Explanation:"
)

print(
    "LR  -> TIER_0"
)

print(
    "RF  -> TIER_2"
)

print(
    "XGB -> TIER_2"
)


print()

print(
    "Prediction/explanation separation:",
    "2 / 3 models"
)


print()

print(
    "H0:",
    expected_hypothesis_statuses[
        "H0"
    ]
)

print(
    "H1:",
    expected_hypothesis_statuses[
        "H1"
    ]
)

print(
    "H2:",
    expected_hypothesis_statuses[
        "H2"
    ]
)

print(
    "H3:",
    expected_hypothesis_statuses[
        "H3"
    ]
)


print()

print(
    "Predictive equivalence claimed:",
    False
)

print(
    "Universal explanation drift claimed:",
    False
)

print(
    "Causal SHAP claim:",
    False
)


print()

print(
    "Status:",
    "FROZEN_CHAPTER11_FINAL_SCIENTIFIC_CONCLUSION"
)


print()

print(
    "Conclusion:",
    CONCLUSION_OUTPUT
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
    "Chapter 11.7 — Final Chapter-11 Freeze"
)