# ============================================================
# MODEL CHAPTER 11.5B
# FINAL HYPOTHESIS EVALUATION
#
# Evaluates H0/H1/H2/H3 using ONLY:
#   - frozen 11.5A interpretation protocol
#   - frozen 11.4 joint prediction/explanation evidence
#   - frozen 11.3 explanation synthesis context
#
# IMPORTANT:
#   - no new statistical test
#   - no new bootstrap
#   - no new null
#   - no new significance threshold
#   - no equivalence claim
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

PROTOCOL_FILE = Path(
    "artifacts/models/final_synthesis/contracts/"
    "chapter11_hypothesis_evaluation_protocol.json"
)


JOINT_CONTRACT_FILE = Path(
    "artifacts/models/final_synthesis/contracts/"
    "joint_prediction_explanation_outcome_contract.json"
)


EXPLANATION_CONTRACT_FILE = Path(
    "artifacts/models/final_synthesis/contracts/"
    "explanation_cross_cycle_synthesis_contract.json"
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


HYPOTHESIS_OUTPUT = (
    METRIC_DIR
    /
    "chapter11_hypothesis_evaluation.csv"
)


CONTRACT_OUTPUT = (
    CONTRACT_DIR
    /
    "chapter11_hypothesis_evaluation_contract.json"
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
    "MODEL CHAPTER 11.5B — "
    "FINAL HYPOTHESIS EVALUATION"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

for path in [
    PROTOCOL_FILE,
    JOINT_CONTRACT_FILE,
    EXPLANATION_CONTRACT_FILE,
]:

    require_file(path)


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    HYPOTHESIS_OUTPUT,
    CONTRACT_OUTPUT,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter-11.5B outputs."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

protocol = load_json(
    PROTOCOL_FILE
)


joint = load_json(
    JOINT_CONTRACT_FILE
)


explanation = load_json(
    EXPLANATION_CONTRACT_FILE
)


# ============================================================
# VERIFY FROZEN INPUT STATUSES
# ============================================================

if (
    protocol.get("status")
    !=
    "FROZEN_CHAPTER11_HYPOTHESIS_EVALUATION_PROTOCOL"
):

    raise RuntimeError(
        "Hypothesis evaluation protocol is not frozen."
    )


if (
    joint.get("status")
    !=
    "FROZEN_CHAPTER11_JOINT_PREDICTION_EXPLANATION_OUTCOME"
):

    raise RuntimeError(
        "Joint evidence contract is not frozen."
    )


if (
    explanation.get("status")
    !=
    "FROZEN_CHAPTER11_EXPLANATION_SYNTHESIS"
):

    raise RuntimeError(
        "Explanation synthesis contract is not frozen."
    )


print(
    "Frozen hypothesis inputs:",
    "PASS"
)


# ============================================================
# VERIFY PROTOCOL SOURCE HASH
# ============================================================

protocol_source = protocol[
    "source_joint_contract"
]


if (
    sha256_file(
        JOINT_CONTRACT_FILE
    )
    !=
    protocol_source[
        "sha256"
    ]
):

    raise RuntimeError(
        "Joint evidence contract changed "
        "after hypothesis protocol freeze."
    )


print(
    "Frozen joint evidence hash:",
    "PASS"
)


# ============================================================
# EXTRACT FROZEN EVIDENCE
# ============================================================

predictive = joint[
    "predictive_state"
]


explanation_state = joint[
    "explanation_state"
]


joint_state = joint[
    "joint_state"
]


robustness = explanation[
    "robustness_context"
]


# ============================================================
# VERIFY EXACT EVIDENCE STATE
# ============================================================

primary_comparisons = int(
    predictive[
        "model_scope_comparisons"
    ]
)


no_measurable_count = int(
    predictive[
        "no_measurable_change_classifications"
    ]
)


if (
    primary_comparisons
    !=
    12
):

    raise RuntimeError(
        "Expected 12 primary TSS comparisons."
    )


if (
    no_measurable_count
    !=
    12
):

    raise RuntimeError(
        "Unexpected primary TSS evidence state."
    )


if (
    predictive[
        "equivalence_established"
    ]
    is not False
):

    raise RuntimeError(
        "Predictive equivalence unexpectedly established."
    )


if (
    predictive[
        "strict_stability_established"
    ]
    is not False
):

    raise RuntimeError(
        "Strict predictive stability unexpectedly established."
    )


tier0_models = int(
    explanation_state[
        "tier0_models"
    ]
)


tier1_models = int(
    explanation_state[
        "tier1_models"
    ]
)


tier2_models = int(
    explanation_state[
        "tier2_models"
    ]
)


if (
    tier0_models,
    tier1_models,
    tier2_models,
) != (
    1,
    0,
    2,
):

    raise RuntimeError(
        "Unexpected explanation evidence-tier state."
    )


separation_models = int(
    joint_state[
        "models_with_no_measurable_tss_change_and_tier2_explanation"
    ]
)


if (
    separation_models
    !=
    2
):

    raise RuntimeError(
        "Unexpected prediction/explanation "
        "separation evidence count."
    )


if (
    robustness[
        "support_reweighting_formal_null_applied"
    ]
    is not False
):

    raise RuntimeError(
        "Support/reweighting unexpectedly "
        "treated as formal null inference."
    )


if (
    robustness[
        "correlated_group_formal_null_applied"
    ]
    is not False
):

    raise RuntimeError(
        "Correlated-group analysis unexpectedly "
        "treated as formal null inference."
    )


if (
    robustness[
        "temporal_null_sample_size_matched"
    ]
    is not False
):

    raise RuntimeError(
        "Temporal null unexpectedly marked "
        "sample-size matched."
    )


print(
    "Frozen hypothesis evidence state:",
    "PASS"
)


# ============================================================
# VERIFY ALLOWED STATUS VALUES
# ============================================================

hypothesis_rules = protocol[
    "hypotheses"
]


required_hypotheses = {
    "H0",
    "H1",
    "H2",
    "H3",
}


if (
    set(
        hypothesis_rules.keys()
    )
    !=
    required_hypotheses
):

    raise RuntimeError(
        "Hypothesis set mismatch."
    )


# ============================================================
# APPLY FROZEN RULES
# ============================================================

# ------------------------------------------------------------
# H1
#
# All 12 primary TSS comparisons show no measurable
# change at the frozen 95% CI criterion.
# ------------------------------------------------------------

if (
    no_measurable_count
    ==
    primary_comparisons
    ==
    12
):

    h1_status = (
        "NOT_SUPPORTED_BY_PRIMARY_TSS_CRITERION"
    )

else:

    h1_status = (
        "SUPPORTED_BY_PRIMARY_TSS_CRITERION"
    )


# ------------------------------------------------------------
# H2
#
# Two models TIER_2, one model TIER_0.
# Therefore model-dependent support.
# ------------------------------------------------------------

if (
    tier2_models
    ==
    3
):

    h2_status = (
        "SUPPORTED_ACROSS_ALL_MODELS"
    )

elif (
    tier2_models
    >=
    1
):

    h2_status = (
        "SUPPORTED_WITH_MODEL_DEPENDENCE"
    )

else:

    h2_status = (
        "NOT_SUPPORTED"
    )


# ------------------------------------------------------------
# H3
#
# Evidence requires:
#   no measurable primary TSS change
#   +
#   TIER_2 explanation evidence
#
# Two of three models satisfy this condition.
# ------------------------------------------------------------

if (
    separation_models
    ==
    3
):

    h3_status = (
        "SUPPORTED_ACROSS_ALL_MODELS"
    )

elif (
    separation_models
    ==
    2
):

    h3_status = (
        "SUPPORTED_WITH_QUALIFICATION_"
        "BY_MAJORITY_MODEL_EVIDENCE"
    )

else:

    h3_status = (
        "NOT_SUPPORTED"
    )


# ------------------------------------------------------------
# H0
#
# H0 cannot explain the complete evidence universally
# because RF/XGB have TIER_2 evidence.
#
# But evidence remains model-dependent and robustness
# analyses retain distribution-shift sensitivity.
# Therefore do NOT turn H0 into a simplistic universal
# binary rejection statement.
# ------------------------------------------------------------

if (
    tier2_models
    ==
    0
):

    h0_status = (
        "CONSISTENT_WITH_ALL_EVIDENCE"
    )

else:

    h0_status = (
        "MIXED_NOT_SUFFICIENT_AS_UNIVERSAL_EXPLANATION"
    )


# ============================================================
# VERIFY AGAINST FROZEN ALLOWED STATUSES
# ============================================================

status_map = {
    "H0":
        h0_status,

    "H1":
        h1_status,

    "H2":
        h2_status,

    "H3":
        h3_status,
}


for hypothesis_id, status in (
    status_map.items()
):

    allowed = (
        hypothesis_rules[
            hypothesis_id
        ][
            "allowed_final_statuses"
        ]
    )


    if status not in allowed:

        raise RuntimeError(
            f"{hypothesis_id}: status {status} "
            "is not allowed by frozen protocol."
        )


print(
    "Frozen hypothesis rules applied:",
    "PASS"
)


# ============================================================
# BUILD HYPOTHESIS TABLE
# ============================================================

rows = [

    {
        "hypothesis_id":
            "H0",

        "hypothesis_statement":
            hypothesis_rules[
                "H0"
            ][
                "statement"
            ],

        "final_status":
            h0_status,

        "primary_evidence":
            (
                "RF and XGB are TIER_2; LR is TIER_0. "
                "Therefore ordinary within-Cycle24 "
                "variability/shift alone is not sufficient "
                "as a universal explanation."
            ),

        "model_support_count":
            2,

        "model_total":
            3,

        "important_qualification":
            (
                "Evidence remains model-dependent. "
                "Common-support/reweighting and correlated-"
                "group analyses are robustness/context "
                "analyses rather than new formal null tests."
            ),

        "equivalence_claimed":
            False,

        "causal_claim":
            False,
    },


    {
        "hypothesis_id":
            "H1",

        "hypothesis_statement":
            hypothesis_rules[
                "H1"
            ][
                "statement"
            ],

        "final_status":
            h1_status,

        "primary_evidence":
            (
                "All 12 frozen model-by-Cycle25-scope "
                "primary TSS comparisons are classified "
                "NO_MEASURABLE_CHANGE_AT_95CI."
            ),

        "model_support_count":
            0,

        "model_total":
            3,

        "important_qualification":
            (
                "Failure to detect measurable TSS change "
                "does not establish predictive equivalence "
                "or strict stability. Secondary metrics "
                "show additional changes but do not redefine "
                "the frozen primary TSS criterion."
            ),

        "equivalence_claimed":
            False,

        "causal_claim":
            False,
    },


    {
        "hypothesis_id":
            "H2",

        "hypothesis_statement":
            hypothesis_rules[
                "H2"
            ][
                "statement"
            ],

        "final_status":
            h2_status,

        "primary_evidence":
            (
                "RF and XGB satisfy frozen TIER_2 "
                "explanation evidence: OVERALL "
                "structure-null exceedance plus "
                "OVERALL featurewise BH-FDR evidence. "
                "LR remains TIER_0 globally."
            ),

        "model_support_count":
            2,

        "model_total":
            3,

        "important_qualification":
            (
                "Support is not universal across model "
                "families. LR has featurewise changes "
                "without global structure-null exceedance."
            ),

        "equivalence_claimed":
            False,

        "causal_claim":
            False,
    },


    {
        "hypothesis_id":
            "H3",

        "hypothesis_statement":
            hypothesis_rules[
                "H3"
            ][
                "statement"
            ],

        "final_status":
            h3_status,

        "primary_evidence":
            (
                "RF and XGB each combine no measurable "
                "primary TSS change with TIER_2 "
                "explanation-change evidence."
            ),

        "model_support_count":
            2,

        "model_total":
            3,

        "important_qualification":
            (
                "This is majority-model evidence for "
                "prediction-explanation separation. "
                "It is not proof of zero predictive loss, "
                "predictive equivalence, universal "
                "explanation drift, or causality."
            ),

        "equivalence_claimed":
            False,

        "causal_claim":
            False,
    },
]


evaluation = pd.DataFrame(
    rows
)


if len(
    evaluation
) != 4:

    raise RuntimeError(
        "Expected exactly four hypothesis rows."
    )


if (
    evaluation[
        "hypothesis_id"
    ].tolist()
    !=
    [
        "H0",
        "H1",
        "H2",
        "H3",
    ]
):

    raise RuntimeError(
        "Hypothesis output order changed."
    )


# ============================================================
# SAVE CSV
# ============================================================

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


temp_csv = HYPOTHESIS_OUTPUT.with_name(
    HYPOTHESIS_OUTPUT.stem
    +
    ".tmp.csv"
)


if temp_csv.exists():

    temp_csv.unlink()


evaluation.to_csv(
    temp_csv,
    index=False,
)


roundtrip = pd.read_csv(
    temp_csv,
    keep_default_na=False,
)


if len(
    roundtrip
) != 4:

    raise RuntimeError(
        "Hypothesis evaluation CSV "
        "round-trip failed."
    )


temp_csv.replace(
    HYPOTHESIS_OUTPUT
)


evaluation_hash = sha256_file(
    HYPOTHESIS_OUTPUT
)


print(
    "Hypothesis evaluation CSV round-trip:",
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
            "Final Hypothesis Evaluation"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CHAPTER11_HYPOTHESIS_EVALUATION",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_11_5B",

    "source_protocol":
        {
            "file":
                PROTOCOL_FILE.as_posix(),

            "sha256":
                sha256_file(
                    PROTOCOL_FILE
                ),

            "status":
                protocol[
                    "status"
                ],
        },

    "source_joint_evidence":
        {
            "file":
                JOINT_CONTRACT_FILE.as_posix(),

            "sha256":
                sha256_file(
                    JOINT_CONTRACT_FILE
                ),

            "status":
                joint[
                    "status"
                ],
        },

    "final_hypothesis_statuses":
        status_map,

    "key_evidence_counts":
        {
            "primary_tss_comparisons":
                12,

            "primary_tss_no_measurable_change":
                12,

            "tier0_explanation_models":
                1,

            "tier1_explanation_models":
                0,

            "tier2_explanation_models":
                2,

            "prediction_explanation_separation_models":
                2,
        },

    "evaluation_output":
        {
            "file":
                HYPOTHESIS_OUTPUT.as_posix(),

            "sha256":
                evaluation_hash,

            "rows":
                4,
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

            "shap_recomputed":
                False,

            "equivalence_test_performed":
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

            "hypotheses_evaluated":
                True,

            "final_paper_conclusion_assigned":
                False,
        },

    "required_interpretation":
        {
            "H0":
                (
                    "Mixed and insufficient as a universal "
                    "explanation because RF and XGB exceed "
                    "the frozen explanation evidence criteria, "
                    "while model/shift dependence remains."
                ),

            "H1":
                (
                    "Not supported by the frozen primary "
                    "TSS criterion; no measurable TSS change "
                    "was detected in any of the 12 comparisons. "
                    "This is not equivalence."
                ),

            "H2":
                (
                    "Supported with model dependence because "
                    "RF and XGB are TIER_2 while LR is TIER_0."
                ),

            "H3":
                (
                    "Supported with qualification by majority-"
                    "model evidence because RF and XGB show "
                    "TIER_2 explanation evidence without a "
                    "commensurate measurable primary TSS change."
                ),
        },

    "next_stage":
        (
            "MODEL_CHAPTER_11_6_"
            "FINAL_SCIENTIFIC_CONCLUSION"
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
    "FROZEN_CHAPTER11_HYPOTHESIS_EVALUATION"
):

    raise RuntimeError(
        "Hypothesis contract round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "hypotheses_evaluated"
    ]
    is not True
):

    raise RuntimeError(
        "Hypotheses not marked evaluated."
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
        "Unexpected equivalence claim."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "final_paper_conclusion_assigned"
    ]
    is not False
):

    raise RuntimeError(
        "Final paper conclusion was unexpectedly assigned."
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
    "MODEL CHAPTER 11.5B — COMPLETE"
)

print("=" * 78)


print()

print(
    "H0:",
    h0_status
)

print(
    "H1:",
    h1_status
)

print(
    "H2:",
    h2_status
)

print(
    "H3:",
    h3_status
)


print()

print(
    "Primary TSS no-measurable-change:",
    "12 / 12"
)

print(
    "TIER_2 explanation models:",
    "2 / 3"
)

print(
    "Prediction/explanation separation models:",
    "2 / 3"
)


print()

print(
    "Predictive equivalence claimed:",
    False
)

print(
    "Strict predictive stability claimed:",
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
    "FROZEN_CHAPTER11_HYPOTHESIS_EVALUATION"
)


print()

print(
    "Evaluation:",
    HYPOTHESIS_OUTPUT
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
    "Chapter 11.6 — Final Scientific Conclusion"
)