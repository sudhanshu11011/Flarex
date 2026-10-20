# ============================================================
# MODEL CHAPTER 11.5A
# FREEZE HYPOTHESIS EVALUATION PROTOCOL
#
# Purpose:
#   Freeze the qualitative interpretation rules used to map
#   the already-frozen Chapter-11 evidence onto H0/H1/H2/H3.
#
# No hypothesis is evaluated in this script.
#
# IMPORTANT:
#   - no new statistical test
#   - no new threshold
#   - no equivalence claim
#   - no universal SHAP drift claim
#   - no causal interpretation
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json


# ============================================================
# INPUT
# ============================================================

JOINT_CONTRACT = Path(
    "artifacts/models/final_synthesis/contracts/"
    "joint_prediction_explanation_outcome_contract.json"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_CONTRACT = Path(
    "artifacts/models/final_synthesis/contracts/"
    "chapter11_hypothesis_evaluation_protocol.json"
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


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 11.5A — "
    "FREEZE HYPOTHESIS EVALUATION PROTOCOL"
)

print("=" * 78)


# ============================================================
# VERIFY INPUT
# ============================================================

if not JOINT_CONTRACT.exists():

    raise FileNotFoundError(
        f"Missing joint synthesis contract: {JOINT_CONTRACT}"
    )


if OUTPUT_CONTRACT.exists():

    raise RuntimeError(
        f"Output already exists: {OUTPUT_CONTRACT}\n"
        "Refusing to overwrite frozen hypothesis protocol."
    )


joint = load_json(
    JOINT_CONTRACT
)


if (
    joint.get("status")
    !=
    "FROZEN_CHAPTER11_JOINT_PREDICTION_EXPLANATION_OUTCOME"
):

    raise RuntimeError(
        "Joint Chapter-11 evidence is not frozen."
    )


# ============================================================
# VERIFY JOINT EVIDENCE STATE
# ============================================================

predictive_state = joint[
    "predictive_state"
]


explanation_state = joint[
    "explanation_state"
]


joint_state = joint[
    "joint_state"
]


if (
    int(
        predictive_state[
            "model_scope_comparisons"
        ]
    )
    !=
    12
):

    raise RuntimeError(
        "Unexpected predictive comparison count."
    )


if (
    int(
        predictive_state[
            "no_measurable_change_classifications"
        ]
    )
    !=
    12
):

    raise RuntimeError(
        "Unexpected primary predictive evidence state."
    )


if (
    predictive_state[
        "equivalence_established"
    ]
    is not False
):

    raise RuntimeError(
        "Predictive equivalence unexpectedly established."
    )


if (
    predictive_state[
        "strict_stability_established"
    ]
    is not False
):

    raise RuntimeError(
        "Strict predictive stability unexpectedly established."
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
        "Unexpected TIER_2 model count."
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
        "Unexpected TIER_0 model count."
    )


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
        "Unexpected prediction/explanation separation count."
    )


print(
    "Frozen joint evidence state:",
    "PASS"
)


# ============================================================
# HYPOTHESIS DEFINITIONS
# ============================================================

hypotheses = {

    "H0": {
        "statement":
            (
                "Observed cross-cycle explanation differences "
                "are consistent with ordinary Cycle24 "
                "variability and/or distributional shift, "
                "without robust explanation-change evidence."
            ),

        "evaluation_rule":
            (
                "H0 cannot serve as a universal explanation "
                "if one or more frozen model families show "
                "TIER_2 explanation evidence. However, "
                "model dependence and shift-sensitivity must "
                "be retained, so H0 is not converted into a "
                "simple universally rejected binary label."
            ),

        "allowed_final_statuses": [
            "CONSISTENT_WITH_ALL_EVIDENCE",
            "MIXED_NOT_SUFFICIENT_AS_UNIVERSAL_EXPLANATION",
        ],
    },


    "H1": {
        "statement":
            (
                "Predictive behavior differs between the "
                "Cycle24 reference evaluation and Cycle25."
            ),

        "primary_metric":
            "TSS",

        "evaluation_rule":
            (
                "Use only the frozen primary TSS cross-cycle "
                "classification. If all 12 model-by-scope "
                "comparisons are NO_MEASURABLE_CHANGE_AT_95CI, "
                "H1 is NOT_SUPPORTED_BY_PRIMARY_TSS_CRITERION. "
                "This does not establish equivalence or strict "
                "predictive stability."
            ),

        "allowed_final_statuses": [
            "SUPPORTED_BY_PRIMARY_TSS_CRITERION",
            "NOT_SUPPORTED_BY_PRIMARY_TSS_CRITERION",
        ],
    },


    "H2": {
        "statement":
            (
                "Cross-cycle SHAP explanation structure "
                "differs beyond ordinary within-Cycle24 "
                "attribution variability."
            ),

        "evaluation_rule":
            (
                "TIER_2 constitutes the frozen model-level "
                "evidence state combining OVERALL structure-null "
                "exceedance with OVERALL featurewise BH-FDR "
                "evidence. If some but not all model families "
                "are TIER_2, H2 is "
                "SUPPORTED_WITH_MODEL_DEPENDENCE rather than "
                "universally supported."
            ),

        "allowed_final_statuses": [
            "SUPPORTED_ACROSS_ALL_MODELS",
            "SUPPORTED_WITH_MODEL_DEPENDENCE",
            "NOT_SUPPORTED",
        ],
    },


    "H3": {
        "statement":
            (
                "Explanation behavior can change without a "
                "commensurate measurable change in primary "
                "predictive skill."
            ),

        "evaluation_rule":
            (
                "A model contributes H3 evidence when its "
                "primary TSS comparisons show no measurable "
                "change at the frozen 95% CI criterion while "
                "its explanation evidence is TIER_2. "
                "If two of three model families satisfy this "
                "condition, H3 is "
                "SUPPORTED_WITH_QUALIFICATION_BY_MAJORITY_MODEL_EVIDENCE. "
                "This must not be reworded as proof of no "
                "predictive loss, predictive equivalence, "
                "or universal explanation drift."
            ),

        "allowed_final_statuses": [
            "SUPPORTED_ACROSS_ALL_MODELS",
            (
                "SUPPORTED_WITH_QUALIFICATION_"
                "BY_MAJORITY_MODEL_EVIDENCE"
            ),
            "NOT_SUPPORTED",
        ],
    },
}


# ============================================================
# LANGUAGE GUARDRAILS
# ============================================================

language_guardrails = {

    "prohibited_claims": [
        "PREDICTIVE_EQUIVALENCE_PROVEN",
        "STRICT_PREDICTIVE_STABILITY_PROVEN",
        "NO_PREDICTIVE_LOSS_PROVEN",
        "UNIVERSAL_EXPLANATION_DRIFT_PROVEN",
        "UNIVERSAL_EXPLANATION_STABILITY_PROVEN",
        "SHAP_CAUSAL_SOLAR_PHYSICS",
        "RIGHT_FOR_THE_WRONG_REASONS",
    ],

    "required_predictive_wording":
        (
            "No measurable primary TSS change was detected "
            "under the frozen 95% bootstrap-CI criterion; "
            "equivalence and strict stability were not established."
        ),

    "required_explanation_wording":
        (
            "Explanation-change evidence is model-dependent: "
            "RF and XGB are TIER_2, while LR is TIER_0 at the "
            "global-structure level despite featurewise changes."
        ),

    "required_joint_wording":
        (
            "The dominant result is majority-model evidence "
            "of explanation change without a commensurate "
            "measurable primary TSS change."
        ),
}


# ============================================================
# CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Chapter-11 "
            "Hypothesis Evaluation Protocol"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CHAPTER11_HYPOTHESIS_EVALUATION_PROTOCOL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_11_5A",

    "source_joint_contract":
        {
            "file":
                JOINT_CONTRACT.as_posix(),

            "sha256":
                sha256_file(
                    JOINT_CONTRACT
                ),

            "status":
                joint[
                    "status"
                ],
        },

    "hypotheses":
        hypotheses,

    "language_guardrails":
        language_guardrails,

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

            "hypothesis_decisions_computed_here":
                False,

            "final_scientific_conclusion_computed_here":
                False,

            "shap_is_causal":
                False,
        },

    "timing_disclosure":
        (
            "The research hypotheses were predefined earlier, "
            "but this explicit evidence-to-status mapping is "
            "being frozen after the underlying Chapter-8 through "
            "Chapter-11.4 results were observed. Therefore this "
            "is a pre-hypothesis-evaluation interpretation freeze, "
            "not a preregistration before Cycle25 analysis."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_11_5B_"
            "HYPOTHESIS_EVALUATION"
        ),
}


# ============================================================
# SAVE
# ============================================================

OUTPUT_CONTRACT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


temp = OUTPUT_CONTRACT.with_name(
    OUTPUT_CONTRACT.stem
    +
    ".tmp.json"
)


if temp.exists():

    temp.unlink()


with temp.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


roundtrip = load_json(
    temp
)


if (
    roundtrip.get("status")
    !=
    "FROZEN_CHAPTER11_HYPOTHESIS_EVALUATION_PROTOCOL"
):

    raise RuntimeError(
        "Hypothesis evaluation protocol "
        "round-trip failed."
    )


if (
    roundtrip[
        "scientific_guardrails"
    ][
        "hypothesis_decisions_computed_here"
    ]
    is not False
):

    raise RuntimeError(
        "Hypothesis decisions were unexpectedly computed."
    )


temp.replace(
    OUTPUT_CONTRACT
)


contract_hash = sha256_file(
    OUTPUT_CONTRACT
)


# ============================================================
# FINAL
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 11.5A — COMPLETE"
)

print("=" * 78)


print(
    "Hypotheses frozen:",
    4
)


print()

print(
    "H0 rule:",
    "MODEL/SHIFT-DEPENDENT NULL INTERPRETATION"
)

print(
    "H1 rule:",
    "PRIMARY TSS ONLY"
)

print(
    "H2 rule:",
    "FROZEN EXPLANATION TIERS"
)

print(
    "H3 rule:",
    "PREDICTIVE-NONDETECTION × TIER_2"
)


print()

print(
    "Equivalence claim allowed:",
    False
)

print(
    "Strict stability claim allowed:",
    False
)

print(
    "Universal SHAP drift claim allowed:",
    False
)

print(
    "Hypotheses evaluated here:",
    False
)


print()

print(
    "Status:",
    "FROZEN_CHAPTER11_HYPOTHESIS_EVALUATION_PROTOCOL"
)


print()

print(
    "Contract:",
    OUTPUT_CONTRACT
)

print(
    "Contract SHA256:",
    contract_hash
)


print()

print(
    "NEXT:",
    "Chapter 11.5B — Hypothesis Evaluation"
)