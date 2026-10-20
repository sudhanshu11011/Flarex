# ============================================================
# MODEL CHAPTER 11.4
# JOINT PREDICTION × EXPLANATION OUTCOME
#
# Purpose:
#   Combine the already-frozen Chapter-11 predictive
#   synthesis and explanation synthesis.
#
# IMPORTANT:
#
# Predictive state is NOT called "stable" because:
#   zero-containing 95% delta-TSS CI != equivalence.
#
# Explanation state is NOT forced into a universal
# binary stable/drift label because Chapter 10 preserved
# model-specific evidence tiers.
#
# This step DOES NOT:
#   - retrain models
#   - generate predictions
#   - compute SHAP
#   - perform bootstrap
#   - create new statistical thresholds
#   - evaluate H0/H1/H2/H3 yet
#   - make causal claims
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


PREDICTIVE_STATE = (
    "NO_MEASURABLE_PRIMARY_TSS_CHANGE_AT_95CI"
)


JOINT_TIER0_LABEL = (
    "NO_MEASURABLE_PRIMARY_TSS_CHANGE_"
    "PLUS_NO_GLOBAL_EXPLANATION_STRUCTURE_"
    "NULL_EXCEEDANCE"
)


JOINT_TIER2_LABEL = (
    "NO_MEASURABLE_PRIMARY_TSS_CHANGE_"
    "PLUS_GLOBAL_AND_FEATUREWISE_"
    "EXPLANATION_CHANGE_EVIDENCE"
)


CROSS_MODEL_JOINT_PATTERN = (
    "MAJORITY_MODEL_EXPLANATION_CHANGE_"
    "EVIDENCE_WITHOUT_MEASURABLE_PRIMARY_TSS_CHANGE"
)


# ============================================================
# INPUTS
# ============================================================

PREDICTIVE_CONTRACT = Path(
    "artifacts/models/final_synthesis/contracts/"
    "predictive_cross_cycle_synthesis_contract.json"
)


PREDICTIVE_MODEL_FILE = Path(
    "artifacts/models/final_synthesis/metrics/"
    "predictive_cross_cycle_model_synthesis.csv"
)


PREDICTIVE_SUMMARY_FILE = Path(
    "artifacts/models/final_synthesis/metrics/"
    "predictive_cross_cycle_summary.csv"
)


EXPLANATION_CONTRACT = Path(
    "artifacts/models/final_synthesis/contracts/"
    "explanation_cross_cycle_synthesis_contract.json"
)


EXPLANATION_MODEL_FILE = Path(
    "artifacts/models/final_synthesis/metrics/"
    "explanation_cross_cycle_model_synthesis.csv"
)


EXPLANATION_SUMMARY_FILE = Path(
    "artifacts/models/final_synthesis/metrics/"
    "explanation_cross_cycle_summary.csv"
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
    "joint_prediction_explanation_model_outcomes.csv"
)


SUMMARY_OUTPUT = (
    METRIC_DIR
    /
    "joint_prediction_explanation_summary.csv"
)


CONTRACT_OUTPUT = (
    CONTRACT_DIR
    /
    "joint_prediction_explanation_outcome_contract.json"
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

    text = str(value).strip().lower()

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
        f"Cannot parse boolean value: {value}"
    )


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 11.4 — "
    "JOINT PREDICTION × EXPLANATION OUTCOME"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    PREDICTIVE_CONTRACT,
    PREDICTIVE_MODEL_FILE,
    PREDICTIVE_SUMMARY_FILE,
    EXPLANATION_CONTRACT,
    EXPLANATION_MODEL_FILE,
    EXPLANATION_SUMMARY_FILE,
]


for path in required_files:

    require_file(path)


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
            "Chapter-11.4 outputs."
        )


# ============================================================
# VERIFY FROZEN CONTRACTS
# ============================================================

predictive_contract = load_json(
    PREDICTIVE_CONTRACT
)


if (
    predictive_contract.get("status")
    !=
    "FROZEN_CHAPTER11_PREDICTIVE_SYNTHESIS"
):

    raise RuntimeError(
        "Predictive synthesis is not frozen."
    )


explanation_contract = load_json(
    EXPLANATION_CONTRACT
)


if (
    explanation_contract.get("status")
    !=
    "FROZEN_CHAPTER11_EXPLANATION_SYNTHESIS"
):

    raise RuntimeError(
        "Explanation synthesis is not frozen."
    )


if (
    predictive_contract[
        "scientific_guardrails"
    ][
        "equivalence_claimed"
    ]
    is not False
):

    raise RuntimeError(
        "Predictive equivalence unexpectedly claimed."
    )


if (
    predictive_contract[
        "scientific_guardrails"
    ][
        "strict_stability_claimed"
    ]
    is not False
):

    raise RuntimeError(
        "Predictive stability unexpectedly claimed."
    )


if (
    explanation_contract[
        "scientific_guardrails"
    ][
        "binary_explanation_drift_label_created"
    ]
    is not False
):

    raise RuntimeError(
        "Universal explanation drift label "
        "unexpectedly exists."
    )


if (
    explanation_contract[
        "scientific_guardrails"
    ][
        "binary_explanation_stability_label_created"
    ]
    is not False
):

    raise RuntimeError(
        "Universal explanation stability label "
        "unexpectedly exists."
    )


print(
    "Frozen predictive/explanation contracts:",
    "PASS"
)


# ============================================================
# VERIFY SOURCE HASHES
# ============================================================

if (
    sha256_file(
        PREDICTIVE_MODEL_FILE
    )
    !=
    predictive_contract[
        "model_synthesis"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Predictive model synthesis artifact changed."
    )


if (
    sha256_file(
        PREDICTIVE_SUMMARY_FILE
    )
    !=
    predictive_contract[
        "cross_model_summary"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Predictive summary artifact changed."
    )


if (
    sha256_file(
        EXPLANATION_MODEL_FILE
    )
    !=
    explanation_contract[
        "model_synthesis"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Explanation model synthesis artifact changed."
    )


if (
    sha256_file(
        EXPLANATION_SUMMARY_FILE
    )
    !=
    explanation_contract[
        "cross_model_summary"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Explanation summary artifact changed."
    )


print(
    "Frozen source hashes:",
    "PASS"
)


# ============================================================
# LOAD TABLES
# ============================================================

predictive_models = pd.read_csv(
    PREDICTIVE_MODEL_FILE,
    keep_default_na=False,
)


predictive_summary = pd.read_csv(
    PREDICTIVE_SUMMARY_FILE,
    keep_default_na=False,
)


explanation_models = pd.read_csv(
    EXPLANATION_MODEL_FILE,
    keep_default_na=False,
)


explanation_summary = pd.read_csv(
    EXPLANATION_SUMMARY_FILE,
    keep_default_na=False,
)


# ============================================================
# INVENTORY CHECKS
# ============================================================

if len(
    predictive_models
) != 3:

    raise RuntimeError(
        "Expected 3 predictive model rows."
    )


if len(
    explanation_models
) != 3:

    raise RuntimeError(
        "Expected 3 explanation model rows."
    )


if len(
    predictive_summary
) != 1:

    raise RuntimeError(
        "Expected 1 predictive summary row."
    )


if len(
    explanation_summary
) != 1:

    raise RuntimeError(
        "Expected 1 explanation summary row."
    )


if set(
    predictive_models[
        "candidate_id"
    ].astype(str)
) != set(
    MODEL_ORDER
):

    raise RuntimeError(
        "Predictive model set mismatch."
    )


if set(
    explanation_models[
        "candidate_id"
    ].astype(str)
) != set(
    MODEL_ORDER
):

    raise RuntimeError(
        "Explanation model set mismatch."
    )


print(
    "Joint evidence inventories:",
    "PASS"
)


# ============================================================
# VERIFY PREDICTIVE CROSS-MODEL STATE
# ============================================================

predictive_summary_row = predictive_summary.iloc[
    0
]


if (
    str(
        predictive_summary_row[
            "primary_cross_model_predictive_pattern"
        ]
    )
    !=
    (
        "NO_MEASURABLE_PRIMARY_TSS_CHANGE_"
        "AT_95CI_ACROSS_ALL_MODELS_AND_SCOPES"
    )
):

    raise RuntimeError(
        "Unexpected predictive cross-model state."
    )


if (
    int(
        predictive_summary_row[
            "no_measurable_change_at_95ci"
        ]
    )
    !=
    12
):

    raise RuntimeError(
        "Expected 12/12 no-measurable-change "
        "primary TSS comparisons."
    )


if parse_bool(
    predictive_summary_row[
        "equivalence_claimed"
    ]
):

    raise RuntimeError(
        "Equivalence unexpectedly claimed."
    )


if parse_bool(
    predictive_summary_row[
        "stability_proven"
    ]
):

    raise RuntimeError(
        "Strict predictive stability "
        "unexpectedly claimed."
    )


print(
    "Predictive synthesis state:",
    "PASS"
)


# ============================================================
# VERIFY EXPLANATION CROSS-MODEL STATE
# ============================================================

explanation_summary_row = explanation_summary.iloc[
    0
]


if (
    str(
        explanation_summary_row[
            "cross_model_explanation_pattern"
        ]
    )
    !=
    "MAJORITY_MODEL_PATTERN"
):

    raise RuntimeError(
        "Unexpected explanation cross-model pattern."
    )


if (
    str(
        explanation_summary_row[
            "dominant_primary_explanation_tier"
        ]
    )
    !=
    "TIER_2"
):

    raise RuntimeError(
        "Unexpected dominant explanation tier."
    )


if (
    int(
        explanation_summary_row[
            "tier2_models"
        ]
    )
    !=
    2
):

    raise RuntimeError(
        "Expected 2 TIER_2 models."
    )


if (
    int(
        explanation_summary_row[
            "tier0_models"
        ]
    )
    !=
    1
):

    raise RuntimeError(
        "Expected 1 TIER_0 model."
    )


print(
    "Explanation synthesis state:",
    "PASS"
)


# ============================================================
# JOIN MODEL EVIDENCE
# ============================================================

joint_rows = []


for candidate_id in MODEL_ORDER:

    pred = predictive_models[
        predictive_models[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    expl = explanation_models[
        explanation_models[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(pred) != 1:

        raise RuntimeError(
            f"{candidate_id}: predictive row mismatch."
        )


    if len(expl) != 1:

        raise RuntimeError(
            f"{candidate_id}: explanation row mismatch."
        )


    pred = pred.iloc[0]

    expl = expl.iloc[0]


    # --------------------------------------------------------
    # VERIFY PREDICTIVE STATE
    # --------------------------------------------------------

    if (
        int(
            pred[
                "no_measurable_change_count"
            ]
        )
        !=
        4
    ):

        raise RuntimeError(
            f"{candidate_id}: expected 4/4 "
            "no-measurable-change TSS comparisons."
        )


    if parse_bool(
        pred[
            "predictive_stability_proven"
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: stability unexpectedly proven."
        )


    if parse_bool(
        pred[
            "predictive_equivalence_proven"
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: equivalence unexpectedly proven."
        )


    # --------------------------------------------------------
    # EXPLANATION STATE
    # --------------------------------------------------------

    explanation_tier = str(
        expl[
            "primary_explanation_evidence_tier"
        ]
    )


    structure_exceedance = parse_bool(
        expl[
            "overall_structure_null_exceedance"
        ]
    )


    fdr_hypotheses = int(
        expl[
            "overall_fdr_total_significant_hypotheses"
        ]
    )


    if explanation_tier == "TIER_0":

        if structure_exceedance:

            raise RuntimeError(
                f"{candidate_id}: invalid TIER_0 state."
            )


        joint_label = (
            JOINT_TIER0_LABEL
        )


        decoupling_evidence_level = (
            "FEATUREWISE_ONLY_WITHOUT_GLOBAL_"
            "STRUCTURE_NULL_EXCEEDANCE"
        )


    elif explanation_tier == "TIER_2":

        if not structure_exceedance:

            raise RuntimeError(
                f"{candidate_id}: invalid TIER_2 "
                "structure state."
            )


        if fdr_hypotheses < 1:

            raise RuntimeError(
                f"{candidate_id}: invalid TIER_2 "
                "featurewise state."
            )


        joint_label = (
            JOINT_TIER2_LABEL
        )


        decoupling_evidence_level = (
            "GLOBAL_AND_FEATUREWISE_"
            "EXPLANATION_CHANGE_EVIDENCE"
        )


    else:

        raise RuntimeError(
            f"{candidate_id}: unexpected "
            f"explanation tier {explanation_tier}"
        )


    # --------------------------------------------------------
    # CREATE JOINT ROW
    # --------------------------------------------------------

    joint_rows.append(
        {
            "candidate_id":
                candidate_id,

            "model_family":
                str(
                    pred[
                        "model_family"
                    ]
                ),

            # --------------------------------------------
            # Predictive side
            # --------------------------------------------

            "predictive_primary_metric":
                "TSS",

            "predictive_cycle25_scopes":
                4,

            "predictive_no_measurable_change_count":
                int(
                    pred[
                        "no_measurable_change_count"
                    ]
                ),

            "predictive_state":
                PREDICTIVE_STATE,

            "predictive_equivalence_proven":
                False,

            "predictive_strict_stability_proven":
                False,

            # --------------------------------------------
            # Explanation side
            # --------------------------------------------

            "explanation_evidence_tier":
                explanation_tier,

            "explanation_evidence_label":
                str(
                    expl[
                        "primary_explanation_evidence_label"
                    ]
                ),

            "overall_structure_null_exceedance":
                structure_exceedance,

            "overall_structure_exceedance_count":
                int(
                    expl[
                        "overall_structure_exceedance_count"
                    ]
                ),

            "overall_fdr_significant_hypotheses":
                fdr_hypotheses,

            "overall_fdr_unique_feature_count":
                int(
                    expl[
                        "overall_fdr_unique_feature_count"
                    ]
                ),

            "target0_fdr_significant_hypotheses":
                int(
                    expl[
                        "target0_fdr_total_significant_hypotheses"
                    ]
                ),

            "target1_fdr_significant_hypotheses":
                int(
                    expl[
                        "target1_fdr_total_significant_hypotheses"
                    ]
                ),

            # --------------------------------------------
            # Joint evidence
            # --------------------------------------------

            "joint_evidence_label":
                joint_label,

            "prediction_explanation_decoupling_evidence":
                decoupling_evidence_level,

            "joint_binary_stable_drift_cell_assigned":
                False,

            "reason_binary_cell_not_assigned":
                (
                    "Predictive equivalence/strict stability "
                    "was not established, and explanation "
                    "evidence remains model-tier based."
                ),

            # --------------------------------------------
            # Guardrails
            # --------------------------------------------

            "hypothesis_decision_made":
                False,

            "causal_claim_made":
                False,
        }
    )


joint = pd.DataFrame(
    joint_rows
)


if len(joint) != 3:

    raise RuntimeError(
        "Expected 3 joint model outcomes."
    )


# ============================================================
# CROSS-MODEL JOINT SUMMARY
# ============================================================

tier2_joint_models = int(
    (
        joint[
            "explanation_evidence_tier"
        ]
        ==
        "TIER_2"
    ).sum()
)


tier0_joint_models = int(
    (
        joint[
            "explanation_evidence_tier"
        ]
        ==
        "TIER_0"
    ).sum()
)


if tier2_joint_models != 2:

    raise RuntimeError(
        "Expected 2 TIER_2 joint models."
    )


if tier0_joint_models != 1:

    raise RuntimeError(
        "Expected 1 TIER_0 joint model."
    )


models_with_no_measurable_predictive_change = int(
    (
        joint[
            "predictive_no_measurable_change_count"
        ]
        ==
        4
    ).sum()
)


if (
    models_with_no_measurable_predictive_change
    !=
    3
):

    raise RuntimeError(
        "Expected all models to have "
        "4/4 no-measurable-change TSS results."
    )


summary = pd.DataFrame(
    [
        {
            "models_evaluated":
                3,

            "models_with_no_measurable_primary_tss_change":
                3,

            "models_with_tier2_explanation_evidence":
                2,

            "models_with_tier0_explanation_evidence":
                1,

            "models_with_no_measurable_tss_change_and_tier2_explanation":
                2,

            "models_with_no_measurable_tss_change_and_tier0_explanation":
                1,

            "cross_model_joint_pattern":
                CROSS_MODEL_JOINT_PATTERN,

            "dominant_joint_evidence":
                (
                    "NO_MEASURABLE_PRIMARY_TSS_CHANGE_"
                    "WITH_TIER2_EXPLANATION_EVIDENCE"
                ),

            "joint_pattern_is_universal_across_models":
                False,

            "predictive_equivalence_claimed":
                False,

            "predictive_strict_stability_claimed":
                False,

            "universal_explanation_drift_claimed":
                False,

            "universal_explanation_stability_claimed":
                False,

            "four_cell_stable_drift_matrix_formally_assigned":
                False,

            "hypotheses_evaluated_here":
                False,

            "primary_interpretation":
                (
                    "All three frozen model families show "
                    "no measurable primary TSS change under "
                    "the predefined 95% bootstrap-CI rule, "
                    "without establishing equivalence. "
                    "At the same time, RF and XGB show "
                    "TIER_2 explanation-change evidence, "
                    "whereas LR remains TIER_0 at the "
                    "global-structure level despite "
                    "featurewise changes. Thus the dominant "
                    "cross-model pattern is a majority-model "
                    "separation between predictive-change "
                    "detection and explanation-change "
                    "evidence, not a universal binary "
                    "stable/drift outcome."
                ),
        }
    ]
)


# ============================================================
# SAVE OUTPUT TABLES
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


joint.to_csv(
    model_temp,
    index=False,
)


summary.to_csv(
    summary_temp,
    index=False,
)


joint_rt = pd.read_csv(
    model_temp,
    keep_default_na=False,
)


summary_rt = pd.read_csv(
    summary_temp,
    keep_default_na=False,
)


if len(joint_rt) != 3:

    raise RuntimeError(
        "Joint model outcome round-trip failed."
    )


if len(summary_rt) != 1:

    raise RuntimeError(
        "Joint summary round-trip failed."
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
    "Joint synthesis CSV round-trip:",
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
            "Frozen Chapter-11 Joint "
            "Prediction-Explanation Outcome"
        ),

    "version":
        "1.0",

    "status":
        (
            "FROZEN_CHAPTER11_JOINT_"
            "PREDICTION_EXPLANATION_OUTCOME"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_11_4",

    "predictive_state":
        {
            "primary_metric":
                "TSS",

            "models_with_no_measurable_change":
                3,

            "model_scope_comparisons":
                12,

            "no_measurable_change_classifications":
                12,

            "equivalence_established":
                False,

            "strict_stability_established":
                False,
        },

    "explanation_state":
        {
            "tier0_models":
                1,

            "tier1_models":
                0,

            "tier2_models":
                2,

            "cross_model_pattern":
                "MAJORITY_MODEL_PATTERN",

            "dominant_tier":
                "TIER_2",

            "universal_drift_label":
                False,

            "universal_stability_label":
                False,
        },

    "joint_state":
        {
            "models_with_no_measurable_tss_change_and_tier2_explanation":
                2,

            "models_with_no_measurable_tss_change_and_tier0_explanation":
                1,

            "cross_model_pattern":
                CROSS_MODEL_JOINT_PATTERN,

            "dominant_joint_evidence":
                (
                    "NO_MEASURABLE_PRIMARY_TSS_CHANGE_"
                    "WITH_TIER2_EXPLANATION_EVIDENCE"
                ),

            "universal_across_models":
                False,

            "formal_four_cell_stable_drift_matrix_assigned":
                False,

            "hypothesis_decision_made":
                False,
        },

    "model_outcomes":
        {
            row[
                "candidate_id"
            ]:
                {
                    "predictive_state":
                        row[
                            "predictive_state"
                        ],

                    "explanation_tier":
                        row[
                            "explanation_evidence_tier"
                        ],

                    "joint_evidence_label":
                        row[
                            "joint_evidence_label"
                        ],
                }

            for _, row
            in joint.iterrows()
        },

    "outputs":
        {
            "model_outcomes":
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
        },

    "scientific_guardrails":
        {
            "new_predictions_generated":
                False,

            "new_shap_generated":
                False,

            "new_bootstrap_performed":
                False,

            "new_null_distribution_created":
                False,

            "new_significance_threshold_created":
                False,

            "models_retrained":
                False,

            "predictive_equivalence_claimed":
                False,

            "predictive_strict_stability_claimed":
                False,

            "universal_explanation_drift_claimed":
                False,

            "universal_explanation_stability_claimed":
                False,

            "causal_shap_claim_made":
                False,

            "hypotheses_evaluated":
                False,
        },

    "required_interpretation":
        (
            "The joint evidence shows that no measurable "
            "primary TSS change was detected for any model "
            "under the frozen 95% bootstrap-CI rule, while "
            "two of three model families (RF and XGB) show "
            "TIER_2 explanation-change evidence. LR shows "
            "featurewise changes but no global structure-null "
            "exceedance and therefore remains TIER_0. "
            "This constitutes a majority-model prediction-"
            "explanation separation pattern, but does not "
            "establish predictive equivalence, universal "
            "explanation drift, or a causal mechanism."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_11_5_"
            "HYPOTHESIS_EVALUATION"
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
    (
        "FROZEN_CHAPTER11_JOINT_"
        "PREDICTION_EXPLANATION_OUTCOME"
    )
):

    raise RuntimeError(
        "Joint outcome contract round-trip failed."
    )


if (
    contract_rt[
        "joint_state"
    ][
        "hypothesis_decision_made"
    ]
    is not False
):

    raise RuntimeError(
        "Hypothesis was unexpectedly evaluated."
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
    "MODEL CHAPTER 11.4 — COMPLETE"
)

print("=" * 78)


for _, row in joint.iterrows():

    print()

    print(
        row[
            "candidate_id"
        ]
    )

    print(
        "Predictive:",
        row[
            "predictive_state"
        ]
    )

    print(
        "Explanation:",
        row[
            "explanation_evidence_tier"
        ]
    )

    print(
        "Joint:",
        row[
            "joint_evidence_label"
        ]
    )


print()

print(
    "Models with no measurable TSS change:",
    3,
    "/ 3"
)


print(
    "Models with TIER_2 explanation evidence:",
    tier2_joint_models,
    "/ 3"
)


print()

print(
    "Cross-model joint pattern:"
)

print(
    CROSS_MODEL_JOINT_PATTERN
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
    "Hypotheses evaluated:",
    False
)


print()

print(
    "Status:",
    (
        "FROZEN_CHAPTER11_JOINT_"
        "PREDICTION_EXPLANATION_OUTCOME"
    )
)


print()

print(
    "Model outcomes:",
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
    "Chapter 11.5 — Hypothesis Evaluation"
)