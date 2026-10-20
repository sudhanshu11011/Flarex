# ============================================================
# MODEL CHAPTER 11.1C
# FREEZE FINAL SYNTHESIS INPUTS
#
# Purpose:
#   Freeze the exact already-computed evidence from:
#
#   Chapter 8  -> predictive cross-cycle evaluation
#   Chapter 9  -> shift/null/control framework
#   Chapter 10 -> cross-cycle SHAP analysis
#
# BEFORE the final joint scientific synthesis.
#
# This step DOES NOT:
#   - retrain models
#   - recompute predictions
#   - recompute SHAP
#   - recompute bootstrap nulls
#   - invent new statistical thresholds
#   - assign final hypotheses
#   - assign final drift/stability labels
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json

import pandas as pd


# ============================================================
# MODEL IDS
# ============================================================

MODEL_IDS = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


# ============================================================
# CHAPTER 8 — PREDICTIVE EVIDENCE
# ============================================================

PREDICTIVE_CONTRACT = Path(
    "artifacts/models/evaluation/contracts/"
    "cross_cycle_predictive_drift_contract.json"
)

PREDICTIVE_DRIFT = Path(
    "artifacts/models/evaluation/metrics/"
    "cross_cycle_predictive_drift.csv"
)

HOLDOUT_RESULTS = Path(
    "artifacts/models/evaluation/manifests/"
    "frozen_holdout_results.csv"
)

CYCLE25_RESULTS = Path(
    "artifacts/models/evaluation/manifests/"
    "frozen_cycle25_predictive_results.csv"
)


# ============================================================
# CHAPTER 9 — SHIFT / NULL CONTROL FRAMEWORK
# ============================================================

CHAPTER9_CONTRACT = Path(
    "artifacts/models/shift_controls/contracts/"
    "chapter9_shift_null_controls_contract.json"
)


# ============================================================
# CHAPTER 10 — EXPLANATION EVIDENCE
# ============================================================

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

MANIFEST_DIR = (
    OUTPUT_ROOT
    /
    "manifests"
)

CONTRACT_DIR = (
    OUTPUT_ROOT
    /
    "contracts"
)

INPUT_REGISTRY = (
    MANIFEST_DIR
    /
    "chapter11_synthesis_input_registry.csv"
)

OUTPUT_CONTRACT = (
    CONTRACT_DIR
    /
    "chapter11_synthesis_inputs_contract.json"
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
            f"Required synthesis input missing: {path}"
        )


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 11.1C — "
    "FREEZE FINAL SYNTHESIS INPUTS"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

input_files = [

    (
        "CHAPTER8",
        "PREDICTIVE_CONTRACT",
        PREDICTIVE_CONTRACT,
    ),

    (
        "CHAPTER8",
        "PREDICTIVE_DRIFT_METRICS",
        PREDICTIVE_DRIFT,
    ),

    (
        "CHAPTER8",
        "CYCLE24_HOLDOUT_RESULTS",
        HOLDOUT_RESULTS,
    ),

    (
        "CHAPTER8",
        "CYCLE25_RESULTS",
        CYCLE25_RESULTS,
    ),

    (
        "CHAPTER9",
        "SHIFT_NULL_CONTROL_CONTRACT",
        CHAPTER9_CONTRACT,
    ),

    (
        "CHAPTER10",
        "CHAPTER10_FINAL_CONTRACT",
        CHAPTER10_CONTRACT,
    ),

    (
        "CHAPTER10",
        "SHAP_FINAL_SYNTHESIS_CONTRACT",
        SHAP_SYNTHESIS_CONTRACT,
    ),

    (
        "CHAPTER10",
        "SHAP_MODEL_SYNTHESIS",
        SHAP_MODEL_SYNTHESIS,
    ),

    (
        "CHAPTER10",
        "SHAP_CROSS_MODEL_SUMMARY",
        SHAP_CROSS_MODEL_SUMMARY,
    ),
]


for _, _, path in input_files:

    require_file(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    INPUT_REGISTRY,
    OUTPUT_CONTRACT,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter-11 synthesis inputs."
        )


# ============================================================
# CHAPTER 8 CONTRACT AUDIT
# ============================================================

predictive_contract = load_json(
    PREDICTIVE_CONTRACT
)


if (
    predictive_contract.get("status")
    !=
    "FROZEN_CROSS_CYCLE_PREDICTIVE_DRIFT"
):

    raise RuntimeError(
        "Predictive drift contract is not frozen."
    )


if (
    predictive_contract.get("primary_metric")
    !=
    "TSS"
):

    raise RuntimeError(
        "Unexpected predictive primary metric."
    )


tss_rule = predictive_contract.get(
    "tss_classification_rule"
)


if not tss_rule:

    raise RuntimeError(
        "Frozen TSS classification rule missing."
    )


constraints = predictive_contract.get(
    "interpretation_constraints",
    {},
)


if (
    constraints.get(
        "zero_containing_delta_ci_is_not_equivalence"
    )
    is not True
):

    raise RuntimeError(
        "Predictive equivalence guardrail missing."
    )


evaluation_state = predictive_contract.get(
    "evaluation_state",
    {},
)


for key in [
    "models_retrained",
    "thresholds_reoptimized",
    "preprocessing_refit",
    "feature_changes",
]:

    if (
        evaluation_state.get(key)
        is not False
    ):

        raise RuntimeError(
            f"Unexpected predictive state: {key}"
        )


print(
    "Chapter-8 predictive contract:",
    "PASS"
)


# ============================================================
# CHAPTER 8 TABLE AUDIT
# ============================================================

predictive_drift = pd.read_csv(
    PREDICTIVE_DRIFT,
    keep_default_na=False,
)

holdout = pd.read_csv(
    HOLDOUT_RESULTS,
    keep_default_na=False,
)

cycle25 = pd.read_csv(
    CYCLE25_RESULTS,
    keep_default_na=False,
)


if len(
    predictive_drift
) != 108:

    raise RuntimeError(
        "Expected 108 cross-cycle metric rows."
    )


if len(
    holdout
) != 3:

    raise RuntimeError(
        "Expected 3 frozen C24 HOLDOUT model rows."
    )


if len(
    cycle25
) != 12:

    raise RuntimeError(
        "Expected 12 frozen Cycle25 predictive rows."
    )


if set(
    holdout["candidate_id"].astype(str)
) != set(
    MODEL_IDS
):

    raise RuntimeError(
        "C24 HOLDOUT model set mismatch."
    )


if set(
    cycle25["candidate_id"].astype(str)
) != set(
    MODEL_IDS
):

    raise RuntimeError(
        "Cycle25 model set mismatch."
    )


primary_tss = predictive_drift[
    predictive_drift[
        "metric"
    ].astype(str).str.lower()
    ==
    "tss"
].copy()


if len(
    primary_tss
) != 12:

    raise RuntimeError(
        "Expected 12 primary TSS "
        "cross-cycle comparisons."
    )


if not (
    primary_tss[
        "predictive_change_classification"
    ].astype(str)
    ==
    "NO_MEASURABLE_CHANGE_AT_95CI"
).all():

    raise RuntimeError(
        "Unexpected primary TSS classification state."
    )


if (
    primary_tss[
        "equivalence_claimed"
    ].astype(str).str.lower()
    ==
    "true"
).any():

    raise RuntimeError(
        "Equivalence was unexpectedly claimed."
    )


print(
    "Chapter-8 predictive evidence tables:",
    "PASS"
)

print(
    "Primary TSS comparisons:",
    len(
        primary_tss
    )
)

print(
    "NO_MEASURABLE_CHANGE_AT_95CI:",
    int(
        (
            primary_tss[
                "predictive_change_classification"
            ]
            ==
            "NO_MEASURABLE_CHANGE_AT_95CI"
        ).sum()
    ),
)

print(
    "Equivalence claimed:",
    False
)


# ============================================================
# CHAPTER 9 CONTRACT AUDIT
# ============================================================

chapter9 = load_json(
    CHAPTER9_CONTRACT
)


if (
    chapter9.get("status")
    !=
    "FROZEN_CHAPTER9_SHIFT_NULL_CONTROLS"
):

    raise RuntimeError(
        "Chapter-9 shift/null controls "
        "are not frozen."
    )


print(
    "Chapter-9 shift/null control contract:",
    "PASS"
)


# ============================================================
# CHAPTER 10 CONTRACT AUDIT
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
        "Final SHAP synthesis is not frozen."
    )


print(
    "Chapter-10 final SHAP contracts:",
    "PASS"
)


# ============================================================
# CHAPTER 10 TABLE AUDIT
# ============================================================

shap_models = pd.read_csv(
    SHAP_MODEL_SYNTHESIS,
    keep_default_na=False,
)

shap_cross = pd.read_csv(
    SHAP_CROSS_MODEL_SUMMARY,
    keep_default_na=False,
)


if len(
    shap_models
) != 3:

    raise RuntimeError(
        "Expected 3 SHAP model synthesis rows."
    )


if len(
    shap_cross
) != 1:

    raise RuntimeError(
        "Expected 1 SHAP cross-model summary row."
    )


if set(
    shap_models[
        "candidate_id"
    ].astype(str)
) != set(
    MODEL_IDS
):

    raise RuntimeError(
        "SHAP synthesis model set mismatch."
    )


expected_tiers = {
    "LR_02_C0p01_BALANCED":
        "TIER_0",

    "RF_04_D12_L5_BALSUB":
        "TIER_2",

    "XGB_02_LR0p03_D3_BAL":
        "TIER_2",
}


actual_tiers = dict(
    zip(
        shap_models[
            "candidate_id"
        ].astype(str),

        shap_models[
            "primary_evidence_tier"
        ].astype(str),
    )
)


if actual_tiers != expected_tiers:

    raise RuntimeError(
        "Unexpected Chapter-10 "
        "model evidence-tier state."
    )


cross_row = shap_cross.iloc[0]


if (
    str(
        cross_row[
            "cross_model_primary_pattern"
        ]
    )
    !=
    "MAJORITY_MODEL_PATTERN"
):

    raise RuntimeError(
        "Unexpected SHAP cross-model pattern."
    )


if (
    str(
        cross_row[
            "dominant_primary_evidence_tier"
        ]
    )
    !=
    "TIER_2"
):

    raise RuntimeError(
        "Unexpected dominant SHAP tier."
    )


print(
    "Chapter-10 SHAP synthesis tables:",
    "PASS"
)


# ============================================================
# CROSS-CHAPTER MODEL ALIGNMENT
# ============================================================

predictive_models = set(
    holdout[
        "candidate_id"
    ].astype(str)
)


cycle25_models = set(
    cycle25[
        "candidate_id"
    ].astype(str)
)


shap_model_ids = set(
    shap_models[
        "candidate_id"
    ].astype(str)
)


if not (
    predictive_models
    ==
    cycle25_models
    ==
    shap_model_ids
    ==
    set(MODEL_IDS)
):

    raise RuntimeError(
        "Cross-chapter model alignment failed."
    )


print(
    "Cross-chapter model alignment:",
    "PASS"
)


# ============================================================
# BUILD INPUT REGISTRY
# ============================================================

registry_rows = []


for index, (
    chapter,
    role,
    path,
) in enumerate(
    input_files,
    start=1,
):

    suffix = (
        path.suffix
        .lower()
        .lstrip(".")
    )


    rows = None

    columns = None

    status = None


    if suffix == "csv":

        frame = pd.read_csv(
            path,
            keep_default_na=False,
        )

        rows = len(
            frame
        )

        columns = len(
            frame.columns
        )


    elif suffix == "json":

        data = load_json(
            path
        )

        status = data.get(
            "status"
        )


    registry_rows.append(
        {
            "input_index":
                index,

            "chapter":
                chapter,

            "role":
                role,

            "path":
                path.as_posix(),

            "artifact_type":
                suffix,

            "rows":
                rows,

            "columns":
                columns,

            "status":
                status,

            "bytes":
                path.stat().st_size,

            "sha256":
                sha256_file(
                    path
                ),

            "frozen_input":
                True,
        }
    )


registry = pd.DataFrame(
    registry_rows
)


if len(
    registry
) != len(
    input_files
):

    raise RuntimeError(
        "Input registry row-count mismatch."
    )


if registry[
    "path"
].duplicated().any():

    raise RuntimeError(
        "Duplicate synthesis input path."
    )


# ============================================================
# SAVE REGISTRY
# ============================================================

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


registry_temp = INPUT_REGISTRY.with_name(
    INPUT_REGISTRY.stem
    +
    ".tmp.csv"
)


if registry_temp.exists():

    registry_temp.unlink()


registry.to_csv(
    registry_temp,
    index=False,
)


registry_rt = pd.read_csv(
    registry_temp,
    keep_default_na=False,
)


if len(
    registry_rt
) != len(
    registry
):

    raise RuntimeError(
        "Synthesis input registry "
        "round-trip failed."
    )


registry_temp.replace(
    INPUT_REGISTRY
)


registry_hash = sha256_file(
    INPUT_REGISTRY
)


print(
    "Synthesis input registry:",
    "PASS"
)


# ============================================================
# FREEZE CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Chapter-11 Final "
            "Cross-Cycle Synthesis Inputs"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CHAPTER11_SYNTHESIS_INPUTS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_11_1C",

    "timing_disclosure":
        {
            "chapter8_results_already_observed":
                True,

            "chapter9_results_already_observed":
                True,

            "chapter10_results_already_observed":
                True,

            "joint_prediction_explanation_synthesis_computed_here":
                False,

            "final_hypothesis_decisions_computed_here":
                False,

            "final_scientific_conclusion_computed_here":
                False,

            "interpretation":
                (
                    "This is a pre-joint-synthesis input "
                    "freeze, not a preregistration before "
                    "Cycle25 observations or prior analyses."
                ),
        },

    "model_ids":
        MODEL_IDS,

    "input_registry":
        {
            "file":
                INPUT_REGISTRY.as_posix(),

            "sha256":
                registry_hash,

            "rows":
                len(
                    registry
                ),
        },

    "predictive_evidence_state":
        {
            "contract_status":
                predictive_contract[
                    "status"
                ],

            "primary_metric":
                "TSS",

            "primary_comparisons":
                12,

            "no_measurable_change_at_95ci":
                12,

            "equivalence_claimed":
                False,

            "models_retrained":
                False,

            "thresholds_reoptimized":
                False,

            "preprocessing_refit":
                False,

            "required_interpretation":
                (
                    "No measurable primary TSS change "
                    "detected at the 95% bootstrap-CI "
                    "criterion. This is not evidence of "
                    "statistical equivalence."
                ),
        },

    "shift_control_state":
        {
            "chapter9_status":
                chapter9[
                    "status"
                ],

            "used_as_context_and_robustness_framework":
                True,

            "new_shift_analysis_performed_here":
                False,
        },

    "explanation_evidence_state":
        {
            "chapter10_status":
                chapter10[
                    "status"
                ],

            "synthesis_status":
                shap_contract[
                    "status"
                ],

            "model_tiers":
                expected_tiers,

            "cross_model_pattern":
                "MAJORITY_MODEL_PATTERN",

            "dominant_primary_evidence_tier":
                "TIER_2",

            "automatic_binary_drift_label":
                False,

            "automatic_binary_stability_label":
                False,
        },

    "scientific_guardrails":
        {
            "cycle25_used_for_training":
                False,

            "model_retraining_allowed":
                False,

            "threshold_retuning_allowed":
                False,

            "preprocessing_refit_allowed":
                False,

            "new_feature_selection_allowed":
                False,

            "new_shap_computation_allowed":
                False,

            "new_null_distribution_allowed":
                False,

            "new_significance_threshold_allowed":
                False,

            "predictive_no_change_equals_equivalence":
                False,

            "shap_is_causal":
                False,

            "final_joint_outcome_assigned":
                False,

            "hypotheses_evaluated":
                False,

            "final_paper_conclusion_assigned":
                False,
        },

    "next_stage":
        (
            "MODEL_CHAPTER_11_2_"
            "PREDICTIVE_SYNTHESIS"
        ),
}


# ============================================================
# SAVE CONTRACT
# ============================================================

CONTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


contract_temp = OUTPUT_CONTRACT.with_name(
    OUTPUT_CONTRACT.stem
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


roundtrip = load_json(
    contract_temp
)


if (
    roundtrip.get("status")
    !=
    "FROZEN_CHAPTER11_SYNTHESIS_INPUTS"
):

    raise RuntimeError(
        "Chapter-11 input contract "
        "round-trip failed."
    )


if (
    roundtrip[
        "scientific_guardrails"
    ][
        "final_joint_outcome_assigned"
    ]
    is not False
):

    raise RuntimeError(
        "Joint conclusion was unexpectedly assigned."
    )


contract_temp.replace(
    OUTPUT_CONTRACT
)


contract_hash = sha256_file(
    OUTPUT_CONTRACT
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 11.1C — COMPLETE"
)

print("=" * 78)


print(
    "Frozen input files:",
    len(
        registry
    )
)


print()

print(
    "Predictive primary metric:",
    "TSS"
)

print(
    "Primary TSS comparisons:",
    12
)

print(
    "NO_MEASURABLE_CHANGE_AT_95CI:",
    12
)

print(
    "Equivalence claimed:",
    False
)


print()

print(
    "Chapter-9 controls:",
    "FROZEN"
)


print()

print(
    "Explanation model tiers:"
)

print(
    "LR:",
    "TIER_0"
)

print(
    "RF:",
    "TIER_2"
)

print(
    "XGB:",
    "TIER_2"
)


print(
    "Cross-model SHAP pattern:",
    "MAJORITY_MODEL_PATTERN"
)


print()

print(
    "Joint prediction/explanation outcome assigned:",
    False
)

print(
    "Hypotheses evaluated:",
    False
)

print(
    "Final paper conclusion assigned:",
    False
)


print()

print(
    "Status:",
    "FROZEN_CHAPTER11_SYNTHESIS_INPUTS"
)


print()

print(
    "Registry:",
    INPUT_REGISTRY
)

print(
    "Registry SHA256:",
    registry_hash
)

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
    "Chapter 11.2 — Predictive Synthesis"
)