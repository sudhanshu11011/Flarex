# ============================================================
# MODEL CHAPTER 11.7
# FINAL CHAPTER-11 SYNTHESIS FREEZE
#
# Purpose:
#   Audit and freeze the complete Chapter-11 final synthesis
#   package after all scientific conclusions are assigned.
#
# This step performs NO new scientific analysis.
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json

import pandas as pd


# ============================================================
# ROOTS
# ============================================================

ROOT = Path(
    "artifacts/models/final_synthesis"
)

CONTRACT_DIR = ROOT / "contracts"
METRIC_DIR = ROOT / "metrics"
MANIFEST_DIR = ROOT / "manifests"


# ============================================================
# REQUIRED CONTRACTS
# ============================================================

REQUIRED_CONTRACTS = {

    "chapter11_synthesis_inputs_contract.json":
        "FROZEN_CHAPTER11_SYNTHESIS_INPUTS",

    "predictive_cross_cycle_synthesis_contract.json":
        "FROZEN_CHAPTER11_PREDICTIVE_SYNTHESIS",

    "explanation_cross_cycle_synthesis_contract.json":
        "FROZEN_CHAPTER11_EXPLANATION_SYNTHESIS",

    "joint_prediction_explanation_outcome_contract.json":
        (
            "FROZEN_CHAPTER11_JOINT_"
            "PREDICTION_EXPLANATION_OUTCOME"
        ),

    "chapter11_hypothesis_evaluation_protocol.json":
        "FROZEN_CHAPTER11_HYPOTHESIS_EVALUATION_PROTOCOL",

    "chapter11_hypothesis_evaluation_contract.json":
        "FROZEN_CHAPTER11_HYPOTHESIS_EVALUATION",

    "chapter11_final_scientific_conclusion_contract.json":
        "FROZEN_CHAPTER11_FINAL_SCIENTIFIC_CONCLUSION",
}


# ============================================================
# REQUIRED METRICS
# ============================================================

REQUIRED_METRICS = [

    "predictive_cross_cycle_model_synthesis.csv",
    "predictive_cross_cycle_summary.csv",

    "explanation_cross_cycle_model_synthesis.csv",
    "explanation_cross_cycle_summary.csv",

    "joint_prediction_explanation_model_outcomes.csv",
    "joint_prediction_explanation_summary.csv",

    "chapter11_hypothesis_evaluation.csv",

    "chapter11_final_scientific_conclusion.csv",
]


# ============================================================
# REQUIRED EXISTING MANIFEST
# ============================================================

INPUT_REGISTRY = (
    MANIFEST_DIR
    /
    "chapter11_synthesis_input_registry.csv"
)


# ============================================================
# FINAL OUTPUTS
# ============================================================

FINAL_INVENTORY = (
    MANIFEST_DIR
    /
    "chapter11_final_synthesis_inventory.csv"
)


FINAL_CONTRACT = (
    CONTRACT_DIR
    /
    "chapter11_final_synthesis_contract.json"
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
            f"Missing required Chapter-11 artifact: {path}"
        )


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 11.7 — "
    "FINAL CHAPTER-11 SYNTHESIS FREEZE"
)

print("=" * 78)


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    FINAL_INVENTORY,
    FINAL_CONTRACT,
]:

    if path.exists():

        raise RuntimeError(
            f"Final freeze output already exists: {path}\n"
            "Refusing to overwrite frozen Chapter-11 package."
        )


# ============================================================
# VERIFY REQUIRED CONTRACTS
# ============================================================

contract_records = []


for filename, expected_status in (
    REQUIRED_CONTRACTS.items()
):

    path = (
        CONTRACT_DIR
        /
        filename
    )


    require_file(path)


    data = load_json(path)


    actual_status = data.get(
        "status"
    )


    if actual_status != expected_status:

        raise RuntimeError(
            f"{filename}: expected status "
            f"{expected_status}, got {actual_status}"
        )


    contract_records.append(
        {
            "path":
                path.as_posix(),

            "status":
                actual_status,

            "sha256":
                sha256_file(path),
        }
    )


print(
    "Required frozen contracts:",
    f"PASS ({len(contract_records)}/7)"
)


# ============================================================
# VERIFY REQUIRED METRIC TABLES
# ============================================================

metric_records = []


for filename in REQUIRED_METRICS:

    path = (
        METRIC_DIR
        /
        filename
    )


    require_file(path)


    frame = pd.read_csv(
        path,
        keep_default_na=False,
    )


    if len(frame) < 1:

        raise RuntimeError(
            f"Metric table is empty: {path}"
        )


    metric_records.append(
        {
            "path":
                path.as_posix(),

            "rows":
                len(frame),

            "columns":
                len(frame.columns),

            "sha256":
                sha256_file(path),
        }
    )


print(
    "Required metric tables:",
    f"PASS ({len(metric_records)}/8)"
)


# ============================================================
# VERIFY INPUT REGISTRY
# ============================================================

require_file(
    INPUT_REGISTRY
)


input_registry = pd.read_csv(
    INPUT_REGISTRY,
    keep_default_na=False,
)


if len(
    input_registry
) != 9:

    raise RuntimeError(
        "Expected 9 frozen Chapter-11 input registry rows."
    )


print(
    "Frozen Chapter-11 input registry:",
    "PASS (9 inputs)"
)


# ============================================================
# VERIFY FINAL SCIENTIFIC STATE
# ============================================================

final_conclusion_contract = load_json(
    CONTRACT_DIR
    /
    "chapter11_final_scientific_conclusion_contract.json"
)


expected_class = (
    "MAJORITY_MODEL_PREDICTION_EXPLANATION_"
    "SEPARATION_WITH_MODEL_DEPENDENCE"
)


if (
    final_conclusion_contract.get(
        "final_conclusion_class"
    )
    !=
    expected_class
):

    raise RuntimeError(
        "Unexpected final Chapter-11 conclusion class."
    )


hypothesis_results = (
    final_conclusion_contract[
        "hypothesis_results"
    ]
)


expected_hypotheses = {

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


if hypothesis_results != expected_hypotheses:

    raise RuntimeError(
        "Final hypothesis state changed."
    )


guardrails = (
    final_conclusion_contract[
        "scientific_guardrails"
    ]
)


required_false_guardrails = [

    "new_statistical_test_performed",
    "new_bootstrap_performed",
    "new_null_distribution_created",
    "new_significance_threshold_created",
    "models_retrained",
    "predictions_recomputed",
    "shap_recomputed",
    "predictive_equivalence_claimed",
    "strict_predictive_stability_claimed",
    "universal_explanation_drift_claimed",
    "universal_explanation_stability_claimed",
    "causal_shap_claim_made",
    "right_for_wrong_reasons_claim_made",
]


for key in required_false_guardrails:

    if guardrails.get(key) is not False:

        raise RuntimeError(
            f"Final scientific guardrail violated: {key}"
        )


if (
    guardrails.get(
        "final_scientific_conclusion_assigned"
    )
    is not True
):

    raise RuntimeError(
        "Final scientific conclusion not marked assigned."
    )


print(
    "Final scientific state:",
    "PASS"
)


# ============================================================
# VERIFY CORE RESULT COUNTS
# ============================================================

predictive_contract = load_json(
    CONTRACT_DIR
    /
    "predictive_cross_cycle_synthesis_contract.json"
)


predictive_state = (
    predictive_contract[
        "evidence_state"
    ]
)


if (
    int(
        predictive_state[
            "primary_comparisons"
        ]
    )
    != 12
):

    raise RuntimeError(
        "Expected 12 primary predictive comparisons."
    )


if (
    int(
        predictive_state[
            "no_measurable_change_at_95ci"
        ]
    )
    != 12
):

    raise RuntimeError(
        "Expected 12/12 no-measurable-change TSS results."
    )


explanation_contract = load_json(
    CONTRACT_DIR
    /
    "explanation_cross_cycle_synthesis_contract.json"
)


explanation_state = (
    explanation_contract[
        "cross_model_evidence"
    ]
)


if (
    int(
        explanation_state[
            "tier0_models"
        ]
    )
    != 1
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
    != 2
):

    raise RuntimeError(
        "Expected two TIER_2 models."
    )


joint_contract = load_json(
    CONTRACT_DIR
    /
    "joint_prediction_explanation_outcome_contract.json"
)


if (
    int(
        joint_contract[
            "joint_state"
        ][
            "models_with_no_measurable_tss_change_and_tier2_explanation"
        ]
    )
    != 2
):

    raise RuntimeError(
        "Expected prediction/explanation "
        "separation in 2/3 models."
    )


print(
    "Core Chapter-11 evidence counts:",
    "PASS"
)


# ============================================================
# BUILD PRE-FREEZE INVENTORY
#
# Expected:
#   contracts = 7
#   metrics   = 8
#   manifests = 1
#   total     = 16
#
# Excludes the inventory and final freeze contract themselves
# to avoid self-referential hashing.
# ============================================================

inventory_rows = []


for path in sorted(
    ROOT.rglob("*")
):

    if not path.is_file():
        continue


    if path in {
        FINAL_INVENTORY,
        FINAL_CONTRACT,
    }:
        continue


    if ".tmp." in path.name:
        continue


    relative = path.relative_to(
        ROOT
    )


    top_level = relative.parts[0]


    if top_level not in {
        "contracts",
        "metrics",
        "manifests",
    }:

        raise RuntimeError(
            f"Unexpected Chapter-11 artifact directory: {path}"
        )


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

        rows = len(frame)

        columns = len(
            frame.columns
        )


    elif suffix == "json":

        data = load_json(path)

        status = data.get(
            "status"
        )


    else:

        raise RuntimeError(
            f"Unexpected Chapter-11 file type: {path}"
        )


    inventory_rows.append(
        {
            "artifact_index":
                len(inventory_rows) + 1,

            "category":
                top_level,

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
                sha256_file(path),

            "frozen":
                True,
        }
    )


inventory = pd.DataFrame(
    inventory_rows
)


if len(
    inventory
) != 16:

    raise RuntimeError(
        "Expected exactly 16 pre-freeze "
        f"Chapter-11 artifacts, found {len(inventory)}."
    )


category_counts = (
    inventory[
        "category"
    ]
    .value_counts()
    .to_dict()
)


if (
    category_counts.get(
        "contracts",
        0,
    )
    != 7
):

    raise RuntimeError(
        "Expected 7 pre-freeze contracts."
    )


if (
    category_counts.get(
        "metrics",
        0,
    )
    != 8
):

    raise RuntimeError(
        "Expected 8 pre-freeze metric tables."
    )


if (
    category_counts.get(
        "manifests",
        0,
    )
    != 1
):

    raise RuntimeError(
        "Expected 1 pre-freeze manifest."
    )


if inventory[
    "path"
].duplicated().any():

    raise RuntimeError(
        "Duplicate Chapter-11 inventory path."
    )


print(
    "Pre-freeze artifact inventory:",
    "PASS (16 artifacts)"
)


# ============================================================
# SAVE INVENTORY
# ============================================================

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


inventory_temp = (
    FINAL_INVENTORY.with_name(
        FINAL_INVENTORY.stem
        +
        ".tmp.csv"
    )
)


if inventory_temp.exists():

    inventory_temp.unlink()


inventory.to_csv(
    inventory_temp,
    index=False,
)


inventory_rt = pd.read_csv(
    inventory_temp,
    keep_default_na=False,
)


if len(
    inventory_rt
) != 16:

    raise RuntimeError(
        "Final inventory round-trip failed."
    )


inventory_temp.replace(
    FINAL_INVENTORY
)


inventory_hash = sha256_file(
    FINAL_INVENTORY
)


print(
    "Final inventory write:",
    "PASS"
)


# ============================================================
# FINAL CHAPTER-11 CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Frozen Chapter-11 Final Cross-Cycle Synthesis Package",

    "version":
        "1.0",

    "status":
        "FROZEN_CHAPTER11_FINAL_SYNTHESIS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_11_7",

    "completed_stages": [
        "11.1_SYNTHESIS_INPUT_FREEZE",
        "11.2_PREDICTIVE_SYNTHESIS",
        "11.3_EXPLANATION_SYNTHESIS",
        "11.4_JOINT_PREDICTION_EXPLANATION_OUTCOME",
        "11.5A_HYPOTHESIS_EVALUATION_PROTOCOL",
        "11.5B_HYPOTHESIS_EVALUATION",
        "11.6_FINAL_SCIENTIFIC_CONCLUSION",
        "11.7_FINAL_PACKAGE_FREEZE",
    ],

    "final_conclusion_class":
        expected_class,

    "final_predictive_state":
        {
            "primary_metric":
                "TSS",

            "comparisons":
                12,

            "no_measurable_change_at_95ci":
                12,

            "predictive_equivalence_established":
                False,

            "strict_predictive_stability_established":
                False,
        },

    "final_explanation_state":
        {
            "LR_02_C0p01_BALANCED":
                "TIER_0",

            "RF_04_D12_L5_BALSUB":
                "TIER_2",

            "XGB_02_LR0p03_D3_BAL":
                "TIER_2",

            "cross_model_pattern":
                "MAJORITY_MODEL_PATTERN",

            "universal_explanation_drift":
                False,
        },

    "final_joint_state":
        {
            "prediction_explanation_separation_models":
                2,

            "total_models":
                3,

            "majority_model_evidence":
                True,

            "universal_across_models":
                False,
        },

    "final_hypothesis_statuses":
        expected_hypotheses,

    "inventory":
        {
            "file":
                FINAL_INVENTORY.as_posix(),

            "sha256":
                inventory_hash,

            "pre_freeze_artifacts":
                16,

            "contracts":
                7,

            "metrics":
                8,

            "existing_manifests":
                1,
        },

    "scientific_guardrails":
        {
            "models_retrained":
                False,

            "predictions_recomputed":
                False,

            "thresholds_reoptimized":
                False,

            "preprocessing_refit":
                False,

            "shap_recomputed":
                False,

            "new_bootstrap_performed":
                False,

            "new_null_distribution_created":
                False,

            "new_significance_threshold_created":
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
        },

    "chapter11_complete":
        True,

    "next_stage":
        "CHAPTER11_GIT_CHECKPOINT",
}


# ============================================================
# SAVE CONTRACT
# ============================================================

CONTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


contract_temp = (
    FINAL_CONTRACT.with_name(
        FINAL_CONTRACT.stem
        +
        ".tmp.json"
    )
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
    "FROZEN_CHAPTER11_FINAL_SYNTHESIS"
):

    raise RuntimeError(
        "Final Chapter-11 contract "
        "round-trip failed."
    )


if (
    contract_rt.get(
        "chapter11_complete"
    )
    is not True
):

    raise RuntimeError(
        "Chapter 11 not marked complete."
    )


contract_temp.replace(
    FINAL_CONTRACT
)


contract_hash = sha256_file(
    FINAL_CONTRACT
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 11.7 — COMPLETE"
)

print("=" * 78)


print()

print(
    "Required frozen contracts:",
    "7 / 7"
)

print(
    "Required metric tables:",
    "8 / 8"
)

print(
    "Existing synthesis manifests:",
    "1 / 1"
)

print(
    "Pre-freeze artifacts:",
    "16"
)


print()

print(
    "Predictive primary TSS:",
    "12 / 12 NO_MEASURABLE_CHANGE_AT_95CI"
)

print(
    "Predictive equivalence established:",
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
    "Final conclusion class:"
)

print(
    expected_class
)


print()

print(
    "Status:",
    "FROZEN_CHAPTER11_FINAL_SYNTHESIS"
)


print()

print(
    "Inventory:",
    FINAL_INVENTORY
)

print(
    "Inventory SHA256:",
    inventory_hash
)

print(
    "Contract:",
    FINAL_CONTRACT
)

print(
    "Contract SHA256:",
    contract_hash
)


print()

print(
    "CHAPTER 11:",
    "COMPLETE"
)

print(
    "NEXT:",
    "Git checkpoint"
)