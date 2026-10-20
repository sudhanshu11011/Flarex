# ============================================================
# MODEL CHAPTER 10.10.3
# FINAL CHAPTER-10 CROSS-CYCLE SHAP FREEZE
#
# Purpose:
#   Freeze the complete final Chapter-10 artifact namespace
#   after all cross-cycle SHAP analyses and synthesis.
#
# This step:
#   - inventories final non-checkpoint artifacts
#   - hashes every frozen artifact
#   - verifies required contract statuses
#   - verifies final synthesis state
#   - creates final Chapter-10 freeze contract
#
# This step DOES NOT:
#   - recompute SHAP
#   - recompute nulls
#   - retrain models
#   - change inference rules
#   - create new statistical conclusions
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
    "artifacts/models/cross_cycle_shap"
)

CONTRACT_DIR = ROOT / "contracts"

MANIFEST_DIR = ROOT / "manifests"


# ============================================================
# OUTPUTS
# ============================================================

INVENTORY_FILE = (
    MANIFEST_DIR
    /
    "chapter10_cross_cycle_shap_inventory.csv"
)


FINAL_CONTRACT_FILE = (
    CONTRACT_DIR
    /
    "chapter10_cross_cycle_shap_contract.json"
)


# ============================================================
# REQUIRED FINAL STATUSES
# ============================================================

REQUIRED_STATUSES = {

    "FROZEN_CYCLE25_SHAP_EXECUTION_INPUTS",

    "FROZEN_CYCLE25_SHAP",

    "FROZEN_OVERALL_CROSS_CYCLE_SHAP_INPUTS",

    "FROZEN_OVERALL_CROSS_CYCLE_SHAP_POINT_ESTIMATES",

    "FROZEN_TEMPORAL_BLOCK_CROSS_CYCLE_SHAP_INPUTS",

    "FROZEN_TEMPORAL_BLOCK_CROSS_CYCLE_SHAP_POINT_ESTIMATES",

    "FROZEN_CLASS_CONDITIONAL_CROSS_CYCLE_SHAP_INPUTS",

    "FROZEN_CLASS_CONDITIONAL_CROSS_CYCLE_SHAP_POINT_ESTIMATES",

    "FROZEN_CROSS_CYCLE_SHAP_STRUCTURE_NULL_COMPARISON",

    "FROZEN_COMMON_SUPPORT_REWEIGHTING_SHAP_SENSITIVITY",

    "FROZEN_CORRELATED_GROUP_SHAP_ROBUSTNESS",

    "FROZEN_FEATUREWISE_INFERENCE_PROTOCOL",

    "FROZEN_FEATUREWISE_CROSS_CYCLE_INFERENCE_BH_FDR",

    "FROZEN_FINAL_CROSS_CYCLE_SHAP_SYNTHESIS_PROTOCOL",

    "FROZEN_CROSS_CYCLE_SHAP_SYNTHESIS",
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def is_final_artifact(
    path: Path,
) -> bool:

    relative = path.relative_to(
        ROOT
    )


    lower_parts = [
        part.lower()
        for part
        in relative.parts
    ]


    # Exclude temporary/checkpoint material.
    if "checkpoints" in lower_parts:

        return False


    name = path.name.lower()


    if ".tmp." in name:

        return False


    if name.endswith(
        ".tmp"
    ):

        return False


    # Exclude outputs of THIS freeze step
    # because they are created after inventory generation.
    if path == INVENTORY_FILE:

        return False


    if path == FINAL_CONTRACT_FILE:

        return False


    return path.is_file()


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.10.3 — "
    "FINAL CHAPTER-10 CROSS-CYCLE SHAP FREEZE"
)

print("=" * 78)


# ============================================================
# BASIC ROOT CHECK
# ============================================================

if not ROOT.exists():

    raise FileNotFoundError(
        f"Missing Chapter-10 root: {ROOT}"
    )


if not CONTRACT_DIR.exists():

    raise FileNotFoundError(
        f"Missing contract directory: {CONTRACT_DIR}"
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    INVENTORY_FILE,
    FINAL_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite final "
            "Chapter-10 freeze artifacts."
        )


# ============================================================
# AUDIT ALL EXISTING CONTRACTS
# ============================================================

contract_files = sorted(
    CONTRACT_DIR.glob(
        "*.json"
    )
)


if len(
    contract_files
) == 0:

    raise RuntimeError(
        "No Chapter-10 contracts found."
    )


contract_rows = []

observed_statuses = set()


for contract_file in contract_files:

    data = load_json(
        contract_file
    )


    status = str(
        data.get(
            "status",
            ""
        )
    ).strip()


    if not status:

        raise RuntimeError(
            f"Contract missing status: {contract_file}"
        )


    observed_statuses.add(
        status
    )


    contract_rows.append(
        {
            "path":
                contract_file.as_posix(),

            "status":
                status,

            "sha256":
                sha256_file(
                    contract_file
                ),
        }
    )


missing_statuses = (
    REQUIRED_STATUSES
    -
    observed_statuses
)


if missing_statuses:

    raise RuntimeError(
        "Missing required Chapter-10 "
        "contract statuses:\n"
        +
        "\n".join(
            sorted(
                missing_statuses
            )
        )
    )


print(
    "Required Chapter-10 contract statuses:",
    "PASS"
)


print(
    "Contract files discovered:",
    len(
        contract_files
    )
)


# ============================================================
# VERIFY FINAL SYNTHESIS CONTRACT
# ============================================================

synthesis_contract_path = (
    CONTRACT_DIR
    /
    "final_cross_cycle_shap_synthesis_contract.json"
)


if not synthesis_contract_path.exists():

    raise FileNotFoundError(
        "Final synthesis contract missing."
    )


synthesis_contract = load_json(
    synthesis_contract_path
)


if (
    synthesis_contract.get(
        "status"
    )
    !=
    "FROZEN_CROSS_CYCLE_SHAP_SYNTHESIS"
):

    raise RuntimeError(
        "Final synthesis is not frozen."
    )


guardrails = synthesis_contract.get(
    "scientific_guardrails",
    {},
)


required_false_guardrails = [
    "new_statistical_test_performed",
    "new_null_distribution_created",
    "new_significance_threshold_created",
    "support_restricted_null_invented",
    "correlated_group_null_invented",
    "temporal_block_specific_null_invented",
    "raw_shap_pooled_across_models",
    "winner_model_selected",
    "automatic_binary_explanation_drift_label_created",
    "automatic_binary_explanation_stability_label_created",
]


for key in required_false_guardrails:

    if guardrails.get(
        key
    ) is not False:

        raise RuntimeError(
            "Unexpected final synthesis "
            f"guardrail state: {key}"
        )


if (
    guardrails.get(
        "final_evidence_synthesis_computed"
    )
    is not True
):

    raise RuntimeError(
        "Final synthesis not marked complete."
    )


print(
    "Final synthesis guardrails:",
    "PASS"
)


# ============================================================
# VERIFY EXPECTED FINAL SYNTHESIS STATE
# ============================================================

cross_model = synthesis_contract[
    "cross_model_summary"
]


if (
    cross_model[
        "cross_model_primary_pattern"
    ]
    !=
    "MAJORITY_MODEL_PATTERN"
):

    raise RuntimeError(
        "Unexpected cross-model synthesis pattern."
    )


if (
    cross_model[
        "dominant_primary_evidence_tier"
    ]
    !=
    "TIER_2"
):

    raise RuntimeError(
        "Unexpected dominant primary evidence tier."
    )


tier_counts = cross_model[
    "tier_counts"
]


if (
    int(
        tier_counts[
            "TIER_0"
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
        tier_counts[
            "TIER_1"
        ]
    )
    !=
    0
):

    raise RuntimeError(
        "Unexpected TIER_1 model count."
    )


if (
    int(
        tier_counts[
            "TIER_2"
        ]
    )
    !=
    2
):

    raise RuntimeError(
        "Unexpected TIER_2 model count."
    )


model_tiers = synthesis_contract[
    "model_primary_evidence_tiers"
]


expected_model_tiers = {

    "LR_02_C0p01_BALANCED":
        "TIER_0",

    "RF_04_D12_L5_BALSUB":
        "TIER_2",

    "XGB_02_LR0p03_D3_BAL":
        "TIER_2",
}


for model_id, expected_tier in (
    expected_model_tiers.items()
):

    actual_tier = (
        model_tiers[
            model_id
        ][
            "tier"
        ]
    )


    if actual_tier != expected_tier:

        raise RuntimeError(
            f"{model_id}: expected "
            f"{expected_tier}, got {actual_tier}"
        )


print(
    "Final synthesis evidence state:",
    "PASS"
)


# ============================================================
# BUILD FINAL NON-CHECKPOINT ARTIFACT INVENTORY
# ============================================================

artifact_files = sorted(
    [
        path
        for path
        in ROOT.rglob(
            "*"
        )
        if is_final_artifact(
            path
        )
    ],
    key=lambda p:
        p.as_posix(),
)


if len(
    artifact_files
) == 0:

    raise RuntimeError(
        "No final Chapter-10 artifacts discovered."
    )


inventory_rows = []


for index, path in enumerate(
    artifact_files,
    start=1,
):

    suffix = (
        path.suffix
        .lower()
        .lstrip(".")
    )


    rows = None

    columns = None

    json_status = None


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


    elif suffix == "parquet":

        frame = pd.read_parquet(
            path
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

        json_status = str(
            data.get(
                "status",
                ""
            )
        )


    inventory_rows.append(
        {
            "inventory_index":
                index,

            "path":
                path.as_posix(),

            "artifact_type":
                suffix,

            "bytes":
                path.stat().st_size,

            "rows":
                rows,

            "columns":
                columns,

            "json_status":
                json_status,

            "sha256":
                sha256_file(
                    path
                ),

            "required":
                True,

            "integrity_pass":
                True,
        }
    )


inventory = pd.DataFrame(
    inventory_rows
)


if inventory[
    "path"
].duplicated().any():

    raise RuntimeError(
        "Duplicate artifact path in inventory."
    )


if not inventory[
    "integrity_pass"
].all():

    raise RuntimeError(
        "Artifact integrity failure."
    )


print(
    "Final artifact inventory:",
    "PASS"
)


print(
    "Non-checkpoint artifacts:",
    len(
        inventory
    )
)


# ============================================================
# SAVE INVENTORY
# ============================================================

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


inventory_temp = INVENTORY_FILE.with_name(
    INVENTORY_FILE.stem
    +
    ".tmp.csv"
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
) != len(
    inventory
):

    raise RuntimeError(
        "Chapter-10 inventory "
        "round-trip failed."
    )


inventory_temp.replace(
    INVENTORY_FILE
)


inventory_hash = sha256_file(
    INVENTORY_FILE
)


print(
    "Inventory CSV round-trip:",
    "PASS"
)


# ============================================================
# FINAL CHAPTER-10 CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Final Frozen Chapter-10 "
            "Cross-Cycle SHAP Artifact Set"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CHAPTER10_CROSS_CYCLE_SHAP",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10",

    "artifact_namespace":
        ROOT.as_posix(),

    "freeze_scope":
        {
            "checkpoint_artifacts_included":
                False,

            "temporary_files_included":
                False,

            "final_noncheckpoint_artifacts":
                len(
                    inventory
                ),

            "all_inventory_entries_required":
                True,

            "all_inventory_entries_integrity_pass":
                True,
        },

    "inventory":
        {
            "file":
                INVENTORY_FILE.as_posix(),

            "sha256":
                inventory_hash,

            "rows":
                len(
                    inventory
                ),
        },

    "required_contract_statuses":
        sorted(
            REQUIRED_STATUSES
        ),

    "observed_contract_statuses":
        sorted(
            observed_statuses
        ),

    "contract_files_discovered":
        len(
            contract_files
        ),

    "final_synthesis":
        {
            "contract_file":
                synthesis_contract_path.as_posix(),

            "contract_sha256":
                sha256_file(
                    synthesis_contract_path
                ),

            "status":
                "FROZEN_CROSS_CYCLE_SHAP_SYNTHESIS",

            "cross_model_primary_pattern":
                "MAJORITY_MODEL_PATTERN",

            "dominant_primary_evidence_tier":
                "TIER_2",

            "tier_counts":
                {
                    "TIER_0":
                        1,

                    "TIER_1":
                        0,

                    "TIER_2":
                        2,
                },

            "model_tiers":
                expected_model_tiers,

            "automatic_binary_drift_label":
                False,

            "automatic_binary_stability_label":
                False,
        },

    "scientific_state":
        {
            "cycle25_used_for_model_training":
                False,

            "models_retrained_in_chapter10":
                False,

            "preprocessing_refit_in_chapter10":
                False,

            "decision_thresholds_retuned":
                False,

            "cycle24_shap_null_recomputed":
                False,

            "correlated_groups_recomputed":
                False,

            "support_weights_recomputed_after_sensitivity":
                False,

            "featurewise_fdr_completed":
                True,

            "correlated_group_robustness_completed":
                True,

            "support_reweighting_sensitivity_completed":
                True,

            "class_conditional_analysis_completed":
                True,

            "temporal_block_analysis_completed":
                True,

            "final_cross_cycle_shap_synthesis_completed":
                True,

            "shap_interpreted_as_causal":
                False,
        },

    "interpretation":
        (
            "Chapter 10 is frozen as the final "
            "cross-cycle SHAP evidence package. "
            "The synthesis shows a majority-model "
            "TIER_2 pattern: RF and XGB exhibit both "
            "global structure-null exceedance and "
            "featurewise BH-FDR evidence, while LR "
            "does not exceed the global structure null "
            "despite featurewise attribution changes. "
            "This pattern is model-dependent and is not "
            "converted into an automatic universal binary "
            "drift/stability label."
        ),

    "next_stage":
        (
            "CHAPTER_10_GIT_CHECKPOINT_"
            "THEN_FINAL_MODEL_SYNTHESIS"
        ),
}


# ============================================================
# SAVE CONTRACT
# ============================================================

CONTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


contract_temp = FINAL_CONTRACT_FILE.with_name(
    FINAL_CONTRACT_FILE.stem
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
    contract_rt.get(
        "status"
    )
    !=
    "FROZEN_CHAPTER10_CROSS_CYCLE_SHAP"
):

    raise RuntimeError(
        "Final Chapter-10 contract "
        "round-trip failed."
    )


if (
    contract_rt[
        "scientific_state"
    ][
        "cycle25_used_for_model_training"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected Cycle25 training state."
    )


if (
    contract_rt[
        "scientific_state"
    ][
        "final_cross_cycle_shap_synthesis_completed"
    ]
    is not True
):

    raise RuntimeError(
        "Final SHAP synthesis not marked complete."
    )


contract_temp.replace(
    FINAL_CONTRACT_FILE
)


contract_hash = sha256_file(
    FINAL_CONTRACT_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 10.10.3 — COMPLETE"
)

print("=" * 78)


print(
    "Final non-checkpoint artifacts:",
    len(
        inventory
    )
)

print(
    "Required contract statuses:",
    len(
        REQUIRED_STATUSES
    )
)

print(
    "Observed contract statuses:",
    len(
        observed_statuses
    )
)


print()

print(
    "Cross-model pattern:",
    "MAJORITY_MODEL_PATTERN"
)

print(
    "Dominant tier:",
    "TIER_2"
)

print(
    "LR tier:",
    "TIER_0"
)

print(
    "RF tier:",
    "TIER_2"
)

print(
    "XGB tier:",
    "TIER_2"
)


print()

print(
    "Automatic binary drift label:",
    False
)

print(
    "Automatic binary stability label:",
    False
)


print()

print(
    "Status:",
    "FROZEN_CHAPTER10_CROSS_CYCLE_SHAP"
)


print()

print(
    "Inventory:",
    INVENTORY_FILE
)

print(
    "Inventory SHA256:",
    inventory_hash
)

print(
    "Contract:",
    FINAL_CONTRACT_FILE
)

print(
    "Contract SHA256:",
    contract_hash
)


print()

print(
    "NEXT:",
    "Chapter-10 Git checkpoint"
)