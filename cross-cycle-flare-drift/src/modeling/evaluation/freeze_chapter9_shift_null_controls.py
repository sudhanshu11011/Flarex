# ============================================================
# MODEL CHAPTER 9.7
# FREEZE CHAPTER 9 SHIFT + NULL CONTROLS
#
# Purpose
# -------
# Create an immutable, hash-addressed Chapter-9 control
# inventory covering:
#
#   9.1  Frozen shift/null protocol
#   9.2  Marginal feature shift
#   9.3  Prevalence + active-region population
#   9.4  Joint domain shift + common support
#   9.5  Correlated-feature groups
#   9.6A Frozen SHAP background
#   9.6B Frozen Cycle24 HOLDOUT SHAP
#   9.6C Within-Cycle24 SHAP null execution
#   9.6D Frozen within-Cycle24 SHAP null baseline
#
# This stage:
#   - DOES NOT train models
#   - DOES NOT compute SHAP
#   - DOES NOT recompute bootstrap results
#   - DOES NOT interpret Cycle25 explanation drift
#   - DOES NOT modify earlier artifacts
#
# It only validates, inventories, hashes, and freezes them.
# ============================================================


from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# ROOT
# ============================================================

SHIFT_ROOT = Path(
    "artifacts/models/shift_controls"
)


# ============================================================
# OUTPUTS
# ============================================================

FINAL_INVENTORY_FILE = (
    SHIFT_ROOT
    /
    "manifests"
    /
    "chapter9_shift_null_controls_inventory.csv"
)


FINAL_CONTRACT_FILE = (
    SHIFT_ROOT
    /
    "contracts"
    /
    "chapter9_shift_null_controls_contract.json"
)


# ============================================================
# EXACT PRE-9.7 ARTIFACT INVENTORY
#
# These paths come from the actual repository inventory.
# Checkpoint files are intentionally excluded.
# ============================================================

ARTIFACTS = [

    # --------------------------------------------------------
    # 9.1 — MASTER SHIFT / NULL PROTOCOL
    # --------------------------------------------------------

    {
        "chapter_stage": "9.1",
        "role": "SHIFT_NULL_PROTOCOL",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "shift_null_control_protocol.json"
        ),
        "required": True,
    },


    # --------------------------------------------------------
    # 9.2 — MARGINAL FEATURE SHIFT
    # --------------------------------------------------------

    {
        "chapter_stage": "9.2",
        "role": "MARGINAL_SHIFT_BOOTSTRAP_REPLICATES",
        "path": Path(
            "artifacts/models/shift_controls/bootstrap/"
            "marginal_shift_group_bootstrap_replicates.parquet"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.2",
        "role": "MARGINAL_SHIFT_POINT_CONTRACT",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "marginal_feature_shift_point_contract.json"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.2",
        "role": "MARGINAL_SHIFT_BOOTSTRAP_CONTRACT",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "marginal_feature_shift_bootstrap_contract.json"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.2",
        "role": "MARGINAL_SHIFT_POINT_METRICS",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "marginal_feature_shift_point.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.2",
        "role": "MARGINAL_SHIFT_BOOTSTRAP_CI",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "marginal_feature_shift_bootstrap_ci.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.2",
        "role": "MISSINGNESS_AUDIT",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "missingness_audit.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.2",
        "role": "SHIFT_FEATURE_REGISTRY",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "shift_feature_registry.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.2",
        "role": "SHIFT_ANALYSIS_REGISTRY",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "shift_analysis_registry.csv"
        ),
        "required": True,
    },


    # --------------------------------------------------------
    # 9.3 — PREVALENCE + ACTIVE REGION POPULATION
    # --------------------------------------------------------

    {
        "chapter_stage": "9.3",
        "role": "LABEL_PREVALENCE_BOOTSTRAP",
        "path": Path(
            "artifacts/models/shift_controls/bootstrap/"
            "label_prevalence_group_bootstrap.parquet"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.3",
        "role": "PREVALENCE_AR_CONTRACT",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "prevalence_ar_population_contract.json"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.3",
        "role": "LABEL_PREVALENCE_METRICS",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "label_prevalence.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.3",
        "role": "ACTIVE_REGION_POPULATION_METRICS",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "active_region_population.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.3",
        "role": "ACTIVE_REGION_GROUP_SUMMARY",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "active_region_group_summary.csv"
        ),
        "required": True,
    },


    # --------------------------------------------------------
    # 9.4 — JOINT DOMAIN SHIFT + COMMON SUPPORT
    # --------------------------------------------------------

    {
        "chapter_stage": "9.4",
        "role": "JOINT_SHIFT_SUPPORT_CONTRACT",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "joint_shift_common_support_contract.json"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.4",
        "role": "JOINT_DOMAIN_SHIFT_CV",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "joint_domain_shift_cv.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.4",
        "role": "COMMON_SUPPORT_SUMMARY",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "common_support_summary.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.4",
        "role": "DOMAIN_OOF_COMMON_SUPPORT",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "domain_oof_common_support.parquet"
        ),
        "required": True,
    },


    # --------------------------------------------------------
    # 9.5 — CORRELATED FEATURE GROUPS
    # --------------------------------------------------------

    {
        "chapter_stage": "9.5",
        "role": "CORRELATED_FEATURE_GROUPS_CONTRACT",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "correlated_feature_groups_contract.json"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.5",
        "role": "CYCLE24_TRAIN_SPEARMAN_MATRIX",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "cycle24_train_spearman_correlation.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.5",
        "role": "CORRELATED_FEATURE_PAIRS",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "correlated_feature_pairs.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.5",
        "role": "CORRELATED_FEATURE_GROUPS",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "correlated_feature_groups.csv"
        ),
        "required": True,
    },


    # --------------------------------------------------------
    # 9.6A — FROZEN SHAP BACKGROUND
    # --------------------------------------------------------

    {
        "chapter_stage": "9.6A",
        "role": "SHAP_BACKGROUND_CONTRACT",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "shap_background_contract.json"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6A",
        "role": "SHAP_BACKGROUND_MANIFEST",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "shap_background_manifest.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6A",
        "role": "SHAP_BACKGROUND_DATA",
        "path": Path(
            "artifacts/models/shift_controls/shap/background/"
            "cycle24_train_shap_background.parquet"
        ),
        "required": True,
    },


    # --------------------------------------------------------
    # 9.6B — CYCLE24 HOLDOUT SHAP
    # --------------------------------------------------------

    {
        "chapter_stage": "9.6B",
        "role": "CYCLE24_HOLDOUT_SHAP_CONTRACT",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "cycle24_holdout_shap_contract.json"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6B",
        "role": "RF_SHAP_AUDIT",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "rf_cycle24_holdout_shap_parallel_audit.json"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6B",
        "role": "XGB_SHAP_AUDIT",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "xgb_cycle24_holdout_shap_parallel_audit.json"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6B",
        "role": "CYCLE24_HOLDOUT_SHAP_REGISTRY",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "cycle24_holdout_shap_registry.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6B",
        "role": "RF_SHAP_CHUNK_MANIFEST",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "rf_cycle24_holdout_shap_chunk_manifest.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6B",
        "role": "XGB_SHAP_CHUNK_MANIFEST",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "xgb_cycle24_holdout_shap_chunk_manifest.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6B",
        "role": "CYCLE24_HOLDOUT_SHAP_SUMMARY",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "cycle24_holdout_shap_summary.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6B",
        "role": "RF_SHAP_SUMMARY",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "rf_cycle24_holdout_shap_summary.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6B",
        "role": "XGB_SHAP_SUMMARY",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "xgb_cycle24_holdout_shap_summary.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6B",
        "role": "LR_HOLDOUT_SHAP",
        "path": Path(
            "artifacts/models/shift_controls/shap/cycle24_holdout/"
            "LR_02_C0p01_BALANCED.parquet"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6B",
        "role": "RF_HOLDOUT_SHAP",
        "path": Path(
            "artifacts/models/shift_controls/shap/cycle24_holdout/"
            "RF_04_D12_L5_BALSUB.parquet"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6B",
        "role": "XGB_HOLDOUT_SHAP",
        "path": Path(
            "artifacts/models/shift_controls/shap/cycle24_holdout/"
            "XGB_02_LR0p03_D3_BAL.parquet"
        ),
        "required": True,
    },


    # --------------------------------------------------------
    # 9.6C — WITHIN-C24 SHAP NULL EXECUTION
    # --------------------------------------------------------

    {
        "chapter_stage": "9.6C",
        "role": "SHAP_NULL_EXECUTION_CONTRACT",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "cycle24_shap_null_execution_contract.json"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6C",
        "role": "SHAP_NULL_EXECUTION_MANIFEST",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "cycle24_shap_null_execution_manifest.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6C",
        "role": "SHAP_NULL_GROUP_ORDER",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "cycle24_shap_null_group_order.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6C",
        "role": "SHAP_NULL_DRAW_COUNTS",
        "path": Path(
            "artifacts/models/shift_controls/shap/null/"
            "cycle24_shap_null_draw_counts.npz"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6C",
        "role": "SHAP_NULL_STRUCTURE_REPLICATES",
        "path": Path(
            "artifacts/models/shift_controls/shap/null/"
            "cycle24_shap_null_replicates.parquet"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6C",
        "role": "SHAP_NULL_FEATUREWISE_REPLICATES",
        "path": Path(
            "artifacts/models/shift_controls/shap/null/"
            "cycle24_shap_null_featurewise_replicates.parquet"
        ),
        "required": True,
    },


    # --------------------------------------------------------
    # 9.6D — FROZEN SHAP NULL BASELINE
    # --------------------------------------------------------

    {
        "chapter_stage": "9.6D",
        "role": "SHAP_NULL_FINAL_CONTRACT",
        "path": Path(
            "artifacts/models/shift_controls/contracts/"
            "cycle24_shap_null_contract.json"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6D",
        "role": "SHAP_NULL_INTEGRITY_MANIFEST",
        "path": Path(
            "artifacts/models/shift_controls/manifests/"
            "cycle24_shap_null_integrity_manifest.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6D",
        "role": "SHAP_NULL_DISTRIBUTION_SUMMARY",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "cycle24_shap_null_distribution_summary.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6D",
        "role": "SHAP_NULL_STRUCTURE_THRESHOLDS",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "cycle24_shap_null_structure_thresholds.csv"
        ),
        "required": True,
    },

    {
        "chapter_stage": "9.6D",
        "role": "SHAP_NULL_FEATUREWISE_THRESHOLDS",
        "path": Path(
            "artifacts/models/shift_controls/metrics/"
            "cycle24_shap_null_featurewise_thresholds.csv"
        ),
        "required": True,
    },
]


# ============================================================
# CRITICAL CONTRACT STATUS ASSERTIONS
#
# Only statuses already frozen/known from earlier phases
# are hard-coded.
# ============================================================

CRITICAL_CONTRACT_STATUSES = {

    Path(
        "artifacts/models/shift_controls/contracts/"
        "shift_null_control_protocol.json"
    ):
        "FROZEN_BEFORE_SHIFT_AND_SHAP_ANALYSIS",

    Path(
        "artifacts/models/shift_controls/contracts/"
        "shap_background_contract.json"
    ):
        "FROZEN_SHAP_BACKGROUND",

    Path(
        "artifacts/models/shift_controls/contracts/"
        "cycle24_holdout_shap_contract.json"
    ):
        "FROZEN_CYCLE24_HOLDOUT_SHAP",

    Path(
        "artifacts/models/shift_controls/contracts/"
        "cycle24_shap_null_execution_contract.json"
    ):
        (
            "CYCLE24_SHAP_NULL_COMPUTED_"
            "PENDING_9_6D_AUDIT"
        ),

    Path(
        "artifacts/models/shift_controls/contracts/"
        "cycle24_shap_null_contract.json"
    ):
        "FROZEN_CYCLE24_SHAP_NULL_BASELINE",
}


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

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def normalize_path(path: Path) -> str:

    return path.as_posix()


def inspect_artifact(path: Path):

    suffix = path.suffix.lower()

    result = {
        "artifact_type": suffix.lstrip("."),
        "rows": None,
        "columns": None,
        "json_status": None,
        "npz_arrays": None,
    }


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if suffix == ".json":

        payload = load_json(
            path
        )


        if not isinstance(
            payload,
            dict,
        ):

            raise RuntimeError(
                f"Expected JSON object: {path}"
            )


        result[
            "json_status"
        ] = payload.get(
            "status"
        )


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    elif suffix == ".csv":

        frame = pd.read_csv(
            path,
            keep_default_na=False,
        )


        result[
            "rows"
        ] = int(
            len(frame)
        )


        result[
            "columns"
        ] = int(
            len(
                frame.columns
            )
        )


        if len(frame) == 0:

            raise RuntimeError(
                f"Unexpected empty CSV: {path}"
            )


    # --------------------------------------------------------
    # PARQUET
    # --------------------------------------------------------

    elif suffix == ".parquet":

        parquet_file = pq.ParquetFile(
            path
        )


        result[
            "rows"
        ] = int(
            parquet_file.metadata.num_rows
        )


        result[
            "columns"
        ] = int(
            parquet_file.metadata.num_columns
        )


        if (
            parquet_file.metadata.num_rows
            <=
            0
        ):

            raise RuntimeError(
                f"Unexpected empty parquet: {path}"
            )


    # --------------------------------------------------------
    # NPZ
    # --------------------------------------------------------

    elif suffix == ".npz":

        with np.load(
            path
        ) as archive:

            names = sorted(
                archive.files
            )


            if len(names) == 0:

                raise RuntimeError(
                    f"Empty NPZ archive: {path}"
                )


            result[
                "npz_arrays"
            ] = ";".join(
                names
            )


            # Specific 9.6C draw-count audit.
            if (
                path.name
                ==
                "cycle24_shap_null_draw_counts.npz"
            ):

                required = {
                    "counts_a",
                    "counts_b",
                }


                if not required.issubset(
                    set(names)
                ):

                    raise RuntimeError(
                        "SHAP null draw-count NPZ "
                        "missing counts_a/counts_b."
                    )


                counts_a = archive[
                    "counts_a"
                ]


                counts_b = archive[
                    "counts_b"
                ]


                if counts_a.shape != (
                    2000,
                    176,
                ):

                    raise RuntimeError(
                        "counts_a shape mismatch."
                    )


                if counts_b.shape != (
                    2000,
                    176,
                ):

                    raise RuntimeError(
                        "counts_b shape mismatch."
                    )


    else:

        raise RuntimeError(
            f"Unsupported artifact type: {path}"
        )


    return result


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 9.7 — "
    "FREEZE SHIFT + NULL CONTROLS"
)

print("=" * 78)


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for output in [
    FINAL_INVENTORY_FILE,
    FINAL_CONTRACT_FILE,
]:

    if output.exists():

        raise RuntimeError(
            f"Output already exists: {output}\n"
            "Refusing to overwrite frozen "
            "Chapter 9.7 outputs."
        )


# ============================================================
# VERIFY EXACT INVENTORY DEFINITION
# ============================================================

defined_paths = [
    entry[
        "path"
    ]
    for entry
    in ARTIFACTS
]


if len(
    defined_paths
) != len(
    set(
        defined_paths
    )
):

    raise RuntimeError(
        "Duplicate artifact path in "
        "Chapter 9.7 inventory definition."
    )


print(
    "Defined freeze artifacts:",
    len(
        defined_paths
    )
)


# ============================================================
# VERIFY ALL REQUIRED FILES EXIST
# ============================================================

missing = [
    path
    for path
    in defined_paths
    if not path.exists()
]


if missing:

    print()

    print(
        "Missing required artifacts:"
    )


    for path in missing:

        print(
            "  ",
            path
        )


    raise RuntimeError(
        "Chapter 9.7 cannot freeze "
        "with missing artifacts."
    )


print(
    "Required artifact existence:",
    "PASS"
)


# ============================================================
# VERIFY THAT EVERY CURRENT NON-CHECKPOINT PRE-9.7 FILE
# IS ACCOUNTED FOR
# ============================================================

current_files = []


for path in SHIFT_ROOT.rglob("*"):

    if not path.is_file():

        continue


    normalized = (
        normalize_path(
            path
        )
    )


    # Skip execution checkpoint directory.
    if "/checkpoints/" in normalized:

        continue


    # 9.7 outputs do not exist yet due to output protection,
    # but keep this guard for clarity.
    if path in {
        FINAL_INVENTORY_FILE,
        FINAL_CONTRACT_FILE,
    }:

        continue


    current_files.append(
        path
    )


current_set = set(
    current_files
)


defined_set = set(
    defined_paths
)


unexpected_files = sorted(
    current_set
    -
    defined_set,
    key=lambda value:
        normalize_path(
            value
        ),
)


undefined_expected = sorted(
    defined_set
    -
    current_set,
    key=lambda value:
        normalize_path(
            value
        ),
)


if unexpected_files:

    print()

    print(
        "Unexpected non-checkpoint files "
        "not included in freeze inventory:"
    )


    for path in unexpected_files:

        print(
            "  ",
            path
        )


    raise RuntimeError(
        "Inventory is incomplete. "
        "Review unexpected files before freeze."
    )


if undefined_expected:

    raise RuntimeError(
        "Defined artifacts missing from "
        "current inventory."
    )


print(
    "Exact non-checkpoint inventory coverage:",
    "PASS"
)


# ============================================================
# CRITICAL CONTRACT STATUS AUDIT
# ============================================================

for (
    contract_path,
    expected_status
) in CRITICAL_CONTRACT_STATUSES.items():

    payload = load_json(
        contract_path
    )


    actual_status = payload.get(
        "status"
    )


    if (
        actual_status
        !=
        expected_status
    ):

        raise RuntimeError(
            "Critical contract status mismatch:\n"
            f"{contract_path}\n"
            f"Expected: {expected_status}\n"
            f"Actual:   {actual_status}"
        )


print(
    "Critical contract statuses:",
    "PASS"
)


# ============================================================
# CROSS-CONTRACT SCIENTIFIC GUARDRAILS
# ============================================================

shap_contract = load_json(
    Path(
        "artifacts/models/shift_controls/contracts/"
        "cycle24_holdout_shap_contract.json"
    )
)


null_execution_contract = load_json(
    Path(
        "artifacts/models/shift_controls/contracts/"
        "cycle24_shap_null_execution_contract.json"
    )
)


null_contract = load_json(
    Path(
        "artifacts/models/shift_controls/contracts/"
        "cycle24_shap_null_contract.json"
    )
)


# ------------------------------------------------------------
# 9.6B
# ------------------------------------------------------------

if (
    shap_contract[
        "scientific_guardrails"
    ][
        "cycle25_read"
    ]
    is not False
):

    raise RuntimeError(
        "9.6B unexpectedly read Cycle25."
    )


if (
    shap_contract[
        "scientific_guardrails"
    ][
        "models_retrained"
    ]
    is not False
):

    raise RuntimeError(
        "9.6B model-retraining guardrail failed."
    )


if (
    shap_contract[
        "scientific_guardrails"
    ][
        "shap_approximation"
    ]
    is not False
):

    raise RuntimeError(
        "9.6B SHAP approximation guardrail failed."
    )


# ------------------------------------------------------------
# 9.6C
# ------------------------------------------------------------

if (
    null_execution_contract[
        "scientific_guardrails"
    ][
        "cycle25_read"
    ]
    is not False
):

    raise RuntimeError(
        "9.6C unexpectedly read Cycle25."
    )


if (
    null_execution_contract[
        "scientific_guardrails"
    ][
        "shap_recomputed"
    ]
    is not False
):

    raise RuntimeError(
        "9.6C unexpectedly recomputed SHAP."
    )


if (
    null_execution_contract[
        "scientific_guardrails"
    ][
        "models_retrained"
    ]
    is not False
):

    raise RuntimeError(
        "9.6C model-retraining guardrail failed."
    )


# ------------------------------------------------------------
# 9.6D
# ------------------------------------------------------------

if (
    null_contract[
        "scientific_guardrails"
    ][
        "cycle25_read"
    ]
    is not False
):

    raise RuntimeError(
        "9.6D unexpectedly read Cycle25."
    )


if (
    null_contract[
        "scientific_guardrails"
    ][
        "cross_cycle_drift_claim_made"
    ]
    is not False
):

    raise RuntimeError(
        "9.6D made a cross-cycle drift claim."
    )


if (
    null_contract[
        "scientific_guardrails"
    ][
        "null_thresholds_frozen_before_cycle25_shap"
    ]
    is not True
):

    raise RuntimeError(
        "Null thresholds were not frozen "
        "before Cycle25 SHAP."
    )


print(
    "9.6 scientific guardrails:",
    "PASS"
)


# ============================================================
# EXPECTED KEY 9.6 ARTIFACT SIZES
# ============================================================

expected_parquet_rows = {

    Path(
        "artifacts/models/shift_controls/shap/background/"
        "cycle24_train_shap_background.parquet"
    ):
        512,

    Path(
        "artifacts/models/shift_controls/shap/cycle24_holdout/"
        "LR_02_C0p01_BALANCED.parquet"
    ):
        13072,

    Path(
        "artifacts/models/shift_controls/shap/cycle24_holdout/"
        "RF_04_D12_L5_BALSUB.parquet"
    ):
        13072,

    Path(
        "artifacts/models/shift_controls/shap/cycle24_holdout/"
        "XGB_02_LR0p03_D3_BAL.parquet"
    ):
        13072,

    Path(
        "artifacts/models/shift_controls/shap/null/"
        "cycle24_shap_null_replicates.parquet"
    ):
        18000,

    Path(
        "artifacts/models/shift_controls/shap/null/"
        "cycle24_shap_null_featurewise_replicates.parquet"
    ):
        450000,
}


# ============================================================
# BUILD HASHED INVENTORY
# ============================================================

inventory_rows = []


print()

print(
    "Hashing + auditing Chapter 9 artifacts..."
)


for index, entry in enumerate(
    ARTIFACTS,
    start=1,
):

    path = entry[
        "path"
    ]


    details = inspect_artifact(
        path
    )


    # --------------------------------------------------------
    # Key known row-count checks
    # --------------------------------------------------------

    if path in expected_parquet_rows:

        expected_rows = (
            expected_parquet_rows[
                path
            ]
        )


        if (
            details[
                "rows"
            ]
            !=
            expected_rows
        ):

            raise RuntimeError(
                f"Row count mismatch: {path}\n"
                f"Expected: {expected_rows}\n"
                f"Actual:   {details['rows']}"
            )


    # --------------------------------------------------------
    # Critical JSON status if applicable
    # --------------------------------------------------------

    expected_status = (
        CRITICAL_CONTRACT_STATUSES.get(
            path
        )
    )


    if (
        expected_status is not None
        and
        details[
            "json_status"
        ]
        !=
        expected_status
    ):

        raise RuntimeError(
            f"Status changed during audit: {path}"
        )


    file_hash = sha256_file(
        path
    )


    file_size = int(
        path.stat().st_size
    )


    if file_size <= 0:

        raise RuntimeError(
            f"Empty artifact file: {path}"
        )


    inventory_rows.append(
        {
            "inventory_index":
                index,

            "chapter_stage":
                entry[
                    "chapter_stage"
                ],

            "role":
                entry[
                    "role"
                ],

            "path":
                normalize_path(
                    path
                ),

            "artifact_type":
                details[
                    "artifact_type"
                ],

            "bytes":
                file_size,

            "rows":
                (
                    ""
                    if details[
                        "rows"
                    ] is None
                    else
                    details[
                        "rows"
                    ]
                ),

            "columns":
                (
                    ""
                    if details[
                        "columns"
                    ] is None
                    else
                    details[
                        "columns"
                    ]
                ),

            "json_status":
                (
                    ""
                    if details[
                        "json_status"
                    ] is None
                    else
                    str(
                        details[
                            "json_status"
                        ]
                    )
                ),

            "npz_arrays":
                (
                    ""
                    if details[
                        "npz_arrays"
                    ] is None
                    else
                    details[
                        "npz_arrays"
                    ]
                ),

            "sha256":
                file_hash,

            "required":
                bool(
                    entry[
                        "required"
                    ]
                ),

            "integrity_pass":
                True,
        }
    )


    print(
        f"[{index:02d}/{len(ARTIFACTS):02d}] "
        f"{entry['chapter_stage']} "
        f"{entry['role']}: PASS"
    )


inventory_df = pd.DataFrame(
    inventory_rows
)


if len(
    inventory_df
) != len(
    ARTIFACTS
):

    raise RuntimeError(
        "Final inventory row-count mismatch."
    )


if inventory_df[
    "path"
].duplicated().any():

    raise RuntimeError(
        "Duplicate artifact path in "
        "final inventory."
    )


if not inventory_df[
    "integrity_pass"
].all():

    raise RuntimeError(
        "At least one Chapter 9 artifact "
        "failed integrity."
    )


print()

print(
    "Artifact integrity:",
    "PASS"
)


# ============================================================
# STAGE COUNTS
# ============================================================

stage_counts = (
    inventory_df
    .groupby(
        "chapter_stage",
        sort=False,
    )
    .size()
    .to_dict()
)


print()

print(
    "Freeze inventory by stage:"
)


for stage, count in stage_counts.items():

    print(
        f"  {stage}: {count}"
    )


# ============================================================
# SAVE INVENTORY
# ============================================================

FINAL_INVENTORY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


FINAL_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


inventory_df.to_csv(
    FINAL_INVENTORY_FILE,
    index=False,
)


inventory_rt = pd.read_csv(
    FINAL_INVENTORY_FILE,
    keep_default_na=False,
)


if len(
    inventory_rt
) != len(
    inventory_df
):

    raise RuntimeError(
        "Final inventory round-trip failed."
    )


inventory_hash = sha256_file(
    FINAL_INVENTORY_FILE
)


print(
    "Freeze inventory round-trip:",
    "PASS"
)


# ============================================================
# IMPORTANT FROZEN SCIENCE SUMMARY
# ============================================================

scientific_state = {

    "cycle24_holdout_shap": {
        "status":
            "FROZEN",

        "models":
            [
                "LR_02_C0p01_BALANCED",
                "RF_04_D12_L5_BALSUB",
                "XGB_02_LR0p03_D3_BAL",
            ],

        "rows_per_model":
            13072,

        "features":
            25,

        "background_rows":
            512,

        "cycle25_read_during_shap_generation":
            False,

        "model_retraining":
            False,
    },


    "within_cycle24_shap_null": {
        "status":
            "FROZEN",

        "bootstrap_unit":
            "ENTITY_GROUP_ID",

        "entity_groups":
            176,

        "replicates":
            2000,

        "seed_a":
            29062,

        "seed_b":
            29063,

        "same_draws_across_models":
            True,

        "same_draws_across_strata":
            True,

        "strata":
            [
                "OVERALL",
                "TARGET_0",
                "TARGET_1",
            ],

        "cycle25_read":
            False,
    },


    "null_thresholds": {
        "status":
            "FROZEN",

        "similarity_rule":
            (
                "future observed similarity "
                "strictly below within-Cycle24 "
                "2.5th percentile"
            ),

        "distance_rule":
            (
                "future observed distance "
                "strictly above within-Cycle24 "
                "97.5th percentile"
            ),

        "featurewise_empirical_null_retained":
            True,

        "future_bh_fdr_q":
            0.05,
    },


    "shift_controls": {
        "marginal_shift":
            "COMPLETE",

        "prevalence_active_region":
            "COMPLETE",

        "joint_domain_common_support":
            "COMPLETE",

        "correlated_feature_groups":
            "COMPLETE",
    },
}


# ============================================================
# FINAL CHAPTER 9 CONTRACT
# ============================================================

final_contract = {

    "contract_name":
        (
            "Frozen Chapter 9 "
            "Distribution-Shift and SHAP-Null Controls"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CHAPTER9_SHIFT_NULL_CONTROLS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_9_7",

    "purpose":
        (
            "Freeze all Chapter-9 distribution-shift, "
            "common-support, correlated-feature, "
            "Cycle24 SHAP, and within-Cycle24 SHAP-null "
            "control artifacts before cross-cycle "
            "Cycle25 SHAP attribution analysis."
        ),

    "source_stages":
        [
            "9.1",
            "9.2",
            "9.3",
            "9.4",
            "9.5",
            "9.6A",
            "9.6B",
            "9.6C",
            "9.6D",
        ],

    "artifact_inventory":
        {
            "file":
                normalize_path(
                    FINAL_INVENTORY_FILE
                ),

            "sha256":
                inventory_hash,

            "rows":
                int(
                    len(
                        inventory_df
                    )
                ),

            "all_integrity_pass":
                True,

            "checkpoint_files_included":
                False,
        },

    "stage_artifact_counts":
        {
            str(stage):
                int(count)
            for stage, count
            in stage_counts.items()
        },

    "critical_contracts":
        {
            normalize_path(
                path
            ):
                {
                    "required_status":
                        expected_status,

                    "sha256":
                        sha256_file(
                            path
                        ),
                }
            for path, expected_status
            in CRITICAL_CONTRACT_STATUSES.items()
        },

    "scientific_state":
        scientific_state,

    "cycle25_usage_scope":
        {
            "chapter9_shift_audits":
                (
                    "Cycle25 distribution/prevalence/"
                    "domain-shift information was legitimately "
                    "used in the predefined Chapter 9.2-9.4 "
                    "transfer-shift diagnostics."
                ),

            "cycle24_shap_generation":
                "NO_CYCLE25_READ",

            "within_cycle24_shap_null":
                "NO_CYCLE25_READ",

            "cycle25_shap_attributions_generated_yet":
                False,
        },

    "frozen_rules_for_chapter10":
        {
            "models_must_remain_frozen":
                True,

            "feature_order_must_remain_frozen":
                True,

            "preprocessing_must_remain_frozen":
                True,

            "classification_thresholds_must_remain_frozen":
                True,

            "shap_background_must_remain_frozen":
                True,

            "shap_output_space_per_model_must_remain_frozen":
                True,

            "within_cycle24_null_must_remain_frozen":
                True,

            "null_thresholds_must_not_be_changed_after_cycle25_shap":
                True,

            "correlated_feature_groups_must_remain_frozen":
                True,

            "common_support_definition_must_remain_frozen":
                True,

            "cycle25_must_not_be_used_for_tuning":
                True,
        },

    "interpretation_guardrails":
        {
            "distribution_shift_is_not_explanation_drift":
                True,

            "domain_classifier_auc_is_not_explanation_drift":
                True,

            "null_threshold_crossing_alone_is_not_final_drift_claim":
                True,

            "shap_is_not_causal":
                True,

            "raw_shap_magnitudes_must_not_be_compared_between_model_families":
                True,

            "same_model_cross_population_comparison_required":
                True,

            "class_conditional_robustness_required":
                True,

            "common_support_reweighting_sensitivity_required":
                True,

            "correlated_feature_group_robustness_required":
                True,

            "featurewise_empirical_inference_and_bh_fdr_required_where_applicable":
                True,
        },

    "chapter9_completion":
        {
            "9.1_protocol":
                "COMPLETE",

            "9.2_marginal_shift":
                "COMPLETE",

            "9.3_prevalence_ar":
                "COMPLETE",

            "9.4_joint_shift_common_support":
                "COMPLETE",

            "9.5_correlated_feature_groups":
                "COMPLETE",

            "9.6a_shap_background":
                "COMPLETE",

            "9.6b_cycle24_holdout_shap":
                "COMPLETE",

            "9.6c_cycle24_shap_null_execution":
                "COMPLETE",

            "9.6d_null_thresholds":
                "COMPLETE",

            "9.7_final_freeze":
                "COMPLETE",
        },

    "next_stage":
        "MODEL_CHAPTER_9_8_GIT_CHECKPOINT",

    "immutability_rule":
        (
            "After this Chapter 9.7 contract is created, "
            "all inventoried Chapter-9 scientific control "
            "artifacts are treated as immutable inputs to "
            "Chapter 10. Any change requires explicit "
            "versioning, rationale, and invalidation of "
            "downstream cross-cycle SHAP results."
        ),
}


with FINAL_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        final_contract,
        file,
        indent=2,
    )


# ============================================================
# FINAL CONTRACT ROUND-TRIP
# ============================================================

contract_rt = load_json(
    FINAL_CONTRACT_FILE
)


if (
    contract_rt.get(
        "status"
    )
    !=
    "FROZEN_CHAPTER9_SHIFT_NULL_CONTROLS"
):

    raise RuntimeError(
        "Chapter 9.7 contract "
        "round-trip failed."
    )


if (
    contract_rt[
        "artifact_inventory"
    ][
        "rows"
    ]
    !=
    len(
        ARTIFACTS
    )
):

    raise RuntimeError(
        "Chapter 9.7 contract "
        "inventory-count mismatch."
    )


if (
    contract_rt[
        "chapter9_completion"
    ][
        "9.7_final_freeze"
    ]
    !=
    "COMPLETE"
):

    raise RuntimeError(
        "Chapter 9.7 completion "
        "state mismatch."
    )


final_contract_hash = sha256_file(
    FINAL_CONTRACT_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 9.7 — COMPLETE"
)

print("=" * 78)


print(
    "Frozen artifacts:",
    len(
        inventory_df
    )
)

print(
    "Artifact integrity:",
    "PASS"
)

print(
    "Critical contract statuses:",
    "PASS"
)

print(
    "9.6 scientific guardrails:",
    "PASS"
)

print(
    "Checkpoint intermediates included:",
    False
)

print(
    "Cycle25 SHAP generated:",
    False
)

print(
    "Chapter 9 controls immutable:",
    True
)

print(
    "Status:",
    "FROZEN_CHAPTER9_SHIFT_NULL_CONTROLS"
)


print()

print(
    "Inventory SHA256:",
    inventory_hash
)

print(
    "Contract SHA256:",
    final_contract_hash
)


print()

print(
    "Saved:",
    FINAL_INVENTORY_FILE
)

print(
    "Saved:",
    FINAL_CONTRACT_FILE
)


print()

print(
    "NEXT:",
    "Chapter 9.8 — Git checkpoint / commit / push"
)