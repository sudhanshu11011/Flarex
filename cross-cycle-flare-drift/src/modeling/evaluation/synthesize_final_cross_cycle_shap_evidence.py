# ============================================================
# MODEL CHAPTER 10.10.2
# FINAL CROSS-CYCLE SHAP EVIDENCE SYNTHESIS
#
# Uses the frozen 10.10.1B synthesis protocol.
#
# Primary model-level tier:
#
#   TIER_0
#     No OVERALL structure-null exceedance.
#
#   TIER_1
#     OVERALL structure-null exceedance,
#     but zero OVERALL FDR-significant hypotheses.
#
#   TIER_2
#     OVERALL structure-null exceedance,
#     and >=1 OVERALL FDR-significant hypothesis.
#
# Additional evidence retained as context/robustness:
#   - TARGET_0 / TARGET_1
#   - common-support sensitivity
#   - C25->C24 reweighting sensitivity
#   - correlated-feature-group robustness
#   - temporal B1/B2/B3 screening
#
# IMPORTANT:
#   - No new statistical threshold is created.
#   - No support-restricted null is invented.
#   - No correlated-group null is invented.
#   - Temporal block results remain screening evidence.
#   - No cross-model raw SHAP magnitude comparison.
#   - No causal interpretation.
#   - No automatic binary explanation-drift/stability label.
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_MODELS = 3

EXPECTED_STRUCTURE_SUMMARY_ROWS = 18
EXPECTED_SUPPORT_ROWS = 288
EXPECTED_GROUP_ROWS = 270
EXPECTED_FDR_ROWS = 18

NUMERIC_ZERO_TOLERANCE = 1e-12


MODEL_ORDER = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


STRUCTURE_METRICS = [
    "spearman_rho",
    "kendall_tau_b",
    "top5_overlap",
    "top10_overlap",
    "normalized_l1_mean_abs",
    "cosine_distance_mean_abs",
    "signed_cosine_similarity",
    "sign_agreement",
]


GROUP_METRICS = [
    "spearman_rho",
    "kendall_tau_b",
    "top5_overlap",
    "normalized_l1_mean_abs",
    "cosine_distance_mean_abs",
]


SIMILARITY_METRICS = {
    "spearman_rho",
    "kendall_tau_b",
    "top5_overlap",
    "top10_overlap",
    "signed_cosine_similarity",
    "sign_agreement",
}


DISTANCE_METRICS = {
    "normalized_l1_mean_abs",
    "cosine_distance_mean_abs",
}


FDR_QUANTITIES = [
    "ABS_DELTA_MEAN_ABS_SHAP",
    "ABS_DELTA_MEAN_SIGNED_SHAP",
]


# ============================================================
# FROZEN SYNTHESIS PROTOCOL
# ============================================================

SYNTHESIS_PROTOCOL_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "final_cross_cycle_shap_synthesis_protocol.json"
)


# ============================================================
# FROZEN EVIDENCE CONTRACTS
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


# ============================================================
# FROZEN EVIDENCE TABLES
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
# OUTPUTS
# ============================================================

MODEL_SYNTHESIS_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "final_cross_cycle_shap_model_synthesis.csv"
)


CROSS_MODEL_SUMMARY_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "final_cross_cycle_shap_cross_model_summary.csv"
)


OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "final_cross_cycle_shap_synthesis_contract.json"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(
    path: Path,
) -> str:

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


def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def verify_exists(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def parse_bool(
    value,
) -> bool:

    if isinstance(
        value,
        (bool, np.bool_),
    ):

        return bool(value)


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
        f"Cannot parse boolean value: {value}"
    )


def parse_feature_names(
    value,
):

    text = str(
        value
    ).strip()


    if text.lower() in {
        "",
        "nan",
        "none",
    }:

        return []


    return [
        item
        for item
        in text.split(";")
        if item
    ]


def directional_change(
    metric: str,
    delta_sensitivity_minus_primary: float,
) -> str:

    delta = float(
        delta_sensitivity_minus_primary
    )


    # Numerical equality only.
    # This is NOT a scientific/significance threshold.
    if abs(delta) <= NUMERIC_ZERO_TOLERANCE:

        return "UNCHANGED_NUMERICALLY"


    if metric in SIMILARITY_METRICS:

        if delta > 0:

            return "TOWARD_GREATER_SIMILARITY"

        return "TOWARD_GREATER_DIFFERENCE"


    if metric in DISTANCE_METRICS:

        if delta < 0:

            return "TOWARD_GREATER_SIMILARITY"

        return "TOWARD_GREATER_DIFFERENCE"


    raise RuntimeError(
        f"Unknown metric direction: {metric}"
    )


def summarize_directional_rows(
    frame: pd.DataFrame,
    delta_column: str,
):

    toward_similarity = 0
    toward_difference = 0
    unchanged = 0


    for _, row in frame.iterrows():

        result = directional_change(
            metric=str(
                row[
                    "metric"
                ]
            ),
            delta_sensitivity_minus_primary=float(
                row[
                    delta_column
                ]
            ),
        )


        if (
            result
            ==
            "TOWARD_GREATER_SIMILARITY"
        ):

            toward_similarity += 1


        elif (
            result
            ==
            "TOWARD_GREATER_DIFFERENCE"
        ):

            toward_difference += 1


        elif (
            result
            ==
            "UNCHANGED_NUMERICALLY"
        ):

            unchanged += 1


        else:

            raise RuntimeError(
                f"Unexpected directional result: {result}"
            )


    if (
        toward_similarity
        +
        toward_difference
        +
        unchanged
        !=
        len(frame)
    ):

        raise RuntimeError(
            "Directional summary count mismatch."
        )


    return {
        "toward_similarity":
            toward_similarity,

        "toward_difference":
            toward_difference,

        "unchanged":
            unchanged,
    }


def get_fdr_summary(
    frame: pd.DataFrame,
    candidate_id: str,
    stratum: str,
):

    subset = frame[
        (
            frame[
                "candidate_id"
            ].astype(str)
            ==
            candidate_id
        )
        &
        (
            frame[
                "stratum"
            ].astype(str)
            ==
            stratum
        )
    ].copy()


    if len(
        subset
    ) != 2:

        raise RuntimeError(
            f"{candidate_id}/{stratum}: "
            "expected exactly two FDR families."
        )


    if set(
        subset[
            "quantity"
        ].astype(str)
    ) != set(
        FDR_QUANTITIES
    ):

        raise RuntimeError(
            f"{candidate_id}/{stratum}: "
            "FDR quantity set mismatch."
        )


    mean_abs_row = subset[
        subset[
            "quantity"
        ].astype(str)
        ==
        "ABS_DELTA_MEAN_ABS_SHAP"
    ].iloc[0]


    mean_signed_row = subset[
        subset[
            "quantity"
        ].astype(str)
        ==
        "ABS_DELTA_MEAN_SIGNED_SHAP"
    ].iloc[0]


    mean_abs_count = int(
        mean_abs_row[
            "fdr_significant_features"
        ]
    )


    mean_signed_count = int(
        mean_signed_row[
            "fdr_significant_features"
        ]
    )


    mean_abs_features = parse_feature_names(
        mean_abs_row[
            "significant_feature_names"
        ]
    )


    mean_signed_features = parse_feature_names(
        mean_signed_row[
            "significant_feature_names"
        ]
    )


    if len(
        mean_abs_features
    ) != mean_abs_count:

        raise RuntimeError(
            f"{candidate_id}/{stratum}: "
            "mean-|SHAP| significant feature "
            "name/count mismatch."
        )


    if len(
        mean_signed_features
    ) != mean_signed_count:

        raise RuntimeError(
            f"{candidate_id}/{stratum}: "
            "signed-SHAP significant feature "
            "name/count mismatch."
        )


    union_features = sorted(
        set(
            mean_abs_features
        ).union(
            mean_signed_features
        )
    )


    return {
        "mean_abs_count":
            mean_abs_count,

        "mean_signed_count":
            mean_signed_count,

        "total_hypotheses":
            (
                mean_abs_count
                +
                mean_signed_count
            ),

        "unique_feature_count":
            len(
                union_features
            ),

        "unique_features":
            union_features,

        "mean_abs_features":
            mean_abs_features,

        "mean_signed_features":
            mean_signed_features,
    }


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.10.2 — "
    "FINAL CROSS-CYCLE SHAP EVIDENCE SYNTHESIS"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    SYNTHESIS_PROTOCOL_FILE,
    NULL_CONTRACT_FILE,
    SUPPORT_CONTRACT_FILE,
    GROUP_CONTRACT_FILE,
    FDR_CONTRACT_FILE,
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

for path in [
    MODEL_SYNTHESIS_FILE,
    CROSS_MODEL_SUMMARY_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter 10.10.2 outputs."
        )


# ============================================================
# VERIFY SYNTHESIS PROTOCOL
# ============================================================

protocol = load_json(
    SYNTHESIS_PROTOCOL_FILE
)


if (
    protocol.get(
        "status"
    )
    !=
    "FROZEN_FINAL_CROSS_CYCLE_SHAP_SYNTHESIS_PROTOCOL"
):

    raise RuntimeError(
        "Final SHAP synthesis protocol "
        "is not frozen."
    )


protocol_hash = sha256_file(
    SYNTHESIS_PROTOCOL_FILE
)


if (
    protocol[
        "scientific_guardrails"
    ][
        "final_synthesis_computed_here"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected synthesis state "
        "inside frozen protocol."
    )


if (
    protocol[
        "primary_model_evidence_rule"
    ][
        "binary_drift_label_created"
    ]
    is not False
):

    raise RuntimeError(
        "Frozen protocol unexpectedly "
        "creates a binary drift label."
    )


if (
    set(
        protocol[
            "primary_model_evidence_rule"
        ][
            "evidence_tiers"
        ].keys()
    )
    !=
    {
        "TIER_0",
        "TIER_1",
        "TIER_2",
    }
):

    raise RuntimeError(
        "Frozen evidence-tier set changed."
    )


print(
    "Frozen 10.10.1B synthesis protocol:",
    "PASS"
)


# ============================================================
# VERIFY EVIDENCE CONTRACTS
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


required_statuses = [
    (
        null_contract.get(
            "status"
        ),
        "FROZEN_CROSS_CYCLE_SHAP_STRUCTURE_NULL_COMPARISON",
    ),
    (
        support_contract.get(
            "status"
        ),
        "FROZEN_COMMON_SUPPORT_REWEIGHTING_SHAP_SENSITIVITY",
    ),
    (
        group_contract.get(
            "status"
        ),
        "FROZEN_CORRELATED_GROUP_SHAP_ROBUSTNESS",
    ),
    (
        fdr_contract.get(
            "status"
        ),
        "FROZEN_FEATUREWISE_CROSS_CYCLE_INFERENCE_BH_FDR",
    ),
]


for actual, expected in required_statuses:

    if actual != expected:

        raise RuntimeError(
            "Required frozen evidence contract "
            f"missing: {expected}"
        )


print(
    "Frozen evidence contracts:",
    "PASS"
)


# ============================================================
# VERIFY SOURCE HASHES AGAINST 10.10.1B PROTOCOL
# ============================================================

source_specs = protocol[
    "frozen_evidence_sources"
]


hash_checks = [
    (
        STRUCTURE_SUMMARY_FILE,
        source_specs[
            "structure_null"
        ][
            "summary_sha256"
        ],
    ),
    (
        SUPPORT_STRUCTURE_FILE,
        source_specs[
            "support_reweighting"
        ][
            "structure_sha256"
        ],
    ),
    (
        GROUP_STRUCTURE_FILE,
        source_specs[
            "correlated_groups"
        ][
            "structure_sha256"
        ],
    ),
    (
        FDR_SUMMARY_FILE,
        source_specs[
            "featurewise_fdr"
        ][
            "summary_sha256"
        ],
    ),
]


for path, expected_hash in hash_checks:

    actual_hash = sha256_file(
        path
    )


    if actual_hash != expected_hash:

        raise RuntimeError(
            f"Frozen synthesis input changed: {path}"
        )


print(
    "Frozen synthesis input hashes:",
    "PASS"
)


# ============================================================
# LOAD EVIDENCE TABLES
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
        "Structure summary row-count mismatch."
    )


if len(
    support_structure
) != EXPECTED_SUPPORT_ROWS:

    raise RuntimeError(
        "Support sensitivity row-count mismatch."
    )


if len(
    group_structure
) != EXPECTED_GROUP_ROWS:

    raise RuntimeError(
        "Group robustness row-count mismatch."
    )


if len(
    fdr_summary
) != EXPECTED_FDR_ROWS:

    raise RuntimeError(
        "FDR summary row-count mismatch."
    )


print(
    "Frozen evidence inventories:",
    "PASS"
)


# ============================================================
# SYNTHESIS CONTAINERS
# ============================================================

model_rows = []


# ============================================================
# MODEL-LEVEL SYNTHESIS
# ============================================================

for candidate_id in MODEL_ORDER:

    print()

    print("=" * 78)

    print(
        "SYNTHESIZE MODEL:",
        candidate_id
    )

    print("=" * 78)


    # --------------------------------------------------------
    # PRIMARY OVERALL STRUCTURE NULL
    # --------------------------------------------------------

    overall_structure = structure_summary[
        (
            structure_summary[
                "candidate_id"
            ].astype(str)
            ==
            candidate_id
        )
        &
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
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: expected exactly "
            "one OVERALL structure-null summary row."
        )


    overall_structure_row = overall_structure.iloc[
        0
    ]


    overall_null_exceedance = parse_bool(
        overall_structure_row[
            "any_null_exceedance"
        ]
    )


    overall_structure_exceedance_count = int(
        overall_structure_row[
            "metrics_exceeding_null"
        ]
    )


    overall_exceeded_metrics = str(
        overall_structure_row[
            "exceeded_metrics"
        ]
    ).strip()


    if overall_exceeded_metrics.lower() == "nan":

        overall_exceeded_metrics = ""


    # --------------------------------------------------------
    # PRIMARY OVERALL FEATUREWISE FDR
    # --------------------------------------------------------

    overall_fdr = get_fdr_summary(
        fdr_summary,
        candidate_id,
        "OVERALL",
    )


    # --------------------------------------------------------
    # ASSIGN FROZEN PRIMARY EVIDENCE TIER
    # --------------------------------------------------------

    if not overall_null_exceedance:

        primary_tier = "TIER_0"


    elif (
        overall_fdr[
            "total_hypotheses"
        ]
        ==
        0
    ):

        primary_tier = "TIER_1"


    else:

        primary_tier = "TIER_2"


    tier_label = (
        protocol[
            "primary_model_evidence_rule"
        ][
            "evidence_tiers"
        ][
            primary_tier
        ][
            "label"
        ]
    )


    # --------------------------------------------------------
    # CLASS-CONDITIONAL STRUCTURE + FDR
    # --------------------------------------------------------

    class_results = {}


    for stratum in [
        "TARGET_0",
        "TARGET_1",
    ]:

        class_structure = structure_summary[
            (
                structure_summary[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            )
            &
            (
                structure_summary[
                    "analysis_scope"
                ].astype(str)
                ==
                "CLASS_CONDITIONAL"
            )
            &
            (
                structure_summary[
                    "observed_stratum"
                ].astype(str)
                ==
                stratum
            )
        ].copy()


        if len(
            class_structure
        ) != 1:

            raise RuntimeError(
                f"{candidate_id}/{stratum}: "
                "expected exactly one class "
                "structure summary row."
            )


        class_row = class_structure.iloc[
            0
        ]


        class_fdr = get_fdr_summary(
            fdr_summary,
            candidate_id,
            stratum,
        )


        class_results[
            stratum
        ] = {
            "structure_null_exceedance":
                parse_bool(
                    class_row[
                        "any_null_exceedance"
                    ]
                ),

            "structure_exceedance_count":
                int(
                    class_row[
                        "metrics_exceeding_null"
                    ]
                ),

            "structure_exceeded_metrics":
                str(
                    class_row[
                        "exceeded_metrics"
                    ]
                ).strip(),

            "fdr":
                class_fdr,
        }


    # --------------------------------------------------------
    # TEMPORAL SCREENING
    # --------------------------------------------------------

    temporal_rows = structure_summary[
        (
            structure_summary[
                "candidate_id"
            ].astype(str)
            ==
            candidate_id
        )
        &
        (
            structure_summary[
                "analysis_scope"
            ].astype(str)
            ==
            "TEMPORAL_BLOCK"
        )
    ].copy()


    if len(
        temporal_rows
    ) != 3:

        raise RuntimeError(
            f"{candidate_id}: expected exactly "
            "3 temporal screening rows."
        )


    expected_blocks = {
        "B1",
        "B2",
        "B3",
    }


    if set(
        temporal_rows[
            "cycle25_block"
        ].astype(str)
    ) != expected_blocks:

        raise RuntimeError(
            f"{candidate_id}: temporal block "
            "inventory mismatch."
        )


    temporal_exceedance_blocks = []


    temporal_total_exceeded_metrics = 0


    for _, temporal_row in temporal_rows.iterrows():

        if parse_bool(
            temporal_row[
                "any_null_exceedance"
            ]
        ):

            temporal_exceedance_blocks.append(
                str(
                    temporal_row[
                        "cycle25_block"
                    ]
                )
            )


        temporal_total_exceeded_metrics += int(
            temporal_row[
                "metrics_exceeding_null"
            ]
        )


        if parse_bool(
            temporal_row[
                "temporal_null_is_sample_size_matched"
            ]
        ):

            raise RuntimeError(
                f"{candidate_id}: temporal null "
                "unexpectedly marked sample-size matched."
            )


    temporal_exceedance_blocks = sorted(
        temporal_exceedance_blocks
    )


    # --------------------------------------------------------
    # COMMON-SUPPORT / REWEIGHTING — OVERALL
    # --------------------------------------------------------

    support_overall = support_structure[
        (
            support_structure[
                "candidate_id"
            ].astype(str)
            ==
            candidate_id
        )
        &
        (
            support_structure[
                "analysis_scope"
            ].astype(str)
            ==
            "OVERALL_POOLED"
        )
        &
        (
            support_structure[
                "scope_key"
            ].astype(str)
            ==
            "OVERALL"
        )
    ].copy()


    if len(
        support_overall
    ) != 16:

        raise RuntimeError(
            f"{candidate_id}: expected "
            "16 OVERALL support sensitivity rows."
        )


    support_direction_results = {}


    for variant in [
        "COMMON_SUPPORT_UNWEIGHTED",
        "COMMON_SUPPORT_REWEIGHTED",
    ]:

        variant_rows = support_overall[
            support_overall[
                "sensitivity_variant"
            ].astype(str)
            ==
            variant
        ].copy()


        if len(
            variant_rows
        ) != 8:

            raise RuntimeError(
                f"{candidate_id}/{variant}: "
                "expected 8 structure metrics."
            )


        if set(
            variant_rows[
                "metric"
            ].astype(str)
        ) != set(
            STRUCTURE_METRICS
        ):

            raise RuntimeError(
                f"{candidate_id}/{variant}: "
                "metric set mismatch."
            )


        support_direction_results[
            variant
        ] = summarize_directional_rows(
            variant_rows,
            "delta_sensitivity_minus_primary",
        )


    support_annotation = (
        "UNWEIGHTED:"
        f"{support_direction_results['COMMON_SUPPORT_UNWEIGHTED']['toward_similarity']}"
        "_TOWARD_SIMILARITY/"
        f"{support_direction_results['COMMON_SUPPORT_UNWEIGHTED']['toward_difference']}"
        "_TOWARD_DIFFERENCE/"
        f"{support_direction_results['COMMON_SUPPORT_UNWEIGHTED']['unchanged']}"
        "_UNCHANGED;"
        "REWEIGHTED:"
        f"{support_direction_results['COMMON_SUPPORT_REWEIGHTED']['toward_similarity']}"
        "_TOWARD_SIMILARITY/"
        f"{support_direction_results['COMMON_SUPPORT_REWEIGHTED']['toward_difference']}"
        "_TOWARD_DIFFERENCE/"
        f"{support_direction_results['COMMON_SUPPORT_REWEIGHTED']['unchanged']}"
        "_UNCHANGED"
    )


    # --------------------------------------------------------
    # CORRELATED-GROUP ROBUSTNESS — OVERALL
    # --------------------------------------------------------

    group_overall = group_structure[
        (
            group_structure[
                "candidate_id"
            ].astype(str)
            ==
            candidate_id
        )
        &
        (
            group_structure[
                "analysis_scope"
            ].astype(str)
            ==
            "OVERALL_POOLED"
        )
        &
        (
            group_structure[
                "scope_key"
            ].astype(str)
            ==
            "OVERALL"
        )
    ].copy()


    expected_group_variants = [
        "PRIMARY_FULL_POPULATION",
        "COMMON_SUPPORT_UNWEIGHTED",
        "COMMON_SUPPORT_REWEIGHTED",
    ]


    if len(
        group_overall
    ) != 15:

        raise RuntimeError(
            f"{candidate_id}: expected "
            "15 OVERALL correlated-group rows."
        )


    group_direction_results = {}


    for variant in expected_group_variants:

        variant_rows = group_overall[
            group_overall[
                "analysis_variant"
            ].astype(str)
            ==
            variant
        ].copy()


        if len(
            variant_rows
        ) != 5:

            raise RuntimeError(
                f"{candidate_id}/{variant}: "
                "expected 5 group-level metrics."
            )


        if set(
            variant_rows[
                "metric"
            ].astype(str)
        ) != set(
            GROUP_METRICS
        ):

            raise RuntimeError(
                f"{candidate_id}/{variant}: "
                "group metric set mismatch."
            )


        if (
            variant_rows[
                "group_level_null_threshold_available"
            ]
            .map(
                parse_bool
            )
            .any()
        ):

            raise RuntimeError(
                f"{candidate_id}/{variant}: "
                "unexpected group-level null."
            )


        group_direction_results[
            variant
        ] = summarize_directional_rows(
            variant_rows,
            "group_minus_individual_metric_value",
        )


    group_annotation = (
        "PRIMARY:"
        f"{group_direction_results['PRIMARY_FULL_POPULATION']['toward_similarity']}"
        "_TOWARD_SIMILARITY/"
        f"{group_direction_results['PRIMARY_FULL_POPULATION']['toward_difference']}"
        "_TOWARD_DIFFERENCE/"
        f"{group_direction_results['PRIMARY_FULL_POPULATION']['unchanged']}"
        "_UNCHANGED;"
        "SUPPORT:"
        f"{group_direction_results['COMMON_SUPPORT_UNWEIGHTED']['toward_similarity']}"
        "_TOWARD_SIMILARITY/"
        f"{group_direction_results['COMMON_SUPPORT_UNWEIGHTED']['toward_difference']}"
        "_TOWARD_DIFFERENCE/"
        f"{group_direction_results['COMMON_SUPPORT_UNWEIGHTED']['unchanged']}"
        "_UNCHANGED;"
        "REWEIGHTED:"
        f"{group_direction_results['COMMON_SUPPORT_REWEIGHTED']['toward_similarity']}"
        "_TOWARD_SIMILARITY/"
        f"{group_direction_results['COMMON_SUPPORT_REWEIGHTED']['toward_difference']}"
        "_TOWARD_DIFFERENCE/"
        f"{group_direction_results['COMMON_SUPPORT_REWEIGHTED']['unchanged']}"
        "_UNCHANGED"
    )


    # --------------------------------------------------------
    # MODEL ROW
    # --------------------------------------------------------

    model_family_values = structure_summary[
        structure_summary[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ][
        "model_family"
    ].astype(str).unique()


    if len(
        model_family_values
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: model-family mismatch."
        )


    target0 = class_results[
        "TARGET_0"
    ]


    target1 = class_results[
        "TARGET_1"
    ]


    model_rows.append(
        {
            "candidate_id":
                candidate_id,

            "model_family":
                model_family_values[
                    0
                ],

            # --------------------------------------------
            # Primary OVERALL evidence
            # --------------------------------------------

            "overall_structure_null_exceedance":
                overall_null_exceedance,

            "overall_structure_exceedance_count":
                overall_structure_exceedance_count,

            "overall_structure_exceeded_metrics":
                overall_exceeded_metrics,

            "overall_fdr_mean_abs_count":
                overall_fdr[
                    "mean_abs_count"
                ],

            "overall_fdr_mean_signed_count":
                overall_fdr[
                    "mean_signed_count"
                ],

            "overall_fdr_total_significant_hypotheses":
                overall_fdr[
                    "total_hypotheses"
                ],

            "overall_fdr_unique_feature_count":
                overall_fdr[
                    "unique_feature_count"
                ],

            "overall_fdr_unique_features":
                ";".join(
                    overall_fdr[
                        "unique_features"
                    ]
                ),

            "primary_evidence_tier":
                primary_tier,

            "primary_evidence_label":
                tier_label,

            # --------------------------------------------
            # TARGET_0
            # --------------------------------------------

            "target0_structure_null_exceedance":
                target0[
                    "structure_null_exceedance"
                ],

            "target0_structure_exceedance_count":
                target0[
                    "structure_exceedance_count"
                ],

            "target0_fdr_mean_abs_count":
                target0[
                    "fdr"
                ][
                    "mean_abs_count"
                ],

            "target0_fdr_mean_signed_count":
                target0[
                    "fdr"
                ][
                    "mean_signed_count"
                ],

            "target0_fdr_total_significant_hypotheses":
                target0[
                    "fdr"
                ][
                    "total_hypotheses"
                ],

            "target0_fdr_unique_feature_count":
                target0[
                    "fdr"
                ][
                    "unique_feature_count"
                ],

            "target0_fdr_unique_features":
                ";".join(
                    target0[
                        "fdr"
                    ][
                        "unique_features"
                    ]
                ),

            # --------------------------------------------
            # TARGET_1
            # --------------------------------------------

            "target1_structure_null_exceedance":
                target1[
                    "structure_null_exceedance"
                ],

            "target1_structure_exceedance_count":
                target1[
                    "structure_exceedance_count"
                ],

            "target1_fdr_mean_abs_count":
                target1[
                    "fdr"
                ][
                    "mean_abs_count"
                ],

            "target1_fdr_mean_signed_count":
                target1[
                    "fdr"
                ][
                    "mean_signed_count"
                ],

            "target1_fdr_total_significant_hypotheses":
                target1[
                    "fdr"
                ][
                    "total_hypotheses"
                ],

            "target1_fdr_unique_feature_count":
                target1[
                    "fdr"
                ][
                    "unique_feature_count"
                ],

            "target1_fdr_unique_features":
                ";".join(
                    target1[
                        "fdr"
                    ][
                        "unique_features"
                    ]
                ),

            # --------------------------------------------
            # Support / reweighting
            # --------------------------------------------

            "support_unweighted_metrics_toward_similarity":
                support_direction_results[
                    "COMMON_SUPPORT_UNWEIGHTED"
                ][
                    "toward_similarity"
                ],

            "support_unweighted_metrics_toward_difference":
                support_direction_results[
                    "COMMON_SUPPORT_UNWEIGHTED"
                ][
                    "toward_difference"
                ],

            "support_unweighted_metrics_unchanged":
                support_direction_results[
                    "COMMON_SUPPORT_UNWEIGHTED"
                ][
                    "unchanged"
                ],

            "support_reweighted_metrics_toward_similarity":
                support_direction_results[
                    "COMMON_SUPPORT_REWEIGHTED"
                ][
                    "toward_similarity"
                ],

            "support_reweighted_metrics_toward_difference":
                support_direction_results[
                    "COMMON_SUPPORT_REWEIGHTED"
                ][
                    "toward_difference"
                ],

            "support_reweighted_metrics_unchanged":
                support_direction_results[
                    "COMMON_SUPPORT_REWEIGHTED"
                ][
                    "unchanged"
                ],

            "support_reweighting_annotation":
                support_annotation,

            # --------------------------------------------
            # Correlated groups
            # --------------------------------------------

            "group_primary_metrics_toward_similarity":
                group_direction_results[
                    "PRIMARY_FULL_POPULATION"
                ][
                    "toward_similarity"
                ],

            "group_primary_metrics_toward_difference":
                group_direction_results[
                    "PRIMARY_FULL_POPULATION"
                ][
                    "toward_difference"
                ],

            "group_primary_metrics_unchanged":
                group_direction_results[
                    "PRIMARY_FULL_POPULATION"
                ][
                    "unchanged"
                ],

            "group_support_metrics_toward_similarity":
                group_direction_results[
                    "COMMON_SUPPORT_UNWEIGHTED"
                ][
                    "toward_similarity"
                ],

            "group_support_metrics_toward_difference":
                group_direction_results[
                    "COMMON_SUPPORT_UNWEIGHTED"
                ][
                    "toward_difference"
                ],

            "group_support_metrics_unchanged":
                group_direction_results[
                    "COMMON_SUPPORT_UNWEIGHTED"
                ][
                    "unchanged"
                ],

            "group_reweighted_metrics_toward_similarity":
                group_direction_results[
                    "COMMON_SUPPORT_REWEIGHTED"
                ][
                    "toward_similarity"
                ],

            "group_reweighted_metrics_toward_difference":
                group_direction_results[
                    "COMMON_SUPPORT_REWEIGHTED"
                ][
                    "toward_difference"
                ],

            "group_reweighted_metrics_unchanged":
                group_direction_results[
                    "COMMON_SUPPORT_REWEIGHTED"
                ][
                    "unchanged"
                ],

            "correlated_group_annotation":
                group_annotation,

            # --------------------------------------------
            # Temporal screen
            # --------------------------------------------

            "temporal_blocks_evaluated":
                3,

            "temporal_blocks_with_null_exceedance":
                len(
                    temporal_exceedance_blocks
                ),

            "temporal_exceedance_blocks":
                ";".join(
                    temporal_exceedance_blocks
                ),

            "temporal_total_exceeded_metrics":
                temporal_total_exceeded_metrics,

            "temporal_null_sample_size_matched":
                False,

            "temporal_screening_annotation":
                (
                    "SCREENING_ONLY_"
                    "NO_BLOCK_SIZE_MATCHED_NULL"
                ),

            # --------------------------------------------
            # Cautions
            # --------------------------------------------

            "target1_cycle24_rows":
                527,

            "target1_cycle24_entity_groups":
                25,

            "sample_size_caution":
                (
                    "TARGET_1_C24_ONLY_25_ENTITY_GROUPS"
                ),

            # --------------------------------------------
            # Final restrictions
            # --------------------------------------------

            "support_results_formally_thresholded":
                False,

            "group_results_formally_thresholded":
                False,

            "final_binary_explanation_drift_label":
                "NOT_ASSIGNED",

            "final_binary_explanation_stability_label":
                "NOT_ASSIGNED",

            "shap_causal_interpretation":
                False,
        }
    )


    # --------------------------------------------------------
    # PRINT MODEL SYNTHESIS
    # --------------------------------------------------------

    print(
        "OVERALL structure exceedance:",
        overall_null_exceedance,
        f"({overall_structure_exceedance_count}/8 metrics)"
    )


    print(
        "OVERALL FDR significant hypotheses:",
        overall_fdr[
            "total_hypotheses"
        ],
        "| unique features:",
        overall_fdr[
            "unique_feature_count"
        ],
    )


    print(
        "Primary evidence tier:",
        primary_tier,
        "|",
        tier_label,
    )


    print(
        "TARGET_0:",
        "structure=",
        target0[
            "structure_null_exceedance"
        ],
        "| FDR hypotheses=",
        target0[
            "fdr"
        ][
            "total_hypotheses"
        ],
    )


    print(
        "TARGET_1:",
        "structure=",
        target1[
            "structure_null_exceedance"
        ],
        "| FDR hypotheses=",
        target1[
            "fdr"
        ][
            "total_hypotheses"
        ],
    )


    print(
        "Temporal exceedance blocks:",
        (
            ";".join(
                temporal_exceedance_blocks
            )
            if
            temporal_exceedance_blocks
            else
            "NONE"
        ),
    )


# ============================================================
# BUILD MODEL SYNTHESIS
# ============================================================

model_synthesis = pd.DataFrame(
    model_rows
)


if len(
    model_synthesis
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 model synthesis rows."
    )


if (
    model_synthesis[
        "candidate_id"
    ].astype(str).tolist()
    !=
    MODEL_ORDER
):

    raise RuntimeError(
        "Model synthesis order changed."
    )


# ============================================================
# CROSS-MODEL PATTERN
# ============================================================

tier_counts = (
    model_synthesis[
        "primary_evidence_tier"
    ]
    .value_counts()
    .to_dict()
)


tier0_count = int(
    tier_counts.get(
        "TIER_0",
        0,
    )
)


tier1_count = int(
    tier_counts.get(
        "TIER_1",
        0,
    )
)


tier2_count = int(
    tier_counts.get(
        "TIER_2",
        0,
    )
)


max_tier_count = max(
    tier0_count,
    tier1_count,
    tier2_count,
)


if max_tier_count == EXPECTED_MODELS:

    cross_model_pattern = (
        "CONSISTENT_ACROSS_ALL_MODELS"
    )


elif max_tier_count == 2:

    cross_model_pattern = (
        "MAJORITY_MODEL_PATTERN"
    )


else:

    cross_model_pattern = (
        "MODEL_DEPENDENT_MIXED_PATTERN"
    )


majority_tiers = [
    tier
    for tier, count
    in [
        (
            "TIER_0",
            tier0_count,
        ),
        (
            "TIER_1",
            tier1_count,
        ),
        (
            "TIER_2",
            tier2_count,
        ),
    ]
    if count
    ==
    max_tier_count
]


if len(
    majority_tiers
) == 1:

    dominant_tier = majority_tiers[
        0
    ]

else:

    dominant_tier = (
        "NO_SINGLE_DOMINANT_TIER"
    )


models_by_tier = {}


for tier in [
    "TIER_0",
    "TIER_1",
    "TIER_2",
]:

    models_by_tier[
        tier
    ] = model_synthesis[
        model_synthesis[
            "primary_evidence_tier"
        ].astype(str)
        ==
        tier
    ][
        "candidate_id"
    ].astype(str).tolist()


cross_model_summary = pd.DataFrame(
    [
        {
            "models_evaluated":
                EXPECTED_MODELS,

            "tier0_models":
                tier0_count,

            "tier1_models":
                tier1_count,

            "tier2_models":
                tier2_count,

            "tier0_model_names":
                ";".join(
                    models_by_tier[
                        "TIER_0"
                    ]
                ),

            "tier1_model_names":
                ";".join(
                    models_by_tier[
                        "TIER_1"
                    ]
                ),

            "tier2_model_names":
                ";".join(
                    models_by_tier[
                        "TIER_2"
                    ]
                ),

            "cross_model_primary_pattern":
                cross_model_pattern,

            "dominant_primary_evidence_tier":
                dominant_tier,

            "all_models_same_tier":
                (
                    max_tier_count
                    ==
                    EXPECTED_MODELS
                ),

            "majority_model_pattern":
                (
                    max_tier_count
                    ==
                    2
                ),

            "raw_shap_magnitudes_pooled_across_models":
                False,

            "winner_model_selected":
                False,

            "automatic_binary_explanation_drift_label":
                "NOT_ASSIGNED",

            "automatic_binary_explanation_stability_label":
                "NOT_ASSIGNED",

            "interpretation":
                (
                    "Cross-model synthesis preserves "
                    "model-specific evidence tiers. "
                    "No pooled raw-SHAP or automatic "
                    "binary drift/stability label is used."
                ),
        }
    ]
)


# ============================================================
# PRINT CROSS-MODEL RESULT
# ============================================================

print()

print("=" * 78)

print(
    "CROSS-MODEL PRIMARY EVIDENCE PATTERN"
)

print("=" * 78)


print(
    "TIER_0 models:",
    tier0_count,
    "|",
    (
        ";".join(
            models_by_tier[
                "TIER_0"
            ]
        )
        or
        "NONE"
    )
)


print(
    "TIER_1 models:",
    tier1_count,
    "|",
    (
        ";".join(
            models_by_tier[
                "TIER_1"
            ]
        )
        or
        "NONE"
    )
)


print(
    "TIER_2 models:",
    tier2_count,
    "|",
    (
        ";".join(
            models_by_tier[
                "TIER_2"
            ]
        )
        or
        "NONE"
    )
)


print(
    "Cross-model pattern:",
    cross_model_pattern
)


print(
    "Dominant tier:",
    dominant_tier
)


print(
    "Automatic binary drift label:",
    False
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

MODEL_SYNTHESIS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


CROSS_MODEL_SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ATOMIC CSV WRITES
# ============================================================

model_temp = MODEL_SYNTHESIS_FILE.with_name(
    MODEL_SYNTHESIS_FILE.stem
    +
    ".tmp.csv"
)


cross_temp = CROSS_MODEL_SUMMARY_FILE.with_name(
    CROSS_MODEL_SUMMARY_FILE.stem
    +
    ".tmp.csv"
)


for path in [
    model_temp,
    cross_temp,
]:

    if path.exists():

        path.unlink()


model_synthesis.to_csv(
    model_temp,
    index=False,
)


cross_model_summary.to_csv(
    cross_temp,
    index=False,
)


# ============================================================
# ROUND-TRIP
# ============================================================

model_rt = pd.read_csv(
    model_temp,
    keep_default_na=False,
)


cross_rt = pd.read_csv(
    cross_temp,
    keep_default_na=False,
)


if len(
    model_rt
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Model synthesis CSV round-trip failed."
    )


if len(
    cross_rt
) != 1:

    raise RuntimeError(
        "Cross-model summary CSV round-trip failed."
    )


model_temp.replace(
    MODEL_SYNTHESIS_FILE
)


cross_temp.replace(
    CROSS_MODEL_SUMMARY_FILE
)


model_hash = sha256_file(
    MODEL_SYNTHESIS_FILE
)


cross_hash = sha256_file(
    CROSS_MODEL_SUMMARY_FILE
)


print()

print(
    "Model synthesis CSV round-trip:",
    "PASS"
)


print(
    "Cross-model summary CSV round-trip:",
    "PASS"
)


# ============================================================
# FINAL CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Final Cross-Cycle SHAP "
            "Evidence Synthesis"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CROSS_CYCLE_SHAP_SYNTHESIS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_10_2",

    "frozen_synthesis_protocol":
        {
            "file":
                SYNTHESIS_PROTOCOL_FILE.as_posix(),

            "sha256":
                protocol_hash,

            "required_status":
                (
                    "FROZEN_FINAL_CROSS_CYCLE_"
                    "SHAP_SYNTHESIS_PROTOCOL"
                ),
        },

    "evidence_sources":
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

    "model_synthesis":
        {
            "file":
                MODEL_SYNTHESIS_FILE.as_posix(),

            "sha256":
                model_hash,

            "rows":
                EXPECTED_MODELS,
        },

    "cross_model_summary":
        {
            "file":
                CROSS_MODEL_SUMMARY_FILE.as_posix(),

            "sha256":
                cross_hash,

            "rows":
                1,

            "cross_model_primary_pattern":
                cross_model_pattern,

            "dominant_primary_evidence_tier":
                dominant_tier,

            "tier_counts":
                {
                    "TIER_0":
                        tier0_count,

                    "TIER_1":
                        tier1_count,

                    "TIER_2":
                        tier2_count,
                },
        },

    "model_primary_evidence_tiers":
        {
            row[
                "candidate_id"
            ]:
                {
                    "tier":
                        row[
                            "primary_evidence_tier"
                        ],

                    "label":
                        row[
                            "primary_evidence_label"
                        ],

                    "overall_structure_null_exceedance":
                        bool(
                            row[
                                "overall_structure_null_exceedance"
                            ]
                        ),

                    "overall_fdr_significant_hypotheses":
                        int(
                            row[
                                "overall_fdr_total_significant_hypotheses"
                            ]
                        ),
                }

            for _, row
            in model_synthesis.iterrows()
        },

    "scientific_guardrails":
        {
            "new_statistical_test_performed":
                False,

            "new_null_distribution_created":
                False,

            "new_significance_threshold_created":
                False,

            "support_restricted_null_invented":
                False,

            "correlated_group_null_invented":
                False,

            "temporal_block_specific_null_invented":
                False,

            "raw_shap_pooled_across_models":
                False,

            "winner_model_selected":
                False,

            "automatic_binary_explanation_drift_label_created":
                False,

            "automatic_binary_explanation_stability_label_created":
                False,

            "final_evidence_synthesis_computed":
                True,

            "shap_is_causal":
                False,
        },

    "interpretation_rule":
        (
            "The synthesis reports frozen model-specific "
            "evidence tiers rather than forcing all models "
            "into one binary explanation-drift label. "
            "TIER_2 requires both an OVERALL structure-null "
            "exceedance and at least one OVERALL "
            "BH-FDR-significant featurewise hypothesis. "
            "TIER_0 means no global structure-null "
            "exceedance and must not be interpreted as "
            "proof of explanation stability, even when "
            "featurewise changes exist. Class-conditional, "
            "support/reweighting, correlated-group, and "
            "temporal results remain robustness/context "
            "evidence under their frozen limitations."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_10_3_"
            "FINAL_CHAPTER10_FREEZE"
        ),
}


# ============================================================
# SAVE CONTRACT ATOMICALLY
# ============================================================

contract_temp = OUTPUT_CONTRACT_FILE.with_name(
    OUTPUT_CONTRACT_FILE.stem
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
    "FROZEN_CROSS_CYCLE_SHAP_SYNTHESIS"
):

    raise RuntimeError(
        "10.10.2 synthesis contract "
        "round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "final_evidence_synthesis_computed"
    ]
    is not True
):

    raise RuntimeError(
        "Final synthesis state not frozen."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "automatic_binary_explanation_drift_label_created"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected binary drift label."
    )


contract_temp.replace(
    OUTPUT_CONTRACT_FILE
)


contract_hash = sha256_file(
    OUTPUT_CONTRACT_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 10.10.2 — COMPLETE"
)

print("=" * 78)


print(
    "Models synthesized:",
    len(
        model_synthesis
    )
)


print()

for _, row in model_synthesis.iterrows():

    print(
        row[
            "candidate_id"
        ],
        "→",
        row[
            "primary_evidence_tier"
        ],
        "|",
        row[
            "primary_evidence_label"
        ],
    )


print()

print(
    "Cross-model pattern:",
    cross_model_pattern
)

print(
    "Dominant primary evidence tier:",
    dominant_tier
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
    "New statistical threshold:",
    False
)

print(
    "Support/group formal null invented:",
    False
)

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
    "FROZEN_CROSS_CYCLE_SHAP_SYNTHESIS"
)


print()

print(
    "Model synthesis:",
    MODEL_SYNTHESIS_FILE
)

print(
    "Cross-model summary:",
    CROSS_MODEL_SUMMARY_FILE
)

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
        "Chapter 10.10.3 — "
        "Final Chapter-10 Freeze"
    )
)