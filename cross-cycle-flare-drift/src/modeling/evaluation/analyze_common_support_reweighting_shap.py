# ============================================================
# MODEL CHAPTER 10.7.2
# COMMON-SUPPORT / REWEIGHTING SHAP SENSITIVITY
#
# Sensitivity variants:
#
#   1. COMMON_SUPPORT_UNWEIGHTED
#      C24 supported rows vs C25 supported rows
#
#   2. COMMON_SUPPORT_REWEIGHTED
#      C24 supported rows vs frozen C25->C24 weighted
#      supported Cycle25 rows
#
# Applied to:
#   - Overall pooled
#   - B1 / B2 / B3
#   - TARGET_0 / TARGET_1
#
# IMPORTANT:
#   - Frozen support flags are reused.
#   - Frozen analysis_weight is reused directly.
#   - NO propensity refit.
#   - NO weight recomputation.
#   - NO re-clipping.
#   - NO SHAP recomputation.
#   - NO Chapter-9 null thresholds applied here because
#     the reference estimand is common-support restricted.
#   - NO final explanation-drift/stability conclusion.
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau


# ============================================================
# CONSTANTS
# ============================================================

EPSILON = 1e-12

EXPECTED_MODELS = 3
EXPECTED_FEATURES = 25

EXPECTED_SUPPORT_ROWS = 92433

EXPECTED_C24_FULL_ROWS = 13072
EXPECTED_C25_FULL_ROWS = 79361

EXPECTED_C24_SUPPORTED = 12153
EXPECTED_C25_SUPPORTED = 64732

EXPECTED_B1_SUPPORTED = 9937
EXPECTED_B2_SUPPORTED = 29084
EXPECTED_B3_SUPPORTED = 25711

EXPECTED_POOLED_WEIGHTED_ESS = 27156.686994464613

EXPECTED_STRUCTURE_ROWS = (
    3          # models
    * 6        # overall + 3 blocks + 2 classes
    * 2        # sensitivity variants
    * 8        # metrics
)

EXPECTED_FEATUREWISE_ROWS = (
    3
    * 6
    * 2
    * 25
)


ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TARGET_COLUMN = "target"
TIME_COLUMN = "observation_time_utc"


FEATURES = [
    "TOTUSJH",
    "TOTPOT",
    "TOTUSJZ",
    "ABSNJZH",
    "SAVNCPP",
    "USFLUX",
    "AREA_ACR",
    "MEANPOT",
    "R_VALUE",
    "SHRGT45",
    "MEANSHR",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "MEANJZH",
    "MEANJZD",
    "MEANALP",
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
]


SHAP_COLUMNS = [
    f"SHAP_{feature}"
    for feature in FEATURES
]


MODEL_ORDER = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


METRIC_ORDER = [
    "spearman_rho",
    "kendall_tau_b",
    "top5_overlap",
    "top10_overlap",
    "normalized_l1_mean_abs",
    "cosine_distance_mean_abs",
    "signed_cosine_similarity",
    "sign_agreement",
]


SENSITIVITY_VARIANTS = [
    "COMMON_SUPPORT_UNWEIGHTED",
    "COMMON_SUPPORT_REWEIGHTED",
]


BLOCK_SCOPE_MAP = {
    "B1": "C25_B1_2020_2021",
    "B2": "C25_B2_2022_2023",
    "B3": "C25_B3_2024_2025",
}


# ============================================================
# INPUT FILES
# ============================================================

SUPPORT_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "joint_shift_common_support_contract.json"
)


SUPPORT_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "domain_oof_common_support.parquet"
)


SUPPORT_SUMMARY_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "common_support_summary.csv"
)


OVERALL_INPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "overall_cross_cycle_shap_inputs_contract.json"
)


OVERALL_INPUT_REGISTRY_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "overall_cross_cycle_shap_input_registry.csv"
)


OVERALL_POINT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "overall_cross_cycle_shap_point_estimates_contract.json"
)


TEMPORAL_POINT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "temporal_block_cross_cycle_shap_point_estimates_contract.json"
)


CLASS_POINT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "class_conditional_cross_cycle_shap_point_estimates_contract.json"
)


NULL_COMPARISON_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "cross_cycle_shap_structure_null_comparison_contract.json"
)


# ============================================================
# OUTPUT FILES
# ============================================================

STRUCTURE_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "common_support_reweighting_shap_structure_sensitivity.csv"
)


FEATUREWISE_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "common_support_reweighting_shap_featurewise_sensitivity.csv"
)


OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "common_support_reweighting_shap_sensitivity_contract.json"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def verify_exists(path):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    value = str(value).strip().lower()

    if value in {"true", "1"}:
        return True

    if value in {"false", "0"}:
        return False

    raise RuntimeError(
        f"Cannot parse boolean value: {value}"
    )


def cosine_similarity(a, b):
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if (
        norm_a <= EPSILON
        and
        norm_b <= EPSILON
    ):
        return 1.0

    if (
        norm_a <= EPSILON
        or
        norm_b <= EPSILON
    ):
        return 0.0

    value = float(
        np.dot(a, b)
        /
        (
            norm_a
            *
            norm_b
        )
    )

    return float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )


def top_k_overlap(a, b, k):
    feature_index = np.arange(
        len(a),
        dtype=np.int64,
    )

    order_a = np.lexsort(
        (
            feature_index,
            -a,
        )
    )

    order_b = np.lexsort(
        (
            feature_index,
            -b,
        )
    )

    top_a = set(
        order_a[:k].tolist()
    )

    top_b = set(
        order_b[:k].tolist()
    )

    return (
        len(
            top_a.intersection(
                top_b
            )
        )
        /
        float(k)
    )


def normalized_l1_distance(a, b):
    numerator = float(
        np.sum(
            np.abs(
                a - b
            )
        )
    )

    denominator = float(
        np.sum(np.abs(a))
        +
        np.sum(np.abs(b))
    )

    if denominator <= EPSILON:
        return 0.0

    return numerator / denominator


def sign_agreement(a, b):
    return float(
        np.mean(
            np.sign(a)
            ==
            np.sign(b)
        )
    )


def calculate_structure_metrics(
    mean_abs_a,
    mean_abs_b,
    mean_signed_a,
    mean_signed_b,
):
    spearman_result = spearmanr(
        mean_abs_a,
        mean_abs_b,
    )

    kendall_result = kendalltau(
        mean_abs_a,
        mean_abs_b,
        variant="b",
    )

    spearman_value = float(
        spearman_result.statistic
    )

    kendall_value = float(
        kendall_result.statistic
    )

    if not np.isfinite(spearman_value):
        raise RuntimeError(
            "Non-finite Spearman rho."
        )

    if not np.isfinite(kendall_value):
        raise RuntimeError(
            "Non-finite Kendall tau-b."
        )

    return {
        "spearman_rho":
            spearman_value,

        "kendall_tau_b":
            kendall_value,

        "top5_overlap":
            top_k_overlap(
                mean_abs_a,
                mean_abs_b,
                5,
            ),

        "top10_overlap":
            top_k_overlap(
                mean_abs_a,
                mean_abs_b,
                10,
            ),

        "normalized_l1_mean_abs":
            normalized_l1_distance(
                mean_abs_a,
                mean_abs_b,
            ),

        "cosine_distance_mean_abs":
            (
                1.0
                -
                cosine_similarity(
                    mean_abs_a,
                    mean_abs_b,
                )
            ),

        "signed_cosine_similarity":
            cosine_similarity(
                mean_signed_a,
                mean_signed_b,
            ),

        "sign_agreement":
            sign_agreement(
                mean_signed_a,
                mean_signed_b,
            ),
    }


def deterministic_ranks(values):
    feature_index = np.arange(
        len(values),
        dtype=np.int64,
    )

    order = np.lexsort(
        (
            feature_index,
            -values,
        )
    )

    ranks = np.empty(
        len(values),
        dtype=np.int64,
    )

    ranks[order] = np.arange(
        1,
        len(values) + 1,
        dtype=np.int64,
    )

    return ranks


def effective_sample_size(weights):
    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    numerator = float(
        np.sum(weights) ** 2
    )

    denominator = float(
        np.sum(
            weights ** 2
        )
    )

    if denominator <= EPSILON:
        raise RuntimeError(
            "Cannot calculate ESS: zero denominator."
        )

    return numerator / denominator


def attribution_vectors(
    shap_matrix,
    weights=None,
):
    shap_matrix = np.asarray(
        shap_matrix,
        dtype=np.float64,
    )

    if shap_matrix.ndim != 2:
        raise RuntimeError(
            "SHAP matrix must be 2-dimensional."
        )

    if not np.isfinite(
        shap_matrix
    ).all():
        raise RuntimeError(
            "Non-finite SHAP values."
        )

    if weights is None:
        mean_abs = np.mean(
            np.abs(shap_matrix),
            axis=0,
        )

        mean_signed = np.mean(
            shap_matrix,
            axis=0,
        )

    else:
        weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        if len(weights) != len(shap_matrix):
            raise RuntimeError(
                "Weight / SHAP row-count mismatch."
            )

        if not np.isfinite(
            weights
        ).all():
            raise RuntimeError(
                "Non-finite analysis weights."
            )

        if not (
            weights > 0
        ).all():
            raise RuntimeError(
                "Analysis weights must be positive."
            )

        mean_abs = np.average(
            np.abs(shap_matrix),
            axis=0,
            weights=weights,
        )

        mean_signed = np.average(
            shap_matrix,
            axis=0,
            weights=weights,
        )

    return (
        np.asarray(
            mean_abs,
            dtype=np.float64,
        ),
        np.asarray(
            mean_signed,
            dtype=np.float64,
        ),
    )


def verify_metadata_alignment(
    joined,
    prefix,
):
    left_group = (
        joined[
            f"{ENTITY_GROUP_COLUMN}_support"
        ]
        .astype(str)
        .to_numpy()
    )

    right_group = (
        joined[
            f"{ENTITY_GROUP_COLUMN}_shap"
        ]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        left_group,
        right_group,
    ):
        raise RuntimeError(
            f"{prefix}: ENTITY_GROUP_ID alignment failed."
        )

    left_target = joined[
        f"{TARGET_COLUMN}_support"
    ].to_numpy(
        dtype=np.int8
    )

    right_target = joined[
        f"{TARGET_COLUMN}_shap"
    ].to_numpy(
        dtype=np.int8
    )

    if not np.array_equal(
        left_target,
        right_target,
    ):
        raise RuntimeError(
            f"{prefix}: target alignment failed."
        )

    left_time = pd.to_datetime(
        joined[
            f"{TIME_COLUMN}_support"
        ],
        utc=True,
        errors="raise",
    ).astype("int64").to_numpy()

    right_time = pd.to_datetime(
        joined[
            f"{TIME_COLUMN}_shap"
        ],
        utc=True,
        errors="raise",
    ).astype("int64").to_numpy()

    if not np.array_equal(
        left_time,
        right_time,
    ):
        raise RuntimeError(
            f"{prefix}: timestamp alignment failed."
        )


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.7.2 — "
    "COMMON-SUPPORT / REWEIGHTING SHAP SENSITIVITY"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

for path in [
    SUPPORT_CONTRACT_FILE,
    SUPPORT_FILE,
    SUPPORT_SUMMARY_FILE,
    OVERALL_INPUT_CONTRACT_FILE,
    OVERALL_INPUT_REGISTRY_FILE,
    OVERALL_POINT_CONTRACT_FILE,
    TEMPORAL_POINT_CONTRACT_FILE,
    CLASS_POINT_CONTRACT_FILE,
    NULL_COMPARISON_CONTRACT_FILE,
]:
    verify_exists(path)


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for output_path in [
    STRUCTURE_OUTPUT_FILE,
    FEATUREWISE_OUTPUT_FILE,
    OUTPUT_CONTRACT_FILE,
]:
    if output_path.exists():
        raise RuntimeError(
            f"Output already exists: {output_path}\n"
            "Refusing to overwrite frozen "
            "Chapter 10.7.2 outputs."
        )


# ============================================================
# VERIFY FROZEN SUPPORT CONTRACT
# ============================================================

support_contract = load_json(
    SUPPORT_CONTRACT_FILE
)


if (
    support_contract.get("status")
    !=
    "FROZEN_JOINT_SHIFT_COMMON_SUPPORT"
):
    raise RuntimeError(
        "Joint shift/common-support contract "
        "is not frozen."
    )


support_contract_hash = sha256_file(
    SUPPORT_CONTRACT_FILE
)


expected_support_hash = (
    support_contract[
        "artifacts"
    ][
        "domain_oof_common_support"
    ][
        "sha256"
    ]
)


if (
    sha256_file(SUPPORT_FILE)
    !=
    expected_support_hash
):
    raise RuntimeError(
        "Frozen common-support parquet changed."
    )


expected_summary_hash = (
    support_contract[
        "artifacts"
    ][
        "common_support_summary"
    ][
        "sha256"
    ]
)


if (
    sha256_file(SUPPORT_SUMMARY_FILE)
    !=
    expected_summary_hash
):
    raise RuntimeError(
        "Frozen common-support summary changed."
    )


reweighting = support_contract[
    "reweighting"
]


if (
    reweighting[
        "direction"
    ]
    !=
    "CYCLE25_TO_CYCLE24"
):
    raise RuntimeError(
        "Unexpected reweighting direction."
    )


if (
    reweighting[
        "applied_only_inside_common_support"
    ]
    is not True
):
    raise RuntimeError(
        "Weights are not frozen as "
        "common-support-only."
    )


if (
    reweighting[
        "normalization"
    ]
    !=
    "Supported Cycle25 clipped weights normalized to mean 1."
):
    raise RuntimeError(
        "Unexpected frozen weight normalization."
    )


if (
    reweighting[
        "primary_or_sensitivity"
    ]
    !=
    "SENSITIVITY_ONLY"
):
    raise RuntimeError(
        "Frozen weights are not marked "
        "sensitivity-only."
    )


print(
    "Frozen common-support/reweighting contract:",
    "PASS"
)


# ============================================================
# LOAD SUPPORT ARTIFACT
# ============================================================

support = pd.read_parquet(
    SUPPORT_FILE
)


if len(support) != EXPECTED_SUPPORT_ROWS:
    raise RuntimeError(
        "Common-support artifact row-count mismatch."
    )


required_support_columns = {
    "domain_name",
    "domain_label",
    "scope_id",
    "source_row_number",
    ENTITY_GROUP_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    "domain_probability_cycle25",
    "in_common_support",
    "raw_c25_to_c24_weight",
    "clipped_c25_to_c24_weight",
    "analysis_weight",
}


missing_support_columns = (
    required_support_columns
    -
    set(support.columns)
)


if missing_support_columns:
    raise RuntimeError(
        "Missing support columns: "
        f"{sorted(missing_support_columns)}"
    )


if set(
    support[
        "domain_label"
    ].astype(int).unique()
) != {0, 1}:
    raise RuntimeError(
        "Invalid domain-label set."
    )


c24_support_all = support[
    support[
        "domain_label"
    ].astype(int)
    ==
    0
].copy()


c25_support_all = support[
    support[
        "domain_label"
    ].astype(int)
    ==
    1
].copy()


if len(c24_support_all) != EXPECTED_C24_FULL_ROWS:
    raise RuntimeError(
        "C24 support-manifest row-count mismatch."
    )


if len(c25_support_all) != EXPECTED_C25_FULL_ROWS:
    raise RuntimeError(
        "C25 support-manifest row-count mismatch."
    )


# ============================================================
# USE FROZEN SUPPORT FLAG — DO NOT RECOMPUTE
# ============================================================

c24_supported = c24_support_all[
    c24_support_all[
        "in_common_support"
    ].map(parse_bool)
].copy()


c25_supported = c25_support_all[
    c25_support_all[
        "in_common_support"
    ].map(parse_bool)
].copy()


if len(c24_supported) != EXPECTED_C24_SUPPORTED:
    raise RuntimeError(
        "C24 supported row-count mismatch."
    )


if len(c25_supported) != EXPECTED_C25_SUPPORTED:
    raise RuntimeError(
        "C25 supported row-count mismatch."
    )


if not np.isfinite(
    c25_supported[
        "analysis_weight"
    ].to_numpy(
        dtype=np.float64
    )
).all():
    raise RuntimeError(
        "Supported C25 analysis weights "
        "contain non-finite values."
    )


if not (
    c25_supported[
        "analysis_weight"
    ].to_numpy(
        dtype=np.float64
    )
    >
    0
).all():
    raise RuntimeError(
        "Supported C25 analysis weights "
        "must all be positive."
    )


pooled_weight_mean = float(
    c25_supported[
        "analysis_weight"
    ].mean()
)


if not np.isclose(
    pooled_weight_mean,
    1.0,
    atol=1e-12,
    rtol=1e-12,
):
    raise RuntimeError(
        "Frozen supported C25 weight mean "
        "is not 1."
    )


pooled_ess = effective_sample_size(
    c25_supported[
        "analysis_weight"
    ].to_numpy(
        dtype=np.float64
    )
)


if not np.isclose(
    pooled_ess,
    EXPECTED_POOLED_WEIGHTED_ESS,
    atol=1e-6,
    rtol=1e-10,
):
    raise RuntimeError(
        "Frozen pooled Cycle25 weighted ESS mismatch."
    )


print(
    "Frozen support flags / analysis weights:",
    "PASS"
)

print(
    "C24 supported rows:",
    len(c24_supported)
)

print(
    "C25 supported rows:",
    len(c25_supported)
)

print(
    "C25 weighted ESS:",
    f"{pooled_ess:.6f}"
)


# ============================================================
# VERIFY BLOCK SUPPORT COUNTS
# ============================================================

expected_block_supported = {
    "B1": EXPECTED_B1_SUPPORTED,
    "B2": EXPECTED_B2_SUPPORTED,
    "B3": EXPECTED_B3_SUPPORTED,
}


for block_name, scope_id in BLOCK_SCOPE_MAP.items():
    actual = len(
        c25_supported[
            c25_supported[
                "scope_id"
            ].astype(str)
            ==
            scope_id
        ]
    )

    if (
        actual
        !=
        expected_block_supported[
            block_name
        ]
    ):
        raise RuntimeError(
            f"{block_name}: supported row-count mismatch."
        )


print(
    "Frozen B1/B2/B3 support counts:",
    "PASS"
)


# ============================================================
# VERIFY PREVIOUS CHAPTER-10 CONTRACTS
# ============================================================

overall_input_contract = load_json(
    OVERALL_INPUT_CONTRACT_FILE
)


overall_point_contract = load_json(
    OVERALL_POINT_CONTRACT_FILE
)


temporal_point_contract = load_json(
    TEMPORAL_POINT_CONTRACT_FILE
)


class_point_contract = load_json(
    CLASS_POINT_CONTRACT_FILE
)


null_comparison_contract = load_json(
    NULL_COMPARISON_CONTRACT_FILE
)


required_statuses = [
    (
        overall_input_contract.get("status"),
        "FROZEN_OVERALL_CROSS_CYCLE_SHAP_INPUTS",
    ),
    (
        overall_point_contract.get("status"),
        "FROZEN_OVERALL_CROSS_CYCLE_SHAP_POINT_ESTIMATES",
    ),
    (
        temporal_point_contract.get("status"),
        "FROZEN_TEMPORAL_BLOCK_CROSS_CYCLE_SHAP_POINT_ESTIMATES",
    ),
    (
        class_point_contract.get("status"),
        "FROZEN_CLASS_CONDITIONAL_CROSS_CYCLE_SHAP_POINT_ESTIMATES",
    ),
    (
        null_comparison_contract.get("status"),
        "FROZEN_CROSS_CYCLE_SHAP_STRUCTURE_NULL_COMPARISON",
    ),
]


for actual, expected in required_statuses:
    if actual != expected:
        raise RuntimeError(
            f"Required frozen contract missing: {expected}"
        )


print(
    "Previous Chapter-10 frozen contracts:",
    "PASS"
)


# ============================================================
# LOAD MODEL PAIR REGISTRY
# ============================================================

overall_registry_hash = sha256_file(
    OVERALL_INPUT_REGISTRY_FILE
)


if (
    overall_registry_hash
    !=
    overall_input_contract[
        "input_registry"
    ][
        "sha256"
    ]
):
    raise RuntimeError(
        "Frozen model-pair registry changed."
    )


registry = pd.read_csv(
    OVERALL_INPUT_REGISTRY_FILE,
    keep_default_na=False,
)


if len(registry) != EXPECTED_MODELS:
    raise RuntimeError(
        "Expected exactly 3 model pairs."
    )


if (
    registry[
        "candidate_id"
    ].astype(str).tolist()
    !=
    MODEL_ORDER
):
    raise RuntimeError(
        "Frozen model order changed."
    )


# ============================================================
# LOAD PRIMARY OBSERVED METRICS
# ============================================================

overall_primary_file = Path(
    overall_point_contract[
        "structure_output"
    ][
        "file"
    ]
)


temporal_primary_file = Path(
    temporal_point_contract[
        "structure_output"
    ][
        "file"
    ]
)


class_primary_file = Path(
    class_point_contract[
        "structure_output"
    ][
        "file"
    ]
)


for path in [
    overall_primary_file,
    temporal_primary_file,
    class_primary_file,
]:
    verify_exists(path)


if (
    sha256_file(overall_primary_file)
    !=
    overall_point_contract[
        "structure_output"
    ][
        "sha256"
    ]
):
    raise RuntimeError(
        "Primary overall metrics changed."
    )


if (
    sha256_file(temporal_primary_file)
    !=
    temporal_point_contract[
        "structure_output"
    ][
        "sha256"
    ]
):
    raise RuntimeError(
        "Primary temporal metrics changed."
    )


if (
    sha256_file(class_primary_file)
    !=
    class_point_contract[
        "structure_output"
    ][
        "sha256"
    ]
):
    raise RuntimeError(
        "Primary class metrics changed."
    )


primary_overall = pd.read_csv(
    overall_primary_file,
    keep_default_na=False,
)


primary_temporal = pd.read_csv(
    temporal_primary_file,
    keep_default_na=False,
)


primary_class = pd.read_csv(
    class_primary_file,
    keep_default_na=False,
)


# ============================================================
# PRIMARY METRIC LOOKUP
# ============================================================

primary_lookup = {}


for _, row in primary_overall.iterrows():
    key = (
        str(row["candidate_id"]),
        "OVERALL_POOLED",
        "OVERALL",
        str(row["metric"]),
    )

    primary_lookup[key] = float(
        row[
            "metric_value"
        ]
    )


for _, row in primary_temporal.iterrows():
    key = (
        str(row["candidate_id"]),
        "TEMPORAL_BLOCK",
        str(row["cycle25_block"]),
        str(row["metric"]),
    )

    primary_lookup[key] = float(
        row[
            "metric_value"
        ]
    )


for _, row in primary_class.iterrows():
    key = (
        str(row["candidate_id"]),
        "CLASS_CONDITIONAL",
        str(row["stratum"]),
        str(row["metric"]),
    )

    primary_lookup[key] = float(
        row[
            "metric_value"
        ]
    )


# ============================================================
# RESULTS
# ============================================================

structure_rows = []
featurewise_rows = []


# ============================================================
# MODEL LOOP
# ============================================================

for candidate_id in MODEL_ORDER:

    print()
    print(
        "=" * 78
    )

    print(
        "MODEL:",
        candidate_id
    )

    print(
        "=" * 78
    )


    match = registry[
        registry[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(match) != 1:
        raise RuntimeError(
            f"{candidate_id}: model registry lookup failed."
        )


    registry_row = match.iloc[0]


    # --------------------------------------------------------
    # VERIFY SAME-MODEL GUARDRAILS
    # --------------------------------------------------------

    for column in [
        "same_frozen_model",
        "same_explainer",
        "same_input_space",
        "same_output_space",
        "same_frozen_background",
        "same_feature_order",
        "cross_cycle_comparison_allowed",
    ]:
        if not parse_bool(
            registry_row[column]
        ):
            raise RuntimeError(
                f"{candidate_id}: "
                f"guardrail failed: {column}"
            )


    c24_file = Path(
        str(
            registry_row[
                "cycle24_shap_file"
            ]
        )
    )


    c25_file = Path(
        str(
            registry_row[
                "cycle25_shap_file"
            ]
        )
    )


    verify_exists(c24_file)
    verify_exists(c25_file)


    if (
        sha256_file(c24_file)
        !=
        str(
            registry_row[
                "cycle24_shap_sha256"
            ]
        )
    ):
        raise RuntimeError(
            f"{candidate_id}: C24 SHAP changed."
        )


    if (
        sha256_file(c25_file)
        !=
        str(
            registry_row[
                "cycle25_shap_sha256"
            ]
        )
    ):
        raise RuntimeError(
            f"{candidate_id}: C25 SHAP changed."
        )


    # --------------------------------------------------------
    # LOAD SHAP ARTIFACTS
    # --------------------------------------------------------

    c24_shap_frame = pd.read_parquet(
        c24_file,
        columns=[
            "source_row_number",
            ENTITY_GROUP_COLUMN,
            TARGET_COLUMN,
            TIME_COLUMN,
            *SHAP_COLUMNS,
        ],
    )


    c25_shap_frame = pd.read_parquet(
        c25_file,
        columns=[
            "source_row_number",
            "cycle25_block",
            ENTITY_GROUP_COLUMN,
            TARGET_COLUMN,
            TIME_COLUMN,
            *SHAP_COLUMNS,
        ],
    )


    if len(c24_shap_frame) != EXPECTED_C24_FULL_ROWS:
        raise RuntimeError(
            f"{candidate_id}: C24 SHAP row mismatch."
        )


    if len(c25_shap_frame) != EXPECTED_C25_FULL_ROWS:
        raise RuntimeError(
            f"{candidate_id}: C25 SHAP row mismatch."
        )


    # --------------------------------------------------------
    # ALIGN C24 SUPPORT ↔ C24 SHAP
    # --------------------------------------------------------

    c24_joined = c24_supported.merge(
        c24_shap_frame,
        on="source_row_number",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_support",
            "_shap",
        ),
    )


    if len(c24_joined) != EXPECTED_C24_SUPPORTED:
        raise RuntimeError(
            f"{candidate_id}: C24 support/SHAP "
            "alignment row mismatch."
        )


    verify_metadata_alignment(
        c24_joined,
        f"{candidate_id} C24",
    )


    # --------------------------------------------------------
    # ALIGN C25 SUPPORT ↔ C25 SHAP
    # --------------------------------------------------------

    c25_joined = c25_supported.merge(
        c25_shap_frame,
        on="source_row_number",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_support",
            "_shap",
        ),
    )


    if len(c25_joined) != EXPECTED_C25_SUPPORTED:
        raise RuntimeError(
            f"{candidate_id}: C25 support/SHAP "
            "alignment row mismatch."
        )


    verify_metadata_alignment(
        c25_joined,
        f"{candidate_id} C25",
    )


    # --------------------------------------------------------
    # VERIFY BLOCK LABEL ↔ FROZEN SCOPE
    # --------------------------------------------------------

    for block_name, scope_id in BLOCK_SCOPE_MAP.items():

        block_rows = c25_joined[
            c25_joined[
                "scope_id"
            ].astype(str)
            ==
            scope_id
        ]


        if not (
            block_rows[
                "cycle25_block"
            ].astype(str)
            ==
            block_name
        ).all():
            raise RuntimeError(
                f"{candidate_id}: {block_name} "
                "scope/block alignment failed."
            )


    print(
        "Support ↔ SHAP row alignment:",
        "PASS"
    )


    # ========================================================
    # INTERNAL ANALYSIS FUNCTION
    # ========================================================

    def process_scope(
        analysis_scope,
        scope_key,
        c24_rows,
        c25_rows,
        reference_label,
        transfer_label,
    ):

        if len(c24_rows) == 0:
            raise RuntimeError(
                f"{candidate_id} {scope_key}: "
                "empty C24 sensitivity subset."
            )


        if len(c25_rows) == 0:
            raise RuntimeError(
                f"{candidate_id} {scope_key}: "
                "empty C25 sensitivity subset."
            )


        c24_matrix = c24_rows[
            SHAP_COLUMNS
        ].to_numpy(
            dtype=np.float64
        )


        c25_matrix = c25_rows[
            SHAP_COLUMNS
        ].to_numpy(
            dtype=np.float64
        )


        c25_weights = c25_rows[
            "analysis_weight"
        ].to_numpy(
            dtype=np.float64
        )


        if not np.isfinite(
            c25_weights
        ).all():
            raise RuntimeError(
                f"{candidate_id} {scope_key}: "
                "non-finite frozen weights."
            )


        if not (
            c25_weights > 0
        ).all():
            raise RuntimeError(
                f"{candidate_id} {scope_key}: "
                "non-positive frozen weights."
            )


        reference_mean_abs, reference_mean_signed = (
            attribution_vectors(
                c24_matrix,
                weights=None,
            )
        )


        reference_ranks = deterministic_ranks(
            reference_mean_abs
        )


        transfer_ess = effective_sample_size(
            c25_weights
        )


        for variant in SENSITIVITY_VARIANTS:

            if (
                variant
                ==
                "COMMON_SUPPORT_UNWEIGHTED"
            ):
                transfer_mean_abs, transfer_mean_signed = (
                    attribution_vectors(
                        c25_matrix,
                        weights=None,
                    )
                )


            elif (
                variant
                ==
                "COMMON_SUPPORT_REWEIGHTED"
            ):
                transfer_mean_abs, transfer_mean_signed = (
                    attribution_vectors(
                        c25_matrix,
                        weights=c25_weights,
                    )
                )


            else:
                raise RuntimeError(
                    f"Unexpected sensitivity variant: "
                    f"{variant}"
                )


            transfer_ranks = deterministic_ranks(
                transfer_mean_abs
            )


            metrics = calculate_structure_metrics(
                mean_abs_a=reference_mean_abs,
                mean_abs_b=transfer_mean_abs,
                mean_signed_a=reference_mean_signed,
                mean_signed_b=transfer_mean_signed,
            )


            metric_text = []


            for metric_name in METRIC_ORDER:

                metric_value = float(
                    metrics[
                        metric_name
                    ]
                )


                primary_key = (
                    candidate_id,
                    analysis_scope,
                    scope_key,
                    metric_name,
                )


                if primary_key not in primary_lookup:
                    raise RuntimeError(
                        f"Missing primary metric: "
                        f"{primary_key}"
                    )


                primary_value = float(
                    primary_lookup[
                        primary_key
                    ]
                )


                structure_rows.append(
                    {
                        "candidate_id":
                            candidate_id,

                        "model_family":
                            str(
                                registry_row[
                                    "model_family"
                                ]
                            ),

                        "output_space":
                            str(
                                registry_row[
                                    "output_space"
                                ]
                            ),

                        "analysis_scope":
                            analysis_scope,

                        "scope_key":
                            scope_key,

                        "reference_population":
                            reference_label,

                        "transfer_population":
                            transfer_label,

                        "sensitivity_variant":
                            variant,

                        "reference_supported_rows":
                            len(c24_rows),

                        "transfer_supported_rows":
                            len(c25_rows),

                        "reference_entity_groups":
                            int(
                                c24_rows[
                                    f"{ENTITY_GROUP_COLUMN}_support"
                                ]
                                .astype(str)
                                .nunique()
                            ),

                        "transfer_entity_groups":
                            int(
                                c25_rows[
                                    f"{ENTITY_GROUP_COLUMN}_support"
                                ]
                                .astype(str)
                                .nunique()
                            ),

                        "transfer_weighted_ess":
                            (
                                transfer_ess
                                if
                                variant
                                ==
                                "COMMON_SUPPORT_REWEIGHTED"
                                else
                                float(len(c25_rows))
                            ),

                        "metric":
                            metric_name,

                        "primary_full_population_value":
                            primary_value,

                        "sensitivity_value":
                            metric_value,

                        "delta_sensitivity_minus_primary":
                            (
                                metric_value
                                -
                                primary_value
                            ),

                        "frozen_null_threshold_applied":
                            False,

                        "final_drift_conclusion":
                            "NOT_MADE",

                        "final_stability_conclusion":
                            "NOT_MADE",
                    }
                )


                metric_text.append(
                    f"{metric_name}="
                    f"{metric_value:.6f}"
                )


            # ------------------------------------------------
            # FEATUREWISE SENSITIVITY
            # ------------------------------------------------

            for feature_index, feature in enumerate(
                FEATURES
            ):

                c24_abs = float(
                    reference_mean_abs[
                        feature_index
                    ]
                )


                c25_abs = float(
                    transfer_mean_abs[
                        feature_index
                    ]
                )


                c24_signed = float(
                    reference_mean_signed[
                        feature_index
                    ]
                )


                c25_signed = float(
                    transfer_mean_signed[
                        feature_index
                    ]
                )


                featurewise_rows.append(
                    {
                        "candidate_id":
                            candidate_id,

                        "model_family":
                            str(
                                registry_row[
                                    "model_family"
                                ]
                            ),

                        "output_space":
                            str(
                                registry_row[
                                    "output_space"
                                ]
                            ),

                        "analysis_scope":
                            analysis_scope,

                        "scope_key":
                            scope_key,

                        "reference_population":
                            reference_label,

                        "transfer_population":
                            transfer_label,

                        "sensitivity_variant":
                            variant,

                        "feature_index":
                            feature_index + 1,

                        "feature":
                            feature,

                        "reference_mean_absolute_shap":
                            c24_abs,

                        "transfer_mean_absolute_shap":
                            c25_abs,

                        "delta_mean_absolute_shap":
                            (
                                c25_abs
                                -
                                c24_abs
                            ),

                        "reference_mean_signed_shap":
                            c24_signed,

                        "transfer_mean_signed_shap":
                            c25_signed,

                        "delta_mean_signed_shap":
                            (
                                c25_signed
                                -
                                c24_signed
                            ),

                        "reference_mean_abs_rank":
                            int(
                                reference_ranks[
                                    feature_index
                                ]
                            ),

                        "transfer_mean_abs_rank":
                            int(
                                transfer_ranks[
                                    feature_index
                                ]
                            ),

                        "rank_change_transfer_minus_reference":
                            int(
                                transfer_ranks[
                                    feature_index
                                ]
                                -
                                reference_ranks[
                                    feature_index
                                ]
                            ),

                        "signed_direction_agreement":
                            bool(
                                np.sign(
                                    c24_signed
                                )
                                ==
                                np.sign(
                                    c25_signed
                                )
                            ),

                        "frozen_null_threshold_applied":
                            False,

                        "featurewise_inference_performed":
                            False,

                        "fdr_applied":
                            False,
                    }
                )


            print(
                f"{analysis_scope} | "
                f"{scope_key} | "
                f"{variant} | "
                f"C24={len(c24_rows)} | "
                f"C25={len(c25_rows)} | "
                f"ESS="
                f"{(transfer_ess if variant == 'COMMON_SUPPORT_REWEIGHTED' else len(c25_rows)):.2f}"
            )

            print(
                "  "
                +
                " | ".join(
                    metric_text
                )
            )


    # ========================================================
    # OVERALL POOLED
    # ========================================================

    process_scope(
        analysis_scope="OVERALL_POOLED",
        scope_key="OVERALL",
        c24_rows=c24_joined,
        c25_rows=c25_joined,
        reference_label="CYCLE24_COMMON_SUPPORT",
        transfer_label="CYCLE25_POOLED_COMMON_SUPPORT",
    )


    # ========================================================
    # TEMPORAL BLOCKS
    # ========================================================

    for block_name, scope_id in (
        BLOCK_SCOPE_MAP.items()
    ):

        c25_block = c25_joined[
            c25_joined[
                "scope_id"
            ].astype(str)
            ==
            scope_id
        ].copy()


        process_scope(
            analysis_scope="TEMPORAL_BLOCK",
            scope_key=block_name,
            c24_rows=c24_joined,
            c25_rows=c25_block,
            reference_label="CYCLE24_COMMON_SUPPORT",
            transfer_label=(
                f"CYCLE25_{block_name}_COMMON_SUPPORT"
            ),
        )


    # ========================================================
    # CLASS-CONDITIONAL
    # ========================================================

    for target_value in [
        0,
        1,
    ]:

        stratum = (
            f"TARGET_{target_value}"
        )


        c24_class = c24_joined[
            c24_joined[
                f"{TARGET_COLUMN}_support"
            ].astype(int)
            ==
            target_value
        ].copy()


        c25_class = c25_joined[
            c25_joined[
                f"{TARGET_COLUMN}_support"
            ].astype(int)
            ==
            target_value
        ].copy()


        process_scope(
            analysis_scope="CLASS_CONDITIONAL",
            scope_key=stratum,
            c24_rows=c24_class,
            c25_rows=c25_class,
            reference_label=(
                f"CYCLE24_COMMON_SUPPORT_{stratum}"
            ),
            transfer_label=(
                f"CYCLE25_COMMON_SUPPORT_{stratum}"
            ),
        )


# ============================================================
# BUILD TABLES
# ============================================================

structure = pd.DataFrame(
    structure_rows
)


featurewise = pd.DataFrame(
    featurewise_rows
)


if len(structure) != EXPECTED_STRUCTURE_ROWS:
    raise RuntimeError(
        "Expected "
        f"{EXPECTED_STRUCTURE_ROWS} structure rows, "
        f"got {len(structure)}."
    )


if len(featurewise) != EXPECTED_FEATUREWISE_ROWS:
    raise RuntimeError(
        "Expected "
        f"{EXPECTED_FEATUREWISE_ROWS} featurewise rows, "
        f"got {len(featurewise)}."
    )


# ============================================================
# VERIFY METRIC INVENTORY
# ============================================================

group_columns = [
    "candidate_id",
    "analysis_scope",
    "scope_key",
    "sensitivity_variant",
]


metric_counts = (
    structure
    .groupby(
        group_columns
    )[
        "metric"
    ]
    .nunique()
)


if not (
    metric_counts
    ==
    8
).all():
    raise RuntimeError(
        "Every sensitivity comparison must "
        "contain exactly 8 metrics."
    )


print()

print(
    "Sensitivity structure inventory:",
    "PASS"
)

print(
    "Sensitivity featurewise inventory:",
    "PASS"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

STRUCTURE_OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


FEATUREWISE_OUTPUT_FILE.parent.mkdir(
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

structure_temp = STRUCTURE_OUTPUT_FILE.with_name(
    STRUCTURE_OUTPUT_FILE.stem
    +
    ".tmp.csv"
)


featurewise_temp = FEATUREWISE_OUTPUT_FILE.with_name(
    FEATUREWISE_OUTPUT_FILE.stem
    +
    ".tmp.csv"
)


for path in [
    structure_temp,
    featurewise_temp,
]:
    if path.exists():
        path.unlink()


structure.to_csv(
    structure_temp,
    index=False,
)


featurewise.to_csv(
    featurewise_temp,
    index=False,
)


structure_rt = pd.read_csv(
    structure_temp,
    keep_default_na=False,
)


featurewise_rt = pd.read_csv(
    featurewise_temp,
    keep_default_na=False,
)


if len(structure_rt) != EXPECTED_STRUCTURE_ROWS:
    raise RuntimeError(
        "Structure sensitivity CSV "
        "round-trip failed."
    )


if len(featurewise_rt) != EXPECTED_FEATUREWISE_ROWS:
    raise RuntimeError(
        "Featurewise sensitivity CSV "
        "round-trip failed."
    )


structure_temp.replace(
    STRUCTURE_OUTPUT_FILE
)


featurewise_temp.replace(
    FEATUREWISE_OUTPUT_FILE
)


structure_hash = sha256_file(
    STRUCTURE_OUTPUT_FILE
)


featurewise_hash = sha256_file(
    FEATUREWISE_OUTPUT_FILE
)


print(
    "Structure sensitivity CSV round-trip:",
    "PASS"
)

print(
    "Featurewise sensitivity CSV round-trip:",
    "PASS"
)


# ============================================================
# CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Common-Support / Reweighting "
            "Cross-Cycle SHAP Sensitivity"
        ),

    "version":
        "1.0",

    "status":
        (
            "FROZEN_COMMON_SUPPORT_REWEIGHTING_"
            "SHAP_SENSITIVITY"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_7_2",

    "frozen_support_source":
        {
            "contract_file":
                SUPPORT_CONTRACT_FILE.as_posix(),

            "contract_sha256":
                support_contract_hash,

            "required_status":
                "FROZEN_JOINT_SHIFT_COMMON_SUPPORT",

            "support_file":
                SUPPORT_FILE.as_posix(),

            "support_sha256":
                sha256_file(
                    SUPPORT_FILE
                ),

            "support_summary_file":
                SUPPORT_SUMMARY_FILE.as_posix(),

            "support_summary_sha256":
                sha256_file(
                    SUPPORT_SUMMARY_FILE
                ),
        },

    "common_support":
        {
            "lower":
                support_contract[
                    "common_support"
                ][
                    "support_lower"
                ],

            "upper":
                support_contract[
                    "common_support"
                ][
                    "support_upper"
                ],

            "cycle24_supported_rows":
                EXPECTED_C24_SUPPORTED,

            "cycle25_supported_rows":
                EXPECTED_C25_SUPPORTED,

            "support_flags_recomputed":
                False,
        },

    "reweighting":
        {
            "direction":
                reweighting[
                    "direction"
                ],

            "frozen_raw_formula":
                reweighting[
                    "raw_weight_formula"
                ],

            "frozen_clip_quantiles":
                reweighting[
                    "raw_weight_clip_quantiles"
                ],

            "frozen_clip_lower":
                reweighting[
                    "raw_weight_clip_lower"
                ],

            "frozen_clip_upper":
                reweighting[
                    "raw_weight_clip_upper"
                ],

            "frozen_normalization":
                reweighting[
                    "normalization"
                ],

            "weights_recomputed":
                False,

            "weights_reclipped":
                False,

            "weights_refit":
                False,

            "analysis_weight_used_directly":
                True,

            "pooled_supported_cycle25_ess":
                pooled_ess,
        },

    "sensitivity_variants":
        SENSITIVITY_VARIANTS,

    "analysis_scopes":
        [
            "OVERALL_POOLED",
            "TEMPORAL_BLOCK_B1",
            "TEMPORAL_BLOCK_B2",
            "TEMPORAL_BLOCK_B3",
            "CLASS_CONDITIONAL_TARGET_0",
            "CLASS_CONDITIONAL_TARGET_1",
        ],

    "metric_definition":
        {
            "epsilon":
                EPSILON,

            "exact_chapter9_structure_formulas_reused":
                True,

            "metrics":
                METRIC_ORDER,
        },

    "structure_output":
        {
            "file":
                STRUCTURE_OUTPUT_FILE.as_posix(),

            "sha256":
                structure_hash,

            "rows":
                EXPECTED_STRUCTURE_ROWS,
        },

    "featurewise_output":
        {
            "file":
                FEATUREWISE_OUTPUT_FILE.as_posix(),

            "sha256":
                featurewise_hash,

            "rows":
                EXPECTED_FEATUREWISE_ROWS,

            "descriptive_only":
                True,
        },

    "scientific_guardrails":
        {
            "cycle25_used_for_training":
                False,

            "models_retrained":
                False,

            "shap_recomputed":
                False,

            "background_changed":
                False,

            "propensity_model_refit":
                False,

            "support_bounds_recomputed":
                False,

            "weights_recomputed":
                False,

            "weights_reclipped":
                False,

            "frozen_analysis_weight_used":
                True,

            "common_support_sensitivity_applied":
                True,

            "reweighting_sensitivity_applied":
                True,

            "chapter9_null_thresholds_applied_to_sensitivity_estimand":
                False,

            "featurewise_inference_performed":
                False,

            "fdr_applied":
                False,

            "correlated_group_robustness_applied":
                False,

            "final_drift_conclusion_made":
                False,

            "final_stability_conclusion_made":
                False,

            "shap_is_causal":
                False,
        },

    "interpretation_rule":
        (
            "Common-support and C25-to-C24 reweighting "
            "results are sensitivity analyses only. "
            "They assess whether observed attribution "
            "structure changes are sensitive to domain "
            "overlap and covariate-distribution mismatch. "
            "The frozen full-population Cycle24 null "
            "thresholds are not directly reapplied because "
            "the reference estimand is support-restricted. "
            "Persistence or attenuation of patterns is "
            "therefore robustness evidence rather than "
            "a standalone drift/stability conclusion."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_7_3_"
            "PREVALENCE_STANDARDIZED_SHAP_SENSITIVITY"
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
    contract_rt.get("status")
    !=
    "FROZEN_COMMON_SUPPORT_REWEIGHTING_SHAP_SENSITIVITY"
):
    raise RuntimeError(
        "10.7.2 contract round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "final_drift_conclusion_made"
    ]
    is not False
):
    raise RuntimeError(
        "Unexpected final drift conclusion."
    )


contract_temp.replace(
    OUTPUT_CONTRACT_FILE
)


contract_hash = sha256_file(
    OUTPUT_CONTRACT_FILE
)


# ============================================================
# FINAL
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 10.7.2 — COMPLETE"
)

print("=" * 78)


print(
    "Models:",
    EXPECTED_MODELS
)

print(
    "Sensitivity variants:",
    2
)

print(
    "Comparison scopes/model:",
    6
)

print(
    "Structure rows:",
    len(structure)
)

print(
    "Featurewise rows:",
    len(featurewise)
)


print()

print(
    "C24 supported rows:",
    EXPECTED_C24_SUPPORTED
)

print(
    "C25 supported rows:",
    EXPECTED_C25_SUPPORTED
)

print(
    "C25 pooled weighted ESS:",
    f"{pooled_ess:.6f}"
)


print()

print(
    "Frozen support flags reused:",
    True
)

print(
    "Frozen weights reused:",
    True
)

print(
    "Weights recomputed:",
    False
)

print(
    "Weights re-clipped:",
    False
)

print(
    "Chapter-9 null applied here:",
    False
)


print()

print(
    "Featurewise inference / FDR:",
    False
)

print(
    "Final drift conclusion:",
    False
)

print(
    "Final stability conclusion:",
    False
)


print()

print(
    "Status:",
    (
        "FROZEN_COMMON_SUPPORT_REWEIGHTING_"
        "SHAP_SENSITIVITY"
    )
)


print()

print(
    "Structure sensitivity:",
    STRUCTURE_OUTPUT_FILE
)

print(
    "Featurewise sensitivity:",
    FEATUREWISE_OUTPUT_FILE
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
        "Chapter 10.7.3 — "
        "Prevalence-Standardized SHAP Sensitivity"
    )
)