# ============================================================
# MODEL CHAPTER 10.8.2
# CORRELATED FEATURE-GROUP SHAP ROBUSTNESS
#
# Frozen grouping source:
#   Cycle24 TRAIN only
#
# Frozen grouping rule:
#   Spearman |rho| >= 0.80
#   Undirected graph
#   Connected components
#   Singleton features preserved
#
# Frozen group-level attribution rule:
#   Sum member-feature mean absolute SHAP magnitudes.
#
# IMPORTANT:
#   - Individual-feature analysis is retained.
#   - Group definitions are NOT recomputed.
#   - NO Cycle25-based regrouping.
#   - NO signed group aggregation is invented.
#   - NO group-level null threshold is applied because
#     no group-level Cycle24 null was frozen.
#   - Top-10 overlap is omitted because there are exactly
#     10 frozen groups, making top-10 overlap identically 1.
#   - NO final drift/stability conclusion is made here.
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json

import numpy as np
import pandas as pd

from scipy.stats import (
    spearmanr,
    kendalltau,
)


# ============================================================
# CONSTANTS
# ============================================================

EPSILON = 1e-12

EXPECTED_FEATURES = 25
EXPECTED_GROUPS = 10
EXPECTED_MODELS = 3

PRIMARY_SCOPES_PER_MODEL = 6
SENSITIVITY_SCOPES_PER_MODEL = 6
SENSITIVITY_VARIANTS = 2

GROUP_METRICS = [
    "spearman_rho",
    "kendall_tau_b",
    "top5_overlap",
    "normalized_l1_mean_abs",
    "cosine_distance_mean_abs",
]


EXPECTED_PRIMARY_STRUCTURE_ROWS = (
    EXPECTED_MODELS
    * PRIMARY_SCOPES_PER_MODEL
    * len(GROUP_METRICS)
)

EXPECTED_SENSITIVITY_STRUCTURE_ROWS = (
    EXPECTED_MODELS
    * SENSITIVITY_SCOPES_PER_MODEL
    * SENSITIVITY_VARIANTS
    * len(GROUP_METRICS)
)

EXPECTED_STRUCTURE_ROWS = (
    EXPECTED_PRIMARY_STRUCTURE_ROWS
    +
    EXPECTED_SENSITIVITY_STRUCTURE_ROWS
)


EXPECTED_PRIMARY_PROFILE_ROWS = (
    EXPECTED_MODELS
    * PRIMARY_SCOPES_PER_MODEL
    * EXPECTED_GROUPS
)

EXPECTED_SENSITIVITY_PROFILE_ROWS = (
    EXPECTED_MODELS
    * SENSITIVITY_SCOPES_PER_MODEL
    * SENSITIVITY_VARIANTS
    * EXPECTED_GROUPS
)

EXPECTED_PROFILE_ROWS = (
    EXPECTED_PRIMARY_PROFILE_ROWS
    +
    EXPECTED_SENSITIVITY_PROFILE_ROWS
)


MODEL_ORDER = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


FEATURE_ORDER = [
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


# ============================================================
# FROZEN GROUP INPUTS
# ============================================================

GROUP_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "correlated_feature_groups_contract.json"
)


GROUP_MANIFEST_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "correlated_feature_groups.csv"
)


# ============================================================
# PRIMARY CROSS-CYCLE CONTRACTS
# ============================================================

OVERALL_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "overall_cross_cycle_shap_point_estimates_contract.json"
)


TEMPORAL_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "temporal_block_cross_cycle_shap_point_estimates_contract.json"
)


CLASS_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "class_conditional_cross_cycle_shap_point_estimates_contract.json"
)


NULL_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "cross_cycle_shap_structure_null_comparison_contract.json"
)


# ============================================================
# SUPPORT / REWEIGHTING CONTRACT
# ============================================================

SENSITIVITY_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "common_support_reweighting_shap_sensitivity_contract.json"
)


# ============================================================
# OUTPUTS
# ============================================================

PROFILE_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "correlated_group_shap_profiles.csv"
)


STRUCTURE_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "correlated_group_shap_structure_robustness.csv"
)


OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "correlated_group_shap_robustness_contract.json"
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

        return json.load(
            file
        )


def verify_exists(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def parse_bool(value) -> bool:

    if isinstance(
        value,
        (bool, np.bool_),
    ):

        return bool(value)


    value = str(
        value
    ).strip().lower()


    if value in {
        "true",
        "1",
    }:

        return True


    if value in {
        "false",
        "0",
    }:

        return False


    raise RuntimeError(
        f"Cannot parse boolean: {value}"
    )


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    norm_a = float(
        np.linalg.norm(a)
    )

    norm_b = float(
        np.linalg.norm(b)
    )


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


def deterministic_top_k_overlap(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
) -> float:

    group_index = np.arange(
        len(a),
        dtype=np.int64,
    )


    order_a = np.lexsort(
        (
            group_index,
            -a,
        )
    )


    order_b = np.lexsort(
        (
            group_index,
            -b,
        )
    )


    top_a = set(
        order_a[
            :k
        ].tolist()
    )


    top_b = set(
        order_b[
            :k
        ].tolist()
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


def normalized_l1_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    numerator = float(
        np.sum(
            np.abs(
                a
                -
                b
            )
        )
    )


    denominator = float(
        np.sum(
            np.abs(a)
        )
        +
        np.sum(
            np.abs(b)
        )
    )


    if denominator <= EPSILON:

        return 0.0


    return numerator / denominator


def calculate_group_structure_metrics(
    reference_vector: np.ndarray,
    transfer_vector: np.ndarray,
):

    spearman_result = spearmanr(
        reference_vector,
        transfer_vector,
    )


    kendall_result = kendalltau(
        reference_vector,
        transfer_vector,
        variant="b",
    )


    spearman_value = float(
        spearman_result.statistic
    )


    kendall_value = float(
        kendall_result.statistic
    )


    if not np.isfinite(
        spearman_value
    ):

        raise RuntimeError(
            "Non-finite group-level Spearman rho."
        )


    if not np.isfinite(
        kendall_value
    ):

        raise RuntimeError(
            "Non-finite group-level Kendall tau-b."
        )


    return {

        "spearman_rho":
            spearman_value,

        "kendall_tau_b":
            kendall_value,

        "top5_overlap":
            deterministic_top_k_overlap(
                reference_vector,
                transfer_vector,
                5,
            ),

        "normalized_l1_mean_abs":
            normalized_l1_distance(
                reference_vector,
                transfer_vector,
            ),

        "cosine_distance_mean_abs":
            (
                1.0
                -
                cosine_similarity(
                    reference_vector,
                    transfer_vector,
                )
            ),
    }


def deterministic_ranks(
    values: np.ndarray,
) -> np.ndarray:

    group_index = np.arange(
        len(values),
        dtype=np.int64,
    )


    order = np.lexsort(
        (
            group_index,
            -values,
        )
    )


    ranks = np.empty(
        len(values),
        dtype=np.int64,
    )


    ranks[
        order
    ] = np.arange(
        1,
        len(values) + 1,
        dtype=np.int64,
    )


    return ranks


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.8.2 — "
    "CORRELATED FEATURE-GROUP SHAP ROBUSTNESS"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

for path in [
    GROUP_CONTRACT_FILE,
    GROUP_MANIFEST_FILE,
    OVERALL_CONTRACT_FILE,
    TEMPORAL_CONTRACT_FILE,
    CLASS_CONTRACT_FILE,
    NULL_CONTRACT_FILE,
    SENSITIVITY_CONTRACT_FILE,
]:

    verify_exists(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    PROFILE_OUTPUT_FILE,
    STRUCTURE_OUTPUT_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter 10.8.2 outputs."
        )


# ============================================================
# VERIFY FROZEN GROUP CONTRACT
# ============================================================

group_contract = load_json(
    GROUP_CONTRACT_FILE
)


if (
    group_contract.get(
        "status"
    )
    !=
    "FROZEN_CORRELATED_FEATURE_GROUPS"
):

    raise RuntimeError(
        "Correlated feature groups "
        "are not frozen."
    )


group_contract_hash = sha256_file(
    GROUP_CONTRACT_FILE
)


if (
    group_contract[
        "source_population"
    ][
        "name"
    ]
    !=
    "CYCLE24_TRAIN_ONLY"
):

    raise RuntimeError(
        "Group source population changed."
    )


design = group_contract[
    "correlation_design"
]


if (
    design[
        "method"
    ]
    !=
    "SPEARMAN"
):

    raise RuntimeError(
        "Unexpected correlation method."
    )


if not np.isclose(
    float(
        design[
            "absolute_rho_threshold"
        ]
    ),
    0.8,
    atol=0.0,
    rtol=0.0,
):

    raise RuntimeError(
        "Unexpected correlation threshold."
    )


if (
    design[
        "edge_rule"
    ]
    !=
    "ABS(SPEARMAN_RHO) >= 0.80"
):

    raise RuntimeError(
        "Unexpected correlation edge rule."
    )


if (
    design[
        "graph_type"
    ]
    !=
    "UNDIRECTED"
):

    raise RuntimeError(
        "Unexpected graph type."
    )


if (
    design[
        "grouping_method"
    ]
    !=
    "CONNECTED_COMPONENTS"
):

    raise RuntimeError(
        "Unexpected grouping method."
    )


if (
    design[
        "transitive_grouping"
    ]
    is not True
):

    raise RuntimeError(
        "Transitive grouping changed."
    )


if (
    design[
        "singleton_features_preserved"
    ]
    is not True
):

    raise RuntimeError(
        "Singleton rule changed."
    )


future_shap_use = group_contract[
    "future_shap_use"
]


if (
    future_shap_use[
        "individual_feature_analysis_retained"
    ]
    is not True
):

    raise RuntimeError(
        "Individual-feature analysis "
        "must remain retained."
    )


if (
    future_shap_use[
        "group_level_analysis_required"
    ]
    is not True
):

    raise RuntimeError(
        "Group-level analysis is not "
        "marked required."
    )


if (
    future_shap_use[
        "group_level_mean_abs_shap_rule"
    ]
    !=
    "Sum member-feature mean absolute SHAP magnitudes."
):

    raise RuntimeError(
        "Frozen group-level SHAP rule changed."
    )


if (
    future_shap_use[
        "groups_may_not_be_changed_using_cycle25"
    ]
    is not True
):

    raise RuntimeError(
        "Cycle25 grouping guardrail changed."
    )


print(
    "Frozen correlated-group contract:",
    "PASS"
)


# ============================================================
# VERIFY GROUP MANIFEST
# ============================================================

expected_group_manifest_hash = (
    group_contract[
        "artifacts"
    ][
        "correlated_feature_groups"
    ][
        "sha256"
    ]
)


if (
    sha256_file(
        GROUP_MANIFEST_FILE
    )
    !=
    expected_group_manifest_hash
):

    raise RuntimeError(
        "Frozen correlated-group manifest changed."
    )


group_manifest = pd.read_csv(
    GROUP_MANIFEST_FILE,
    keep_default_na=False,
)


if len(
    group_manifest
) != EXPECTED_FEATURES:

    raise RuntimeError(
        "Expected exactly 25 feature-to-group rows."
    )


if (
    group_manifest[
        "feature"
    ].astype(str).tolist()
    !=
    FEATURE_ORDER
):

    raise RuntimeError(
        "Frozen feature order changed."
    )


if (
    group_manifest[
        "feature"
    ].astype(str).nunique()
    !=
    EXPECTED_FEATURES
):

    raise RuntimeError(
        "Feature-to-group mapping "
        "contains duplicate features."
    )


GROUP_ORDER = [
    str(
        group[
            "group_id"
        ]
    )
    for group
    in group_contract[
        "group_results"
    ][
        "groups"
    ]
]


if len(
    GROUP_ORDER
) != EXPECTED_GROUPS:

    raise RuntimeError(
        "Expected exactly 10 frozen groups."
    )


if (
    group_manifest[
        "group_id"
    ].astype(str).nunique()
    !=
    EXPECTED_GROUPS
):

    raise RuntimeError(
        "Frozen group count mismatch."
    )


FEATURE_TO_GROUP = dict(
    zip(
        group_manifest[
            "feature"
        ].astype(str),

        group_manifest[
            "group_id"
        ].astype(str),
    )
)


GROUP_METADATA = {}


for group_id in GROUP_ORDER:

    subset = group_manifest[
        group_manifest[
            "group_id"
        ].astype(str)
        ==
        group_id
    ].copy()


    if len(
        subset
    ) == 0:

        raise RuntimeError(
            f"Missing group: {group_id}"
        )


    declared_size = int(
        subset.iloc[
            0
        ][
            "group_size"
        ]
    )


    if len(
        subset
    ) != declared_size:

        raise RuntimeError(
            f"{group_id}: group-size mismatch."
        )


    members = subset[
        "feature"
    ].astype(str).tolist()


    GROUP_METADATA[
        group_id
    ] = {
        "group_size":
            declared_size,

        "members":
            members,

        "group_members":
            "|".join(
                members
            ),

        "is_singleton":
            bool(
                declared_size
                ==
                1
            ),
    }


print(
    "Frozen 25-feature → 10-group mapping:",
    "PASS"
)


# ============================================================
# VERIFY REQUIRED CHAPTER-10 CONTRACTS
# ============================================================

overall_contract = load_json(
    OVERALL_CONTRACT_FILE
)


temporal_contract = load_json(
    TEMPORAL_CONTRACT_FILE
)


class_contract = load_json(
    CLASS_CONTRACT_FILE
)


null_contract = load_json(
    NULL_CONTRACT_FILE
)


sensitivity_contract = load_json(
    SENSITIVITY_CONTRACT_FILE
)


required_statuses = [
    (
        overall_contract.get("status"),
        "FROZEN_OVERALL_CROSS_CYCLE_SHAP_POINT_ESTIMATES",
    ),
    (
        temporal_contract.get("status"),
        "FROZEN_TEMPORAL_BLOCK_CROSS_CYCLE_SHAP_POINT_ESTIMATES",
    ),
    (
        class_contract.get("status"),
        "FROZEN_CLASS_CONDITIONAL_CROSS_CYCLE_SHAP_POINT_ESTIMATES",
    ),
    (
        null_contract.get("status"),
        "FROZEN_CROSS_CYCLE_SHAP_STRUCTURE_NULL_COMPARISON",
    ),
    (
        sensitivity_contract.get("status"),
        "FROZEN_COMMON_SUPPORT_REWEIGHTING_SHAP_SENSITIVITY",
    ),
]


for actual, expected in required_statuses:

    if actual != expected:

        raise RuntimeError(
            "Required frozen contract "
            f"missing: {expected}"
        )


print(
    "Required Chapter-10 frozen contracts:",
    "PASS"
)


# ============================================================
# INPUT FILES FROM CONTRACTS
# ============================================================

overall_feature_file = Path(
    overall_contract[
        "featurewise_output"
    ][
        "file"
    ]
)


overall_structure_file = Path(
    overall_contract[
        "structure_output"
    ][
        "file"
    ]
)


temporal_feature_file = Path(
    temporal_contract[
        "featurewise_output"
    ][
        "file"
    ]
)


temporal_structure_file = Path(
    temporal_contract[
        "structure_output"
    ][
        "file"
    ]
)


class_feature_file = Path(
    class_contract[
        "featurewise_output"
    ][
        "file"
    ]
)


class_structure_file = Path(
    class_contract[
        "structure_output"
    ][
        "file"
    ]
)


sensitivity_feature_file = Path(
    sensitivity_contract[
        "featurewise_output"
    ][
        "file"
    ]
)


sensitivity_structure_file = Path(
    sensitivity_contract[
        "structure_output"
    ][
        "file"
    ]
)


null_comparison_file = Path(
    null_contract[
        "comparison_output"
    ][
        "file"
    ]
)


for path in [
    overall_feature_file,
    overall_structure_file,
    temporal_feature_file,
    temporal_structure_file,
    class_feature_file,
    class_structure_file,
    sensitivity_feature_file,
    sensitivity_structure_file,
    null_comparison_file,
]:

    verify_exists(
        path
    )


# ============================================================
# HASH VALIDATION
# ============================================================

hash_checks = [
    (
        overall_feature_file,
        overall_contract[
            "featurewise_output"
        ][
            "sha256"
        ],
    ),
    (
        overall_structure_file,
        overall_contract[
            "structure_output"
        ][
            "sha256"
        ],
    ),
    (
        temporal_feature_file,
        temporal_contract[
            "featurewise_output"
        ][
            "sha256"
        ],
    ),
    (
        temporal_structure_file,
        temporal_contract[
            "structure_output"
        ][
            "sha256"
        ],
    ),
    (
        class_feature_file,
        class_contract[
            "featurewise_output"
        ][
            "sha256"
        ],
    ),
    (
        class_structure_file,
        class_contract[
            "structure_output"
        ][
            "sha256"
        ],
    ),
    (
        sensitivity_feature_file,
        sensitivity_contract[
            "featurewise_output"
        ][
            "sha256"
        ],
    ),
    (
        sensitivity_structure_file,
        sensitivity_contract[
            "structure_output"
        ][
            "sha256"
        ],
    ),
    (
        null_comparison_file,
        null_contract[
            "comparison_output"
        ][
            "sha256"
        ],
    ),
]


for path, expected_hash in hash_checks:

    if (
        sha256_file(
            path
        )
        !=
        expected_hash
    ):

        raise RuntimeError(
            f"Frozen input changed: {path}"
        )


print(
    "Frozen input artifact hashes:",
    "PASS"
)


# ============================================================
# LOAD INPUT TABLES
# ============================================================

overall_feature = pd.read_csv(
    overall_feature_file,
    keep_default_na=False,
)


overall_structure = pd.read_csv(
    overall_structure_file,
    keep_default_na=False,
)


temporal_feature = pd.read_csv(
    temporal_feature_file,
    keep_default_na=False,
)


temporal_structure = pd.read_csv(
    temporal_structure_file,
    keep_default_na=False,
)


class_feature = pd.read_csv(
    class_feature_file,
    keep_default_na=False,
)


class_structure = pd.read_csv(
    class_structure_file,
    keep_default_na=False,
)


sensitivity_feature = pd.read_csv(
    sensitivity_feature_file,
    keep_default_na=False,
)


sensitivity_structure = pd.read_csv(
    sensitivity_structure_file,
    keep_default_na=False,
)


null_comparison = pd.read_csv(
    null_comparison_file,
    keep_default_na=False,
)


# ============================================================
# EXPECTED INPUT INVENTORY
# ============================================================

if len(
    overall_feature
) != 75:

    raise RuntimeError(
        "Overall featurewise row-count mismatch."
    )


if len(
    temporal_feature
) != 225:

    raise RuntimeError(
        "Temporal featurewise row-count mismatch."
    )


if len(
    class_feature
) != 150:

    raise RuntimeError(
        "Class featurewise row-count mismatch."
    )


if len(
    sensitivity_feature
) != 900:

    raise RuntimeError(
        "Sensitivity featurewise row-count mismatch."
    )


if len(
    overall_structure
) != 24:

    raise RuntimeError(
        "Overall structure row-count mismatch."
    )


if len(
    temporal_structure
) != 72:

    raise RuntimeError(
        "Temporal structure row-count mismatch."
    )


if len(
    class_structure
) != 48:

    raise RuntimeError(
        "Class structure row-count mismatch."
    )


if len(
    sensitivity_structure
) != 288:

    raise RuntimeError(
        "Sensitivity structure row-count mismatch."
    )


if len(
    null_comparison
) != 144:

    raise RuntimeError(
        "Null-comparison row-count mismatch."
    )


print(
    "Input table inventories:",
    "PASS"
)


# ============================================================
# INDIVIDUAL-FEATURE METRIC LOOKUP
# ============================================================

individual_metric_lookup = {}


# Overall
for _, row in overall_structure.iterrows():

    metric = str(
        row[
            "metric"
        ]
    )


    if metric not in GROUP_METRICS:

        continue


    key = (
        "PRIMARY_FULL_POPULATION",
        str(
            row[
                "candidate_id"
            ]
        ),
        "OVERALL_POOLED",
        "OVERALL",
        metric,
    )


    individual_metric_lookup[
        key
    ] = float(
        row[
            "metric_value"
        ]
    )


# Temporal
for _, row in temporal_structure.iterrows():

    metric = str(
        row[
            "metric"
        ]
    )


    if metric not in GROUP_METRICS:

        continue


    key = (
        "PRIMARY_FULL_POPULATION",
        str(
            row[
                "candidate_id"
            ]
        ),
        "TEMPORAL_BLOCK",
        str(
            row[
                "cycle25_block"
            ]
        ),
        metric,
    )


    individual_metric_lookup[
        key
    ] = float(
        row[
            "metric_value"
        ]
    )


# Class
for _, row in class_structure.iterrows():

    metric = str(
        row[
            "metric"
        ]
    )


    if metric not in GROUP_METRICS:

        continue


    key = (
        "PRIMARY_FULL_POPULATION",
        str(
            row[
                "candidate_id"
            ]
        ),
        "CLASS_CONDITIONAL",
        str(
            row[
                "stratum"
            ]
        ),
        metric,
    )


    individual_metric_lookup[
        key
    ] = float(
        row[
            "metric_value"
        ]
    )


# Support / reweight sensitivity
for _, row in sensitivity_structure.iterrows():

    metric = str(
        row[
            "metric"
        ]
    )


    if metric not in GROUP_METRICS:

        continue


    key = (
        str(
            row[
                "sensitivity_variant"
            ]
        ),
        str(
            row[
                "candidate_id"
            ]
        ),
        str(
            row[
                "analysis_scope"
            ]
        ),
        str(
            row[
                "scope_key"
            ]
        ),
        metric,
    )


    individual_metric_lookup[
        key
    ] = float(
        row[
            "sensitivity_value"
        ]
    )


# ============================================================
# INDIVIDUAL-FEATURE NULL EXCEEDANCE LOOKUP
# ============================================================

null_lookup = {}


for _, row in null_comparison.iterrows():

    analysis_scope = str(
        row[
            "analysis_scope"
        ]
    )


    if analysis_scope == "OVERALL_POOLED":

        scope_key = "OVERALL"


    elif analysis_scope == "TEMPORAL_BLOCK":

        scope_key = str(
            row[
                "cycle25_block"
            ]
        )


    elif analysis_scope == "CLASS_CONDITIONAL":

        scope_key = str(
            row[
                "observed_stratum"
            ]
        )


    else:

        raise RuntimeError(
            "Unexpected null comparison scope: "
            f"{analysis_scope}"
        )


    metric = str(
        row[
            "metric"
        ]
    )


    if metric not in GROUP_METRICS:

        continue


    key = (
        str(
            row[
                "candidate_id"
            ]
        ),
        analysis_scope,
        scope_key,
        metric,
    )


    null_lookup[
        key
    ] = parse_bool(
        row[
            "null_exceedance"
        ]
    )


# ============================================================
# OUTPUT CONTAINERS
# ============================================================

profile_rows = []

structure_rows = []


# ============================================================
# PROCESS ONE COMPARISON
# ============================================================

def process_comparison(
    *,
    candidate_id,
    model_family,
    output_space,
    analysis_variant,
    analysis_scope,
    scope_key,
    reference_population,
    transfer_population,
    feature_frame,
    reference_column,
    transfer_column,
):

    frame = feature_frame.copy()


    if len(
        frame
    ) != EXPECTED_FEATURES:

        raise RuntimeError(
            f"{candidate_id} {analysis_scope} "
            f"{scope_key} {analysis_variant}: "
            "expected 25 feature rows."
        )


    if set(
        frame[
            "feature"
        ].astype(str)
    ) != set(
        FEATURE_ORDER
    ):

        raise RuntimeError(
            f"{candidate_id} {analysis_scope} "
            f"{scope_key}: feature set mismatch."
        )


    frame[
        "feature"
    ] = frame[
        "feature"
    ].astype(str)


    frame = (
        frame
        .set_index(
            "feature"
        )
        .loc[
            FEATURE_ORDER
        ]
        .reset_index()
    )


    reference_values = frame[
        reference_column
    ].to_numpy(
        dtype=np.float64
    )


    transfer_values = frame[
        transfer_column
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        reference_values
    ).all():

        raise RuntimeError(
            "Non-finite reference mean-|SHAP|."
        )


    if not np.isfinite(
        transfer_values
    ).all():

        raise RuntimeError(
            "Non-finite transfer mean-|SHAP|."
        )


    if (
        reference_values
        <
        -EPSILON
    ).any():

        raise RuntimeError(
            "Negative reference mean-|SHAP|."
        )


    if (
        transfer_values
        <
        -EPSILON
    ).any():

        raise RuntimeError(
            "Negative transfer mean-|SHAP|."
        )


    # --------------------------------------------------------
    # EXACT FROZEN GROUP AGGREGATION:
    # SUM member-feature mean absolute SHAP
    # --------------------------------------------------------

    reference_group_values = []

    transfer_group_values = []


    for group_id in GROUP_ORDER:

        members = GROUP_METADATA[
            group_id
        ][
            "members"
        ]


        subset = frame[
            frame[
                "feature"
            ].isin(
                members
            )
        ]


        if len(
            subset
        ) != len(
            members
        ):

            raise RuntimeError(
                f"{group_id}: member coverage failed."
            )


        reference_group_values.append(
            float(
                subset[
                    reference_column
                ].sum()
            )
        )


        transfer_group_values.append(
            float(
                subset[
                    transfer_column
                ].sum()
            )
        )


    reference_group_values = np.asarray(
        reference_group_values,
        dtype=np.float64,
    )


    transfer_group_values = np.asarray(
        transfer_group_values,
        dtype=np.float64,
    )


    if reference_group_values.shape != (
        EXPECTED_GROUPS,
    ):

        raise RuntimeError(
            "Reference group vector shape mismatch."
        )


    if transfer_group_values.shape != (
        EXPECTED_GROUPS,
    ):

        raise RuntimeError(
            "Transfer group vector shape mismatch."
        )


    # --------------------------------------------------------
    # CONSERVATION CHECK
    #
    # Sum of all group magnitudes must equal
    # sum of all individual-feature magnitudes.
    # --------------------------------------------------------

    if not np.isclose(
        float(
            np.sum(
                reference_group_values
            )
        ),
        float(
            np.sum(
                reference_values
            )
        ),
        atol=1e-12,
        rtol=1e-10,
    ):

        raise RuntimeError(
            "Reference group aggregation "
            "conservation check failed."
        )


    if not np.isclose(
        float(
            np.sum(
                transfer_group_values
            )
        ),
        float(
            np.sum(
                transfer_values
            )
        ),
        atol=1e-12,
        rtol=1e-10,
    ):

        raise RuntimeError(
            "Transfer group aggregation "
            "conservation check failed."
        )


    reference_ranks = deterministic_ranks(
        reference_group_values
    )


    transfer_ranks = deterministic_ranks(
        transfer_group_values
    )


    reference_total = float(
        np.sum(
            reference_group_values
        )
    )


    transfer_total = float(
        np.sum(
            transfer_group_values
        )
    )


    # --------------------------------------------------------
    # GROUP PROFILE OUTPUT
    # --------------------------------------------------------

    for group_index, group_id in enumerate(
        GROUP_ORDER
    ):

        metadata = GROUP_METADATA[
            group_id
        ]


        reference_value = float(
            reference_group_values[
                group_index
            ]
        )


        transfer_value = float(
            transfer_group_values[
                group_index
            ]
        )


        profile_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "model_family":
                    model_family,

                "output_space":
                    output_space,

                "analysis_variant":
                    analysis_variant,

                "analysis_scope":
                    analysis_scope,

                "scope_key":
                    scope_key,

                "reference_population":
                    reference_population,

                "transfer_population":
                    transfer_population,

                "group_order":
                    group_index + 1,

                "group_id":
                    group_id,

                "group_size":
                    metadata[
                        "group_size"
                    ],

                "group_members":
                    metadata[
                        "group_members"
                    ],

                "is_singleton":
                    metadata[
                        "is_singleton"
                    ],

                "reference_group_mean_absolute_shap":
                    reference_value,

                "transfer_group_mean_absolute_shap":
                    transfer_value,

                "delta_group_mean_absolute_shap":
                    (
                        transfer_value
                        -
                        reference_value
                    ),

                "absolute_delta_group_mean_absolute_shap":
                    abs(
                        transfer_value
                        -
                        reference_value
                    ),

                "reference_group_share":
                    (
                        reference_value
                        /
                        reference_total
                        if
                        reference_total
                        >
                        EPSILON
                        else
                        0.0
                    ),

                "transfer_group_share":
                    (
                        transfer_value
                        /
                        transfer_total
                        if
                        transfer_total
                        >
                        EPSILON
                        else
                        0.0
                    ),

                "delta_group_share":
                    (
                        (
                            transfer_value
                            /
                            transfer_total
                            if
                            transfer_total
                            >
                            EPSILON
                            else
                            0.0
                        )
                        -
                        (
                            reference_value
                            /
                            reference_total
                            if
                            reference_total
                            >
                            EPSILON
                            else
                            0.0
                        )
                    ),

                "reference_group_rank":
                    int(
                        reference_ranks[
                            group_index
                        ]
                    ),

                "transfer_group_rank":
                    int(
                        transfer_ranks[
                            group_index
                        ]
                    ),

                "rank_change_transfer_minus_reference":
                    int(
                        transfer_ranks[
                            group_index
                        ]
                        -
                        reference_ranks[
                            group_index
                        ]
                    ),

                "aggregation_rule":
                    (
                        "SUM_MEMBER_FEATURE_"
                        "MEAN_ABSOLUTE_SHAP"
                    ),

                "group_level_null_applied":
                    False,

                "final_drift_conclusion":
                    "NOT_MADE",
            }
        )


    # --------------------------------------------------------
    # GROUP STRUCTURE METRICS
    # --------------------------------------------------------

    metrics = calculate_group_structure_metrics(
        reference_group_values,
        transfer_group_values,
    )


    for metric_name in GROUP_METRICS:

        metric_value = float(
            metrics[
                metric_name
            ]
        )


        individual_key = (
            analysis_variant,
            candidate_id,
            analysis_scope,
            scope_key,
            metric_name,
        )


        if (
            individual_key
            not in
            individual_metric_lookup
        ):

            raise RuntimeError(
                "Missing individual-feature metric: "
                f"{individual_key}"
            )


        individual_value = float(
            individual_metric_lookup[
                individual_key
            ]
        )


        if (
            analysis_variant
            ==
            "PRIMARY_FULL_POPULATION"
        ):

            null_key = (
                candidate_id,
                analysis_scope,
                scope_key,
                metric_name,
            )


            if (
                null_key
                not in
                null_lookup
            ):

                raise RuntimeError(
                    "Missing individual-feature "
                    f"null screen: {null_key}"
                )


            individual_null_exceedance = (
                null_lookup[
                    null_key
                ]
            )


        else:

            individual_null_exceedance = None


        structure_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "model_family":
                    model_family,

                "output_space":
                    output_space,

                "analysis_variant":
                    analysis_variant,

                "analysis_scope":
                    analysis_scope,

                "scope_key":
                    scope_key,

                "reference_population":
                    reference_population,

                "transfer_population":
                    transfer_population,

                "group_count":
                    EXPECTED_GROUPS,

                "metric":
                    metric_name,

                "individual_feature_metric_value":
                    individual_value,

                "group_level_metric_value":
                    metric_value,

                "group_minus_individual_metric_value":
                    (
                        metric_value
                        -
                        individual_value
                    ),

                "individual_feature_null_exceedance":
                    individual_null_exceedance,

                "group_level_null_threshold_available":
                    False,

                "group_level_null_applied":
                    False,

                "cross_dimension_metric_comparison":
                    "DESCRIPTIVE_ONLY",

                "final_group_robustness_conclusion":
                    "NOT_MADE",

                "final_drift_conclusion":
                    "NOT_MADE",
            }
        )


    print(
        f"{analysis_variant} | "
        f"{candidate_id} | "
        f"{analysis_scope} | "
        f"{scope_key}"
    )


    print(
        "  Group Spearman:",
        f"{metrics['spearman_rho']:.6f}",
        "| Kendall:",
        f"{metrics['kendall_tau_b']:.6f}",
        "| Top5:",
        f"{metrics['top5_overlap']:.6f}",
        "| L1:",
        f"{metrics['normalized_l1_mean_abs']:.6f}",
        "| CosDist:",
        f"{metrics['cosine_distance_mean_abs']:.6f}",
    )


# ============================================================
# PRIMARY — OVERALL
# ============================================================

for candidate_id in MODEL_ORDER:

    feature_subset = overall_feature[
        overall_feature[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ].copy()


    first = feature_subset.iloc[
        0
    ]


    process_comparison(
        candidate_id=candidate_id,
        model_family=str(
            first[
                "model_family"
            ]
        ),
        output_space=str(
            first[
                "output_space"
            ]
        ),
        analysis_variant="PRIMARY_FULL_POPULATION",
        analysis_scope="OVERALL_POOLED",
        scope_key="OVERALL",
        reference_population="CYCLE24_HOLDOUT",
        transfer_population="CYCLE25_POOLED",
        feature_frame=feature_subset,
        reference_column="cycle24_mean_absolute_shap",
        transfer_column="cycle25_mean_absolute_shap",
    )


# ============================================================
# PRIMARY — TEMPORAL B1/B2/B3
# ============================================================

for candidate_id in MODEL_ORDER:

    for block_name in [
        "B1",
        "B2",
        "B3",
    ]:

        feature_subset = temporal_feature[
            (
                temporal_feature[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            )
            &
            (
                temporal_feature[
                    "cycle25_block"
                ].astype(str)
                ==
                block_name
            )
        ].copy()


        first = feature_subset.iloc[
            0
        ]


        process_comparison(
            candidate_id=candidate_id,
            model_family=str(
                first[
                    "model_family"
                ]
            ),
            output_space=str(
                first[
                    "output_space"
                ]
            ),
            analysis_variant="PRIMARY_FULL_POPULATION",
            analysis_scope="TEMPORAL_BLOCK",
            scope_key=block_name,
            reference_population="CYCLE24_HOLDOUT",
            transfer_population=(
                f"CYCLE25_{block_name}"
            ),
            feature_frame=feature_subset,
            reference_column="cycle24_mean_absolute_shap",
            transfer_column=(
                "cycle25_block_mean_absolute_shap"
            ),
        )


# ============================================================
# PRIMARY — CLASS-CONDITIONAL
# ============================================================

for candidate_id in MODEL_ORDER:

    for stratum_name in [
        "TARGET_0",
        "TARGET_1",
    ]:

        feature_subset = class_feature[
            (
                class_feature[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            )
            &
            (
                class_feature[
                    "stratum"
                ].astype(str)
                ==
                stratum_name
            )
        ].copy()


        first = feature_subset.iloc[
            0
        ]


        process_comparison(
            candidate_id=candidate_id,
            model_family=str(
                first[
                    "model_family"
                ]
            ),
            output_space=str(
                first[
                    "output_space"
                ]
            ),
            analysis_variant="PRIMARY_FULL_POPULATION",
            analysis_scope="CLASS_CONDITIONAL",
            scope_key=stratum_name,
            reference_population=(
                f"CYCLE24_HOLDOUT_{stratum_name}"
            ),
            transfer_population=(
                f"CYCLE25_POOLED_{stratum_name}"
            ),
            feature_frame=feature_subset,
            reference_column="cycle24_mean_absolute_shap",
            transfer_column="cycle25_mean_absolute_shap",
        )


# ============================================================
# SUPPORT / REWEIGHTING SENSITIVITY
# ============================================================

for candidate_id in MODEL_ORDER:

    for analysis_scope, scope_keys in [
        (
            "OVERALL_POOLED",
            [
                "OVERALL",
            ],
        ),
        (
            "TEMPORAL_BLOCK",
            [
                "B1",
                "B2",
                "B3",
            ],
        ),
        (
            "CLASS_CONDITIONAL",
            [
                "TARGET_0",
                "TARGET_1",
            ],
        ),
    ]:

        for scope_key in scope_keys:

            for sensitivity_variant in [
                "COMMON_SUPPORT_UNWEIGHTED",
                "COMMON_SUPPORT_REWEIGHTED",
            ]:

                feature_subset = sensitivity_feature[
                    (
                        sensitivity_feature[
                            "candidate_id"
                        ].astype(str)
                        ==
                        candidate_id
                    )
                    &
                    (
                        sensitivity_feature[
                            "analysis_scope"
                        ].astype(str)
                        ==
                        analysis_scope
                    )
                    &
                    (
                        sensitivity_feature[
                            "scope_key"
                        ].astype(str)
                        ==
                        scope_key
                    )
                    &
                    (
                        sensitivity_feature[
                            "sensitivity_variant"
                        ].astype(str)
                        ==
                        sensitivity_variant
                    )
                ].copy()


                if len(
                    feature_subset
                ) == 0:

                    raise RuntimeError(
                        "Missing sensitivity feature "
                        f"subset: {candidate_id}, "
                        f"{analysis_scope}, "
                        f"{scope_key}, "
                        f"{sensitivity_variant}"
                    )


                first = feature_subset.iloc[
                    0
                ]


                process_comparison(
                    candidate_id=candidate_id,
                    model_family=str(
                        first[
                            "model_family"
                        ]
                    ),
                    output_space=str(
                        first[
                            "output_space"
                        ]
                    ),
                    analysis_variant=sensitivity_variant,
                    analysis_scope=analysis_scope,
                    scope_key=scope_key,
                    reference_population=str(
                        first[
                            "reference_population"
                        ]
                    ),
                    transfer_population=str(
                        first[
                            "transfer_population"
                        ]
                    ),
                    feature_frame=feature_subset,
                    reference_column=(
                        "reference_mean_absolute_shap"
                    ),
                    transfer_column=(
                        "transfer_mean_absolute_shap"
                    ),
                )


# ============================================================
# BUILD OUTPUT TABLES
# ============================================================

profiles = pd.DataFrame(
    profile_rows
)


structure = pd.DataFrame(
    structure_rows
)


if len(
    profiles
) != EXPECTED_PROFILE_ROWS:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_PROFILE_ROWS} group-profile rows, "
        f"got {len(profiles)}."
    )


if len(
    structure
) != EXPECTED_STRUCTURE_ROWS:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_STRUCTURE_ROWS} group-structure rows, "
        f"got {len(structure)}."
    )


# ============================================================
# DUPLICATE / INVENTORY CHECKS
# ============================================================

if profiles[
    [
        "candidate_id",
        "analysis_variant",
        "analysis_scope",
        "scope_key",
        "group_id",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate group-profile keys detected."
    )


if structure[
    [
        "candidate_id",
        "analysis_variant",
        "analysis_scope",
        "scope_key",
        "metric",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate group-structure keys detected."
    )


metric_counts = (
    structure
    .groupby(
        [
            "candidate_id",
            "analysis_variant",
            "analysis_scope",
            "scope_key",
        ]
    )[
        "metric"
    ]
    .nunique()
)


if not (
    metric_counts
    ==
    len(
        GROUP_METRICS
    )
).all():

    raise RuntimeError(
        "Every group comparison must "
        "contain exactly 5 group metrics."
    )


group_counts = (
    profiles
    .groupby(
        [
            "candidate_id",
            "analysis_variant",
            "analysis_scope",
            "scope_key",
        ]
    )[
        "group_id"
    ]
    .nunique()
)


if not (
    group_counts
    ==
    EXPECTED_GROUPS
).all():

    raise RuntimeError(
        "Every comparison must contain "
        "exactly 10 frozen groups."
    )


print()

print(
    "Correlated-group profile inventory:",
    "PASS"
)

print(
    "Correlated-group structure inventory:",
    "PASS"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

PROFILE_OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


STRUCTURE_OUTPUT_FILE.parent.mkdir(
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

profile_temp = PROFILE_OUTPUT_FILE.with_name(
    PROFILE_OUTPUT_FILE.stem
    +
    ".tmp.csv"
)


structure_temp = STRUCTURE_OUTPUT_FILE.with_name(
    STRUCTURE_OUTPUT_FILE.stem
    +
    ".tmp.csv"
)


for path in [
    profile_temp,
    structure_temp,
]:

    if path.exists():

        path.unlink()


profiles.to_csv(
    profile_temp,
    index=False,
)


structure.to_csv(
    structure_temp,
    index=False,
)


profile_rt = pd.read_csv(
    profile_temp,
    keep_default_na=False,
)


structure_rt = pd.read_csv(
    structure_temp,
    keep_default_na=False,
)


if len(
    profile_rt
) != EXPECTED_PROFILE_ROWS:

    raise RuntimeError(
        "Group-profile CSV round-trip failed."
    )


if len(
    structure_rt
) != EXPECTED_STRUCTURE_ROWS:

    raise RuntimeError(
        "Group-structure CSV round-trip failed."
    )


profile_temp.replace(
    PROFILE_OUTPUT_FILE
)


structure_temp.replace(
    STRUCTURE_OUTPUT_FILE
)


profile_hash = sha256_file(
    PROFILE_OUTPUT_FILE
)


structure_hash = sha256_file(
    STRUCTURE_OUTPUT_FILE
)


print(
    "Group-profile CSV round-trip:",
    "PASS"
)

print(
    "Group-structure CSV round-trip:",
    "PASS"
)


# ============================================================
# FINAL CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Correlated Feature-Group "
            "Cross-Cycle SHAP Robustness"
        ),

    "version":
        "1.0",

    "status":
        (
            "FROZEN_CORRELATED_GROUP_"
            "SHAP_ROBUSTNESS"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_8_2",

    "frozen_group_source":
        {
            "contract_file":
                GROUP_CONTRACT_FILE.as_posix(),

            "contract_sha256":
                group_contract_hash,

            "required_status":
                "FROZEN_CORRELATED_FEATURE_GROUPS",

            "manifest_file":
                GROUP_MANIFEST_FILE.as_posix(),

            "manifest_sha256":
                sha256_file(
                    GROUP_MANIFEST_FILE
                ),

            "source_population":
                "CYCLE24_TRAIN_ONLY",

            "correlation_method":
                "SPEARMAN",

            "absolute_rho_threshold":
                0.80,

            "grouping_method":
                "CONNECTED_COMPONENTS",

            "transitive_grouping":
                True,

            "groups":
                EXPECTED_GROUPS,

            "features":
                EXPECTED_FEATURES,
        },

    "aggregation_rule":
        {
            "group_level_mean_absolute_shap":
                (
                    "SUM_MEMBER_FEATURE_"
                    "MEAN_ABSOLUTE_SHAP"
                ),

            "matches_frozen_chapter9_contract":
                True,

            "signed_group_aggregation_defined_by_frozen_contract":
                False,

            "signed_group_metrics_computed":
                False,

            "top10_overlap_computed":
                False,

            "top10_omission_reason":
                (
                    "There are exactly 10 frozen groups, "
                    "so top-10 overlap would be "
                    "identically 1 and non-informative."
                ),
        },

    "group_structure_metrics":
        GROUP_METRICS,

    "analysis_variants":
        [
            "PRIMARY_FULL_POPULATION",
            "COMMON_SUPPORT_UNWEIGHTED",
            "COMMON_SUPPORT_REWEIGHTED",
        ],

    "analysis_scopes":
        [
            "OVERALL_POOLED",
            "TEMPORAL_BLOCK_B1",
            "TEMPORAL_BLOCK_B2",
            "TEMPORAL_BLOCK_B3",
            "CLASS_CONDITIONAL_TARGET_0",
            "CLASS_CONDITIONAL_TARGET_1",
        ],

    "profile_output":
        {
            "file":
                PROFILE_OUTPUT_FILE.as_posix(),

            "sha256":
                profile_hash,

            "rows":
                EXPECTED_PROFILE_ROWS,
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

    "null_handling":
        {
            "individual_feature_null_screen_retained":
                True,

            "group_level_cycle24_null_available":
                False,

            "group_level_null_threshold_applied":
                False,

            "reason":
                (
                    "Chapter 9 froze the null baseline "
                    "for 25 individual features, not "
                    "for the 10 aggregated correlated "
                    "feature groups."
                ),
        },

    "scientific_guardrails":
        {
            "cycle25_used_to_define_groups":
                False,

            "groups_recomputed":
                False,

            "correlation_threshold_changed":
                False,

            "group_membership_changed":
                False,

            "individual_feature_analysis_retained":
                True,

            "frozen_group_aggregation_rule_used":
                True,

            "signed_group_rule_invented":
                False,

            "group_level_null_invented":
                False,

            "common_support_group_robustness_checked":
                True,

            "reweighting_group_robustness_checked":
                True,

            "featurewise_inference_performed":
                False,

            "fdr_applied":
                False,

            "final_drift_conclusion_made":
                False,

            "final_stability_conclusion_made":
                False,

            "correlation_is_causality":
                False,

            "shap_is_causal":
                False,
        },

    "interpretation_rule":
        (
            "Correlated-group analysis is a robustness "
            "check for individual-feature attribution "
            "structure. Member-feature mean absolute SHAP "
            "magnitudes are summed using the frozen "
            "Cycle24-TRAIN grouping. Group-level results "
            "must not be treated as causal solar-physics "
            "factors or as statistically significant "
            "drift because no group-level Cycle24 null "
            "distribution was frozen. CFG_01 contains "
            "13 features and its magnitude reflects the "
            "predefined summation rule as well as group size."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_9_"
            "FEATUREWISE_INFERENCE_AND_FDR"
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
    "FROZEN_CORRELATED_GROUP_SHAP_ROBUSTNESS"
):

    raise RuntimeError(
        "10.8.2 contract round-trip failed."
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
    "MODEL CHAPTER 10.8.2 — COMPLETE"
)

print("=" * 78)


print(
    "Frozen features:",
    EXPECTED_FEATURES
)

print(
    "Frozen correlated groups:",
    EXPECTED_GROUPS
)

print(
    "Primary comparison units:",
    (
        EXPECTED_MODELS
        *
        PRIMARY_SCOPES_PER_MODEL
    )
)

print(
    "Sensitivity comparison units:",
    (
        EXPECTED_MODELS
        *
        SENSITIVITY_SCOPES_PER_MODEL
        *
        SENSITIVITY_VARIANTS
    )
)


print()

print(
    "Group profile rows:",
    len(
        profiles
    )
)

print(
    "Group structure rows:",
    len(
        structure
    )
)


print()

print(
    "Aggregation:",
    "SUM member-feature mean-|SHAP|"
)

print(
    "Signed group metrics:",
    False
)

print(
    "Top-10 group overlap:",
    False
)

print(
    "Group-level null applied:",
    False
)


print()

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
    "FROZEN_CORRELATED_GROUP_SHAP_ROBUSTNESS"
)


print()

print(
    "Group profiles:",
    PROFILE_OUTPUT_FILE
)

print(
    "Group structure:",
    STRUCTURE_OUTPUT_FILE
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
        "Chapter 10.9 — "
        "Featurewise Inference + BH-FDR"
    )
)