# ============================================================
# MODEL CHAPTER 10.3.2B
# OVERALL CYCLE24 vs CYCLE25 SHAP POINT ESTIMATES
#
# Comparison:
#
#   Cycle24 HOLDOUT
#       vs
#   Cycle25 POOLED
#
# Within same frozen model only.
#
# This script reproduces EXACTLY the structure-metric
# definitions frozen in Chapter 9:
#
#   - Spearman rho
#   - Kendall tau-b
#   - Top-5 overlap
#   - Top-10 overlap
#   - normalized L1 on mean-|SHAP|
#   - cosine distance on mean-|SHAP|
#   - signed cosine similarity
#   - sign agreement
#
# IMPORTANT:
#   - NO null thresholds applied here
#   - NO significance claim
#   - NO drift/stability conclusion
#   - NO model retraining
#   - NO SHAP recomputation
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
# FROZEN CONSTANTS
# ============================================================

EPSILON = 1e-12

EXPECTED_MODELS = 3

EXPECTED_FEATURES = 25

EXPECTED_C24_ROWS = 13072

EXPECTED_C25_ROWS = 79361

EXPECTED_STRUCTURE_ROWS = (
    EXPECTED_MODELS
    *
    8
)

EXPECTED_FEATUREWISE_ROWS = (
    EXPECTED_MODELS
    *
    EXPECTED_FEATURES
)


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


METRIC_DIRECTION = {

    "spearman_rho":
        "HIGHER_IS_MORE_SIMILAR",

    "kendall_tau_b":
        "HIGHER_IS_MORE_SIMILAR",

    "top5_overlap":
        "HIGHER_IS_MORE_SIMILAR",

    "top10_overlap":
        "HIGHER_IS_MORE_SIMILAR",

    "normalized_l1_mean_abs":
        "LOWER_IS_MORE_SIMILAR",

    "cosine_distance_mean_abs":
        "LOWER_IS_MORE_SIMILAR",

    "signed_cosine_similarity":
        "HIGHER_IS_MORE_SIMILAR",

    "sign_agreement":
        "HIGHER_IS_MORE_SIMILAR",
}


# ============================================================
# INPUTS
# ============================================================

INPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "overall_cross_cycle_shap_inputs_contract.json"
)


INPUT_REGISTRY_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "overall_cross_cycle_shap_input_registry.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

STRUCTURE_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "overall_cross_cycle_shap_structure_point_estimates.csv"
)


FEATUREWISE_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "overall_cross_cycle_shap_featurewise_point_estimates.csv"
)


OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "overall_cross_cycle_shap_point_estimates_contract.json"
)


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

        return json.load(
            file
        )


def verify_exists(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


# ============================================================
# EXACT CHAPTER-9 VECTOR METRICS
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    norm_a = float(
        np.linalg.norm(
            a
        )
    )

    norm_b = float(
        np.linalg.norm(
            b
        )
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
        np.dot(
            a,
            b,
        )
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


def top_k_overlap(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
) -> float:

    feature_index = np.arange(
        len(
            a
        ),
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
        float(
            k
        )
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
            np.abs(
                a
            )
        )
        +
        np.sum(
            np.abs(
                b
            )
        )
    )


    if denominator <= EPSILON:

        return 0.0


    return (
        numerator
        /
        denominator
    )


def sign_agreement(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(
        np.mean(
            np.sign(
                a
            )
            ==
            np.sign(
                b
            )
        )
    )


def calculate_structure_metrics(
    mean_abs_a: np.ndarray,
    mean_abs_b: np.ndarray,
    mean_signed_a: np.ndarray,
    mean_signed_b: np.ndarray,
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


    if not np.isfinite(
        spearman_value
    ):

        raise RuntimeError(
            "Non-finite Spearman rho."
        )


    if not np.isfinite(
        kendall_value
    ):

        raise RuntimeError(
            "Non-finite Kendall tau-b."
        )


    mean_abs_cosine_similarity = (
        cosine_similarity(
            mean_abs_a,
            mean_abs_b,
        )
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
                mean_abs_cosine_similarity
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


# ============================================================
# DETERMINISTIC RANKING
#
# Same deterministic concept used for top-k:
# descending mean-|SHAP|,
# feature order as tie-break.
# ============================================================

def deterministic_ranks(
    values: np.ndarray,
) -> np.ndarray:

    feature_index = np.arange(
        len(
            values
        ),
        dtype=np.int64,
    )


    order = np.lexsort(
        (
            feature_index,
            -values,
        )
    )


    ranks = np.empty(
        len(
            values
        ),
        dtype=np.int64,
    )


    ranks[
        order
    ] = np.arange(
        1,
        len(
            values
        )
        +
        1,
        dtype=np.int64,
    )


    return ranks


# ============================================================
# START
# ============================================================

print(
    "=" * 78
)

print(
    "MODEL CHAPTER 10.3.2B — "
    "OVERALL CROSS-CYCLE SHAP POINT ESTIMATES"
)

print(
    "=" * 78
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

for path in [
    INPUT_CONTRACT_FILE,
    INPUT_REGISTRY_FILE,
]:

    verify_exists(
        path
    )


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
            "10.3.2B point estimates."
        )


# ============================================================
# VERIFY 10.3.1 CONTRACT
# ============================================================

input_contract = load_json(
    INPUT_CONTRACT_FILE
)


if (
    input_contract.get(
        "status"
    )
    !=
    "FROZEN_OVERALL_CROSS_CYCLE_SHAP_INPUTS"
):

    raise RuntimeError(
        "Chapter 10.3.1 comparison inputs "
        "are not frozen."
    )


input_contract_hash = sha256_file(
    INPUT_CONTRACT_FILE
)


if (
    input_contract[
        "scientific_guardrails"
    ][
        "overall_drift_metrics_computed"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected previous overall-metric state."
    )


print(
    "Chapter 10.3.1 frozen contract:",
    "PASS"
)


# ============================================================
# VERIFY INPUT REGISTRY
# ============================================================

input_registry_hash = sha256_file(
    INPUT_REGISTRY_FILE
)


if (
    input_registry_hash
    !=
    input_contract[
        "input_registry"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Frozen 10.3.1 comparison registry "
        "hash mismatch."
    )


registry = pd.read_csv(
    INPUT_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    registry
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 matched model pairs."
    )


if (
    registry[
        "candidate_id"
    ].astype(str).tolist()
    !=
    MODEL_ORDER
):

    raise RuntimeError(
        "Frozen model ordering changed."
    )


print(
    "Frozen comparison registry:",
    "PASS"
)


# ============================================================
# VERIFY FEATURE ORDER CONTRACT
# ============================================================

if (
    input_contract[
        "feature_order"
    ]
    !=
    FEATURES
):

    raise RuntimeError(
        "Frozen feature order changed."
    )


print(
    "Frozen 25-feature order:",
    "PASS"
)


# ============================================================
# RESULTS
# ============================================================

structure_rows = []

featurewise_rows = []


# ============================================================
# EACH SAME-MODEL CROSS-CYCLE COMPARISON
# ============================================================

for candidate_id in MODEL_ORDER:

    print()

    print(
        "ANALYZE:",
        candidate_id
    )


    registry_match = registry[
        registry[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(
        registry_match
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: registry lookup failed."
        )


    row = registry_match.iloc[
        0
    ]


    # --------------------------------------------------------
    # FROZEN COMPARISON RULES
    # --------------------------------------------------------

    required_true_columns = [
        "same_frozen_model",
        "same_explainer",
        "same_input_space",
        "same_output_space",
        "same_frozen_background",
        "same_feature_order",
        "cross_cycle_comparison_allowed",
    ]


    for column in required_true_columns:

        value = str(
            row[
                column
            ]
        ).strip().lower()


        if value not in {
            "true",
            "1",
        }:

            raise RuntimeError(
                f"{candidate_id}: frozen comparison "
                f"guardrail failed: {column}"
            )


    cross_model_allowed = str(
        row[
            "cross_model_raw_magnitude_comparison_allowed"
        ]
    ).strip().lower()


    if cross_model_allowed not in {
        "false",
        "0",
    }:

        raise RuntimeError(
            f"{candidate_id}: cross-model raw "
            "magnitude guardrail changed."
        )


    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    c24_file = Path(
        str(
            row[
                "cycle24_shap_file"
            ]
        )
    )


    c25_file = Path(
        str(
            row[
                "cycle25_shap_file"
            ]
        )
    )


    verify_exists(
        c24_file
    )


    verify_exists(
        c25_file
    )


    # --------------------------------------------------------
    # HASHES
    # --------------------------------------------------------

    c24_hash = sha256_file(
        c24_file
    )


    c25_hash = sha256_file(
        c25_file
    )


    if (
        c24_hash
        !=
        str(
            row[
                "cycle24_shap_sha256"
            ]
        )
    ):

        raise RuntimeError(
            f"{candidate_id}: Cycle24 SHAP "
            "artifact changed."
        )


    if (
        c25_hash
        !=
        str(
            row[
                "cycle25_shap_sha256"
            ]
        )
    ):

        raise RuntimeError(
            f"{candidate_id}: Cycle25 SHAP "
            "artifact changed."
        )


    # --------------------------------------------------------
    # LOAD SHAP ONLY
    # --------------------------------------------------------

    c24_frame = pd.read_parquet(
        c24_file,
        columns=SHAP_COLUMNS,
    )


    c25_frame = pd.read_parquet(
        c25_file,
        columns=SHAP_COLUMNS,
    )


    if len(
        c24_frame
    ) != EXPECTED_C24_ROWS:

        raise RuntimeError(
            f"{candidate_id}: Cycle24 "
            "row-count mismatch."
        )


    if len(
        c25_frame
    ) != EXPECTED_C25_ROWS:

        raise RuntimeError(
            f"{candidate_id}: Cycle25 "
            "row-count mismatch."
        )


    c24_shap = c24_frame.to_numpy(
        dtype=np.float64
    )


    c25_shap = c25_frame.to_numpy(
        dtype=np.float64
    )


    if c24_shap.shape != (
        EXPECTED_C24_ROWS,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            f"{candidate_id}: Cycle24 "
            "SHAP shape mismatch."
        )


    if c25_shap.shape != (
        EXPECTED_C25_ROWS,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            f"{candidate_id}: Cycle25 "
            "SHAP shape mismatch."
        )


    if not np.isfinite(
        c24_shap
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "Cycle24 SHAP values."
        )


    if not np.isfinite(
        c25_shap
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "Cycle25 SHAP values."
        )


    # --------------------------------------------------------
    # GLOBAL ATTRIBUTION VECTORS
    # --------------------------------------------------------

    c24_mean_abs = np.mean(
        np.abs(
            c24_shap
        ),
        axis=0,
    )


    c25_mean_abs = np.mean(
        np.abs(
            c25_shap
        ),
        axis=0,
    )


    c24_mean_signed = np.mean(
        c24_shap,
        axis=0,
    )


    c25_mean_signed = np.mean(
        c25_shap,
        axis=0,
    )


    for vector_name, vector in [
        (
            "c24_mean_abs",
            c24_mean_abs,
        ),
        (
            "c25_mean_abs",
            c25_mean_abs,
        ),
        (
            "c24_mean_signed",
            c24_mean_signed,
        ),
        (
            "c25_mean_signed",
            c25_mean_signed,
        ),
    ]:

        if vector.shape != (
            EXPECTED_FEATURES,
        ):

            raise RuntimeError(
                f"{candidate_id}: {vector_name} "
                "shape mismatch."
            )


        if not np.isfinite(
            vector
        ).all():

            raise RuntimeError(
                f"{candidate_id}: {vector_name} "
                "contains non-finite values."
            )


    # --------------------------------------------------------
    # EXACT CHAPTER-9 STRUCTURE METRICS
    # --------------------------------------------------------

    metrics = calculate_structure_metrics(
        mean_abs_a=c24_mean_abs,
        mean_abs_b=c25_mean_abs,
        mean_signed_a=c24_mean_signed,
        mean_signed_b=c25_mean_signed,
    )


    if set(
        metrics.keys()
    ) != set(
        METRIC_ORDER
    ):

        raise RuntimeError(
            f"{candidate_id}: structure metric "
            "set mismatch."
        )


    for metric_name in METRIC_ORDER:

        metric_value = float(
            metrics[
                metric_name
            ]
        )


        if not np.isfinite(
            metric_value
        ):

            raise RuntimeError(
                f"{candidate_id}: non-finite "
                f"{metric_name}."
            )


        structure_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "model_family":
                    str(
                        row[
                            "model_family"
                        ]
                    ),

                "comparison":
                    (
                        "CYCLE24_HOLDOUT"
                        "_VS_"
                        "CYCLE25_POOLED"
                    ),

                "stratum":
                    "OVERALL",

                "cycle24_rows":
                    EXPECTED_C24_ROWS,

                "cycle25_rows":
                    EXPECTED_C25_ROWS,

                "metric":
                    metric_name,

                "metric_value":
                    metric_value,

                "metric_direction":
                    METRIC_DIRECTION[
                        metric_name
                    ],

                "null_threshold_applied":
                    False,

                "significance_evaluated":
                    False,

                "drift_classification":
                    "NOT_EVALUATED",
            }
        )


    # --------------------------------------------------------
    # FEATUREWISE DESCRIPTIVE POINT ESTIMATES
    # --------------------------------------------------------

    c24_ranks = deterministic_ranks(
        c24_mean_abs
    )


    c25_ranks = deterministic_ranks(
        c25_mean_abs
    )


    for feature_index, feature in enumerate(
        FEATURES
    ):

        c24_abs_value = float(
            c24_mean_abs[
                feature_index
            ]
        )


        c25_abs_value = float(
            c25_mean_abs[
                feature_index
            ]
        )


        c24_signed_value = float(
            c24_mean_signed[
                feature_index
            ]
        )


        c25_signed_value = float(
            c25_mean_signed[
                feature_index
            ]
        )


        featurewise_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "model_family":
                    str(
                        row[
                            "model_family"
                        ]
                    ),

                "comparison":
                    (
                        "CYCLE24_HOLDOUT"
                        "_VS_"
                        "CYCLE25_POOLED"
                    ),

                "stratum":
                    "OVERALL",

                "feature_index":
                    feature_index + 1,

                "feature":
                    feature,

                "output_space":
                    str(
                        row[
                            "output_space"
                        ]
                    ),

                "cycle24_mean_absolute_shap":
                    c24_abs_value,

                "cycle25_mean_absolute_shap":
                    c25_abs_value,

                "delta_mean_absolute_shap":
                    (
                        c25_abs_value
                        -
                        c24_abs_value
                    ),

                "absolute_delta_mean_absolute_shap":
                    abs(
                        c25_abs_value
                        -
                        c24_abs_value
                    ),

                "cycle24_mean_signed_shap":
                    c24_signed_value,

                "cycle25_mean_signed_shap":
                    c25_signed_value,

                "delta_mean_signed_shap":
                    (
                        c25_signed_value
                        -
                        c24_signed_value
                    ),

                "cycle24_mean_abs_rank":
                    int(
                        c24_ranks[
                            feature_index
                        ]
                    ),

                "cycle25_mean_abs_rank":
                    int(
                        c25_ranks[
                            feature_index
                        ]
                    ),

                "rank_change_cycle25_minus_cycle24":
                    int(
                        c25_ranks[
                            feature_index
                        ]
                        -
                        c24_ranks[
                            feature_index
                        ]
                    ),

                "cycle24_signed_direction":
                    int(
                        np.sign(
                            c24_signed_value
                        )
                    ),

                "cycle25_signed_direction":
                    int(
                        np.sign(
                            c25_signed_value
                        )
                    ),

                "signed_direction_agreement":
                    bool(
                        np.sign(
                            c24_signed_value
                        )
                        ==
                        np.sign(
                            c25_signed_value
                        )
                    ),

                "null_threshold_applied":
                    False,

                "featurewise_inference_performed":
                    False,

                "fdr_applied":
                    False,

                "drift_classification":
                    "NOT_EVALUATED",
            }
        )


    # --------------------------------------------------------
    # PRINT POINT ESTIMATES
    # --------------------------------------------------------

    print(
        "Cycle24 SHAP:",
        c24_shap.shape,
        "PASS"
    )

    print(
        "Cycle25 SHAP:",
        c25_shap.shape,
        "PASS"
    )


    print(
        "Spearman:",
        f"{metrics['spearman_rho']:.6f}"
    )

    print(
        "Kendall:",
        f"{metrics['kendall_tau_b']:.6f}"
    )

    print(
        "Top-5 overlap:",
        f"{metrics['top5_overlap']:.6f}"
    )

    print(
        "Top-10 overlap:",
        f"{metrics['top10_overlap']:.6f}"
    )

    print(
        "Normalized L1:",
        f"{metrics['normalized_l1_mean_abs']:.6f}"
    )

    print(
        "Cosine distance:",
        f"{metrics['cosine_distance_mean_abs']:.6f}"
    )

    print(
        "Signed cosine:",
        f"{metrics['signed_cosine_similarity']:.6f}"
    )

    print(
        "Sign agreement:",
        f"{metrics['sign_agreement']:.6f}"
    )


# ============================================================
# BUILD FINAL TABLES
# ============================================================

structure = pd.DataFrame(
    structure_rows
)


featurewise = pd.DataFrame(
    featurewise_rows
)


if len(
    structure
) != EXPECTED_STRUCTURE_ROWS:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_STRUCTURE_ROWS} structure rows, "
        f"got {len(structure)}."
    )


if len(
    featurewise
) != EXPECTED_FEATUREWISE_ROWS:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_FEATUREWISE_ROWS} featurewise rows, "
        f"got {len(featurewise)}."
    )


# ============================================================
# VERIFY ALL 8 METRICS PER MODEL
# ============================================================

for candidate_id in MODEL_ORDER:

    subset = structure[
        structure[
            "candidate_id"
        ]
        ==
        candidate_id
    ]


    if (
        subset[
            "metric"
        ].tolist()
        !=
        METRIC_ORDER
    ):

        raise RuntimeError(
            f"{candidate_id}: final metric "
            "ordering mismatch."
        )


print()

print(
    "Structure metric inventory:",
    "PASS"
)

print(
    "Featurewise descriptive inventory:",
    "PASS"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
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


for temp_path in [
    structure_temp,
    featurewise_temp,
]:

    if temp_path.exists():

        temp_path.unlink()


structure.to_csv(
    structure_temp,
    index=False,
)


featurewise.to_csv(
    featurewise_temp,
    index=False,
)


# ============================================================
# ROUND-TRIP
# ============================================================

structure_rt = pd.read_csv(
    structure_temp,
    keep_default_na=False,
)


featurewise_rt = pd.read_csv(
    featurewise_temp,
    keep_default_na=False,
)


if len(
    structure_rt
) != EXPECTED_STRUCTURE_ROWS:

    raise RuntimeError(
        "Structure point-estimate "
        "round-trip failed."
    )


if len(
    featurewise_rt
) != EXPECTED_FEATUREWISE_ROWS:

    raise RuntimeError(
        "Featurewise point-estimate "
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
    "Structure CSV round-trip:",
    "PASS"
)

print(
    "Featurewise CSV round-trip:",
    "PASS"
)


# ============================================================
# FINAL CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Overall Cross-Cycle "
            "SHAP Point Estimates"
        ),

    "version":
        "1.0",

    "status":
        (
            "FROZEN_OVERALL_CROSS_CYCLE_"
            "SHAP_POINT_ESTIMATES"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_3_2B",

    "source_comparison_inputs":
        {
            "contract_file":
                INPUT_CONTRACT_FILE.as_posix(),

            "contract_sha256":
                input_contract_hash,

            "required_status":
                (
                    "FROZEN_OVERALL_"
                    "CROSS_CYCLE_SHAP_INPUTS"
                ),

            "registry_file":
                INPUT_REGISTRY_FILE.as_posix(),

            "registry_sha256":
                input_registry_hash,
        },

    "comparison":
        {
            "reference_population":
                "CYCLE24_HOLDOUT",

            "transfer_population":
                "CYCLE25_POOLED",

            "stratum":
                "OVERALL",

            "cycle24_rows":
                EXPECTED_C24_ROWS,

            "cycle25_rows":
                EXPECTED_C25_ROWS,

            "models":
                EXPECTED_MODELS,

            "features":
                EXPECTED_FEATURES,

            "comparison_unit":
                "WITHIN_SAME_FROZEN_MODEL",
        },

    "metric_definition_source":
        {
            "source_script":
                (
                    "src/modeling/evaluation/"
                    "run_cycle24_shap_null_bootstrap.py"
                ),

            "epsilon":
                EPSILON,

            "exact_chapter9_formulas_reused":
                True,

            "spearman":
                (
                    "scipy.stats.spearmanr on "
                    "mean-|SHAP| vectors"
                ),

            "kendall":
                (
                    "scipy.stats.kendalltau "
                    "variant='b' on mean-|SHAP| vectors"
                ),

            "top_k":
                (
                    "descending mean-|SHAP| with frozen "
                    "feature-order tie break; intersection/k"
                ),

            "normalized_l1":
                (
                    "sum(abs(A-B)) / "
                    "(sum(abs(A)) + sum(abs(B)))"
                ),

            "cosine_distance":
                (
                    "1 - cosine_similarity of "
                    "mean-|SHAP| vectors"
                ),

            "signed_cosine":
                (
                    "cosine similarity of mean "
                    "signed-SHAP vectors"
                ),

            "sign_agreement":
                (
                    "fraction of features with identical "
                    "numpy.sign of mean signed SHAP"
                ),
        },

    "structure_output":
        {
            "file":
                STRUCTURE_OUTPUT_FILE.as_posix(),

            "sha256":
                structure_hash,

            "rows":
                EXPECTED_STRUCTURE_ROWS,

            "metrics_per_model":
                8,

            "metrics":
                METRIC_ORDER,
        },

    "featurewise_output":
        {
            "file":
                FEATUREWISE_OUTPUT_FILE.as_posix(),

            "sha256":
                featurewise_hash,

            "rows":
                EXPECTED_FEATUREWISE_ROWS,

            "features_per_model":
                EXPECTED_FEATURES,

            "descriptive_only":
                True,
        },

    "comparison_rules":
        {
            "cross_cycle_within_model_only":
                True,

            "cross_model_raw_shap_magnitude_comparison":
                False,

            "model_specific_output_spaces_preserved":
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

            "cycle24_null_changed":
                False,

            "observed_point_estimates_computed":
                True,

            "null_thresholds_applied":
                False,

            "null_exceedance_evaluated":
                False,

            "featurewise_inference_performed":
                False,

            "fdr_applied":
                False,

            "common_support_sensitivity_applied":
                False,

            "correlated_group_robustness_applied":
                False,

            "drift_conclusion_made":
                False,

            "stability_conclusion_made":
                False,

            "shap_is_causal":
                False,
        },

    "interpretation_rule":
        (
            "These are observed descriptive cross-cycle "
            "point estimates only. No metric may be used "
            "to declare explanation drift or stability "
            "until it is compared against the frozen "
            "within-Cycle24 null baseline and required "
            "robustness analyses are completed."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_4_"
            "CYCLE25_TEMPORAL_BLOCK_SHAP"
        ),
}


# ============================================================
# ATOMIC CONTRACT WRITE
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
    "FROZEN_OVERALL_CROSS_CYCLE_SHAP_POINT_ESTIMATES"
):

    raise RuntimeError(
        "10.3.2B contract round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "null_thresholds_applied"
    ]
    is not False
):

    raise RuntimeError(
        "Null thresholds unexpectedly applied."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "drift_conclusion_made"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected drift conclusion state."
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

print(
    "=" * 78
)

print(
    "MODEL CHAPTER 10.3.2B — COMPLETE"
)

print(
    "=" * 78
)


print(
    "Models:",
    EXPECTED_MODELS
)

print(
    "Features:",
    EXPECTED_FEATURES
)

print(
    "Structure metrics/model:",
    8
)

print(
    "Structure rows:",
    len(
        structure
    )
)

print(
    "Featurewise rows:",
    len(
        featurewise
    )
)


print()

print(
    "Cycle24 reference:",
    "HOLDOUT"
)

print(
    "Cycle24 rows:",
    EXPECTED_C24_ROWS
)

print(
    "Cycle25 transfer:",
    "POOLED"
)

print(
    "Cycle25 rows:",
    EXPECTED_C25_ROWS
)


print()

print(
    "Exact Chapter-9 metric formulas:",
    "PASS"
)

print(
    "EPSILON:",
    EPSILON
)

print(
    "Null thresholds applied:",
    False
)

print(
    "Featurewise inference performed:",
    False
)

print(
    "Drift conclusion made:",
    False
)


print()

print(
    "Status:",
    (
        "FROZEN_OVERALL_CROSS_CYCLE_"
        "SHAP_POINT_ESTIMATES"
    )
)


print()

print(
    "Structure estimates:",
    STRUCTURE_OUTPUT_FILE
)

print(
    "Featurewise estimates:",
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
        "Chapter 10.4 — "
        "Cycle25 Temporal-Block SHAP Analysis"
    )
)