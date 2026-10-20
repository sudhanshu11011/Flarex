# ============================================================
# MODEL CHAPTER 10.4.2
# TEMPORAL-BLOCK CROSS-CYCLE SHAP POINT ESTIMATES
#
# Comparisons:
#
#   Cycle24 HOLDOUT vs Cycle25 B1
#   Cycle24 HOLDOUT vs Cycle25 B2
#   Cycle24 HOLDOUT vs Cycle25 B3
#
# Within same frozen model only.
#
# Exact Chapter-9 structure metric definitions are reused.
#
# THIS STEP DOES NOT:
#   - apply Cycle24 null thresholds
#   - perform featurewise inference
#   - apply FDR
#   - make drift/stability conclusions
#   - recompute SHAP
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

EXPECTED_STRUCTURE_ROWS = (
    3
    *
    3
    *
    8
)

EXPECTED_FEATUREWISE_ROWS = (
    3
    *
    3
    *
    25
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


BLOCK_ORDER = [
    "B1",
    "B2",
    "B3",
]


BLOCK_SPECS = {

    "B1": {
        "years": "2020-2021",
        "rows": 10411,
        "positives": 133,
        "groups": 146,
    },

    "B2": {
        "years": "2022-2023",
        "rows": 33132,
        "positives": 1801,
        "groups": 414,
    },

    "B3": {
        "years": "2024-2025",
        "rows": 35818,
        "positives": 2902,
        "groups": 458,
    },
}


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
    "temporal_block_cross_cycle_shap_inputs_contract.json"
)


INPUT_REGISTRY_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "temporal_block_cross_cycle_shap_input_registry.csv"
)


OVERALL_POINT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "overall_cross_cycle_shap_point_estimates_contract.json"
)


# ============================================================
# OUTPUTS
# ============================================================

STRUCTURE_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "temporal_block_cross_cycle_shap_structure_point_estimates.csv"
)


FEATUREWISE_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "temporal_block_cross_cycle_shap_featurewise_point_estimates.csv"
)


OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "temporal_block_cross_cycle_shap_point_estimates_contract.json"
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
# EXACT CHAPTER-9 METRIC DEFINITIONS
# ============================================================

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


def top_k_overlap(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
) -> float:

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
            np.sign(a)
            ==
            np.sign(b)
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


def deterministic_ranks(
    values: np.ndarray,
) -> np.ndarray:

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


    ranks[
        order
    ] = np.arange(
        1,
        len(values) + 1,
        dtype=np.int64,
    )


    return ranks


# ============================================================
# EXACT 10.4.1 BLOCK MEMBERSHIP FINGERPRINT
# ============================================================

def block_membership_fingerprint(
    frame: pd.DataFrame,
) -> str:

    digest = hashlib.sha256()


    timestamps = pd.to_datetime(
        frame[
            TIME_COLUMN
        ],
        utc=True,
        errors="raise",
    )


    for (
        source_row,
        block,
        group,
        target,
        timestamp,
    ) in zip(
        frame[
            "source_row_number"
        ].astype(
            np.int64
        ),
        frame[
            "cycle25_block"
        ].astype(str),
        frame[
            ENTITY_GROUP_COLUMN
        ].astype(str),
        frame[
            TARGET_COLUMN
        ].astype(
            np.int8
        ),
        timestamps,
    ):

        line = (
            f"{int(source_row)}|"
            f"{block}|"
            f"{group}|"
            f"{int(target)}|"
            f"{timestamp.isoformat()}\n"
        )


        digest.update(
            line.encode(
                "utf-8"
            )
        )


    return digest.hexdigest()


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.4.2 — "
    "TEMPORAL-BLOCK SHAP POINT ESTIMATES"
)

print("=" * 78)


# ============================================================
# REQUIRED INPUTS
# ============================================================

for path in [
    INPUT_CONTRACT_FILE,
    INPUT_REGISTRY_FILE,
    OVERALL_POINT_CONTRACT_FILE,
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
            "10.4.2 point estimates."
        )


# ============================================================
# VERIFY 10.4.1 CONTRACT
# ============================================================

input_contract = load_json(
    INPUT_CONTRACT_FILE
)


if (
    input_contract.get(
        "status"
    )
    !=
    "FROZEN_TEMPORAL_BLOCK_CROSS_CYCLE_SHAP_INPUTS"
):

    raise RuntimeError(
        "Chapter 10.4.1 temporal inputs "
        "are not frozen."
    )


input_contract_hash = sha256_file(
    INPUT_CONTRACT_FILE
)


if (
    input_contract[
        "scientific_guardrails"
    ][
        "temporal_metrics_computed"
    ]
    is not False
):

    raise RuntimeError(
        "Temporal metrics were unexpectedly "
        "computed before 10.4.2."
    )


print(
    "Chapter 10.4.1 frozen inputs:",
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
        "10.4.1 temporal input registry "
        "hash mismatch."
    )


registry = pd.read_csv(
    INPUT_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    registry
) != 9:

    raise RuntimeError(
        "Expected exactly 9 frozen "
        "model/block comparison rows."
    )


print(
    "Frozen temporal registry:",
    "PASS"
)


# ============================================================
# VERIFY PREVIOUS OVERALL POINT ESTIMATES
# ============================================================

overall_contract = load_json(
    OVERALL_POINT_CONTRACT_FILE
)


if (
    overall_contract.get(
        "status"
    )
    !=
    "FROZEN_OVERALL_CROSS_CYCLE_SHAP_POINT_ESTIMATES"
):

    raise RuntimeError(
        "Overall cross-cycle point estimates "
        "are not frozen."
    )


if (
    overall_contract[
        "metric_definition_source"
    ][
        "epsilon"
    ]
    !=
    EPSILON
):

    raise RuntimeError(
        "EPSILON differs from frozen "
        "10.3 metric definition."
    )


if (
    overall_contract[
        "metric_definition_source"
    ][
        "exact_chapter9_formulas_reused"
    ]
    is not True
):

    raise RuntimeError(
        "10.3 did not record exact "
        "Chapter-9 formulas."
    )


print(
    "Exact Chapter-9 metric definition:",
    "PASS"
)


# ============================================================
# RESULT CONTAINERS
# ============================================================

structure_rows = []

featurewise_rows = []


# ============================================================
# ANALYZE EACH MODEL
# ============================================================

for candidate_id in MODEL_ORDER:

    print()

    print(
        "ANALYZE MODEL:",
        candidate_id
    )


    model_rows = registry[
        registry[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ].copy()


    if len(
        model_rows
    ) != 3:

        raise RuntimeError(
            f"{candidate_id}: expected "
            "exactly 3 temporal blocks."
        )


    if (
        model_rows[
            "cycle25_block"
        ].astype(str).tolist()
        !=
        BLOCK_ORDER
    ):

        raise RuntimeError(
            f"{candidate_id}: temporal "
            "block ordering mismatch."
        )


    # --------------------------------------------------------
    # REFERENCE C24 FILE MUST BE SAME FOR ALL THREE BLOCKS
    # --------------------------------------------------------

    c24_files = (
        model_rows[
            "reference_shap_file"
        ]
        .astype(str)
        .unique()
    )


    c24_hashes = (
        model_rows[
            "reference_shap_sha256"
        ]
        .astype(str)
        .unique()
    )


    if len(
        c24_files
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: multiple "
            "Cycle24 reference files."
        )


    if len(
        c24_hashes
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: multiple "
            "Cycle24 reference hashes."
        )


    c24_file = Path(
        c24_files[0]
    )


    verify_exists(
        c24_file
    )


    if (
        sha256_file(
            c24_file
        )
        !=
        c24_hashes[0]
    ):

        raise RuntimeError(
            f"{candidate_id}: frozen "
            "Cycle24 SHAP changed."
        )


    # --------------------------------------------------------
    # LOAD C24 SHAP ONCE
    # --------------------------------------------------------

    c24_frame = pd.read_parquet(
        c24_file,
        columns=SHAP_COLUMNS,
    )


    if len(
        c24_frame
    ) != EXPECTED_C24_ROWS:

        raise RuntimeError(
            f"{candidate_id}: Cycle24 "
            "row-count mismatch."
        )


    c24_shap = c24_frame.to_numpy(
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


    if not np.isfinite(
        c24_shap
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "Cycle24 SHAP."
        )


    c24_mean_abs = np.mean(
        np.abs(
            c24_shap
        ),
        axis=0,
    )


    c24_mean_signed = np.mean(
        c24_shap,
        axis=0,
    )


    c24_ranks = deterministic_ranks(
        c24_mean_abs
    )


    # --------------------------------------------------------
    # SAME C25 ARTIFACT FOR B1/B2/B3
    # --------------------------------------------------------

    c25_files = (
        model_rows[
            "cycle25_shap_file"
        ]
        .astype(str)
        .unique()
    )


    c25_hashes = (
        model_rows[
            "cycle25_shap_sha256"
        ]
        .astype(str)
        .unique()
    )


    if len(
        c25_files
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: multiple "
            "Cycle25 SHAP files."
        )


    if len(
        c25_hashes
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: multiple "
            "Cycle25 SHAP hashes."
        )


    c25_file = Path(
        c25_files[0]
    )


    verify_exists(
        c25_file
    )


    if (
        sha256_file(
            c25_file
        )
        !=
        c25_hashes[0]
    ):

        raise RuntimeError(
            f"{candidate_id}: frozen "
            "Cycle25 SHAP changed."
        )


    # --------------------------------------------------------
    # LOAD C25 DATA ONCE
    # --------------------------------------------------------

    c25_frame = pd.read_parquet(
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


    if len(
        c25_frame
    ) != 79361:

        raise RuntimeError(
            f"{candidate_id}: Cycle25 "
            "row-count mismatch."
        )


    # ========================================================
    # EACH TEMPORAL BLOCK
    # ========================================================

    for block_name in BLOCK_ORDER:

        print()

        print(
            "BLOCK:",
            block_name
        )


        spec = BLOCK_SPECS[
            block_name
        ]


        registry_match = model_rows[
            model_rows[
                "cycle25_block"
            ].astype(str)
            ==
            block_name
        ]


        if len(
            registry_match
        ) != 1:

            raise RuntimeError(
                f"{candidate_id} {block_name}: "
                "registry lookup failed."
            )


        registry_row = registry_match.iloc[
            0
        ]


        # ----------------------------------------------------
        # BLOCK SUBSET
        # ----------------------------------------------------

        block_frame = c25_frame[
            c25_frame[
                "cycle25_block"
            ].astype(str)
            ==
            block_name
        ].copy()


        block_frame = block_frame.sort_values(
            "source_row_number",
            kind="stable",
        ).reset_index(
            drop=True
        )


        if len(
            block_frame
        ) != spec[
            "rows"
        ]:

            raise RuntimeError(
                f"{candidate_id} {block_name}: "
                "block row-count mismatch."
            )


        positive_rows = int(
            (
                block_frame[
                    TARGET_COLUMN
                ]
                ==
                1
            ).sum()
        )


        entity_groups = int(
            block_frame[
                ENTITY_GROUP_COLUMN
            ]
            .astype(str)
            .nunique()
        )


        if (
            positive_rows
            !=
            spec[
                "positives"
            ]
        ):

            raise RuntimeError(
                f"{candidate_id} {block_name}: "
                "positive-count mismatch."
            )


        if (
            entity_groups
            !=
            spec[
                "groups"
            ]
        ):

            raise RuntimeError(
                f"{candidate_id} {block_name}: "
                "group-count mismatch."
            )


        # ----------------------------------------------------
        # BLOCK MEMBERSHIP HASH
        # ----------------------------------------------------

        membership_frame = block_frame[
            [
                "source_row_number",
                "cycle25_block",
                ENTITY_GROUP_COLUMN,
                TARGET_COLUMN,
                TIME_COLUMN,
            ]
        ].copy()


        current_membership_hash = (
            block_membership_fingerprint(
                membership_frame
            )
        )


        expected_membership_hash = str(
            registry_row[
                "block_membership_sha256"
            ]
        )


        if (
            current_membership_hash
            !=
            expected_membership_hash
        ):

            raise RuntimeError(
                f"{candidate_id} {block_name}: "
                "frozen block membership changed."
            )


        # ----------------------------------------------------
        # SHAP MATRIX
        # ----------------------------------------------------

        block_shap = block_frame[
            SHAP_COLUMNS
        ].to_numpy(
            dtype=np.float64
        )


        if block_shap.shape != (
            spec[
                "rows"
            ],
            EXPECTED_FEATURES,
        ):

            raise RuntimeError(
                f"{candidate_id} {block_name}: "
                "SHAP matrix shape mismatch."
            )


        if not np.isfinite(
            block_shap
        ).all():

            raise RuntimeError(
                f"{candidate_id} {block_name}: "
                "non-finite SHAP values."
            )


        # ----------------------------------------------------
        # BLOCK GLOBAL VECTORS
        # ----------------------------------------------------

        block_mean_abs = np.mean(
            np.abs(
                block_shap
            ),
            axis=0,
        )


        block_mean_signed = np.mean(
            block_shap,
            axis=0,
        )


        block_ranks = deterministic_ranks(
            block_mean_abs
        )


        # ----------------------------------------------------
        # EXACT STRUCTURE METRICS
        # ----------------------------------------------------

        metrics = calculate_structure_metrics(
            mean_abs_a=c24_mean_abs,
            mean_abs_b=block_mean_abs,
            mean_signed_a=c24_mean_signed,
            mean_signed_b=block_mean_signed,
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
                    f"{candidate_id} "
                    f"{block_name}: non-finite "
                    f"{metric_name}."
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

                    "reference_population":
                        "CYCLE24_HOLDOUT",

                    "transfer_population":
                        f"CYCLE25_{block_name}",

                    "cycle25_block":
                        block_name,

                    "year_definition":
                        spec[
                            "years"
                        ],

                    "stratum":
                        "OVERALL",

                    "reference_rows":
                        EXPECTED_C24_ROWS,

                    "transfer_rows":
                        spec[
                            "rows"
                        ],

                    "transfer_positive_rows":
                        positive_rows,

                    "transfer_entity_groups":
                        entity_groups,

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

                    "null_exceedance_evaluated":
                        False,

                    "drift_classification":
                        "NOT_EVALUATED",
                }
            )


        # ----------------------------------------------------
        # FEATUREWISE DESCRIPTIVE ESTIMATES
        # ----------------------------------------------------

        for feature_index, feature in enumerate(
            FEATURES
        ):

            c24_abs = float(
                c24_mean_abs[
                    feature_index
                ]
            )


            c25_abs = float(
                block_mean_abs[
                    feature_index
                ]
            )


            c24_signed = float(
                c24_mean_signed[
                    feature_index
                ]
            )


            c25_signed = float(
                block_mean_signed[
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

                    "reference_population":
                        "CYCLE24_HOLDOUT",

                    "transfer_population":
                        f"CYCLE25_{block_name}",

                    "cycle25_block":
                        block_name,

                    "year_definition":
                        spec[
                            "years"
                        ],

                    "stratum":
                        "OVERALL",

                    "feature_index":
                        feature_index + 1,

                    "feature":
                        feature,

                    "cycle24_mean_absolute_shap":
                        c24_abs,

                    "cycle25_block_mean_absolute_shap":
                        c25_abs,

                    "delta_mean_absolute_shap":
                        (
                            c25_abs
                            -
                            c24_abs
                        ),

                    "absolute_delta_mean_absolute_shap":
                        abs(
                            c25_abs
                            -
                            c24_abs
                        ),

                    "cycle24_mean_signed_shap":
                        c24_signed,

                    "cycle25_block_mean_signed_shap":
                        c25_signed,

                    "delta_mean_signed_shap":
                        (
                            c25_signed
                            -
                            c24_signed
                        ),

                    "cycle24_mean_abs_rank":
                        int(
                            c24_ranks[
                                feature_index
                            ]
                        ),

                    "cycle25_block_mean_abs_rank":
                        int(
                            block_ranks[
                                feature_index
                            ]
                        ),

                    "rank_change_cycle25_minus_cycle24":
                        int(
                            block_ranks[
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
                                c24_signed
                            )
                        ),

                    "cycle25_signed_direction":
                        int(
                            np.sign(
                                c25_signed
                            )
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


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            "Rows:",
            spec[
                "rows"
            ],
            "PASS"
        )

        print(
            "Membership fingerprint:",
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
# BUILD OUTPUT TABLES
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
# VERIFY METRIC INVENTORY
# ============================================================

for candidate_id in MODEL_ORDER:

    for block_name in BLOCK_ORDER:

        subset = structure[
            (
                structure[
                    "candidate_id"
                ]
                ==
                candidate_id
            )
            &
            (
                structure[
                    "cycle25_block"
                ]
                ==
                block_name
            )
        ]


        if (
            subset[
                "metric"
            ].tolist()
            !=
            METRIC_ORDER
        ):

            raise RuntimeError(
                f"{candidate_id} {block_name}: "
                "metric inventory mismatch."
            )


print()

print(
    "Temporal structure metric inventory:",
    "PASS"
)

print(
    "Temporal featurewise inventory:",
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
# ATOMIC CSV WRITE
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


for temp_file in [
    structure_temp,
    featurewise_temp,
]:

    if temp_file.exists():

        temp_file.unlink()


structure.to_csv(
    structure_temp,
    index=False,
)


featurewise.to_csv(
    featurewise_temp,
    index=False,
)


# ============================================================
# ROUND-TRIP VALIDATION
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
        "Temporal structure CSV "
        "round-trip failed."
    )


if len(
    featurewise_rt
) != EXPECTED_FEATUREWISE_ROWS:

    raise RuntimeError(
        "Temporal featurewise CSV "
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
    "Temporal structure CSV round-trip:",
    "PASS"
)

print(
    "Temporal featurewise CSV round-trip:",
    "PASS"
)


# ============================================================
# FINAL CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Temporal-Block "
            "Cross-Cycle SHAP Point Estimates"
        ),

    "version":
        "1.0",

    "status":
        (
            "FROZEN_TEMPORAL_BLOCK_"
            "CROSS_CYCLE_SHAP_POINT_ESTIMATES"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_4_2",

    "source_temporal_inputs":
        {
            "contract_file":
                INPUT_CONTRACT_FILE.as_posix(),

            "contract_sha256":
                input_contract_hash,

            "required_status":
                (
                    "FROZEN_TEMPORAL_BLOCK_"
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

            "transfer_populations":
                [
                    "CYCLE25_B1",
                    "CYCLE25_B2",
                    "CYCLE25_B3",
                ],

            "models":
                EXPECTED_MODELS,

            "features":
                EXPECTED_FEATURES,

            "stratum":
                "OVERALL",

            "comparison_unit":
                "WITHIN_SAME_FROZEN_MODEL",
        },

    "metric_definition":
        {
            "epsilon":
                EPSILON,

            "exact_chapter9_formulas_reused":
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

            "metrics_per_model_per_block":
                8,
        },

    "featurewise_output":
        {
            "file":
                FEATUREWISE_OUTPUT_FILE.as_posix(),

            "sha256":
                featurewise_hash,

            "rows":
                EXPECTED_FEATUREWISE_ROWS,

            "features_per_model_per_block":
                EXPECTED_FEATURES,

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

            "cycle24_null_changed":
                False,

            "temporal_point_estimates_computed":
                True,

            "null_thresholds_applied":
                False,

            "null_exceedance_evaluated":
                False,

            "featurewise_inference_performed":
                False,

            "fdr_applied":
                False,

            "class_conditional_analysis_performed":
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
            "Temporal-block values are observed "
            "descriptive point estimates only. "
            "No block may be classified as "
            "explanation drift or stability until "
            "the frozen within-Cycle24 null baseline "
            "and later robustness analyses are applied."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_5_"
            "CLASS_CONDITIONAL_SHAP_ANALYSIS"
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
    "FROZEN_TEMPORAL_BLOCK_CROSS_CYCLE_SHAP_POINT_ESTIMATES"
):

    raise RuntimeError(
        "10.4.2 contract round-trip failed."
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

print("=" * 78)

print(
    "MODEL CHAPTER 10.4.2 — COMPLETE"
)

print("=" * 78)


print(
    "Models:",
    EXPECTED_MODELS
)

print(
    "Temporal blocks:",
    3
)

print(
    "Features:",
    EXPECTED_FEATURES
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
    "Exact Chapter-9 formulas:",
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
    "Featurewise inference:",
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
        "FROZEN_TEMPORAL_BLOCK_"
        "CROSS_CYCLE_SHAP_POINT_ESTIMATES"
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
        "Chapter 10.5 — "
        "Class-Conditional SHAP Analysis"
    )
)