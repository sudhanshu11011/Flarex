# ============================================================
# MODEL CHAPTER 10.5.2
# CLASS-CONDITIONAL CROSS-CYCLE SHAP POINT ESTIMATES
#
# Comparisons:
#
#   Cycle24 HOLDOUT TARGET_0
#       vs
#   Cycle25 POOLED TARGET_0
#
#   Cycle24 HOLDOUT TARGET_1
#       vs
#   Cycle25 POOLED TARGET_1
#
# Within same frozen model only.
#
# Exact Chapter-9 structure metrics are reused.
#
# THIS STEP DOES NOT:
#   - apply Cycle24 null thresholds
#   - perform featurewise inference
#   - apply FDR
#   - perform common-support sensitivity
#   - perform correlated-group robustness
#   - make drift/stability conclusions
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

EXPECTED_STRUCTURE_ROWS = (
    3
    *
    2
    *
    8
)

EXPECTED_FEATUREWISE_ROWS = (
    3
    *
    2
    *
    25
)


ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TARGET_COLUMN = "target"


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


STRATUM_ORDER = [
    "TARGET_0",
    "TARGET_1",
]


STRATUM_SPECS = {

    "TARGET_0": {
        "target_value": 0,
        "cycle24_rows": 12545,
        "cycle25_rows": 74525,
    },

    "TARGET_1": {
        "target_value": 1,
        "cycle24_rows": 527,
        "cycle25_rows": 4836,
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
    "class_conditional_cross_cycle_shap_inputs_contract.json"
)


INPUT_REGISTRY_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "class_conditional_cross_cycle_shap_input_registry.csv"
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
    "class_conditional_cross_cycle_shap_structure_point_estimates.csv"
)


FEATUREWISE_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "class_conditional_cross_cycle_shap_featurewise_point_estimates.csv"
)


OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "class_conditional_cross_cycle_shap_point_estimates_contract.json"
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
# EXACT CHAPTER-9 STRUCTURE METRICS
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
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.5.2 — "
    "CLASS-CONDITIONAL SHAP POINT ESTIMATES"
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
            "10.5.2 outputs."
        )


# ============================================================
# VERIFY CLASS INPUT CONTRACT
# ============================================================

input_contract = load_json(
    INPUT_CONTRACT_FILE
)


if (
    input_contract.get(
        "status"
    )
    !=
    "FROZEN_CLASS_CONDITIONAL_CROSS_CYCLE_SHAP_INPUTS"
):

    raise RuntimeError(
        "Class-conditional inputs are not frozen."
    )


input_contract_hash = sha256_file(
    INPUT_CONTRACT_FILE
)


if (
    input_contract[
        "scientific_guardrails"
    ][
        "class_conditional_metrics_computed"
    ]
    is not False
):

    raise RuntimeError(
        "Class-conditional metrics unexpectedly "
        "computed before 10.5.2."
    )


print(
    "Frozen class-conditional inputs:",
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
        "Class-conditional input registry changed."
    )


registry = pd.read_csv(
    INPUT_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    registry
) != 6:

    raise RuntimeError(
        "Expected exactly 6 frozen "
        "model/stratum rows."
    )


print(
    "Frozen class registry:",
    "PASS"
)


# ============================================================
# VERIFY EXACT METRIC DEFINITION SOURCE
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
        "Overall SHAP metric contract "
        "is not frozen."
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
        "EPSILON mismatch with frozen "
        "Chapter-10.3 metric definition."
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
        "Exact Chapter-9 metric formulas "
        "were not frozen."
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
# MODEL × CLASS
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
    ) != 2:

        raise RuntimeError(
            f"{candidate_id}: expected "
            "2 class strata."
        )


    if (
        model_rows[
            "stratum"
        ].astype(str).tolist()
        !=
        STRATUM_ORDER
    ):

        raise RuntimeError(
            f"{candidate_id}: class-stratum "
            "ordering mismatch."
        )


    # ========================================================
    # EACH STRATUM
    # ========================================================

    for stratum_name in STRATUM_ORDER:

        print()

        print(
            "STRATUM:",
            stratum_name
        )


        spec = STRATUM_SPECS[
            stratum_name
        ]


        target_value = spec[
            "target_value"
        ]


        match = model_rows[
            model_rows[
                "stratum"
            ].astype(str)
            ==
            stratum_name
        ]


        if len(
            match
        ) != 1:

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "registry lookup failed."
            )


        row = match.iloc[
            0
        ]


        # ----------------------------------------------------
        # FROZEN GUARDRAILS
        # ----------------------------------------------------

        for column in [
            "same_frozen_model",
            "same_feature_order",
            "same_output_space",
            "same_frozen_background",
        ]:

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
                    f"{candidate_id} {stratum_name}: "
                    f"guardrail failed: {column}"
                )


        # ----------------------------------------------------
        # FILES + HASHES
        # ----------------------------------------------------

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


        if (
            sha256_file(
                c24_file
            )
            !=
            str(
                row[
                    "cycle24_shap_sha256"
                ]
            )
        ):

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "Cycle24 SHAP changed."
            )


        if (
            sha256_file(
                c25_file
            )
            !=
            str(
                row[
                    "cycle25_shap_sha256"
                ]
            )
        ):

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "Cycle25 SHAP changed."
            )


        # ----------------------------------------------------
        # LOAD CLASS-CONDITIONED C24
        # ----------------------------------------------------

        c24 = pd.read_parquet(
            c24_file,
            columns=[
                TARGET_COLUMN,
                *SHAP_COLUMNS,
            ],
        )


        c24_subset = c24[
            c24[
                TARGET_COLUMN
            ]
            ==
            target_value
        ]


        if len(
            c24_subset
        ) != spec[
            "cycle24_rows"
        ]:

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "Cycle24 class row-count mismatch."
            )


        # ----------------------------------------------------
        # LOAD CLASS-CONDITIONED C25
        # ----------------------------------------------------

        c25 = pd.read_parquet(
            c25_file,
            columns=[
                TARGET_COLUMN,
                *SHAP_COLUMNS,
            ],
        )


        c25_subset = c25[
            c25[
                TARGET_COLUMN
            ]
            ==
            target_value
        ]


        if len(
            c25_subset
        ) != spec[
            "cycle25_rows"
        ]:

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "Cycle25 class row-count mismatch."
            )


        # ----------------------------------------------------
        # SHAP MATRICES
        # ----------------------------------------------------

        c24_shap = c24_subset[
            SHAP_COLUMNS
        ].to_numpy(
            dtype=np.float64
        )


        c25_shap = c25_subset[
            SHAP_COLUMNS
        ].to_numpy(
            dtype=np.float64
        )


        if c24_shap.shape != (
            spec[
                "cycle24_rows"
            ],
            EXPECTED_FEATURES,
        ):

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "C24 SHAP shape mismatch."
            )


        if c25_shap.shape != (
            spec[
                "cycle25_rows"
            ],
            EXPECTED_FEATURES,
        ):

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "C25 SHAP shape mismatch."
            )


        if not np.isfinite(
            c24_shap
        ).all():

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "non-finite C24 SHAP."
            )


        if not np.isfinite(
            c25_shap
        ).all():

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "non-finite C25 SHAP."
            )


        # ----------------------------------------------------
        # GLOBAL CLASS-CONDITIONAL VECTORS
        # ----------------------------------------------------

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


        c24_ranks = deterministic_ranks(
            c24_mean_abs
        )


        c25_ranks = deterministic_ranks(
            c25_mean_abs
        )


        # ----------------------------------------------------
        # EXACT STRUCTURE METRICS
        # ----------------------------------------------------

        metrics = calculate_structure_metrics(
            mean_abs_a=c24_mean_abs,
            mean_abs_b=c25_mean_abs,
            mean_signed_a=c24_mean_signed,
            mean_signed_b=c25_mean_signed,
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
                    f"{candidate_id} {stratum_name}: "
                    f"non-finite {metric_name}."
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

                    "reference_population":
                        "CYCLE24_HOLDOUT",

                    "transfer_population":
                        "CYCLE25_POOLED",

                    "stratum":
                        stratum_name,

                    "target_value":
                        target_value,

                    "cycle24_rows":
                        spec[
                            "cycle24_rows"
                        ],

                    "cycle24_entity_groups":
                        int(
                            row[
                                "cycle24_entity_groups"
                            ]
                        ),

                    "cycle25_rows":
                        spec[
                            "cycle25_rows"
                        ],

                    "cycle25_entity_groups":
                        int(
                            row[
                                "cycle25_entity_groups"
                            ]
                        ),

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
                c25_mean_abs[
                    feature_index
                ]
            )


            c24_signed = float(
                c24_mean_signed[
                    feature_index
                ]
            )


            c25_signed = float(
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

                    "output_space":
                        str(
                            row[
                                "output_space"
                            ]
                        ),

                    "reference_population":
                        "CYCLE24_HOLDOUT",

                    "transfer_population":
                        "CYCLE25_POOLED",

                    "stratum":
                        stratum_name,

                    "target_value":
                        target_value,

                    "feature_index":
                        feature_index + 1,

                    "feature":
                        feature,

                    "cycle24_mean_absolute_shap":
                        c24_abs,

                    "cycle25_mean_absolute_shap":
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

                    "cycle25_mean_signed_shap":
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
            "C24 rows:",
            spec[
                "cycle24_rows"
            ],
            "PASS"
        )

        print(
            "C25 rows:",
            spec[
                "cycle25_rows"
            ],
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
# VERIFY INVENTORY
# ============================================================

for candidate_id in MODEL_ORDER:

    for stratum_name in STRATUM_ORDER:

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
                    "stratum"
                ]
                ==
                stratum_name
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
                f"{candidate_id} {stratum_name}: "
                "metric inventory mismatch."
            )


print()

print(
    "Class-conditioned metric inventory:",
    "PASS"
)

print(
    "Class-conditioned featurewise inventory:",
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
        "Class structure CSV round-trip failed."
    )


if len(
    featurewise_rt
) != EXPECTED_FEATUREWISE_ROWS:

    raise RuntimeError(
        "Class featurewise CSV round-trip failed."
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
    "Class structure CSV round-trip:",
    "PASS"
)

print(
    "Class featurewise CSV round-trip:",
    "PASS"
)


# ============================================================
# FINAL CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Class-Conditional "
            "Cross-Cycle SHAP Point Estimates"
        ),

    "version":
        "1.0",

    "status":
        (
            "FROZEN_CLASS_CONDITIONAL_"
            "CROSS_CYCLE_SHAP_POINT_ESTIMATES"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_5_2",

    "source_class_inputs":
        {
            "contract_file":
                INPUT_CONTRACT_FILE.as_posix(),

            "contract_sha256":
                input_contract_hash,

            "required_status":
                (
                    "FROZEN_CLASS_CONDITIONAL_"
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

            "strata":
                STRATUM_ORDER,

            "models":
                EXPECTED_MODELS,

            "features":
                EXPECTED_FEATURES,

            "comparison_unit":
                "WITHIN_SAME_FROZEN_MODEL_AND_CLASS",
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

            "metrics_per_model_per_stratum":
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

            "features_per_model_per_stratum":
                EXPECTED_FEATURES,

            "descriptive_only":
                True,
        },

    "sample_size_caution":
        {
            "target_1_cycle24_rows":
                527,

            "target_1_cycle24_entity_groups":
                25,

            "interpret_c24_target_1_with_caution":
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

            "class_conditional_point_estimates_computed":
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
            "These class-conditional values are "
            "descriptive point estimates only. "
            "They help distinguish overall changes "
            "associated with class composition from "
            "within-class attribution changes. "
            "No drift or stability claim is permitted "
            "until comparison against the frozen "
            "within-Cycle24 null baseline and later "
            "robustness analyses are completed."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_6_"
            "OBSERVED_DRIFT_VS_CYCLE24_NULL"
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
    "FROZEN_CLASS_CONDITIONAL_CROSS_CYCLE_SHAP_POINT_ESTIMATES"
):

    raise RuntimeError(
        "10.5.2 contract round-trip failed."
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
    "MODEL CHAPTER 10.5.2 — COMPLETE"
)

print("=" * 78)


print(
    "Models:",
    EXPECTED_MODELS
)

print(
    "Class strata:",
    2
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
    "TARGET_1 C24 group caution:",
    "25 groups"
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
        "FROZEN_CLASS_CONDITIONAL_"
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
        "Chapter 10.6 — "
        "Observed Drift vs Frozen Cycle24 Null"
    )
)