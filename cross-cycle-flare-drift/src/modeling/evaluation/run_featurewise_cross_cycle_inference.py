# ============================================================
# MODEL CHAPTER 10.9.2
# FEATUREWISE EMPIRICAL INFERENCE + BH-FDR
#
# Frozen protocol:
#
# empirical p =
#   (1 + count(null_abs_delta >= observed_abs_delta))
#   /
#   (2000 + 1)
#
# BH family:
#   candidate_id x stratum x quantity
#
# Formal strata:
#   OVERALL
#   TARGET_0
#   TARGET_1
#
# Quantities:
#   - absolute change in mean-|SHAP|
#   - absolute change in mean signed SHAP
#
# 18 BH families:
#   3 models x 3 strata x 2 quantities
#
# 25 features per family.
#
# Total formal hypotheses = 450.
#
# Temporal B1/B2/B3 are NOT formally tested because
# no frozen block-specific / sample-size-matched
# featurewise Cycle24 null exists.
#
# THIS STEP DOES NOT:
#   - retrain models
#   - recompute SHAP
#   - recompute Cycle24 nulls
#   - change FDR families
#   - infer causal solar physics
#   - make final explanation-drift/stability conclusion
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

N_NULL_REPLICATES = 2000

FDR_Q = 0.05

EXPECTED_MODELS = 3
EXPECTED_STRATA = 3
EXPECTED_FEATURES = 25
EXPECTED_QUANTITIES = 2

EXPECTED_FEATURE_CONTEXTS = (
    EXPECTED_MODELS
    *
    EXPECTED_STRATA
    *
    EXPECTED_FEATURES
)

EXPECTED_INFERENCE_ROWS = (
    EXPECTED_FEATURE_CONTEXTS
    *
    EXPECTED_QUANTITIES
)

EXPECTED_BH_FAMILIES = (
    EXPECTED_MODELS
    *
    EXPECTED_STRATA
    *
    EXPECTED_QUANTITIES
)

EXPECTED_SUMMARY_ROWS = EXPECTED_BH_FAMILIES


MODEL_ORDER = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


STRATUM_ORDER = [
    "OVERALL",
    "TARGET_0",
    "TARGET_1",
]


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


QUANTITY_ORDER = [
    "ABS_DELTA_MEAN_ABS_SHAP",
    "ABS_DELTA_MEAN_SIGNED_SHAP",
]


QUANTITY_CONFIG = {

    "ABS_DELTA_MEAN_ABS_SHAP": {

        "null_column":
            "abs_delta_mean_abs_shap",

        "observed_signed_delta_column":
            "delta_mean_absolute_shap",

        "observed_absolute_delta_column":
            "absolute_delta_mean_absolute_shap",

        "scientific_quantity":
            (
                "Absolute cross-cycle change "
                "in feature mean-|SHAP|"
            ),
    },

    "ABS_DELTA_MEAN_SIGNED_SHAP": {

        "null_column":
            "abs_delta_mean_signed_shap",

        "observed_signed_delta_column":
            "delta_mean_signed_shap",

        # Not required to already exist in observed CSV.
        # It is calculated as abs(delta_mean_signed_shap).
        "observed_absolute_delta_column":
            "absolute_delta_mean_signed_shap",

        "scientific_quantity":
            (
                "Absolute cross-cycle change "
                "in feature mean signed SHAP"
            ),
    },
}


# ============================================================
# FROZEN PROTOCOL
# ============================================================

PROTOCOL_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "featurewise_inference_protocol.json"
)


# ============================================================
# FROZEN CHAPTER-9 NULL
# ============================================================

NULL_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "cycle24_shap_null_contract.json"
)


FEATUREWISE_NULL_FILE = Path(
    "artifacts/models/shift_controls/shap/null/"
    "cycle24_shap_null_featurewise_replicates.parquet"
)


# ============================================================
# FROZEN OBSERVED INPUTS
# ============================================================

OVERALL_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "overall_cross_cycle_shap_point_estimates_contract.json"
)


CLASS_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "class_conditional_cross_cycle_shap_point_estimates_contract.json"
)


TEMPORAL_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "temporal_block_cross_cycle_shap_point_estimates_contract.json"
)


GROUP_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "correlated_group_shap_robustness_contract.json"
)


SUPPORT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "common_support_reweighting_shap_sensitivity_contract.json"
)


# ============================================================
# OUTPUTS
# ============================================================

INFERENCE_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "featurewise_cross_cycle_inference_bh_fdr.csv"
)


SUMMARY_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "featurewise_cross_cycle_inference_summary.csv"
)


OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "featurewise_cross_cycle_inference_contract.json"
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


def verify_exists(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def bh_adjust(
    p_values: np.ndarray,
) -> np.ndarray:

    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )


    if p_values.ndim != 1:

        raise RuntimeError(
            "BH p-value input must be 1-D."
        )


    if len(
        p_values
    ) != EXPECTED_FEATURES:

        raise RuntimeError(
            "BH family must contain exactly "
            f"{EXPECTED_FEATURES} p-values."
        )


    if not np.isfinite(
        p_values
    ).all():

        raise RuntimeError(
            "Non-finite empirical p-value."
        )


    if (
        (
            p_values
            <
            0.0
        ).any()
        or
        (
            p_values
            >
            1.0
        ).any()
    ):

        raise RuntimeError(
            "Empirical p-value outside [0, 1]."
        )


    m = len(
        p_values
    )


    # Stable deterministic sorting.
    order = np.argsort(
        p_values,
        kind="mergesort",
    )


    sorted_p = p_values[
        order
    ]


    ranks = np.arange(
        1,
        m + 1,
        dtype=np.float64,
    )


    raw_adjusted = (
        sorted_p
        *
        float(m)
        /
        ranks
    )


    # BH monotonicity:
    # q_i = min_{j>=i}(m*p_j/j)
    adjusted_sorted = np.minimum.accumulate(
        raw_adjusted[::-1]
    )[::-1]


    adjusted_sorted = np.minimum(
        adjusted_sorted,
        1.0,
    )


    adjusted = np.empty(
        m,
        dtype=np.float64,
    )


    adjusted[
        order
    ] = adjusted_sorted


    return adjusted


def empirical_p_value(
    null_absolute_values: np.ndarray,
    observed_absolute_value: float,
):

    null_absolute_values = np.asarray(
        null_absolute_values,
        dtype=np.float64,
    )


    if len(
        null_absolute_values
    ) != N_NULL_REPLICATES:

        raise RuntimeError(
            "Expected exactly "
            f"{N_NULL_REPLICATES} null values."
        )


    if not np.isfinite(
        null_absolute_values
    ).all():

        raise RuntimeError(
            "Non-finite null distribution."
        )


    if (
        null_absolute_values
        <
        -1e-15
    ).any():

        raise RuntimeError(
            "Negative value in absolute null distribution."
        )


    if not np.isfinite(
        observed_absolute_value
    ):

        raise RuntimeError(
            "Observed absolute delta is non-finite."
        )


    if observed_absolute_value < -1e-15:

        raise RuntimeError(
            "Observed absolute delta is negative."
        )


    # Frozen 10.9.1C rule:
    #
    # p =
    # (1 + count(null_abs >= observed_abs))
    # /
    # (R + 1)
    #
    exceedance_count = int(
        np.sum(
            null_absolute_values
            >=
            observed_absolute_value
        )
    )


    p_value = (
        1.0
        +
        float(
            exceedance_count
        )
    ) / (
        float(
            N_NULL_REPLICATES
        )
        +
        1.0
    )


    return (
        exceedance_count,
        float(
            p_value
        ),
    )


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.9.2 — "
    "FEATUREWISE EMPIRICAL INFERENCE + BH-FDR"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    PROTOCOL_FILE,
    NULL_CONTRACT_FILE,
    FEATUREWISE_NULL_FILE,
    OVERALL_CONTRACT_FILE,
    CLASS_CONTRACT_FILE,
    TEMPORAL_CONTRACT_FILE,
    GROUP_CONTRACT_FILE,
    SUPPORT_CONTRACT_FILE,
]


for path in required_files:

    verify_exists(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    INFERENCE_OUTPUT_FILE,
    SUMMARY_OUTPUT_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter 10.9.2 outputs."
        )


# ============================================================
# VERIFY 10.9.1C PROTOCOL
# ============================================================

protocol = load_json(
    PROTOCOL_FILE
)


if (
    protocol.get(
        "status"
    )
    !=
    "FROZEN_FEATUREWISE_INFERENCE_PROTOCOL"
):

    raise RuntimeError(
        "Featurewise inference protocol "
        "is not frozen."
    )


protocol_hash = sha256_file(
    PROTOCOL_FILE
)


p_rule = protocol[
    "empirical_p_value_rule"
]


if (
    p_rule[
        "tail"
    ]
    !=
    "UPPER_TAIL_ON_ABSOLUTE_DELTA"
):

    raise RuntimeError(
        "Unexpected empirical-p tail."
    )


if (
    p_rule[
        "monte_carlo_correction"
    ]
    !=
    "PLUS_ONE_NUMERATOR_AND_DENOMINATOR"
):

    raise RuntimeError(
        "Unexpected Monte-Carlo correction."
    )


if (
    int(
        p_rule[
            "n_null_replicates"
        ]
    )
    !=
    N_NULL_REPLICATES
):

    raise RuntimeError(
        "Null-replicate count differs "
        "from frozen protocol."
    )


if (
    int(
        p_rule[
            "denominator"
        ]
    )
    !=
    N_NULL_REPLICATES + 1
):

    raise RuntimeError(
        "Empirical-p denominator differs "
        "from frozen protocol."
    )


if (
    p_rule[
        "equality_counts_as_null_exceedance"
    ]
    is not True
):

    raise RuntimeError(
        "Frozen empirical-p equality rule changed."
    )


multiple_testing = protocol[
    "multiple_testing_rule"
]


if (
    multiple_testing[
        "method"
    ]
    !=
    "BENJAMINI_HOCHBERG"
):

    raise RuntimeError(
        "Unexpected multiple-testing method."
    )


if not np.isclose(
    float(
        multiple_testing[
            "fdr_q"
        ]
    ),
    FDR_Q,
    atol=0.0,
    rtol=0.0,
):

    raise RuntimeError(
        "Frozen FDR q changed."
    )


if (
    multiple_testing[
        "family_definition"
    ]
    !=
    "CANDIDATE_ID_X_STRATUM_X_QUANTITY"
):

    raise RuntimeError(
        "Frozen BH family definition changed."
    )


if (
    int(
        multiple_testing[
            "family_count"
        ]
    )
    !=
    EXPECTED_BH_FAMILIES
):

    raise RuntimeError(
        "Frozen BH family count changed."
    )


if (
    int(
        multiple_testing[
            "tests_per_family"
        ]
    )
    !=
    EXPECTED_FEATURES
):

    raise RuntimeError(
        "Frozen tests-per-family changed."
    )


if (
    int(
        multiple_testing[
            "formal_hypothesis_count"
        ]
    )
    !=
    EXPECTED_INFERENCE_ROWS
):

    raise RuntimeError(
        "Frozen hypothesis count changed."
    )


formal_scope = protocol[
    "formal_test_scope"
]


if (
    formal_scope[
        "temporal_blocks_formally_tested"
    ]
    is not False
):

    raise RuntimeError(
        "Temporal featurewise inference "
        "was unexpectedly enabled."
    )


if (
    formal_scope[
        "strata"
    ]
    !=
    STRATUM_ORDER
):

    raise RuntimeError(
        "Frozen formal stratum order changed."
    )


print(
    "Frozen 10.9.1C inference protocol:",
    "PASS"
)


# ============================================================
# VERIFY FROZEN NULL CONTRACT + HASH
# ============================================================

null_contract = load_json(
    NULL_CONTRACT_FILE
)


if (
    null_contract.get(
        "status"
    )
    !=
    "FROZEN_CYCLE24_SHAP_NULL_BASELINE"
):

    raise RuntimeError(
        "Cycle24 SHAP null baseline "
        "is not frozen."
    )


expected_null_hash = (
    protocol[
        "frozen_chapter9_basis"
    ][
        "featurewise_null_file"
    ][
        "sha256"
    ]
)


if (
    sha256_file(
        FEATUREWISE_NULL_FILE
    )
    !=
    expected_null_hash
):

    raise RuntimeError(
        "Frozen full featurewise null changed."
    )


print(
    "Frozen full Cycle24 featurewise null:",
    "PASS"
)


# ============================================================
# VERIFY REQUIRED CHAPTER-10 CONTRACTS
# ============================================================

overall_contract = load_json(
    OVERALL_CONTRACT_FILE
)


class_contract = load_json(
    CLASS_CONTRACT_FILE
)


temporal_contract = load_json(
    TEMPORAL_CONTRACT_FILE
)


group_contract = load_json(
    GROUP_CONTRACT_FILE
)


support_contract = load_json(
    SUPPORT_CONTRACT_FILE
)


required_statuses = [
    (
        overall_contract.get(
            "status"
        ),
        "FROZEN_OVERALL_CROSS_CYCLE_SHAP_POINT_ESTIMATES",
    ),
    (
        class_contract.get(
            "status"
        ),
        "FROZEN_CLASS_CONDITIONAL_CROSS_CYCLE_SHAP_POINT_ESTIMATES",
    ),
    (
        temporal_contract.get(
            "status"
        ),
        "FROZEN_TEMPORAL_BLOCK_CROSS_CYCLE_SHAP_POINT_ESTIMATES",
    ),
    (
        group_contract.get(
            "status"
        ),
        "FROZEN_CORRELATED_GROUP_SHAP_ROBUSTNESS",
    ),
    (
        support_contract.get(
            "status"
        ),
        "FROZEN_COMMON_SUPPORT_REWEIGHTING_SHAP_SENSITIVITY",
    ),
]


for actual, expected in required_statuses:

    if actual != expected:

        raise RuntimeError(
            "Required frozen Chapter-10 "
            f"contract missing: {expected}"
        )


print(
    "Required Chapter-10 robustness analyses:",
    "PASS"
)


# ============================================================
# OBSERVED FEATUREWISE FILES
# ============================================================

overall_feature_file = Path(
    overall_contract[
        "featurewise_output"
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


for path in [
    overall_feature_file,
    class_feature_file,
]:

    verify_exists(
        path
    )


if (
    sha256_file(
        overall_feature_file
    )
    !=
    overall_contract[
        "featurewise_output"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Frozen overall featurewise estimates changed."
    )


if (
    sha256_file(
        class_feature_file
    )
    !=
    class_contract[
        "featurewise_output"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Frozen class-conditional "
        "featurewise estimates changed."
    )


print(
    "Frozen observed featurewise estimates:",
    "PASS"
)


# ============================================================
# LOAD OBSERVED FEATUREWISE TABLES
# ============================================================

overall = pd.read_csv(
    overall_feature_file,
    keep_default_na=False,
)


class_conditional = pd.read_csv(
    class_feature_file,
    keep_default_na=False,
)


if len(
    overall
) != (
    EXPECTED_MODELS
    *
    EXPECTED_FEATURES
):

    raise RuntimeError(
        "Overall featurewise row-count mismatch."
    )


if len(
    class_conditional
) != (
    EXPECTED_MODELS
    *
    2
    *
    EXPECTED_FEATURES
):

    raise RuntimeError(
        "Class featurewise row-count mismatch."
    )


required_observed_columns = {
    "candidate_id",
    "model_family",
    "output_space",
    "feature_index",
    "feature",
    "delta_mean_absolute_shap",
    "absolute_delta_mean_absolute_shap",
    "delta_mean_signed_shap",
}


for name, frame in [
    (
        "overall",
        overall,
    ),
    (
        "class_conditional",
        class_conditional,
    ),
]:

    missing = (
        required_observed_columns
        -
        set(
            frame.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"{name}: missing observed columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# NORMALIZE FORMAL STRATA
# ============================================================

if "stratum" not in overall.columns:

    raise RuntimeError(
        "Overall featurewise table is missing stratum."
    )


if not (
    overall[
        "stratum"
    ].astype(str)
    ==
    "OVERALL"
).all():

    raise RuntimeError(
        "Overall featurewise table contains "
        "unexpected stratum."
    )


if "stratum" not in class_conditional.columns:

    raise RuntimeError(
        "Class featurewise table is missing stratum."
    )


if set(
    class_conditional[
        "stratum"
    ].astype(str).unique()
) != {
    "TARGET_0",
    "TARGET_1",
}:

    raise RuntimeError(
        "Class featurewise stratum set mismatch."
    )


observed = pd.concat(
    [
        overall,
        class_conditional,
    ],
    ignore_index=True,
    sort=False,
)


if len(
    observed
) != EXPECTED_FEATURE_CONTEXTS:

    raise RuntimeError(
        "Formal observed feature-context "
        "row-count mismatch."
    )


# ============================================================
# OBSERVED FEATURE INVENTORY
# ============================================================

for candidate_id in MODEL_ORDER:

    for stratum in STRATUM_ORDER:

        subset = observed[
            (
                observed[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            )
            &
            (
                observed[
                    "stratum"
                ].astype(str)
                ==
                stratum
            )
        ].copy()


        if len(
            subset
        ) != EXPECTED_FEATURES:

            raise RuntimeError(
                f"{candidate_id}/{stratum}: "
                "expected exactly 25 observed features."
            )


        if set(
            subset[
                "feature"
            ].astype(str)
        ) != set(
            FEATURES
        ):

            raise RuntimeError(
                f"{candidate_id}/{stratum}: "
                "observed feature set mismatch."
            )


        if (
            subset[
                "feature"
            ].astype(str).nunique()
            !=
            EXPECTED_FEATURES
        ):

            raise RuntimeError(
                f"{candidate_id}/{stratum}: "
                "duplicate observed feature."
            )


print(
    "Formal observed feature inventory:",
    "PASS"
)


# ============================================================
# LOAD ONLY REQUIRED NULL COLUMNS
# ============================================================

null_df = pd.read_parquet(
    FEATUREWISE_NULL_FILE,
    columns=[
        "replicate_id",
        "candidate_id",
        "model_family",
        "stratum",
        "feature_index",
        "feature",
        "abs_delta_mean_abs_shap",
        "abs_delta_mean_signed_shap",
    ],
)


expected_null_rows = (
    EXPECTED_MODELS
    *
    EXPECTED_STRATA
    *
    EXPECTED_FEATURES
    *
    N_NULL_REPLICATES
)


if len(
    null_df
) != expected_null_rows:

    raise RuntimeError(
        "Full featurewise null row-count mismatch."
    )


if null_df[
    [
        "candidate_id",
        "stratum",
        "feature",
        "replicate_id",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate null replicate key."
    )


print(
    "Formal null inventory:",
    "PASS"
)


# ============================================================
# FEATUREWISE EMPIRICAL TESTS
# ============================================================

inference_rows = []


for candidate_id in MODEL_ORDER:

    print()
    print("=" * 78)
    print(
        "MODEL:",
        candidate_id
    )
    print("=" * 78)


    for stratum in STRATUM_ORDER:

        observed_subset = observed[
            (
                observed[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            )
            &
            (
                observed[
                    "stratum"
                ].astype(str)
                ==
                stratum
            )
        ].copy()


        observed_subset[
            "feature"
        ] = observed_subset[
            "feature"
        ].astype(str)


        observed_subset = (
            observed_subset
            .set_index(
                "feature"
            )
            .loc[
                FEATURES
            ]
            .reset_index()
        )


        null_subset = null_df[
            (
                null_df[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            )
            &
            (
                null_df[
                    "stratum"
                ].astype(str)
                ==
                stratum
            )
        ].copy()


        if len(
            null_subset
        ) != (
            EXPECTED_FEATURES
            *
            N_NULL_REPLICATES
        ):

            raise RuntimeError(
                f"{candidate_id}/{stratum}: "
                "null subset row-count mismatch."
            )


        print()
        print(
            "STRATUM:",
            stratum
        )


        for _, observed_row in (
            observed_subset.iterrows()
        ):

            feature = str(
                observed_row[
                    "feature"
                ]
            )


            feature_index = int(
                observed_row[
                    "feature_index"
                ]
            )


            expected_feature_index = (
                FEATURES.index(
                    feature
                )
                +
                1
            )


            if (
                feature_index
                !=
                expected_feature_index
            ):

                raise RuntimeError(
                    f"{candidate_id}/{stratum}/{feature}: "
                    "feature index mismatch."
                )


            feature_null = null_subset[
                null_subset[
                    "feature"
                ].astype(str)
                ==
                feature
            ].copy()


            if len(
                feature_null
            ) != N_NULL_REPLICATES:

                raise RuntimeError(
                    f"{candidate_id}/{stratum}/{feature}: "
                    "expected 2000 null replicates."
                )


            if (
                feature_null[
                    "replicate_id"
                ].nunique()
                !=
                N_NULL_REPLICATES
            ):

                raise RuntimeError(
                    f"{candidate_id}/{stratum}/{feature}: "
                    "null replicate IDs incomplete."
                )


            # ------------------------------------------------
            # QUANTITY 1:
            # ABS DELTA MEAN-|SHAP|
            # ------------------------------------------------

            observed_abs_mean_abs = float(
                observed_row[
                    "absolute_delta_mean_absolute_shap"
                ]
            )


            signed_delta_mean_abs = float(
                observed_row[
                    "delta_mean_absolute_shap"
                ]
            )


            if not np.isclose(
                observed_abs_mean_abs,
                abs(
                    signed_delta_mean_abs
                ),
                atol=1e-12,
                rtol=1e-10,
            ):

                raise RuntimeError(
                    f"{candidate_id}/{stratum}/{feature}: "
                    "observed mean-|SHAP| absolute "
                    "delta integrity failed."
                )


            null_abs_mean_abs = feature_null[
                "abs_delta_mean_abs_shap"
            ].to_numpy(
                dtype=np.float64
            )


            (
                exceedance_count_mean_abs,
                p_mean_abs,
            ) = empirical_p_value(
                null_abs_mean_abs,
                observed_abs_mean_abs,
            )


            q975_mean_abs = float(
                np.quantile(
                    null_abs_mean_abs,
                    0.975,
                    method="linear",
                )
            )


            inference_rows.append(
                {
                    "candidate_id":
                        candidate_id,

                    "model_family":
                        str(
                            observed_row[
                                "model_family"
                            ]
                        ),

                    "output_space":
                        str(
                            observed_row[
                                "output_space"
                            ]
                        ),

                    "stratum":
                        stratum,

                    "feature_index":
                        feature_index,

                    "feature":
                        feature,

                    "quantity":
                        "ABS_DELTA_MEAN_ABS_SHAP",

                    "scientific_quantity":
                        QUANTITY_CONFIG[
                            "ABS_DELTA_MEAN_ABS_SHAP"
                        ][
                            "scientific_quantity"
                        ],

                    "observed_signed_delta":
                        signed_delta_mean_abs,

                    "observed_absolute_delta":
                        observed_abs_mean_abs,

                    "null_absolute_quantity":
                        "abs_delta_mean_abs_shap",

                    "n_null_replicates":
                        N_NULL_REPLICATES,

                    "null_q975":
                        q975_mean_abs,

                    "strict_q975_exceedance":
                        bool(
                            observed_abs_mean_abs
                            >
                            q975_mean_abs
                        ),

                    "null_values_ge_observed":
                        exceedance_count_mean_abs,

                    "empirical_p_value":
                        p_mean_abs,

                    "bh_family_id":
                        (
                            f"{candidate_id}|"
                            f"{stratum}|"
                            "ABS_DELTA_MEAN_ABS_SHAP"
                        ),

                    "bh_adjusted_p_value":
                        np.nan,

                    "fdr_q":
                        FDR_Q,

                    "fdr_significant":
                        False,

                    "inference_status":
                        "PENDING_BH_ADJUSTMENT",

                    "final_drift_conclusion":
                        "NOT_MADE",
                }
            )


            # ------------------------------------------------
            # QUANTITY 2:
            # ABS DELTA MEAN SIGNED SHAP
            # ------------------------------------------------

            signed_delta_mean_signed = float(
                observed_row[
                    "delta_mean_signed_shap"
                ]
            )


            observed_abs_mean_signed = abs(
                signed_delta_mean_signed
            )


            null_abs_mean_signed = feature_null[
                "abs_delta_mean_signed_shap"
            ].to_numpy(
                dtype=np.float64
            )


            (
                exceedance_count_mean_signed,
                p_mean_signed,
            ) = empirical_p_value(
                null_abs_mean_signed,
                observed_abs_mean_signed,
            )


            q975_mean_signed = float(
                np.quantile(
                    null_abs_mean_signed,
                    0.975,
                    method="linear",
                )
            )


            inference_rows.append(
                {
                    "candidate_id":
                        candidate_id,

                    "model_family":
                        str(
                            observed_row[
                                "model_family"
                            ]
                        ),

                    "output_space":
                        str(
                            observed_row[
                                "output_space"
                            ]
                        ),

                    "stratum":
                        stratum,

                    "feature_index":
                        feature_index,

                    "feature":
                        feature,

                    "quantity":
                        "ABS_DELTA_MEAN_SIGNED_SHAP",

                    "scientific_quantity":
                        QUANTITY_CONFIG[
                            "ABS_DELTA_MEAN_SIGNED_SHAP"
                        ][
                            "scientific_quantity"
                        ],

                    "observed_signed_delta":
                        signed_delta_mean_signed,

                    "observed_absolute_delta":
                        observed_abs_mean_signed,

                    "null_absolute_quantity":
                        "abs_delta_mean_signed_shap",

                    "n_null_replicates":
                        N_NULL_REPLICATES,

                    "null_q975":
                        q975_mean_signed,

                    "strict_q975_exceedance":
                        bool(
                            observed_abs_mean_signed
                            >
                            q975_mean_signed
                        ),

                    "null_values_ge_observed":
                        exceedance_count_mean_signed,

                    "empirical_p_value":
                        p_mean_signed,

                    "bh_family_id":
                        (
                            f"{candidate_id}|"
                            f"{stratum}|"
                            "ABS_DELTA_MEAN_SIGNED_SHAP"
                        ),

                    "bh_adjusted_p_value":
                        np.nan,

                    "fdr_q":
                        FDR_Q,

                    "fdr_significant":
                        False,

                    "inference_status":
                        "PENDING_BH_ADJUSTMENT",

                    "final_drift_conclusion":
                        "NOT_MADE",
                }
            )


# ============================================================
# BUILD RAW INFERENCE TABLE
# ============================================================

inference = pd.DataFrame(
    inference_rows
)


if len(
    inference
) != EXPECTED_INFERENCE_ROWS:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_INFERENCE_ROWS} inference rows, "
        f"got {len(inference)}."
    )


if inference[
    [
        "candidate_id",
        "stratum",
        "feature",
        "quantity",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate featurewise hypothesis."
    )


if not np.isfinite(
    inference[
        "empirical_p_value"
    ].to_numpy(
        dtype=np.float64
    )
).all():

    raise RuntimeError(
        "Non-finite empirical p-values."
    )


minimum_possible_p = (
    1.0
    /
    (
        N_NULL_REPLICATES
        +
        1
    )
)


if (
    inference[
        "empirical_p_value"
    ].to_numpy(
        dtype=np.float64
    )
    <
    (
        minimum_possible_p
        -
        1e-15
    )
).any():

    raise RuntimeError(
        "Empirical p-value below frozen "
        "minimum possible p."
    )


print()

print(
    "Raw empirical featurewise tests:",
    "PASS"
)


# ============================================================
# APPLY BH WITHIN EACH FROZEN FAMILY
# ============================================================

family_columns = [
    "candidate_id",
    "stratum",
    "quantity",
]


family_count = 0


for family_key, family in inference.groupby(
    family_columns,
    sort=False,
):

    family_count += 1


    if len(
        family
    ) != EXPECTED_FEATURES:

        raise RuntimeError(
            f"{family_key}: BH family "
            "does not contain 25 tests."
        )


    if set(
        family[
            "feature"
        ].astype(str)
    ) != set(
        FEATURES
    ):

        raise RuntimeError(
            f"{family_key}: BH family "
            "feature set mismatch."
        )


    adjusted = bh_adjust(
        family[
            "empirical_p_value"
        ].to_numpy(
            dtype=np.float64
        )
    )


    significant = (
        adjusted
        <=
        FDR_Q
    )


    inference.loc[
        family.index,
        "bh_adjusted_p_value",
    ] = adjusted


    inference.loc[
        family.index,
        "fdr_significant",
    ] = significant


    inference.loc[
        family.index,
        "inference_status",
    ] = np.where(
        significant,
        "FDR_SIGNIFICANT_FEATUREWISE_CHANGE",
        "NOT_FDR_SIGNIFICANT",
    )


if (
    family_count
    !=
    EXPECTED_BH_FAMILIES
):

    raise RuntimeError(
        "BH family count mismatch."
    )


if inference[
    "bh_adjusted_p_value"
].isna().any():

    raise RuntimeError(
        "BH adjusted p-values contain missing values."
    )


if not np.isfinite(
    inference[
        "bh_adjusted_p_value"
    ].to_numpy(
        dtype=np.float64
    )
).all():

    raise RuntimeError(
        "Non-finite BH adjusted p-values."
    )


if (
    inference[
        "bh_adjusted_p_value"
    ].to_numpy(
        dtype=np.float64
    )
    <
    0.0
).any():

    raise RuntimeError(
        "BH adjusted p-value below zero."
    )


if (
    inference[
        "bh_adjusted_p_value"
    ].to_numpy(
        dtype=np.float64
    )
    >
    1.0
).any():

    raise RuntimeError(
        "BH adjusted p-value above one."
    )


print(
    "Benjamini-Hochberg adjustment:",
    "PASS"
)

print(
    "BH families:",
    family_count
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


print()

print("=" * 78)

print(
    "FEATUREWISE BH-FDR SUMMARY"
)

print("=" * 78)


for (
    candidate_id,
    stratum,
    quantity,
), family in inference.groupby(
    family_columns,
    sort=False,
):

    family = family.copy()


    significant = family[
        family[
            "fdr_significant"
        ]
        ==
        True
    ].copy()


    significant = significant.sort_values(
        [
            "bh_adjusted_p_value",
            "empirical_p_value",
            "feature_index",
        ],
        kind="stable",
    )


    significant_features = (
        significant[
            "feature"
        ]
        .astype(str)
        .tolist()
    )


    pointwise_exceedance_count = int(
        family[
            "strict_q975_exceedance"
        ].sum()
    )


    significant_count = len(
        significant_features
    )


    min_empirical_p = float(
        family[
            "empirical_p_value"
        ].min()
    )


    min_bh_p = float(
        family[
            "bh_adjusted_p_value"
        ].min()
    )


    if significant_count > 0:

        family_status = (
            "FDR_SIGNIFICANT_FEATUREWISE_CHANGE_PRESENT"
        )

    else:

        family_status = (
            "NO_FDR_SIGNIFICANT_FEATUREWISE_CHANGE"
        )


    summary_rows.append(
        {
            "family_id":
                (
                    f"{candidate_id}|"
                    f"{stratum}|"
                    f"{quantity}"
                ),

            "candidate_id":
                candidate_id,

            "stratum":
                stratum,

            "quantity":
                quantity,

            "features_tested":
                EXPECTED_FEATURES,

            "null_replicates_per_feature":
                N_NULL_REPLICATES,

            "pointwise_q975_exceedances":
                pointwise_exceedance_count,

            "fdr_significant_features":
                significant_count,

            "significant_feature_names":
                ";".join(
                    significant_features
                ),

            "minimum_empirical_p":
                min_empirical_p,

            "minimum_bh_adjusted_p":
                min_bh_p,

            "fdr_q":
                FDR_Q,

            "family_status":
                family_status,

            "final_drift_conclusion":
                "NOT_MADE",
        }
    )


    print()

    print(
        candidate_id,
        "|",
        stratum,
        "|",
        quantity,
    )


    print(
        "Pointwise q97.5 exceedances:",
        pointwise_exceedance_count,
        "/",
        EXPECTED_FEATURES,
    )


    print(
        "BH-FDR significant:",
        significant_count,
        "/",
        EXPECTED_FEATURES,
    )


    if significant_features:

        print(
            "Features:",
            ";".join(
                significant_features
            )
        )

    else:

        print(
            "Features:",
            "NONE"
        )


    print(
        "Min empirical p:",
        f"{min_empirical_p:.8f}"
    )


    print(
        "Min BH-adjusted p:",
        f"{min_bh_p:.8f}"
    )


summary = pd.DataFrame(
    summary_rows
)


if len(
    summary
) != EXPECTED_SUMMARY_ROWS:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_SUMMARY_ROWS} BH summary rows, "
        f"got {len(summary)}."
    )


# ============================================================
# FINAL INVENTORY CHECKS
# ============================================================

if (
    inference[
        "candidate_id"
    ].nunique()
    !=
    EXPECTED_MODELS
):

    raise RuntimeError(
        "Inference model count mismatch."
    )


if (
    inference[
        "stratum"
    ].nunique()
    !=
    EXPECTED_STRATA
):

    raise RuntimeError(
        "Inference stratum count mismatch."
    )


if (
    inference[
        "quantity"
    ].nunique()
    !=
    EXPECTED_QUANTITIES
):

    raise RuntimeError(
        "Inference quantity count mismatch."
    )


print()

print(
    "Final featurewise inference inventory:",
    "PASS"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

INFERENCE_OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY_OUTPUT_FILE.parent.mkdir(
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

inference_temp = INFERENCE_OUTPUT_FILE.with_name(
    INFERENCE_OUTPUT_FILE.stem
    +
    ".tmp.csv"
)


summary_temp = SUMMARY_OUTPUT_FILE.with_name(
    SUMMARY_OUTPUT_FILE.stem
    +
    ".tmp.csv"
)


for path in [
    inference_temp,
    summary_temp,
]:

    if path.exists():

        path.unlink()


inference.to_csv(
    inference_temp,
    index=False,
)


summary.to_csv(
    summary_temp,
    index=False,
)


# ============================================================
# ROUND-TRIP
# ============================================================

inference_rt = pd.read_csv(
    inference_temp,
    keep_default_na=False,
)


summary_rt = pd.read_csv(
    summary_temp,
    keep_default_na=False,
)


if len(
    inference_rt
) != EXPECTED_INFERENCE_ROWS:

    raise RuntimeError(
        "Inference CSV round-trip failed."
    )


if len(
    summary_rt
) != EXPECTED_SUMMARY_ROWS:

    raise RuntimeError(
        "Inference summary CSV "
        "round-trip failed."
    )


inference_temp.replace(
    INFERENCE_OUTPUT_FILE
)


summary_temp.replace(
    SUMMARY_OUTPUT_FILE
)


inference_hash = sha256_file(
    INFERENCE_OUTPUT_FILE
)


summary_hash = sha256_file(
    SUMMARY_OUTPUT_FILE
)


print(
    "Inference CSV round-trip:",
    "PASS"
)

print(
    "Inference summary CSV round-trip:",
    "PASS"
)


# ============================================================
# CONTRACT
# ============================================================

total_fdr_significant = int(
    inference[
        "fdr_significant"
    ].sum()
)


families_with_significant = int(
    (
        summary[
            "fdr_significant_features"
        ].astype(int)
        >
        0
    ).sum()
)


contract = {

    "contract_name":
        (
            "Frozen Featurewise Cross-Cycle "
            "SHAP Empirical Inference + BH-FDR"
        ),

    "version":
        "1.0",

    "status":
        (
            "FROZEN_FEATUREWISE_CROSS_CYCLE_"
            "INFERENCE_BH_FDR"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_9_2",

    "frozen_protocol":
        {
            "file":
                PROTOCOL_FILE.as_posix(),

            "sha256":
                protocol_hash,

            "required_status":
                "FROZEN_FEATUREWISE_INFERENCE_PROTOCOL",
        },

    "null_source":
        {
            "file":
                FEATUREWISE_NULL_FILE.as_posix(),

            "sha256":
                sha256_file(
                    FEATUREWISE_NULL_FILE
                ),

            "replicates_per_feature":
                N_NULL_REPLICATES,

            "source_population":
                "CYCLE24_HOLDOUT",

            "null_type":
                (
                    "WITHIN_CYCLE24_ENTITY_GROUP_"
                    "BOOTSTRAP_ABSOLUTE_DELTA"
                ),
        },

    "formal_scope":
        {
            "models":
                MODEL_ORDER,

            "strata":
                STRATUM_ORDER,

            "features":
                FEATURES,

            "feature_count":
                EXPECTED_FEATURES,

            "quantities":
                QUANTITY_ORDER,

            "bh_families":
                EXPECTED_BH_FAMILIES,

            "formal_hypotheses":
                EXPECTED_INFERENCE_ROWS,

            "temporal_B1_B2_B3_formal_inference":
                False,
        },

    "empirical_p_rule":
        {
            "formula":
                (
                    "(1 + count(null_abs_delta >= "
                    "observed_abs_delta)) / 2001"
                ),

            "plus_one_correction":
                True,

            "equality_counts_as_null_exceedance":
                True,

            "minimum_possible_p":
                minimum_possible_p,
        },

    "multiple_testing":
        {
            "method":
                "BENJAMINI_HOCHBERG",

            "family":
                "CANDIDATE_ID_X_STRATUM_X_QUANTITY",

            "tests_per_family":
                EXPECTED_FEATURES,

            "family_count":
                EXPECTED_BH_FAMILIES,

            "fdr_q":
                FDR_Q,

            "significant_rule":
                "BH_ADJUSTED_P <= 0.05",
        },

    "results":
        {
            "total_featurewise_hypotheses":
                EXPECTED_INFERENCE_ROWS,

            "total_fdr_significant_hypotheses":
                total_fdr_significant,

            "families_with_at_least_one_fdr_significant_feature":
                families_with_significant,

            "family_count":
                EXPECTED_BH_FAMILIES,
        },

    "inference_output":
        {
            "file":
                INFERENCE_OUTPUT_FILE.as_posix(),

            "sha256":
                inference_hash,

            "rows":
                EXPECTED_INFERENCE_ROWS,
        },

    "summary_output":
        {
            "file":
                SUMMARY_OUTPUT_FILE.as_posix(),

            "sha256":
                summary_hash,

            "rows":
                EXPECTED_SUMMARY_ROWS,
        },

    "scientific_guardrails":
        {
            "cycle25_used_for_training":
                False,

            "models_retrained":
                False,

            "shap_recomputed":
                False,

            "cycle24_null_recomputed":
                False,

            "featurewise_null_modified":
                False,

            "inference_protocol_modified_after_results":
                False,

            "empirical_p_values_computed":
                True,

            "bh_fdr_applied":
                True,

            "temporal_formal_featurewise_inference":
                False,

            "common_support_featurewise_fdr_applied":
                False,

            "correlated_group_fdr_applied":
                False,

            "final_explanation_drift_conclusion_made":
                False,

            "final_explanation_stability_conclusion_made":
                False,

            "shap_is_causal":
                False,
        },

    "sample_size_caution":
        {
            "cycle24_target_1_rows":
                527,

            "cycle24_target_1_entity_groups":
                25,

            "target_1_results_require_caution":
                True,
        },

    "interpretation_rule":
        (
            "An FDR-significant feature indicates that "
            "its observed absolute cross-cycle SHAP change "
            "is unusually large relative to the frozen "
            "same-model, same-stratum, same-feature "
            "within-Cycle24 null distribution and survives "
            "the predefined 25-feature Benjamini-Hochberg "
            "family at q=0.05. This is evidence of "
            "feature-level attribution change, not causal "
            "solar physics and not by itself the final "
            "cross-cycle explanation-drift conclusion."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_10_"
            "FINAL_CROSS_CYCLE_SHAP_FREEZE"
        ),
}


# ============================================================
# SAVE CONTRACT
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
    "FROZEN_FEATUREWISE_CROSS_CYCLE_INFERENCE_BH_FDR"
):

    raise RuntimeError(
        "10.9.2 contract round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "bh_fdr_applied"
    ]
    is not True
):

    raise RuntimeError(
        "BH-FDR state not frozen correctly."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "final_explanation_drift_conclusion_made"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected final explanation-drift conclusion."
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
    "MODEL CHAPTER 10.9.2 — COMPLETE"
)

print("=" * 78)


print(
    "Formal hypotheses:",
    len(
        inference
    )
)

print(
    "BH families:",
    len(
        summary
    )
)

print(
    "Features / family:",
    EXPECTED_FEATURES
)

print(
    "Null replicates / feature:",
    N_NULL_REPLICATES
)


print()

print(
    "FDR q:",
    FDR_Q
)

print(
    "Total FDR-significant hypotheses:",
    total_fdr_significant
)

print(
    "Families with >=1 significant feature:",
    families_with_significant
)


print()

print(
    "Temporal formal featurewise inference:",
    False
)

print(
    "Common-support FDR:",
    False
)

print(
    "Group-level FDR:",
    False
)

print(
    "Final explanation-drift conclusion:",
    False
)


print()

print(
    "Status:",
    (
        "FROZEN_FEATUREWISE_CROSS_CYCLE_"
        "INFERENCE_BH_FDR"
    )
)


print()

print(
    "Inference:",
    INFERENCE_OUTPUT_FILE
)

print(
    "Summary:",
    SUMMARY_OUTPUT_FILE
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
        "Chapter 10.10 — "
        "Final Cross-Cycle SHAP Freeze + Synthesis"
    )
)