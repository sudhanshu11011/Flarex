# ============================================================
# MODEL CHAPTER 10.9.1C
# FREEZE FEATUREWISE INFERENCE + BH-FDR PROTOCOL
#
# Purpose:
#   Lock the exact empirical-p and multiple-testing procedure
#   BEFORE Chapter 10.9.2 computes inferential results.
#
# Frozen Chapter-9 design already specifies:
#
#   - full 2000-replicate absolute-delta null distribution
#   - mean-|SHAP| absolute delta
#   - mean-signed-SHAP absolute delta
#   - empirical p-values computed later
#   - BH-FDR q = 0.05 later
#
# This file freezes the previously unspecified implementation:
#
#   empirical p =
#       (1 + count(null_abs_delta >= observed_abs_delta))
#       /
#       (R + 1)
#
#   with R = 2000.
#
# BH family:
#   candidate_id × stratum × quantity
#
# Formal strata:
#   OVERALL
#   TARGET_0
#   TARGET_1
#
# Each family contains the same 25 frozen SHARP features.
#
# Temporal B1/B2/B3 are EXCLUDED from formal featurewise
# inference because no temporal-block-specific / sample-size-
# matched featurewise Cycle24 null was frozen in Chapter 9.
#
# THIS STEP DOES NOT:
#   - compute observed empirical p-values
#   - perform BH adjustment
#   - label significant features
#   - make explanation-drift/stability conclusions
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

EXPECTED_MODELS = 3
EXPECTED_STRATA = 3
EXPECTED_FEATURES = 25
EXPECTED_QUANTITIES = 2

EXPECTED_BH_FAMILIES = (
    EXPECTED_MODELS
    *
    EXPECTED_STRATA
    *
    EXPECTED_QUANTITIES
)

EXPECTED_FORMAL_HYPOTHESES = (
    EXPECTED_BH_FAMILIES
    *
    EXPECTED_FEATURES
)

EXPECTED_NULL_ROWS = (
    EXPECTED_MODELS
    *
    EXPECTED_STRATA
    *
    N_NULL_REPLICATES
    *
    EXPECTED_FEATURES
)


FDR_Q = 0.05


MODEL_ORDER = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


STRATA = [
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


QUANTITIES = {

    "ABS_DELTA_MEAN_ABS_SHAP": {
        "observed_quantity":
            "absolute_delta_mean_absolute_shap",

        "null_quantity":
            "abs_delta_mean_abs_shap",

        "scientific_quantity":
            (
                "Absolute cross-cycle change in "
                "feature mean-|SHAP|"
            ),
    },

    "ABS_DELTA_MEAN_SIGNED_SHAP": {
        "observed_quantity":
            "absolute_delta_mean_signed_shap",

        "null_quantity":
            "abs_delta_mean_signed_shap",

        "scientific_quantity":
            (
                "Absolute cross-cycle change in "
                "feature mean signed SHAP"
            ),
    },
}


# ============================================================
# FROZEN CHAPTER-9 INPUTS
# ============================================================

CHAPTER9_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "chapter9_shift_null_controls_contract.json"
)


CHAPTER9_INVENTORY_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "chapter9_shift_null_controls_inventory.csv"
)


CYCLE24_NULL_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "cycle24_shap_null_contract.json"
)


FEATUREWISE_NULL_FILE = Path(
    "artifacts/models/shift_controls/shap/null/"
    "cycle24_shap_null_featurewise_replicates.parquet"
)


FEATUREWISE_THRESHOLD_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "cycle24_shap_null_featurewise_thresholds.csv"
)


# ============================================================
# FROZEN CHAPTER-10 OBSERVED SOURCES
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


CORRELATED_GROUP_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "correlated_group_shap_robustness_contract.json"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "featurewise_inference_protocol.json"
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


def normalized_path(
    path,
) -> str:

    return Path(
        str(path)
    ).as_posix()


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.9.1C — "
    "FREEZE FEATUREWISE INFERENCE + BH-FDR PROTOCOL"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    CHAPTER9_CONTRACT_FILE,
    CHAPTER9_INVENTORY_FILE,
    CYCLE24_NULL_CONTRACT_FILE,
    FEATUREWISE_NULL_FILE,
    FEATUREWISE_THRESHOLD_FILE,
    OVERALL_CONTRACT_FILE,
    CLASS_CONTRACT_FILE,
    TEMPORAL_CONTRACT_FILE,
    CORRELATED_GROUP_CONTRACT_FILE,
]


for path in required_files:

    verify_exists(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

if OUTPUT_CONTRACT_FILE.exists():

    raise RuntimeError(
        f"Output already exists: {OUTPUT_CONTRACT_FILE}\n"
        "Refusing to overwrite frozen "
        "Chapter 10.9.1C protocol."
    )


# ============================================================
# VERIFY CHAPTER 9 FREEZE
# ============================================================

chapter9_contract = load_json(
    CHAPTER9_CONTRACT_FILE
)


if (
    chapter9_contract.get(
        "status"
    )
    !=
    "FROZEN_CHAPTER9_SHIFT_NULL_CONTROLS"
):

    raise RuntimeError(
        "Chapter-9 shift/null controls "
        "are not frozen."
    )


print(
    "Chapter-9 shift/null controls:",
    "PASS"
)


# ============================================================
# VERIFY CYCLE24 NULL CONTRACT
# ============================================================

null_contract = load_json(
    CYCLE24_NULL_CONTRACT_FILE
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


featurewise_rule = null_contract.get(
    "featurewise_inference_rule"
)


if not isinstance(
    featurewise_rule,
    dict,
):

    raise RuntimeError(
        "Frozen Cycle24 null contract does not "
        "contain featurewise_inference_rule."
    )


if (
    featurewise_rule.get(
        "primary_future_null"
    )
    !=
    "FULL_2000_REPLICATE_ABSOLUTE_DELTA_DISTRIBUTION"
):

    raise RuntimeError(
        "Unexpected frozen featurewise null rule."
    )


if (
    featurewise_rule.get(
        "mean_abs_shap_quantity"
    )
    !=
    "abs_delta_mean_abs_shap"
):

    raise RuntimeError(
        "Unexpected frozen mean-|SHAP| quantity."
    )


if (
    featurewise_rule.get(
        "mean_signed_shap_quantity"
    )
    !=
    "abs_delta_mean_signed_shap"
):

    raise RuntimeError(
        "Unexpected frozen mean-signed-SHAP quantity."
    )


if (
    featurewise_rule.get(
        "empirical_p_values"
    )
    !=
    "COMPUTED_LATER_AGAINST_FULL_NULL_REPLICATES"
):

    raise RuntimeError(
        "Unexpected frozen empirical-p rule."
    )


if (
    featurewise_rule.get(
        "multiple_testing"
    )
    !=
    "BH_FDR_Q_0.05_LATER"
):

    raise RuntimeError(
        "Unexpected frozen multiple-testing rule."
    )


if (
    featurewise_rule.get(
        "bh_fdr_applied_in_9_6d"
    )
    is not False
):

    raise RuntimeError(
        "Chapter 9 unexpectedly already "
        "applied featurewise FDR."
    )


print(
    "Frozen Chapter-9 featurewise inference intent:",
    "PASS"
)


# ============================================================
# VERIFY CHAPTER-9 INVENTORY + HASHES
# ============================================================

chapter9_inventory = pd.read_csv(
    CHAPTER9_INVENTORY_FILE,
    keep_default_na=False,
)


inventory_hash_map = {

    normalized_path(
        row[
            "path"
        ]
    ):
        str(
            row[
                "sha256"
            ]
        )

    for _, row
    in chapter9_inventory.iterrows()
}


for artifact_path in [
    FEATUREWISE_NULL_FILE,
    FEATUREWISE_THRESHOLD_FILE,
    CYCLE24_NULL_CONTRACT_FILE,
]:

    key = normalized_path(
        artifact_path
    )


    if key not in inventory_hash_map:

        raise RuntimeError(
            "Required frozen Chapter-9 artifact "
            f"missing from inventory: {artifact_path}"
        )


    actual_hash = sha256_file(
        artifact_path
    )


    expected_hash = inventory_hash_map[
        key
    ]


    if actual_hash != expected_hash:

        raise RuntimeError(
            "Frozen Chapter-9 artifact changed: "
            f"{artifact_path}"
        )


print(
    "Frozen Chapter-9 artifact hashes:",
    "PASS"
)


# ============================================================
# VERIFY FULL FEATUREWISE NULL
# ============================================================

null_df = pd.read_parquet(
    FEATUREWISE_NULL_FILE
)


if len(
    null_df
) != EXPECTED_NULL_ROWS:

    raise RuntimeError(
        "Expected "
        f"{EXPECTED_NULL_ROWS} featurewise null rows, "
        f"got {len(null_df)}."
    )


required_null_columns = {
    "replicate_id",
    "candidate_id",
    "model_family",
    "stratum",
    "feature_index",
    "feature",
    "delta_mean_abs_shap",
    "abs_delta_mean_abs_shap",
    "delta_mean_signed_shap",
    "abs_delta_mean_signed_shap",
}


missing_columns = (
    required_null_columns
    -
    set(
        null_df.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "Missing featurewise null columns: "
        f"{sorted(missing_columns)}"
    )


if set(
    null_df[
        "candidate_id"
    ].astype(str).unique()
) != set(
    MODEL_ORDER
):

    raise RuntimeError(
        "Featurewise null model set mismatch."
    )


if set(
    null_df[
        "stratum"
    ].astype(str).unique()
) != set(
    STRATA
):

    raise RuntimeError(
        "Featurewise null stratum set mismatch."
    )


if set(
    null_df[
        "feature"
    ].astype(str).unique()
) != set(
    FEATURES
):

    raise RuntimeError(
        "Featurewise null feature set mismatch."
    )


if null_df[
    [
        "candidate_id",
        "stratum",
        "replicate_id",
        "feature",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate featurewise null replicate key."
    )


# ============================================================
# VERIFY 2000 REPLICATES PER MODEL × STRATUM × FEATURE
# ============================================================

replicate_counts = (
    null_df
    .groupby(
        [
            "candidate_id",
            "stratum",
            "feature",
        ]
    )[
        "replicate_id"
    ]
    .nunique()
)


if not (
    replicate_counts
    ==
    N_NULL_REPLICATES
).all():

    raise RuntimeError(
        "Every model/stratum/feature must "
        "contain exactly 2000 null replicates."
    )


for quantity in [
    "abs_delta_mean_abs_shap",
    "abs_delta_mean_signed_shap",
]:

    values = null_df[
        quantity
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"Non-finite frozen null values: {quantity}"
        )


    if (
        values
        <
        -1e-15
    ).any():

        raise RuntimeError(
            f"Negative absolute null values: {quantity}"
        )


print(
    "Full 2000-replicate featurewise null:",
    "PASS"
)


# ============================================================
# VERIFY THRESHOLD SUMMARY
# ============================================================

thresholds = pd.read_csv(
    FEATUREWISE_THRESHOLD_FILE,
    keep_default_na=False,
)


expected_threshold_rows = (
    EXPECTED_MODELS
    *
    EXPECTED_STRATA
    *
    EXPECTED_FEATURES
)


if len(
    thresholds
) != expected_threshold_rows:

    raise RuntimeError(
        "Featurewise threshold row-count mismatch."
    )


if (
    thresholds[
        "future_primary_featurewise_test"
    ].astype(str)
    !=
    "EMPIRICAL_ABSOLUTE_DELTA_AGAINST_FULL_NULL"
).all():

    raise RuntimeError(
        "Unexpected future featurewise test rule."
    )


if not (
    thresholds[
        "bh_fdr_applied_here"
    ].astype(str).str.lower()
    .isin(
        [
            "false",
            "0",
        ]
    )
).all():

    raise RuntimeError(
        "Featurewise thresholds indicate "
        "unexpected prior BH-FDR."
    )


print(
    "Frozen featurewise threshold summaries:",
    "PASS"
)


# ============================================================
# VERIFY REQUIRED CHAPTER-10 RESULTS EXIST
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
    CORRELATED_GROUP_CONTRACT_FILE
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
]


for actual, expected in required_statuses:

    if actual != expected:

        raise RuntimeError(
            "Required frozen Chapter-10 "
            f"contract missing: {expected}"
        )


print(
    "Required Chapter-10 observed analyses:",
    "PASS"
)


# ============================================================
# FREEZE BH FAMILY INVENTORY
# ============================================================

bh_family_rows = []


for candidate_id in MODEL_ORDER:

    for stratum in STRATA:

        for quantity_name in QUANTITIES:

            family_id = (
                f"{candidate_id}|"
                f"{stratum}|"
                f"{quantity_name}"
            )


            bh_family_rows.append(
                {
                    "family_id":
                        family_id,

                    "candidate_id":
                        candidate_id,

                    "stratum":
                        stratum,

                    "quantity":
                        quantity_name,

                    "tests":
                        EXPECTED_FEATURES,

                    "features":
                        FEATURES,
                }
            )


if len(
    bh_family_rows
) != EXPECTED_BH_FAMILIES:

    raise RuntimeError(
        "BH family inventory mismatch."
    )


# ============================================================
# FINAL PROTOCOL CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Featurewise Empirical-p "
            "and BH-FDR Protocol"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_FEATUREWISE_INFERENCE_PROTOCOL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_9_1C",

    "timing_disclosure":
        {
            "cycle25_shap_point_estimates_already_exist":
                True,

            "structure_null_screen_already_exists":
                True,

            "common_support_sensitivity_already_exists":
                True,

            "correlated_group_robustness_already_exists":
                True,

            "featurewise_empirical_p_values_computed_before_this_freeze":
                False,

            "featurewise_bh_fdr_computed_before_this_freeze":
                False,

            "interpretation":
                (
                    "This is an implementation freeze "
                    "before featurewise inferential "
                    "results, not a preregistration "
                    "before observing Cycle25."
                ),
        },

    "frozen_chapter9_basis":
        {
            "cycle24_null_contract":
                {
                    "file":
                        CYCLE24_NULL_CONTRACT_FILE.as_posix(),

                    "sha256":
                        sha256_file(
                            CYCLE24_NULL_CONTRACT_FILE
                        ),

                    "required_status":
                        "FROZEN_CYCLE24_SHAP_NULL_BASELINE",
                },

            "featurewise_null_file":
                {
                    "file":
                        FEATUREWISE_NULL_FILE.as_posix(),

                    "sha256":
                        sha256_file(
                            FEATUREWISE_NULL_FILE
                        ),

                    "rows":
                        EXPECTED_NULL_ROWS,

                    "replicates_per_model_stratum_feature":
                        N_NULL_REPLICATES,
                },

            "featurewise_threshold_file":
                {
                    "file":
                        FEATUREWISE_THRESHOLD_FILE.as_posix(),

                    "sha256":
                        sha256_file(
                            FEATUREWISE_THRESHOLD_FILE
                        ),

                    "rows":
                        expected_threshold_rows,
                },

            "frozen_future_test":
                (
                    "EMPIRICAL_ABSOLUTE_DELTA_"
                    "AGAINST_FULL_NULL"
                ),

            "frozen_multiple_testing":
                "BH_FDR_Q_0.05_LATER",
        },

    "formal_test_scope":
        {
            "models":
                MODEL_ORDER,

            "strata":
                STRATA,

            "features":
                FEATURES,

            "feature_count":
                EXPECTED_FEATURES,

            "quantities":
                QUANTITIES,

            "temporal_blocks_formally_tested":
                False,

            "temporal_exclusion_reason":
                (
                    "No frozen temporal-block-specific "
                    "or sample-size-matched featurewise "
                    "Cycle24 null distribution exists."
                ),
        },

    "empirical_p_value_rule":
        {
            "tail":
                "UPPER_TAIL_ON_ABSOLUTE_DELTA",

            "comparison":
                (
                    "NULL_ABSOLUTE_DELTA >= "
                    "OBSERVED_ABSOLUTE_DELTA"
                ),

            "monte_carlo_correction":
                "PLUS_ONE_NUMERATOR_AND_DENOMINATOR",

            "formula":
                (
                    "(1 + count(null_abs_delta >= "
                    "observed_abs_delta)) / "
                    "(N_NULL_REPLICATES + 1)"
                ),

            "n_null_replicates":
                N_NULL_REPLICATES,

            "denominator":
                N_NULL_REPLICATES + 1,

            "minimum_possible_p":
                (
                    1.0
                    /
                    (
                        N_NULL_REPLICATES
                        +
                        1
                    )
                ),

            "maximum_possible_p":
                1.0,

            "equality_counts_as_null_exceedance":
                True,

            "zero_empirical_p_allowed":
                False,
        },

    "multiple_testing_rule":
        {
            "method":
                "BENJAMINI_HOCHBERG",

            "fdr_q":
                FDR_Q,

            "family_definition":
                (
                    "CANDIDATE_ID_X_STRATUM_X_QUANTITY"
                ),

            "families":
                bh_family_rows,

            "family_count":
                EXPECTED_BH_FAMILIES,

            "tests_per_family":
                EXPECTED_FEATURES,

            "formal_hypothesis_count":
                EXPECTED_FORMAL_HYPOTHESES,

            "models_pooled_into_same_family":
                False,

            "strata_pooled_into_same_family":
                False,

            "mean_abs_and_mean_signed_pooled_into_same_family":
                False,
        },

    "bh_algorithm":
        {
            "input":
                "25 empirical p-values within one frozen family",

            "ordering":
                "ASCENDING_P_VALUE",

            "adjusted_p_rule":
                (
                    "q_i = min_{j>=i}("
                    "m * p_(j) / j), capped at 1"
                ),

            "m":
                EXPECTED_FEATURES,

            "significant_rule":
                "BH_ADJUSTED_P <= 0.05",

            "tie_handling":
                (
                    "Stable deterministic ordering; "
                    "ties retain identical inferential "
                    "meaning."
                ),
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

            "empirical_p_values_computed_here":
                False,

            "bh_fdr_applied_here":
                False,

            "feature_significance_labels_created_here":
                False,

            "temporal_featurewise_formal_inference":
                False,

            "final_explanation_drift_conclusion_made":
                False,

            "final_explanation_stability_conclusion_made":
                False,

            "shap_is_causal":
                False,
        },

    "interpretation_rule":
        (
            "Chapter 10.9.2 may call a feature "
            "FDR-significant only when its observed "
            "absolute cross-cycle SHAP delta is tested "
            "against the corresponding same-model, "
            "same-stratum, same-feature full 2000-"
            "replicate Cycle24 null distribution and "
            "its empirical p-value survives the frozen "
            "Benjamini-Hochberg family at q=0.05. "
            "Featurewise significance is evidence of "
            "attribution change, not causal solar "
            "physics and not by itself the final "
            "cross-cycle explanation-drift conclusion."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_9_2_"
            "FEATUREWISE_INFERENCE_AND_BH_FDR"
        ),
}


# ============================================================
# SAVE OUTPUT
# ============================================================

OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


temp_file = OUTPUT_CONTRACT_FILE.with_name(
    OUTPUT_CONTRACT_FILE.stem
    +
    ".tmp.json"
)


if temp_file.exists():

    temp_file.unlink()


with temp_file.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


roundtrip = load_json(
    temp_file
)


if (
    roundtrip.get(
        "status"
    )
    !=
    "FROZEN_FEATUREWISE_INFERENCE_PROTOCOL"
):

    raise RuntimeError(
        "10.9.1C contract round-trip failed."
    )


if (
    roundtrip[
        "scientific_guardrails"
    ][
        "empirical_p_values_computed_here"
    ]
    is not False
):

    raise RuntimeError(
        "Protocol freeze unexpectedly "
        "computed empirical p-values."
    )


if (
    roundtrip[
        "scientific_guardrails"
    ][
        "bh_fdr_applied_here"
    ]
    is not False
):

    raise RuntimeError(
        "Protocol freeze unexpectedly "
        "applied BH-FDR."
    )


if (
    roundtrip[
        "multiple_testing_rule"
    ][
        "family_count"
    ]
    !=
    EXPECTED_BH_FAMILIES
):

    raise RuntimeError(
        "Frozen BH family count mismatch."
    )


temp_file.replace(
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
    "MODEL CHAPTER 10.9.1C — COMPLETE"
)

print("=" * 78)


print(
    "Null replicates:",
    N_NULL_REPLICATES
)

print(
    "Formal models:",
    EXPECTED_MODELS
)

print(
    "Formal strata:",
    EXPECTED_STRATA
)

print(
    "Features / BH family:",
    EXPECTED_FEATURES
)

print(
    "Featurewise quantities:",
    EXPECTED_QUANTITIES
)


print()

print(
    "BH families:",
    EXPECTED_BH_FAMILIES
)

print(
    "Formal hypotheses:",
    EXPECTED_FORMAL_HYPOTHESES
)

print(
    "FDR q:",
    FDR_Q
)


print()

print(
    "Empirical p formula:"
)

print(
    "  (1 + count(null >= observed)) / 2001"
)


print()

print(
    "Formal temporal B1/B2/B3 inference:",
    False
)

print(
    "Empirical p-values computed:",
    False
)

print(
    "BH-FDR applied:",
    False
)

print(
    "Final drift conclusion:",
    False
)


print()

print(
    "Status:",
    "FROZEN_FEATUREWISE_INFERENCE_PROTOCOL"
)


print()

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
        "Chapter 10.9.2 — "
        "Featurewise Empirical Inference + BH-FDR"
    )
)