# ============================================================
# MODEL CHAPTER 10.6.1B
# OBSERVED CROSS-CYCLE SHAP STRUCTURE
# VS FROZEN WITHIN-CYCLE24 NULL THRESHOLDS
#
# Inputs:
#   - Overall pooled point estimates
#   - Temporal B1/B2/B3 point estimates
#   - Class-conditional TARGET_0/TARGET_1 point estimates
#   - Frozen Chapter-9 Cycle24 SHAP null thresholds
#
# Null mapping:
#
#   Overall pooled -> OVERALL null
#   B1/B2/B3       -> OVERALL null
#   TARGET_0       -> TARGET_0 null
#   TARGET_1       -> TARGET_1 null
#
# IMPORTANT:
#   Temporal blocks do NOT have a block-size-matched null.
#   Therefore temporal exceedance is a screening result,
#   not a final drift conclusion.
#
# THIS STEP DOES NOT:
#   - perform common-support sensitivity
#   - perform correlated-feature robustness
#   - perform featurewise inference
#   - apply FDR
#   - declare final explanation drift
#   - declare explanation stability
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
EXPECTED_METRICS = 8

EXPECTED_THRESHOLD_ROWS = 72

EXPECTED_OVERALL_ROWS = 24
EXPECTED_TEMPORAL_ROWS = 72
EXPECTED_CLASS_ROWS = 48

EXPECTED_COMPARISON_ROWS = (
    EXPECTED_OVERALL_ROWS
    +
    EXPECTED_TEMPORAL_ROWS
    +
    EXPECTED_CLASS_ROWS
)

EXPECTED_SUMMARY_ROWS = (
    3     # overall: 3 models
    +
    9     # temporal: 3 models x 3 blocks
    +
    6     # class: 3 models x 2 strata
)


MODEL_ORDER = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


METRIC_SET = {
    "spearman_rho",
    "kendall_tau_b",
    "top5_overlap",
    "top10_overlap",
    "signed_cosine_similarity",
    "sign_agreement",
    "normalized_l1_mean_abs",
    "cosine_distance_mean_abs",
}


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


NULL_THRESHOLD_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "cycle24_shap_null_structure_thresholds.csv"
)


# ============================================================
# FROZEN CHAPTER-10 OBSERVED INPUTS
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


# ============================================================
# OUTPUTS
# ============================================================

COMPARISON_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "cross_cycle_shap_structure_null_comparison.csv"
)


SUMMARY_OUTPUT_FILE = Path(
    "artifacts/models/cross_cycle_shap/metrics/"
    "cross_cycle_shap_structure_null_summary.csv"
)


OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "cross_cycle_shap_structure_null_comparison_contract.json"
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


def parse_bool(
    value,
) -> bool:

    if isinstance(
        value,
        (bool, np.bool_),
    ):

        return bool(
            value
        )


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
    }:

        return False


    raise RuntimeError(
        f"Cannot parse boolean value: {value}"
    )


def evaluate_threshold(
    observed: float,
    threshold: float,
    side_rule: str,
    strict: bool,
):

    if side_rule == "OBSERVED_BELOW_THRESHOLD":

        if strict:

            exceedance = (
                observed
                <
                threshold
            )

        else:

            exceedance = (
                observed
                <=
                threshold
            )


        signed_margin = (
            threshold
            -
            observed
        )


    elif side_rule == "OBSERVED_ABOVE_THRESHOLD":

        if strict:

            exceedance = (
                observed
                >
                threshold
            )

        else:

            exceedance = (
                observed
                >=
                threshold
            )


        signed_margin = (
            observed
            -
            threshold
        )


    else:

        raise RuntimeError(
            f"Unsupported null side rule: {side_rule}"
        )


    return (
        bool(
            exceedance
        ),
        float(
            signed_margin
        ),
    )


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.6.1B — "
    "OBSERVED SHAP STRUCTURE VS FROZEN CYCLE24 NULL"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

for path in [
    CHAPTER9_CONTRACT_FILE,
    CHAPTER9_INVENTORY_FILE,
    NULL_THRESHOLD_FILE,
    OVERALL_CONTRACT_FILE,
    TEMPORAL_CONTRACT_FILE,
    CLASS_CONTRACT_FILE,
]:

    verify_exists(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    COMPARISON_OUTPUT_FILE,
    SUMMARY_OUTPUT_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter 10.6.1B outputs."
        )


# ============================================================
# VERIFY CHAPTER-9 FREEZE
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
        "Chapter 9 shift/null controls "
        "are not frozen."
    )


chapter9_inventory_hash = sha256_file(
    CHAPTER9_INVENTORY_FILE
)


if (
    chapter9_inventory_hash
    !=
    chapter9_contract[
        "artifact_inventory"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Frozen Chapter-9 inventory changed."
    )


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


threshold_path_key = normalized_path(
    NULL_THRESHOLD_FILE
)


if (
    threshold_path_key
    not in
    inventory_hash_map
):

    raise RuntimeError(
        "Null threshold file is not present "
        "in the frozen Chapter-9 inventory."
    )


threshold_file_hash = sha256_file(
    NULL_THRESHOLD_FILE
)


if (
    threshold_file_hash
    !=
    inventory_hash_map[
        threshold_path_key
    ]
):

    raise RuntimeError(
        "Frozen Cycle24 null threshold "
        "artifact changed."
    )


print(
    "Frozen Chapter-9 null artifact:",
    "PASS"
)


# ============================================================
# LOAD + VERIFY NULL THRESHOLDS
# ============================================================

thresholds = pd.read_csv(
    NULL_THRESHOLD_FILE,
    keep_default_na=False,
)


if len(
    thresholds
) != EXPECTED_THRESHOLD_ROWS:

    raise RuntimeError(
        "Expected exactly 72 frozen "
        "null threshold rows."
    )


required_threshold_columns = {
    "candidate_id",
    "model_family",
    "stratum",
    "metric",
    "metric_type",
    "n_null_replicates",
    "threshold_type",
    "threshold_quantile",
    "threshold_value",
    "future_drift_side_rule",
    "comparison_is_strict",
}


missing_threshold_columns = (
    required_threshold_columns
    -
    set(
        thresholds.columns
    )
)


if missing_threshold_columns:

    raise RuntimeError(
        "Missing threshold columns: "
        f"{sorted(missing_threshold_columns)}"
    )


if thresholds[
    [
        "candidate_id",
        "stratum",
        "metric",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate null threshold keys detected."
    )


if set(
    thresholds[
        "candidate_id"
    ].astype(str).unique()
) != set(
    MODEL_ORDER
):

    raise RuntimeError(
        "Null threshold model set mismatch."
    )


if set(
    thresholds[
        "stratum"
    ].astype(str).unique()
) != {
    "OVERALL",
    "TARGET_0",
    "TARGET_1",
}:

    raise RuntimeError(
        "Null threshold stratum set mismatch."
    )


if set(
    thresholds[
        "metric"
    ].astype(str).unique()
) != METRIC_SET:

    raise RuntimeError(
        "Null threshold metric set mismatch."
    )


if not (
    thresholds[
        "n_null_replicates"
    ].astype(int)
    ==
    2000
).all():

    raise RuntimeError(
        "Expected exactly 2000 null replicates "
        "for every threshold."
    )


# ============================================================
# VERIFY FROZEN DIRECTION RULES
# ============================================================

for _, row in thresholds.iterrows():

    metric = str(
        row[
            "metric"
        ]
    )

    metric_type = str(
        row[
            "metric_type"
        ]
    )

    threshold_type = str(
        row[
            "threshold_type"
        ]
    )

    quantile = float(
        row[
            "threshold_quantile"
        ]
    )

    side_rule = str(
        row[
            "future_drift_side_rule"
        ]
    )

    strict = parse_bool(
        row[
            "comparison_is_strict"
        ]
    )


    if not strict:

        raise RuntimeError(
            "Frozen null comparison is expected "
            "to be strict for all metrics."
        )


    if metric in SIMILARITY_METRICS:

        if metric_type != "SIMILARITY":

            raise RuntimeError(
                f"{metric}: expected SIMILARITY."
            )


        if threshold_type != "LOWER_2P5_SIMILARITY":

            raise RuntimeError(
                f"{metric}: wrong threshold type."
            )


        if not np.isclose(
            quantile,
            0.025,
            atol=0.0,
            rtol=0.0,
        ):

            raise RuntimeError(
                f"{metric}: wrong similarity quantile."
            )


        if side_rule != "OBSERVED_BELOW_THRESHOLD":

            raise RuntimeError(
                f"{metric}: wrong similarity side rule."
            )


    elif metric in DISTANCE_METRICS:

        if metric_type != "DISTANCE":

            raise RuntimeError(
                f"{metric}: expected DISTANCE."
            )


        if threshold_type != "UPPER_97P5_DISTANCE":

            raise RuntimeError(
                f"{metric}: wrong threshold type."
            )


        if not np.isclose(
            quantile,
            0.975,
            atol=0.0,
            rtol=0.0,
        ):

            raise RuntimeError(
                f"{metric}: wrong distance quantile."
            )


        if side_rule != "OBSERVED_ABOVE_THRESHOLD":

            raise RuntimeError(
                f"{metric}: wrong distance side rule."
            )


    else:

        raise RuntimeError(
            f"Unknown metric: {metric}"
        )


print(
    "Frozen null threshold schema/rules:",
    "PASS"
)


# ============================================================
# LOAD FROZEN OBSERVED POINT-ESTIMATE CONTRACTS
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


if (
    overall_contract.get(
        "status"
    )
    !=
    "FROZEN_OVERALL_CROSS_CYCLE_SHAP_POINT_ESTIMATES"
):

    raise RuntimeError(
        "Overall point estimates are not frozen."
    )


if (
    temporal_contract.get(
        "status"
    )
    !=
    "FROZEN_TEMPORAL_BLOCK_CROSS_CYCLE_SHAP_POINT_ESTIMATES"
):

    raise RuntimeError(
        "Temporal point estimates are not frozen."
    )


if (
    class_contract.get(
        "status"
    )
    !=
    "FROZEN_CLASS_CONDITIONAL_CROSS_CYCLE_SHAP_POINT_ESTIMATES"
):

    raise RuntimeError(
        "Class-conditional point estimates "
        "are not frozen."
    )


# ============================================================
# OBSERVED FILE PATHS FROM CONTRACTS
# ============================================================

overall_file = Path(
    overall_contract[
        "structure_output"
    ][
        "file"
    ]
)


temporal_file = Path(
    temporal_contract[
        "structure_output"
    ][
        "file"
    ]
)


class_file = Path(
    class_contract[
        "structure_output"
    ][
        "file"
    ]
)


for path in [
    overall_file,
    temporal_file,
    class_file,
]:

    verify_exists(
        path
    )


if (
    sha256_file(
        overall_file
    )
    !=
    overall_contract[
        "structure_output"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Frozen overall structure estimates changed."
    )


if (
    sha256_file(
        temporal_file
    )
    !=
    temporal_contract[
        "structure_output"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Frozen temporal structure estimates changed."
    )


if (
    sha256_file(
        class_file
    )
    !=
    class_contract[
        "structure_output"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Frozen class structure estimates changed."
    )


print(
    "Frozen observed structure artifacts:",
    "PASS"
)


# ============================================================
# LOAD OBSERVED TABLES
# ============================================================

overall = pd.read_csv(
    overall_file,
    keep_default_na=False,
)


temporal = pd.read_csv(
    temporal_file,
    keep_default_na=False,
)


class_conditional = pd.read_csv(
    class_file,
    keep_default_na=False,
)


if len(
    overall
) != EXPECTED_OVERALL_ROWS:

    raise RuntimeError(
        "Overall observed row-count mismatch."
    )


if len(
    temporal
) != EXPECTED_TEMPORAL_ROWS:

    raise RuntimeError(
        "Temporal observed row-count mismatch."
    )


if len(
    class_conditional
) != EXPECTED_CLASS_ROWS:

    raise RuntimeError(
        "Class observed row-count mismatch."
    )


# ============================================================
# VERIFY OBSERVED METRIC SETS
# ============================================================

for name, frame in [
    (
        "overall",
        overall,
    ),
    (
        "temporal",
        temporal,
    ),
    (
        "class_conditional",
        class_conditional,
    ),
]:

    if set(
        frame[
            "metric"
        ].astype(str).unique()
    ) != METRIC_SET:

        raise RuntimeError(
            f"{name}: observed metric set mismatch."
        )


print(
    "Observed metric inventories:",
    "PASS"
)


# ============================================================
# THRESHOLD LOOKUP
# ============================================================

threshold_lookup = {}


for _, row in thresholds.iterrows():

    key = (
        str(
            row[
                "candidate_id"
            ]
        ),
        str(
            row[
                "stratum"
            ]
        ),
        str(
            row[
                "metric"
            ]
        ),
    )


    threshold_lookup[
        key
    ] = row


# ============================================================
# RESULT CONTAINERS
# ============================================================

comparison_rows = []


# ============================================================
# COMMON PROCESSOR
# ============================================================

def add_comparison(
    *,
    analysis_scope: str,
    candidate_id: str,
    model_family: str,
    transfer_population: str,
    observed_stratum: str,
    null_stratum: str,
    metric: str,
    observed_value: float,
    cycle25_block: str = "",
    temporal_null_is_sample_size_matched: bool | None = None,
):

    threshold_key = (
        candidate_id,
        null_stratum,
        metric,
    )


    if (
        threshold_key
        not in
        threshold_lookup
    ):

        raise RuntimeError(
            "Missing frozen null threshold for "
            f"{threshold_key}"
        )


    threshold_row = threshold_lookup[
        threshold_key
    ]


    threshold_value = float(
        threshold_row[
            "threshold_value"
        ]
    )


    side_rule = str(
        threshold_row[
            "future_drift_side_rule"
        ]
    )


    strict = parse_bool(
        threshold_row[
            "comparison_is_strict"
        ]
    )


    exceedance, signed_margin = (
        evaluate_threshold(
            observed=observed_value,
            threshold=threshold_value,
            side_rule=side_rule,
            strict=strict,
        )
    )


    comparison_id = (
        f"{candidate_id}|"
        f"{analysis_scope}|"
        f"{transfer_population}|"
        f"{observed_stratum}"
    )


    if exceedance:

        null_screen_status = (
            "EXCEEDS_FROZEN_CYCLE24_NULL"
        )

    else:

        null_screen_status = (
            "DOES_NOT_EXCEED_FROZEN_CYCLE24_NULL"
        )


    comparison_rows.append(
        {
            "comparison_id":
                comparison_id,

            "analysis_scope":
                analysis_scope,

            "candidate_id":
                candidate_id,

            "model_family":
                model_family,

            "reference_population":
                "CYCLE24_HOLDOUT",

            "transfer_population":
                transfer_population,

            "cycle25_block":
                cycle25_block,

            "observed_stratum":
                observed_stratum,

            "null_stratum":
                null_stratum,

            "metric":
                metric,

            "metric_type":
                str(
                    threshold_row[
                        "metric_type"
                    ]
                ),

            "observed_value":
                observed_value,

            "threshold_type":
                str(
                    threshold_row[
                        "threshold_type"
                    ]
                ),

            "threshold_quantile":
                float(
                    threshold_row[
                        "threshold_quantile"
                    ]
                ),

            "threshold_value":
                threshold_value,

            "future_drift_side_rule":
                side_rule,

            "comparison_is_strict":
                strict,

            "signed_margin_beyond_threshold":
                signed_margin,

            "null_exceedance":
                exceedance,

            "null_screen_status":
                null_screen_status,

            "n_null_replicates":
                int(
                    threshold_row[
                        "n_null_replicates"
                    ]
                ),

            "temporal_null_is_sample_size_matched":
                (
                    temporal_null_is_sample_size_matched
                ),

            "final_drift_conclusion":
                "NOT_MADE",

            "final_stability_conclusion":
                "NOT_MADE",
        }
    )


# ============================================================
# 1. OVERALL POOLED
# ============================================================

for _, row in overall.iterrows():

    candidate_id = str(
        row[
            "candidate_id"
        ]
    )


    observed_stratum = str(
        row[
            "stratum"
        ]
    )


    if observed_stratum != "OVERALL":

        raise RuntimeError(
            "Overall table contains "
            "non-OVERALL stratum."
        )


    add_comparison(
        analysis_scope="OVERALL_POOLED",
        candidate_id=candidate_id,
        model_family=str(
            row[
                "model_family"
            ]
        ),
        transfer_population="CYCLE25_POOLED",
        observed_stratum="OVERALL",
        null_stratum="OVERALL",
        metric=str(
            row[
                "metric"
            ]
        ),
        observed_value=float(
            row[
                "metric_value"
            ]
        ),
        temporal_null_is_sample_size_matched=None,
    )


# ============================================================
# 2. TEMPORAL BLOCKS
#
# Frozen Chapter-9 null has no block-specific stratum.
# Each class-mixed temporal block is screened against
# the OVERALL Cycle24 null.
#
# This is NOT sample-size-matched.
# ============================================================

for _, row in temporal.iterrows():

    observed_stratum = str(
        row[
            "stratum"
        ]
    )


    if observed_stratum != "OVERALL":

        raise RuntimeError(
            "Temporal table contains "
            "non-OVERALL stratum."
        )


    block = str(
        row[
            "cycle25_block"
        ]
    )


    if block not in {
        "B1",
        "B2",
        "B3",
    }:

        raise RuntimeError(
            f"Unexpected temporal block: {block}"
        )


    add_comparison(
        analysis_scope="TEMPORAL_BLOCK",
        candidate_id=str(
            row[
                "candidate_id"
            ]
        ),
        model_family=str(
            row[
                "model_family"
            ]
        ),
        transfer_population=str(
            row[
                "transfer_population"
            ]
        ),
        cycle25_block=block,
        observed_stratum="OVERALL",
        null_stratum="OVERALL",
        metric=str(
            row[
                "metric"
            ]
        ),
        observed_value=float(
            row[
                "metric_value"
            ]
        ),
        temporal_null_is_sample_size_matched=False,
    )


# ============================================================
# 3. CLASS-CONDITIONAL
# ============================================================

for _, row in class_conditional.iterrows():

    observed_stratum = str(
        row[
            "stratum"
        ]
    )


    if observed_stratum not in {
        "TARGET_0",
        "TARGET_1",
    }:

        raise RuntimeError(
            "Unexpected class stratum: "
            f"{observed_stratum}"
        )


    add_comparison(
        analysis_scope="CLASS_CONDITIONAL",
        candidate_id=str(
            row[
                "candidate_id"
            ]
        ),
        model_family=str(
            row[
                "model_family"
            ]
        ),
        transfer_population="CYCLE25_POOLED",
        observed_stratum=observed_stratum,
        null_stratum=observed_stratum,
        metric=str(
            row[
                "metric"
            ]
        ),
        observed_value=float(
            row[
                "metric_value"
            ]
        ),
        temporal_null_is_sample_size_matched=None,
    )


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    comparison_rows
)


if len(
    comparison
) != EXPECTED_COMPARISON_ROWS:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_COMPARISON_ROWS} null-comparison "
        f"rows, got {len(comparison)}."
    )


if comparison[
    [
        "comparison_id",
        "metric",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate comparison/metric rows detected."
    )


# ============================================================
# VERIFY 8 METRICS PER COMPARISON UNIT
# ============================================================

metric_counts = (
    comparison
    .groupby(
        "comparison_id"
    )[
        "metric"
    ]
    .nunique()
)


if not (
    metric_counts
    ==
    EXPECTED_METRICS
).all():

    raise RuntimeError(
        "Every comparison unit must contain "
        "exactly 8 unique structure metrics."
    )


print(
    "Observed-vs-null comparison inventory:",
    "PASS"
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_rows = []


for comparison_id, group in comparison.groupby(
    "comparison_id",
    sort=False,
):

    exceeded = group[
        group[
            "null_exceedance"
        ]
        ==
        True
    ]


    exceeded_metrics = (
        exceeded[
            "metric"
        ]
        .astype(str)
        .tolist()
    )


    n_exceeded = len(
        exceeded_metrics
    )


    if n_exceeded > 0:

        null_screen_status = (
            "NULL_EXCEEDANCE_PRESENT_PENDING_ROBUSTNESS"
        )

    else:

        null_screen_status = (
            "NO_STRUCTURE_NULL_EXCEEDANCE_DETECTED"
        )


    first = group.iloc[
        0
    ]


    summary_rows.append(
        {
            "comparison_id":
                comparison_id,

            "analysis_scope":
                first[
                    "analysis_scope"
                ],

            "candidate_id":
                first[
                    "candidate_id"
                ],

            "model_family":
                first[
                    "model_family"
                ],

            "reference_population":
                first[
                    "reference_population"
                ],

            "transfer_population":
                first[
                    "transfer_population"
                ],

            "cycle25_block":
                first[
                    "cycle25_block"
                ],

            "observed_stratum":
                first[
                    "observed_stratum"
                ],

            "null_stratum":
                first[
                    "null_stratum"
                ],

            "metrics_evaluated":
                EXPECTED_METRICS,

            "metrics_exceeding_null":
                n_exceeded,

            "exceeded_metrics":
                ";".join(
                    exceeded_metrics
                ),

            "any_null_exceedance":
                (
                    n_exceeded
                    >
                    0
                ),

            "all_structure_metrics_within_null_threshold":
                (
                    n_exceeded
                    ==
                    0
                ),

            "null_screen_status":
                null_screen_status,

            "temporal_null_is_sample_size_matched":
                first[
                    "temporal_null_is_sample_size_matched"
                ],

            "final_drift_conclusion":
                "NOT_MADE",

            "final_stability_conclusion":
                "NOT_MADE",
        }
    )


summary = pd.DataFrame(
    summary_rows
)


if len(
    summary
) != EXPECTED_SUMMARY_ROWS:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_SUMMARY_ROWS} summary rows, "
        f"got {len(summary)}."
    )


# ============================================================
# PRINT SCREEN RESULTS
# ============================================================

print()

print("=" * 78)

print(
    "NULL SCREEN SUMMARY"
)

print("=" * 78)


for _, row in summary.iterrows():

    print()

    print(
        f"{row['analysis_scope']} | "
        f"{row['candidate_id']} | "
        f"{row['transfer_population']} | "
        f"{row['observed_stratum']}"
    )


    print(
        "Metrics exceeding frozen null:",
        int(
            row[
                "metrics_exceeding_null"
            ]
        ),
        "/",
        EXPECTED_METRICS,
    )


    exceeded_text = str(
        row[
            "exceeded_metrics"
        ]
    )


    if exceeded_text:

        print(
            "Exceeded metrics:",
            exceeded_text
        )

    else:

        print(
            "Exceeded metrics:",
            "NONE"
        )


    print(
        "Screen status:",
        row[
            "null_screen_status"
        ]
    )


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

COMPARISON_OUTPUT_FILE.parent.mkdir(
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

comparison_temp = COMPARISON_OUTPUT_FILE.with_name(
    COMPARISON_OUTPUT_FILE.stem
    +
    ".tmp.csv"
)


summary_temp = SUMMARY_OUTPUT_FILE.with_name(
    SUMMARY_OUTPUT_FILE.stem
    +
    ".tmp.csv"
)


for path in [
    comparison_temp,
    summary_temp,
]:

    if path.exists():

        path.unlink()


comparison.to_csv(
    comparison_temp,
    index=False,
)


summary.to_csv(
    summary_temp,
    index=False,
)


# ============================================================
# ROUND-TRIP VALIDATION
# ============================================================

comparison_rt = pd.read_csv(
    comparison_temp,
    keep_default_na=False,
)


summary_rt = pd.read_csv(
    summary_temp,
    keep_default_na=False,
)


if len(
    comparison_rt
) != EXPECTED_COMPARISON_ROWS:

    raise RuntimeError(
        "Null comparison CSV round-trip failed."
    )


if len(
    summary_rt
) != EXPECTED_SUMMARY_ROWS:

    raise RuntimeError(
        "Null summary CSV round-trip failed."
    )


comparison_temp.replace(
    COMPARISON_OUTPUT_FILE
)


summary_temp.replace(
    SUMMARY_OUTPUT_FILE
)


comparison_hash = sha256_file(
    COMPARISON_OUTPUT_FILE
)


summary_hash = sha256_file(
    SUMMARY_OUTPUT_FILE
)


print()

print(
    "Null comparison CSV round-trip:",
    "PASS"
)

print(
    "Null summary CSV round-trip:",
    "PASS"
)


# ============================================================
# FINAL CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Observed Cross-Cycle SHAP Structure "
            "vs Frozen Cycle24 Null"
        ),

    "version":
        "1.0",

    "status":
        (
            "FROZEN_CROSS_CYCLE_SHAP_"
            "STRUCTURE_NULL_COMPARISON"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_6_1B",

    "null_source":
        {
            "threshold_file":
                NULL_THRESHOLD_FILE.as_posix(),

            "threshold_sha256":
                threshold_file_hash,

            "rows":
                EXPECTED_THRESHOLD_ROWS,

            "replicates_per_model_stratum_metric":
                2000,

            "comparison_is_strict":
                True,

            "similarity_rule":
                "OBSERVED_BELOW_LOWER_2P5_THRESHOLD",

            "distance_rule":
                "OBSERVED_ABOVE_UPPER_97P5_THRESHOLD",
        },

    "observed_sources":
        {
            "overall":
                {
                    "contract":
                        OVERALL_CONTRACT_FILE.as_posix(),

                    "contract_sha256":
                        sha256_file(
                            OVERALL_CONTRACT_FILE
                        ),

                    "structure_file":
                        overall_file.as_posix(),

                    "structure_sha256":
                        sha256_file(
                            overall_file
                        ),
                },

            "temporal":
                {
                    "contract":
                        TEMPORAL_CONTRACT_FILE.as_posix(),

                    "contract_sha256":
                        sha256_file(
                            TEMPORAL_CONTRACT_FILE
                        ),

                    "structure_file":
                        temporal_file.as_posix(),

                    "structure_sha256":
                        sha256_file(
                            temporal_file
                        ),
                },

            "class_conditional":
                {
                    "contract":
                        CLASS_CONTRACT_FILE.as_posix(),

                    "contract_sha256":
                        sha256_file(
                            CLASS_CONTRACT_FILE
                        ),

                    "structure_file":
                        class_file.as_posix(),

                    "structure_sha256":
                        sha256_file(
                            class_file
                        ),
                },
        },

    "null_mapping":
        {
            "overall_pooled":
                "OVERALL",

            "temporal_B1":
                "OVERALL",

            "temporal_B2":
                "OVERALL",

            "temporal_B3":
                "OVERALL",

            "class_TARGET_0":
                "TARGET_0",

            "class_TARGET_1":
                "TARGET_1",
        },

    "temporal_null_caveat":
        {
            "block_specific_null_available":
                False,

            "temporal_blocks_use_overall_null":
                True,

            "sample_size_matched":
                False,

            "interpretation":
                (
                    "Temporal block null exceedance is "
                    "a screening result because the frozen "
                    "Cycle24 null was not constructed to "
                    "match each Cycle25 block sample size."
                ),
        },

    "comparison_output":
        {
            "file":
                COMPARISON_OUTPUT_FILE.as_posix(),

            "sha256":
                comparison_hash,

            "rows":
                EXPECTED_COMPARISON_ROWS,
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

            "background_changed":
                False,

            "cycle24_null_changed":
                False,

            "frozen_null_thresholds_applied":
                True,

            "null_exceedance_evaluated":
                True,

            "temporal_block_specific_null_used":
                False,

            "featurewise_inference_performed":
                False,

            "fdr_applied":
                False,

            "common_support_sensitivity_applied":
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
            "A structure metric crossing its frozen "
            "within-Cycle24 threshold indicates that the "
            "observed cross-cycle change exceeds the "
            "corresponding pre-frozen null threshold for "
            "that metric. This is not by itself a final "
            "explanation-drift conclusion. Conversely, "
            "failure to cross the threshold is not proof "
            "of explanation stability."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_7_"
            "COMMON_SUPPORT_REWEIGHTING_SENSITIVITY"
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
    "FROZEN_CROSS_CYCLE_SHAP_STRUCTURE_NULL_COMPARISON"
):

    raise RuntimeError(
        "10.6.1B contract round-trip failed."
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
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 10.6.1B — COMPLETE"
)

print("=" * 78)


print(
    "Frozen null threshold rows:",
    EXPECTED_THRESHOLD_ROWS
)

print(
    "Observed comparison rows:",
    len(
        comparison
    )
)

print(
    "Comparison summary rows:",
    len(
        summary
    )
)


print()

print(
    "Strict threshold comparison:",
    True
)

print(
    "Overall pooled screened:",
    True
)

print(
    "Temporal blocks screened:",
    True
)

print(
    "Class-conditioned screened:",
    True
)


print()

print(
    "Temporal block-specific null:",
    False
)

print(
    "Temporal sample-size-matched null:",
    False
)


print()

print(
    "Common-support sensitivity applied:",
    False
)

print(
    "Correlated-group robustness applied:",
    False
)

print(
    "Featurewise inference / FDR:",
    False
)

print(
    "Final drift conclusion made:",
    False
)

print(
    "Final stability conclusion made:",
    False
)


print()

print(
    "Status:",
    (
        "FROZEN_CROSS_CYCLE_SHAP_"
        "STRUCTURE_NULL_COMPARISON"
    )
)


print()

print(
    "Comparison:",
    COMPARISON_OUTPUT_FILE
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
        "Chapter 10.7 — "
        "Common-Support / Reweighting Sensitivity"
    )
)