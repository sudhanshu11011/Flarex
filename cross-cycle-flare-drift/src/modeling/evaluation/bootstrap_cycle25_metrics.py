from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)


# ============================================================
# PATHS
# ============================================================

PREDICTION_FILE = Path(
    "artifacts/models/evaluation/predictions/"
    "cycle25_predictions.parquet"
)

PREDICTION_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_prediction_contract.json"
)

CYCLE25_PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_evaluation_protocol.json"
)

POINT_METRICS_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "cycle25_metrics.csv"
)

POINT_METRICS_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_metrics_contract.json"
)

MODEL_REGISTRY_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_frozen_model_registry.csv"
)

OUTPUT_REPLICATES_FILE = Path(
    "artifacts/models/evaluation/bootstrap/"
    "cycle25_group_bootstrap_replicates.parquet"
)

OUTPUT_CI_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "cycle25_bootstrap_ci.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_bootstrap_contract.json"
)


# ============================================================
# FROZEN SCOPES
# ============================================================

PRIMARY_BLOCKS = [
    "C25_B1_2020_2021",
    "C25_B2_2022_2023",
    "C25_B3_2024_2025",
]

POOLED_SCOPE = "C25_ALL_2020_2025"

ALL_SCOPES = (
    PRIMARY_BLOCKS
    +
    [POOLED_SCOPE]
)


# ============================================================
# FROZEN BOOTSTRAP SETTINGS
# ============================================================

VALID_REPLICATES_REQUIRED = 2000

CI_LOWER_QUANTILE = 0.025
CI_UPPER_QUANTILE = 0.975

SCOPE_SEEDS = {
    "C25_B1_2020_2021": 25025,
    "C25_B2_2022_2023": 25026,
    "C25_B3_2024_2025": 25027,
    "C25_ALL_2020_2025": 25028,
}


BOOTSTRAP_METRICS = [
    "tss",
    "hss",
    "f1",
    "sensitivity",
    "specificity",
    "precision",
    "pr_auc",
    "roc_auc",
    "brier_score",
]


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

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def calculate_hss(
    tp,
    fp,
    tn,
    fn,
):

    numerator = 2.0 * (
        tp * tn
        -
        fn * fp
    )

    denominator = (
        (tp + fn)
        *
        (fn + tn)
        +
        (tp + fp)
        *
        (fp + tn)
    )

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def calculate_f1(
    tp,
    fp,
    fn,
):

    denominator = (
        2.0 * tp
        +
        fp
        +
        fn
    )

    if denominator == 0:
        return 0.0

    return float(
        (2.0 * tp)
        /
        denominator
    )


def compute_weighted_metrics(
    y_true,
    probabilities,
    predictions,
    sample_weights,
):

    weights = np.asarray(
        sample_weights,
        dtype=np.float64,
    )

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )


    weighted_positive = float(
        np.sum(
            weights
            *
            (y_true == 1)
        )
    )


    weighted_negative = float(
        np.sum(
            weights
            *
            (y_true == 0)
        )
    )


    if weighted_positive <= 0:

        raise RuntimeError(
            "Bootstrap sample contains "
            "no positive target weight."
        )


    if weighted_negative <= 0:

        raise RuntimeError(
            "Bootstrap sample contains "
            "no negative target weight."
        )


    tp = float(
        np.sum(
            weights
            *
            (
                (predictions == 1)
                &
                (y_true == 1)
            )
        )
    )


    fp = float(
        np.sum(
            weights
            *
            (
                (predictions == 1)
                &
                (y_true == 0)
            )
        )
    )


    tn = float(
        np.sum(
            weights
            *
            (
                (predictions == 0)
                &
                (y_true == 0)
            )
        )
    )


    fn = float(
        np.sum(
            weights
            *
            (
                (predictions == 0)
                &
                (y_true == 1)
            )
        )
    )


    sensitivity = safe_divide(
        tp,
        tp + fn,
    )


    fpr = safe_divide(
        fp,
        fp + tn,
    )


    specificity = safe_divide(
        tn,
        tn + fp,
    )


    precision = safe_divide(
        tp,
        tp + fp,
    )


    tss = float(
        sensitivity - fpr
    )


    hss = calculate_hss(
        tp,
        fp,
        tn,
        fn,
    )


    f1 = calculate_f1(
        tp,
        fp,
        fn,
    )


    pr_auc = float(
        average_precision_score(
            y_true,
            probabilities,
            sample_weight=weights,
        )
    )


    roc_auc = float(
        roc_auc_score(
            y_true,
            probabilities,
            sample_weight=weights,
        )
    )


    brier = float(
        brier_score_loss(
            y_true,
            probabilities,
            sample_weight=weights,
        )
    )


    return {
        "tss": tss,
        "hss": hss,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
    }


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 8.4 — CYCLE25 ENTITY_GROUP BOOTSTRAP"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    PREDICTION_FILE,
    PREDICTION_CONTRACT_FILE,
    CYCLE25_PROTOCOL_FILE,
    POINT_METRICS_FILE,
    POINT_METRICS_CONTRACT_FILE,
    MODEL_REGISTRY_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# PROTECT OUTPUTS
# ============================================================

for path in [
    OUTPUT_REPLICATES_FILE,
    OUTPUT_CI_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Cycle25 bootstrap results."
        )


# ============================================================
# CONTRACT VERIFICATION
# ============================================================

prediction_contract = load_json(
    PREDICTION_CONTRACT_FILE
)

protocol = load_json(
    CYCLE25_PROTOCOL_FILE
)

metrics_contract = load_json(
    POINT_METRICS_CONTRACT_FILE
)


if prediction_contract.get(
    "status"
) != "FROZEN_CYCLE25_PREDICTIONS":

    raise RuntimeError(
        "Cycle25 prediction artifact "
        "is not frozen."
    )


if protocol.get(
    "status"
) != "FROZEN_BEFORE_CYCLE25_EVALUATION":

    raise RuntimeError(
        "Cycle25 evaluation protocol "
        "is not frozen."
    )


if metrics_contract.get(
    "status"
) != "FROZEN_CYCLE25_POINT_METRICS":

    raise RuntimeError(
        "Cycle25 point metrics "
        "are not frozen."
    )


print(
    "Cycle25 predictions:",
    "FROZEN"
)

print(
    "Cycle25 evaluation protocol:",
    "FROZEN"
)

print(
    "Cycle25 point metrics:",
    "FROZEN"
)


# ============================================================
# VERIFY PROTOCOL BOOTSTRAP SETTINGS
# ============================================================

bootstrap_protocol = protocol[
    "bootstrap"
]


if bootstrap_protocol[
    "unit"
] != "ENTITY_GROUP":

    raise RuntimeError(
        "Frozen bootstrap unit mismatch."
    )


if int(
    bootstrap_protocol[
        "valid_replicates_per_evaluation"
    ]
) != VALID_REPLICATES_REQUIRED:

    raise RuntimeError(
        "Frozen bootstrap replicate "
        "count mismatch."
    )


if bootstrap_protocol[
    "ci_method"
] != "percentile":

    raise RuntimeError(
        "Frozen bootstrap CI method mismatch."
    )


if float(
    bootstrap_protocol[
        "lower_quantile"
    ]
) != CI_LOWER_QUANTILE:

    raise RuntimeError(
        "Frozen lower quantile mismatch."
    )


if float(
    bootstrap_protocol[
        "upper_quantile"
    ]
) != CI_UPPER_QUANTILE:

    raise RuntimeError(
        "Frozen upper quantile mismatch."
    )


protocol_seeds = bootstrap_protocol[
    "block_specific_seeds"
]


for scope_id, expected_seed in SCOPE_SEEDS.items():

    if int(
        protocol_seeds[
            scope_id
        ]
    ) != expected_seed:

        raise RuntimeError(
            f"{scope_id}: frozen seed mismatch."
        )


print(
    "Bootstrap unit:",
    "ENTITY_GROUP"
)

print(
    "Valid replicates/scope:",
    VALID_REPLICATES_REQUIRED
)

print(
    "CI method:",
    "95% percentile"
)

print(
    "Scope-specific seeds:",
    "PASS"
)


# ============================================================
# HASH VERIFICATION
# ============================================================

prediction_hash = sha256_file(
    PREDICTION_FILE
)


if (
    prediction_hash
    !=
    prediction_contract[
        "artifacts"
    ][
        "prediction_artifact"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle25 prediction artifact "
        "hash mismatch."
    )


metrics_hash = sha256_file(
    POINT_METRICS_FILE
)


if (
    metrics_hash
    !=
    metrics_contract[
        "artifacts"
    ][
        "cycle25_metrics"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle25 point metrics "
        "hash mismatch."
    )


print(
    "Prediction artifact hash:",
    "PASS"
)

print(
    "Point metrics artifact hash:",
    "PASS"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

predictions = pd.read_parquet(
    PREDICTION_FILE
)


point_metrics = pd.read_csv(
    POINT_METRICS_FILE,
    keep_default_na=False,
)


registry = pd.read_csv(
    MODEL_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    registry
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen models."
    )


if len(
    point_metrics
) != 12:

    raise RuntimeError(
        "Expected exactly 12 Cycle25 "
        "point metric rows."
    )


# ============================================================
# MODEL INFO
# ============================================================

model_info = []


for _, row in registry.iterrows():

    family = str(
        row[
            "family"
        ]
    )


    candidate_id = str(
        row[
            "candidate_id"
        ]
    )


    threshold = float(
        row[
            "frozen_threshold"
        ]
    )


    probability_column = (
        f"{candidate_id}"
        "__probability"
    )


    prediction_column = (
        f"{candidate_id}"
        "__prediction"
    )


    if probability_column not in predictions.columns:

        raise RuntimeError(
            f"{candidate_id}: missing "
            "probability column."
        )


    if prediction_column not in predictions.columns:

        raise RuntimeError(
            f"{candidate_id}: missing "
            "prediction column."
        )


    probabilities = predictions[
        probability_column
    ].to_numpy(
        dtype=np.float64
    )


    classes = predictions[
        prediction_column
    ].to_numpy(
        dtype=np.int64
    )


    recomputed = (
        probabilities
        >=
        threshold
    ).astype(
        np.int64
    )


    if not np.array_equal(
        recomputed,
        classes,
    ):

        raise RuntimeError(
            f"{candidate_id}: frozen "
            "threshold application mismatch."
        )


    model_info.append(
        {
            "family":
                family,

            "candidate_id":
                candidate_id,

            "threshold":
                threshold,

            "probabilities":
                probabilities,

            "predictions":
                classes,
        }
    )


print(
    "Frozen thresholds reverified:",
    "PASS"
)


# ============================================================
# BUILD SCOPE MASKS
# ============================================================

scope_masks = {}


for scope_id in PRIMARY_BLOCKS:

    scope_masks[
        scope_id
    ] = (
        predictions[
            "primary_block_id"
        ].astype(str)
        ==
        scope_id
    ).to_numpy()


scope_masks[
    POOLED_SCOPE
] = np.ones(
    len(
        predictions
    ),
    dtype=bool,
)


# ============================================================
# BOOTSTRAP ALL SCOPES
# ============================================================

bootstrap_rows = []

scope_run_summary = []


print(
    "\n===================================="
)

print(
    "RUNNING CYCLE25 ENTITY_GROUP BOOTSTRAPS"
)

print(
    "===================================="
)


for scope_id in ALL_SCOPES:

    scope_mask = scope_masks[
        scope_id
    ]


    scope_indices = np.flatnonzero(
        scope_mask
    )


    if len(
        scope_indices
    ) == 0:

        raise RuntimeError(
            f"{scope_id}: empty scope."
        )


    y_scope = predictions.loc[
        scope_mask,
        "target"
    ].to_numpy(
        dtype=np.int64
    )


    group_values = predictions.loc[
        scope_mask,
        "ENTITY_GROUP"
    ]


    if group_values.isna().any():

        raise RuntimeError(
            f"{scope_id}: missing "
            "ENTITY_GROUP values."
        )


    group_codes, unique_groups = pd.factorize(
        group_values,
        sort=False,
    )


    group_codes = np.asarray(
        group_codes,
        dtype=np.int64,
    )


    group_count = int(
        len(
            unique_groups
        )
    )


    if group_count <= 1:

        raise RuntimeError(
            f"{scope_id}: insufficient "
            "ENTITY_GROUP clusters."
        )


    official_rows = point_metrics[
        point_metrics[
            "scope_id"
        ].astype(str)
        ==
        scope_id
    ]


    if len(
        official_rows
    ) != 3:

        raise RuntimeError(
            f"{scope_id}: expected "
            "3 point-metric rows."
        )


    expected_group_count = int(
        official_rows.iloc[
            0
        ][
            "entity_groups"
        ]
    )


    if group_count != expected_group_count:

        raise RuntimeError(
            f"{scope_id}: ENTITY_GROUP "
            "count differs from point metrics."
        )


    seed = SCOPE_SEEDS[
        scope_id
    ]


    rng = np.random.default_rng(
        seed
    )


    valid_replicates = 0
    invalid_replicates = 0
    attempts = 0

    MAX_ATTEMPTS = 100000


    print(
        f"\n{scope_id}: "
        f"rows={len(scope_indices)}, "
        f"ENTITY_GROUPs={group_count}, "
        f"seed={seed}"
    )


    while (
        valid_replicates
        <
        VALID_REPLICATES_REQUIRED
    ):

        attempts += 1


        if attempts > MAX_ATTEMPTS:

            raise RuntimeError(
                f"{scope_id}: exceeded "
                "maximum bootstrap attempts."
            )


        # ----------------------------------------------------
        # Sample G groups with replacement
        # ----------------------------------------------------

        sampled_group_codes = rng.integers(
            low=0,
            high=group_count,
            size=group_count,
        )


        multiplicities = np.bincount(
            sampled_group_codes,
            minlength=group_count,
        ).astype(
            np.float64
        )


        # Each row gets the multiplicity
        # of its ENTITY_GROUP.
        row_weights = multiplicities[
            group_codes
        ]


        bootstrap_positive = float(
            np.sum(
                row_weights
                *
                (y_scope == 1)
            )
        )


        bootstrap_negative = float(
            np.sum(
                row_weights
                *
                (y_scope == 0)
            )
        )


        if (
            bootstrap_positive <= 0
            or
            bootstrap_negative <= 0
        ):

            invalid_replicates += 1
            continue


        valid_replicates += 1


        sampled_unique_group_count = int(
            np.count_nonzero(
                multiplicities
            )
        )


        bootstrap_row_weight_total = int(
            np.sum(
                row_weights
            )
        )


        for model in model_info:

            probabilities_scope = model[
                "probabilities"
            ][
                scope_indices
            ]


            predictions_scope = model[
                "predictions"
            ][
                scope_indices
            ]


            metrics = compute_weighted_metrics(
                y_scope,
                probabilities_scope,
                predictions_scope,
                row_weights,
            )


            bootstrap_rows.append(
                {
                    "scope_id":
                        scope_id,

                    "replicate_id":
                        valid_replicates,

                    "family":
                        model[
                            "family"
                        ],

                    "candidate_id":
                        model[
                            "candidate_id"
                        ],

                    "bootstrap_seed":
                        seed,

                    "group_draw_count":
                        group_count,

                    "sampled_unique_group_count":
                        sampled_unique_group_count,

                    "bootstrap_row_weight_total":
                        bootstrap_row_weight_total,

                    "bootstrap_positive_weight":
                        bootstrap_positive,

                    "bootstrap_negative_weight":
                        bootstrap_negative,

                    **metrics,
                }
            )


        if (
            valid_replicates % 250
            ==
            0
        ):

            print(
                f"  Valid replicates: "
                f"{valid_replicates} / "
                f"{VALID_REPLICATES_REQUIRED}"
            )


    scope_run_summary.append(
        {
            "scope_id":
                scope_id,

            "rows":
                int(
                    len(
                        scope_indices
                    )
                ),

            "entity_groups":
                group_count,

            "seed":
                seed,

            "valid_replicates":
                valid_replicates,

            "invalid_replicates_redrawn":
                invalid_replicates,

            "total_attempts":
                attempts,
        }
    )


    print(
        f"{scope_id}: "
        f"bootstrap PASS | "
        f"valid={valid_replicates}, "
        f"invalid={invalid_replicates}, "
        f"attempts={attempts}"
    )


# ============================================================
# BUILD BOOTSTRAP TABLE
# ============================================================

bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


expected_rows = (
    len(
        ALL_SCOPES
    )
    *
    VALID_REPLICATES_REQUIRED
    *
    3
)


if len(
    bootstrap_df
) != expected_rows:

    raise RuntimeError(
        "Unexpected Cycle25 bootstrap "
        "replicate-table row count."
    )


# ============================================================
# STRUCTURAL CHECK
# ============================================================

for scope_id in ALL_SCOPES:

    scope_bootstrap = bootstrap_df[
        bootstrap_df[
            "scope_id"
        ]
        ==
        scope_id
    ]


    if scope_bootstrap[
        "replicate_id"
    ].nunique() != VALID_REPLICATES_REQUIRED:

        raise RuntimeError(
            f"{scope_id}: replicate "
            "ID count mismatch."
        )


    replicate_counts = (
        scope_bootstrap
        .groupby(
            "replicate_id"
        )
        .size()
    )


    if not (
        replicate_counts
        ==
        3
    ).all():

        raise RuntimeError(
            f"{scope_id}: not every "
            "replicate contains 3 models."
        )


print(
    "\nBootstrap structural integrity:",
    "PASS"
)

print(
    "Total bootstrap rows:",
    len(
        bootstrap_df
    )
)


# ============================================================
# CONFIDENCE INTERVALS
# ============================================================

ci_rows = []


for scope_id in ALL_SCOPES:

    for model in model_info:

        candidate_id = model[
            "candidate_id"
        ]


        official = point_metrics[
            (
                point_metrics[
                    "scope_id"
                ].astype(str)
                ==
                scope_id
            )
            &
            (
                point_metrics[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            )
        ]


        if len(
            official
        ) != 1:

            raise RuntimeError(
                f"{scope_id}/{candidate_id}: "
                "point estimate row missing."
            )


        official = official.iloc[
            0
        ]


        candidate_bootstrap = bootstrap_df[
            (
                bootstrap_df[
                    "scope_id"
                ]
                ==
                scope_id
            )
            &
            (
                bootstrap_df[
                    "candidate_id"
                ]
                ==
                candidate_id
            )
        ]


        if len(
            candidate_bootstrap
        ) != VALID_REPLICATES_REQUIRED:

            raise RuntimeError(
                f"{scope_id}/{candidate_id}: "
                "bootstrap replicate count "
                "mismatch."
            )


        for metric_name in BOOTSTRAP_METRICS:

            values = candidate_bootstrap[
                metric_name
            ].to_numpy(
                dtype=np.float64
            )


            if not np.isfinite(
                values
            ).all():

                raise RuntimeError(
                    f"{scope_id}/"
                    f"{candidate_id}/"
                    f"{metric_name}: "
                    "non-finite bootstrap values."
                )


            lower = float(
                np.quantile(
                    values,
                    CI_LOWER_QUANTILE,
                    method="linear",
                )
            )


            upper = float(
                np.quantile(
                    values,
                    CI_UPPER_QUANTILE,
                    method="linear",
                )
            )


            point_estimate = float(
                official[
                    metric_name
                ]
            )


            ci_rows.append(
                {
                    "scope_id":
                        scope_id,

                    "scope_type":
                        str(
                            official[
                                "scope_type"
                            ]
                        ),

                    "family":
                        model[
                            "family"
                        ],

                    "candidate_id":
                        candidate_id,

                    "metric":
                        metric_name,

                    "point_estimate":
                        point_estimate,

                    "bootstrap_mean":
                        float(
                            values.mean()
                        ),

                    "bootstrap_median":
                        float(
                            np.median(
                                values
                            )
                        ),

                    "ci_lower_2p5":
                        lower,

                    "ci_upper_97p5":
                        upper,

                    "confidence_level":
                        0.95,

                    "ci_method":
                        "percentile",

                    "quantile_method":
                        "linear",

                    "valid_replicates":
                        VALID_REPLICATES_REQUIRED,

                    "bootstrap_unit":
                        "ENTITY_GROUP",

                    "bootstrap_seed":
                        SCOPE_SEEDS[
                            scope_id
                        ],

                    "point_estimate_inside_ci":
                        bool(
                            lower
                            <=
                            point_estimate
                            <=
                            upper
                        ),
                }
            )


ci_df = pd.DataFrame(
    ci_rows
)


expected_ci_rows = (
    len(
        ALL_SCOPES
    )
    *
    3
    *
    len(
        BOOTSTRAP_METRICS
    )
)


if len(
    ci_df
) != expected_ci_rows:

    raise RuntimeError(
        "Unexpected Cycle25 bootstrap "
        "CI row count."
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_REPLICATES_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CI_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


bootstrap_df.to_parquet(
    OUTPUT_REPLICATES_FILE,
    index=False,
)


ci_df.to_csv(
    OUTPUT_CI_FILE,
    index=False,
)


replicates_hash = sha256_file(
    OUTPUT_REPLICATES_FILE
)


ci_hash = sha256_file(
    OUTPUT_CI_FILE
)


# ============================================================
# ROUND-TRIP
# ============================================================

replicate_roundtrip = pd.read_parquet(
    OUTPUT_REPLICATES_FILE
)


ci_roundtrip = pd.read_csv(
    OUTPUT_CI_FILE,
    keep_default_na=False,
)


if len(
    replicate_roundtrip
) != expected_rows:

    raise RuntimeError(
        "Cycle25 bootstrap parquet "
        "round-trip mismatch."
    )


if len(
    ci_roundtrip
) != expected_ci_rows:

    raise RuntimeError(
        "Cycle25 bootstrap CI CSV "
        "round-trip mismatch."
    )


print(
    "Bootstrap parquet round-trip:",
    "PASS"
)

print(
    "Bootstrap CI round-trip:",
    "PASS"
)


# ============================================================
# CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Frozen Cycle25 ENTITY_GROUP Bootstrap Confidence Intervals",

    "version":
        "1.0",

    "status":
        "FROZEN_CYCLE25_BOOTSTRAP_CI",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_8_4",

    "scopes":
        scope_run_summary,

    "bootstrap_design":
        {
            "unit":
                "ENTITY_GROUP",

            "sampling":
                "with_replacement",

            "group_draw_rule":
                (
                    "Within each evaluation scope, "
                    "draw exactly G ENTITY_GROUP "
                    "clusters with replacement, where "
                    "G is the number of distinct groups "
                    "present in that scope."
                ),

            "row_replication_implementation":
                (
                    "Integer sample weights equal to "
                    "ENTITY_GROUP draw multiplicities; "
                    "mathematically equivalent to "
                    "duplicating every row of each "
                    "sampled group once per draw."
                ),

            "paired_draws_across_models":
                True,

            "valid_replicates_per_scope":
                VALID_REPLICATES_REQUIRED,

            "invalid_replicates":
                (
                    "Redrawn when either target "
                    "class has zero bootstrap weight."
                ),

            "scope_specific_seeds":
                SCOPE_SEEDS,
        },

    "confidence_interval":
        {
            "method":
                "percentile",

            "confidence_level":
                0.95,

            "lower_quantile":
                CI_LOWER_QUANTILE,

            "upper_quantile":
                CI_UPPER_QUANTILE,

            "numpy_quantile_method":
                "linear",
        },

    "metrics":
        BOOTSTRAP_METRICS,

    "threshold_policy":
        {
            "source":
                "CYCLE24_VALIDATION",

            "unchanged":
                True,

            "reoptimized_during_bootstrap":
                False,
        },

    "model_policy":
        {
            "frozen_models":
                True,

            "retraining":
                False,

            "candidate_replacement":
                False,

            "preprocessing_refit":
                False,
        },

    "artifacts":
        {
            "bootstrap_replicates":
                {
                    "file":
                        str(
                            OUTPUT_REPLICATES_FILE
                        ),

                    "sha256":
                        replicates_hash,

                    "rows":
                        len(
                            bootstrap_df
                        ),
                },

            "bootstrap_ci_summary":
                {
                    "file":
                        str(
                            OUTPUT_CI_FILE
                        ),

                    "sha256":
                        ci_hash,

                    "rows":
                        len(
                            ci_df
                        ),
                },
        },

    "evaluation_state":
        {
            "cycle25_point_metrics_frozen":
                True,

            "cycle25_bootstrap_ci_computed":
                True,

            "cross_cycle_delta_computed":
                False,

            "predictive_drift_classification_computed":
                False,

            "new_model_fitting":
                False,

            "threshold_tuning":
                False,

            "preprocessing_refit":
                False,
        },

    "data_access":
        {
            "frozen_cycle25_predictions_read":
                True,

            "raw_cycle25_dataset_reread":
                False,

            "cycle24_train_read":
                False,

            "cycle24_validation_read":
                False,

            "cycle24_holdout_raw_read":
                False,
        },

    "provenance":
        {
            "cycle25_prediction_artifact":
                {
                    "file":
                        str(
                            PREDICTION_FILE
                        ),

                    "sha256":
                        prediction_hash,
                },

            "cycle25_prediction_contract":
                {
                    "file":
                        str(
                            PREDICTION_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            PREDICTION_CONTRACT_FILE
                        ),
                },

            "cycle25_evaluation_protocol":
                {
                    "file":
                        str(
                            CYCLE25_PROTOCOL_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CYCLE25_PROTOCOL_FILE
                        ),
                },

            "cycle25_point_metrics":
                {
                    "file":
                        str(
                            POINT_METRICS_FILE
                        ),

                    "sha256":
                        metrics_hash,
                },

            "cycle25_point_metrics_contract":
                {
                    "file":
                        str(
                            POINT_METRICS_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            POINT_METRICS_CONTRACT_FILE
                        ),
                },

            "cycle25_model_registry":
                {
                    "file":
                        str(
                            MODEL_REGISTRY_FILE
                        ),

                    "sha256":
                        sha256_file(
                            MODEL_REGISTRY_FILE
                        ),
                },
        },

    "interpretation_rule":
        (
            "Cycle25 confidence intervals quantify "
            "sampling uncertainty under connected "
            "ENTITY_GROUP resampling. Cross-cycle "
            "predictive drift must not be classified "
            "from marginal Cycle24 and Cycle25 CIs "
            "alone; the frozen difference-bootstrap "
            "analysis is performed in Chapter 8.5."
        ),
}


# ============================================================
# SAVE CONTRACT
# ============================================================

with OUTPUT_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "CYCLE25 TSS 95% ENTITY_GROUP BOOTSTRAP CI"
)

print(
    "===================================="
)


tss_ci = ci_df[
    ci_df[
        "metric"
    ]
    ==
    "tss"
]


for scope_id in ALL_SCOPES:

    print(
        f"\n{scope_id}"
    )


    subset = tss_ci[
        tss_ci[
            "scope_id"
        ]
        ==
        scope_id
    ]


    for _, row in subset.iterrows():

        print(
            f"  {row['candidate_id']}: "
            f"TSS={row['point_estimate']:.6f}, "
            f"95% CI=["
            f"{row['ci_lower_2p5']:.6f}, "
            f"{row['ci_upper_97p5']:.6f}]"
        )


print(
    "\n===================================="
)

print(
    "BOOTSTRAP STATE"
)

print(
    "===================================="
)


print(
    "Evaluation scopes:",
    4
)

print(
    "Models/scope:",
    3
)

print(
    "Valid replicates/scope:",
    2000
)

print(
    "Expected replicate rows:",
    expected_rows
)

print(
    "CI rows:",
    expected_ci_rows
)

print(
    "Bootstrap unit:",
    "ENTITY_GROUP"
)

print(
    "Paired draws across models:",
    True
)

print(
    "Models retrained:",
    False
)

print(
    "Thresholds reoptimized:",
    False
)

print(
    "Raw Cycle25 dataset reread:",
    False
)

print(
    "Cross-cycle drift computed:",
    False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_REPLICATES_FILE
)

print(
    OUTPUT_CI_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 8.4 COMPLETE — CYCLE25 BOOTSTRAP CIs FROZEN"
)

print(
    "===================================="
)