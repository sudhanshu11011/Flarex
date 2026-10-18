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
    "cycle24_holdout_predictions.parquet"
)

PREDICTION_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_prediction_contract.json"
)

HOLDOUT_PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_evaluation_protocol.json"
)

POINT_METRICS_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "holdout_metrics.csv"
)

POINT_METRICS_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_metrics_contract.json"
)

MODEL_REGISTRY_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "holdout_frozen_model_registry.csv"
)

OUTPUT_REPLICATES_FILE = Path(
    "artifacts/models/evaluation/bootstrap/"
    "holdout_group_bootstrap_replicates.parquet"
)

OUTPUT_CI_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "holdout_bootstrap_ci.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_bootstrap_contract.json"
)


# ============================================================
# FROZEN BOOTSTRAP DESIGN
# ============================================================

EXPECTED_ROWS = 13072
EXPECTED_POSITIVES = 527
EXPECTED_NEGATIVES = 12545
EXPECTED_ENTITY_GROUPS = 176

BOOTSTRAP_SEED = 24025
VALID_REPLICATES_REQUIRED = 2000

CI_LOWER_QUANTILE = 0.025
CI_UPPER_QUANTILE = 0.975


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
        numerator
        /
        denominator
    )


def calculate_hss(
    tp,
    fp,
    tn,
    fn,
):

    numerator = 2 * (
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
        numerator
        /
        denominator
    )


def calculate_f1(
    tp,
    fp,
    fn,
):

    denominator = (
        2 * tp
        +
        fp
        +
        fn
    )

    if denominator == 0:
        return 0.0

    return float(
        2 * tp
        /
        denominator
    )


def compute_metrics(
    y_true,
    probabilities,
    predictions,
):

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


    positive_count = int(
        y_true.sum()
    )

    negative_count = int(
        len(y_true)
        -
        positive_count
    )


    if positive_count <= 0:

        raise RuntimeError(
            "Bootstrap sample contains no "
            "positive targets."
        )


    if negative_count <= 0:

        raise RuntimeError(
            "Bootstrap sample contains no "
            "negative targets."
        )


    tp = int(
        np.sum(
            (predictions == 1)
            &
            (y_true == 1)
        )
    )

    fp = int(
        np.sum(
            (predictions == 1)
            &
            (y_true == 0)
        )
    )

    tn = int(
        np.sum(
            (predictions == 0)
            &
            (y_true == 0)
        )
    )

    fn = int(
        np.sum(
            (predictions == 0)
            &
            (y_true == 1)
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
        sensitivity
        -
        fpr
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
        )
    )


    roc_auc = float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


    brier = float(
        brier_score_loss(
            y_true,
            probabilities,
        )
    )


    return {
        "tss":
            tss,

        "hss":
            hss,

        "f1":
            f1,

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "precision":
            precision,

        "pr_auc":
            pr_auc,

        "roc_auc":
            roc_auc,

        "brier_score":
            brier,
    }


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 7.4 — ENTITY_GROUP BOOTSTRAP CONFIDENCE INTERVALS"
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
    HOLDOUT_PROTOCOL_FILE,
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

for output_path in [
    OUTPUT_REPLICATES_FILE,
    OUTPUT_CI_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Output already exists: {output_path}\n"
            "Refusing to overwrite frozen "
            "bootstrap provenance."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

prediction_contract = load_json(
    PREDICTION_CONTRACT_FILE
)

holdout_protocol = load_json(
    HOLDOUT_PROTOCOL_FILE
)

point_metrics_contract = load_json(
    POINT_METRICS_CONTRACT_FILE
)


if prediction_contract.get(
    "status"
) != "FROZEN_HOLDOUT_PREDICTIONS":

    raise RuntimeError(
        "Holdout prediction contract "
        "is not frozen."
    )


if holdout_protocol.get(
    "status"
) != "FROZEN_BEFORE_HOLDOUT_EVALUATION":

    raise RuntimeError(
        "Holdout evaluation protocol "
        "is not frozen."
    )


if point_metrics_contract.get(
    "status"
) != "FROZEN_HOLDOUT_POINT_METRICS":

    raise RuntimeError(
        "Holdout point metrics are "
        "not frozen."
    )


print(
    "Holdout prediction contract:",
    "FROZEN"
)

print(
    "Holdout evaluation protocol:",
    "FROZEN"
)

print(
    "Holdout point metrics:",
    "FROZEN"
)


# ============================================================
# VERIFY FROZEN BOOTSTRAP PROTOCOL
# ============================================================

bootstrap_protocol = (
    holdout_protocol[
        "bootstrap"
    ]
)


if bootstrap_protocol[
    "unit"
] != "ENTITY_GROUP":

    raise RuntimeError(
        "Frozen bootstrap unit is not "
        "ENTITY_GROUP."
    )


if int(
    bootstrap_protocol[
        "random_seed"
    ]
) != BOOTSTRAP_SEED:

    raise RuntimeError(
        "Frozen bootstrap seed differs "
        "from expected seed."
    )


if int(
    bootstrap_protocol[
        "valid_replicates"
    ]
) != VALID_REPLICATES_REQUIRED:

    raise RuntimeError(
        "Frozen bootstrap replicate count "
        "differs from expected value."
    )


if bootstrap_protocol[
    "ci_method"
] != "percentile":

    raise RuntimeError(
        "Frozen bootstrap CI method "
        "is not percentile."
    )


if float(
    bootstrap_protocol[
        "lower_quantile"
    ]
) != CI_LOWER_QUANTILE:

    raise RuntimeError(
        "Frozen lower CI quantile mismatch."
    )


if float(
    bootstrap_protocol[
        "upper_quantile"
    ]
) != CI_UPPER_QUANTILE:

    raise RuntimeError(
        "Frozen upper CI quantile mismatch."
    )


print(
    "Bootstrap unit:",
    "ENTITY_GROUP"
)

print(
    "Bootstrap seed:",
    BOOTSTRAP_SEED
)

print(
    "Required valid replicates:",
    VALID_REPLICATES_REQUIRED
)

print(
    "CI method:",
    "95% percentile"
)


# ============================================================
# VERIFY INPUT ARTIFACT HASHES
# ============================================================

actual_prediction_hash = sha256_file(
    PREDICTION_FILE
)


if (
    actual_prediction_hash
    !=
    prediction_contract[
        "prediction_artifact"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Frozen holdout prediction "
        "artifact hash mismatch."
    )


actual_point_metrics_hash = sha256_file(
    POINT_METRICS_FILE
)


if (
    actual_point_metrics_hash
    !=
    point_metrics_contract[
        "artifacts"
    ][
        "holdout_metrics"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Frozen holdout point-metric "
        "artifact hash mismatch."
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


# ============================================================
# BASIC INTEGRITY
# ============================================================

if len(
    predictions
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Unexpected holdout prediction "
        "row count."
    )


if len(
    point_metrics
) != 3:

    raise RuntimeError(
        "Expected 3 frozen point-metric rows."
    )


if len(
    registry
) != 3:

    raise RuntimeError(
        "Expected 3 frozen model registry rows."
    )


required_columns = {
    "ENTITY_GROUP",
    "target",
}


if not required_columns.issubset(
    predictions.columns
):

    raise RuntimeError(
        "Prediction artifact missing "
        "bootstrap-required columns."
    )


if predictions[
    "ENTITY_GROUP"
].isna().any():

    raise RuntimeError(
        "Missing ENTITY_GROUP value."
    )


y_full = predictions[
    "target"
].to_numpy(
    dtype=np.int64
)


if int(
    y_full.sum()
) != EXPECTED_POSITIVES:

    raise RuntimeError(
        "Unexpected holdout positive count."
    )


if int(
    (
        y_full
        ==
        0
    ).sum()
) != EXPECTED_NEGATIVES:

    raise RuntimeError(
        "Unexpected holdout negative count."
    )


# ============================================================
# BUILD GROUP -> ROW-INDEX MAP
# ============================================================

group_indices_raw = (
    predictions
    .groupby(
        "ENTITY_GROUP",
        sort=False,
    )
    .indices
)


group_to_indices = {

    group:
        np.asarray(
            indices,
            dtype=np.int64,
        )

    for group, indices
    in group_indices_raw.items()
}


unique_groups = np.asarray(
    list(
        group_to_indices.keys()
    ),
    dtype=object,
)


if len(
    unique_groups
) != EXPECTED_ENTITY_GROUPS:

    raise RuntimeError(
        "Expected exactly 176 "
        "ENTITY_GROUP clusters."
    )


print(
    "Holdout rows:",
    len(
        predictions
    )
)

print(
    "ENTITY_GROUP clusters:",
    len(
        unique_groups
    )
)


# ============================================================
# VERIFY MODEL COLUMNS AND FROZEN THRESHOLDS
# ============================================================

model_information = []


for _, registry_row in registry.iterrows():

    family = str(
        registry_row[
            "family"
        ]
    )


    candidate_id = str(
        registry_row[
            "candidate_id"
        ]
    )


    threshold = float(
        registry_row[
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
            f"{candidate_id}: probability "
            "column missing."
        )


    if prediction_column not in predictions.columns:

        raise RuntimeError(
            f"{candidate_id}: prediction "
            "column missing."
        )


    probability_values = predictions[
        probability_column
    ].to_numpy(
        dtype=np.float64
    )


    prediction_values = predictions[
        prediction_column
    ].to_numpy(
        dtype=np.int64
    )


    recomputed_classes = (
        probability_values
        >=
        threshold
    ).astype(
        np.int64
    )


    if not np.array_equal(
        recomputed_classes,
        prediction_values,
    ):

        raise RuntimeError(
            f"{candidate_id}: stored classes "
            "do not match frozen threshold."
        )


    point_row = point_metrics[
        point_metrics[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(
        point_row
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: frozen point "
            "metric row missing."
        )


    model_information.append(
        {
            "family":
                family,

            "candidate_id":
                candidate_id,

            "threshold":
                threshold,

            "probabilities":
                probability_values,

            "predictions":
                prediction_values,

            "point_metrics":
                point_row.iloc[
                    0
                ],
        }
    )


print(
    "Frozen model prediction columns:",
    "PASS"
)

print(
    "Frozen thresholds reverified:",
    "PASS"
)


# ============================================================
# INDEPENDENT POINT-METRIC RECHECK
# ============================================================

for model in model_information:

    recalculated = compute_metrics(
        y_full,
        model[
            "probabilities"
        ],
        model[
            "predictions"
        ],
    )


    point_row = model[
        "point_metrics"
    ]


    for metric_name in BOOTSTRAP_METRICS:

        expected_value = float(
            point_row[
                metric_name
            ]
        )


        actual_value = float(
            recalculated[
                metric_name
            ]
        )


        if not np.isclose(
            expected_value,
            actual_value,
            rtol=0.0,
            atol=1e-12,
        ):

            raise RuntimeError(
                f"{model['candidate_id']}: "
                f"point metric mismatch for "
                f"{metric_name}."
            )


print(
    "Independent point-metric recheck:",
    "PASS"
)


# ============================================================
# BOOTSTRAP
# ============================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


bootstrap_rows = []

valid_replicates = 0
invalid_replicates = 0
attempts = 0

MAX_ATTEMPTS = 100000


print(
    "\n===================================="
)

print(
    "RUNNING CONNECTED ENTITY-GROUP BOOTSTRAP"
)

print(
    "===================================="
)


while (
    valid_replicates
    <
    VALID_REPLICATES_REQUIRED
):

    attempts += 1


    if attempts > MAX_ATTEMPTS:

        raise RuntimeError(
            "Exceeded maximum bootstrap "
            "attempts before obtaining "
            "required valid replicates."
        )


    sampled_groups = rng.choice(
        unique_groups,
        size=EXPECTED_ENTITY_GROUPS,
        replace=True,
    )


    sampled_indices = np.concatenate(
        [
            group_to_indices[
                group
            ]
            for group in sampled_groups
        ]
    )


    y_bootstrap = y_full[
        sampled_indices
    ]


    bootstrap_positive = int(
        y_bootstrap.sum()
    )


    bootstrap_negative = int(
        len(
            y_bootstrap
        )
        -
        bootstrap_positive
    )


    # --------------------------------------------------------
    # Frozen invalid-replicate rule
    # --------------------------------------------------------

    if (
        bootstrap_positive == 0
        or
        bootstrap_negative == 0
    ):

        invalid_replicates += 1
        continue


    valid_replicates += 1


    sampled_unique_group_count = int(
        len(
            set(
                sampled_groups.tolist()
            )
        )
    )


    for model in model_information:

        probabilities_bootstrap = (
            model[
                "probabilities"
            ][
                sampled_indices
            ]
        )


        predictions_bootstrap = (
            model[
                "predictions"
            ][
                sampled_indices
            ]
        )


        metrics = compute_metrics(
            y_bootstrap,
            probabilities_bootstrap,
            predictions_bootstrap,
        )


        bootstrap_rows.append(
            {
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

                "bootstrap_sample_rows":
                    len(
                        sampled_indices
                    ),

                "bootstrap_positive":
                    bootstrap_positive,

                "bootstrap_negative":
                    bootstrap_negative,

                "group_draw_count":
                    EXPECTED_ENTITY_GROUPS,

                "sampled_unique_group_count":
                    sampled_unique_group_count,

                **metrics,
            }
        )


    if (
        valid_replicates % 250
        ==
        0
    ):

        print(
            f"Valid replicates: "
            f"{valid_replicates} / "
            f"{VALID_REPLICATES_REQUIRED}"
        )


# ============================================================
# BUILD REPLICATE TABLE
# ============================================================

bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


expected_bootstrap_rows = (
    VALID_REPLICATES_REQUIRED
    *
    3
)


if len(
    bootstrap_df
) != expected_bootstrap_rows:

    raise RuntimeError(
        "Unexpected bootstrap output "
        "row count."
    )


if bootstrap_df[
    "replicate_id"
].nunique() != VALID_REPLICATES_REQUIRED:

    raise RuntimeError(
        "Bootstrap replicate IDs are "
        "not complete."
    )


for replicate_id, group in bootstrap_df.groupby(
    "replicate_id"
):

    if len(
        group
    ) != 3:

        raise RuntimeError(
            f"Replicate {replicate_id}: "
            "does not contain all 3 models."
        )


print(
    "\nBootstrap generation:",
    "PASS"
)

print(
    "Valid replicates:",
    valid_replicates
)

print(
    "Invalid/redrawn replicates:",
    invalid_replicates
)

print(
    "Total sampling attempts:",
    attempts
)

print(
    "Models per replicate:",
    3
)

print(
    "Paired group draws across models:",
    True
)


# ============================================================
# BUILD CONFIDENCE-INTERVAL SUMMARY
# ============================================================

ci_rows = []


for model in model_information:

    candidate_id = model[
        "candidate_id"
    ]


    family = model[
        "family"
    ]


    candidate_bootstrap = bootstrap_df[
        bootstrap_df[
            "candidate_id"
        ]
        ==
        candidate_id
    ]


    if len(
        candidate_bootstrap
    ) != VALID_REPLICATES_REQUIRED:

        raise RuntimeError(
            f"{candidate_id}: expected "
            "2000 bootstrap replicates."
        )


    point_row = model[
        "point_metrics"
    ]


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
                f"{candidate_id}: "
                f"{metric_name} contains "
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
            point_row[
                metric_name
            ]
        )


        ci_rows.append(
            {
                "family":
                    family,

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
                    BOOTSTRAP_SEED,

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


if len(
    ci_df
) != (
    3
    *
    len(
        BOOTSTRAP_METRICS
    )
):

    raise RuntimeError(
        "Unexpected bootstrap CI "
        "summary row count."
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
# ROUND-TRIP VERIFICATION
# ============================================================

replicates_roundtrip = pd.read_parquet(
    OUTPUT_REPLICATES_FILE
)


ci_roundtrip = pd.read_csv(
    OUTPUT_CI_FILE,
    keep_default_na=False,
)


if len(
    replicates_roundtrip
) != expected_bootstrap_rows:

    raise RuntimeError(
        "Bootstrap parquet round-trip "
        "row mismatch."
    )


if len(
    ci_roundtrip
) != len(
    ci_df
):

    raise RuntimeError(
        "Bootstrap CI CSV round-trip "
        "row mismatch."
    )


print(
    "\nBootstrap parquet round-trip:",
    "PASS"
)

print(
    "Bootstrap CI round-trip:",
    "PASS"
)


# ============================================================
# BUILD CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Cycle24 Holdout ENTITY_GROUP Bootstrap Confidence Intervals",

    "version":
        "1.0",

    "status":
        "FROZEN_HOLDOUT_BOOTSTRAP_CI",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "bootstrap_design":
        {
            "unit":
                "ENTITY_GROUP",

            "group_count":
                EXPECTED_ENTITY_GROUPS,

            "groups_drawn_per_replicate":
                EXPECTED_ENTITY_GROUPS,

            "sampling":
                "with_replacement",

            "cluster_rows_preserved":
                True,

            "paired_draws_across_models":
                True,

            "seed":
                BOOTSTRAP_SEED,

            "valid_replicates":
                valid_replicates,

            "invalid_replicates_redrawn":
                invalid_replicates,

            "total_sampling_attempts":
                attempts,
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

            "frozen_unchanged":
                True,

            "reoptimized_during_bootstrap":
                False,
        },

    "model_policy":
        {
            "models_frozen":
                True,

            "retraining":
                False,

            "candidate_changes":
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

    "data_isolation":
        {
            "frozen_holdout_predictions_read":
                True,

            "raw_holdout_dataset_reread":
                False,

            "train_read":
                False,

            "validation_read":
                False,

            "cycle25_read":
                False,

            "new_model_fitting":
                False,

            "threshold_tuning":
                False,
        },

    "provenance":
        {
            "holdout_prediction_artifact":
                {
                    "file":
                        str(
                            PREDICTION_FILE
                        ),

                    "sha256":
                        actual_prediction_hash,
                },

            "holdout_prediction_contract":
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

            "holdout_evaluation_protocol":
                {
                    "file":
                        str(
                            HOLDOUT_PROTOCOL_FILE
                        ),

                    "sha256":
                        sha256_file(
                            HOLDOUT_PROTOCOL_FILE
                        ),
                },

            "holdout_point_metrics":
                {
                    "file":
                        str(
                            POINT_METRICS_FILE
                        ),

                    "sha256":
                        actual_point_metrics_hash,
                },

            "holdout_point_metrics_contract":
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

            "frozen_model_registry":
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
            "Confidence intervals quantify uncertainty "
            "under connected ENTITY_GROUP resampling. "
            "They must not be used to retroactively "
            "change the frozen models, thresholds, "
            "features, preprocessing, or model-family "
            "selection."
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
    "95% ENTITY_GROUP BOOTSTRAP CI — TSS"
)

print(
    "===================================="
)


for _, row in ci_df[
    ci_df[
        "metric"
    ]
    ==
    "tss"
].iterrows():

    print(
        f"{row['candidate_id']}: "
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
    "Valid replicates:",
    valid_replicates
)

print(
    "Invalid replicates redrawn:",
    invalid_replicates
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
    "Confidence interval:",
    "95% percentile"
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
    "Cycle25 read:",
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
    "MODEL CHAPTER 7.4 COMPLETE — HOLDOUT BOOTSTRAP CIs FROZEN"
)

print(
    "===================================="
)