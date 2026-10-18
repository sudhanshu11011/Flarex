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

PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_evaluation_protocol.json"
)

MODEL_REGISTRY_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "holdout_frozen_model_registry.csv"
)

POINT_METRICS_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "holdout_metrics.csv"
)

POINT_METRICS_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_metrics_contract.json"
)

BOOTSTRAP_REPLICATES_FILE = Path(
    "artifacts/models/evaluation/bootstrap/"
    "holdout_group_bootstrap_replicates.parquet"
)

BOOTSTRAP_CI_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "holdout_bootstrap_ci.csv"
)

BOOTSTRAP_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_bootstrap_contract.json"
)

OUTPUT_AUDIT_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "holdout_evaluation_audit.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_evaluation_audit.json"
)


# ============================================================
# FROZEN EXPECTATIONS
# ============================================================

EXPECTED_ROWS = 13072
EXPECTED_POSITIVES = 527
EXPECTED_NEGATIVES = 12545
EXPECTED_ENTITY_GROUPS = 176

EXPECTED_MODELS = 3

EXPECTED_VALID_REPLICATES = 2000
EXPECTED_BOOTSTRAP_ROWS = 6000

BOOTSTRAP_SEED = 24025

CI_LOWER = 0.025
CI_UPPER = 0.975


METRICS = [
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
        numerator / denominator
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
        (2 * tp) / denominator
    )


def compute_point_metrics(
    y_true,
    probabilities,
    predictions,
):

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


    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,

        "tss":
            float(
                sensitivity - fpr
            ),

        "hss":
            calculate_hss(
                tp,
                fp,
                tn,
                fn,
            ),

        "f1":
            calculate_f1(
                tp,
                fp,
                fn,
            ),

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "precision":
            precision,

        "pr_auc":
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y_true,
                    probabilities,
                )
            ),
    }


def close_enough(
    left,
    right,
    atol=1e-12,
):

    return bool(
        np.isclose(
            float(left),
            float(right),
            rtol=0.0,
            atol=atol,
        )
    )


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 7.5 — INDEPENDENT HOLDOUT EVALUATION AUDIT"
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
    PROTOCOL_FILE,
    MODEL_REGISTRY_FILE,
    POINT_METRICS_FILE,
    POINT_METRICS_CONTRACT_FILE,
    BOOTSTRAP_REPLICATES_FILE,
    BOOTSTRAP_CI_FILE,
    BOOTSTRAP_CONTRACT_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


for output_path in [
    OUTPUT_AUDIT_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Audit output already exists: "
            f"{output_path}\n"
            "Refusing to overwrite independent audit."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

prediction_contract = load_json(
    PREDICTION_CONTRACT_FILE
)

protocol = load_json(
    PROTOCOL_FILE
)

metrics_contract = load_json(
    POINT_METRICS_CONTRACT_FILE
)

bootstrap_contract = load_json(
    BOOTSTRAP_CONTRACT_FILE
)


# ============================================================
# CONTRACT STATE VERIFICATION
# ============================================================

if prediction_contract.get(
    "status"
) != "FROZEN_HOLDOUT_PREDICTIONS":

    raise RuntimeError(
        "Holdout prediction contract "
        "is not frozen."
    )


if protocol.get(
    "status"
) != "FROZEN_BEFORE_HOLDOUT_EVALUATION":

    raise RuntimeError(
        "Holdout protocol is not frozen."
    )


if metrics_contract.get(
    "status"
) != "FROZEN_HOLDOUT_POINT_METRICS":

    raise RuntimeError(
        "Holdout point metrics are "
        "not frozen."
    )


if bootstrap_contract.get(
    "status"
) != "FROZEN_HOLDOUT_BOOTSTRAP_CI":

    raise RuntimeError(
        "Holdout bootstrap CI contract "
        "is not frozen."
    )


print(
    "Holdout protocol:",
    "VERIFIED"
)

print(
    "Prediction contract:",
    "VERIFIED"
)

print(
    "Point metrics contract:",
    "VERIFIED"
)

print(
    "Bootstrap contract:",
    "VERIFIED"
)


# ============================================================
# VERIFY ARTIFACT HASH CHAIN
# ============================================================

prediction_hash = sha256_file(
    PREDICTION_FILE
)


if (
    prediction_hash
    !=
    prediction_contract[
        "prediction_artifact"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Prediction artifact hash mismatch."
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
        "holdout_metrics"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Point metrics hash mismatch."
    )


bootstrap_replicates_hash = sha256_file(
    BOOTSTRAP_REPLICATES_FILE
)


if (
    bootstrap_replicates_hash
    !=
    bootstrap_contract[
        "artifacts"
    ][
        "bootstrap_replicates"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Bootstrap replicate artifact "
        "hash mismatch."
    )


bootstrap_ci_hash = sha256_file(
    BOOTSTRAP_CI_FILE
)


if (
    bootstrap_ci_hash
    !=
    bootstrap_contract[
        "artifacts"
    ][
        "bootstrap_ci_summary"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Bootstrap CI artifact "
        "hash mismatch."
    )


print(
    "Artifact hash chain:",
    "PASS"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

predictions = pd.read_parquet(
    PREDICTION_FILE
)

registry = pd.read_csv(
    MODEL_REGISTRY_FILE,
    keep_default_na=False,
)

point_metrics = pd.read_csv(
    POINT_METRICS_FILE,
    keep_default_na=False,
)

bootstrap_df = pd.read_parquet(
    BOOTSTRAP_REPLICATES_FILE
)

ci_df = pd.read_csv(
    BOOTSTRAP_CI_FILE,
    keep_default_na=False,
)


# ============================================================
# POPULATION INTEGRITY
# ============================================================

if len(
    predictions
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Unexpected prediction row count."
    )


y_true = predictions[
    "target"
].to_numpy(
    dtype=np.int64
)


if int(
    y_true.sum()
) != EXPECTED_POSITIVES:

    raise RuntimeError(
        "Unexpected positive count."
    )


if int(
    (y_true == 0).sum()
) != EXPECTED_NEGATIVES:

    raise RuntimeError(
        "Unexpected negative count."
    )


if int(
    predictions[
        "ENTITY_GROUP"
    ].nunique(
        dropna=False
    )
) != EXPECTED_ENTITY_GROUPS:

    raise RuntimeError(
        "Unexpected ENTITY_GROUP count."
    )


if predictions[
    "ENTITY_GROUP"
].isna().any():

    raise RuntimeError(
        "Missing ENTITY_GROUP values."
    )


print(
    "Holdout population integrity:",
    "PASS"
)


# ============================================================
# REGISTRY INTEGRITY
# ============================================================

if len(
    registry
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 models."
    )


if len(
    point_metrics
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 point "
        "metric rows."
    )


if set(
    registry[
        "candidate_id"
    ].astype(str)
) != set(
    point_metrics[
        "candidate_id"
    ].astype(str)
):

    raise RuntimeError(
        "Point metric candidate set differs "
        "from frozen registry."
    )


# ============================================================
# INDEPENDENT POINT-METRIC RECOMPUTATION
# ============================================================

audit_rows = []


print(
    "\n===================================="
)

print(
    "INDEPENDENT POINT-METRIC AUDIT"
)

print(
    "===================================="
)


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


    probabilities = predictions[
        probability_column
    ].to_numpy(
        dtype=np.float64
    )


    stored_predictions = predictions[
        prediction_column
    ].to_numpy(
        dtype=np.int64
    )


    threshold_predictions = (
        probabilities
        >=
        threshold
    ).astype(
        np.int64
    )


    threshold_match = bool(
        np.array_equal(
            threshold_predictions,
            stored_predictions,
        )
    )


    if not threshold_match:

        raise RuntimeError(
            f"{candidate_id}: frozen threshold "
            "application mismatch."
        )


    recomputed = compute_point_metrics(
        y_true,
        probabilities,
        stored_predictions,
    )


    official = point_metrics[
        point_metrics[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(
        official
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: expected one "
            "official metric row."
        )


    official = official.iloc[
        0
    ]


    confusion_match = bool(
        int(
            official[
                "tp"
            ]
        )
        ==
        recomputed[
            "tp"
        ]
        and
        int(
            official[
                "fp"
            ]
        )
        ==
        recomputed[
            "fp"
        ]
        and
        int(
            official[
                "tn"
            ]
        )
        ==
        recomputed[
            "tn"
        ]
        and
        int(
            official[
                "fn"
            ]
        )
        ==
        recomputed[
            "fn"
        ]
    )


    metric_matches = {}


    for metric_name in METRICS:

        metric_matches[
            metric_name
        ] = close_enough(
            official[
                metric_name
            ],
            recomputed[
                metric_name
            ],
            atol=1e-12,
        )


    all_metrics_match = bool(
        all(
            metric_matches.values()
        )
    )


    point_audit_pass = bool(
        threshold_match
        and
        confusion_match
        and
        all_metrics_match
    )


    audit_rows.append(
        {
            "family":
                family,

            "candidate_id":
                candidate_id,

            "frozen_threshold":
                threshold,

            "threshold_application_match":
                threshold_match,

            "confusion_matrix_match":
                confusion_match,

            "point_metrics_match":
                all_metrics_match,

            "official_tss":
                float(
                    official[
                        "tss"
                    ]
                ),

            "recomputed_tss":
                float(
                    recomputed[
                        "tss"
                    ]
                ),

            "official_pr_auc":
                float(
                    official[
                        "pr_auc"
                    ]
                ),

            "recomputed_pr_auc":
                float(
                    recomputed[
                        "pr_auc"
                    ]
                ),

            "official_roc_auc":
                float(
                    official[
                        "roc_auc"
                    ]
                ),

            "recomputed_roc_auc":
                float(
                    recomputed[
                        "roc_auc"
                    ]
                ),

            "official_brier":
                float(
                    official[
                        "brier_score"
                    ]
                ),

            "recomputed_brier":
                float(
                    recomputed[
                        "brier_score"
                    ]
                ),

            "point_metric_audit_pass":
                point_audit_pass,
        }
    )


    print(
        f"{candidate_id}: "
        f"threshold=PASS, "
        f"confusion=PASS, "
        f"TSS={recomputed['tss']:.6f}, "
        f"metrics_match={all_metrics_match}, "
        f"PASS={point_audit_pass}"
    )


# ============================================================
# BOOTSTRAP STRUCTURAL AUDIT
# ============================================================

print(
    "\n===================================="
)

print(
    "BOOTSTRAP STRUCTURAL AUDIT"
)

print(
    "===================================="
)


if len(
    bootstrap_df
) != EXPECTED_BOOTSTRAP_ROWS:

    raise RuntimeError(
        "Expected exactly 6000 "
        "bootstrap rows."
    )


if bootstrap_df[
    "replicate_id"
].nunique() != EXPECTED_VALID_REPLICATES:

    raise RuntimeError(
        "Expected exactly 2000 unique "
        "bootstrap replicate IDs."
    )


replicate_counts = bootstrap_df.groupby(
    "replicate_id"
).size()


if not (
    replicate_counts
    ==
    EXPECTED_MODELS
).all():

    raise RuntimeError(
        "Not every replicate contains "
        "all 3 model families."
    )


if not (
    bootstrap_df[
        "group_draw_count"
    ].astype(
        int
    )
    ==
    EXPECTED_ENTITY_GROUPS
).all():

    raise RuntimeError(
        "Bootstrap group draw count "
        "differs from 176."
    )


if (
    bootstrap_df[
        "bootstrap_positive"
    ]
    <=
    0
).any():

    raise RuntimeError(
        "Bootstrap replicate with "
        "zero positives detected."
    )


if (
    bootstrap_df[
        "bootstrap_negative"
    ]
    <=
    0
).any():

    raise RuntimeError(
        "Bootstrap replicate with "
        "zero negatives detected."
    )


for column in METRICS:

    values = bootstrap_df[
        column
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"Bootstrap metric {column} "
            "contains non-finite values."
        )


print(
    "Bootstrap rows:",
    len(
        bootstrap_df
    )
)

print(
    "Unique valid replicates:",
    bootstrap_df[
        "replicate_id"
    ].nunique()
)

print(
    "Models per replicate:",
    3
)

print(
    "Groups drawn per replicate:",
    176
)

print(
    "Class-valid replicates:",
    "PASS"
)


# ============================================================
# INDEPENDENT CI RECOMPUTATION
# ============================================================

print(
    "\n===================================="
)

print(
    "INDEPENDENT CI AUDIT"
)

print(
    "===================================="
)


expected_ci_rows = (
    EXPECTED_MODELS
    *
    len(
        METRICS
    )
)


if len(
    ci_df
) != expected_ci_rows:

    raise RuntimeError(
        f"Expected {expected_ci_rows} "
        "bootstrap CI rows."
    )


ci_pass_count = 0


for _, ci_row in ci_df.iterrows():

    candidate_id = str(
        ci_row[
            "candidate_id"
        ]
    )


    metric_name = str(
        ci_row[
            "metric"
        ]
    )


    if metric_name not in METRICS:

        raise RuntimeError(
            f"Unexpected bootstrap metric: "
            f"{metric_name}"
        )


    values = bootstrap_df[
        bootstrap_df[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ][
        metric_name
    ].to_numpy(
        dtype=np.float64
    )


    if len(
        values
    ) != EXPECTED_VALID_REPLICATES:

        raise RuntimeError(
            f"{candidate_id}/{metric_name}: "
            "replicate count mismatch."
        )


    recomputed_lower = float(
        np.quantile(
            values,
            CI_LOWER,
            method="linear",
        )
    )


    recomputed_upper = float(
        np.quantile(
            values,
            CI_UPPER,
            method="linear",
        )
    )


    lower_match = close_enough(
        ci_row[
            "ci_lower_2p5"
        ],
        recomputed_lower,
        atol=1e-12,
    )


    upper_match = close_enough(
        ci_row[
            "ci_upper_97p5"
        ],
        recomputed_upper,
        atol=1e-12,
    )


    if not (
        lower_match
        and
        upper_match
    ):

        raise RuntimeError(
            f"{candidate_id}/{metric_name}: "
            "bootstrap CI mismatch."
        )


    ci_pass_count += 1


print(
    "CI rows independently verified:",
    ci_pass_count,
    "/",
    expected_ci_rows,
)

print(
    "CI quantile method:",
    "linear"
)

print(
    "95% percentile CI reproduction:",
    "PASS"
)


# ============================================================
# VERIFY BOOTSTRAP CONTRACT
# ============================================================

bootstrap_design = (
    bootstrap_contract[
        "bootstrap_design"
    ]
)


if bootstrap_design[
    "unit"
] != "ENTITY_GROUP":

    raise RuntimeError(
        "Bootstrap contract unit mismatch."
    )


if int(
    bootstrap_design[
        "seed"
    ]
) != BOOTSTRAP_SEED:

    raise RuntimeError(
        "Bootstrap contract seed mismatch."
    )


if int(
    bootstrap_design[
        "valid_replicates"
    ]
) != EXPECTED_VALID_REPLICATES:

    raise RuntimeError(
        "Bootstrap contract replicate "
        "count mismatch."
    )


if not bool(
    bootstrap_design[
        "paired_draws_across_models"
    ]
):

    raise RuntimeError(
        "Bootstrap draws are not marked "
        "paired across models."
    )


# ============================================================
# BUILD AUDIT MANIFEST
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


if len(
    audit_df
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 audit rows."
    )


models_passing = int(
    audit_df[
        "point_metric_audit_pass"
    ].sum()
)


overall_pass = bool(
    models_passing
    ==
    EXPECTED_MODELS
    and
    ci_pass_count
    ==
    expected_ci_rows
)


# ============================================================
# SAVE AUDIT MANIFEST
# ============================================================

OUTPUT_AUDIT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


audit_df.to_csv(
    OUTPUT_AUDIT_FILE,
    index=False,
)


audit_hash = sha256_file(
    OUTPUT_AUDIT_FILE
)


# ============================================================
# BUILD AUDIT CONTRACT
# ============================================================

audit_contract = {

    "contract_name":
        "Independent Cycle24 Holdout Evaluation Audit",

    "version":
        "1.0",

    "status":
        (
            "PASS"
            if overall_pass
            else
            "FAIL"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "scope":
        {
            "point_metric_recomputation":
                True,

            "threshold_application_verification":
                True,

            "confusion_matrix_verification":
                True,

            "bootstrap_structure_verification":
                True,

            "bootstrap_ci_recomputation":
                True,

            "new_bootstrap_sampling":
                False,

            "raw_holdout_dataset_read":
                False,

            "model_files_loaded":
                False,

            "model_retraining":
                False,

            "threshold_reoptimization":
                False,

            "cycle25_read":
                False,
        },

    "population_verified":
        {
            "rows":
                EXPECTED_ROWS,

            "positive":
                EXPECTED_POSITIVES,

            "negative":
                EXPECTED_NEGATIVES,

            "entity_groups":
                EXPECTED_ENTITY_GROUPS,
        },

    "models_audited":
        EXPECTED_MODELS,

    "models_passing":
        models_passing,

    "bootstrap_verified":
        {
            "unit":
                "ENTITY_GROUP",

            "valid_replicates":
                EXPECTED_VALID_REPLICATES,

            "rows":
                EXPECTED_BOOTSTRAP_ROWS,

            "groups_drawn_per_replicate":
                EXPECTED_ENTITY_GROUPS,

            "paired_draws_across_models":
                True,

            "seed":
                BOOTSTRAP_SEED,

            "ci_rows_verified":
                ci_pass_count,

            "expected_ci_rows":
                expected_ci_rows,
        },

    "overall_holdout_evaluation_audit_pass":
        overall_pass,

    "audit_manifest":
        {
            "file":
                str(
                    OUTPUT_AUDIT_FILE
                ),

            "sha256":
                audit_hash,
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
                        prediction_hash,
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

            "holdout_protocol":
                {
                    "file":
                        str(
                            PROTOCOL_FILE
                        ),

                    "sha256":
                        sha256_file(
                            PROTOCOL_FILE
                        ),
                },

            "point_metrics":
                {
                    "file":
                        str(
                            POINT_METRICS_FILE
                        ),

                    "sha256":
                        metrics_hash,
                },

            "point_metrics_contract":
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

            "bootstrap_replicates":
                {
                    "file":
                        str(
                            BOOTSTRAP_REPLICATES_FILE
                        ),

                    "sha256":
                        bootstrap_replicates_hash,
                },

            "bootstrap_ci":
                {
                    "file":
                        str(
                            BOOTSTRAP_CI_FILE
                        ),

                    "sha256":
                        bootstrap_ci_hash,
                },

            "bootstrap_contract":
                {
                    "file":
                        str(
                            BOOTSTRAP_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            BOOTSTRAP_CONTRACT_FILE
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

    "interpretation":
        (
            "The independent audit verifies that the "
            "reported Cycle24 holdout point estimates "
            "are exactly reproducible from the frozen "
            "prediction artifact and frozen thresholds, "
            "and that the reported confidence intervals "
            "are reproducible from the saved connected-"
            "ENTITY_GROUP bootstrap replicates. No model "
            "fitting, threshold optimization, new bootstrap "
            "sampling, or Cycle25 access is performed."
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
        audit_contract,
        file,
        indent=2,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "AUDIT SUMMARY"
)

print(
    "===================================="
)


print(
    "Models audited:",
    EXPECTED_MODELS
)

print(
    "Models passing:",
    models_passing,
    "/",
    EXPECTED_MODELS,
)

print(
    "Bootstrap replicates verified:",
    EXPECTED_VALID_REPLICATES
)

print(
    "Bootstrap CI rows verified:",
    ci_pass_count,
    "/",
    expected_ci_rows,
)

print(
    "Artifact hash chain:",
    "PASS"
)


print(
    "\n===================================="
)

print(
    "DATA / DEVELOPMENT ISOLATION"
)

print(
    "===================================="
)


print(
    "Raw holdout dataset reread:",
    False
)

print(
    "Model files loaded:",
    False
)

print(
    "New model fitting:",
    False
)

print(
    "Threshold reoptimization:",
    False
)

print(
    "New bootstrap sampling:",
    False
)

print(
    "Cycle25 read:",
    False
)


print(
    "\n===================================="
)

print(
    "CHAPTER 7.5 RESULT"
)

print(
    "===================================="
)


print(
    "Independent holdout evaluation audit:",
    (
        "PASS"
        if overall_pass
        else
        "FAIL"
    )
)


print(
    "\nSaved:"
)

print(
    OUTPUT_AUDIT_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


if not overall_pass:

    raise RuntimeError(
        "Chapter 7.5 independent "
        "holdout evaluation audit failed."
    )


print(
    "\nMODEL CHAPTER 7.5 COMPLETE."
)