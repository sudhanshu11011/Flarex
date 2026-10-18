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

CYCLE25_PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_evaluation_protocol.json"
)

CYCLE25_PREDICTION_FILE = Path(
    "artifacts/models/evaluation/predictions/"
    "cycle25_predictions.parquet"
)

CYCLE25_PREDICTION_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_prediction_contract.json"
)

CYCLE25_METRICS_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "cycle25_metrics.csv"
)

CYCLE25_METRICS_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_metrics_contract.json"
)

CYCLE25_BOOTSTRAP_FILE = Path(
    "artifacts/models/evaluation/bootstrap/"
    "cycle25_group_bootstrap_replicates.parquet"
)

CYCLE25_BOOTSTRAP_CI_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "cycle25_bootstrap_ci.csv"
)

CYCLE25_BOOTSTRAP_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_bootstrap_contract.json"
)

CYCLE24_METRICS_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "holdout_metrics.csv"
)

CYCLE24_BOOTSTRAP_FILE = Path(
    "artifacts/models/evaluation/bootstrap/"
    "holdout_group_bootstrap_replicates.parquet"
)

DRIFT_DELTA_FILE = Path(
    "artifacts/models/evaluation/bootstrap/"
    "cross_cycle_metric_delta_replicates.parquet"
)

DRIFT_SUMMARY_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "cross_cycle_predictive_drift.csv"
)

DRIFT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cross_cycle_predictive_drift_contract.json"
)

OUTPUT_AUDIT_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_evaluation_audit.csv"
)

OUTPUT_AUDIT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_evaluation_audit.json"
)


# ============================================================
# FROZEN EXPECTATIONS
# ============================================================

EXPECTED_ROWS = 79361
EXPECTED_POSITIVE = 4836
EXPECTED_NEGATIVE = 74525

EXPECTED_MODELS = {
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
}

SCOPES = [
    "C25_B1_2020_2021",
    "C25_B2_2022_2023",
    "C25_B3_2024_2025",
    "C25_ALL_2020_2025",
]

PRIMARY_BLOCKS = [
    "C25_B1_2020_2021",
    "C25_B2_2022_2023",
    "C25_B3_2024_2025",
]

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

VALID_REPLICATES = 2000

EXPECTED_C25_METRIC_ROWS = 12
EXPECTED_C25_CI_ROWS = 108
EXPECTED_DRIFT_ROWS = 108
EXPECTED_DELTA_ROWS = 216000

ATOL = 1e-12


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


def compute_metrics(
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


    tss = float(
        sensitivity - fpr
    )


    hss_denominator = (
        (tp + fn)
        *
        (fn + tn)
        +
        (tp + fp)
        *
        (fp + tn)
    )


    hss = (
        0.0
        if hss_denominator == 0
        else
        float(
            2.0
            *
            (
                tp * tn
                -
                fn * fp
            )
            /
            hss_denominator
        )
    )


    f1_denominator = (
        2 * tp
        +
        fp
        +
        fn
    )


    f1 = (
        0.0
        if f1_denominator == 0
        else
        float(
            (2.0 * tp)
            /
            f1_denominator
        )
    )


    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tss": tss,
        "hss": hss,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
    }


def classify_tss(
    lower,
    upper,
):

    if upper < 0.0:
        return "MEASURABLE_DECREASE"

    if lower > 0.0:
        return "MEASURABLE_INCREASE"

    return "NO_MEASURABLE_CHANGE_AT_95CI"


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 8.6 — INDEPENDENT CYCLE25 EVALUATION AUDIT"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    CYCLE25_PROTOCOL_FILE,
    CYCLE25_PREDICTION_FILE,
    CYCLE25_PREDICTION_CONTRACT_FILE,
    CYCLE25_METRICS_FILE,
    CYCLE25_METRICS_CONTRACT_FILE,
    CYCLE25_BOOTSTRAP_FILE,
    CYCLE25_BOOTSTRAP_CI_FILE,
    CYCLE25_BOOTSTRAP_CONTRACT_FILE,
    CYCLE24_METRICS_FILE,
    CYCLE24_BOOTSTRAP_FILE,
    DRIFT_DELTA_FILE,
    DRIFT_SUMMARY_FILE,
    DRIFT_CONTRACT_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required audit input: {path}"
        )


for output_path in [
    OUTPUT_AUDIT_FILE,
    OUTPUT_AUDIT_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Audit output already exists: "
            f"{output_path}"
        )


# ============================================================
# CONTRACT STATUS
# ============================================================

protocol = load_json(
    CYCLE25_PROTOCOL_FILE
)

prediction_contract = load_json(
    CYCLE25_PREDICTION_CONTRACT_FILE
)

metrics_contract = load_json(
    CYCLE25_METRICS_CONTRACT_FILE
)

bootstrap_contract = load_json(
    CYCLE25_BOOTSTRAP_CONTRACT_FILE
)

drift_contract = load_json(
    DRIFT_CONTRACT_FILE
)


if protocol.get(
    "status"
) != "FROZEN_BEFORE_CYCLE25_EVALUATION":

    raise RuntimeError(
        "Cycle25 protocol status mismatch."
    )


if prediction_contract.get(
    "status"
) != "FROZEN_CYCLE25_PREDICTIONS":

    raise RuntimeError(
        "Cycle25 prediction contract "
        "status mismatch."
    )


if metrics_contract.get(
    "status"
) != "FROZEN_CYCLE25_POINT_METRICS":

    raise RuntimeError(
        "Cycle25 metric contract "
        "status mismatch."
    )


if bootstrap_contract.get(
    "status"
) != "FROZEN_CYCLE25_BOOTSTRAP_CI":

    raise RuntimeError(
        "Cycle25 bootstrap contract "
        "status mismatch."
    )


if drift_contract.get(
    "status"
) != "FROZEN_CROSS_CYCLE_PREDICTIVE_DRIFT":

    raise RuntimeError(
        "Cross-cycle drift contract "
        "status mismatch."
    )


print(
    "Cycle25 protocol: VERIFIED"
)

print(
    "Prediction contract: VERIFIED"
)

print(
    "Point metrics contract: VERIFIED"
)

print(
    "Bootstrap contract: VERIFIED"
)

print(
    "Predictive-drift contract: VERIFIED"
)


# ============================================================
# HASH CHAIN
# ============================================================

prediction_hash = sha256_file(
    CYCLE25_PREDICTION_FILE
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
        "Cycle25 prediction hash mismatch."
    )


metrics_hash = sha256_file(
    CYCLE25_METRICS_FILE
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
        "Cycle25 metric hash mismatch."
    )


bootstrap_hash = sha256_file(
    CYCLE25_BOOTSTRAP_FILE
)

if (
    bootstrap_hash
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
        "Cycle25 bootstrap hash mismatch."
    )


ci_hash = sha256_file(
    CYCLE25_BOOTSTRAP_CI_FILE
)

if (
    ci_hash
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
        "Cycle25 bootstrap CI hash mismatch."
    )


delta_hash = sha256_file(
    DRIFT_DELTA_FILE
)

if (
    delta_hash
    !=
    drift_contract[
        "artifacts"
    ][
        "delta_bootstrap_replicates"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cross-cycle delta artifact "
        "hash mismatch."
    )


drift_hash = sha256_file(
    DRIFT_SUMMARY_FILE
)

if (
    drift_hash
    !=
    drift_contract[
        "artifacts"
    ][
        "predictive_drift_summary"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cross-cycle drift summary "
        "hash mismatch."
    )


print(
    "Artifact hash chain: PASS"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

predictions = pd.read_parquet(
    CYCLE25_PREDICTION_FILE
)

c25_metrics = pd.read_csv(
    CYCLE25_METRICS_FILE,
    keep_default_na=False,
)

c25_bootstrap = pd.read_parquet(
    CYCLE25_BOOTSTRAP_FILE
)

c25_ci = pd.read_csv(
    CYCLE25_BOOTSTRAP_CI_FILE,
    keep_default_na=False,
)

c24_metrics = pd.read_csv(
    CYCLE24_METRICS_FILE,
    keep_default_na=False,
)

c24_bootstrap = pd.read_parquet(
    CYCLE24_BOOTSTRAP_FILE
)

delta_saved = pd.read_parquet(
    DRIFT_DELTA_FILE
)

drift_saved = pd.read_csv(
    DRIFT_SUMMARY_FILE,
    keep_default_na=False,
)


# ============================================================
# POPULATION
# ============================================================

if len(
    predictions
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Cycle25 row count mismatch."
    )


target = predictions[
    "target"
].to_numpy(
    dtype=np.int64
)


if int(
    target.sum()
) != EXPECTED_POSITIVE:

    raise RuntimeError(
        "Cycle25 positive count mismatch."
    )


if int(
    (target == 0).sum()
) != EXPECTED_NEGATIVE:

    raise RuntimeError(
        "Cycle25 negative count mismatch."
    )


if set(
    predictions[
        "primary_block_id"
    ].astype(str).unique()
) != set(
    PRIMARY_BLOCKS
):

    raise RuntimeError(
        "Cycle25 temporal block set mismatch."
    )


print(
    "Cycle25 population integrity: PASS"
)


# ============================================================
# INDEPENDENT POINT METRIC AUDIT
# ============================================================

point_audit_rows = []


print(
    "\n===================================="
)

print(
    "INDEPENDENT POINT-METRIC AUDIT"
)

print(
    "===================================="
)


for scope_id in SCOPES:

    if scope_id in PRIMARY_BLOCKS:

        mask = (
            predictions[
                "primary_block_id"
            ].astype(str)
            ==
            scope_id
        ).to_numpy()

    else:

        mask = np.ones(
            len(
                predictions
            ),
            dtype=bool,
        )


    y_scope = target[
        mask
    ]


    for candidate_id in sorted(
        EXPECTED_MODELS
    ):

        official = c25_metrics[
            (
                c25_metrics[
                    "scope_id"
                ].astype(str)
                ==
                scope_id
            )
            &
            (
                c25_metrics[
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
                "official point row missing."
            )


        official = official.iloc[
            0
        ]


        threshold = float(
            official[
                "frozen_threshold"
            ]
        )


        probability_column = (
            f"{candidate_id}"
            "__probability"
        )


        stored_prediction_column = (
            f"{candidate_id}"
            "__prediction"
        )


        probabilities = predictions.loc[
            mask,
            probability_column
        ].to_numpy(
            dtype=np.float64
        )


        stored_classes = predictions.loc[
            mask,
            stored_prediction_column
        ].to_numpy(
            dtype=np.int64
        )


        recomputed_classes = (
            probabilities
            >=
            threshold
        ).astype(
            np.int64
        )


        threshold_pass = bool(
            np.array_equal(
                stored_classes,
                recomputed_classes,
            )
        )


        recomputed = compute_metrics(
            y_scope,
            probabilities,
            recomputed_classes,
        )


        confusion_pass = all(
            int(
                official[
                    key
                ]
            )
            ==
            int(
                recomputed[
                    key
                ]
            )
            for key in [
                "tp",
                "fp",
                "tn",
                "fn",
            ]
        )


        metrics_match = all(
            np.isclose(
                float(
                    official[
                        metric
                    ]
                ),
                float(
                    recomputed[
                        metric
                    ]
                ),
                rtol=0.0,
                atol=ATOL,
            )
            for metric in METRICS
        )


        overall_pass = bool(
            threshold_pass
            and
            confusion_pass
            and
            metrics_match
        )


        point_audit_rows.append(
            {
                "scope_id":
                    scope_id,

                "candidate_id":
                    candidate_id,

                "threshold_pass":
                    threshold_pass,

                "confusion_pass":
                    confusion_pass,

                "metrics_match":
                    metrics_match,

                "tss":
                    recomputed[
                        "tss"
                    ],

                "pass":
                    overall_pass,
            }
        )


        print(
            f"{scope_id} | "
            f"{candidate_id}: "
            f"threshold=PASS, "
            f"confusion={'PASS' if confusion_pass else 'FAIL'}, "
            f"TSS={recomputed['tss']:.6f}, "
            f"metrics_match={metrics_match}, "
            f"PASS={overall_pass}"
        )


point_audit_df = pd.DataFrame(
    point_audit_rows
)


if len(
    point_audit_df
) != 12:

    raise RuntimeError(
        "Expected exactly 12 point "
        "metric audit rows."
    )


if not point_audit_df[
    "pass"
].all():

    raise RuntimeError(
        "Independent Cycle25 point "
        "metric audit failed."
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
    c25_bootstrap
) != 24000:

    raise RuntimeError(
        "Cycle25 bootstrap row count "
        "must be 24000."
    )


for scope_id in SCOPES:

    subset = c25_bootstrap[
        c25_bootstrap[
            "scope_id"
        ].astype(str)
        ==
        scope_id
    ]


    if subset[
        "replicate_id"
    ].nunique() != VALID_REPLICATES:

        raise RuntimeError(
            f"{scope_id}: bootstrap "
            "replicate count mismatch."
        )


    counts = (
        subset.groupby(
            "replicate_id"
        ).size()
    )


    if not (
        counts == 3
    ).all():

        raise RuntimeError(
            f"{scope_id}: each replicate "
            "must contain 3 models."
        )


    if (
        subset[
            "bootstrap_positive_weight"
        ]
        <=
        0
    ).any():

        raise RuntimeError(
            f"{scope_id}: invalid "
            "positive-weight bootstrap."
        )


    if (
        subset[
            "bootstrap_negative_weight"
        ]
        <=
        0
    ).any():

        raise RuntimeError(
            f"{scope_id}: invalid "
            "negative-weight bootstrap."
        )


    for metric in METRICS:

        if not np.isfinite(
            subset[
                metric
            ].to_numpy(
                dtype=np.float64
            )
        ).all():

            raise RuntimeError(
                f"{scope_id}/{metric}: "
                "non-finite bootstrap values."
            )


print(
    "Bootstrap rows: 24000"
)

print(
    "Valid replicates per scope: 2000"
)

print(
    "Models per replicate: 3"
)

print(
    "Class-valid replicates: PASS"
)


# ============================================================
# INDEPENDENT CYCLE25 CI AUDIT
# ============================================================

ci_verified = 0


print(
    "\n===================================="
)

print(
    "INDEPENDENT CYCLE25 CI AUDIT"
)

print(
    "===================================="
)


for _, official in c25_ci.iterrows():

    scope_id = str(
        official[
            "scope_id"
        ]
    )

    candidate_id = str(
        official[
            "candidate_id"
        ]
    )

    metric = str(
        official[
            "metric"
        ]
    )


    values = c25_bootstrap[
        (
            c25_bootstrap[
                "scope_id"
            ].astype(str)
            ==
            scope_id
        )
        &
        (
            c25_bootstrap[
                "candidate_id"
            ].astype(str)
            ==
            candidate_id
        )
    ][
        metric
    ].to_numpy(
        dtype=np.float64
    )


    lower = float(
        np.quantile(
            values,
            0.025,
            method="linear",
        )
    )


    upper = float(
        np.quantile(
            values,
            0.975,
            method="linear",
        )
    )


    if not np.isclose(
        lower,
        float(
            official[
                "ci_lower_2p5"
            ]
        ),
        rtol=0.0,
        atol=ATOL,
    ):

        raise RuntimeError(
            f"{scope_id}/{candidate_id}/"
            f"{metric}: lower CI mismatch."
        )


    if not np.isclose(
        upper,
        float(
            official[
                "ci_upper_97p5"
            ]
        ),
        rtol=0.0,
        atol=ATOL,
    ):

        raise RuntimeError(
            f"{scope_id}/{candidate_id}/"
            f"{metric}: upper CI mismatch."
        )


    ci_verified += 1


if ci_verified != EXPECTED_C25_CI_ROWS:

    raise RuntimeError(
        "Cycle25 CI verification "
        "row count mismatch."
    )


print(
    f"Cycle25 CI rows verified: "
    f"{ci_verified} / {EXPECTED_C25_CI_ROWS}"
)

print(
    "95% percentile CI reproduction: PASS"
)


# ============================================================
# INDEPENDENT DELTA-BOOTSTRAP AUDIT
# ============================================================

print(
    "\n===================================="
)

print(
    "INDEPENDENT CROSS-CYCLE DELTA AUDIT"
)

print(
    "===================================="
)


if len(
    delta_saved
) != EXPECTED_DELTA_ROWS:

    raise RuntimeError(
        "Cross-cycle delta replicate "
        "row count mismatch."
    )


delta_rows_verified = 0
summary_rows_verified = 0
tss_classifications_verified = 0


for scope_id in SCOPES:

    for candidate_id in sorted(
        EXPECTED_MODELS
    ):

        c24_boot = c24_bootstrap[
            c24_bootstrap[
                "candidate_id"
            ].astype(str)
            ==
            candidate_id
        ].sort_values(
            "replicate_id"
        ).reset_index(
            drop=True
        )


        c25_boot = c25_bootstrap[
            (
                c25_bootstrap[
                    "scope_id"
                ].astype(str)
                ==
                scope_id
            )
            &
            (
                c25_bootstrap[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            )
        ].sort_values(
            "replicate_id"
        ).reset_index(
            drop=True
        )


        if len(
            c24_boot
        ) != VALID_REPLICATES:

            raise RuntimeError(
                f"{candidate_id}: Cycle24 "
                "bootstrap count mismatch."
            )


        if len(
            c25_boot
        ) != VALID_REPLICATES:

            raise RuntimeError(
                f"{scope_id}/{candidate_id}: "
                "Cycle25 bootstrap count mismatch."
            )


        if not np.array_equal(
            c24_boot[
                "replicate_id"
            ].to_numpy(),
            c25_boot[
                "replicate_id"
            ].to_numpy(),
        ):

            raise RuntimeError(
                f"{scope_id}/{candidate_id}: "
                "replicate alignment mismatch."
            )


        for metric in METRICS:

            c24_values = c24_boot[
                metric
            ].to_numpy(
                dtype=np.float64
            )


            c25_values = c25_boot[
                metric
            ].to_numpy(
                dtype=np.float64
            )


            expected_delta = (
                c25_values
                -
                c24_values
            )


            stored_delta = delta_saved[
                (
                    delta_saved[
                        "scope_id"
                    ].astype(str)
                    ==
                    scope_id
                )
                &
                (
                    delta_saved[
                        "candidate_id"
                    ].astype(str)
                    ==
                    candidate_id
                )
                &
                (
                    delta_saved[
                        "metric"
                    ].astype(str)
                    ==
                    metric
                )
            ].sort_values(
                "replicate_id"
            )


            if len(
                stored_delta
            ) != VALID_REPLICATES:

                raise RuntimeError(
                    f"{scope_id}/{candidate_id}/"
                    f"{metric}: saved delta "
                    "replicate count mismatch."
                )


            actual_delta = stored_delta[
                "delta_cycle25_minus_cycle24"
            ].to_numpy(
                dtype=np.float64
            )


            if not np.allclose(
                expected_delta,
                actual_delta,
                rtol=0.0,
                atol=ATOL,
            ):

                raise RuntimeError(
                    f"{scope_id}/{candidate_id}/"
                    f"{metric}: delta replicate "
                    "values mismatch."
                )


            delta_rows_verified += (
                VALID_REPLICATES
            )


            lower = float(
                np.quantile(
                    expected_delta,
                    0.025,
                    method="linear",
                )
            )


            upper = float(
                np.quantile(
                    expected_delta,
                    0.975,
                    method="linear",
                )
            )


            c24_point = c24_metrics[
                c24_metrics[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            ]


            c25_point = c25_metrics[
                (
                    c25_metrics[
                        "scope_id"
                    ].astype(str)
                    ==
                    scope_id
                )
                &
                (
                    c25_metrics[
                        "candidate_id"
                    ].astype(str)
                    ==
                    candidate_id
                )
            ]


            if len(
                c24_point
            ) != 1 or len(
                c25_point
            ) != 1:

                raise RuntimeError(
                    "Point-metric lookup failed."
                )


            point_delta = float(
                c25_point.iloc[
                    0
                ][
                    metric
                ]
                -
                c24_point.iloc[
                    0
                ][
                    metric
                ]
            )


            official_summary = drift_saved[
                (
                    drift_saved[
                        "scope_id"
                    ].astype(str)
                    ==
                    scope_id
                )
                &
                (
                    drift_saved[
                        "candidate_id"
                    ].astype(str)
                    ==
                    candidate_id
                )
                &
                (
                    drift_saved[
                        "metric"
                    ].astype(str)
                    ==
                    metric
                )
            ]


            if len(
                official_summary
            ) != 1:

                raise RuntimeError(
                    f"{scope_id}/{candidate_id}/"
                    f"{metric}: drift summary "
                    "row missing."
                )


            official_summary = (
                official_summary.iloc[
                    0
                ]
            )


            if not np.isclose(
                point_delta,
                float(
                    official_summary[
                        "delta_cycle25_minus_cycle24"
                    ]
                ),
                rtol=0.0,
                atol=ATOL,
            ):

                raise RuntimeError(
                    f"{scope_id}/{candidate_id}/"
                    f"{metric}: point delta mismatch."
                )


            if not np.isclose(
                lower,
                float(
                    official_summary[
                        "delta_ci_lower_2p5"
                    ]
                ),
                rtol=0.0,
                atol=ATOL,
            ):

                raise RuntimeError(
                    f"{scope_id}/{candidate_id}/"
                    f"{metric}: delta lower "
                    "CI mismatch."
                )


            if not np.isclose(
                upper,
                float(
                    official_summary[
                        "delta_ci_upper_97p5"
                    ]
                ),
                rtol=0.0,
                atol=ATOL,
            ):

                raise RuntimeError(
                    f"{scope_id}/{candidate_id}/"
                    f"{metric}: delta upper "
                    "CI mismatch."
                )


            summary_rows_verified += 1


            if metric == "tss":

                expected_classification = (
                    classify_tss(
                        lower,
                        upper,
                    )
                )


                actual_classification = str(
                    official_summary[
                        "predictive_change_classification"
                    ]
                )


                if (
                    expected_classification
                    !=
                    actual_classification
                ):

                    raise RuntimeError(
                        f"{scope_id}/{candidate_id}: "
                        "TSS classification mismatch."
                    )


                tss_classifications_verified += 1


if delta_rows_verified != EXPECTED_DELTA_ROWS:

    raise RuntimeError(
        "Delta replicate verification "
        "count mismatch."
    )


if summary_rows_verified != EXPECTED_DRIFT_ROWS:

    raise RuntimeError(
        "Drift summary verification "
        "count mismatch."
    )


if tss_classifications_verified != 12:

    raise RuntimeError(
        "TSS classification verification "
        "count mismatch."
    )


print(
    f"Delta-bootstrap rows verified: "
    f"{delta_rows_verified} / "
    f"{EXPECTED_DELTA_ROWS}"
)

print(
    f"Drift summary rows verified: "
    f"{summary_rows_verified} / "
    f"{EXPECTED_DRIFT_ROWS}"
)

print(
    "TSS classifications verified: 12 / 12"
)


# ============================================================
# SAVE AUDIT MANIFEST
# ============================================================

OUTPUT_AUDIT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_AUDIT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


point_audit_df.to_csv(
    OUTPUT_AUDIT_FILE,
    index=False,
)


audit_manifest_hash = sha256_file(
    OUTPUT_AUDIT_FILE
)


# ============================================================
# FINAL STATUS
# ============================================================

overall_pass = bool(
    point_audit_df[
        "pass"
    ].all()
    and
    ci_verified
    ==
    EXPECTED_C25_CI_ROWS
    and
    delta_rows_verified
    ==
    EXPECTED_DELTA_ROWS
    and
    summary_rows_verified
    ==
    EXPECTED_DRIFT_ROWS
    and
    tss_classifications_verified
    ==
    12
)


if not overall_pass:

    raise RuntimeError(
        "Independent Cycle25 evaluation "
        "audit failed."
    )


# ============================================================
# AUDIT CONTRACT
# ============================================================

audit_contract = {

    "contract_name":
        "Independent Cycle25 Predictive Evaluation Audit",

    "version":
        "1.0",

    "status":
        "PASS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_8_6",

    "overall_cycle25_evaluation_audit_pass":
        True,

    "verification":
        {
            "point_metric_rows":
                12,

            "point_metric_rows_passing":
                12,

            "cycle25_bootstrap_ci_rows_verified":
                ci_verified,

            "cycle25_bootstrap_expected_ci_rows":
                EXPECTED_C25_CI_ROWS,

            "cross_cycle_delta_rows_verified":
                delta_rows_verified,

            "cross_cycle_delta_expected_rows":
                EXPECTED_DELTA_ROWS,

            "cross_cycle_summary_rows_verified":
                summary_rows_verified,

            "cross_cycle_summary_expected_rows":
                EXPECTED_DRIFT_ROWS,

            "tss_classifications_verified":
                tss_classifications_verified,

            "tss_classifications_expected":
                12,

            "artifact_hash_chain":
                "PASS",
        },

    "audit_scope":
        {
            "point_metrics_independently_recomputed":
                True,

            "cycle25_ci_independently_recomputed":
                True,

            "cross_cycle_delta_values_independently_recomputed":
                True,

            "cross_cycle_delta_ci_independently_recomputed":
                True,

            "tss_classifications_independently_recomputed":
                True,

            "bootstrap_samples_regenerated":
                False,

            "bootstrap_rng_sequence_reproduced":
                False,
        },

    "development_isolation":
        {
            "raw_cycle25_dataset_reread":
                False,

            "raw_cycle24_holdout_reread":
                False,

            "model_files_loaded":
                False,

            "model_retraining":
                False,

            "preprocessing_refit":
                False,

            "threshold_reoptimization":
                False,

            "new_bootstrap_sampling":
                False,

            "temporal_blocks_changed":
                False,

            "shap_analysis_performed":
                False,
        },

    "audit_manifest":
        {
            "file":
                str(
                    OUTPUT_AUDIT_FILE
                ),

            "sha256":
                audit_manifest_hash,

            "rows":
                len(
                    point_audit_df
                ),
        },

    "provenance":
        {
            "cycle25_prediction_artifact_sha256":
                prediction_hash,

            "cycle25_metrics_sha256":
                metrics_hash,

            "cycle25_bootstrap_sha256":
                bootstrap_hash,

            "cycle25_bootstrap_ci_sha256":
                ci_hash,

            "cross_cycle_delta_sha256":
                delta_hash,

            "cross_cycle_drift_summary_sha256":
                drift_hash,
        },

    "interpretation_constraints":
        {
            "no_measurable_change_is_not_equivalence":
                True,

            "predictive_stability_not_yet_declared_as_equivalence":
                True,

            "predictive_results_do_not_imply_shap_results":
                True,

            "shap_not_causal_physics":
                True,
        },
}


with OUTPUT_AUDIT_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        audit_contract,
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
    "AUDIT SUMMARY"
)

print(
    "===================================="
)


print(
    "Point-metric rows passing:",
    "12 / 12"
)

print(
    "Cycle25 bootstrap CI rows verified:",
    f"{ci_verified} / {EXPECTED_C25_CI_ROWS}"
)

print(
    "Cross-cycle delta rows verified:",
    f"{delta_rows_verified} / {EXPECTED_DELTA_ROWS}"
)

print(
    "Cross-cycle summary rows verified:",
    f"{summary_rows_verified} / {EXPECTED_DRIFT_ROWS}"
)

print(
    "TSS classifications verified:",
    "12 / 12"
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
    "Raw Cycle25 dataset reread:",
    False
)

print(
    "Raw Cycle24 holdout reread:",
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
    "SHAP analysis:",
    False
)


print(
    "\n===================================="
)

print(
    "CHAPTER 8.6 RESULT"
)

print(
    "===================================="
)


print(
    "Independent Cycle25 evaluation audit:",
    "PASS"
)


print(
    "\nSaved:"
)

print(
    OUTPUT_AUDIT_FILE
)

print(
    OUTPUT_AUDIT_CONTRACT_FILE
)


print(
    "\nMODEL CHAPTER 8.6 COMPLETE."
)