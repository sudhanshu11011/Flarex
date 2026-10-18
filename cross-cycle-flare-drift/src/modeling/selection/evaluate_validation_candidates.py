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
    "artifacts/models/selection/predictions/"
    "cycle24_validation_predictions.parquet"
)

PREDICTION_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "validation_prediction_contract.json"
)

SELECTION_PROTOCOL_FILE = Path(
    "artifacts/models/selection/contracts/"
    "model_selection_protocol.json"
)

CANDIDATE_REGISTRY_FILE = Path(
    "artifacts/models/selection/manifests/"
    "frozen_candidate_registry.csv"
)

THRESHOLD_SWEEP_FILE = Path(
    "artifacts/models/selection/metrics/"
    "validation_threshold_sweep.parquet"
)

CANDIDATE_METRICS_FILE = Path(
    "artifacts/models/selection/manifests/"
    "validation_candidate_metrics.csv"
)

METRICS_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "validation_metrics_contract.json"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path):

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
        (tp * tn)
        -
        (fn * fp)
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
        (2 * tp)
        /
        denominator
    )


# ============================================================
# EXACT THRESHOLD SWEEP
# ============================================================

def exact_threshold_sweep(
    y_true,
    probabilities,
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )


    if y_true.shape != probabilities.shape:

        raise RuntimeError(
            "Target and probability vector "
            "shapes differ."
        )


    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            "Non-finite probabilities detected."
        )


    positives = int(
        y_true.sum()
    )

    negatives = int(
        len(y_true)
        -
        positives
    )


    if positives <= 0:

        raise RuntimeError(
            "Threshold sweep requires positive "
            "validation samples."
        )


    if negatives <= 0:

        raise RuntimeError(
            "Threshold sweep requires negative "
            "validation samples."
        )


    # --------------------------------------------------------
    # Sort from highest probability to lowest probability.
    #
    # Stable sorting is used for deterministic handling of
    # repeated probability values.
    # --------------------------------------------------------

    order = np.argsort(
        -probabilities,
        kind="mergesort",
    )


    sorted_probabilities = probabilities[
        order
    ]


    sorted_targets = y_true[
        order
    ]


    cumulative_tp = np.cumsum(
        sorted_targets,
        dtype=np.int64,
    )


    cumulative_fp = np.cumsum(
        1 - sorted_targets,
        dtype=np.int64,
    )


    # --------------------------------------------------------
    # A threshold must include every sample whose probability
    # equals that threshold.
    #
    # Therefore evaluate only at the LAST index of each unique
    # probability value.
    # --------------------------------------------------------

    if len(
        sorted_probabilities
    ) == 1:

        group_end_indices = np.array(
            [
                0,
            ],
            dtype=np.int64,
        )

    else:

        change_indices = np.flatnonzero(
            sorted_probabilities[
                :-1
            ]
            !=
            sorted_probabilities[
                1:
            ]
        )


        group_end_indices = np.concatenate(
            [
                change_indices,
                np.array(
                    [
                        len(
                            sorted_probabilities
                        )
                        -
                        1
                    ],
                    dtype=np.int64,
                ),
            ]
        )


    thresholds = sorted_probabilities[
        group_end_indices
    ]


    tp = cumulative_tp[
        group_end_indices
    ]


    fp = cumulative_fp[
        group_end_indices
    ]


    fn = (
        positives
        -
        tp
    )


    tn = (
        negatives
        -
        fp
    )


    # --------------------------------------------------------
    # Frozen protocol requires an all-negative sentinel
    # immediately above maximum predicted probability.
    # --------------------------------------------------------

    sentinel_threshold = np.nextafter(
        float(
            sorted_probabilities[
                0
            ]
        ),
        np.inf,
    )


    thresholds = np.concatenate(
        [
            np.array(
                [
                    sentinel_threshold,
                ],
                dtype=np.float64,
            ),
            thresholds.astype(
                np.float64
            ),
        ]
    )


    tp = np.concatenate(
        [
            np.array(
                [
                    0,
                ],
                dtype=np.int64,
            ),
            tp,
        ]
    )


    fp = np.concatenate(
        [
            np.array(
                [
                    0,
                ],
                dtype=np.int64,
            ),
            fp,
        ]
    )


    fn = np.concatenate(
        [
            np.array(
                [
                    positives,
                ],
                dtype=np.int64,
            ),
            fn,
        ]
    )


    tn = np.concatenate(
        [
            np.array(
                [
                    negatives,
                ],
                dtype=np.int64,
            ),
            tn,
        ]
    )


    # --------------------------------------------------------
    # Exact TSS comparison.
    #
    # TSS = TP/P - FP/N
    #
    # Since P and N are fixed:
    #
    # TSS numerator =
    # TP*N - FP*P
    #
    # Comparing this INTEGER numerator avoids floating-point
    # ambiguity when applying the exact tie rule.
    # --------------------------------------------------------

    tss_numerator = (
        tp
        *
        negatives
        -
        fp
        *
        positives
    ).astype(
        np.int64
    )


    tss_denominator = (
        positives
        *
        negatives
    )


    tss = (
        tss_numerator.astype(
            np.float64
        )
        /
        float(
            tss_denominator
        )
    )


    sensitivity = (
        tp.astype(
            np.float64
        )
        /
        float(
            positives
        )
    )


    fpr = (
        fp.astype(
            np.float64
        )
        /
        float(
            negatives
        )
    )


    specificity = (
        tn.astype(
            np.float64
        )
        /
        float(
            negatives
        )
    )


    predicted_positive = (
        tp
        +
        fp
    )


    precision = np.divide(
        tp.astype(
            np.float64
        ),
        predicted_positive.astype(
            np.float64
        ),
        out=np.zeros_like(
            tp,
            dtype=np.float64,
        ),
        where=(
            predicted_positive
            !=
            0
        ),
    )


    f1_denominator = (
        2 * tp
        +
        fp
        +
        fn
    )


    f1 = np.divide(
        2
        *
        tp.astype(
            np.float64
        ),
        f1_denominator.astype(
            np.float64
        ),
        out=np.zeros_like(
            tp,
            dtype=np.float64,
        ),
        where=(
            f1_denominator
            !=
            0
        ),
    )


    hss_numerator = (
        2
        *
        (
            tp * tn
            -
            fn * fp
        )
    ).astype(
        np.float64
    )


    hss_denominator = (
        (tp + fn)
        *
        (fn + tn)
        +
        (tp + fp)
        *
        (fp + tn)
    ).astype(
        np.float64
    )


    hss = np.divide(
        hss_numerator,
        hss_denominator,
        out=np.zeros_like(
            hss_numerator,
            dtype=np.float64,
        ),
        where=(
            hss_denominator
            !=
            0
        ),
    )


    max_tss_numerator = int(
        tss_numerator.max()
    )


    tied_indices = np.flatnonzero(
        tss_numerator
        ==
        max_tss_numerator
    )


    if len(
        tied_indices
    ) == 0:

        raise RuntimeError(
            "No maximum-TSS threshold found."
        )


    # Threshold array is descending.
    # First exact-TSS tie is therefore highest threshold.
    selected_index = int(
        tied_indices[
            0
        ]
    )


    selected_threshold = float(
        thresholds[
            selected_index
        ]
    )


    return {
        "thresholds":
            thresholds,

        "tp":
            tp,

        "fp":
            fp,

        "tn":
            tn,

        "fn":
            fn,

        "tss_numerator":
            tss_numerator,

        "tss_denominator":
            tss_denominator,

        "tss":
            tss,

        "sensitivity":
            sensitivity,

        "fpr":
            fpr,

        "specificity":
            specificity,

        "precision":
            precision,

        "f1":
            f1,

        "hss":
            hss,

        "predicted_positive":
            predicted_positive,

        "selected_index":
            selected_index,

        "selected_threshold":
            selected_threshold,

        "max_tss_numerator":
            max_tss_numerator,

        "tss_tie_count":
            int(
                len(
                    tied_indices
                )
            ),

        "sentinel_threshold":
            sentinel_threshold,
    }


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 6.3 — VALIDATION THRESHOLD SWEEP + METRICS"
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
    SELECTION_PROTOCOL_FILE,
    CANDIDATE_REGISTRY_FILE,
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
    THRESHOLD_SWEEP_FILE,
    CANDIDATE_METRICS_FILE,
    METRICS_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Output already exists: {output_path}\n"
            "Refusing to overwrite validation "
            "metric provenance."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

prediction_contract = load_json(
    PREDICTION_CONTRACT_FILE
)

selection_protocol = load_json(
    SELECTION_PROTOCOL_FILE
)


if prediction_contract.get(
    "status"
) != "FROZEN_VALIDATION_PREDICTIONS":

    raise RuntimeError(
        "Validation prediction contract "
        "is not frozen."
    )


if selection_protocol.get(
    "status"
) != "FROZEN_BEFORE_VALIDATION":

    raise RuntimeError(
        "Selection protocol is not frozen."
    )


# ============================================================
# VERIFY PREDICTION ARTIFACT HASH
# ============================================================

expected_prediction_hash = (
    prediction_contract[
        "prediction_artifact"
    ][
        "sha256"
    ]
)


actual_prediction_hash = sha256_file(
    PREDICTION_FILE
)


if (
    actual_prediction_hash
    !=
    expected_prediction_hash
):

    raise RuntimeError(
        "Validation prediction artifact hash "
        "does not match frozen contract."
    )


print(
    "Validation prediction artifact hash: PASS"
)


# ============================================================
# LOAD CANDIDATE REGISTRY
# ============================================================

registry = pd.read_csv(
    CANDIDATE_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    registry
) != 34:

    raise RuntimeError(
        "Expected exactly 34 frozen candidates."
    )


if registry[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate IDs in registry."
    )


family_counts = (
    registry[
        "family"
    ]
    .value_counts()
    .to_dict()
)


if family_counts.get(
    "logistic"
) != 10:

    raise RuntimeError(
        "Expected 10 Logistic candidates."
    )


if family_counts.get(
    "random_forest"
) != 12:

    raise RuntimeError(
        "Expected 12 Random Forest candidates."
    )


if family_counts.get(
    "xgboost"
) != 12:

    raise RuntimeError(
        "Expected 12 XGBoost candidates."
    )


# ============================================================
# LOAD FROZEN VALIDATION PREDICTIONS
# ============================================================

predictions = pd.read_parquet(
    PREDICTION_FILE
)


if predictions.shape != (
    13299,
    36,
):

    raise RuntimeError(
        f"Unexpected validation prediction "
        f"shape: {predictions.shape}"
    )


required_base_columns = {
    "validation_row_id",
    "target",
}


if not required_base_columns.issubset(
    predictions.columns
):

    raise RuntimeError(
        "Validation prediction artifact "
        "missing required base columns."
    )


candidate_ids = registry[
    "candidate_id"
].astype(str).tolist()


if set(
    candidate_ids
) != (
    set(
        predictions.columns
    )
    -
    required_base_columns
):

    raise RuntimeError(
        "Candidate prediction columns differ "
        "from frozen candidate registry."
    )


y_validation = predictions[
    "target"
].to_numpy(
    dtype=np.int64
)


if set(
    np.unique(
        y_validation
    ).tolist()
) != {
    0,
    1,
}:

    raise RuntimeError(
        "Validation target must be binary."
    )


positive_count = int(
    y_validation.sum()
)


negative_count = int(
    (
        y_validation
        ==
        0
    ).sum()
)


if positive_count != 515:

    raise RuntimeError(
        "Unexpected validation positive count."
    )


if negative_count != 12784:

    raise RuntimeError(
        "Unexpected validation negative count."
    )


print(
    "Validation rows:",
    len(
        y_validation
    )
)

print(
    "Validation positives:",
    positive_count
)

print(
    "Validation negatives:",
    negative_count
)

print(
    "Candidates:",
    len(
        candidate_ids
    )
)


# ============================================================
# METRIC IMPLEMENTATION FREEZE
# ============================================================

print(
    "\n===================================="
)

print(
    "METRIC IMPLEMENTATION"
)

print(
    "===================================="
)


print(
    "TSS: TPR - FPR"
)

print(
    "TSS comparison: exact integer numerator"
)

print(
    "PR-AUC implementation: sklearn average_precision_score"
)

print(
    "ROC-AUC implementation: sklearn roc_auc_score"
)

print(
    "Brier implementation: sklearn brier_score_loss"
)

print(
    "Threshold tie-break: highest threshold"
)


# ============================================================
# EVALUATE CANDIDATES
# ============================================================

sweep_frames = []

metrics_rows = []


print(
    "\n===================================="
)

print(
    "CANDIDATE VALIDATION EVALUATION"
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


    probabilities = predictions[
        candidate_id
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "probability values."
        )


    if (
        probabilities.min() < 0.0
        or
        probabilities.max() > 1.0
    ):

        raise RuntimeError(
            f"{candidate_id}: probability "
            "outside [0,1]."
        )


    # ========================================================
    # THRESHOLD-INDEPENDENT METRICS
    # ========================================================

    validation_pr_auc = float(
        average_precision_score(
            y_validation,
            probabilities,
        )
    )


    validation_roc_auc = float(
        roc_auc_score(
            y_validation,
            probabilities,
        )
    )


    validation_brier = float(
        brier_score_loss(
            y_validation,
            probabilities,
        )
    )


    # ========================================================
    # EXACT THRESHOLD SWEEP
    # ========================================================

    sweep = exact_threshold_sweep(
        y_validation,
        probabilities,
    )


    selected_index = sweep[
        "selected_index"
    ]


    selected_threshold = float(
        sweep[
            "selected_threshold"
        ]
    )


    selected_tp = int(
        sweep[
            "tp"
        ][
            selected_index
        ]
    )


    selected_fp = int(
        sweep[
            "fp"
        ][
            selected_index
        ]
    )


    selected_tn = int(
        sweep[
            "tn"
        ][
            selected_index
        ]
    )


    selected_fn = int(
        sweep[
            "fn"
        ][
            selected_index
        ]
    )


    selected_tss = float(
        sweep[
            "tss"
        ][
            selected_index
        ]
    )


    selected_sensitivity = float(
        sweep[
            "sensitivity"
        ][
            selected_index
        ]
    )


    selected_specificity = float(
        sweep[
            "specificity"
        ][
            selected_index
        ]
    )


    selected_precision = float(
        sweep[
            "precision"
        ][
            selected_index
        ]
    )


    selected_f1 = float(
        sweep[
            "f1"
        ][
            selected_index
        ]
    )


    selected_hss = float(
        sweep[
            "hss"
        ][
            selected_index
        ]
    )


    selected_fpr = float(
        sweep[
            "fpr"
        ][
            selected_index
        ]
    )


    # ========================================================
    # INDEPENDENT SELECTED-THRESHOLD RECHECK
    # ========================================================

    selected_predictions = (
        probabilities
        >=
        selected_threshold
    ).astype(
        np.int64
    )


    recheck_tp = int(
        np.sum(
            (
                selected_predictions
                ==
                1
            )
            &
            (
                y_validation
                ==
                1
            )
        )
    )


    recheck_fp = int(
        np.sum(
            (
                selected_predictions
                ==
                1
            )
            &
            (
                y_validation
                ==
                0
            )
        )
    )


    recheck_tn = int(
        np.sum(
            (
                selected_predictions
                ==
                0
            )
            &
            (
                y_validation
                ==
                0
            )
        )
    )


    recheck_fn = int(
        np.sum(
            (
                selected_predictions
                ==
                0
            )
            &
            (
                y_validation
                ==
                1
            )
        )
    )


    if (
        recheck_tp
        !=
        selected_tp
        or
        recheck_fp
        !=
        selected_fp
        or
        recheck_tn
        !=
        selected_tn
        or
        recheck_fn
        !=
        selected_fn
    ):

        raise RuntimeError(
            f"{candidate_id}: selected-threshold "
            "confusion-matrix recheck failed."
        )


    recheck_tss = (
        safe_divide(
            recheck_tp,
            recheck_tp
            +
            recheck_fn,
        )
        -
        safe_divide(
            recheck_fp,
            recheck_fp
            +
            recheck_tn,
        )
    )


    if not np.isclose(
        recheck_tss,
        selected_tss,
        rtol=0.0,
        atol=1e-15,
    ):

        raise RuntimeError(
            f"{candidate_id}: selected TSS "
            "recheck failed."
        )


    # ========================================================
    # STORE COMPLETE SWEEP
    # ========================================================

    sweep_count = len(
        sweep[
            "thresholds"
        ]
    )


    sweep_frame = pd.DataFrame(
        {
            "family":
                family,

            "candidate_id":
                candidate_id,

            "threshold_rank_descending":
                np.arange(
                    sweep_count,
                    dtype=np.int64,
                ),

            "threshold":
                sweep[
                    "thresholds"
                ],

            "is_all_negative_sentinel":
                np.arange(
                    sweep_count
                )
                ==
                0,

            "tp":
                sweep[
                    "tp"
                ],

            "fp":
                sweep[
                    "fp"
                ],

            "tn":
                sweep[
                    "tn"
                ],

            "fn":
                sweep[
                    "fn"
                ],

            "predicted_positive":
                sweep[
                    "predicted_positive"
                ],

            "sensitivity":
                sweep[
                    "sensitivity"
                ],

            "fpr":
                sweep[
                    "fpr"
                ],

            "specificity":
                sweep[
                    "specificity"
                ],

            "precision":
                sweep[
                    "precision"
                ],

            "f1":
                sweep[
                    "f1"
                ],

            "hss":
                sweep[
                    "hss"
                ],

            "tss_numerator":
                sweep[
                    "tss_numerator"
                ],

            "tss_denominator":
                int(
                    sweep[
                        "tss_denominator"
                    ]
                ),

            "tss":
                sweep[
                    "tss"
                ],

            "selected_by_max_tss":
                np.arange(
                    sweep_count
                )
                ==
                selected_index,
        }
    )


    sweep_frames.append(
        sweep_frame
    )


    # ========================================================
    # STORE CANDIDATE SUMMARY
    # ========================================================

    metrics_rows.append(
        {
            "family":
                family,

            "model_family":
                str(
                    registry_row[
                        "model_family"
                    ]
                ),

            "candidate_id":
                candidate_id,

            "validation_rows":
                len(
                    y_validation
                ),

            "validation_positive":
                positive_count,

            "validation_negative":
                negative_count,

            "unique_probability_count":
                int(
                    np.unique(
                        probabilities
                    ).size
                ),

            "threshold_count_including_sentinel":
                sweep_count,

            "selected_threshold":
                selected_threshold,

            "selected_threshold_is_sentinel":
                bool(
                    selected_index
                    ==
                    0
                ),

            "validation_tss":
                selected_tss,

            "validation_pr_auc":
                validation_pr_auc,

            "validation_roc_auc":
                validation_roc_auc,

            "validation_brier_score":
                validation_brier,

            "validation_hss":
                selected_hss,

            "validation_f1":
                selected_f1,

            "validation_sensitivity":
                selected_sensitivity,

            "validation_specificity":
                selected_specificity,

            "validation_fpr":
                selected_fpr,

            "validation_precision":
                selected_precision,

            "tp":
                selected_tp,

            "fp":
                selected_fp,

            "tn":
                selected_tn,

            "fn":
                selected_fn,

            "predicted_positive":
                selected_tp
                +
                selected_fp,

            "tss_exact_numerator":
                int(
                    sweep[
                        "max_tss_numerator"
                    ]
                ),

            "tss_exact_denominator":
                int(
                    sweep[
                        "tss_denominator"
                    ]
                ),

            "maximum_tss_threshold_tie_count":
                int(
                    sweep[
                        "tss_tie_count"
                    ]
                ),

            "threshold_tie_break":
                "HIGHEST_THRESHOLD",

            "threshold_selected":
                True,

            "family_winner_selected":
                False,

            "holdout_used":
                False,

            "cycle25_used":
                False,
        }
    )


    print(
        f"{candidate_id}: "
        f"threshold={selected_threshold:.10g}, "
        f"TSS={selected_tss:.6f}, "
        f"PR_AUC={validation_pr_auc:.6f}, "
        f"ROC_AUC={validation_roc_auc:.6f}, "
        f"Brier={validation_brier:.6f}, "
        f"ties={sweep['tss_tie_count']}, "
        f"PASS=True"
    )


# ============================================================
# COMBINE OUTPUTS
# ============================================================

threshold_sweep_df = pd.concat(
    sweep_frames,
    ignore_index=True,
)


candidate_metrics_df = pd.DataFrame(
    metrics_rows
)


if len(
    candidate_metrics_df
) != 34:

    raise RuntimeError(
        "Candidate metrics table does not "
        "contain 34 rows."
    )


if candidate_metrics_df[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Candidate metrics table contains "
        "duplicate candidate IDs."
    )


if int(
    threshold_sweep_df[
        "selected_by_max_tss"
    ].sum()
) != 34:

    raise RuntimeError(
        "Expected exactly one selected "
        "threshold per candidate."
    )


# ============================================================
# NUMERIC SANITY
# ============================================================

bounded_metric_columns = [
    "validation_pr_auc",
    "validation_roc_auc",
    "validation_brier_score",
    "validation_f1",
    "validation_sensitivity",
    "validation_specificity",
    "validation_fpr",
    "validation_precision",
]


for column in bounded_metric_columns:

    values = candidate_metrics_df[
        column
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"{column} contains non-finite values."
        )


    if (
        values.min() < 0.0
        or
        values.max() > 1.0
    ):

        raise RuntimeError(
            f"{column} contains value outside [0,1]."
        )


for column in [
    "validation_tss",
    "validation_hss",
]:

    values = candidate_metrics_df[
        column
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"{column} contains non-finite values."
        )


    if (
        values.min() < -1.0
        or
        values.max() > 1.0
    ):

        raise RuntimeError(
            f"{column} contains value outside [-1,1]."
        )


# ============================================================
# SAVE
# ============================================================

THRESHOLD_SWEEP_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


CANDIDATE_METRICS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


METRICS_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


threshold_sweep_df.to_parquet(
    THRESHOLD_SWEEP_FILE,
    index=False,
)


candidate_metrics_df.to_csv(
    CANDIDATE_METRICS_FILE,
    index=False,
)


threshold_sweep_hash = sha256_file(
    THRESHOLD_SWEEP_FILE
)


candidate_metrics_hash = sha256_file(
    CANDIDATE_METRICS_FILE
)


# ============================================================
# ROUND-TRIP VERIFICATION
# ============================================================

sweep_roundtrip = pd.read_parquet(
    THRESHOLD_SWEEP_FILE
)


metrics_roundtrip = pd.read_csv(
    CANDIDATE_METRICS_FILE,
    keep_default_na=False,
)


if len(
    sweep_roundtrip
) != len(
    threshold_sweep_df
):

    raise RuntimeError(
        "Threshold-sweep parquet "
        "round-trip row mismatch."
    )


if len(
    metrics_roundtrip
) != 34:

    raise RuntimeError(
        "Candidate-metrics CSV "
        "round-trip row mismatch."
    )


print(
    "\nThreshold sweep round-trip: PASS"
)

print(
    "Candidate metrics round-trip: PASS"
)


# ============================================================
# CONTRACT
# ============================================================

metrics_contract = {

    "contract_name":
        "Cycle24 Validation Candidate Metrics",

    "version":
        "1.0",

    "status":
        "FROZEN_CANDIDATE_VALIDATION_METRICS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "validation_population":
        {
            "rows":
                len(
                    y_validation
                ),

            "positive":
                positive_count,

            "negative":
                negative_count,
        },

    "candidate_count":
        34,

    "family_counts":
        {
            "logistic":
                10,

            "random_forest":
                12,

            "xgboost":
                12,
        },

    "threshold_protocol":
        {
            "prediction_rule":
                "probability >= threshold",

            "threshold_candidates":
                (
                    "Every unique validation probability "
                    "plus one all-negative sentinel above "
                    "the maximum probability."
                ),

            "all_negative_sentinel":
                "numpy.nextafter(max_probability, +infinity)",

            "primary_metric":
                "TSS",

            "tss_definition":
                "TPR - FPR",

            "exact_tss_comparison":
                (
                    "Compare integer quantity "
                    "TP * N_negative - FP * N_positive."
                ),

            "tie_break":
                "highest threshold",

            "default_threshold_0_5_forced":
                False,
        },

    "metric_implementations":
        {
            "PR_AUC":
                (
                    "sklearn.metrics.average_precision_score"
                ),

            "PR_AUC_note":
                (
                    "Non-interpolated average precision "
                    "implementation used as the operational "
                    "PR-AUC quantity for the frozen "
                    "selection hierarchy."
                ),

            "ROC_AUC":
                "sklearn.metrics.roc_auc_score",

            "Brier_score":
                "sklearn.metrics.brier_score_loss",

            "HSS":
                (
                    "2*(TP*TN - FN*FP) / "
                    "((TP+FN)*(FN+TN) + "
                    "(TP+FP)*(FP+TN))"
                ),

            "F1":
                "2*TP / (2*TP + FP + FN)",
        },

    "artifacts":
        {
            "threshold_sweep":
                {
                    "file":
                        str(
                            THRESHOLD_SWEEP_FILE
                        ),

                    "sha256":
                        threshold_sweep_hash,

                    "rows":
                        int(
                            len(
                                threshold_sweep_df
                            )
                        ),
                },

            "candidate_metrics":
                {
                    "file":
                        str(
                            CANDIDATE_METRICS_FILE
                        ),

                    "sha256":
                        candidate_metrics_hash,

                    "rows":
                        34,
                },
        },

    "selection_state":
        {
            "candidate_thresholds_selected":
                True,

            "candidate_validation_metrics_computed":
                True,

            "family_winners_selected":
                False,

            "cross_family_winner_selected":
                False,

            "model_retraining_performed":
                False,
        },

    "data_isolation":
        {
            "validation_predictions_used":
                True,

            "raw_validation_dataset_reread":
                False,

            "holdout_read":
                False,

            "cycle25_read":
                False,
        },

    "immutability_rule":
        (
            "Candidate-specific validation thresholds "
            "and validation metrics are now frozen. "
            "They must not be changed after viewing "
            "candidate performance. Chapter 6.4 may "
            "only apply the already-frozen family "
            "selection hierarchy to these results."
        ),

    "provenance":
        {
            "validation_prediction_artifact":
                {
                    "file":
                        str(
                            PREDICTION_FILE
                        ),

                    "sha256":
                        actual_prediction_hash,
                },

            "validation_prediction_contract":
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

            "selection_protocol":
                {
                    "file":
                        str(
                            SELECTION_PROTOCOL_FILE
                        ),

                    "sha256":
                        sha256_file(
                            SELECTION_PROTOCOL_FILE
                        ),
                },

            "candidate_registry":
                {
                    "file":
                        str(
                            CANDIDATE_REGISTRY_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CANDIDATE_REGISTRY_FILE
                        ),
                },
        },
}


with METRICS_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics_contract,
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
    "VALIDATION METRIC SUMMARY"
)

print(
    "===================================="
)


print(
    "Candidates evaluated:",
    len(
        candidate_metrics_df
    )
)

print(
    "Candidate thresholds selected:",
    int(
        candidate_metrics_df[
            "threshold_selected"
        ].sum()
    )
)

print(
    "Threshold sweep rows:",
    len(
        threshold_sweep_df
    )
)

print(
    "Minimum validation TSS:",
    float(
        candidate_metrics_df[
            "validation_tss"
        ].min()
    )
)

print(
    "Maximum validation TSS:",
    float(
        candidate_metrics_df[
            "validation_tss"
        ].max()
    )
)

print(
    "Minimum PR-AUC:",
    float(
        candidate_metrics_df[
            "validation_pr_auc"
        ].min()
    )
)

print(
    "Maximum PR-AUC:",
    float(
        candidate_metrics_df[
            "validation_pr_auc"
        ].max()
    )
)


# ============================================================
# FAMILY-WINNER STATE
# ============================================================

print(
    "\n===================================="
)

print(
    "SELECTION STATE"
)

print(
    "===================================="
)


print(
    "Candidate-specific thresholds selected:",
    True
)

print(
    "Validation metrics computed:",
    True
)

print(
    "Family winners selected:",
    False
)

print(
    "Cross-family winner selected:",
    False
)

print(
    "Models retrained:",
    False
)


# ============================================================
# DATA ISOLATION
# ============================================================

print(
    "\n===================================="
)

print(
    "DATA ISOLATION"
)

print(
    "===================================="
)


print(
    "Cycle24 validation predictions used:",
    True
)

print(
    "Raw validation dataset reread:",
    False
)

print(
    "Cycle24 HOLDOUT read:",
    False
)

print(
    "Cycle25 read:",
    False
)


# ============================================================
# OUTPUTS
# ============================================================

print(
    "\nSaved:"
)

print(
    THRESHOLD_SWEEP_FILE
)

print(
    CANDIDATE_METRICS_FILE
)

print(
    METRICS_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 6.3 COMPLETE — VALIDATION METRICS FROZEN"
)

print(
    "===================================="
)