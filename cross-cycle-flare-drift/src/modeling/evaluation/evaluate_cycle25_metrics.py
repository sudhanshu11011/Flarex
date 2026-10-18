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

MODEL_REGISTRY_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_frozen_model_registry.csv"
)

OUTPUT_METRICS_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "cycle25_metrics.csv"
)

OUTPUT_CALIBRATION_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "cycle25_calibration.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_metrics_contract.json"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_ROWS = 79361
EXPECTED_POSITIVES = 4836
EXPECTED_NEGATIVES = 74525

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

CALIBRATION_EDGES = np.array(
    [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    dtype=np.float64,
)


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
        (2 * tp)
        /
        denominator
    )


def evaluate_scope(
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
        "tp":
            tp,

        "fp":
            fp,

        "tn":
            tn,

        "fn":
            fn,

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

        "fpr":
            fpr,

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
    "MODEL CHAPTER 8.3 — CYCLE25 BLOCK + POOLED METRICS"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

for path in [
    PREDICTION_FILE,
    PREDICTION_CONTRACT_FILE,
    CYCLE25_PROTOCOL_FILE,
    MODEL_REGISTRY_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# PROTECT OUTPUTS
# ============================================================

for path in [
    OUTPUT_METRICS_FILE,
    OUTPUT_CALIBRATION_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Cycle25 point evaluation."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

prediction_contract = load_json(
    PREDICTION_CONTRACT_FILE
)

protocol = load_json(
    CYCLE25_PROTOCOL_FILE
)


if prediction_contract.get(
    "status"
) != "FROZEN_CYCLE25_PREDICTIONS":

    raise RuntimeError(
        "Cycle25 predictions are not frozen."
    )


if protocol.get(
    "status"
) != "FROZEN_BEFORE_CYCLE25_EVALUATION":

    raise RuntimeError(
        "Cycle25 evaluation protocol "
        "is not frozen."
    )


print(
    "Cycle25 prediction contract:",
    "FROZEN"
)

print(
    "Cycle25 evaluation protocol:",
    "FROZEN"
)


# ============================================================
# VERIFY PREDICTION HASH
# ============================================================

actual_prediction_hash = sha256_file(
    PREDICTION_FILE
)


expected_prediction_hash = (
    prediction_contract[
        "artifacts"
    ][
        "prediction_artifact"
    ][
        "sha256"
    ]
)


if (
    actual_prediction_hash
    !=
    expected_prediction_hash
):

    raise RuntimeError(
        "Cycle25 frozen prediction "
        "artifact hash mismatch."
    )


print(
    "Cycle25 prediction artifact hash:",
    "PASS"
)


# ============================================================
# VERIFY REGISTRY HASH
# ============================================================

actual_registry_hash = sha256_file(
    MODEL_REGISTRY_FILE
)


expected_registry_hash = (
    protocol[
        "artifacts"
    ][
        "cycle25_frozen_model_registry"
    ][
        "sha256"
    ]
)


if (
    actual_registry_hash
    !=
    expected_registry_hash
):

    raise RuntimeError(
        "Cycle25 model registry "
        "hash mismatch."
    )


# ============================================================
# LOAD FROZEN ARTIFACTS
# ============================================================

predictions = pd.read_parquet(
    PREDICTION_FILE
)


registry = pd.read_csv(
    MODEL_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    predictions
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Unexpected Cycle25 prediction "
        "row count."
    )


if len(
    registry
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen models."
    )


# ============================================================
# BASE SCHEMA
# ============================================================

required_base_columns = {
    "cycle25_row_id",
    "primary_block_id",
    "ENTITY_GROUP",
    "ENTITY_GROUP_ID",
    "HARPNUM",
    "target",
}


missing = (
    required_base_columns
    -
    set(
        predictions.columns
    )
)


if missing:

    raise RuntimeError(
        "Cycle25 prediction artifact missing "
        f"columns: {sorted(missing)}"
    )


# ============================================================
# POPULATION INTEGRITY
# ============================================================

target_all = predictions[
    "target"
].to_numpy(
    dtype=np.int64
)


if int(
    target_all.sum()
) != EXPECTED_POSITIVES:

    raise RuntimeError(
        "Cycle25 positive count mismatch."
    )


if int(
    (target_all == 0).sum()
) != EXPECTED_NEGATIVES:

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
        "Cycle25 primary temporal "
        "block set mismatch."
    )


print(
    "Cycle25 rows:",
    len(
        predictions
    )
)

print(
    "Cycle25 positives:",
    EXPECTED_POSITIVES
)

print(
    "Cycle25 negatives:",
    EXPECTED_NEGATIVES
)

print(
    "Primary temporal block set:",
    "PASS"
)


# ============================================================
# BUILD EVALUATION SCOPES
# ============================================================

scope_masks = {}


for block_id in PRIMARY_BLOCKS:

    mask = (
        predictions[
            "primary_block_id"
        ].astype(str)
        ==
        block_id
    ).to_numpy()


    if int(
        mask.sum()
    ) == 0:

        raise RuntimeError(
            f"{block_id}: no rows."
        )


    scope_masks[
        block_id
    ] = mask


scope_masks[
    POOLED_SCOPE
] = np.ones(
    len(
        predictions
    ),
    dtype=bool,
)


# ============================================================
# VERIFY PARTITION
# ============================================================

primary_membership_count = np.zeros(
    len(
        predictions
    ),
    dtype=np.int64,
)


for block_id in PRIMARY_BLOCKS:

    primary_membership_count += (
        scope_masks[
            block_id
        ].astype(
            np.int64
        )
    )


if not np.all(
    primary_membership_count
    ==
    1
):

    raise RuntimeError(
        "Cycle25 primary blocks do not "
        "partition rows exactly once."
    )


print(
    "Primary block partition:",
    "PASS"
)


# ============================================================
# METRIC / CALIBRATION STORAGE
# ============================================================

metric_rows = []
calibration_rows = []


print(
    "\n===================================="
)

print(
    "CYCLE25 POINT RESULTS"
)

print(
    "===================================="
)


# ============================================================
# MODEL LOOP
# ============================================================

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
            f"{candidate_id}: missing "
            "probability column."
        )


    if prediction_column not in predictions.columns:

        raise RuntimeError(
            f"{candidate_id}: missing "
            "prediction column."
        )


    probabilities_all = predictions[
        probability_column
    ].to_numpy(
        dtype=np.float64
    )


    classes_all = predictions[
        prediction_column
    ].to_numpy(
        dtype=np.int64
    )


    if not np.isfinite(
        probabilities_all
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "probabilities."
        )


    # --------------------------------------------------------
    # FROZEN THRESHOLD VERIFICATION
    # --------------------------------------------------------

    recomputed_classes = (
        probabilities_all
        >=
        threshold
    ).astype(
        np.int64
    )


    if not np.array_equal(
        recomputed_classes,
        classes_all,
    ):

        raise RuntimeError(
            f"{candidate_id}: stored "
            "predictions do not match "
            "frozen threshold."
        )


    # ========================================================
    # SCOPE LOOP
    # ========================================================

    for scope_id in ALL_SCOPES:

        mask = scope_masks[
            scope_id
        ]


        y_true = target_all[
            mask
        ]


        probabilities = probabilities_all[
            mask
        ]


        classes = classes_all[
            mask
        ]


        rows = int(
            mask.sum()
        )


        positives = int(
            y_true.sum()
        )


        negatives = int(
            rows
            -
            positives
        )


        if positives <= 0:

            raise RuntimeError(
                f"{scope_id}: no positive "
                "samples."
            )


        if negatives <= 0:

            raise RuntimeError(
                f"{scope_id}: no negative "
                "samples."
            )


        entity_groups = int(
            predictions.loc[
                mask,
                "ENTITY_GROUP"
            ].nunique(
                dropna=False
            )
        )


        values = evaluate_scope(
            y_true,
            probabilities,
            classes,
        )


        metric_rows.append(
            {
                "scope_id":
                    scope_id,

                "scope_type":
                    (
                        "PRIMARY_TEMPORAL_BLOCK"
                        if scope_id
                        in PRIMARY_BLOCKS
                        else
                        "SECONDARY_POOLED"
                    ),

                "family":
                    family,

                "candidate_id":
                    candidate_id,

                "frozen_threshold":
                    threshold,

                "rows":
                    rows,

                "positive":
                    positives,

                "negative":
                    negatives,

                "positive_prevalence":
                    float(
                        positives / rows
                    ),

                "entity_groups":
                    entity_groups,

                "predicted_positive":
                    int(
                        classes.sum()
                    ),

                **values,

                "threshold_source":
                    "CYCLE24_VALIDATION",

                "threshold_reoptimized":
                    False,

                "model_retrained":
                    False,

                "preprocessing_refit":
                    False,
            }
        )


        # ====================================================
        # CALIBRATION
        # ====================================================

        bin_indices = np.floor(
            probabilities
            *
            10.0
        ).astype(
            np.int64
        )


        bin_indices = np.clip(
            bin_indices,
            0,
            9,
        )


        for bin_index in range(10):

            bin_mask = (
                bin_indices
                ==
                bin_index
            )


            row_count = int(
                bin_mask.sum()
            )


            lower = float(
                CALIBRATION_EDGES[
                    bin_index
                ]
            )


            upper = float(
                CALIBRATION_EDGES[
                    bin_index + 1
                ]
            )


            if row_count > 0:

                positive_in_bin = int(
                    y_true[
                        bin_mask
                    ].sum()
                )


                mean_probability = float(
                    probabilities[
                        bin_mask
                    ].mean()
                )


                observed_rate = float(
                    y_true[
                        bin_mask
                    ].mean()
                )


            else:

                positive_in_bin = 0
                mean_probability = np.nan
                observed_rate = np.nan


            calibration_rows.append(
                {
                    "scope_id":
                        scope_id,

                    "scope_type":
                        (
                            "PRIMARY_TEMPORAL_BLOCK"
                            if scope_id
                            in PRIMARY_BLOCKS
                            else
                            "SECONDARY_POOLED"
                        ),

                    "family":
                        family,

                    "candidate_id":
                        candidate_id,

                    "bin_index":
                        bin_index,

                    "bin_lower":
                        lower,

                    "bin_upper":
                        upper,

                    "bin_interval":
                        (
                            f"[{lower:.1f}, "
                            f"{upper:.1f}"
                            +
                            (
                                "]"
                                if bin_index == 9
                                else ")"
                            )
                        ),

                    "row_count":
                        row_count,

                    "positive_count":
                        positive_in_bin,

                    "mean_predicted_probability":
                        mean_probability,

                    "observed_event_rate":
                        observed_rate,
                }
            )


        print(
            f"{scope_id} | "
            f"{candidate_id}: "
            f"rows={rows}, "
            f"groups={entity_groups}, "
            f"prevalence={positives / rows:.6f}, "
            f"TSS={values['tss']:.6f}, "
            f"HSS={values['hss']:.6f}, "
            f"F1={values['f1']:.6f}, "
            f"PR_AUC={values['pr_auc']:.6f}, "
            f"ROC_AUC={values['roc_auc']:.6f}, "
            f"Brier={values['brier_score']:.6f}"
        )


# ============================================================
# BUILD TABLES
# ============================================================

metrics_df = pd.DataFrame(
    metric_rows
)


calibration_df = pd.DataFrame(
    calibration_rows
)


if len(
    metrics_df
) != 12:

    raise RuntimeError(
        "Expected 12 Cycle25 metric rows "
        "(4 scopes x 3 models)."
    )


if len(
    calibration_df
) != 120:

    raise RuntimeError(
        "Expected 120 calibration rows "
        "(4 scopes x 3 models x 10 bins)."
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

if metrics_df.duplicated(
    subset=[
        "scope_id",
        "candidate_id",
    ]
).any():

    raise RuntimeError(
        "Duplicate Cycle25 metric scope/model."
    )


# ============================================================
# METRIC SANITY
# ============================================================

for column in [
    "f1",
    "sensitivity",
    "specificity",
    "fpr",
    "precision",
    "pr_auc",
    "roc_auc",
    "brier_score",
]:

    values = metrics_df[
        column
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"{column}: non-finite value."
        )


    if (
        values.min() < 0.0
        or
        values.max() > 1.0
    ):

        raise RuntimeError(
            f"{column}: value outside [0,1]."
        )


for column in [
    "tss",
    "hss",
]:

    values = metrics_df[
        column
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"{column}: non-finite value."
        )


    if (
        values.min() < -1.0
        or
        values.max() > 1.0
    ):

        raise RuntimeError(
            f"{column}: value outside [-1,1]."
        )


# ============================================================
# CALIBRATION ACCOUNTING
# ============================================================

for _, metric_row in metrics_df.iterrows():

    scope_id = str(
        metric_row[
            "scope_id"
        ]
    )

    candidate_id = str(
        metric_row[
            "candidate_id"
        ]
    )


    subset = calibration_df[
        (
            calibration_df[
                "scope_id"
            ].astype(str)
            ==
            scope_id
        )
        &
        (
            calibration_df[
                "candidate_id"
            ].astype(str)
            ==
            candidate_id
        )
    ]


    if int(
        subset[
            "row_count"
        ].sum()
    ) != int(
        metric_row[
            "rows"
        ]
    ):

        raise RuntimeError(
            f"{scope_id}/{candidate_id}: "
            "calibration row accounting mismatch."
        )


    if int(
        subset[
            "positive_count"
        ].sum()
    ) != int(
        metric_row[
            "positive"
        ]
    ):

        raise RuntimeError(
            f"{scope_id}/{candidate_id}: "
            "calibration positive accounting mismatch."
        )


print(
    "\nCalibration accounting:",
    "PASS"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_METRICS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CALIBRATION_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metrics_df.to_csv(
    OUTPUT_METRICS_FILE,
    index=False,
)


calibration_df.to_csv(
    OUTPUT_CALIBRATION_FILE,
    index=False,
)


metrics_hash = sha256_file(
    OUTPUT_METRICS_FILE
)


calibration_hash = sha256_file(
    OUTPUT_CALIBRATION_FILE
)


# ============================================================
# ROUND TRIP
# ============================================================

metrics_roundtrip = pd.read_csv(
    OUTPUT_METRICS_FILE,
    keep_default_na=False,
)


calibration_roundtrip = pd.read_csv(
    OUTPUT_CALIBRATION_FILE,
    keep_default_na=False,
)


if len(
    metrics_roundtrip
) != 12:

    raise RuntimeError(
        "Cycle25 metrics CSV "
        "round-trip mismatch."
    )


if len(
    calibration_roundtrip
) != 120:

    raise RuntimeError(
        "Cycle25 calibration CSV "
        "round-trip mismatch."
    )


print(
    "Cycle25 metrics round-trip:",
    "PASS"
)

print(
    "Cycle25 calibration round-trip:",
    "PASS"
)


# ============================================================
# CONTRACT
# ============================================================

scope_summary = []


for scope_id in ALL_SCOPES:

    scope_rows = metrics_df[
        metrics_df[
            "scope_id"
        ]
        ==
        scope_id
    ]


    first = scope_rows.iloc[
        0
    ]


    scope_summary.append(
        {
            "scope_id":
                scope_id,

            "scope_type":
                str(
                    first[
                        "scope_type"
                    ]
                ),

            "rows":
                int(
                    first[
                        "rows"
                    ]
                ),

            "positive":
                int(
                    first[
                        "positive"
                    ]
                ),

            "negative":
                int(
                    first[
                        "negative"
                    ]
                ),

            "positive_prevalence":
                float(
                    first[
                        "positive_prevalence"
                    ]
                ),

            "entity_groups":
                int(
                    first[
                        "entity_groups"
                    ]
                ),
        }
    )


contract = {

    "contract_name":
        "Frozen Solar Cycle 25 Point Metrics and Calibration",

    "version":
        "1.0",

    "status":
        "FROZEN_CYCLE25_POINT_METRICS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_8_3",

    "evaluation_scopes":
        scope_summary,

    "primary_temporal_blocks":
        PRIMARY_BLOCKS,

    "secondary_pooled_scope":
        POOLED_SCOPE,

    "model_count":
        3,

    "metric_rows":
        12,

    "metric_implementations":
        {
            "TSS":
                "TPR - FPR",

            "HSS":
                (
                    "2*(TP*TN - FN*FP) / "
                    "((TP+FN)*(FN+TN) + "
                    "(TP+FP)*(FP+TN))"
                ),

            "F1":
                (
                    "2*TP / "
                    "(2*TP + FP + FN)"
                ),

            "PR_AUC":
                (
                    "sklearn.metrics."
                    "average_precision_score"
                ),

            "PR_AUC_note":
                (
                    "Operationally Average Precision; "
                    "name retained for compatibility "
                    "with frozen Cycle24 contracts."
                ),

            "ROC_AUC":
                (
                    "sklearn.metrics."
                    "roc_auc_score"
                ),

            "Brier_score":
                (
                    "sklearn.metrics."
                    "brier_score_loss"
                ),
        },

    "threshold_policy":
        {
            "source":
                "CYCLE24_VALIDATION",

            "prediction_rule":
                (
                    "probability >= "
                    "frozen_threshold"
                ),

            "reoptimized_on_cycle25":
                False,
        },

    "calibration":
        {
            "method":
                "fixed_probability_width",

            "bin_count":
                10,

            "bin_edges":
                CALIBRATION_EDGES.tolist(),

            "recalibration_performed":
                False,

            "diagnostic_only":
                True,
        },

    "artifacts":
        {
            "cycle25_metrics":
                {
                    "file":
                        str(
                            OUTPUT_METRICS_FILE
                        ),

                    "sha256":
                        metrics_hash,

                    "rows":
                        12,
                },

            "cycle25_calibration":
                {
                    "file":
                        str(
                            OUTPUT_CALIBRATION_FILE
                        ),

                    "sha256":
                        calibration_hash,

                    "rows":
                        120,
                },
        },

    "evaluation_state":
        {
            "point_metrics_computed":
                True,

            "calibration_computed":
                True,

            "bootstrap_confidence_intervals_computed":
                False,

            "cross_cycle_metric_deltas_computed":
                False,

            "predictive_drift_classification_computed":
                False,

            "models_retrained":
                False,

            "preprocessing_refit":
                False,

            "thresholds_reoptimized":
                False,

            "candidate_replacement":
                False,
        },

    "data_access":
        {
            "frozen_cycle25_prediction_artifact_read":
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
                        actual_prediction_hash,
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

            "cycle25_model_registry":
                {
                    "file":
                        str(
                            MODEL_REGISTRY_FILE
                        ),

                    "sha256":
                        actual_registry_hash,
                },
        },

    "interpretation_rule":
        (
            "These are fixed-pipeline Cycle25 point "
            "estimates and calibration diagnostics. "
            "Do not infer statistically measurable "
            "cross-cycle change from point estimates "
            "alone. ENTITY_GROUP bootstrap uncertainty "
            "and the predeclared Cycle24-vs-Cycle25 "
            "difference analysis must be completed "
            "before predictive-drift conclusions."
        ),

    "immutability_rule":
        (
            "Cycle25 point results must not trigger "
            "model retraining, candidate replacement, "
            "feature changes, preprocessing refitting, "
            "threshold optimization, temporal-block "
            "changes, or probability recalibration."
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
# FINAL SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "CYCLE25 METRIC SUMMARY"
)

print(
    "===================================="
)


for _, row in metrics_df.iterrows():

    print(
        f"{row['scope_id']} | "
        f"{row['candidate_id']} | "
        f"TSS={row['tss']:.6f} | "
        f"HSS={row['hss']:.6f} | "
        f"F1={row['f1']:.6f} | "
        f"PR_AUC={row['pr_auc']:.6f} | "
        f"ROC_AUC={row['roc_auc']:.6f} | "
        f"Brier={row['brier_score']:.6f}"
    )


print(
    "\n===================================="
)

print(
    "EVALUATION STATE"
)

print(
    "===================================="
)


print(
    "Cycle25 point metrics:",
    True
)

print(
    "Cycle25 calibration:",
    True
)

print(
    "Bootstrap confidence intervals:",
    False
)

print(
    "Cross-cycle drift analysis:",
    False
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
    "\nSaved:"
)

print(
    OUTPUT_METRICS_FILE
)

print(
    OUTPUT_CALIBRATION_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 8.3 COMPLETE — CYCLE25 POINT METRICS FROZEN"
)

print(
    "===================================="
)