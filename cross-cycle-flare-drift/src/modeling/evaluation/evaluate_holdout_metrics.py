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

MODEL_REGISTRY_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "holdout_frozen_model_registry.csv"
)

OUTPUT_METRICS_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "holdout_metrics.csv"
)

OUTPUT_CALIBRATION_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "holdout_calibration.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_metrics_contract.json"
)


# ============================================================
# FROZEN EXPECTATIONS
# ============================================================

EXPECTED_ROWS = 13072
EXPECTED_POSITIVES = 527
EXPECTED_NEGATIVES = 12545
EXPECTED_ENTITY_GROUPS = 176

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


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 7.3 — HOLDOUT METRICS + CALIBRATION"
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
    OUTPUT_METRICS_FILE,
    OUTPUT_CALIBRATION_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Output already exists: {output_path}\n"
            "Refusing to overwrite frozen "
            "holdout evaluation results."
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


print(
    "Holdout prediction contract:",
    "FROZEN"
)

print(
    "Holdout evaluation protocol:",
    "FROZEN"
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
        "Holdout prediction artifact "
        "hash mismatch."
    )


print(
    "Holdout prediction artifact hash:",
    "PASS"
)


# ============================================================
# LOAD FROZEN MODEL REGISTRY
# ============================================================

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


if set(
    registry[
        "family"
    ].astype(str)
) != {
    "logistic",
    "random_forest",
    "xgboost",
}:

    raise RuntimeError(
        "Frozen model-family set is incorrect."
    )


if registry[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate in frozen registry."
    )


print(
    "Frozen model registry:",
    "PASS"
)


# ============================================================
# LOAD HOLDOUT PREDICTION ARTIFACT
# ============================================================

predictions = pd.read_parquet(
    PREDICTION_FILE
)


if len(
    predictions
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Unexpected holdout prediction row count."
    )


required_base_columns = {
    "holdout_row_id",
    "ENTITY_GROUP",
    "ENTITY_GROUP_ID",
    "HARPNUM",
    "target",
}


missing_base_columns = (
    required_base_columns
    -
    set(
        predictions.columns
    )
)


if missing_base_columns:

    raise RuntimeError(
        "Holdout prediction artifact missing "
        f"required columns: "
        f"{sorted(missing_base_columns)}"
    )


# ============================================================
# TARGET / ENTITY CHECK
# ============================================================

y_true = predictions[
    "target"
].to_numpy(
    dtype=np.int64
)


if set(
    np.unique(
        y_true
    ).tolist()
) != {
    0,
    1,
}:

    raise RuntimeError(
        "Holdout target must contain "
        "both classes 0 and 1."
    )


positive_count = int(
    y_true.sum()
)


negative_count = int(
    len(
        y_true
    )
    -
    positive_count
)


if positive_count != EXPECTED_POSITIVES:

    raise RuntimeError(
        "Unexpected holdout positive count."
    )


if negative_count != EXPECTED_NEGATIVES:

    raise RuntimeError(
        "Unexpected holdout negative count."
    )


entity_groups = int(
    predictions[
        "ENTITY_GROUP"
    ].nunique(
        dropna=False
    )
)


if entity_groups != EXPECTED_ENTITY_GROUPS:

    raise RuntimeError(
        "Unexpected holdout ENTITY_GROUP count."
    )


if predictions[
    "ENTITY_GROUP"
].isna().any():

    raise RuntimeError(
        "Missing ENTITY_GROUP detected."
    )


print(
    "Holdout rows:",
    len(
        predictions
    )
)

print(
    "Holdout positives:",
    positive_count
)

print(
    "Holdout negatives:",
    negative_count
)

print(
    "ENTITY_GROUPs:",
    entity_groups
)


# ============================================================
# METRIC IMPLEMENTATION
# ============================================================

print(
    "\n===================================="
)

print(
    "FROZEN METRIC IMPLEMENTATION"
)

print(
    "===================================="
)


print(
    "Thresholds:",
    "CYCLE24 VALIDATION-DERIVED / UNCHANGED"
)

print(
    "TSS:",
    "TPR - FPR"
)

print(
    "PR-AUC:",
    "sklearn.average_precision_score"
)

print(
    "ROC-AUC:",
    "sklearn.roc_auc_score"
)

print(
    "Brier:",
    "sklearn.brier_score_loss"
)

print(
    "Calibration:",
    "10 fixed-width probability bins"
)


# ============================================================
# EVALUATE EACH FROZEN MODEL
# ============================================================

metric_rows = []
calibration_rows = []


print(
    "\n===================================="
)

print(
    "HOLDOUT RESULTS"
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
        f"{candidate_id}__probability"
    )

    prediction_column = (
        f"{candidate_id}__prediction"
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


    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "probabilities detected."
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
    # VERIFY FROZEN THRESHOLD APPLICATION
    # ========================================================

    recomputed_predictions = (
        probabilities
        >=
        threshold
    ).astype(
        np.int64
    )


    if not np.array_equal(
        recomputed_predictions,
        stored_predictions,
    ):

        raise RuntimeError(
            f"{candidate_id}: stored prediction "
            "does not match frozen threshold."
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    tp = int(
        np.sum(
            (stored_predictions == 1)
            &
            (y_true == 1)
        )
    )

    fp = int(
        np.sum(
            (stored_predictions == 1)
            &
            (y_true == 0)
        )
    )

    tn = int(
        np.sum(
            (stored_predictions == 0)
            &
            (y_true == 0)
        )
    )

    fn = int(
        np.sum(
            (stored_predictions == 0)
            &
            (y_true == 1)
        )
    )


    if (
        tp + fn
        !=
        positive_count
    ):

        raise RuntimeError(
            f"{candidate_id}: positive "
            "accounting mismatch."
        )


    if (
        tn + fp
        !=
        negative_count
    ):

        raise RuntimeError(
            f"{candidate_id}: negative "
            "accounting mismatch."
        )


    # ========================================================
    # THRESHOLD-DEPENDENT METRICS
    # ========================================================

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


    # ========================================================
    # THRESHOLD-INDEPENDENT METRICS
    # ========================================================

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


    # ========================================================
    # SAVE METRIC ROW
    # ========================================================

    metric_rows.append(
        {
            "family":
                family,

            "candidate_id":
                candidate_id,

            "frozen_threshold":
                threshold,

            "rows":
                len(
                    y_true
                ),

            "positive":
                positive_count,

            "negative":
                negative_count,

            "predicted_positive":
                int(
                    stored_predictions.sum()
                ),

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

            "threshold_source":
                "CYCLE24_VALIDATION",

            "threshold_reoptimized":
                False,

            "model_retrained":
                False,

            "cycle25_used":
                False,
        }
    )


    # ========================================================
    # FIXED-WIDTH CALIBRATION TABLE
    # ========================================================

    bin_indices = np.floor(
        probabilities
        *
        10.0
    ).astype(
        np.int64
    )


    # Probability == 1.0 belongs to final bin.
    bin_indices = np.clip(
        bin_indices,
        0,
        9,
    )


    for bin_index in range(10):

        mask = (
            bin_indices
            ==
            bin_index
        )


        row_count = int(
            mask.sum()
        )


        bin_lower = float(
            CALIBRATION_EDGES[
                bin_index
            ]
        )


        bin_upper = float(
            CALIBRATION_EDGES[
                bin_index + 1
            ]
        )


        if row_count > 0:

            positive_in_bin = int(
                y_true[
                    mask
                ].sum()
            )


            mean_probability = float(
                probabilities[
                    mask
                ].mean()
            )


            observed_event_rate = float(
                y_true[
                    mask
                ].mean()
            )


        else:

            positive_in_bin = 0
            mean_probability = np.nan
            observed_event_rate = np.nan


        calibration_rows.append(
            {
                "family":
                    family,

                "candidate_id":
                    candidate_id,

                "bin_index":
                    bin_index,

                "bin_lower":
                    bin_lower,

                "bin_upper":
                    bin_upper,

                "bin_interval":
                    (
                        f"[{bin_lower:.1f}, "
                        f"{bin_upper:.1f}"
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
                    observed_event_rate,
            }
        )


    print(
        f"{candidate_id}: "
        f"TSS={tss:.6f}, "
        f"HSS={hss:.6f}, "
        f"F1={f1:.6f}, "
        f"PR_AUC={pr_auc:.6f}, "
        f"ROC_AUC={roc_auc:.6f}, "
        f"Brier={brier:.6f}, "
        f"TP={tp}, FP={fp}, "
        f"TN={tn}, FN={fn}, "
        f"PASS=True"
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
) != 3:

    raise RuntimeError(
        "Expected exactly 3 holdout "
        "metric rows."
    )


if len(
    calibration_df
) != 30:

    raise RuntimeError(
        "Expected exactly 30 calibration "
        "rows: 3 models x 10 bins."
    )


if metrics_df[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate holdout metric candidate."
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
            f"{column}: non-finite "
            "metric value detected."
        )


    if (
        values.min() < 0.0
        or
        values.max() > 1.0
    ):

        raise RuntimeError(
            f"{column}: metric value "
            "outside [0,1]."
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
            f"{column}: non-finite "
            "metric value detected."
        )


    if (
        values.min() < -1.0
        or
        values.max() > 1.0
    ):

        raise RuntimeError(
            f"{column}: metric value "
            "outside [-1,1]."
        )


# ============================================================
# CALIBRATION ACCOUNTING CHECK
# ============================================================

for candidate_id in registry[
    "candidate_id"
].astype(str):

    candidate_calibration = (
        calibration_df[
            calibration_df[
                "candidate_id"
            ]
            ==
            candidate_id
        ]
    )


    if int(
        candidate_calibration[
            "row_count"
        ].sum()
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            f"{candidate_id}: calibration "
            "row accounting mismatch."
        )


    if int(
        candidate_calibration[
            "positive_count"
        ].sum()
    ) != EXPECTED_POSITIVES:

        raise RuntimeError(
            f"{candidate_id}: calibration "
            "positive accounting mismatch."
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
# ROUND-TRIP CHECK
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
) != 3:

    raise RuntimeError(
        "Holdout metrics CSV round-trip "
        "row mismatch."
    )


if len(
    calibration_roundtrip
) != 30:

    raise RuntimeError(
        "Calibration CSV round-trip "
        "row mismatch."
    )


print(
    "Holdout metrics round-trip:",
    "PASS"
)

print(
    "Calibration round-trip:",
    "PASS"
)


# ============================================================
# BUILD CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Cycle24 Independent Holdout Metrics",

    "version":
        "1.0",

    "status":
        "FROZEN_HOLDOUT_POINT_METRICS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "evaluation_population":
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

    "model_count":
        3,

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
                "2*TP / (2*TP + FP + FN)",

            "PR_AUC":
                (
                    "sklearn.metrics."
                    "average_precision_score"
                ),

            "ROC_AUC":
                "sklearn.metrics.roc_auc_score",

            "Brier_score":
                "sklearn.metrics.brier_score_loss",
        },

    "threshold_policy":
        {
            "source":
                "CYCLE24_VALIDATION",

            "reoptimized_on_holdout":
                False,

            "prediction_rule":
                "probability >= frozen_threshold",
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
            "holdout_metrics":
                {
                    "file":
                        str(
                            OUTPUT_METRICS_FILE
                        ),

                    "sha256":
                        metrics_hash,

                    "rows":
                        3,
                },

            "holdout_calibration":
                {
                    "file":
                        str(
                            OUTPUT_CALIBRATION_FILE
                        ),

                    "sha256":
                        calibration_hash,

                    "rows":
                        30,
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

            "models_retrained":
                False,

            "thresholds_reoptimized":
                False,

            "family_selection_changed":
                False,

            "cycle25_used":
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

            "holdout_model_registry":
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

    "immutability_rule":
        (
            "These Cycle24 holdout point metrics and "
            "calibration diagnostics were computed using "
            "the already-frozen models and validation-derived "
            "thresholds. They must not trigger retraining, "
            "threshold changes, candidate replacement, "
            "preprocessing changes, or feature changes."
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
    "HOLDOUT METRIC SUMMARY"
)

print(
    "===================================="
)


for _, row in metrics_df.iterrows():

    print(
        f"{row['family']}: "
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
    "Holdout point metrics computed:",
    True
)

print(
    "Calibration diagnostics computed:",
    True
)

print(
    "Bootstrap confidence intervals:",
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
    "Cycle25 read:",
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
    "MODEL CHAPTER 7.3 COMPLETE — HOLDOUT POINT METRICS FROZEN"
)

print(
    "===================================="
)