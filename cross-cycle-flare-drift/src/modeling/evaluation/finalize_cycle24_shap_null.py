# ============================================================
# MODEL CHAPTER 9.6D
# WITHIN-CYCLE24 SHAP NULL THRESHOLDS + INTEGRITY AUDIT
#
# Purpose
# -------
# Audit the frozen Chapter 9.6C null distributions and derive
# model-specific + stratum-specific null thresholds that will
# later be used to evaluate observed Cycle24 -> Cycle25 SHAP
# drift.
#
# Frozen interpretation:
#
# Similarity metrics:
#   drift-side threshold = lower 2.5 percentile
#
# Distance metrics:
#   drift-side threshold = upper 97.5 percentile
#
# Similarity metrics:
#   - Spearman rho
#   - Kendall tau-b
#   - Top-5 overlap
#   - Top-10 overlap
#   - Signed cosine similarity
#   - Sign agreement
#
# Distance metrics:
#   - Normalized L1 distance on mean-|SHAP|
#   - Cosine distance on mean-|SHAP|
#
# Featurewise null summaries:
#   - |delta mean-|SHAP||
#   - |delta mean signed SHAP|
#
# IMPORTANT:
#   - No Cycle25
#   - No model loading
#   - No SHAP recomputation
#   - No drift conclusion
#   - No BH-FDR yet
# ============================================================


from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# FROZEN SETTINGS
# ============================================================

EXPECTED_REPLICATES = 2000

EXPECTED_MODELS = 3
EXPECTED_STRATA = 3
EXPECTED_FEATURES = 25

EXPECTED_METRIC_ROWS = (
    EXPECTED_REPLICATES
    *
    EXPECTED_MODELS
    *
    EXPECTED_STRATA
)

EXPECTED_FEATUREWISE_ROWS = (
    EXPECTED_REPLICATES
    *
    EXPECTED_MODELS
    *
    EXPECTED_STRATA
    *
    EXPECTED_FEATURES
)

LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975

QUANTILE_METHOD = "linear"


MODEL_IDS = [
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


SIMILARITY_METRICS = [
    "spearman_rho",
    "kendall_tau_b",
    "top5_overlap",
    "top10_overlap",
    "signed_cosine_similarity",
    "sign_agreement",
]


DISTANCE_METRICS = [
    "normalized_l1_mean_abs",
    "cosine_distance_mean_abs",
]


ALL_STRUCTURE_METRICS = (
    SIMILARITY_METRICS
    +
    DISTANCE_METRICS
)


# ============================================================
# INPUT PATHS
# ============================================================

NULL_REPLICATES_FILE = Path(
    "artifacts/models/shift_controls/shap/null/"
    "cycle24_shap_null_replicates.parquet"
)


FEATUREWISE_REPLICATES_FILE = Path(
    "artifacts/models/shift_controls/shap/null/"
    "cycle24_shap_null_featurewise_replicates.parquet"
)


DRAW_COUNTS_FILE = Path(
    "artifacts/models/shift_controls/shap/null/"
    "cycle24_shap_null_draw_counts.npz"
)


GROUP_ORDER_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "cycle24_shap_null_group_order.csv"
)


EXECUTION_MANIFEST_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "cycle24_shap_null_execution_manifest.csv"
)


EXECUTION_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "cycle24_shap_null_execution_contract.json"
)


SHAP_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "cycle24_holdout_shap_contract.json"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

STRUCTURE_THRESHOLDS_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "cycle24_shap_null_structure_thresholds.csv"
)


FEATUREWISE_THRESHOLDS_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "cycle24_shap_null_featurewise_thresholds.csv"
)


NULL_SUMMARY_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "cycle24_shap_null_distribution_summary.csv"
)


AUDIT_MANIFEST_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "cycle24_shap_null_integrity_manifest.csv"
)


FINAL_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "cycle24_shap_null_contract.json"
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


def verify_exists(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def quantile(
    values: np.ndarray,
    q: float,
) -> float:

    return float(
        np.quantile(
            values,
            q,
            method=QUANTILE_METHOD,
        )
    )


# ============================================================
# START
# ============================================================

print("=" * 76)

print(
    "MODEL CHAPTER 9.6D — "
    "NULL THRESHOLDS + INTEGRITY AUDIT"
)

print("=" * 76)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_inputs = [
    NULL_REPLICATES_FILE,
    FEATUREWISE_REPLICATES_FILE,
    DRAW_COUNTS_FILE,
    GROUP_ORDER_FILE,
    EXECUTION_MANIFEST_FILE,
    EXECUTION_CONTRACT_FILE,
    SHAP_CONTRACT_FILE,
]


for path in required_inputs:

    verify_exists(path)


# ============================================================
# OUTPUT PROTECTION
# ============================================================

outputs = [
    STRUCTURE_THRESHOLDS_FILE,
    FEATUREWISE_THRESHOLDS_FILE,
    NULL_SUMMARY_FILE,
    AUDIT_MANIFEST_FILE,
    FINAL_CONTRACT_FILE,
]


for path in outputs:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter 9.6D output."
        )


# ============================================================
# VERIFY 9.6B CONTRACT
# ============================================================

shap_contract = load_json(
    SHAP_CONTRACT_FILE
)


if (
    shap_contract.get("status")
    !=
    "FROZEN_CYCLE24_HOLDOUT_SHAP"
):

    raise RuntimeError(
        "Chapter 9.6B SHAP contract "
        "is not frozen."
    )


print(
    "Chapter 9.6B SHAP contract:",
    "PASS"
)


# ============================================================
# VERIFY 9.6C CONTRACT
# ============================================================

execution_contract = load_json(
    EXECUTION_CONTRACT_FILE
)


expected_status = (
    "CYCLE24_SHAP_NULL_COMPUTED_"
    "PENDING_9_6D_AUDIT"
)


if (
    execution_contract.get("status")
    !=
    expected_status
):

    raise RuntimeError(
        "Chapter 9.6C execution contract "
        "has unexpected status."
    )


bootstrap_design = execution_contract[
    "bootstrap_design"
]


if int(
    bootstrap_design[
        "replicates"
    ]
) != EXPECTED_REPLICATES:

    raise RuntimeError(
        "Bootstrap replicate-count mismatch."
    )


if int(
    bootstrap_design[
        "a_seed"
    ]
) != 29062:

    raise RuntimeError(
        "Bootstrap seed A mismatch."
    )


if int(
    bootstrap_design[
        "b_seed"
    ]
) != 29063:

    raise RuntimeError(
        "Bootstrap seed B mismatch."
    )


if (
    bootstrap_design[
        "unit"
    ]
    !=
    "ENTITY_GROUP_ID"
):

    raise RuntimeError(
        "Bootstrap unit mismatch."
    )


if (
    bootstrap_design[
        "same_draws_across_models"
    ]
    is not True
):

    raise RuntimeError(
        "Draw-sharing across models mismatch."
    )


if (
    bootstrap_design[
        "same_draws_across_strata"
    ]
    is not True
):

    raise RuntimeError(
        "Draw-sharing across strata mismatch."
    )


print(
    "Chapter 9.6C execution contract:",
    "PASS"
)


# ============================================================
# VERIFY 9.6C ARTIFACT HASHES
# ============================================================

artifacts = execution_contract[
    "artifacts"
]


artifact_checks = [
    (
        NULL_REPLICATES_FILE,
        artifacts[
            "metric_replicates"
        ][
            "sha256"
        ],
    ),

    (
        FEATUREWISE_REPLICATES_FILE,
        artifacts[
            "featurewise_replicates"
        ][
            "sha256"
        ],
    ),

    (
        DRAW_COUNTS_FILE,
        artifacts[
            "draw_counts"
        ][
            "sha256"
        ],
    ),

    (
        GROUP_ORDER_FILE,
        artifacts[
            "group_order"
        ][
            "sha256"
        ],
    ),

    (
        EXECUTION_MANIFEST_FILE,
        artifacts[
            "execution_manifest"
        ][
            "sha256"
        ],
    ),
]


for path, expected_hash in artifact_checks:

    actual_hash = sha256_file(
        path
    )


    if actual_hash != expected_hash:

        raise RuntimeError(
            f"Artifact hash mismatch: {path}"
        )


print(
    "9.6C artifact hashes:",
    "PASS"
)


# ============================================================
# VERIFY DRAW COUNTS
# ============================================================

draw_archive = np.load(
    DRAW_COUNTS_FILE
)


if (
    "counts_a"
    not in draw_archive.files
    or
    "counts_b"
    not in draw_archive.files
):

    raise RuntimeError(
        "Bootstrap draw-count archive "
        "missing A/B arrays."
    )


counts_a = draw_archive[
    "counts_a"
]


counts_b = draw_archive[
    "counts_b"
]


if counts_a.shape != (
    EXPECTED_REPLICATES,
    176,
):

    raise RuntimeError(
        "Bootstrap A shape mismatch."
    )


if counts_b.shape != (
    EXPECTED_REPLICATES,
    176,
):

    raise RuntimeError(
        "Bootstrap B shape mismatch."
    )


if not np.all(
    counts_a.sum(axis=1)
    ==
    176
):

    raise RuntimeError(
        "Bootstrap A group-count integrity failed."
    )


if not np.all(
    counts_b.sum(axis=1)
    ==
    176
):

    raise RuntimeError(
        "Bootstrap B group-count integrity failed."
    )


if np.array_equal(
    counts_a,
    counts_b,
):

    raise RuntimeError(
        "Bootstrap A and B are identical."
    )


print(
    "A/B bootstrap draw integrity:",
    "PASS"
)


# ============================================================
# GROUP ORDER AUDIT
# ============================================================

group_order = pd.read_csv(
    GROUP_ORDER_FILE,
    keep_default_na=False,
)


if len(
    group_order
) != 176:

    raise RuntimeError(
        "Expected exactly 176 "
        "ENTITY_GROUP entries."
    )


if group_order[
    "ENTITY_GROUP_ID"
].astype(str).duplicated().any():

    raise RuntimeError(
        "Duplicate ENTITY_GROUP in "
        "frozen group order."
    )


print(
    "ENTITY_GROUP order integrity:",
    "PASS"
)


# ============================================================
# EXECUTION MANIFEST AUDIT
# ============================================================

execution_manifest = pd.read_csv(
    EXECUTION_MANIFEST_FILE,
    keep_default_na=False,
)


if len(
    execution_manifest
) != 9:

    raise RuntimeError(
        "Expected exactly 9 execution "
        "manifest rows."
    )


if int(
    execution_manifest[
        "empty_a_replicates"
    ].sum()
) != 0:

    raise RuntimeError(
        "Empty A bootstrap strata detected."
    )


if int(
    execution_manifest[
        "empty_b_replicates"
    ].sum()
) != 0:

    raise RuntimeError(
        "Empty B bootstrap strata detected."
    )


print(
    "Bootstrap execution manifest:",
    "PASS"
)


# ============================================================
# LOAD STRUCTURE NULL
# ============================================================

null_df = pd.read_parquet(
    NULL_REPLICATES_FILE
)


if len(
    null_df
) != EXPECTED_METRIC_ROWS:

    raise RuntimeError(
        "Structure-null row count mismatch."
    )


required_metric_columns = {
    "replicate_id",
    "candidate_id",
    "model_family",
    "stratum",
    "a_rows",
    "b_rows",
    "a_unique_groups",
    "b_unique_groups",
    *ALL_STRUCTURE_METRICS,
}


missing_metric_columns = (
    required_metric_columns
    -
    set(
        null_df.columns
    )
)


if missing_metric_columns:

    raise RuntimeError(
        "Structure-null missing columns: "
        f"{sorted(missing_metric_columns)}"
    )


if null_df.duplicated(
    subset=[
        "candidate_id",
        "stratum",
        "replicate_id",
    ]
).any():

    raise RuntimeError(
        "Duplicate structure-null replicate."
    )


if set(
    null_df[
        "candidate_id"
    ].astype(str)
) != set(
    MODEL_IDS
):

    raise RuntimeError(
        "Structure-null model set mismatch."
    )


if set(
    null_df[
        "stratum"
    ].astype(str)
) != set(
    STRATA
):

    raise RuntimeError(
        "Structure-null stratum set mismatch."
    )


for metric in ALL_STRUCTURE_METRICS:

    values = null_df[
        metric
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"Non-finite structure metric: {metric}"
        )


print(
    "Structure-null artifact integrity:",
    "PASS"
)


# ============================================================
# METRIC RANGE AUDITS
# ============================================================

bounded_minus1_plus1 = [
    "spearman_rho",
    "kendall_tau_b",
    "signed_cosine_similarity",
]


bounded_zero_one = [
    "top5_overlap",
    "top10_overlap",
    "normalized_l1_mean_abs",
    "sign_agreement",
]


for metric in bounded_minus1_plus1:

    values = null_df[
        metric
    ].to_numpy(
        dtype=np.float64
    )


    if (
        np.any(
            values < -1.0000001
        )
        or
        np.any(
            values > 1.0000001
        )
    ):

        raise RuntimeError(
            f"Metric out of [-1,1]: {metric}"
        )


for metric in bounded_zero_one:

    values = null_df[
        metric
    ].to_numpy(
        dtype=np.float64
    )


    if (
        np.any(
            values < -1e-10
        )
        or
        np.any(
            values > 1.0000001
        )
    ):

        raise RuntimeError(
            f"Metric out of [0,1]: {metric}"
        )


cosine_distance_values = null_df[
    "cosine_distance_mean_abs"
].to_numpy(
    dtype=np.float64
)


if (
    np.any(
        cosine_distance_values < -1e-10
    )
    or
    np.any(
        cosine_distance_values > 2.0000001
    )
):

    raise RuntimeError(
        "Cosine distance out of [0,2]."
    )


print(
    "Structure metric ranges:",
    "PASS"
)


# ============================================================
# REPLICATE COMPLETENESS
# ============================================================

for candidate_id in MODEL_IDS:

    for stratum in STRATA:

        subset = null_df[
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
        ]


        if len(
            subset
        ) != EXPECTED_REPLICATES:

            raise RuntimeError(
                f"{candidate_id}/{stratum}: "
                "replicate-count mismatch."
            )


        replicate_ids = np.sort(
            subset[
                "replicate_id"
            ].to_numpy(
                dtype=np.int64
            )
        )


        if not np.array_equal(
            replicate_ids,
            np.arange(
                EXPECTED_REPLICATES,
                dtype=np.int64,
            ),
        ):

            raise RuntimeError(
                f"{candidate_id}/{stratum}: "
                "replicate ID integrity failed."
            )


print(
    "2000-replicate completeness:",
    "PASS"
)


# ============================================================
# STRUCTURE NULL DISTRIBUTION SUMMARY
# ============================================================

summary_rows = []

threshold_rows = []


for candidate_id in MODEL_IDS:

    model_df = null_df[
        null_df[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    model_families = model_df[
        "model_family"
    ].astype(str).unique()


    if len(
        model_families
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: inconsistent "
            "model family."
        )


    model_family = model_families[
        0
    ]


    for stratum in STRATA:

        subset = model_df[
            model_df[
                "stratum"
            ].astype(str)
            ==
            stratum
        ]


        for metric in ALL_STRUCTURE_METRICS:

            values = subset[
                metric
            ].to_numpy(
                dtype=np.float64
            )


            q025 = quantile(
                values,
                LOWER_QUANTILE,
            )


            q25 = quantile(
                values,
                0.25,
            )


            q50 = quantile(
                values,
                0.50,
            )


            q75 = quantile(
                values,
                0.75,
            )


            q975 = quantile(
                values,
                UPPER_QUANTILE,
            )


            summary_rows.append(
                {
                    "candidate_id":
                        candidate_id,

                    "model_family":
                        model_family,

                    "stratum":
                        stratum,

                    "metric":
                        metric,

                    "n_replicates":
                        EXPECTED_REPLICATES,

                    "mean":
                        float(
                            values.mean()
                        ),

                    "std":
                        float(
                            values.std(
                                ddof=1
                            )
                        ),

                    "minimum":
                        float(
                            values.min()
                        ),

                    "q025":
                        q025,

                    "q25":
                        q25,

                    "median":
                        q50,

                    "q75":
                        q75,

                    "q975":
                        q975,

                    "maximum":
                        float(
                            values.max()
                        ),
                }
            )


            if metric in SIMILARITY_METRICS:

                threshold_type = (
                    "LOWER_2P5_SIMILARITY"
                )

                threshold_value = q025

                drift_direction = (
                    "OBSERVED_BELOW_THRESHOLD"
                )


            elif metric in DISTANCE_METRICS:

                threshold_type = (
                    "UPPER_97P5_DISTANCE"
                )

                threshold_value = q975

                drift_direction = (
                    "OBSERVED_ABOVE_THRESHOLD"
                )


            else:

                raise RuntimeError(
                    f"Metric orientation undefined: "
                    f"{metric}"
                )


            threshold_rows.append(
                {
                    "candidate_id":
                        candidate_id,

                    "model_family":
                        model_family,

                    "stratum":
                        stratum,

                    "metric":
                        metric,

                    "metric_type":
                        (
                            "SIMILARITY"
                            if metric
                            in SIMILARITY_METRICS
                            else
                            "DISTANCE"
                        ),

                    "n_null_replicates":
                        EXPECTED_REPLICATES,

                    "threshold_type":
                        threshold_type,

                    "threshold_quantile":
                        (
                            LOWER_QUANTILE
                            if metric
                            in SIMILARITY_METRICS
                            else
                            UPPER_QUANTILE
                        ),

                    "threshold_value":
                        threshold_value,

                    "future_drift_side_rule":
                        drift_direction,

                    "comparison_is_strict":
                        True,
                }
            )


summary_df = pd.DataFrame(
    summary_rows
)


thresholds_df = pd.DataFrame(
    threshold_rows
)


expected_summary_rows = (
    EXPECTED_MODELS
    *
    EXPECTED_STRATA
    *
    len(
        ALL_STRUCTURE_METRICS
    )
)


if len(
    summary_df
) != expected_summary_rows:

    raise RuntimeError(
        "Null summary row count mismatch."
    )


if len(
    thresholds_df
) != expected_summary_rows:

    raise RuntimeError(
        "Structure-threshold row "
        "count mismatch."
    )


if thresholds_df.duplicated(
    subset=[
        "candidate_id",
        "stratum",
        "metric",
    ]
).any():

    raise RuntimeError(
        "Duplicate structure threshold."
    )


print(
    "Structure thresholds derived:",
    len(
        thresholds_df
    ),
    "PASS"
)


# ============================================================
# FEATUREWISE NULL AUDIT
# ============================================================

featurewise_df = pd.read_parquet(
    FEATUREWISE_REPLICATES_FILE
)


if len(
    featurewise_df
) != EXPECTED_FEATUREWISE_ROWS:

    raise RuntimeError(
        "Featurewise-null row count mismatch."
    )


required_featurewise_columns = {
    "replicate_id",
    "candidate_id",
    "model_family",
    "stratum",
    "feature_index",
    "feature",
    "a_mean_abs_shap",
    "b_mean_abs_shap",
    "delta_mean_abs_shap",
    "abs_delta_mean_abs_shap",
    "a_mean_signed_shap",
    "b_mean_signed_shap",
    "delta_mean_signed_shap",
    "abs_delta_mean_signed_shap",
}


missing_featurewise_columns = (
    required_featurewise_columns
    -
    set(
        featurewise_df.columns
    )
)


if missing_featurewise_columns:

    raise RuntimeError(
        "Featurewise-null missing columns: "
        f"{sorted(missing_featurewise_columns)}"
    )


numeric_featurewise_columns = [
    "a_mean_abs_shap",
    "b_mean_abs_shap",
    "delta_mean_abs_shap",
    "abs_delta_mean_abs_shap",
    "a_mean_signed_shap",
    "b_mean_signed_shap",
    "delta_mean_signed_shap",
    "abs_delta_mean_signed_shap",
]


for column in numeric_featurewise_columns:

    values = featurewise_df[
        column
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"Non-finite featurewise "
            f"null values: {column}"
        )


if np.any(
    featurewise_df[
        "abs_delta_mean_abs_shap"
    ].to_numpy(
        dtype=np.float64
    )
    <
    -1e-15
):

    raise RuntimeError(
        "Negative absolute mean-|SHAP| "
        "delta detected."
    )


if np.any(
    featurewise_df[
        "abs_delta_mean_signed_shap"
    ].to_numpy(
        dtype=np.float64
    )
    <
    -1e-15
):

    raise RuntimeError(
        "Negative absolute signed-SHAP "
        "delta detected."
    )


print(
    "Featurewise-null artifact integrity:",
    "PASS"
)


# ============================================================
# FEATUREWISE COMPLETENESS
# ============================================================

expected_feature_set = set(
    FEATURES
)


if set(
    featurewise_df[
        "feature"
    ].astype(str)
) != expected_feature_set:

    raise RuntimeError(
        "Featurewise-null feature set mismatch."
    )


for candidate_id in MODEL_IDS:

    for stratum in STRATA:

        subset = featurewise_df[
            (
                featurewise_df[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            )
            &
            (
                featurewise_df[
                    "stratum"
                ].astype(str)
                ==
                stratum
            )
        ]


        expected_rows = (
            EXPECTED_REPLICATES
            *
            EXPECTED_FEATURES
        )


        if len(
            subset
        ) != expected_rows:

            raise RuntimeError(
                f"{candidate_id}/{stratum}: "
                "featurewise replicate "
                "count mismatch."
            )


        for feature in FEATURES:

            feature_subset = subset[
                subset[
                    "feature"
                ].astype(str)
                ==
                feature
            ]


            if len(
                feature_subset
            ) != EXPECTED_REPLICATES:

                raise RuntimeError(
                    f"{candidate_id}/{stratum}/"
                    f"{feature}: expected "
                    "2000 featurewise replicates."
                )


print(
    "Featurewise 2000-replicate completeness:",
    "PASS"
)


# ============================================================
# FEATUREWISE NULL THRESHOLDS
# ============================================================

feature_threshold_rows = []


for candidate_id in MODEL_IDS:

    for stratum in STRATA:

        subset = featurewise_df[
            (
                featurewise_df[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            )
            &
            (
                featurewise_df[
                    "stratum"
                ].astype(str)
                ==
                stratum
            )
        ]


        model_families = subset[
            "model_family"
        ].astype(str).unique()


        if len(
            model_families
        ) != 1:

            raise RuntimeError(
                f"{candidate_id}/{stratum}: "
                "model family mismatch."
            )


        model_family = model_families[
            0
        ]


        for feature_index, feature in enumerate(
            FEATURES,
            start=1,
        ):

            feature_subset = subset[
                subset[
                    "feature"
                ].astype(str)
                ==
                feature
            ]


            abs_delta_abs = feature_subset[
                "abs_delta_mean_abs_shap"
            ].to_numpy(
                dtype=np.float64
            )


            abs_delta_signed = feature_subset[
                "abs_delta_mean_signed_shap"
            ].to_numpy(
                dtype=np.float64
            )


            signed_delta_abs = feature_subset[
                "delta_mean_abs_shap"
            ].to_numpy(
                dtype=np.float64
            )


            signed_delta_signed = feature_subset[
                "delta_mean_signed_shap"
            ].to_numpy(
                dtype=np.float64
            )


            feature_threshold_rows.append(
                {
                    "candidate_id":
                        candidate_id,

                    "model_family":
                        model_family,

                    "stratum":
                        stratum,

                    "feature_index":
                        feature_index,

                    "feature":
                        feature,

                    "n_null_replicates":
                        EXPECTED_REPLICATES,

                    "mean_abs_delta_mean_abs_shap":
                        float(
                            abs_delta_abs.mean()
                        ),

                    "median_abs_delta_mean_abs_shap":
                        quantile(
                            abs_delta_abs,
                            0.50,
                        ),

                    "q95_abs_delta_mean_abs_shap":
                        quantile(
                            abs_delta_abs,
                            0.95,
                        ),

                    "q975_abs_delta_mean_abs_shap":
                        quantile(
                            abs_delta_abs,
                            0.975,
                        ),

                    "mean_abs_delta_mean_signed_shap":
                        float(
                            abs_delta_signed.mean()
                        ),

                    "median_abs_delta_mean_signed_shap":
                        quantile(
                            abs_delta_signed,
                            0.50,
                        ),

                    "q95_abs_delta_mean_signed_shap":
                        quantile(
                            abs_delta_signed,
                            0.95,
                        ),

                    "q975_abs_delta_mean_signed_shap":
                        quantile(
                            abs_delta_signed,
                            0.975,
                        ),

                    "q025_signed_delta_mean_abs_shap":
                        quantile(
                            signed_delta_abs,
                            0.025,
                        ),

                    "q975_signed_delta_mean_abs_shap":
                        quantile(
                            signed_delta_abs,
                            0.975,
                        ),

                    "q025_signed_delta_mean_signed_shap":
                        quantile(
                            signed_delta_signed,
                            0.025,
                        ),

                    "q975_signed_delta_mean_signed_shap":
                        quantile(
                            signed_delta_signed,
                            0.975,
                        ),

                    "future_primary_featurewise_test":
                        (
                            "EMPIRICAL_ABSOLUTE_DELTA_"
                            "AGAINST_FULL_NULL"
                        ),

                    "bh_fdr_applied_here":
                        False,
                }
            )


feature_thresholds_df = pd.DataFrame(
    feature_threshold_rows
)


expected_feature_threshold_rows = (
    EXPECTED_MODELS
    *
    EXPECTED_STRATA
    *
    EXPECTED_FEATURES
)


if len(
    feature_thresholds_df
) != expected_feature_threshold_rows:

    raise RuntimeError(
        "Featurewise threshold row-count "
        "mismatch."
    )


if feature_thresholds_df.duplicated(
    subset=[
        "candidate_id",
        "stratum",
        "feature",
    ]
).any():

    raise RuntimeError(
        "Duplicate featurewise threshold."
    )


print(
    "Featurewise threshold summaries:",
    len(
        feature_thresholds_df
    ),
    "PASS"
)


# ============================================================
# AUDIT MANIFEST
# ============================================================

audit_rows = []


for candidate_id in MODEL_IDS:

    for stratum in STRATA:

        structure_subset = null_df[
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
        ]


        feature_subset = featurewise_df[
            (
                featurewise_df[
                    "candidate_id"
                ].astype(str)
                ==
                candidate_id
            )
            &
            (
                featurewise_df[
                    "stratum"
                ].astype(str)
                ==
                stratum
            )
        ]


        audit_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "stratum":
                    stratum,

                "structure_replicates":
                    len(
                        structure_subset
                    ),

                "featurewise_rows":
                    len(
                        feature_subset
                    ),

                "features":
                    int(
                        feature_subset[
                            "feature"
                        ].nunique()
                    ),

                "structure_metrics":
                    len(
                        ALL_STRUCTURE_METRICS
                    ),

                "finite_structure_metrics":
                    all(
                        np.isfinite(
                            structure_subset[
                                metric
                            ].to_numpy(
                                dtype=np.float64
                            )
                        ).all()
                        for metric
                        in ALL_STRUCTURE_METRICS
                    ),

                "finite_featurewise_values":
                    bool(
                        np.isfinite(
                            feature_subset[
                                numeric_featurewise_columns
                            ].to_numpy(
                                dtype=np.float64
                            )
                        ).all()
                    ),

                "integrity_pass":
                    True,
            }
        )


audit_manifest_df = pd.DataFrame(
    audit_rows
)


if len(
    audit_manifest_df
) != 9:

    raise RuntimeError(
        "Expected 9 audit manifest rows."
    )


if not audit_manifest_df[
    "integrity_pass"
].all():

    raise RuntimeError(
        "At least one null integrity "
        "audit failed."
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

STRUCTURE_THRESHOLDS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FEATUREWISE_THRESHOLDS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

NULL_SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT_MANIFEST_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


thresholds_df.to_csv(
    STRUCTURE_THRESHOLDS_FILE,
    index=False,
)


feature_thresholds_df.to_csv(
    FEATUREWISE_THRESHOLDS_FILE,
    index=False,
)


summary_df.to_csv(
    NULL_SUMMARY_FILE,
    index=False,
)


audit_manifest_df.to_csv(
    AUDIT_MANIFEST_FILE,
    index=False,
)


# ============================================================
# ROUND-TRIP AUDIT
# ============================================================

thresholds_rt = pd.read_csv(
    STRUCTURE_THRESHOLDS_FILE,
    keep_default_na=False,
)


feature_thresholds_rt = pd.read_csv(
    FEATUREWISE_THRESHOLDS_FILE,
    keep_default_na=False,
)


summary_rt = pd.read_csv(
    NULL_SUMMARY_FILE,
    keep_default_na=False,
)


audit_rt = pd.read_csv(
    AUDIT_MANIFEST_FILE,
    keep_default_na=False,
)


if len(
    thresholds_rt
) != expected_summary_rows:

    raise RuntimeError(
        "Structure threshold "
        "round-trip failed."
    )


if len(
    feature_thresholds_rt
) != expected_feature_threshold_rows:

    raise RuntimeError(
        "Featurewise threshold "
        "round-trip failed."
    )


if len(
    summary_rt
) != expected_summary_rows:

    raise RuntimeError(
        "Null summary round-trip failed."
    )


if len(
    audit_rt
) != 9:

    raise RuntimeError(
        "Audit manifest round-trip failed."
    )


print(
    "Output round-trip integrity:",
    "PASS"
)


# ============================================================
# HASH OUTPUTS
# ============================================================

structure_threshold_hash = sha256_file(
    STRUCTURE_THRESHOLDS_FILE
)


featurewise_threshold_hash = sha256_file(
    FEATUREWISE_THRESHOLDS_FILE
)


summary_hash = sha256_file(
    NULL_SUMMARY_FILE
)


audit_manifest_hash = sha256_file(
    AUDIT_MANIFEST_FILE
)


# ============================================================
# FINAL NULL CONTRACT
# ============================================================

final_contract = {

    "contract_name":
        (
            "Frozen Within-Cycle24 "
            "SHAP Null Baseline"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CYCLE24_SHAP_NULL_BASELINE",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_9_6D",

    "source_execution":
        {
            "file":
                str(
                    EXECUTION_CONTRACT_FILE
                ),

            "sha256":
                sha256_file(
                    EXECUTION_CONTRACT_FILE
                ),

            "status":
                expected_status,
        },

    "source_shap_contract":
        {
            "file":
                str(
                    SHAP_CONTRACT_FILE
                ),

            "sha256":
                sha256_file(
                    SHAP_CONTRACT_FILE
                ),

            "status":
                "FROZEN_CYCLE24_HOLDOUT_SHAP",
        },

    "null_design":
        {
            "population":
                "CYCLE24_HOLDOUT",

            "bootstrap_unit":
                "ENTITY_GROUP_ID",

            "replicates":
                EXPECTED_REPLICATES,

            "seed_a":
                29062,

            "seed_b":
                29063,

            "same_draws_across_models":
                True,

            "same_draws_across_strata":
                True,

            "models":
                MODEL_IDS,

            "strata":
                STRATA,
        },

    "structure_threshold_rule":
        {
            "similarity_metrics":
                SIMILARITY_METRICS,

            "similarity_drift_side":
                (
                    "OBSERVED_VALUE_STRICTLY_BELOW_"
                    "WITHIN_C24_2P5_PERCENTILE"
                ),

            "distance_metrics":
                DISTANCE_METRICS,

            "distance_drift_side":
                (
                    "OBSERVED_VALUE_STRICTLY_ABOVE_"
                    "WITHIN_C24_97P5_PERCENTILE"
                ),

            "lower_quantile":
                LOWER_QUANTILE,

            "upper_quantile":
                UPPER_QUANTILE,

            "quantile_method":
                QUANTILE_METHOD,

            "equality_counts_as_exceedance":
                False,
        },

    "featurewise_inference_rule":
        {
            "primary_future_null":
                (
                    "FULL_2000_REPLICATE_"
                    "ABSOLUTE_DELTA_DISTRIBUTION"
                ),

            "mean_abs_shap_quantity":
                "abs_delta_mean_abs_shap",

            "mean_signed_shap_quantity":
                "abs_delta_mean_signed_shap",

            "threshold_summary_quantile":
                0.975,

            "empirical_p_values":
                (
                    "COMPUTED_LATER_AGAINST_FULL_"
                    "NULL_REPLICATES"
                ),

            "multiple_testing":
                "BH_FDR_Q_0.05_LATER",

            "bh_fdr_applied_in_9_6d":
                False,
        },

    "artifacts":
        {
            "structure_thresholds":
                {
                    "file":
                        str(
                            STRUCTURE_THRESHOLDS_FILE
                        ),

                    "sha256":
                        structure_threshold_hash,

                    "rows":
                        expected_summary_rows,
                },

            "featurewise_thresholds":
                {
                    "file":
                        str(
                            FEATUREWISE_THRESHOLDS_FILE
                        ),

                    "sha256":
                        featurewise_threshold_hash,

                    "rows":
                        expected_feature_threshold_rows,
                },

            "distribution_summary":
                {
                    "file":
                        str(
                            NULL_SUMMARY_FILE
                        ),

                    "sha256":
                        summary_hash,

                    "rows":
                        expected_summary_rows,
                },

            "integrity_manifest":
                {
                    "file":
                        str(
                            AUDIT_MANIFEST_FILE
                        ),

                    "sha256":
                        audit_manifest_hash,

                    "rows":
                        9,
                },

            "full_structure_null":
                {
                    "file":
                        str(
                            NULL_REPLICATES_FILE
                        ),

                    "sha256":
                        sha256_file(
                            NULL_REPLICATES_FILE
                        ),

                    "rows":
                        EXPECTED_METRIC_ROWS,
                },

            "full_featurewise_null":
                {
                    "file":
                        str(
                            FEATUREWISE_REPLICATES_FILE
                        ),

                    "sha256":
                        sha256_file(
                            FEATUREWISE_REPLICATES_FILE
                        ),

                    "rows":
                        EXPECTED_FEATUREWISE_ROWS,
                },
        },

    "scientific_guardrails":
        {
            "cycle25_read":
                False,

            "models_loaded":
                False,

            "models_retrained":
                False,

            "shap_recomputed":
                False,

            "null_thresholds_frozen_before_cycle25_shap":
                True,

            "cross_cycle_drift_claim_made":
                False,

            "featurewise_fdr_performed":
                False,
        },

    "interpretation":
        (
            "These thresholds quantify ordinary "
            "within-Cycle24 explanation variability. "
            "A future Cycle25 SHAP difference can only "
            "be considered beyond the within-Cycle24 "
            "null when the corresponding same-model "
            "and same-stratum observed metric crosses "
            "the frozen drift-side threshold. Crossing "
            "a null threshold alone will not constitute "
            "the final explanation-drift conclusion; "
            "class-conditional, common-support/"
            "reweighting, correlated-feature-group, "
            "and featurewise robustness analyses remain "
            "required."
        ),

    "next_stage":
        "MODEL_CHAPTER_9_7_FREEZE_SHIFT_NULL_CONTROLS",
}


with FINAL_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        final_contract,
        file,
        indent=2,
    )


# ============================================================
# FINAL CONTRACT ROUND-TRIP
# ============================================================

contract_rt = load_json(
    FINAL_CONTRACT_FILE
)


if (
    contract_rt[
        "status"
    ]
    !=
    "FROZEN_CYCLE24_SHAP_NULL_BASELINE"
):

    raise RuntimeError(
        "Final 9.6D contract "
        "round-trip failed."
    )


# ============================================================
# PRINT COMPACT THRESHOLD PREVIEW
# ============================================================

print()

print("-" * 76)

print(
    "OVERALL NULL THRESHOLD PREVIEW"
)

print("-" * 76)


preview = thresholds_df[
    thresholds_df[
        "stratum"
    ]
    ==
    "OVERALL"
][
    [
        "candidate_id",
        "metric",
        "threshold_type",
        "threshold_value",
    ]
]


for candidate_id in MODEL_IDS:

    print()

    print(
        candidate_id
    )


    model_preview = preview[
        preview[
            "candidate_id"
        ]
        ==
        candidate_id
    ]


    for _, row in model_preview.iterrows():

        print(
            "  "
            f"{row['metric']}: "
            f"{float(row['threshold_value']):.6f} "
            f"[{row['threshold_type']}]"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 76)

print(
    "MODEL CHAPTER 9.6D — COMPLETE"
)

print("=" * 76)


print(
    "Structure null rows:",
    len(
        null_df
    )
)

print(
    "Featurewise null rows:",
    len(
        featurewise_df
    )
)

print(
    "Structure thresholds:",
    len(
        thresholds_df
    )
)

print(
    "Featurewise threshold rows:",
    len(
        feature_thresholds_df
    )
)

print(
    "Integrity manifest rows:",
    len(
        audit_manifest_df
    )
)

print(
    "Similarity threshold:",
    "LOWER 2.5%"
)

print(
    "Distance threshold:",
    "UPPER 97.5%"
)

print(
    "Quantile method:",
    QUANTILE_METHOD
)

print(
    "Cycle25 read:",
    False
)

print(
    "Models loaded:",
    False
)

print(
    "SHAP recomputed:",
    False
)

print(
    "Drift conclusion made:",
    False
)

print(
    "Status:",
    "FROZEN_CYCLE24_SHAP_NULL_BASELINE"
)


print()

print(
    "Saved:",
    STRUCTURE_THRESHOLDS_FILE
)

print(
    "Saved:",
    FEATUREWISE_THRESHOLDS_FILE
)

print(
    "Saved:",
    NULL_SUMMARY_FILE
)

print(
    "Saved:",
    AUDIT_MANIFEST_FILE
)

print(
    "Saved:",
    FINAL_CONTRACT_FILE
)


print()

print(
    "NEXT:",
    "Chapter 9.7 — Freeze Chapter 9 shift/null controls"
)