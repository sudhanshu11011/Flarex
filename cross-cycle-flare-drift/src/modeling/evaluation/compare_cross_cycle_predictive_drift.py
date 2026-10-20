from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CYCLE24_METRICS_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "holdout_metrics.csv"
)

CYCLE24_METRICS_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_metrics_contract.json"
)

CYCLE24_BOOTSTRAP_FILE = Path(
    "artifacts/models/evaluation/bootstrap/"
    "holdout_group_bootstrap_replicates.parquet"
)

CYCLE24_BOOTSTRAP_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_bootstrap_contract.json"
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

CYCLE25_BOOTSTRAP_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_bootstrap_contract.json"
)

CYCLE25_PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_evaluation_protocol.json"
)

OUTPUT_DELTA_REPLICATES_FILE = Path(
    "artifacts/models/evaluation/bootstrap/"
    "cross_cycle_metric_delta_replicates.parquet"
)

OUTPUT_DRIFT_FILE = Path(
    "artifacts/models/evaluation/metrics/"
    "cross_cycle_predictive_drift.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cross_cycle_predictive_drift_contract.json"
)


# ============================================================
# FROZEN SETTINGS
# ============================================================

CYCLE25_SCOPES = [
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


POOLED_SCOPE = "C25_ALL_2020_2025"


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


EXPECTED_MODELS = {
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
}


VALID_REPLICATES = 2000

LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975


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


def classify_tss_delta(
    lower,
    upper,
):

    if upper < 0.0:

        return (
            "MEASURABLE_DECREASE"
        )


    if lower > 0.0:

        return (
            "MEASURABLE_INCREASE"
        )


    return (
        "NO_MEASURABLE_CHANGE_AT_95CI"
    )


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 8.5 — CROSS-CYCLE PREDICTIVE-DRIFT ANALYSIS"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    CYCLE24_METRICS_FILE,
    CYCLE24_METRICS_CONTRACT_FILE,
    CYCLE24_BOOTSTRAP_FILE,
    CYCLE24_BOOTSTRAP_CONTRACT_FILE,
    CYCLE25_METRICS_FILE,
    CYCLE25_METRICS_CONTRACT_FILE,
    CYCLE25_BOOTSTRAP_FILE,
    CYCLE25_BOOTSTRAP_CONTRACT_FILE,
    CYCLE25_PROTOCOL_FILE,
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
    OUTPUT_DELTA_REPLICATES_FILE,
    OUTPUT_DRIFT_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "cross-cycle predictive-drift results."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

c24_metrics_contract = load_json(
    CYCLE24_METRICS_CONTRACT_FILE
)

c24_bootstrap_contract = load_json(
    CYCLE24_BOOTSTRAP_CONTRACT_FILE
)

c25_metrics_contract = load_json(
    CYCLE25_METRICS_CONTRACT_FILE
)

c25_bootstrap_contract = load_json(
    CYCLE25_BOOTSTRAP_CONTRACT_FILE
)

protocol = load_json(
    CYCLE25_PROTOCOL_FILE
)


# ============================================================
# CONTRACT STATUS CHECK
# ============================================================

if c24_metrics_contract.get(
    "status"
) != "FROZEN_HOLDOUT_POINT_METRICS":

    raise RuntimeError(
        "Cycle24 holdout point metrics "
        "are not frozen."
    )


if c24_bootstrap_contract.get(
    "status"
) != "FROZEN_HOLDOUT_BOOTSTRAP_CI":

    raise RuntimeError(
        "Cycle24 bootstrap results "
        "are not frozen."
    )


if c25_metrics_contract.get(
    "status"
) != "FROZEN_CYCLE25_POINT_METRICS":

    raise RuntimeError(
        "Cycle25 point metrics "
        "are not frozen."
    )


if c25_bootstrap_contract.get(
    "status"
) != "FROZEN_CYCLE25_BOOTSTRAP_CI":

    raise RuntimeError(
        "Cycle25 bootstrap results "
        "are not frozen."
    )


if protocol.get(
    "status"
) != "FROZEN_BEFORE_CYCLE25_EVALUATION":

    raise RuntimeError(
        "Cycle25 evaluation protocol "
        "is not frozen."
    )


print(
    "Cycle24 point metrics:",
    "FROZEN"
)

print(
    "Cycle24 bootstrap:",
    "FROZEN"
)

print(
    "Cycle25 point metrics:",
    "FROZEN"
)

print(
    "Cycle25 bootstrap:",
    "FROZEN"
)

print(
    "Cycle25 evaluation protocol:",
    "FROZEN"
)


# ============================================================
# VERIFY PREDECLARED COMPARISON RULE
# ============================================================

comparison_protocol = protocol[
    "cross_cycle_comparison"
]


if comparison_protocol[
    "reference"
] != "CYCLE24_HOLDOUT":

    raise RuntimeError(
        "Cross-cycle reference mismatch."
    )


if comparison_protocol[
    "primary_delta_metric"
] != "TSS":

    raise RuntimeError(
        "Primary delta metric is not TSS."
    )


if int(
    comparison_protocol[
        "difference_uncertainty"
    ][
        "replicates"
    ]
) != VALID_REPLICATES:

    raise RuntimeError(
        "Cross-cycle bootstrap "
        "replicate count mismatch."
    )


if comparison_protocol[
    "difference_uncertainty"
][
    "ci_method"
] != "percentile":

    raise RuntimeError(
        "Cross-cycle CI method mismatch."
    )


print(
    "Cross-cycle reference:",
    "CYCLE24_HOLDOUT"
)

print(
    "Primary delta metric:",
    "TSS"
)

print(
    "Delta definition:",
    "Cycle25 - Cycle24"
)

print(
    "Difference-bootstrap design:",
    "VERIFIED"
)


# ============================================================
# HASH VERIFICATION
# ============================================================

c24_metrics_hash = sha256_file(
    CYCLE24_METRICS_FILE
)


if (
    c24_metrics_hash
    !=
    c24_metrics_contract[
        "artifacts"
    ][
        "holdout_metrics"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle24 point metrics "
        "hash mismatch."
    )


c24_bootstrap_hash = sha256_file(
    CYCLE24_BOOTSTRAP_FILE
)


if (
    c24_bootstrap_hash
    !=
    c24_bootstrap_contract[
        "artifacts"
    ][
        "bootstrap_replicates"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle24 bootstrap replicate "
        "hash mismatch."
    )


c25_metrics_hash = sha256_file(
    CYCLE25_METRICS_FILE
)


if (
    c25_metrics_hash
    !=
    c25_metrics_contract[
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


c25_bootstrap_hash = sha256_file(
    CYCLE25_BOOTSTRAP_FILE
)


if (
    c25_bootstrap_hash
    !=
    c25_bootstrap_contract[
        "artifacts"
    ][
        "bootstrap_replicates"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle25 bootstrap replicate "
        "hash mismatch."
    )


print(
    "Cross-cycle artifact hash chain:",
    "PASS"
)


# ============================================================
# LOAD FROZEN ARTIFACTS
# ============================================================

c24_metrics = pd.read_csv(
    CYCLE24_METRICS_FILE,
    keep_default_na=False,
)


c24_bootstrap = pd.read_parquet(
    CYCLE24_BOOTSTRAP_FILE
)


c25_metrics = pd.read_csv(
    CYCLE25_METRICS_FILE,
    keep_default_na=False,
)


c25_bootstrap = pd.read_parquet(
    CYCLE25_BOOTSTRAP_FILE
)


# ============================================================
# POINT-METRIC STRUCTURE
# ============================================================

if len(
    c24_metrics
) != 3:

    raise RuntimeError(
        "Expected 3 Cycle24 holdout "
        "point-metric rows."
    )


if len(
    c25_metrics
) != 12:

    raise RuntimeError(
        "Expected 12 Cycle25 "
        "point-metric rows."
    )


if set(
    c24_metrics[
        "candidate_id"
    ].astype(str)
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Cycle24 candidate set mismatch."
    )


if set(
    c25_metrics[
        "candidate_id"
    ].astype(str)
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Cycle25 candidate set mismatch."
    )


if set(
    c25_metrics[
        "scope_id"
    ].astype(str)
) != set(
    CYCLE25_SCOPES
):

    raise RuntimeError(
        "Cycle25 scope set mismatch."
    )


# ============================================================
# BOOTSTRAP STRUCTURE
# ============================================================

if c24_bootstrap[
    "replicate_id"
].nunique() != VALID_REPLICATES:

    raise RuntimeError(
        "Cycle24 bootstrap replicate "
        "count mismatch."
    )


if set(
    c24_bootstrap[
        "candidate_id"
    ].astype(str)
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Cycle24 bootstrap candidate "
        "set mismatch."
    )


for scope_id in CYCLE25_SCOPES:

    scope_bootstrap = c25_bootstrap[
        c25_bootstrap[
            "scope_id"
        ].astype(str)
        ==
        scope_id
    ]


    if scope_bootstrap[
        "replicate_id"
    ].nunique() != VALID_REPLICATES:

        raise RuntimeError(
            f"{scope_id}: Cycle25 "
            "bootstrap replicate count "
            "mismatch."
        )


    if set(
        scope_bootstrap[
            "candidate_id"
        ].astype(str)
    ) != EXPECTED_MODELS:

        raise RuntimeError(
            f"{scope_id}: candidate "
            "set mismatch."
        )


print(
    "Cycle24 bootstrap structure:",
    "PASS"
)

print(
    "Cycle25 bootstrap structure:",
    "PASS"
)


# ============================================================
# BUILD DELTA REPLICATES
# ============================================================

delta_replicate_rows = []
summary_rows = []


print(
    "\n===================================="
)

print(
    "CROSS-CYCLE DELTA ANALYSIS"
)

print(
    "===================================="
)


for scope_id in CYCLE25_SCOPES:

    for candidate_id in sorted(
        EXPECTED_MODELS
    ):

        # ----------------------------------------------------
        # FAMILY
        # ----------------------------------------------------

        c25_point_row = c25_metrics[
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
            c25_point_row
        ) != 1:

            raise RuntimeError(
                f"{scope_id}/{candidate_id}: "
                "missing Cycle25 point row."
            )


        c25_point_row = (
            c25_point_row.iloc[
                0
            ]
        )


        family = str(
            c25_point_row[
                "family"
            ]
        )


        c24_point_row = c24_metrics[
            c24_metrics[
                "candidate_id"
            ].astype(str)
            ==
            candidate_id
        ]


        if len(
            c24_point_row
        ) != 1:

            raise RuntimeError(
                f"{candidate_id}: missing "
                "Cycle24 point row."
            )


        c24_point_row = (
            c24_point_row.iloc[
                0
            ]
        )


        # ----------------------------------------------------
        # BOOTSTRAP ROWS
        # ----------------------------------------------------

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
                "bootstrap length mismatch."
            )


        if len(
            c25_boot
        ) != VALID_REPLICATES:

            raise RuntimeError(
                f"{scope_id}/{candidate_id}: "
                "Cycle25 bootstrap length "
                "mismatch."
            )


        c24_ids = c24_boot[
            "replicate_id"
        ].to_numpy(
            dtype=np.int64
        )


        c25_ids = c25_boot[
            "replicate_id"
        ].to_numpy(
            dtype=np.int64
        )


        if not np.array_equal(
            c24_ids,
            c25_ids,
        ):

            raise RuntimeError(
                f"{scope_id}/{candidate_id}: "
                "replicate IDs do not align."
            )


        # ----------------------------------------------------
        # METRIC LOOP
        # ----------------------------------------------------

        for metric_name in METRICS:

            c24_values = c24_boot[
                metric_name
            ].to_numpy(
                dtype=np.float64
            )


            c25_values = c25_boot[
                metric_name
            ].to_numpy(
                dtype=np.float64
            )


            if not np.isfinite(
                c24_values
            ).all():

                raise RuntimeError(
                    f"{candidate_id}/"
                    f"{metric_name}: "
                    "non-finite Cycle24 "
                    "bootstrap value."
                )


            if not np.isfinite(
                c25_values
            ).all():

                raise RuntimeError(
                    f"{scope_id}/"
                    f"{candidate_id}/"
                    f"{metric_name}: "
                    "non-finite Cycle25 "
                    "bootstrap value."
                )


            deltas = (
                c25_values
                -
                c24_values
            )


            if not np.isfinite(
                deltas
            ).all():

                raise RuntimeError(
                    f"{scope_id}/"
                    f"{candidate_id}/"
                    f"{metric_name}: "
                    "non-finite delta."
                )


            point_c24 = float(
                c24_point_row[
                    metric_name
                ]
            )


            point_c25 = float(
                c25_point_row[
                    metric_name
                ]
            )


            point_delta = float(
                point_c25
                -
                point_c24
            )


            lower = float(
                np.quantile(
                    deltas,
                    LOWER_QUANTILE,
                    method="linear",
                )
            )


            upper = float(
                np.quantile(
                    deltas,
                    UPPER_QUANTILE,
                    method="linear",
                )
            )


            if metric_name == "tss":

                classification = (
                    classify_tss_delta(
                        lower,
                        upper,
                    )
                )

            else:

                classification = (
                    "NOT_CLASSIFIED_"
                    "PRIMARY_TSS_ONLY"
                )


            summary_rows.append(
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

                    "metric":
                        metric_name,

                    "cycle24_holdout_point":
                        point_c24,

                    "cycle25_point":
                        point_c25,

                    "delta_cycle25_minus_cycle24":
                        point_delta,

                    "bootstrap_delta_mean":
                        float(
                            deltas.mean()
                        ),

                    "bootstrap_delta_median":
                        float(
                            np.median(
                                deltas
                            )
                        ),

                    "delta_ci_lower_2p5":
                        lower,

                    "delta_ci_upper_97p5":
                        upper,

                    "confidence_level":
                        0.95,

                    "ci_method":
                        "percentile",

                    "quantile_method":
                        "linear",

                    "valid_replicates":
                        VALID_REPLICATES,

                    "predictive_change_classification":
                        classification,

                    "equivalence_claimed":
                        False,
                }
            )


            for i in range(
                VALID_REPLICATES
            ):

                delta_replicate_rows.append(
                    {
                        "scope_id":
                            scope_id,

                        "family":
                            family,

                        "candidate_id":
                            candidate_id,

                        "replicate_id":
                            int(
                                c24_ids[
                                    i
                                ]
                            ),

                        "metric":
                            metric_name,

                        "cycle24_bootstrap_value":
                            float(
                                c24_values[
                                    i
                                ]
                            ),

                        "cycle25_bootstrap_value":
                            float(
                                c25_values[
                                    i
                                ]
                            ),

                        "delta_cycle25_minus_cycle24":
                            float(
                                deltas[
                                    i
                                ]
                            ),
                    }
                )


# ============================================================
# BUILD TABLES
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


delta_replicates_df = pd.DataFrame(
    delta_replicate_rows
)


EXPECTED_SUMMARY_ROWS = (
    4
    *
    3
    *
    len(
        METRICS
    )
)


EXPECTED_DELTA_REPLICATE_ROWS = (
    4
    *
    3
    *
    len(
        METRICS
    )
    *
    VALID_REPLICATES
)


if len(
    summary_df
) != EXPECTED_SUMMARY_ROWS:

    raise RuntimeError(
        "Unexpected predictive-drift "
        "summary row count."
    )


if len(
    delta_replicates_df
) != EXPECTED_DELTA_REPLICATE_ROWS:

    raise RuntimeError(
        "Unexpected delta-bootstrap "
        "replicate row count."
    )


if summary_df.duplicated(
    subset=[
        "scope_id",
        "candidate_id",
        "metric",
    ]
).any():

    raise RuntimeError(
        "Duplicate predictive-drift "
        "summary row."
    )


# ============================================================
# TSS CLASSIFICATION SUMMARY
# ============================================================

tss_summary = summary_df[
    summary_df[
        "metric"
    ]
    ==
    "tss"
].copy()


if len(
    tss_summary
) != 12:

    raise RuntimeError(
        "Expected exactly 12 TSS "
        "cross-cycle comparisons."
    )


print(
    "\n===================================="
)

print(
    "TSS CROSS-CYCLE RESULTS"
)

print(
    "===================================="
)


for scope_id in CYCLE25_SCOPES:

    print(
        f"\n{scope_id}"
    )


    subset = tss_summary[
        tss_summary[
            "scope_id"
        ]
        ==
        scope_id
    ]


    for _, row in subset.iterrows():

        print(
            f"  {row['candidate_id']}: "
            f"C24={row['cycle24_holdout_point']:.6f}, "
            f"C25={row['cycle25_point']:.6f}, "
            f"delta={row['delta_cycle25_minus_cycle24']:+.6f}, "
            f"95% CI=["
            f"{row['delta_ci_lower_2p5']:.6f}, "
            f"{row['delta_ci_upper_97p5']:.6f}], "
            f"{row['predictive_change_classification']}"
        )


# ============================================================
# SAVE
# ============================================================

OUTPUT_DELTA_REPLICATES_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_DRIFT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


delta_replicates_df.to_parquet(
    OUTPUT_DELTA_REPLICATES_FILE,
    index=False,
)


summary_df.to_csv(
    OUTPUT_DRIFT_FILE,
    index=False,
)


delta_replicates_hash = sha256_file(
    OUTPUT_DELTA_REPLICATES_FILE
)


summary_hash = sha256_file(
    OUTPUT_DRIFT_FILE
)


# ============================================================
# ROUND-TRIP
# ============================================================

delta_roundtrip = pd.read_parquet(
    OUTPUT_DELTA_REPLICATES_FILE
)


summary_roundtrip = pd.read_csv(
    OUTPUT_DRIFT_FILE,
    keep_default_na=False,
)


if len(
    delta_roundtrip
) != EXPECTED_DELTA_REPLICATE_ROWS:

    raise RuntimeError(
        "Delta-bootstrap parquet "
        "round-trip mismatch."
    )


if len(
    summary_roundtrip
) != EXPECTED_SUMMARY_ROWS:

    raise RuntimeError(
        "Predictive-drift CSV "
        "round-trip mismatch."
    )


print(
    "\nDelta-bootstrap parquet round-trip:",
    "PASS"
)

print(
    "Predictive-drift summary round-trip:",
    "PASS"
)


# ============================================================
# CONTRACT
# ============================================================

tss_contract_results = []


for _, row in tss_summary.iterrows():

    tss_contract_results.append(
        {
            "scope_id":
                str(
                    row[
                        "scope_id"
                    ]
                ),

            "scope_type":
                str(
                    row[
                        "scope_type"
                    ]
                ),

            "family":
                str(
                    row[
                        "family"
                    ]
                ),

            "candidate_id":
                str(
                    row[
                        "candidate_id"
                    ]
                ),

            "cycle24_holdout_tss":
                float(
                    row[
                        "cycle24_holdout_point"
                    ]
                ),

            "cycle25_tss":
                float(
                    row[
                        "cycle25_point"
                    ]
                ),

            "delta_tss":
                float(
                    row[
                        "delta_cycle25_minus_cycle24"
                    ]
                ),

            "delta_ci95_lower":
                float(
                    row[
                        "delta_ci_lower_2p5"
                    ]
                ),

            "delta_ci95_upper":
                float(
                    row[
                        "delta_ci_upper_97p5"
                    ]
                ),

            "classification":
                str(
                    row[
                        "predictive_change_classification"
                    ]
                ),
        }
    )


contract = {

    "contract_name":
        "Frozen Cycle24-to-Cycle25 Predictive Drift Comparison",

    "version":
        "1.0",

    "status":
        "FROZEN_CROSS_CYCLE_PREDICTIVE_DRIFT",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_8_5",

    "reference_domain":
        "CYCLE24_HOLDOUT",

    "transfer_domain":
        "CYCLE25",

    "delta_definition":
        "Cycle25 - Cycle24 holdout",

    "primary_metric":
        "TSS",

    "comparison_scopes":
        CYCLE25_SCOPES,

    "primary_temporal_blocks":
        PRIMARY_BLOCKS,

    "secondary_pooled_scope":
        POOLED_SCOPE,

    "metrics":
        METRICS,

    "difference_bootstrap":
        {
            "cycle24_unit":
                "ENTITY_GROUP",

            "cycle25_unit":
                "ENTITY_GROUP",

            "domains_independent":
                True,

            "replicate_combination":
                (
                    "Match independently generated "
                    "Cycle24 and Cycle25 bootstrap "
                    "replicates by replicate_id and "
                    "compute Cycle25 minus Cycle24."
                ),

            "valid_replicates":
                VALID_REPLICATES,

            "confidence_level":
                0.95,

            "ci_method":
                "percentile",

            "lower_quantile":
                LOWER_QUANTILE,

            "upper_quantile":
                UPPER_QUANTILE,

            "numpy_quantile_method":
                "linear",
        },

    "tss_classification_rule":
        {
            "MEASURABLE_DECREASE":
                (
                    "95% bootstrap delta-TSS CI "
                    "lies entirely below zero."
                ),

            "MEASURABLE_INCREASE":
                (
                    "95% bootstrap delta-TSS CI "
                    "lies entirely above zero."
                ),

            "NO_MEASURABLE_CHANGE_AT_95CI":
                (
                    "95% bootstrap delta-TSS CI "
                    "contains zero."
                ),

            "equivalence_rule":
                (
                    "A CI containing zero is not "
                    "interpreted as statistical "
                    "equivalence or proof of stability."
                ),
        },

    "tss_results":
        tss_contract_results,

    "cross_family_policy":
        {
            "overall_winner_declared":
                False,

            "cross_family_superiority_test":
                False,

            "comparison_is_same_model_across_domains":
                True,
        },

    "artifacts":
        {
            "delta_bootstrap_replicates":
                {
                    "file":
                        str(
                            OUTPUT_DELTA_REPLICATES_FILE
                        ),

                    "sha256":
                        delta_replicates_hash,

                    "rows":
                        len(
                            delta_replicates_df
                        ),
                },

            "predictive_drift_summary":
                {
                    "file":
                        str(
                            OUTPUT_DRIFT_FILE
                        ),

                    "sha256":
                        summary_hash,

                    "rows":
                        len(
                            summary_df
                        ),
                },
        },

    "evaluation_state":
        {
            "cross_cycle_point_deltas_computed":
                True,

            "cross_cycle_bootstrap_delta_ci_computed":
                True,

            "primary_tss_change_classified":
                True,

            "models_retrained":
                False,

            "thresholds_reoptimized":
                False,

            "preprocessing_refit":
                False,

            "feature_changes":
                False,

            "shap_analysis_performed":
                False,

            "distribution_shift_analysis_performed":
                False,
        },

    "data_access":
        {
            "raw_cycle24_holdout_read":
                False,

            "raw_cycle25_dataset_read":
                False,

            "cycle24_bootstrap_artifact_read":
                True,

            "cycle25_bootstrap_artifact_read":
                True,

            "cycle24_point_metrics_read":
                True,

            "cycle25_point_metrics_read":
                True,
        },

    "provenance":
        {
            "cycle24_metrics":
                {
                    "file":
                        str(
                            CYCLE24_METRICS_FILE
                        ),

                    "sha256":
                        c24_metrics_hash,
                },

            "cycle24_bootstrap_replicates":
                {
                    "file":
                        str(
                            CYCLE24_BOOTSTRAP_FILE
                        ),

                    "sha256":
                        c24_bootstrap_hash,
                },

            "cycle25_metrics":
                {
                    "file":
                        str(
                            CYCLE25_METRICS_FILE
                        ),

                    "sha256":
                        c25_metrics_hash,
                },

            "cycle25_bootstrap_replicates":
                {
                    "file":
                        str(
                            CYCLE25_BOOTSTRAP_FILE
                        ),

                    "sha256":
                        c25_bootstrap_hash,
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
        },

    "interpretation_constraints":
        {
            "zero_containing_delta_ci_is_not_equivalence":
                True,

            "marginal_ci_overlap_not_used_for_drift_classification":
                True,

            "cycle25_point_results_do_not_trigger_model_changes":
                True,

            "predictive_drift_does_not_imply_explanation_drift":
                True,

            "cycle25_is_temporal_domain_shift":
                True,

            "cycle25_not_automatically_called_ood":
                True,

            "shap_not_causal_solar_physics":
                True,
        },

    "immutability_rule":
        (
            "These cross-cycle predictive comparisons "
            "are final outputs of the frozen predictive "
            "pipeline. Their results must not be used "
            "to retrain models, change features, modify "
            "preprocessing, reoptimize thresholds, or "
            "alter Cycle25 temporal blocks."
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
    "PREDICTIVE-DRIFT SUMMARY"
)

print(
    "===================================="
)


for scope_id in CYCLE25_SCOPES:

    subset = tss_summary[
        tss_summary[
            "scope_id"
        ]
        ==
        scope_id
    ]


    decrease = int(
        (
            subset[
                "predictive_change_classification"
            ]
            ==
            "MEASURABLE_DECREASE"
        ).sum()
    )


    increase = int(
        (
            subset[
                "predictive_change_classification"
            ]
            ==
            "MEASURABLE_INCREASE"
        ).sum()
    )


    no_change = int(
        (
            subset[
                "predictive_change_classification"
            ]
            ==
            "NO_MEASURABLE_CHANGE_AT_95CI"
        ).sum()
    )


    print(
        f"{scope_id}: "
        f"decrease={decrease}, "
        f"increase={increase}, "
        f"no_measurable_change={no_change}"
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
    "Cross-cycle point deltas:",
    True
)

print(
    "Cross-cycle delta CIs:",
    True
)

print(
    "TSS predictive-change classification:",
    True
)

print(
    "Cross-family winner declared:",
    False
)

print(
    "Model retraining:",
    False
)

print(
    "Threshold reoptimization:",
    False
)

print(
    "SHAP drift evaluated:",
    False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_DELTA_REPLICATES_FILE
)

print(
    OUTPUT_DRIFT_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 8.5 COMPLETE — CROSS-CYCLE PREDICTIVE DRIFT FROZEN"
)

print(
    "===================================="
)