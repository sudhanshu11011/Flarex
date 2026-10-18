from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

METRICS_FILE = Path(
    "artifacts/models/selection/manifests/"
    "validation_candidate_metrics.csv"
)

METRICS_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "validation_metrics_contract.json"
)

SELECTION_PROTOCOL_FILE = Path(
    "artifacts/models/selection/contracts/"
    "model_selection_protocol.json"
)

CANDIDATE_REGISTRY_FILE = Path(
    "artifacts/models/selection/manifests/"
    "frozen_candidate_registry.csv"
)

LR_GRID_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_candidate_grid.csv"
)

RF_GRID_FILE = Path(
    "artifacts/models/random_forest/manifests/"
    "random_forest_candidate_grid.csv"
)

XGB_GRID_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_candidate_grid.csv"
)

OUTPUT_FILE = Path(
    "artifacts/models/selection/manifests/"
    "selected_family_winners.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "family_selection_result.json"
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


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 6.4 — SELECT FAMILY WINNERS"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    METRICS_FILE,
    METRICS_CONTRACT_FILE,
    SELECTION_PROTOCOL_FILE,
    CANDIDATE_REGISTRY_FILE,
    LR_GRID_FILE,
    RF_GRID_FILE,
    XGB_GRID_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


for output_path in [
    OUTPUT_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Output already exists: {output_path}\n"
            "Refusing to overwrite selection result."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

metrics_contract = load_json(
    METRICS_CONTRACT_FILE
)

selection_protocol = load_json(
    SELECTION_PROTOCOL_FILE
)


if metrics_contract.get(
    "status"
) != "FROZEN_CANDIDATE_VALIDATION_METRICS":

    raise RuntimeError(
        "Validation metrics contract is not frozen."
    )


if selection_protocol.get(
    "status"
) != "FROZEN_BEFORE_VALIDATION":

    raise RuntimeError(
        "Selection protocol is not frozen."
    )


# ============================================================
# LOAD TABLES
# ============================================================

metrics = pd.read_csv(
    METRICS_FILE,
    keep_default_na=False,
)

registry = pd.read_csv(
    CANDIDATE_REGISTRY_FILE,
    keep_default_na=False,
)

lr_grid = pd.read_csv(
    LR_GRID_FILE,
    keep_default_na=False,
)

rf_grid = pd.read_csv(
    RF_GRID_FILE,
    keep_default_na=False,
)

xgb_grid = pd.read_csv(
    XGB_GRID_FILE,
    keep_default_na=False,
)


# ============================================================
# BASIC INTEGRITY
# ============================================================

if len(metrics) != 34:

    raise RuntimeError(
        "Expected 34 candidate metric rows."
    )


if metrics[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidate IDs in metric table."
    )


if len(registry) != 34:

    raise RuntimeError(
        "Expected 34 frozen registry rows."
    )


if set(
    metrics[
        "candidate_id"
    ].astype(str)
) != set(
    registry[
        "candidate_id"
    ].astype(str)
):

    raise RuntimeError(
        "Metric candidate set differs "
        "from frozen registry."
    )


if not metrics[
    "threshold_selected"
].astype(str).str.lower().eq(
    "true"
).all():

    raise RuntimeError(
        "Not all candidates have frozen thresholds."
    )


if metrics[
    "family_winner_selected"
].astype(str).str.lower().eq(
    "true"
).any():

    raise RuntimeError(
        "Premature family selection detected."
    )


print(
    "Frozen candidate metrics: VERIFIED"
)

print(
    "Candidates available: 34"
)


# ============================================================
# JOIN MODEL ARTIFACT INFORMATION
# ============================================================

registry_subset = registry[
    [
        "family",
        "candidate_id",
        "model_file",
        "model_sha256",
    ]
].copy()


metrics = metrics.merge(
    registry_subset,
    on=[
        "family",
        "candidate_id",
    ],
    how="left",
    validate="one_to_one",
)


if metrics[
    "model_file"
].eq("").any():

    raise RuntimeError(
        "Missing frozen model-file reference."
    )


# ============================================================
# FAMILY SELECTION FUNCTIONS
# ============================================================

def select_logistic(
    family_metrics,
):

    data = family_metrics.merge(
        lr_grid,
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_grid",
        ),
    )


    if len(data) != 10:

        raise RuntimeError(
            "Expected 10 Logistic candidates."
        )


    # Frozen hierarchy:
    #
    # 1. higher TSS
    # 2. higher PR-AUC
    # 3. smaller C
    # 4. None class_weight before balanced
    # 5. lexicographically smaller ID

    data[
        "_class_weight_rank"
    ] = data[
        "class_weight"
    ].astype(str).map(
        {
            "None": 0,
            "balanced": 1,
        }
    )


    if data[
        "_class_weight_rank"
    ].isna().any():

        raise RuntimeError(
            "Unexpected Logistic class_weight."
        )


    ranked = data.sort_values(
        by=[
            "tss_exact_numerator",
            "validation_pr_auc",
            "C",
            "_class_weight_rank",
            "candidate_id",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            True,
        ],
        kind="mergesort",
    )


    return ranked.iloc[
        0
    ]


def select_random_forest(
    family_metrics,
):

    data = family_metrics.merge(
        rf_grid,
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_grid",
        ),
    )


    if len(data) != 12:

        raise RuntimeError(
            "Expected 12 Random Forest candidates."
        )


    # Frozen depth complexity order:
    # 12 < 24 < None/unbounded

    def depth_rank(value):

        text = str(
            value
        ).strip()

        if text == "12":
            return 0

        if text == "24":
            return 1

        if text.lower() in {
            "none",
            "nan",
            "",
        }:
            return 2

        raise RuntimeError(
            f"Unexpected RF max_depth: {value}"
        )


    data[
        "_max_depth_rank"
    ] = data[
        "max_depth"
    ].apply(
        depth_rank
    )


    data[
        "_class_weight_rank"
    ] = data[
        "class_weight"
    ].astype(str).map(
        {
            "None": 0,
            "balanced_subsample": 1,
        }
    )


    if data[
        "_class_weight_rank"
    ].isna().any():

        raise RuntimeError(
            "Unexpected RF class_weight."
        )


    # Frozen hierarchy:
    #
    # 1. higher TSS
    # 2. higher PR-AUC
    # 3. smaller max_depth
    # 4. larger min_samples_leaf
    # 5. None before balanced_subsample
    # 6. lexical ID

    ranked = data.sort_values(
        by=[
            "tss_exact_numerator",
            "validation_pr_auc",
            "_max_depth_rank",
            "min_samples_leaf",
            "_class_weight_rank",
            "candidate_id",
        ],
        ascending=[
            False,
            False,
            True,
            False,
            True,
            True,
        ],
        kind="mergesort",
    )


    return ranked.iloc[
        0
    ]


def select_xgboost(
    family_metrics,
):

    data = family_metrics.merge(
        xgb_grid,
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_grid",
        ),
    )


    if len(data) != 12:

        raise RuntimeError(
            "Expected 12 XGBoost candidates."
        )


    balanced_spw = max(
        pd.to_numeric(
            data[
                "scale_pos_weight"
            ],
            errors="raise",
        )
    )


    def spw_rank(value):

        numeric = float(
            value
        )

        if np.isclose(
            numeric,
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            return 0

        if np.isclose(
            numeric,
            balanced_spw,
            rtol=0.0,
            atol=1e-12,
        ):
            return 1

        raise RuntimeError(
            f"Unexpected XGBoost scale_pos_weight: "
            f"{value}"
        )


    data[
        "_scale_pos_weight_rank"
    ] = data[
        "scale_pos_weight"
    ].apply(
        spw_rank
    )


    # Frozen hierarchy:
    #
    # 1. higher TSS
    # 2. higher PR-AUC
    # 3. smaller max_depth
    # 4. smaller learning_rate
    # 5. SPW 1 before balanced
    # 6. lexical ID

    ranked = data.sort_values(
        by=[
            "tss_exact_numerator",
            "validation_pr_auc",
            "max_depth",
            "learning_rate",
            "_scale_pos_weight_rank",
            "candidate_id",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            True,
            True,
        ],
        kind="mergesort",
    )


    return ranked.iloc[
        0
    ]


# ============================================================
# SELECT EXACTLY ONE WINNER PER FAMILY
# ============================================================

winner_rows = []


for family in [
    "logistic",
    "random_forest",
    "xgboost",
]:

    family_metrics = metrics[
        metrics[
            "family"
        ]
        ==
        family
    ].copy()


    if family == "logistic":

        winner = select_logistic(
            family_metrics
        )


    elif family == "random_forest":

        winner = select_random_forest(
            family_metrics
        )


    elif family == "xgboost":

        winner = select_xgboost(
            family_metrics
        )


    else:

        raise RuntimeError(
            f"Unknown family: {family}"
        )


    winner_rows.append(
        {
            "family":
                family,

            "candidate_id":
                str(
                    winner[
                        "candidate_id"
                    ]
                ),

            "selected_threshold":
                float(
                    winner[
                        "selected_threshold"
                    ]
                ),

            "validation_tss":
                float(
                    winner[
                        "validation_tss"
                    ]
                ),

            "validation_pr_auc":
                float(
                    winner[
                        "validation_pr_auc"
                    ]
                ),

            "validation_roc_auc":
                float(
                    winner[
                        "validation_roc_auc"
                    ]
                ),

            "validation_brier_score":
                float(
                    winner[
                        "validation_brier_score"
                    ]
                ),

            "validation_hss":
                float(
                    winner[
                        "validation_hss"
                    ]
                ),

            "validation_f1":
                float(
                    winner[
                        "validation_f1"
                    ]
                ),

            "validation_sensitivity":
                float(
                    winner[
                        "validation_sensitivity"
                    ]
                ),

            "validation_specificity":
                float(
                    winner[
                        "validation_specificity"
                    ]
                ),

            "validation_precision":
                float(
                    winner[
                        "validation_precision"
                    ]
                ),

            "tp":
                int(
                    winner[
                        "tp"
                    ]
                ),

            "fp":
                int(
                    winner[
                        "fp"
                    ]
                ),

            "tn":
                int(
                    winner[
                        "tn"
                    ]
                ),

            "fn":
                int(
                    winner[
                        "fn"
                    ]
                ),

            "tss_exact_numerator":
                int(
                    winner[
                        "tss_exact_numerator"
                    ]
                ),

            "tss_exact_denominator":
                int(
                    winner[
                        "tss_exact_denominator"
                    ]
                ),

            "model_file":
                str(
                    winner[
                        "model_file"
                    ]
                ),

            "model_sha256":
                str(
                    winner[
                        "model_sha256"
                    ]
                ),

            "selection_dataset":
                "CYCLE24_VALIDATION",

            "primary_selection_metric":
                "TSS",

            "threshold_frozen":
                True,

            "model_retrained":
                False,

            "holdout_used":
                False,

            "cycle25_used":
                False,
        }
    )


# ============================================================
# BUILD WINNER TABLE
# ============================================================

winners = pd.DataFrame(
    winner_rows
)


if len(
    winners
) != 3:

    raise RuntimeError(
        "Expected exactly 3 family winners."
    )


if winners[
    "family"
].nunique() != 3:

    raise RuntimeError(
        "Expected one unique winner per family."
    )


if winners[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Same candidate selected for "
        "multiple model families."
    )


# ============================================================
# EXPECTED PRIMARY-METRIC WINNER CHECK
# ============================================================

for _, winner in winners.iterrows():

    family = winner[
        "family"
    ]


    family_metrics = metrics[
        metrics[
            "family"
        ]
        ==
        family
    ]


    maximum_tss_numerator = int(
        family_metrics[
            "tss_exact_numerator"
        ].max()
    )


    if int(
        winner[
            "tss_exact_numerator"
        ]
    ) != maximum_tss_numerator:

        raise RuntimeError(
            f"{family}: selected candidate does "
            "not have maximum validation TSS."
        )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


winners.to_csv(
    OUTPUT_FILE,
    index=False,
)


winner_file_hash = sha256_file(
    OUTPUT_FILE
)


# ============================================================
# CONTRACT
# ============================================================

selection_result = {

    "contract_name":
        "Cycle24 Validation Family Selection Result",

    "version":
        "1.0",

    "status":
        "SELECTED_PENDING_INDEPENDENT_AUDIT",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "selection_dataset":
        "CYCLE24_VALIDATION",

    "primary_metric":
        "TSS",

    "secondary_metric":
        "PR_AUC",

    "family_winner_count":
        3,

    "cross_family_winner_selected":
        False,

    "winners":
        [
            {
                "family":
                    row[
                        "family"
                    ],

                "candidate_id":
                    row[
                        "candidate_id"
                    ],

                "selected_threshold":
                    float(
                        row[
                            "selected_threshold"
                        ]
                    ),

                "validation_tss":
                    float(
                        row[
                            "validation_tss"
                        ]
                    ),

                "validation_pr_auc":
                    float(
                        row[
                            "validation_pr_auc"
                        ]
                    ),

                "model_file":
                    row[
                        "model_file"
                    ],

                "model_sha256":
                    row[
                        "model_sha256"
                    ],
            }

            for _, row
            in winners.iterrows()
        ],

    "post_selection_state":
        {
            "winner_models_retrained":
                False,

            "train_plus_validation_refit":
                False,

            "selected_thresholds_frozen":
                True,

            "holdout_used":
                False,

            "cycle25_used":
                False,

            "independent_selection_audit":
                "PENDING_CHAPTER_6_5",
        },

    "winner_manifest":
        {
            "file":
                str(
                    OUTPUT_FILE
                ),

            "sha256":
                winner_file_hash,
        },

    "provenance":
        {
            "candidate_metrics":
                {
                    "file":
                        str(
                            METRICS_FILE
                        ),

                    "sha256":
                        sha256_file(
                            METRICS_FILE
                        ),
                },

            "validation_metrics_contract":
                {
                    "file":
                        str(
                            METRICS_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            METRICS_CONTRACT_FILE
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

            "logistic_grid":
                {
                    "file":
                        str(
                            LR_GRID_FILE
                        ),

                    "sha256":
                        sha256_file(
                            LR_GRID_FILE
                        ),
                },

            "random_forest_grid":
                {
                    "file":
                        str(
                            RF_GRID_FILE
                        ),

                    "sha256":
                        sha256_file(
                            RF_GRID_FILE
                        ),
                },

            "xgboost_grid":
                {
                    "file":
                        str(
                            XGB_GRID_FILE
                        ),

                    "sha256":
                        sha256_file(
                            XGB_GRID_FILE
                        ),
                },
        },

    "immutability_rule":
        (
            "The selected candidate and validation-derived "
            "probability threshold for each model family "
            "must not be changed after this selection. "
            "Chapter 6.5 may independently audit the result "
            "but may not alter the frozen selection hierarchy. "
            "Winner models must not be retrained on validation."
        ),
}


with OUTPUT_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        selection_result,
        file,
        indent=2,
    )


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "SELECTED FAMILY WINNERS"
)

print(
    "===================================="
)


for _, row in winners.iterrows():

    print(
        f"{row['family']}: "
        f"{row['candidate_id']} | "
        f"threshold={row['selected_threshold']:.10g} | "
        f"TSS={row['validation_tss']:.6f} | "
        f"PR_AUC={row['validation_pr_auc']:.6f}"
    )


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
    "Family winners selected:",
    True
)

print(
    "Winners selected:",
    len(
        winners
    )
)

print(
    "Cross-family winner selected:",
    False
)

print(
    "Selected thresholds frozen:",
    True
)

print(
    "Models retrained:",
    False
)

print(
    "Independent selection audit:",
    "PENDING"
)


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
    "Cycle24 validation metrics used:",
    True
)

print(
    "Cycle24 HOLDOUT read:",
    False
)

print(
    "Cycle25 read:",
    False
)

print(
    "TRAIN + VALIDATION refit:",
    False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 6.4 COMPLETE — FAMILY WINNERS SELECTED"
)

print(
    "===================================="
)