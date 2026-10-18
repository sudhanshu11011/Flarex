from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SELECTION_PROTOCOL_FILE = Path(
    "artifacts/models/selection/contracts/"
    "model_selection_protocol.json"
)

METRICS_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "validation_metrics_contract.json"
)

CANDIDATE_METRICS_FILE = Path(
    "artifacts/models/selection/manifests/"
    "validation_candidate_metrics.csv"
)

THRESHOLD_SWEEP_FILE = Path(
    "artifacts/models/selection/metrics/"
    "validation_threshold_sweep.parquet"
)

CANDIDATE_REGISTRY_FILE = Path(
    "artifacts/models/selection/manifests/"
    "frozen_candidate_registry.csv"
)

WINNERS_FILE = Path(
    "artifacts/models/selection/manifests/"
    "selected_family_winners.csv"
)

SELECTION_RESULT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "family_selection_result.json"
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

AUDIT_FILE = Path(
    "artifacts/models/selection/manifests/"
    "family_selection_audit.csv"
)

AUDIT_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "family_selection_audit.json"
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


def exact_float_match(
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
    "MODEL CHAPTER 6.5 — INDEPENDENT FAMILY SELECTION AUDIT"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    SELECTION_PROTOCOL_FILE,
    METRICS_CONTRACT_FILE,
    CANDIDATE_METRICS_FILE,
    THRESHOLD_SWEEP_FILE,
    CANDIDATE_REGISTRY_FILE,
    WINNERS_FILE,
    SELECTION_RESULT_FILE,
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
    AUDIT_FILE,
    AUDIT_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Audit output already exists: {output_path}\n"
            "Refusing to overwrite independent audit."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

selection_protocol = load_json(
    SELECTION_PROTOCOL_FILE
)

metrics_contract = load_json(
    METRICS_CONTRACT_FILE
)

selection_result = load_json(
    SELECTION_RESULT_FILE
)


if selection_protocol.get(
    "status"
) != "FROZEN_BEFORE_VALIDATION":

    raise RuntimeError(
        "Selection protocol is not frozen."
    )


if metrics_contract.get(
    "status"
) != "FROZEN_CANDIDATE_VALIDATION_METRICS":

    raise RuntimeError(
        "Validation metrics contract is not frozen."
    )


if selection_result.get(
    "status"
) != "SELECTED_PENDING_INDEPENDENT_AUDIT":

    raise RuntimeError(
        "Family selection result is not "
        "pending independent audit."
    )


print(
    "Selection protocol: VERIFIED"
)

print(
    "Validation metrics contract: VERIFIED"
)

print(
    "Selection result state: PENDING AUDIT"
)


# ============================================================
# VERIFY FROZEN SELECTION RULES
# ============================================================

rules = selection_protocol[
    "family_selection_rules"
]


expected_lr_order = [
    "higher_validation_TSS",
    "higher_validation_PR_AUC",
    "smaller_C",
    "class_weight_None_before_balanced",
    "lexicographically_smaller_candidate_id",
]


expected_rf_order = [
    "higher_validation_TSS",
    "higher_validation_PR_AUC",
    "smaller_max_depth",
    "larger_min_samples_leaf",
    "class_weight_None_before_balanced_subsample",
    "lexicographically_smaller_candidate_id",
]


expected_xgb_order = [
    "higher_validation_TSS",
    "higher_validation_PR_AUC",
    "smaller_max_depth",
    "smaller_learning_rate",
    "scale_pos_weight_1_before_balanced",
    "lexicographically_smaller_candidate_id",
]


if (
    rules[
        "logistic_regression"
    ][
        "winner_order"
    ]
    != expected_lr_order
):

    raise RuntimeError(
        "Logistic selection hierarchy differs "
        "from expected frozen protocol."
    )


if (
    rules[
        "random_forest"
    ][
        "winner_order"
    ]
    != expected_rf_order
):

    raise RuntimeError(
        "Random Forest selection hierarchy differs "
        "from expected frozen protocol."
    )


if (
    rules[
        "xgboost"
    ][
        "winner_order"
    ]
    != expected_xgb_order
):

    raise RuntimeError(
        "XGBoost selection hierarchy differs "
        "from expected frozen protocol."
    )


if (
    rules[
        "random_forest"
    ][
        "max_depth_complexity_order"
    ]
    != [
        12,
        24,
        "None",
    ]
):

    raise RuntimeError(
        "Random Forest max-depth complexity "
        "ordering differs from frozen protocol."
    )


print(
    "Frozen family-selection hierarchies: PASS"
)


# ============================================================
# VERIFY INPUT ARTIFACT HASHES
# ============================================================

metrics_contract_expected_hash = (
    metrics_contract[
        "artifacts"
    ][
        "candidate_metrics"
    ][
        "sha256"
    ]
)


if (
    sha256_file(
        CANDIDATE_METRICS_FILE
    )
    !=
    metrics_contract_expected_hash
):

    raise RuntimeError(
        "Candidate metrics hash mismatch."
    )


threshold_sweep_expected_hash = (
    metrics_contract[
        "artifacts"
    ][
        "threshold_sweep"
    ][
        "sha256"
    ]
)


if (
    sha256_file(
        THRESHOLD_SWEEP_FILE
    )
    !=
    threshold_sweep_expected_hash
):

    raise RuntimeError(
        "Threshold sweep hash mismatch."
    )


if (
    sha256_file(
        WINNERS_FILE
    )
    !=
    selection_result[
        "winner_manifest"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Selected winner manifest hash mismatch."
    )


print(
    "Frozen artifact hashes: PASS"
)


# ============================================================
# LOAD TABLES
# ============================================================

metrics = pd.read_csv(
    CANDIDATE_METRICS_FILE,
    keep_default_na=False,
)

threshold_sweep = pd.read_parquet(
    THRESHOLD_SWEEP_FILE
)

registry = pd.read_csv(
    CANDIDATE_REGISTRY_FILE,
    keep_default_na=False,
)

selected_winners = pd.read_csv(
    WINNERS_FILE,
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
# GLOBAL INTEGRITY
# ============================================================

if len(metrics) != 34:

    raise RuntimeError(
        "Expected 34 candidate metric rows."
    )


if metrics[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate candidates in metric table."
    )


if len(registry) != 34:

    raise RuntimeError(
        "Expected 34 frozen candidates."
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
        "Metrics candidate set differs from registry."
    )


if len(selected_winners) != 3:

    raise RuntimeError(
        "Expected exactly 3 selected winners."
    )


if selected_winners[
    "family"
].nunique() != 3:

    raise RuntimeError(
        "Expected one winner per model family."
    )


# ============================================================
# VERIFY ONE SELECTED THRESHOLD PER CANDIDATE
# ============================================================

selected_threshold_rows = threshold_sweep[
    threshold_sweep[
        "selected_by_max_tss"
    ].astype(bool)
].copy()


if len(
    selected_threshold_rows
) != 34:

    raise RuntimeError(
        "Expected exactly 34 selected threshold rows."
    )


if selected_threshold_rows[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "More than one selected threshold exists "
        "for a candidate."
    )


threshold_lookup = selected_threshold_rows.set_index(
    "candidate_id"
)


for _, row in metrics.iterrows():

    candidate_id = str(
        row[
            "candidate_id"
        ]
    )


    if candidate_id not in threshold_lookup.index:

        raise RuntimeError(
            f"{candidate_id}: selected threshold "
            "missing from sweep."
        )


    sweep_row = threshold_lookup.loc[
        candidate_id
    ]


    if not exact_float_match(
        row[
            "selected_threshold"
        ],
        sweep_row[
            "threshold"
        ],
    ):

        raise RuntimeError(
            f"{candidate_id}: selected threshold "
            "differs between metrics and sweep."
        )


    if int(
        row[
            "tss_exact_numerator"
        ]
    ) != int(
        sweep_row[
            "tss_numerator"
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: exact TSS numerator mismatch."
        )


print(
    "34 candidate thresholds independently linked: PASS"
)


# ============================================================
# JOIN MODEL REFERENCES
# ============================================================

registry_lookup = registry[
    [
        "family",
        "candidate_id",
        "model_file",
        "model_sha256",
    ]
].copy()


metrics = metrics.merge(
    registry_lookup,
    on=[
        "family",
        "candidate_id",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# INDEPENDENT FAMILY RANKING
# ============================================================

def independently_rank_logistic(
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
            "Independent LR audit expected "
            "10 candidates."
        )


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
            "Unexpected Logistic class weight."
        )


    return data.sort_values(
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
    ).reset_index(
        drop=True
    )


def independently_rank_rf(
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
            "Independent RF audit expected "
            "12 candidates."
        )


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
            "",
            "nan",
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
            "Unexpected RF class weight."
        )


    return data.sort_values(
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
    ).reset_index(
        drop=True
    )


def independently_rank_xgb(
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
            "Independent XGBoost audit expected "
            "12 candidates."
        )


    numeric_spw = pd.to_numeric(
        data[
            "scale_pos_weight"
        ],
        errors="raise",
    )


    balanced_spw = float(
        numeric_spw.max()
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
            f"Unexpected XGBoost SPW: {value}"
        )


    data[
        "_spw_rank"
    ] = data[
        "scale_pos_weight"
    ].apply(
        spw_rank
    )


    return data.sort_values(
        by=[
            "tss_exact_numerator",
            "validation_pr_auc",
            "max_depth",
            "learning_rate",
            "_spw_rank",
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
    ).reset_index(
        drop=True
    )


# ============================================================
# AUDIT EACH FAMILY
# ============================================================

audit_rows = []


print(
    "\n===================================="
)

print(
    "INDEPENDENT FAMILY AUDIT"
)

print(
    "===================================="
)


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

        ranked = independently_rank_logistic(
            family_metrics
        )


    elif family == "random_forest":

        ranked = independently_rank_rf(
            family_metrics
        )


    elif family == "xgboost":

        ranked = independently_rank_xgb(
            family_metrics
        )


    else:

        raise RuntimeError(
            f"Unknown family: {family}"
        )


    expected_winner = ranked.iloc[
        0
    ]


    official = selected_winners[
        selected_winners[
            "family"
        ]
        ==
        family
    ]


    if len(
        official
    ) != 1:

        raise RuntimeError(
            f"{family}: expected one official winner."
        )


    official = official.iloc[
        0
    ]


    expected_candidate_id = str(
        expected_winner[
            "candidate_id"
        ]
    )


    official_candidate_id = str(
        official[
            "candidate_id"
        ]
    )


    candidate_match = (
        expected_candidate_id
        ==
        official_candidate_id
    )


    threshold_match = exact_float_match(
        expected_winner[
            "selected_threshold"
        ],
        official[
            "selected_threshold"
        ],
    )


    tss_match = exact_float_match(
        expected_winner[
            "validation_tss"
        ],
        official[
            "validation_tss"
        ],
        atol=1e-15,
    )


    pr_auc_match = exact_float_match(
        expected_winner[
            "validation_pr_auc"
        ],
        official[
            "validation_pr_auc"
        ],
        atol=1e-15,
    )


    model_hash_match = (
        str(
            expected_winner[
                "model_sha256"
            ]
        )
        ==
        str(
            official[
                "model_sha256"
            ]
        )
    )


    # --------------------------------------------------------
    # Determine whether any TSS tie existed at first level
    # --------------------------------------------------------

    best_tss_numerator = int(
        ranked.iloc[
            0
        ][
            "tss_exact_numerator"
        ]
    )


    primary_tie_count = int(
        (
            ranked[
                "tss_exact_numerator"
            ]
            ==
            best_tss_numerator
        ).sum()
    )


    if primary_tie_count == 1:

        selection_decided_by = (
            "PRIMARY_TSS"
        )

    else:

        tied = ranked[
            ranked[
                "tss_exact_numerator"
            ]
            ==
            best_tss_numerator
        ]


        best_pr_auc = float(
            tied[
                "validation_pr_auc"
            ].max()
        )


        secondary_tie_count = int(
            np.isclose(
                tied[
                    "validation_pr_auc"
                ].to_numpy(
                    dtype=np.float64
                ),
                best_pr_auc,
                rtol=0.0,
                atol=0.0,
            ).sum()
        )


        if secondary_tie_count == 1:

            selection_decided_by = (
                "SECONDARY_PR_AUC"
            )

        else:

            selection_decided_by = (
                "PRE_REGISTERED_COMPLEXITY_TIE_BREAK"
            )


    family_audit_pass = bool(

        candidate_match

        and
        threshold_match

        and
        tss_match

        and
        pr_auc_match

        and
        model_hash_match
    )


    audit_rows.append(
        {
            "family":
                family,

            "independently_expected_winner":
                expected_candidate_id,

            "official_selected_winner":
                official_candidate_id,

            "candidate_match":
                candidate_match,

            "expected_threshold":
                float(
                    expected_winner[
                        "selected_threshold"
                    ]
                ),

            "official_threshold":
                float(
                    official[
                        "selected_threshold"
                    ]
                ),

            "threshold_match":
                threshold_match,

            "expected_validation_tss":
                float(
                    expected_winner[
                        "validation_tss"
                    ]
                ),

            "official_validation_tss":
                float(
                    official[
                        "validation_tss"
                    ]
                ),

            "tss_match":
                tss_match,

            "expected_validation_pr_auc":
                float(
                    expected_winner[
                        "validation_pr_auc"
                    ]
                ),

            "official_validation_pr_auc":
                float(
                    official[
                        "validation_pr_auc"
                    ]
                ),

            "pr_auc_match":
                pr_auc_match,

            "model_sha256_match":
                model_hash_match,

            "primary_tss_tie_count":
                primary_tie_count,

            "selection_decided_by":
                selection_decided_by,

            "family_selection_audit_pass":
                family_audit_pass,
        }
    )


    print(
        f"{family}: "
        f"expected={expected_candidate_id}, "
        f"official={official_candidate_id}, "
        f"TSS={float(expected_winner['validation_tss']):.6f}, "
        f"primary_ties={primary_tie_count}, "
        f"decided_by={selection_decided_by}, "
        f"PASS={family_audit_pass}"
    )


# ============================================================
# BUILD AUDIT TABLE
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


if len(
    audit_df
) != 3:

    raise RuntimeError(
        "Expected 3 family audit rows."
    )


passed_families = int(
    audit_df[
        "family_selection_audit_pass"
    ].sum()
)


overall_pass = bool(
    passed_families
    ==
    3
)


# ============================================================
# VERIFY CONTRACT WINNER LIST TOO
# ============================================================

contract_winners = {
    item[
        "family"
    ]:
    item[
        "candidate_id"
    ]

    for item in selection_result[
        "winners"
    ]
}


manifest_winners = {
    str(
        row[
            "family"
        ]
    ):
    str(
        row[
            "candidate_id"
        ]
    )

    for _, row
    in selected_winners.iterrows()
}


contract_manifest_match = (
    contract_winners
    ==
    manifest_winners
)


if not contract_manifest_match:

    raise RuntimeError(
        "Selection result contract and winner "
        "manifest disagree."
    )


# ============================================================
# SAVE AUDIT MANIFEST
# ============================================================

AUDIT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


AUDIT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


audit_df.to_csv(
    AUDIT_FILE,
    index=False,
)


audit_hash = sha256_file(
    AUDIT_FILE
)


# ============================================================
# BUILD AUDIT CONTRACT
# ============================================================

audit_contract = {

    "contract_name":
        "Independent Cycle24 Validation Family Selection Audit",

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

    "families_audited":
        3,

    "families_passed":
        passed_families,

    "selection_protocol_verified":
        True,

    "candidate_threshold_links_verified":
        34,

    "winner_contract_manifest_match":
        contract_manifest_match,

    "overall_selection_audit_pass":
        overall_pass,

    "audit_scope":
        {
            "independent_family_ranking":
                True,

            "frozen_threshold_verification":
                True,

            "winner_model_hash_verification":
                True,

            "model_retraining":
                False,

            "threshold_reoptimization":
                False,

            "raw_validation_dataset_read":
                False,

            "holdout_read":
                False,

            "cycle25_read":
                False,
        },

    "family_results":
        [
            {
                "family":
                    row[
                        "family"
                    ],

                "winner":
                    row[
                        "official_selected_winner"
                    ],

                "selection_decided_by":
                    row[
                        "selection_decided_by"
                    ],

                "primary_tss_tie_count":
                    int(
                        row[
                            "primary_tss_tie_count"
                        ]
                    ),

                "audit_pass":
                    bool(
                        row[
                            "family_selection_audit_pass"
                        ]
                    ),
            }

            for _, row
            in audit_df.iterrows()
        ],

    "audit_manifest":
        {
            "file":
                str(
                    AUDIT_FILE
                ),

            "sha256":
                audit_hash,
        },

    "provenance":
        {
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

            "candidate_metrics":
                {
                    "file":
                        str(
                            CANDIDATE_METRICS_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CANDIDATE_METRICS_FILE
                        ),
                },

            "threshold_sweep":
                {
                    "file":
                        str(
                            THRESHOLD_SWEEP_FILE
                        ),

                    "sha256":
                        sha256_file(
                            THRESHOLD_SWEEP_FILE
                        ),
                },

            "selected_winners":
                {
                    "file":
                        str(
                            WINNERS_FILE
                        ),

                    "sha256":
                        sha256_file(
                            WINNERS_FILE
                        ),
                },

            "selection_result":
                {
                    "file":
                        str(
                            SELECTION_RESULT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            SELECTION_RESULT_FILE
                        ),
                },
        },

    "interpretation":
        (
            "This audit independently reapplies the "
            "pre-registered within-family ranking hierarchy "
            "to the frozen validation metrics. It does not "
            "introduce new model fitting, threshold tuning, "
            "candidate definitions, or performance criteria."
        ),
}


with AUDIT_CONTRACT_FILE.open(
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
    "Families audited:",
    3
)

print(
    "Families passing:",
    passed_families,
    "/ 3"
)

print(
    "Candidate threshold links verified:",
    34,
    "/ 34"
)

print(
    "Winner contract/manifest agreement:",
    contract_manifest_match
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
    "Raw validation dataset read:",
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

print(
    "Models retrained:",
    False
)

print(
    "Thresholds reoptimized:",
    False
)


print(
    "\n===================================="
)

print(
    "CHAPTER 6.5 RESULT"
)

print(
    "===================================="
)


print(
    "Independent family selection audit:",
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
    AUDIT_FILE
)

print(
    AUDIT_CONTRACT_FILE
)


if not overall_pass:

    raise RuntimeError(
        "Chapter 6.5 independent "
        "selection audit failed."
    )


print(
    "\nMODEL CHAPTER 6.5 COMPLETE."
)