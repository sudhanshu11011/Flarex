from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

WINNERS_FILE = Path(
    "artifacts/models/selection/manifests/"
    "selected_family_winners.csv"
)

SELECTION_RESULT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "family_selection_result.json"
)

SELECTION_AUDIT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "family_selection_audit.json"
)

SELECTION_PROTOCOL_FILE = Path(
    "artifacts/models/selection/contracts/"
    "model_selection_protocol.json"
)

VALIDATION_METRICS_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "validation_metrics_contract.json"
)

CANDIDATE_REGISTRY_FILE = Path(
    "artifacts/models/selection/manifests/"
    "frozen_candidate_registry.csv"
)

OUTPUT_MANIFEST_FILE = Path(
    "artifacts/models/selection/manifests/"
    "frozen_selected_models.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "selected_models_contract.json"
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
    "MODEL CHAPTER 6.6 — FREEZE SELECTED MODELS + THRESHOLDS"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    WINNERS_FILE,
    SELECTION_RESULT_FILE,
    SELECTION_AUDIT_FILE,
    SELECTION_PROTOCOL_FILE,
    VALIDATION_METRICS_CONTRACT_FILE,
    CANDIDATE_REGISTRY_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# PROTECT FROZEN OUTPUTS
# ============================================================

for output_path in [
    OUTPUT_MANIFEST_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Frozen output already exists: {output_path}\n"
            "Refusing to overwrite selected-model freeze."
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

selection_result = load_json(
    SELECTION_RESULT_FILE
)

selection_audit = load_json(
    SELECTION_AUDIT_FILE
)

selection_protocol = load_json(
    SELECTION_PROTOCOL_FILE
)

validation_metrics_contract = load_json(
    VALIDATION_METRICS_CONTRACT_FILE
)


# ============================================================
# VERIFY CONTRACT STATES
# ============================================================

if selection_result.get(
    "status"
) != "SELECTED_PENDING_INDEPENDENT_AUDIT":

    raise RuntimeError(
        "Family selection result is not in "
        "expected pre-freeze state."
    )


if selection_audit.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "Independent family selection audit "
        "did not PASS."
    )


if not selection_audit.get(
    "overall_selection_audit_pass",
    False,
):

    raise RuntimeError(
        "Independent family selection audit "
        "overall PASS flag is False."
    )


if selection_protocol.get(
    "status"
) != "FROZEN_BEFORE_VALIDATION":

    raise RuntimeError(
        "Selection protocol is not frozen."
    )


if validation_metrics_contract.get(
    "status"
) != "FROZEN_CANDIDATE_VALIDATION_METRICS":

    raise RuntimeError(
        "Validation metrics contract is not frozen."
    )


print(
    "Selection protocol: VERIFIED"
)

print(
    "Validation metrics: FROZEN"
)

print(
    "Family selection result: VERIFIED"
)

print(
    "Independent selection audit: PASS"
)


# ============================================================
# LOAD WINNERS + REGISTRY
# ============================================================

winners = pd.read_csv(
    WINNERS_FILE,
    keep_default_na=False,
)

registry = pd.read_csv(
    CANDIDATE_REGISTRY_FILE,
    keep_default_na=False,
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
        "Expected one winner per family."
    )


expected_families = {
    "logistic",
    "random_forest",
    "xgboost",
}


if set(
    winners[
        "family"
    ].astype(str)
) != expected_families:

    raise RuntimeError(
        "Selected winner families are not exactly "
        "Logistic, Random Forest, and XGBoost."
    )


if len(
    registry
) != 34:

    raise RuntimeError(
        "Frozen candidate registry must contain "
        "34 candidates."
    )


# ============================================================
# VERIFY SELECTION RESULT AGREES WITH MANIFEST
# ============================================================

result_winners = {
    item[
        "family"
    ]:
    item

    for item in selection_result[
        "winners"
    ]
}


if set(
    result_winners.keys()
) != expected_families:

    raise RuntimeError(
        "Selection-result contract does not "
        "contain exactly three families."
    )


# ============================================================
# VERIFY EACH WINNER
# ============================================================

freeze_rows = []


print(
    "\n===================================="
)

print(
    "VERIFY SELECTED FAMILY WINNERS"
)

print(
    "===================================="
)


for _, winner in winners.iterrows():

    family = str(
        winner[
            "family"
        ]
    )


    candidate_id = str(
        winner[
            "candidate_id"
        ]
    )


    threshold = float(
        winner[
            "selected_threshold"
        ]
    )


    validation_tss = float(
        winner[
            "validation_tss"
        ]
    )


    validation_pr_auc = float(
        winner[
            "validation_pr_auc"
        ]
    )


    model_file = Path(
        str(
            winner[
                "model_file"
            ]
        )
    )


    expected_model_hash = str(
        winner[
            "model_sha256"
        ]
    )


    # --------------------------------------------------------
    # Candidate must exist in frozen 34-model registry
    # --------------------------------------------------------

    registry_row = registry[
        (
            registry[
                "family"
            ].astype(str)
            ==
            family
        )
        &
        (
            registry[
                "candidate_id"
            ].astype(str)
            ==
            candidate_id
        )
    ]


    if len(
        registry_row
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: winner not uniquely present "
            "in frozen candidate registry."
        )


    registry_row = registry_row.iloc[
        0
    ]


    # --------------------------------------------------------
    # Model artifact must still exist
    # --------------------------------------------------------

    if not model_file.exists():

        raise FileNotFoundError(
            f"{candidate_id}: selected model "
            f"artifact missing: {model_file}"
        )


    actual_model_hash = sha256_file(
        model_file
    )


    if (
        actual_model_hash
        !=
        expected_model_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: selected model "
            "artifact hash changed."
        )


    if (
        str(
            registry_row[
                "model_sha256"
            ]
        )
        !=
        actual_model_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: winner artifact hash "
            "differs from frozen candidate registry."
        )


    # --------------------------------------------------------
    # Verify result contract
    # --------------------------------------------------------

    contract_entry = result_winners[
        family
    ]


    if (
        str(
            contract_entry[
                "candidate_id"
            ]
        )
        !=
        candidate_id
    ):

        raise RuntimeError(
            f"{family}: selected candidate differs "
            "from selection-result contract."
        )


    if abs(
        float(
            contract_entry[
                "selected_threshold"
            ]
        )
        -
        threshold
    ) > 1e-12:

        raise RuntimeError(
            f"{candidate_id}: threshold differs "
            "from selection-result contract."
        )


    if abs(
        float(
            contract_entry[
                "validation_tss"
            ]
        )
        -
        validation_tss
    ) > 1e-15:

        raise RuntimeError(
            f"{candidate_id}: validation TSS differs "
            "from selection-result contract."
        )


    if abs(
        float(
            contract_entry[
                "validation_pr_auc"
            ]
        )
        -
        validation_pr_auc
    ) > 1e-15:

        raise RuntimeError(
            f"{candidate_id}: validation PR-AUC differs "
            "from selection-result contract."
        )


    freeze_rows.append(
        {
            "family":
                family,

            "candidate_id":
                candidate_id,

            "model_file":
                str(
                    model_file
                ),

            "model_sha256":
                actual_model_hash,

            "selected_threshold":
                threshold,

            "validation_tss":
                validation_tss,

            "validation_pr_auc":
                validation_pr_auc,

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

            "selection_dataset":
                "CYCLE24_VALIDATION",

            "selection_metric":
                "TSS",

            "model_training_scope":
                "CYCLE24_TRAIN_ONLY",

            "model_retrained_after_selection":
                False,

            "threshold_frozen":
                True,

            "holdout_used_for_selection":
                False,

            "cycle25_used_for_selection":
                False,
        }
    )


    print(
        f"{family}: "
        f"{candidate_id} | "
        f"threshold={threshold:.10g} | "
        f"TSS={validation_tss:.6f} | "
        f"model_hash=PASS | "
        f"FROZEN=YES"
    )


# ============================================================
# BUILD FINAL MANIFEST
# ============================================================

frozen_manifest = pd.DataFrame(
    freeze_rows
)


if len(
    frozen_manifest
) != 3:

    raise RuntimeError(
        "Frozen selected-model manifest must "
        "contain exactly 3 rows."
    )


if frozen_manifest[
    "family"
].nunique() != 3:

    raise RuntimeError(
        "Frozen manifest must contain "
        "three unique model families."
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

OUTPUT_MANIFEST_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


frozen_manifest.to_csv(
    OUTPUT_MANIFEST_FILE,
    index=False,
)


manifest_hash = sha256_file(
    OUTPUT_MANIFEST_FILE
)


# ============================================================
# BUILD FINAL DEVELOPMENT-FREEZE CONTRACT
# ============================================================

selected_models_contract = {

    "contract_name":
        "Frozen Selected Models and Thresholds",

    "version":
        "1.0",

    "status":
        "FROZEN",

    "frozen_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "development_stage":
        "MODEL_SELECTION_COMPLETE",

    "selected_model_count":
        3,

    "cross_family_winner_selected":
        False,

    "selection_dataset":
        "CYCLE24_VALIDATION",

    "selection_metric":
        "TSS",

    "selected_models":
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

                "model_file":
                    row[
                        "model_file"
                    ],

                "model_sha256":
                    row[
                        "model_sha256"
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
            }

            for _, row
            in frozen_manifest.iterrows()
        ],

    "model_policy":
        {
            "training_scope":
                "CYCLE24_TRAIN_ONLY",

            "retrain_on_train_plus_validation":
                False,

            "retraining_after_selection":
                False,

            "hyperparameter_changes_after_selection":
                False,

            "feature_changes_after_selection":
                False,

            "preprocessing_changes_after_selection":
                False,

            "candidate_substitution_after_selection":
                False,
        },

    "threshold_policy":
        {
            "source":
                "CYCLE24_VALIDATION",

            "primary_metric":
                "TSS",

            "frozen":
                True,

            "reoptimization_on_holdout":
                False,

            "reoptimization_on_cycle25":
                False,

            "application_rule":
                "predict_positive_if_probability_greater_than_or_equal_to_threshold",

            "apply_unchanged_to":
                [
                    "CYCLE24_HOLDOUT",
                    "CYCLE25",
                ],
        },

    "next_evaluation_policy":
        {
            "cycle24_holdout":
                {
                    "role":
                        "independent within-cycle test",

                    "tuning_allowed":
                        False,

                    "threshold_changes_allowed":
                        False,

                    "retraining_allowed":
                        False,
                },

            "cycle25":
                {
                    "role":
                        "temporal cross-cycle test",

                    "tuning_allowed":
                        False,

                    "threshold_changes_allowed":
                        False,

                    "retraining_allowed":
                        False,

                    "feature_selection_allowed":
                        False,

                    "preprocessing_refit_allowed":
                        False,
                },
        },

    "scientific_scope":
        {
            "retain_all_three_primary_model_families":
                True,

            "overall_best_model_declared":
                False,

            "prediction_stability_and_explanation_stability_evaluated_separately":
                True,

            "shap_interpretation":
                "MODEL_ATTRIBUTION_NOT_CAUSAL_PHYSICS",
        },

    "verification":
        {
            "chapter_6_1_selection_protocol":
                "PASS",

            "chapter_6_2_validation_predictions":
                "PASS",

            "chapter_6_3_validation_metrics":
                "PASS",

            "chapter_6_4_family_selection":
                "PASS",

            "chapter_6_5_independent_selection_audit":
                "PASS",
        },

    "data_isolation_at_freeze":
        {
            "raw_validation_dataset_read_by_freeze_script":
                False,

            "holdout_read":
                False,

            "cycle25_read":
                False,

            "models_retrained":
                False,

            "thresholds_reoptimized":
                False,
        },

    "immutability_rule":
        (
            "These three selected TRAIN-only model artifacts "
            "and their Cycle24-validation-derived probability "
            "thresholds are frozen. They must be applied "
            "unchanged to the Cycle24 holdout and Cycle25 "
            "evaluation datasets. No retraining, threshold "
            "reoptimization, preprocessing refit, feature "
            "selection, hyperparameter modification, or "
            "candidate substitution is permitted in response "
            "to holdout or Cycle25 outcomes."
        ),

    "frozen_manifest":
        {
            "file":
                str(
                    OUTPUT_MANIFEST_FILE
                ),

            "sha256":
                manifest_hash,
        },

    "provenance":
        {
            "selected_family_winners":
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

            "family_selection_result":
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

            "family_selection_audit":
                {
                    "file":
                        str(
                            SELECTION_AUDIT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            SELECTION_AUDIT_FILE
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

            "validation_metrics_contract":
                {
                    "file":
                        str(
                            VALIDATION_METRICS_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            VALIDATION_METRICS_CONTRACT_FILE
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


with OUTPUT_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        selected_models_contract,
        file,
        indent=2,
    )


# ============================================================
# VERIFY FINAL OUTPUT
# ============================================================

frozen_contract = load_json(
    OUTPUT_CONTRACT_FILE
)


if frozen_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Selected-model contract was not frozen."
    )


if int(
    frozen_contract.get(
        "selected_model_count"
    )
) != 3:

    raise RuntimeError(
        "Frozen selected-model count is not 3."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "FROZEN SELECTED MODELS"
)

print(
    "===================================="
)


for _, row in frozen_manifest.iterrows():

    print(
        f"{row['family']}: "
        f"{row['candidate_id']} | "
        f"threshold={row['selected_threshold']:.10g} | "
        f"TSS={row['validation_tss']:.6f}"
    )


print(
    "\n===================================="
)

print(
    "DEVELOPMENT FREEZE"
)

print(
    "===================================="
)


print(
    "Selected model count:",
    3
)

print(
    "Overall cross-family winner:",
    False
)

print(
    "Winner model retraining allowed:",
    False
)

print(
    "Threshold reoptimization allowed:",
    False
)

print(
    "Hyperparameter modification allowed:",
    False
)

print(
    "Preprocessing refit allowed:",
    False
)

print(
    "Feature changes allowed:",
    False
)


print(
    "\n===================================="
)

print(
    "FUTURE EVALUATION POLICY"
)

print(
    "===================================="
)


print(
    "Cycle24 holdout threshold changes:",
    False
)

print(
    "Cycle25 threshold changes:",
    False
)

print(
    "Cycle25 model tuning:",
    False
)

print(
    "Cycle25 preprocessing refit:",
    False
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
    "\nSaved:"
)

print(
    OUTPUT_MANIFEST_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 6.6 COMPLETE — DEVELOPMENT STATE FROZEN"
)

print(
    "===================================="
)