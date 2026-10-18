from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

LR_FAMILY_CONTRACT = Path(
    "artifacts/models/logistic/contracts/"
    "logistic_candidate_family_contract.json"
)

RF_FAMILY_CONTRACT = Path(
    "artifacts/models/random_forest/contracts/"
    "random_forest_candidate_family_contract.json"
)

XGB_FAMILY_CONTRACT = Path(
    "artifacts/models/xgboost/contracts/"
    "xgboost_candidate_family_contract.json"
)

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/"
    "cycle24_split_contract.json"
)

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "model_feature_contract.json"
)

PREPROCESSING_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "preprocessing_contract.json"
)

OUTPUT_CONTRACT = Path(
    "artifacts/models/selection/contracts/"
    "model_selection_protocol.json"
)

CANDIDATE_REGISTRY = Path(
    "artifacts/models/selection/manifests/"
    "frozen_candidate_registry.csv"
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
    "MODEL CHAPTER 6.1 — FREEZE VALIDATION SELECTION PROTOCOL"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    LR_FAMILY_CONTRACT,
    RF_FAMILY_CONTRACT,
    XGB_FAMILY_CONTRACT,
    SPLIT_CONTRACT_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required frozen input: {path}"
        )


# ============================================================
# LOAD CONTRACTS
# ============================================================

lr_contract = load_json(
    LR_FAMILY_CONTRACT
)

rf_contract = load_json(
    RF_FAMILY_CONTRACT
)

xgb_contract = load_json(
    XGB_FAMILY_CONTRACT
)

split_contract = load_json(
    SPLIT_CONTRACT_FILE
)

feature_contract = load_json(
    FEATURE_CONTRACT_FILE
)

preprocessing_contract = load_json(
    PREPROCESSING_CONTRACT_FILE
)


# ============================================================
# VERIFY GLOBAL CONTRACT STATES
# ============================================================

if split_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 split contract is not FROZEN."
    )


if feature_contract.get(
    "status"
) != "VERIFIED":

    raise RuntimeError(
        "Feature contract is not VERIFIED."
    )


if preprocessing_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Preprocessing contract is not FROZEN."
    )


features = feature_contract.get(
    "feature_order",
    []
)


if len(features) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


# ============================================================
# VERIFY MODEL FAMILY CONTRACTS
# ============================================================

family_specs = [
    {
        "name":
            "logistic",

        "display_name":
            "Logistic Regression",

        "expected_model_family":
            "LogisticRegression",

        "expected_candidate_count":
            10,

        "contract_path":
            LR_FAMILY_CONTRACT,

        "contract":
            lr_contract,
    },

    {
        "name":
            "random_forest",

        "display_name":
            "Random Forest",

        "expected_model_family":
            "RandomForestClassifier",

        "expected_candidate_count":
            12,

        "contract_path":
            RF_FAMILY_CONTRACT,

        "contract":
            rf_contract,
    },

    {
        "name":
            "xgboost",

        "display_name":
            "XGBoost",

        "expected_model_family":
            "XGBClassifier",

        "expected_candidate_count":
            12,

        "contract_path":
            XGB_FAMILY_CONTRACT,

        "contract":
            xgb_contract,
    },
]


candidate_rows = []


for spec in family_specs:

    contract = spec[
        "contract"
    ]


    if contract.get(
        "status"
    ) != "FROZEN":

        raise RuntimeError(
            f"{spec['display_name']} "
            "candidate family is not FROZEN."
        )


    if contract.get(
        "model_family"
    ) != spec[
        "expected_model_family"
    ]:

        raise RuntimeError(
            f"{spec['display_name']} model family "
            "does not match expected value."
        )


    candidate_count = int(
        contract.get(
            "candidate_count"
        )
    )


    if candidate_count != spec[
        "expected_candidate_count"
    ]:

        raise RuntimeError(
            f"{spec['display_name']} candidate count "
            f"expected {spec['expected_candidate_count']} "
            f"but found {candidate_count}."
        )


    candidate_ids = contract.get(
        "candidate_ids",
        []
    )


    if len(
        candidate_ids
    ) != candidate_count:

        raise RuntimeError(
            f"{spec['display_name']} candidate ID "
            "count mismatch."
        )


    if len(
        set(
            candidate_ids
        )
    ) != candidate_count:

        raise RuntimeError(
            f"{spec['display_name']} contains "
            "duplicate candidate IDs."
        )


    candidate_files = contract.get(
        "candidate_files",
        []
    )


    if len(
        candidate_files
    ) != candidate_count:

        raise RuntimeError(
            f"{spec['display_name']} candidate-file "
            "count mismatch."
        )


    file_map = {
        entry[
            "candidate_id"
        ]:
        entry

        for entry in candidate_files
    }


    if set(
        file_map.keys()
    ) != set(
        candidate_ids
    ):

        raise RuntimeError(
            f"{spec['display_name']} candidate-file "
            "IDs differ from candidate IDs."
        )


    for candidate_id in candidate_ids:

        entry = file_map[
            candidate_id
        ]


        candidate_rows.append(
            {
                "family":
                    spec[
                        "name"
                    ],

                "family_display_name":
                    spec[
                        "display_name"
                    ],

                "model_family":
                    spec[
                        "expected_model_family"
                    ],

                "candidate_id":
                    candidate_id,

                "model_file":
                    entry[
                        "file"
                    ],

                "model_sha256":
                    entry[
                        "sha256"
                    ],

                "family_contract_file":
                    str(
                        spec[
                            "contract_path"
                        ]
                    ),

                "family_contract_sha256":
                    sha256_file(
                        spec[
                            "contract_path"
                        ]
                    ),

                "family_contract_status":
                    "FROZEN",
            }
        )


    print(
        f"{spec['display_name']}: "
        f"{candidate_count} frozen candidates"
    )


# ============================================================
# BUILD CANDIDATE REGISTRY
# ============================================================

registry = pd.DataFrame(
    candidate_rows
)


if len(
    registry
) != 34:

    raise RuntimeError(
        "Expected exactly 34 frozen candidates "
        "across LR + RF + XGBoost."
    )


if registry[
    "candidate_id"
].duplicated().any():

    raise RuntimeError(
        "Candidate IDs are not globally unique."
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
        "Candidate registry expected "
        "10 Logistic candidates."
    )


if family_counts.get(
    "random_forest"
) != 12:

    raise RuntimeError(
        "Candidate registry expected "
        "12 Random Forest candidates."
    )


if family_counts.get(
    "xgboost"
) != 12:

    raise RuntimeError(
        "Candidate registry expected "
        "12 XGBoost candidates."
    )


# ============================================================
# VALIDATION SPLIT PROVENANCE
# ============================================================

validation_info = split_contract.get(
    "validation"
)


if not isinstance(
    validation_info,
    dict,
):

    raise RuntimeError(
        "Frozen split contract does not contain "
        "validation provenance."
    )


validation_hash = validation_info.get(
    "sha256"
)


if not validation_hash:

    raise RuntimeError(
        "Validation SHA256 missing from "
        "frozen split contract."
    )


# ============================================================
# FREEZE SELECTION / THRESHOLD RULES
# ============================================================

selection_protocol = {

    "protocol_name":
        "Cycle24 Validation Model and Threshold Selection Protocol",

    "version":
        "1.0",

    "status":
        "FROZEN_BEFORE_VALIDATION",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "scientific_role":
        (
            "Select exactly one frozen candidate and one "
            "frozen probability threshold independently "
            "within each primary model family using only "
            "the Cycle24 validation partition."
        ),

    "model_families":
        [
            "LogisticRegression",
            "RandomForestClassifier",
            "XGBClassifier",
        ],

    "candidate_counts":
        {
            "logistic_regression":
                10,

            "random_forest":
                12,

            "xgboost":
                12,

            "total":
                34,
        },

    "selection_scope":
        {
            "winner_per_family":
                1,

            "family_winners_total":
                3,

            "cross_family_winner_selected":
                False,

            "reason":
                (
                    "All three primary model families are "
                    "retained for later predictive and SHAP "
                    "cross-cycle stability analysis."
                ),
        },

    "validation_dataset":
        {
            "name":
                "CYCLE24_VALIDATION",

            "source":
                "frozen Cycle24 split",

            "sha256":
                validation_hash,

            "used_for":
                [
                    "candidate evaluation",
                    "candidate-specific threshold selection",
                    "within-family candidate selection",
                ],

            "used_for_training":
                False,

            "used_for_retraining":
                False,
        },

    "threshold_protocol":
        {
            "prediction_rule":
                "predict_positive_if_probability_greater_than_or_equal_to_threshold",

            "candidate_threshold_set":
                (
                    "All unique predicted probabilities on "
                    "Cycle24 validation for that frozen "
                    "candidate, plus one all-negative sentinel "
                    "threshold immediately above the maximum "
                    "predicted probability."
                ),

            "all_negative_sentinel":
                "numpy.nextafter(max_probability, +infinity)",

            "primary_threshold_metric":
                "TSS",

            "tss_definition":
                "TPR_minus_FPR",

            "tpr_definition":
                "TP / (TP + FN)",

            "fpr_definition":
                "FP / (FP + TN)",

            "threshold_winner_rule":
                "maximum_validation_TSS",

            "exact_threshold_tie_break":
                (
                    "If multiple thresholds produce exactly "
                    "the same maximum TSS, select the highest "
                    "threshold as a deterministic tie break."
                ),

            "default_0_5_threshold_used":
                False,

            "threshold_optimized_on_holdout":
                False,

            "threshold_optimized_on_cycle25":
                False,
        },

    "validation_metrics":
        {
            "selection_metrics":
                [
                    "TSS",
                    "PR_AUC",
                ],

            "diagnostic_metrics":
                [
                    "HSS",
                    "F1",
                    "ROC_AUC",
                    "Brier_score",
                    "sensitivity",
                    "specificity",
                    "precision",
                ],

            "primary_candidate_selection_metric":
                "TSS",

            "secondary_candidate_selection_metric":
                "PR_AUC",

            "note":
                (
                    "Diagnostic metrics may be reported but "
                    "must not override the pre-registered "
                    "family-specific selection hierarchy."
                ),
        },

    "family_selection_rules":
        {

            "logistic_regression":
                {
                    "winner_order":
                        [
                            "higher_validation_TSS",
                            "higher_validation_PR_AUC",
                            "smaller_C",
                            "class_weight_None_before_balanced",
                            "lexicographically_smaller_candidate_id",
                        ],
                },

            "random_forest":
                {
                    "winner_order":
                        [
                            "higher_validation_TSS",
                            "higher_validation_PR_AUC",
                            "smaller_max_depth",
                            "larger_min_samples_leaf",
                            "class_weight_None_before_balanced_subsample",
                            "lexicographically_smaller_candidate_id",
                        ],

                    "max_depth_complexity_order":
                        [
                            12,
                            24,
                            "None",
                        ],

                    "note":
                        (
                            "max_depth=None is treated as "
                            "unbounded and therefore ranks after "
                            "numeric depths 12 and 24 when a "
                            "complexity tie-break is required."
                        ),
                },

            "xgboost":
                {
                    "winner_order":
                        [
                            "higher_validation_TSS",
                            "higher_validation_PR_AUC",
                            "smaller_max_depth",
                            "smaller_learning_rate",
                            "scale_pos_weight_1_before_balanced",
                            "lexicographically_smaller_candidate_id",
                        ],
                },
        },

    "post_selection_rule":
        {
            "retraining_after_validation":
                False,

            "winner_model_artifact":
                (
                    "Use the already-frozen TRAIN-only model "
                    "artifact. Do not refit on TRAIN+VALIDATION."
                ),

            "winner_threshold":
                (
                    "Freeze the Cycle24-validation-selected "
                    "threshold and apply it unchanged to "
                    "Cycle24 holdout and Cycle25."
                ),

            "hyperparameter_changes":
                False,

            "feature_changes":
                False,

            "preprocessing_changes":
                False,
        },

    "holdout_policy":
        {
            "used_in_chapter_6_selection":
                False,

            "allowed_after_selection_freeze":
                True,

            "role":
                (
                    "Independent Cycle24 performance estimate "
                    "after all model and threshold development "
                    "decisions are frozen."
                ),
        },

    "cycle25_policy":
        {
            "used_in_chapter_6_selection":
                False,

            "allowed_after_development_freeze":
                True,

            "role":
                (
                    "Temporal cross-cycle test only; never "
                    "used for tuning, threshold selection, "
                    "feature selection, or preprocessing fit."
                ),
        },

    "interpretation_constraints":
        [
            (
                "Validation-selected thresholds are operating "
                "points, not intrinsic physical flare-probability "
                "boundaries."
            ),

            (
                "SHAP values later describe model attribution "
                "and must not be interpreted as causal solar "
                "physics."
            ),

            (
                "Predictive stability and explanation stability "
                "will be evaluated separately."
            ),
        ],

    "immutability_rule":
        (
            "After this protocol is frozen and before viewing "
            "validation outcomes, the threshold-search method, "
            "selection metrics, family-specific tie-break rules, "
            "candidate sets, preprocessing, and model artifacts "
            "must not be changed in response to validation, "
            "holdout, or Cycle25 results."
        ),

    "data_isolation_at_freeze":
        {
            "candidate_family_contracts_read":
                True,

            "validation_dataset_read":
                False,

            "validation_predictions_generated":
                False,

            "validation_metrics_computed":
                False,

            "thresholds_selected":
                False,

            "family_winners_selected":
                False,

            "holdout_read":
                False,

            "cycle25_read":
                False,
        },

    "provenance":
        {
            "split_contract":
                {
                    "file":
                        str(
                            SPLIT_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            SPLIT_CONTRACT_FILE
                        ),
                },

            "feature_contract":
                {
                    "file":
                        str(
                            FEATURE_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            FEATURE_CONTRACT_FILE
                        ),
                },

            "preprocessing_contract":
                {
                    "file":
                        str(
                            PREPROCESSING_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            PREPROCESSING_CONTRACT_FILE
                        ),
                },

            "logistic_family_contract":
                {
                    "file":
                        str(
                            LR_FAMILY_CONTRACT
                        ),

                    "sha256":
                        sha256_file(
                            LR_FAMILY_CONTRACT
                        ),
                },

            "random_forest_family_contract":
                {
                    "file":
                        str(
                            RF_FAMILY_CONTRACT
                        ),

                    "sha256":
                        sha256_file(
                            RF_FAMILY_CONTRACT
                        ),
                },

            "xgboost_family_contract":
                {
                    "file":
                        str(
                            XGB_FAMILY_CONTRACT
                        ),

                    "sha256":
                        sha256_file(
                            XGB_FAMILY_CONTRACT
                        ),
                },
        },
}


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_CONTRACT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


CANDIDATE_REGISTRY.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAFE FREEZE
# ============================================================

if OUTPUT_CONTRACT.exists():

    raise RuntimeError(
        "Validation selection protocol already exists. "
        "Refusing to overwrite frozen selection rules."
    )


if CANDIDATE_REGISTRY.exists():

    raise RuntimeError(
        "Frozen candidate registry already exists. "
        "Refusing to overwrite."
    )


# ============================================================
# SAVE
# ============================================================

with OUTPUT_CONTRACT.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        selection_protocol,
        file,
        indent=2,
    )


registry.to_csv(
    CANDIDATE_REGISTRY,
    index=False,
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "FROZEN CANDIDATE REGISTRY"
)

print(
    "===================================="
)


print(
    "Logistic Regression candidates:",
    family_counts[
        "logistic"
    ]
)

print(
    "Random Forest candidates:",
    family_counts[
        "random_forest"
    ]
)

print(
    "XGBoost candidates:",
    family_counts[
        "xgboost"
    ]
)

print(
    "Total frozen candidates:",
    len(
        registry
    )
)


print(
    "\n===================================="
)

print(
    "THRESHOLD SELECTION PROTOCOL"
)

print(
    "===================================="
)


print(
    "Primary metric: TSS"
)

print(
    "Decision rule: probability >= threshold"
)

print(
    "Threshold candidates: unique validation probabilities"
)

print(
    "All-negative sentinel included: True"
)

print(
    "Threshold tie-break: highest threshold"
)

print(
    "Default threshold 0.5 forced:",
    False
)


print(
    "\n===================================="
)

print(
    "MODEL FAMILY SELECTION"
)

print(
    "===================================="
)


print(
    "Winner per model family:",
    1
)

print(
    "Total family winners:",
    3
)

print(
    "Cross-family winner:",
    False
)

print(
    "Primary candidate metric: TSS"
)

print(
    "Secondary candidate metric: PR-AUC"
)


print(
    "\n===================================="
)

print(
    "POST-SELECTION RULE"
)

print(
    "===================================="
)


print(
    "Retrain winner on TRAIN + VALIDATION:",
    False
)

print(
    "Use original frozen TRAIN-only model:",
    True
)

print(
    "Freeze selected validation threshold:",
    True
)

print(
    "Threshold reused unchanged on holdout/Cycle25:",
    True
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
    "Validation dataset read:",
    False
)

print(
    "Validation predictions generated:",
    False
)

print(
    "Validation metrics computed:",
    False
)

print(
    "Thresholds selected:",
    False
)

print(
    "Family winners selected:",
    False
)

print(
    "HOLDOUT read:",
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
    OUTPUT_CONTRACT
)

print(
    CANDIDATE_REGISTRY
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 6.1 COMPLETE — SELECTION PROTOCOL FROZEN"
)

print(
    "===================================="
)