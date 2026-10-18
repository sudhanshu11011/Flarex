from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROTOCOL_FILE = Path(
    "artifacts/models/xgboost/contracts/"
    "xgboost_training_protocol.json"
)

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "model_feature_contract.json"
)

PREPROCESSING_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/"
    "preprocessing_contract.json"
)

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/"
    "cycle24_split_contract.json"
)

GRID_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_candidate_grid.csv"
)

TRAINING_SUMMARY_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_training_summary.csv"
)

CANDIDATE_AUDIT_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_candidate_audit.csv"
)

ARTIFACT_INVENTORY_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_artifact_inventory.csv"
)

ARTIFACT_VERIFICATION_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_artifact_verification_summary.csv"
)

REPRO_AUDIT_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_reproducibility_audit.csv"
)

REPRO_SUMMARY_FILE = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_reproducibility_summary.csv"
)

MODEL_DIR = Path(
    "artifacts/models/xgboost/models"
)

CONTRACT_FILE = Path(
    "artifacts/models/xgboost/contracts/"
    "xgboost_candidate_family_contract.json"
)

FILES_MANIFEST = Path(
    "artifacts/models/xgboost/manifests/"
    "xgboost_candidate_family_files.csv"
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


def as_bool(value):

    if isinstance(value, bool):
        return value

    return str(
        value
    ).strip().lower() == "true"


def float_match(
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
    "MODEL CHAPTER 5.6 — FREEZE XGBOOST CANDIDATE FAMILY"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    PROTOCOL_FILE,
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
    SPLIT_CONTRACT_FILE,
    GRID_FILE,
    TRAINING_SUMMARY_FILE,
    CANDIDATE_AUDIT_FILE,
    ARTIFACT_INVENTORY_FILE,
    ARTIFACT_VERIFICATION_FILE,
    REPRO_AUDIT_FILE,
    REPRO_SUMMARY_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


if not MODEL_DIR.exists():

    raise FileNotFoundError(
        f"XGBoost model directory missing: {MODEL_DIR}"
    )


# ============================================================
# LOAD CONTRACTS
# ============================================================

with PROTOCOL_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    protocol = json.load(file)


with FEATURE_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    feature_contract = json.load(file)


with PREPROCESSING_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    preprocessing_contract = json.load(file)


with SPLIT_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    split_contract = json.load(file)


# ============================================================
# VERIFY CONTRACT STATES
# ============================================================

if protocol.get(
    "status"
) != "FROZEN_BEFORE_TRAINING":

    raise RuntimeError(
        "XGBoost training protocol is not frozen."
    )


if protocol.get(
    "model_family"
) != "XGBClassifier":

    raise RuntimeError(
        "Unexpected model family."
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


if split_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 split contract is not FROZEN."
    )


features = feature_contract.get(
    "feature_order",
    []
)


if len(features) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen features."
    )


print(
    "XGBoost training protocol: FROZEN"
)

print(
    "Feature contract: VERIFIED"
)

print(
    "Preprocessing contract: FROZEN"
)

print(
    "Cycle24 split: FROZEN"
)


# ============================================================
# LOAD MANIFESTS
# ============================================================

grid = pd.read_csv(
    GRID_FILE,
    keep_default_na=False,
)


training_summary = pd.read_csv(
    TRAINING_SUMMARY_FILE,
    keep_default_na=False,
)


candidate_audit = pd.read_csv(
    CANDIDATE_AUDIT_FILE,
    keep_default_na=False,
)


artifact_inventory = pd.read_csv(
    ARTIFACT_INVENTORY_FILE,
    keep_default_na=False,
)


artifact_verification = pd.read_csv(
    ARTIFACT_VERIFICATION_FILE,
    keep_default_na=False,
)


repro_audit = pd.read_csv(
    REPRO_AUDIT_FILE,
    keep_default_na=False,
)


repro_summary = pd.read_csv(
    REPRO_SUMMARY_FILE,
    keep_default_na=False,
)


# ============================================================
# ROW COUNTS
# ============================================================

for name, frame in [
    (
        "candidate grid",
        grid,
    ),
    (
        "training summary",
        training_summary,
    ),
    (
        "candidate audit",
        candidate_audit,
    ),
    (
        "artifact inventory",
        artifact_inventory,
    ),
    (
        "reproducibility audit",
        repro_audit,
    ),
]:

    if len(frame) != 12:

        raise RuntimeError(
            f"Expected 12 rows in {name}."
        )


if len(
    artifact_verification
) != 1:

    raise RuntimeError(
        "Artifact verification summary must "
        "contain exactly one row."
    )


if len(
    repro_summary
) != 1:

    raise RuntimeError(
        "Reproducibility summary must "
        "contain exactly one row."
    )


# ============================================================
# CANDIDATE SET CONSISTENCY
# ============================================================

candidate_ids = list(
    grid[
        "candidate_id"
    ].astype(str)
)


candidate_set = set(
    candidate_ids
)


if len(
    candidate_set
) != 12:

    raise RuntimeError(
        "XGBoost grid does not contain "
        "12 unique candidates."
    )


for name, frame in [
    (
        "training summary",
        training_summary,
    ),
    (
        "candidate audit",
        candidate_audit,
    ),
    (
        "artifact inventory",
        artifact_inventory,
    ),
    (
        "reproducibility audit",
        repro_audit,
    ),
]:

    current_set = set(
        frame[
            "candidate_id"
        ].astype(str)
    )


    if current_set != candidate_set:

        raise RuntimeError(
            f"{name} candidate set differs "
            "from frozen XGBoost grid."
        )


# ============================================================
# VERIFY CHAPTER 5.2 TRAINING
# ============================================================

if not training_summary[
    "boosted_rounds"
].astype(int).eq(
    500
).all():

    raise RuntimeError(
        "One or more XGBoost candidates "
        "do not contain 500 boosted rounds."
    )


total_training_warnings = int(
    training_summary[
        "warning_count"
    ].astype(int).sum()
)


if total_training_warnings != 0:

    raise RuntimeError(
        "XGBoost training warnings present."
    )


if not training_summary[
    "serialization_probability_match"
].apply(
    as_bool
).all():

    raise RuntimeError(
        "One or more serialized XGBoost models "
        "failed probability verification."
    )


print(
    "Chapter 5.2 training state: PASS"
)

print(
    "Training warnings: 0"
)


# ============================================================
# VERIFY CHAPTER 5.3 AUDIT
# ============================================================

chapter53_pass = bool(
    candidate_audit[
        "candidate_audit_pass"
    ].apply(
        as_bool
    ).all()
)


if not chapter53_pass:

    raise RuntimeError(
        "Chapter 5.3 candidate audit "
        "did not fully PASS."
    )


print(
    "Chapter 5.3 candidate audit: PASS"
)


# ============================================================
# VERIFY CHAPTER 5.4 ARTIFACTS
# ============================================================

artifact_row = artifact_verification.iloc[
    0
]


chapter54_pass = as_bool(
    artifact_row[
        "overall_artifact_verification_pass"
    ]
)


if not chapter54_pass:

    raise RuntimeError(
        "Chapter 5.4 artifact verification "
        "did not PASS."
    )


if int(
    artifact_row[
        "artifact_pass_count"
    ]
) != 12:

    raise RuntimeError(
        "Not all 12 XGBoost artifacts passed."
    )


if int(
    artifact_row[
        "missing_artifacts"
    ]
) != 0:

    raise RuntimeError(
        "Missing XGBoost artifacts detected."
    )


if int(
    artifact_row[
        "unexpected_artifacts"
    ]
) != 0:

    raise RuntimeError(
        "Unexpected XGBoost artifacts detected."
    )


if int(
    artifact_row[
        "selection_metadata_clean_count"
    ]
) != 12:

    raise RuntimeError(
        "Premature selection metadata detected."
    )


if int(
    artifact_row[
        "early_stopping_clean_count"
    ]
) != 12:

    raise RuntimeError(
        "Early-stopping contamination detected."
    )


if int(
    artifact_row[
        "validation_clean_count"
    ]
) != 12:

    raise RuntimeError(
        "Validation contamination detected."
    )


print(
    "Chapter 5.4 artifact verification: PASS"
)


# ============================================================
# VERIFY CHAPTER 5.5 REPRODUCIBILITY
# ============================================================

repro_row = repro_summary.iloc[
    0
]


chapter55_pass = as_bool(
    repro_row[
        "reproducibility_pass"
    ]
)


if not chapter55_pass:

    raise RuntimeError(
        "Chapter 5.5 reproducibility "
        "did not PASS."
    )


if int(
    repro_row[
        "candidates_reproducible"
    ]
) != 12:

    raise RuntimeError(
        "Not all 12 XGBoost candidates "
        "are reproducible."
    )


if int(
    repro_row[
        "exact_tree_dump_match_count"
    ]
) != 12:

    raise RuntimeError(
        "Not all XGBoost tree structures "
        "reproduced exactly."
    )


max_importance_diff = float(
    repro_row[
        "max_feature_importance_abs_difference"
    ]
)


importance_atol = float(
    repro_row[
        "feature_importance_atol"
    ]
)


max_probability_diff = float(
    repro_row[
        "max_probability_abs_difference"
    ]
)


probability_atol = float(
    repro_row[
        "probability_atol"
    ]
)


if max_importance_diff > importance_atol:

    raise RuntimeError(
        "Feature-importance reproducibility "
        "exceeds frozen tolerance."
    )


if max_probability_diff > probability_atol:

    raise RuntimeError(
        "Probability reproducibility "
        "exceeds frozen tolerance."
    )


if int(
    repro_row[
        "fresh_warning_count"
    ]
) != 0:

    raise RuntimeError(
        "Fresh reproducibility training "
        "generated warnings."
    )


print(
    "Chapter 5.5 reproducibility: PASS"
)


# ============================================================
# FIXED PROTOCOL VALUES
# ============================================================

fixed = protocol[
    "model_fixed_parameters"
]


EXPECTED_N_ESTIMATORS = int(
    fixed[
        "n_estimators"
    ]
)


EXPECTED_BOOSTER = str(
    fixed[
        "booster"
    ]
)


EXPECTED_OBJECTIVE = str(
    fixed[
        "objective"
    ]
)


EXPECTED_EVAL_METRIC = str(
    fixed[
        "eval_metric"
    ]
)


EXPECTED_TREE_METHOD = str(
    fixed[
        "tree_method"
    ]
)


EXPECTED_RANDOM_STATE = int(
    fixed[
        "random_state"
    ]
)


EXPECTED_N_JOBS = int(
    fixed[
        "n_jobs"
    ]
)


# ============================================================
# SOURCE HASHES
# ============================================================

protocol_hash = sha256_file(
    PROTOCOL_FILE
)


grid_hash = sha256_file(
    GRID_FILE
)


expected_train_hash = (
    split_contract[
        "train"
    ][
        "sha256"
    ]
)


# ============================================================
# VERIFY ALL MODEL ARTIFACTS
# ============================================================

family_rows = []


print(
    "\n===================================="
)

print(
    "CANDIDATE FAMILY"
)

print(
    "===================================="
)


for candidate_id in candidate_ids:

    model_file = (
        MODEL_DIR
        /
        f"{candidate_id}.joblib"
    )


    if not model_file.exists():

        raise FileNotFoundError(
            f"Missing XGBoost model: "
            f"{candidate_id}"
        )


    training_row = training_summary[
        training_summary[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(training_row) != 1:

        raise RuntimeError(
            f"{candidate_id}: training-summary "
            "row not unique."
        )


    training_row = training_row.iloc[
        0
    ]


    actual_model_hash = sha256_file(
        model_file
    )


    expected_model_hash = str(
        training_row[
            "model_sha256"
        ]
    )


    if actual_model_hash != expected_model_hash:

        raise RuntimeError(
            f"{candidate_id}: model hash mismatch."
        )


    bundle = joblib.load(
        model_file
    )


    model = bundle.get(
        "model"
    )


    if model is None:

        raise RuntimeError(
            f"{candidate_id}: model object missing."
        )


    booster_object = model.get_booster()


    if bundle.get(
        "candidate_id"
    ) != candidate_id:

        raise RuntimeError(
            f"{candidate_id}: bundle ID mismatch."
        )


    if bundle.get(
        "model_family"
    ) != "XGBClassifier":

        raise RuntimeError(
            f"{candidate_id}: model-family mismatch."
        )


    if bundle.get(
        "feature_order"
    ) != features:

        raise RuntimeError(
            f"{candidate_id}: feature-order mismatch."
        )


    if int(
        bundle.get(
            "feature_count"
        )
    ) != 25:

        raise RuntimeError(
            f"{candidate_id}: feature-count mismatch."
        )


    if bundle.get(
        "preprocessing"
    ) != "IDENTITY_RAW_FEATURES":

        raise RuntimeError(
            f"{candidate_id}: preprocessing mismatch."
        )


    if bundle.get(
        "training_scope"
    ) != "CYCLE24_TRAIN_ONLY":

        raise RuntimeError(
            f"{candidate_id}: training scope mismatch."
        )


    if bundle.get(
        "train_sha256"
    ) != expected_train_hash:

        raise RuntimeError(
            f"{candidate_id}: TRAIN hash mismatch."
        )


    if bundle.get(
        "protocol_sha256"
    ) != protocol_hash:

        raise RuntimeError(
            f"{candidate_id}: protocol hash mismatch."
        )


    if bundle.get(
        "candidate_grid_sha256"
    ) != grid_hash:

        raise RuntimeError(
            f"{candidate_id}: candidate-grid "
            "hash mismatch."
        )


    if int(
        bundle.get(
            "n_estimators"
        )
    ) != EXPECTED_N_ESTIMATORS:

        raise RuntimeError(
            f"{candidate_id}: n_estimators mismatch."
        )


    if str(
        bundle.get(
            "booster"
        )
    ) != EXPECTED_BOOSTER:

        raise RuntimeError(
            f"{candidate_id}: booster mismatch."
        )


    if str(
        bundle.get(
            "objective"
        )
    ) != EXPECTED_OBJECTIVE:

        raise RuntimeError(
            f"{candidate_id}: objective mismatch."
        )


    if str(
        bundle.get(
            "eval_metric"
        )
    ) != EXPECTED_EVAL_METRIC:

        raise RuntimeError(
            f"{candidate_id}: eval_metric mismatch."
        )


    if str(
        bundle.get(
            "tree_method"
        )
    ) != EXPECTED_TREE_METHOD:

        raise RuntimeError(
            f"{candidate_id}: tree_method mismatch."
        )


    if int(
        bundle.get(
            "random_state"
        )
    ) != EXPECTED_RANDOM_STATE:

        raise RuntimeError(
            f"{candidate_id}: random_state mismatch."
        )


    if int(
        bundle.get(
            "n_jobs"
        )
    ) != EXPECTED_N_JOBS:

        raise RuntimeError(
            f"{candidate_id}: n_jobs mismatch."
        )


    if bundle.get(
        "early_stopping_used"
    ) is not False:

        raise RuntimeError(
            f"{candidate_id}: early stopping "
            "must be False."
        )


    if bundle.get(
        "validation_used"
    ) is not False:

        raise RuntimeError(
            f"{candidate_id}: validation "
            "contamination detected."
        )


    boosted_rounds = int(
        booster_object.num_boosted_rounds()
    )


    if boosted_rounds != 500:

        raise RuntimeError(
            f"{candidate_id}: boosted-round "
            "count mismatch."
        )


    dumped_tree_count = len(
        booster_object.get_dump(
            dump_format="json"
        )
    )


    if dumped_tree_count != 500:

        raise RuntimeError(
            f"{candidate_id}: dumped-tree "
            "count mismatch."
        )


    family_rows.append(
        {
            "candidate_id":
                candidate_id,

            "learning_rate":
                float(
                    bundle.get(
                        "learning_rate"
                    )
                ),

            "max_depth":
                int(
                    bundle.get(
                        "max_depth"
                    )
                ),

            "scale_pos_weight":
                float(
                    bundle.get(
                        "scale_pos_weight"
                    )
                ),

            "n_estimators":
                int(
                    bundle.get(
                        "n_estimators"
                    )
                ),

            "boosted_rounds":
                boosted_rounds,

            "dumped_tree_count":
                dumped_tree_count,

            "random_state":
                int(
                    bundle.get(
                        "random_state"
                    )
                ),

            "model_file":
                str(
                    model_file
                ),

            "model_sha256":
                actual_model_hash,

            "size_bytes":
                model_file.stat().st_size,

            "train_sha256":
                str(
                    bundle.get(
                        "train_sha256"
                    )
                ),

            "protocol_sha256":
                str(
                    bundle.get(
                        "protocol_sha256"
                    )
                ),

            "candidate_grid_sha256":
                str(
                    bundle.get(
                        "candidate_grid_sha256"
                    )
                ),

            "sklearn_version":
                str(
                    bundle.get(
                        "sklearn_version"
                    )
                ),

            "xgboost_version":
                str(
                    bundle.get(
                        "xgboost_version"
                    )
                ),
        }
    )


    print(
        f"{candidate_id}: "
        f"rounds={boosted_rounds}, "
        f"trees={dumped_tree_count}, "
        f"hash=PASS, "
        f"training_scope=TRAIN_ONLY, "
        f"frozen=YES"
    )


family_df = pd.DataFrame(
    family_rows
)


if len(
    family_df
) != 12:

    raise RuntimeError(
        "Frozen XGBoost family does not "
        "contain 12 models."
    )


# ============================================================
# SOFTWARE CONSISTENCY
# ============================================================

sklearn_versions = sorted(
    family_df[
        "sklearn_version"
    ].unique()
)


xgboost_versions = sorted(
    family_df[
        "xgboost_version"
    ].unique()
)


if len(
    sklearn_versions
) != 1:

    raise RuntimeError(
        "Inconsistent scikit-learn versions."
    )


if len(
    xgboost_versions
) != 1:

    raise RuntimeError(
        "Inconsistent XGBoost versions."
    )


# ============================================================
# PROVENANCE
# ============================================================

provenance_files = {
    "training_protocol":
        PROTOCOL_FILE,

    "feature_contract":
        FEATURE_CONTRACT_FILE,

    "preprocessing_contract":
        PREPROCESSING_CONTRACT_FILE,

    "split_contract":
        SPLIT_CONTRACT_FILE,

    "candidate_grid":
        GRID_FILE,

    "training_summary":
        TRAINING_SUMMARY_FILE,

    "candidate_audit":
        CANDIDATE_AUDIT_FILE,

    "artifact_inventory":
        ARTIFACT_INVENTORY_FILE,

    "artifact_verification":
        ARTIFACT_VERIFICATION_FILE,

    "reproducibility_audit":
        REPRO_AUDIT_FILE,

    "reproducibility_summary":
        REPRO_SUMMARY_FILE,
}


provenance = {}


for name, path in provenance_files.items():

    provenance[
        name
    ] = {
        "file":
            str(
                path
            ),

        "sha256":
            sha256_file(
                path
            ),
    }


# ============================================================
# BUILD CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Frozen XGBoost Candidate Family",

    "version":
        "1.0",

    "status":
        "FROZEN",

    "frozen_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "model_family":
        "XGBClassifier",

    "candidate_count":
        12,

    "candidate_ids":
        candidate_ids,

    "training_scope":
        "CYCLE24_TRAIN_ONLY",

    "feature_count":
        25,

    "feature_order":
        features,

    "preprocessing":
        {
            "method":
                "IDENTITY_RAW_FEATURES",

            "scaling":
                False,

            "imputation":
                False,

            "feature_selection":
                False,

            "refit_allowed":
                False,
        },

    "training_protocol":
        {
            "n_estimators":
                EXPECTED_N_ESTIMATORS,

            "booster":
                EXPECTED_BOOSTER,

            "objective":
                EXPECTED_OBJECTIVE,

            "eval_metric":
                EXPECTED_EVAL_METRIC,

            "tree_method":
                EXPECTED_TREE_METHOD,

            "random_state":
                EXPECTED_RANDOM_STATE,

            "n_jobs":
                EXPECTED_N_JOBS,

            "candidate_search_space":
                protocol[
                    "candidate_search_space"
                ],

            "early_stopping":
                False,
        },

    "verification":
        {
            "chapter_5_2_training":
                "PASS",

            "candidate_count_trained":
                12,

            "boosted_rounds_per_candidate":
                500,

            "total_boosted_rounds":
                6000,

            "training_warning_count":
                total_training_warnings,

            "chapter_5_3_candidate_audit":
                "PASS",

            "chapter_5_4_artifact_verification":
                "PASS",

            "chapter_5_5_reproducibility":
                "PASS",

            "exact_tree_dump_reproducibility":
                "PASS",

            "max_feature_importance_difference":
                max_importance_diff,

            "feature_importance_tolerance":
                importance_atol,

            "max_probability_difference":
                max_probability_diff,

            "probability_tolerance":
                probability_atol,
        },

    "selection_state":
        {
            "winner_selected":
                False,

            "threshold_selected":
                False,

            "validation_metrics_computed":
                False,

            "selection_stage":
                "MODEL_CHAPTER_6",

            "selection_dataset":
                "CYCLE24_VALIDATION",

            "primary_selection_metric":
                "TSS",

            "threshold_strategy":
                (
                    "Candidate-specific probability "
                    "threshold maximizing TSS on "
                    "Cycle24 validation only."
                ),
        },

    "data_isolation":
        {
            "training_data":
                "CYCLE24_TRAIN_ONLY",

            "validation_used":
                False,

            "holdout_used":
                False,

            "cycle25_used":
                False,
        },

    "interpretation_note":
        (
            "Built-in XGBoost feature importance is "
            "retained only as a model diagnostic. "
            "It is not treated as SHAP explanation, "
            "causal solar-physics evidence, or a basis "
            "for feature selection."
        ),

    "immutability_rule":
        (
            "The 12 XGBoost candidate models, their "
            "hyperparameters, feature order, TRAIN "
            "provenance, class-weight definitions, "
            "boosted tree structures, candidate grid, "
            "and artifact hashes are frozen. "
            "Candidates must not be retrained, added, "
            "removed, or modified in response to "
            "validation, holdout, or Cycle25 results. "
            "Chapter 6 may only evaluate and select "
            "among this frozen candidate family."
        ),

    "software":
        {
            "scikit_learn_version":
                sklearn_versions[
                    0
                ],

            "xgboost_version":
                xgboost_versions[
                    0
                ],
        },

    "candidate_files":
        [
            {
                "candidate_id":
                    row[
                        "candidate_id"
                    ],

                "file":
                    row[
                        "model_file"
                    ],

                "sha256":
                    row[
                        "model_sha256"
                    ],
            }

            for _, row
            in family_df.iterrows()
        ],

    "provenance":
        provenance,
}


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

FILES_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)


CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAFE FREEZE
# ============================================================

if CONTRACT_FILE.exists():

    with CONTRACT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        existing = json.load(
            file
        )


    if existing.get(
        "status"
    ) != "FROZEN":

        raise RuntimeError(
            "Existing XGBoost family contract "
            "is not marked FROZEN."
        )


    existing_candidates = {
        item[
            "candidate_id"
        ]:
        item[
            "sha256"
        ]

        for item
        in existing.get(
            "candidate_files",
            []
        )
    }


    current_candidates = {
        item[
            "candidate_id"
        ]:
        item[
            "sha256"
        ]

        for item
        in contract[
            "candidate_files"
        ]
    }


    if existing_candidates != current_candidates:

        raise RuntimeError(
            "Existing frozen XGBoost candidate-family "
            "contract references different model "
            "artifacts. Refusing to overwrite."
        )


    print(
        "Existing frozen XGBoost contract "
        "matches current candidate hashes."
    )


else:

    # --------------------------------------------------------
    # Save family manifest
    # --------------------------------------------------------

    family_df.to_csv(
        FILES_MANIFEST,
        index=False,
    )


    # --------------------------------------------------------
    # Save frozen contract
    # --------------------------------------------------------

    with CONTRACT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            contract,
            file,
            indent=2,
        )


# ============================================================
# VERIFY OUTPUTS
# ============================================================

if not CONTRACT_FILE.exists():

    raise RuntimeError(
        "Frozen XGBoost family contract "
        "was not created."
    )


if not FILES_MANIFEST.exists():

    # Existing contract from previous safe run may exist;
    # manifest must still exist.
    family_df.to_csv(
        FILES_MANIFEST,
        index=False,
    )


with CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    frozen_contract = json.load(
        file
    )


if frozen_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "XGBoost candidate-family contract "
        "is not FROZEN."
    )


if int(
    frozen_contract.get(
        "candidate_count"
    )
) != 12:

    raise RuntimeError(
        "Frozen XGBoost candidate count "
        "is not 12."
    )


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "FROZEN XGBOOST CANDIDATE FAMILY"
)

print(
    "===================================="
)


print(
    "Model family: XGBClassifier"
)

print(
    "Candidate count:",
    12
)

print(
    "Training scope: CYCLE24 TRAIN ONLY"
)

print(
    "Features:",
    25
)

print(
    "Boosted rounds per candidate:",
    500
)

print(
    "Total frozen boosted rounds:",
    6000
)

print(
    "Training warnings:",
    total_training_warnings
)

print(
    "Exact tree reproducibility:",
    "PASS"
)

print(
    "Feature-importance reproducibility:",
    "PASS"
)

print(
    "Probability reproducibility:",
    "PASS"
)


print(
    "\n===================================="
)

print(
    "SOFTWARE PROVENANCE"
)

print(
    "===================================="
)


print(
    "scikit-learn:",
    sklearn_versions[
        0
    ]
)

print(
    "XGBoost:",
    xgboost_versions[
        0
    ]
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
    "Winner selected:",
    False
)

print(
    "Threshold selected:",
    False
)

print(
    "Validation metrics computed:",
    False
)

print(
    "Selection stage: Chapter 6"
)

print(
    "Selection dataset: CYCLE24 VALIDATION"
)

print(
    "Selection metric: TSS"
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
    "Dataset files read by freeze script:",
    False
)

print(
    "VALIDATION used:",
    False
)

print(
    "HOLDOUT used:",
    False
)

print(
    "Cycle25 used:",
    False
)


print(
    "\n===================================="
)

print(
    "CANDIDATE FAMILY CONTRACT"
)

print(
    "===================================="
)


print(
    "Version: 1.0"
)

print(
    "Status: FROZEN"
)

print(
    "Candidate retraining after freeze: FORBIDDEN"
)

print(
    "Candidate addition/removal after freeze: FORBIDDEN"
)

print(
    "Performance-driven candidate changes: FORBIDDEN"
)


print(
    "\nSaved:"
)

print(
    FILES_MANIFEST
)

print(
    CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 5.6 COMPLETE — XGBOOST FAMILY FROZEN"
)

print(
    "===================================="
)