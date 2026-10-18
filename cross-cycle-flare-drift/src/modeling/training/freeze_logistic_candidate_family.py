from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROTOCOL_FILE = Path(
    "artifacts/models/logistic/contracts/"
    "logistic_training_protocol.json"
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
    "artifacts/models/logistic/manifests/"
    "logistic_candidate_grid.csv"
)

TRAINING_SUMMARY_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_training_summary.csv"
)

CANDIDATE_AUDIT_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_candidate_audit.csv"
)

ARTIFACT_INVENTORY_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_artifact_inventory.csv"
)

ARTIFACT_VERIFICATION_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_artifact_verification_summary.csv"
)

REPRO_AUDIT_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_reproducibility_audit.csv"
)

REPRO_SUMMARY_FILE = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_reproducibility_summary.csv"
)


MODEL_DIR = Path(
    "artifacts/models/logistic/models"
)


CONTRACT_FILE = Path(
    "artifacts/models/logistic/contracts/"
    "logistic_candidate_family_contract.json"
)

FILES_MANIFEST = Path(
    "artifacts/models/logistic/manifests/"
    "logistic_candidate_family_files.csv"
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


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 3.6 — FREEZE LR CANDIDATE FAMILY"
)

print(
    "===================================="
)


# ============================================================
# VERIFY REQUIRED INPUTS
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


if protocol.get(
    "status"
) != "FROZEN_BEFORE_TRAINING":

    raise RuntimeError(
        "LR training protocol is not frozen."
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
    "LR training protocol: FROZEN"
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
# LOAD CHAPTER 3 MANIFESTS
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
# BASIC COUNTS
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

    if len(frame) != 10:

        raise RuntimeError(
            f"Expected 10 rows in {name}."
        )


if len(
    artifact_verification
) != 1:

    raise RuntimeError(
        "Artifact verification summary "
        "must contain exactly one row."
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


expected_set = set(
    candidate_ids
)


if len(expected_set) != 10:

    raise RuntimeError(
        "Frozen grid does not contain "
        "10 unique candidate IDs."
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

    if current_set != expected_set:

        raise RuntimeError(
            f"{name} candidate set does not "
            "match frozen grid."
        )


# ============================================================
# VERIFY 3.3
# ============================================================

candidate_audit_pass = bool(
    candidate_audit[
        "candidate_audit_pass"
    ].apply(
        as_bool
    ).all()
)


if not candidate_audit_pass:

    raise RuntimeError(
        "Chapter 3.3 candidate audit did not "
        "fully PASS."
    )


print(
    "Chapter 3.3 candidate audit: PASS"
)


# ============================================================
# VERIFY 3.4
# ============================================================

artifact_row = artifact_verification.iloc[
    0
]


artifact_pass = as_bool(
    artifact_row[
        "overall_artifact_verification_pass"
    ]
)


if not artifact_pass:

    raise RuntimeError(
        "Chapter 3.4 artifact verification "
        "did not PASS."
    )


if int(
    artifact_row[
        "artifact_pass_count"
    ]
) != 10:

    raise RuntimeError(
        "Not all 10 artifacts passed "
        "Chapter 3.4 verification."
    )


print(
    "Chapter 3.4 artifact verification: PASS"
)


# ============================================================
# VERIFY 3.5
# ============================================================

repro_row = repro_summary.iloc[
    0
]


repro_pass = as_bool(
    repro_row[
        "reproducibility_pass"
    ]
)


if not repro_pass:

    raise RuntimeError(
        "Chapter 3.5 reproducibility "
        "did not PASS."
    )


if int(
    repro_row[
        "candidates_reproducible"
    ]
) != 10:

    raise RuntimeError(
        "Not all 10 candidates are reproducible."
    )


if float(
    repro_row[
        "max_coefficient_abs_difference"
    ]
) != 0.0:

    raise RuntimeError(
        "Coefficient reproducibility "
        "difference is not zero."
    )


if float(
    repro_row[
        "max_intercept_abs_difference"
    ]
) != 0.0:

    raise RuntimeError(
        "Intercept reproducibility "
        "difference is not zero."
    )


if float(
    repro_row[
        "max_probability_abs_difference"
    ]
) != 0.0:

    raise RuntimeError(
        "Probability reproducibility "
        "difference is not zero."
    )


print(
    "Chapter 3.5 reproducibility: PASS"
)


# ============================================================
# VERIFY TRAINING WARNINGS
# ============================================================

total_convergence_warnings = int(
    training_summary[
        "convergence_warning_count"
    ].astype(int).sum()
)


total_future_warnings = int(
    training_summary[
        "future_warning_count"
    ].astype(int).sum()
)


total_other_warnings = int(
    training_summary[
        "other_warning_count"
    ].astype(int).sum()
)


if total_convergence_warnings != 0:

    raise RuntimeError(
        "Convergence warnings present."
    )


if total_other_warnings != 0:

    raise RuntimeError(
        "Unexpected training warnings present."
    )


print(
    "Convergence warnings: 0"
)

print(
    "FutureWarnings:",
    total_future_warnings,
    "(documented / non-blocking)"
)

print(
    "Unexpected warnings: 0"
)


# ============================================================
# VERIFY MODEL ARTIFACTS + BUILD FAMILY
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
            f"Missing candidate model: "
            f"{candidate_id}"
        )


    training_row = training_summary[
        training_summary[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ].iloc[0]


    actual_hash = sha256_file(
        model_file
    )


    expected_hash = str(
        training_row[
            "model_sha256"
        ]
    )


    if actual_hash != expected_hash:

        raise RuntimeError(
            f"{candidate_id}: model hash "
            "does not match training manifest."
        )


    bundle = joblib.load(
        model_file
    )


    if bundle.get(
        "candidate_id"
    ) != candidate_id:

        raise RuntimeError(
            f"{candidate_id}: bundle ID mismatch."
        )


    if bundle.get(
        "feature_order"
    ) != features:

        raise RuntimeError(
            f"{candidate_id}: feature order mismatch."
        )


    if bundle.get(
        "training_scope"
    ) != "CYCLE24_TRAIN_ONLY":

        raise RuntimeError(
            f"{candidate_id}: training scope mismatch."
        )


    family_rows.append(
        {
            "candidate_id":
                candidate_id,

            "C":
                float(
                    bundle[
                        "C"
                    ]
                ),

            "class_weight":
                (
                    "None"
                    if bundle[
                        "class_weight"
                    ]
                    is None
                    else str(
                        bundle[
                            "class_weight"
                        ]
                    )
                ),

            "penalty":
                str(
                    bundle[
                        "penalty"
                    ]
                ),

            "solver":
                str(
                    bundle[
                        "solver"
                    ]
                ),

            "random_state":
                int(
                    bundle[
                        "random_state"
                    ]
                ),

            "n_iter":
                int(
                    bundle[
                        "n_iter"
                    ]
                ),

            "model_file":
                str(
                    model_file
                ),

            "model_sha256":
                actual_hash,

            "size_bytes":
                model_file.stat().st_size,

            "train_sha256":
                str(
                    bundle[
                        "train_sha256"
                    ]
                ),

            "scaler_sha256":
                str(
                    bundle[
                        "scaler_sha256"
                    ]
                ),

            "sklearn_version":
                str(
                    bundle[
                        "sklearn_version"
                    ]
                ),
        }
    )


    print(
        f"{candidate_id}: "
        f"hash=PASS, "
        f"training_scope=TRAIN_ONLY, "
        f"frozen=YES"
    )


family_df = pd.DataFrame(
    family_rows
)


if len(
    family_df
) != 10:

    raise RuntimeError(
        "Candidate family does not "
        "contain 10 models."
    )


# ============================================================
# SOFTWARE CONSISTENCY
# ============================================================

software_versions = sorted(
    family_df[
        "sklearn_version"
    ].unique()
)


if len(
    software_versions
) != 1:

    raise RuntimeError(
        "Candidate family uses inconsistent "
        "scikit-learn versions."
    )


# ============================================================
# FILE PROVENANCE
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
            str(path),

        "sha256":
            sha256_file(
                path
            ),
    }


# ============================================================
# WRITE FAMILY FILE MANIFEST
# ============================================================

FILES_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)


family_df.to_csv(
    FILES_MANIFEST,
    index=False,
)


# ============================================================
# BUILD FROZEN CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Frozen Logistic Regression Candidate Family",

    "version":
        "1.0",

    "status":
        "FROZEN",

    "frozen_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "model_family":
        "LogisticRegression",

    "candidate_count":
        10,

    "candidate_ids":
        candidate_ids,

    "training_scope":
        "CYCLE24_TRAIN_ONLY",

    "feature_count":
        25,

    "feature_order":
        features,

    "preprocessing": {

        "method":
            "StandardScaler",

        "fit_scope":
            "CYCLE24_TRAIN_ONLY",

        "artifact_sha256":
            preprocessing_contract[
                "logistic_regression"
            ][
                "artifact_sha256"
            ],

        "refit_allowed":
            False,
    },

    "training_protocol": {

        "penalty":
            protocol[
                "model_fixed_parameters"
            ][
                "penalty"
            ],

        "solver":
            protocol[
                "model_fixed_parameters"
            ][
                "solver"
            ],

        "max_iter":
            protocol[
                "model_fixed_parameters"
            ][
                "max_iter"
            ],

        "tol":
            protocol[
                "model_fixed_parameters"
            ][
                "tol"
            ],

        "random_state":
            protocol[
                "model_fixed_parameters"
            ][
                "random_state"
            ],

        "candidate_search_space":
            protocol[
                "candidate_search_space"
            ],
    },

    "verification": {

        "chapter_3_2_training_complete":
            True,

        "candidate_count_trained":
            10,

        "candidate_count_converged":
            10,

        "convergence_warning_count":
            total_convergence_warnings,

        "future_warning_count":
            total_future_warnings,

        "future_warning_status":
            "DOCUMENTED_NON_BLOCKING",

        "unexpected_warning_count":
            total_other_warnings,

        "chapter_3_3_candidate_audit":
            "PASS",

        "chapter_3_4_artifact_verification":
            "PASS",

        "chapter_3_5_reproducibility":
            "PASS",

        "max_reproducibility_coefficient_difference":
            float(
                repro_row[
                    "max_coefficient_abs_difference"
                ]
            ),

        "max_reproducibility_intercept_difference":
            float(
                repro_row[
                    "max_intercept_abs_difference"
                ]
            ),

        "max_reproducibility_probability_difference":
            float(
                repro_row[
                    "max_probability_abs_difference"
                ]
            ),
    },

    "selection_state": {

        "winner_selected":
            False,

        "threshold_selected":
            False,

        "validation_metrics_computed":
            False,

        "selection_stage":
            "MODEL_CHAPTER_6",

        "primary_selection_metric":
            "TSS",

        "selection_dataset":
            "CYCLE24_VALIDATION",
    },

    "data_isolation": {

        "training_data":
            "CYCLE24_TRAIN_ONLY",

        "validation_used":
            False,

        "holdout_used":
            False,

        "cycle25_used":
            False,
    },

    "immutability_rule":
        (
            "The 10 Logistic Regression candidate models, "
            "their hyperparameters, feature order, fitted "
            "parameters, TRAIN provenance, scaler provenance, "
            "and model hashes are frozen. Candidates must not "
            "be retrained, removed, added, or modified in "
            "response to validation, holdout, or Cycle25 "
            "performance. Chapter 6 may only evaluate and "
            "select among this frozen candidate family."
        ),

    "future_warning_note":
        (
            "Training under scikit-learn 1.9.1 emitted one "
            "FutureWarning per candidate because explicit "
            "penalty='l2' is deprecated. No convergence "
            "warnings or unexpected warnings occurred. "
            "The warning is documented and non-blocking; "
            "the pre-registered training protocol was "
            "preserved."
        ),

    "software": {

        "scikit_learn_version":
            software_versions[
                0
            ],
    },

    "candidate_files": [

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
# SAFE CONTRACT WRITE
# ============================================================

CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


if CONTRACT_FILE.exists():

    with CONTRACT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        existing = json.load(
            file
        )


    existing_candidates = {
        entry[
            "candidate_id"
        ]:
        entry[
            "sha256"
        ]

        for entry
        in existing.get(
            "candidate_files",
            []
        )
    }


    current_candidates = {
        entry[
            "candidate_id"
        ]:
        entry[
            "sha256"
        ]

        for entry
        in contract[
            "candidate_files"
        ]
    }


    if (
        existing_candidates
        !=
        current_candidates
    ):

        raise RuntimeError(
            "Existing frozen LR candidate-family "
            "contract references different model "
            "artifacts. It will NOT be overwritten."
        )


    print(
        "Existing candidate-family contract "
        "matches current model hashes."
    )


else:

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
# OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "FROZEN LR CANDIDATE FAMILY"
)

print(
    "===================================="
)


print(
    "Model family: LogisticRegression"
)

print(
    "Candidate count:",
    10
)

print(
    "Training scope: CYCLE24 TRAIN ONLY"
)

print(
    "Features:",
    25
)

print(
    "Candidates converged:",
    10
)

print(
    "Convergence warnings:",
    0
)

print(
    "FutureWarnings:",
    total_future_warnings,
    "(documented)"
)

print(
    "Unexpected warnings:",
    0
)

print(
    "Reproducibility:",
    "PASS"
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
    "TRAIN used:",
    True
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
    "MODEL CHAPTER 3.6 COMPLETE — LR FAMILY FROZEN"
)

print(
    "===================================="
)