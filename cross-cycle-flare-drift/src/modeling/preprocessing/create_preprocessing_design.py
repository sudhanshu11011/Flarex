from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


# ============================================================
# PATHS
# ============================================================

FEATURE_CONTRACT_FILE = Path(
    "artifacts/preprocessing/contracts/model_feature_contract.json"
)

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/cycle24_split_contract.json"
)

PRE_MODEL_GATE_FILE = Path(
    "data_manifests/pre_model_gate.json"
)

OUTPUT_FILE = Path(
    "artifacts/preprocessing/contracts/preprocessing_design.json"
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


# ============================================================
# VERIFY INPUTS
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 2.2 — PREPROCESSING DESIGN"
)

print(
    "===================================="
)


for path in [
    FEATURE_CONTRACT_FILE,
    SPLIT_CONTRACT_FILE,
    PRE_MODEL_GATE_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


with FEATURE_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    feature_contract = json.load(file)


with SPLIT_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    split_contract = json.load(file)


with PRE_MODEL_GATE_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    gate = json.load(file)


# ============================================================
# VERIFY UPSTREAM STATE
# ============================================================

if feature_contract.get(
    "status"
) != "VERIFIED":

    raise RuntimeError(
        "Feature contract is not VERIFIED."
    )


if split_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 split is not FROZEN."
    )


if gate.get(
    "status"
) != "OPEN_FOR_MODELING":

    raise RuntimeError(
        "Pre-model gate is not open."
    )


features = feature_contract.get(
    "feature_order",
    []
)


if len(features) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen model features."
    )


print(
    "Feature contract: VERIFIED"
)

print(
    "Cycle24 split: FROZEN"
)

print(
    "Pre-model gate: OPEN"
)

print(
    "Frozen features:",
    len(features)
)


# ============================================================
# PREPROCESSING DESIGN
# ============================================================

design = {

    "contract_name":
        "Model Preprocessing Design",

    "version":
        "1.0",

    "status":
        "DESIGN_FROZEN",

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "feature_contract_version":
        feature_contract.get(
            "version"
        ),

    "split_version":
        split_contract.get(
            "split_version"
        ),

    "feature_count":
        25,

    "feature_order":
        features,

    # --------------------------------------------------------
    # GLOBAL RULES
    # --------------------------------------------------------

    "global_rules": {

        "missing_value_imputation":
            False,

        "reason_no_imputation":
            (
                "Frozen Dataset v2 and all Cycle24 splits "
                "contain no missing or non-finite model "
                "feature values."
            ),

        "feature_selection":
            False,

        "pca":
            False,

        "winsorization":
            False,

        "outlier_clipping":
            False,

        "log_transformation":
            False,

        "polynomial_features":
            False,

        "interaction_feature_engineering":
            False,

        "feature_order_fixed":
            True,

        "preserve_original_physical_features":
            True,
    },

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    "logistic_regression": {

        "preprocessor":
            "StandardScaler",

        "with_mean":
            True,

        "with_std":
            True,

        "fit_scope":
            "TRAIN_ONLY",

        "validation_policy":
            "TRANSFORM_ONLY",

        "holdout_policy":
            (
                "TRANSFORM_ONLY_AFTER_MODEL_"
                "DEVELOPMENT_PIPELINE_IS_FROZEN"
            ),

        "cycle25_policy":
            (
                "TRANSFORM_ONLY_AFTER_FULL_"
                "CYCLE24_PIPELINE_IS_FROZEN"
            ),

        "reason":
            (
                "Logistic Regression is scale-sensitive, "
                "especially under regularization. "
                "StandardScaler places frozen features on "
                "comparable numerical scales without "
                "changing feature identity."
            ),
    },

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    "random_forest": {

        "preprocessor":
            "IDENTITY",

        "scaling":
            False,

        "fit_scope":
            "NONE",

        "validation_policy":
            "RAW_FROZEN_FEATURES",

        "holdout_policy":
            "RAW_FROZEN_FEATURES",

        "cycle25_policy":
            "RAW_FROZEN_FEATURES",

        "reason":
            (
                "Tree-based split decisions are invariant "
                "to monotonic feature scaling. Preserving "
                "raw feature values also simplifies "
                "physical interpretation."
            ),
    },

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    "xgboost": {

        "preprocessor":
            "IDENTITY",

        "scaling":
            False,

        "fit_scope":
            "NONE",

        "validation_policy":
            "RAW_FROZEN_FEATURES",

        "holdout_policy":
            "RAW_FROZEN_FEATURES",

        "cycle25_policy":
            "RAW_FROZEN_FEATURES",

        "reason":
            (
                "Gradient-boosted decision trees do not "
                "require feature standardization for "
                "tree split construction."
            ),
    },

    # --------------------------------------------------------
    # DATA ISOLATION
    # --------------------------------------------------------

    "data_isolation": {

        "train":
            {
                "may_fit_preprocessing":
                    True,

                "may_fit_models":
                    True,
            },

        "validation":
            {
                "may_fit_preprocessing":
                    False,

                "may_transform":
                    True,

                "may_tune_models":
                    True,
            },

        "holdout":
            {
                "may_fit_preprocessing":
                    False,

                "may_tune_models":
                    False,

                "may_select_threshold":
                    False,

                "use_before_pipeline_freeze":
                    False,
            },

        "cycle25":
            {
                "may_fit_preprocessing":
                    False,

                "may_fit_models":
                    False,

                "may_tune_models":
                    False,

                "may_select_threshold":
                    False,

                "use_before_full_pipeline_freeze":
                    False,
            },
    },

    # --------------------------------------------------------
    # CLASS IMBALANCE
    # --------------------------------------------------------

    "class_imbalance_policy": {

        "handled_in_preprocessing":
            False,

        "reason":
            (
                "Class weighting and model-specific imbalance "
                "parameters belong to the model-training and "
                "validation stage rather than the feature "
                "preprocessing stage."
            ),
    },

    # --------------------------------------------------------
    # SHAP / INTERPRETABILITY
    # --------------------------------------------------------

    "interpretability_policy": {

        "preserve_original_feature_names":
            True,

        "preserve_25_feature_identity":
            True,

        "no_pca":
            True,

        "reason":
            (
                "Cross-cycle explanation-drift analysis "
                "requires attribution to the same physical "
                "SHARP/Lorentz feature definitions."
            ),
    },

    # --------------------------------------------------------
    # CHANGE CONTROL
    # --------------------------------------------------------

    "change_control": {

        "performance_driven_preprocessing_changes":
            False,

        "cycle25_driven_changes":
            False,

        "distribution_audit_may_trigger_change":
            (
                "Only if a genuine numerical or data-quality "
                "failure is demonstrated before model "
                "training. Any change requires a new "
                "preprocessing-design version."
            ),
    },

    # --------------------------------------------------------
    # PROVENANCE
    # --------------------------------------------------------

    "provenance": {

        "feature_contract_file":
            str(
                FEATURE_CONTRACT_FILE
            ),

        "feature_contract_sha256":
            sha256_file(
                FEATURE_CONTRACT_FILE
            ),

        "split_contract_file":
            str(
                SPLIT_CONTRACT_FILE
            ),

        "split_contract_sha256":
            sha256_file(
                SPLIT_CONTRACT_FILE
            ),

        "pre_model_gate_file":
            str(
                PRE_MODEL_GATE_FILE
            ),

        "pre_model_gate_sha256":
            sha256_file(
                PRE_MODEL_GATE_FILE
            ),
    },
}


# ============================================================
# SAFE WRITE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


if OUTPUT_FILE.exists():

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        existing = json.load(file)


    if existing.get(
        "version"
    ) != "1.0":

        raise RuntimeError(
            "Existing preprocessing design has "
            "a different version."
        )


    if existing.get(
        "feature_order"
    ) != features:

        raise RuntimeError(
            "Existing preprocessing design belongs "
            "to a different feature contract."
        )


    print(
        "Existing preprocessing design verified."
    )


else:

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            design,
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
    "MODEL-SPECIFIC PREPROCESSING"
)

print(
    "===================================="
)


print(
    "Logistic Regression:"
)

print(
    " StandardScaler"
)

print(
    " Fit on: TRAIN only"
)


print(
    "\nRandom Forest:"
)

print(
    " Identity / no scaling"
)


print(
    "\nXGBoost:"
)

print(
    " Identity / no scaling"
)


print(
    "\n===================================="
)

print(
    "GLOBAL PREPROCESSING RULES"
)

print(
    "===================================="
)


print(
    "Imputation: False"
)

print(
    "PCA: False"
)

print(
    "Feature selection: False"
)

print(
    "Outlier clipping: False"
)

print(
    "Log transform: False"
)

print(
    "Frozen feature count: 25"
)

print(
    "Cycle25 used: False"
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
    "TRAIN       → fit preprocessing"
)

print(
    "VALIDATION  → transform only"
)

print(
    "HOLDOUT     → untouched for development"
)

print(
    "CYCLE25     → untouched until full freeze"
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\nMODEL CHAPTER 2.2 COMPLETE."
)