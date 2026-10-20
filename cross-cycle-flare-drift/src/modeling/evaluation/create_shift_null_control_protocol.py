from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

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

SELECTED_MODELS_CONTRACT_FILE = Path(
    "artifacts/models/selection/contracts/"
    "selected_models_contract.json"
)

CYCLE24_RESULTS_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "holdout_results_contract.json"
)

CYCLE25_RESULTS_CONTRACT_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_results_contract.json"
)

CYCLE25_PROTOCOL_FILE = Path(
    "artifacts/models/evaluation/contracts/"
    "cycle25_evaluation_protocol.json"
)

CYCLE25_BLOCKS_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_temporal_blocks.csv"
)

FROZEN_MODEL_REGISTRY_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_frozen_model_registry.csv"
)


# Future data inputs.
# These files are NOT opened in Chapter 9.1.

CYCLE24_TRAIN_FILE = Path(
    "artifacts/splitting/data/"
    "cycle24_train.parquet"
)

CYCLE24_HOLDOUT_FILE = Path(
    "artifacts/splitting/data/"
    "cycle24_holdout.parquet"
)

CYCLE25_DATASET_FILE = Path(
    "data/final/dataset_v2/"
    "cycle25_dataset_v2.parquet"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_PROTOCOL_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "shift_null_control_protocol.json"
)

OUTPUT_FEATURE_REGISTRY_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "shift_feature_registry.csv"
)

OUTPUT_ANALYSIS_REGISTRY_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "shift_analysis_registry.csv"
)


# ============================================================
# FROZEN DATA EXPECTATIONS
# ============================================================

EXPECTED_CYCLE24_TRAIN_SHA256 = (
    "4801a6e3b11d39cef8d4b0ef0eea4ae1"
    "67c27862431662c3f43ff55e4c433def"
)

EXPECTED_CYCLE24_HOLDOUT_SHA256 = (
    "8abb3929febcde7d778c4781ac890ff5e"
    "658933c2ef2a06aa19da085e7e255fc"
)

EXPECTED_CYCLE25_SHA256 = (
    "0bd6a364e152685d13c7eb955da074a4"
    "b9f019f1851c6b5dafebf295c406cd4d"
)

EXPECTED_CYCLE24_TRAIN_ROWS = 62064
EXPECTED_CYCLE24_HOLDOUT_ROWS = 13072
EXPECTED_CYCLE25_ROWS = 79361

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TARGET_COLUMN = "target"
TIME_COLUMN = "observation_time_utc"


# ============================================================
# FROZEN FEATURES
# ============================================================

SHARP_FEATURES = [
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
]

LORENTZ_FEATURES = [
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
]

FEATURES = (
    SHARP_FEATURES
    +
    LORENTZ_FEATURES
)


# ============================================================
# FROZEN CYCLE25 SCOPES
# ============================================================

PRIMARY_CYCLE25_BLOCKS = [
    "C25_B1_2020_2021",
    "C25_B2_2022_2023",
    "C25_B3_2024_2025",
]

POOLED_CYCLE25_SCOPE = (
    "C25_ALL_2020_2025"
)


# ============================================================
# RANDOM SEEDS
# ============================================================

SEEDS = {
    "distribution_group_bootstrap": 29021,
    "prevalence_group_bootstrap": 29031,
    "domain_classifier_cv": 29041,
    "shap_background_group_sample": 29061,
    "shap_null_pseudo_sample_a": 29062,
    "shap_null_pseudo_sample_b": 29063,
}


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


def contains_exact_list(
    obj,
    expected,
):

    if isinstance(obj, list):

        if obj == expected:
            return True

        return any(
            contains_exact_list(
                item,
                expected,
            )
            for item in obj
        )

    if isinstance(obj, dict):

        return any(
            contains_exact_list(
                value,
                expected,
            )
            for value in obj.values()
        )

    return False


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 9.1 — FREEZE SHIFT + SHAP NULL-CONTROL PROTOCOL"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED UPSTREAM ARTIFACTS
# ============================================================

required_files = [
    FEATURE_CONTRACT_FILE,
    PREPROCESSING_CONTRACT_FILE,
    SPLIT_CONTRACT_FILE,
    SELECTED_MODELS_CONTRACT_FILE,
    CYCLE24_RESULTS_CONTRACT_FILE,
    CYCLE25_RESULTS_CONTRACT_FILE,
    CYCLE25_PROTOCOL_FILE,
    CYCLE25_BLOCKS_FILE,
    FROZEN_MODEL_REGISTRY_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required upstream "
            f"artifact: {path}"
        )


# ============================================================
# FUTURE RAW/DERIVED DATA EXISTENCE ONLY
# ============================================================

for path in [
    CYCLE24_TRAIN_FILE,
    CYCLE24_HOLDOUT_FILE,
    CYCLE25_DATASET_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required frozen dataset "
            f"not found: {path}"
        )


print(
    "Frozen analysis datasets exist:",
    True
)

print(
    "Dataset content read in 9.1:",
    False
)


# ============================================================
# PROTECT OUTPUTS
# ============================================================

for path in [
    OUTPUT_PROTOCOL_FILE,
    OUTPUT_FEATURE_REGISTRY_FILE,
    OUTPUT_ANALYSIS_REGISTRY_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Protocol output already exists: "
            f"{path}\n"
            "Refusing to overwrite frozen "
            "Chapter 9.1 design."
        )


# ============================================================
# LOAD CONTRACTS ONLY
# ============================================================

feature_contract = load_json(
    FEATURE_CONTRACT_FILE
)

preprocessing_contract = load_json(
    PREPROCESSING_CONTRACT_FILE
)

split_contract = load_json(
    SPLIT_CONTRACT_FILE
)

selected_models_contract = load_json(
    SELECTED_MODELS_CONTRACT_FILE
)

cycle24_results_contract = load_json(
    CYCLE24_RESULTS_CONTRACT_FILE
)

cycle25_results_contract = load_json(
    CYCLE25_RESULTS_CONTRACT_FILE
)

cycle25_protocol = load_json(
    CYCLE25_PROTOCOL_FILE
)


# ============================================================
# VERIFY UPSTREAM FREEZE STATE
# ============================================================

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


if selected_models_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Selected-model contract is not FROZEN."
    )


if cycle24_results_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 holdout results are not FROZEN."
    )


if cycle25_results_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle25 predictive results are not FROZEN."
    )


if cycle25_results_contract.get(
    "evaluation_stage"
) != "CYCLE25_PREDICTIVE_EVALUATION_COMPLETE":

    raise RuntimeError(
        "Cycle25 predictive evaluation "
        "is not marked complete."
    )


if cycle25_protocol.get(
    "status"
) != "FROZEN_BEFORE_CYCLE25_EVALUATION":

    raise RuntimeError(
        "Cycle25 temporal protocol "
        "is not frozen."
    )


print(
    "Feature contract:",
    "VERIFIED"
)

print(
    "Preprocessing:",
    "FROZEN"
)

print(
    "Cycle24 split:",
    "FROZEN"
)

print(
    "Selected models:",
    "FROZEN"
)

print(
    "Cycle24 predictive baseline:",
    "FROZEN"
)

print(
    "Cycle25 predictive evaluation:",
    "FROZEN"
)


# ============================================================
# FEATURE ORDER
# ============================================================

if len(
    FEATURES
) != 25:

    raise RuntimeError(
        "Expected exactly 25 features."
    )


if len(
    set(FEATURES)
) != 25:

    raise RuntimeError(
        "Duplicate feature detected."
    )


if not contains_exact_list(
    feature_contract,
    FEATURES,
):

    raise RuntimeError(
        "Exact frozen feature order "
        "not found in feature contract."
    )


print(
    "Frozen 25-feature order:",
    "PASS"
)


# ============================================================
# VERIFY CYCLE25 TEMPORAL MANIFEST
# ============================================================

blocks = pd.read_csv(
    CYCLE25_BLOCKS_FILE,
    keep_default_na=False,
)


if set(
    blocks[
        "block_id"
    ].astype(str)
) != set(
    PRIMARY_CYCLE25_BLOCKS
    +
    [
        POOLED_CYCLE25_SCOPE
    ]
):

    raise RuntimeError(
        "Cycle25 temporal scope "
        "registry mismatch."
    )


expected_blocks_hash = (
    cycle25_protocol[
        "artifacts"
    ][
        "cycle25_temporal_blocks"
    ][
        "sha256"
    ]
)


actual_blocks_hash = sha256_file(
    CYCLE25_BLOCKS_FILE
)


if (
    actual_blocks_hash
    !=
    expected_blocks_hash
):

    raise RuntimeError(
        "Cycle25 temporal-block "
        "manifest hash mismatch."
    )


print(
    "Cycle25 temporal blocks:",
    "PASS"
)


# ============================================================
# VERIFY FROZEN MODELS
# ============================================================

model_registry = pd.read_csv(
    FROZEN_MODEL_REGISTRY_FILE,
    keep_default_na=False,
)


expected_models = {
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
}


if len(
    model_registry
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen models."
    )


if set(
    model_registry[
        "candidate_id"
    ].astype(str)
) != expected_models:

    raise RuntimeError(
        "Frozen model identity mismatch."
    )


print(
    "Frozen model registry:",
    "PASS"
)


# ============================================================
# FEATURE REGISTRY
# ============================================================

feature_rows = []


for index, feature in enumerate(
    FEATURES,
    start=1,
):

    feature_family = (
        "SHARP_DIRECT"
        if feature
        in SHARP_FEATURES
        else
        "LORENTZ"
    )


    feature_rows.append(
        {
            "feature_index":
                index,

            "feature_name":
                feature,

            "feature_family":
                feature_family,

            "marginal_shift_audit":
                True,

            "class_conditional_shift_audit":
                True,

            "missingness_audit":
                True,

            "correlation_grouping":
                True,

            "shap_null_analysis":
                True,

            "feature_selection_allowed":
                False,
        }
    )


feature_registry = pd.DataFrame(
    feature_rows
)


# ============================================================
# ANALYSIS REGISTRY
# ============================================================

analysis_registry = pd.DataFrame(
    [
        {
            "analysis_id":
                "SHIFT_MARGINAL_OVERALL",

            "chapter":
                "9.2",

            "primary":
                True,

            "population":
                "C24_HOLDOUT_VS_C25_SCOPES",

            "unit":
                "ENTITY_GROUP",

            "outcome_dependent":
                False,
        },

        {
            "analysis_id":
                "SHIFT_MARGINAL_CLASS_0",

            "chapter":
                "9.2",

            "primary":
                False,

            "population":
                "TARGET_0_ONLY",

            "unit":
                "ENTITY_GROUP",

            "outcome_dependent":
                True,
        },

        {
            "analysis_id":
                "SHIFT_MARGINAL_CLASS_1",

            "chapter":
                "9.2",

            "primary":
                False,

            "population":
                "TARGET_1_ONLY",

            "unit":
                "ENTITY_GROUP",

            "outcome_dependent":
                True,
        },

        {
            "analysis_id":
                "LABEL_PREVALENCE",

            "chapter":
                "9.3",

            "primary":
                True,

            "population":
                "ALL_SCOPES",

            "unit":
                "ENTITY_GROUP",

            "outcome_dependent":
                True,
        },

        {
            "analysis_id":
                "ACTIVE_REGION_POPULATION",

            "chapter":
                "9.3",

            "primary":
                True,

            "population":
                "ALL_SCOPES",

            "unit":
                "ENTITY_GROUP",

            "outcome_dependent":
                False,
        },

        {
            "analysis_id":
                "JOINT_DOMAIN_CLASSIFIER",

            "chapter":
                "9.4",

            "primary":
                True,

            "population":
                "C24_HOLDOUT_VS_C25",

            "unit":
                "ENTITY_GROUP",

            "outcome_dependent":
                False,
        },

        {
            "analysis_id":
                "COMMON_SUPPORT",

            "chapter":
                "9.4",

            "primary":
                False,

            "population":
                "C24_HOLDOUT_VS_C25",

            "unit":
                "ENTITY_GROUP",

            "outcome_dependent":
                False,
        },

        {
            "analysis_id":
                "CORRELATED_FEATURE_GROUPS",

            "chapter":
                "9.5",

            "primary":
                True,

            "population":
                "C24_TRAIN_ONLY",

            "unit":
                "FEATURE",

            "outcome_dependent":
                False,
        },

        {
            "analysis_id":
                "WITHIN_C24_SHAP_NULL_OVERALL",

            "chapter":
                "9.6",

            "primary":
                True,

            "population":
                "C24_HOLDOUT",

            "unit":
                "ENTITY_GROUP",

            "outcome_dependent":
                False,
        },

        {
            "analysis_id":
                "WITHIN_C24_SHAP_NULL_CLASS_0",

            "chapter":
                "9.6",

            "primary":
                False,

            "population":
                "C24_HOLDOUT_TARGET_0",

            "unit":
                "ENTITY_GROUP",

            "outcome_dependent":
                True,
        },

        {
            "analysis_id":
                "WITHIN_C24_SHAP_NULL_CLASS_1",

            "chapter":
                "9.6",

            "primary":
                False,

            "population":
                "C24_HOLDOUT_TARGET_1",

            "unit":
                "ENTITY_GROUP",

            "outcome_dependent":
                True,
        },
    ]
)


# ============================================================
# SAVE REGISTRIES
# ============================================================

OUTPUT_FEATURE_REGISTRY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_ANALYSIS_REGISTRY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PROTOCOL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


feature_registry.to_csv(
    OUTPUT_FEATURE_REGISTRY_FILE,
    index=False,
)


analysis_registry.to_csv(
    OUTPUT_ANALYSIS_REGISTRY_FILE,
    index=False,
)


feature_registry_hash = sha256_file(
    OUTPUT_FEATURE_REGISTRY_FILE
)


analysis_registry_hash = sha256_file(
    OUTPUT_ANALYSIS_REGISTRY_FILE
)


# ============================================================
# FROZEN PROTOCOL
# ============================================================

protocol = {

    "contract_name":
        (
            "Distribution Shift and "
            "Within-Cycle24 SHAP Null-Control Protocol"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_BEFORE_SHIFT_AND_SHAP_ANALYSIS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_9",

    # --------------------------------------------------------
    # DESIGN TIMING
    # --------------------------------------------------------

    "design_timing":
        {
            "cycle24_predictive_results_exist":
                True,

            "cycle25_predictive_results_exist":
                True,

            "distribution_shift_results_seen":
                False,

            "within_cycle24_shap_null_seen":
                False,

            "cross_cycle_shap_results_seen":
                False,

            "important_note":
                (
                    "This protocol is frozen after "
                    "predictive evaluation but before "
                    "distribution-shift and SHAP "
                    "analyses. Predictive outcomes must "
                    "not be used to alter the shift or "
                    "SHAP-control methodology."
                ),
        },

    # --------------------------------------------------------
    # DATA POPULATIONS
    # --------------------------------------------------------

    "data_populations":
        {
            "distribution_reference":
                "CYCLE24_HOLDOUT",

            "distribution_transfer":
                "CYCLE25",

            "shap_background_source":
                "CYCLE24_TRAIN_ONLY",

            "shap_null_population":
                "CYCLE24_HOLDOUT",

            "cycle25_primary_blocks":
                PRIMARY_CYCLE25_BLOCKS,

            "cycle25_secondary_pooled_scope":
                POOLED_CYCLE25_SCOPE,

            "entity_group_column":
                ENTITY_GROUP_COLUMN,

            "target_column":
                TARGET_COLUMN,

            "time_column":
                TIME_COLUMN,
        },

    # --------------------------------------------------------
    # DATASET EXPECTATIONS
    # --------------------------------------------------------

    "dataset_expectations":
        {
            "cycle24_train":
                {
                    "file":
                        str(
                            CYCLE24_TRAIN_FILE
                        ),

                    "sha256":
                        EXPECTED_CYCLE24_TRAIN_SHA256,

                    "rows":
                        EXPECTED_CYCLE24_TRAIN_ROWS,
                },

            "cycle24_holdout":
                {
                    "file":
                        str(
                            CYCLE24_HOLDOUT_FILE
                        ),

                    "sha256":
                        EXPECTED_CYCLE24_HOLDOUT_SHA256,

                    "rows":
                        EXPECTED_CYCLE24_HOLDOUT_ROWS,
                },

            "cycle25":
                {
                    "file":
                        str(
                            CYCLE25_DATASET_FILE
                        ),

                    "sha256":
                        EXPECTED_CYCLE25_SHA256,

                    "rows":
                        EXPECTED_CYCLE25_ROWS,
                },
        },

    # --------------------------------------------------------
    # FEATURE CONTRACT
    # --------------------------------------------------------

    "feature_contract":
        {
            "feature_count":
                25,

            "feature_order":
                FEATURES,

            "feature_selection_allowed":
                False,

            "cycle25_driven_feature_selection":
                False,
        },

    # --------------------------------------------------------
    # 9.2 MARGINAL DISTRIBUTION SHIFT
    # --------------------------------------------------------

    "marginal_distribution_shift":
        {
            "comparison":
                (
                    "Cycle24 holdout versus each "
                    "Cycle25 temporal block and "
                    "Cycle25 pooled scope"
                ),

            "strata":
                [
                    "OVERALL",
                    "TARGET_0",
                    "TARGET_1",
                ],

            "metrics":
                {
                    "ks_statistic":
                        (
                            "Two-sample Kolmogorov-"
                            "Smirnov D statistic. "
                            "Used as an effect-size-like "
                            "distribution discrepancy; "
                            "naive IID p-values are not "
                            "used for scientific claims."
                        ),

                    "wasserstein_distance":
                        (
                            "1D Wasserstein distance "
                            "on the frozen raw feature."
                        ),

                    "normalized_wasserstein":
                        (
                            "Wasserstein distance divided "
                            "by Cycle24-holdout IQR. "
                            "If IQR is zero, record NA "
                            "and flag the feature."
                        ),

                    "median_shift_iqr_units":
                        (
                            "(median_C25 - median_C24) "
                            "/ Cycle24-holdout IQR."
                        ),
                },

            "uncertainty":
                {
                    "unit":
                        "ENTITY_GROUP",

                    "replicates":
                        2000,

                    "seed":
                        SEEDS[
                            "distribution_group_bootstrap"
                        ],

                    "ci":
                        "95% percentile",

                    "same_group_draws_across_features":
                        True,
                },

            "binary_drift_threshold":
                None,

            "interpretation":
                (
                    "No single marginal metric or "
                    "arbitrary threshold is sufficient "
                    "to declare explanation drift."
                ),
        },

    # --------------------------------------------------------
    # MISSINGNESS
    # --------------------------------------------------------

    "missingness_audit":
        {
            "features":
                FEATURES,

            "measure":
                [
                    "missing_fraction",
                    "nonfinite_fraction",
                ],

            "comparison_scopes":
                [
                    "C24_HOLDOUT",
                    *PRIMARY_CYCLE25_BLOCKS,
                    POOLED_CYCLE25_SCOPE,
                ],

            "expected_primary_dataset_state":
                (
                    "No missing/nonfinite model "
                    "features after frozen data "
                    "pipeline; any violation is "
                    "reported, not silently imputed."
                ),
        },

    # --------------------------------------------------------
    # 9.3 PREVALENCE + AR POPULATION
    # --------------------------------------------------------

    "label_prevalence":
        {
            "scopes":
                [
                    "C24_HOLDOUT",
                    *PRIMARY_CYCLE25_BLOCKS,
                    POOLED_CYCLE25_SCOPE,
                ],

            "metrics":
                [
                    "row_count",
                    "positive_count",
                    "negative_count",
                    "positive_prevalence",
                ],

            "uncertainty":
                {
                    "unit":
                        "ENTITY_GROUP",

                    "replicates":
                        2000,

                    "seed":
                        SEEDS[
                            "prevalence_group_bootstrap"
                        ],

                    "ci":
                        "95% percentile",
                },

            "prevalence_difference_is_real_shift":
                (
                    "Report observed prevalence "
                    "difference; do not tune it away."
                ),
        },

    "active_region_population":
        {
            "unit":
                "ENTITY_GROUP",

            "metrics":
                [
                    "entity_group_count",
                    "rows_per_group_median",
                    "rows_per_group_iqr",
                    "positive_groups",
                    "positive_group_fraction",
                    "group_positive_rate_distribution",
                ],

            "cycle24_cycle25_group_overlap_expected":
                0,

            "interpretation":
                (
                    "Population composition differences "
                    "are potential confounders for "
                    "cross-cycle SHAP comparisons."
                ),
        },

    # --------------------------------------------------------
    # 9.4 JOINT SHIFT / COMMON SUPPORT
    # --------------------------------------------------------

    "joint_distribution_shift":
        {
            "primary_method":
                "DOMAIN_CLASSIFIER",

            "features":
                FEATURES,

            "domain_target":
                {
                    "CYCLE24_HOLDOUT":
                        0,

                    "CYCLE25":
                        1,
                },

            "preprocessing":
                (
                    "Use the already-frozen "
                    "Cycle24-TRAIN StandardScaler "
                    "for both domains. Do not refit "
                    "scaling on Cycle24 holdout or "
                    "Cycle25."
                ),

            "classifier":
                {
                    "type":
                        "LogisticRegression",

                    "penalty":
                        "l2",

                    "C":
                        1.0,

                    "class_weight":
                        "balanced",

                    "solver":
                        "lbfgs",

                    "max_iter":
                        5000,

                    "random_state":
                        SEEDS[
                            "domain_classifier_cv"
                        ],
                },

            "validation":
                {
                    "type":
                        "StratifiedGroupKFold",

                    "n_splits":
                        5,

                    "shuffle":
                        True,

                    "random_state":
                        SEEDS[
                            "domain_classifier_cv"
                        ],

                    "group":
                        (
                            "cycle-prefixed "
                            "ENTITY_GROUP_ID"
                        ),
                },

            "metrics":
                [
                    "ROC_AUC",
                    "balanced_accuracy",
                ],

            "purpose":
                (
                    "Quantify multivariate "
                    "domain separability only. "
                    "This classifier is not a "
                    "solar-flare predictor."
                ),
        },

    "common_support_reweighting":
        {
            "propensity":
                (
                    "OOF probability that a row "
                    "belongs to Cycle25 from the "
                    "frozen domain-classifier design."
                ),

            "common_support_interval":
                (
                    "Intersection of the Cycle24 and "
                    "Cycle25 OOF propensity-score "
                    "1st-to-99th percentile intervals."
                ),

            "primary_support_rule":
                "retain rows inside common support",

            "cycle25_to_cycle24_weight":
                (
                    "((1-p)/p) * "
                    "(N_C25/N_C24)"
                ),

            "weight_clipping":
                (
                    "After common-support restriction, "
                    "clip Cycle25 raw weights to their "
                    "1st and 99th percentiles."
                ),

            "weight_normalization":
                (
                    "Normalize retained Cycle25 "
                    "weights to mean 1."
                ),

            "use":
                (
                    "Sensitivity analysis only; "
                    "does not replace the primary "
                    "unweighted cross-cycle analysis."
                ),
        },

    # --------------------------------------------------------
    # 9.5 CORRELATED FEATURES
    # --------------------------------------------------------

    "correlated_feature_control":
        {
            "source_population":
                "CYCLE24_TRAIN_ONLY",

            "correlation":
                "Spearman",

            "edge_rule":
                "absolute_rho >= 0.80",

            "grouping":
                (
                    "Construct an undirected feature "
                    "graph and define connected "
                    "components as correlated-feature "
                    "groups. Singleton features remain "
                    "single-member groups."
                ),

            "cycle25_used_to_form_groups":
                False,

            "group_level_shap_magnitude":
                (
                    "Sum mean absolute SHAP magnitude "
                    "of member features."
                ),

            "purpose":
                (
                    "Check whether apparent feature-level "
                    "explanation drift persists after "
                    "correlated SHARP variables are "
                    "interpreted jointly."
                ),
        },

    # --------------------------------------------------------
    # 9.6 SHAP BACKGROUND
    # --------------------------------------------------------

    "shap_background":
        {
            "source":
                "CYCLE24_TRAIN_ONLY",

            "size":
                512,

            "sampling_unit":
                "ENTITY_GROUP",

            "selection":
                (
                    "Sort unique Cycle24-TRAIN "
                    "ENTITY_GROUP_ID values, sample "
                    "512 groups without replacement "
                    "using the frozen seed, then select "
                    "the temporal-median row from each "
                    "chosen group after sorting by "
                    "observation_time_utc."
                ),

            "seed":
                SEEDS[
                    "shap_background_group_sample"
                ],

            "cycle25_used":
                False,

            "validation_used":
                False,

            "holdout_used":
                False,
        },

    # --------------------------------------------------------
    # MODEL-SPECIFIC SHAP
    # --------------------------------------------------------

    "shap_explainers":
        {
            "logistic":
                {
                    "candidate_id":
                        "LR_02_C0p01_BALANCED",

                    "explainer":
                        "shap.LinearExplainer",

                    "input_space":
                        (
                            "Frozen StandardScaler "
                            "output"
                        ),

                    "background":
                        "same frozen 512-row TRAIN background",

                    "comparison":
                        (
                            "Within this model only; "
                            "absolute SHAP magnitudes "
                            "are not compared across "
                            "different model families."
                        ),
                },

            "random_forest":
                {
                    "candidate_id":
                        "RF_04_D12_L5_BALSUB",

                    "explainer":
                        "shap.TreeExplainer",

                    "feature_perturbation":
                        "interventional",

                    "input_space":
                        "raw frozen 25 features",

                    "background":
                        "same frozen 512-row TRAIN background",
                },

            "xgboost":
                {
                    "candidate_id":
                        "XGB_02_LR0p03_D3_BAL",

                    "explainer":
                        "shap.TreeExplainer",

                    "feature_perturbation":
                        "interventional",

                    "input_space":
                        "raw frozen 25 features",

                    "background":
                        "same frozen 512-row TRAIN background",
                },
        },

    # --------------------------------------------------------
    # WITHIN-C24 SHAP NULL
    # --------------------------------------------------------

    "within_cycle24_shap_null":
        {
            "population":
                "CYCLE24_HOLDOUT",

            "unit":
                "ENTITY_GROUP",

            "replicates":
                2000,

            "pseudo_sample_A_seed":
                SEEDS[
                    "shap_null_pseudo_sample_a"
                ],

            "pseudo_sample_B_seed":
                SEEDS[
                    "shap_null_pseudo_sample_b"
                ],

            "sampling_rule":
                (
                    "For every replicate independently "
                    "draw G Cycle24-holdout ENTITY_GROUP "
                    "clusters with replacement for "
                    "pseudo-sample A and independently "
                    "draw G clusters for pseudo-sample B, "
                    "where G equals the number of "
                    "Cycle24-holdout groups."
                ),

            "same_group_draws_across_models":
                True,

            "strata":
                [
                    "OVERALL",
                    "TARGET_0",
                    "TARGET_1",
                ],

            "valid_replicate_rule":
                (
                    "Both pseudo-samples must contain "
                    "non-zero row weight for TARGET_0 "
                    "and TARGET_1 so overall and "
                    "class-conditional summaries can "
                    "share the same replicate."
                ),

            "shap_summary_vector":
                [
                    "mean_absolute_shap_per_feature",
                    "mean_signed_shap_per_feature",
                ],

            "null_metrics":
                {
                    "spearman_rho":
                        "similarity",

                    "kendall_tau_b":
                        "similarity",

                    "top5_overlap_fraction":
                        "similarity",

                    "top10_overlap_fraction":
                        "similarity",

                    "normalized_l1_mean_abs":
                        "distance",

                    "cosine_distance_mean_abs":
                        "distance",

                    "signed_mean_cosine_similarity":
                        "similarity",

                    "signed_direction_agreement_fraction":
                        "similarity",
                },

            "null_thresholds":
                {
                    "similarity_metrics":
                        "2.5th percentile",

                    "distance_metrics":
                        "97.5th percentile",

                    "confidence_level":
                        0.95,
                },

            "role":
                (
                    "Defines how much SHAP rank, "
                    "magnitude and direction variation "
                    "can occur naturally within Cycle24 "
                    "under ENTITY_GROUP resampling."
                ),
        },

    # --------------------------------------------------------
    # LATER CROSS-CYCLE SHAP DECISION RULE
    # --------------------------------------------------------

    "future_explanation_drift_rule":
        {
            "cross_cycle_analysis_not_performed_in_chapter_9_1":
                True,

            "raw_shap_difference_sufficient_for_drift":
                False,

            "required_evidence":
                [
                    (
                        "Cross-cycle SHAP difference "
                        "exceeds the corresponding "
                        "within-Cycle24 null threshold."
                    ),

                    (
                        "Result is checked under "
                        "class-conditional analysis."
                    ),

                    (
                        "Result is checked under "
                        "common-support/reweighting "
                        "sensitivity."
                    ),

                    (
                        "Result is checked at correlated "
                        "feature-group level."
                    ),
                ],

            "featurewise_multiple_testing_if_used":
                {
                    "method":
                        "Benjamini-Hochberg FDR",

                    "q":
                        0.05,
                },

            "shap_is_causal":
                False,
        },

    # --------------------------------------------------------
    # STRICT PROHIBITIONS
    # --------------------------------------------------------

    "strict_prohibitions":
        [
            "retrain_flare_models",
            "replace_frozen_models",
            "change_frozen_thresholds",
            "change_25_feature_contract",
            "refit_predictive_preprocessing",
            "change_cycle25_temporal_blocks",
            "select_shift_metrics_after_viewing_shift_results",
            "select_shap_null_metrics_after_viewing_shap_results",
            "use_cycle25_to_create_correlated_feature_groups",
            "use_cycle25_to_select_shap_background",
            "declare_explanation_drift_from_raw_shap_difference_only",
            "interpret_shap_as_causal_solar_physics",
        ],

    # --------------------------------------------------------
    # ARTIFACTS
    # --------------------------------------------------------

    "artifacts":
        {
            "feature_registry":
                {
                    "file":
                        str(
                            OUTPUT_FEATURE_REGISTRY_FILE
                        ),

                    "sha256":
                        feature_registry_hash,

                    "rows":
                        len(
                            feature_registry
                        ),
                },

            "analysis_registry":
                {
                    "file":
                        str(
                            OUTPUT_ANALYSIS_REGISTRY_FILE
                        ),

                    "sha256":
                        analysis_registry_hash,

                    "rows":
                        len(
                            analysis_registry
                        ),
                },
        },

    # --------------------------------------------------------
    # PROVENANCE
    # --------------------------------------------------------

    "provenance":
        {
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

            "selected_models_contract":
                {
                    "file":
                        str(
                            SELECTED_MODELS_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            SELECTED_MODELS_CONTRACT_FILE
                        ),
                },

            "cycle24_results_contract":
                {
                    "file":
                        str(
                            CYCLE24_RESULTS_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CYCLE24_RESULTS_CONTRACT_FILE
                        ),
                },

            "cycle25_results_contract":
                {
                    "file":
                        str(
                            CYCLE25_RESULTS_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CYCLE25_RESULTS_CONTRACT_FILE
                        ),
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

            "cycle25_temporal_blocks":
                {
                    "file":
                        str(
                            CYCLE25_BLOCKS_FILE
                        ),

                    "sha256":
                        actual_blocks_hash,
                },

            "frozen_model_registry":
                {
                    "file":
                        str(
                            FROZEN_MODEL_REGISTRY_FILE
                        ),

                    "sha256":
                        sha256_file(
                            FROZEN_MODEL_REGISTRY_FILE
                        ),
                },
        },

    # --------------------------------------------------------
    # 9.1 ISOLATION
    # --------------------------------------------------------

    "chapter_9_1_isolation":
        {
            "cycle24_train_content_read":
                False,

            "cycle24_holdout_content_read":
                False,

            "cycle25_content_read":
                False,

            "distribution_shift_metrics_computed":
                False,

            "domain_classifier_trained":
                False,

            "correlation_groups_computed":
                False,

            "shap_values_computed":
                False,

            "shap_null_computed":
                False,

            "cross_cycle_shap_computed":
                False,
        },

    "immutability_rule":
        (
            "After Chapter 9.1 is frozen, the "
            "distribution-shift metrics, domain-"
            "classifier design, common-support rule, "
            "correlated-feature grouping rule, SHAP "
            "background selection, within-Cycle24 null "
            "sampling design, null metrics, and null "
            "threshold directions must not be changed "
            "in response to observed Cycle25 feature "
            "or SHAP results."
        ),
}


# ============================================================
# SAVE PROTOCOL
# ============================================================

with OUTPUT_PROTOCOL_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        protocol,
        file,
        indent=2,
    )


# ============================================================
# VERIFY SAVE
# ============================================================

saved = load_json(
    OUTPUT_PROTOCOL_FILE
)


if saved.get(
    "status"
) != "FROZEN_BEFORE_SHIFT_AND_SHAP_ANALYSIS":

    raise RuntimeError(
        "Chapter 9.1 protocol "
        "was not frozen."
    )


if saved[
    "chapter_9_1_isolation"
][
    "cycle25_content_read"
]:

    raise RuntimeError(
        "Chapter 9.1 incorrectly "
        "reports Cycle25 data access."
    )


if saved[
    "feature_contract"
][
    "feature_count"
] != 25:

    raise RuntimeError(
        "Feature count mismatch."
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "CHAPTER 9 CONTROL DESIGN"
)

print(
    "===================================="
)


print(
    "Frozen features:",
    25
)

print(
    "Distribution reference:",
    "CYCLE24_HOLDOUT"
)

print(
    "Transfer domain:",
    "CYCLE25"
)

print(
    "Cycle25 primary blocks:",
    3
)

print(
    "Marginal shift strata:",
    "OVERALL + TARGET_0 + TARGET_1"
)

print(
    "Distribution bootstrap:",
    "2000 ENTITY_GROUP replicates"
)

print(
    "Joint shift:",
    "5-fold group-safe domain classifier"
)

print(
    "Common-support sensitivity:",
    "FROZEN"
)

print(
    "Correlation grouping:",
    "|Spearman rho| >= 0.80 on C24 TRAIN"
)

print(
    "SHAP background:",
    "512 C24-TRAIN ENTITY_GROUP-balanced rows"
)

print(
    "Within-C24 SHAP null:",
    "2000 pseudo-sample comparisons"
)

print(
    "SHAP null strata:",
    "OVERALL + TARGET_0 + TARGET_1"
)


print(
    "\n===================================="
)

print(
    "CHAPTER 9.1 ISOLATION"
)

print(
    "===================================="
)


print(
    "Cycle24 TRAIN content read:",
    False
)

print(
    "Cycle24 HOLDOUT content read:",
    False
)

print(
    "Cycle25 content read:",
    False
)

print(
    "Distribution metrics computed:",
    False
)

print(
    "Domain classifier trained:",
    False
)

print(
    "Correlation groups computed:",
    False
)

print(
    "SHAP values computed:",
    False
)

print(
    "SHAP null computed:",
    False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FEATURE_REGISTRY_FILE
)

print(
    OUTPUT_ANALYSIS_REGISTRY_FILE
)

print(
    OUTPUT_PROTOCOL_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 9.1 COMPLETE — SHIFT + SHAP NULL PROTOCOL FROZEN"
)

print(
    "===================================="
)