# ============================================================
# MODEL CHAPTER 10.3.1
# FREEZE OVERALL CROSS-CYCLE SHAP COMPARISON INPUTS
#
# Comparison:
#
#   Cycle24 HOLDOUT
#        vs
#   Cycle25 POOLED
#
# Within each frozen model only:
#
#   LR  C24 vs LR  C25
#   RF  C24 vs RF  C25
#   XGB C24 vs XGB C25
#
# THIS STEP DOES NOT:
#   - calculate drift metrics
#   - compare against null thresholds
#   - perform hypothesis testing
#   - make drift/stability claims
#   - modify any SHAP artifact
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

C24_ROWS = 13072
C24_POSITIVES = 527
C24_ENTITY_GROUPS = 176

C25_ROWS = 79361
C25_POSITIVES = 4836
C25_ENTITY_GROUPS = 1011

EXPECTED_FEATURES = 25
EXPECTED_MODELS = 3


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


SHAP_COLUMNS = [
    f"SHAP_{feature}"
    for feature in FEATURES
]


MODEL_ORDER = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


# ============================================================
# CHAPTER 9 / CYCLE24 FROZEN INPUTS
# ============================================================

CHAPTER9_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "chapter9_shift_null_controls_contract.json"
)


CHAPTER9_INVENTORY_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "chapter9_shift_null_controls_inventory.csv"
)


C24_SHAP_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "cycle24_holdout_shap_contract.json"
)


C24_SHAP_FILES = {

    "LR_02_C0p01_BALANCED":
        Path(
            "artifacts/models/shift_controls/"
            "shap/cycle24_holdout/"
            "LR_02_C0p01_BALANCED.parquet"
        ),

    "RF_04_D12_L5_BALSUB":
        Path(
            "artifacts/models/shift_controls/"
            "shap/cycle24_holdout/"
            "RF_04_D12_L5_BALSUB.parquet"
        ),

    "XGB_02_LR0p03_D3_BAL":
        Path(
            "artifacts/models/shift_controls/"
            "shap/cycle24_holdout/"
            "XGB_02_LR0p03_D3_BAL.parquet"
        ),
}


# ============================================================
# CHAPTER 10 / CYCLE25 FROZEN INPUTS
# ============================================================

C25_EXECUTION_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "cycle25_shap_execution_inputs_contract.json"
)


C25_SHAP_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "cycle25_shap_contract.json"
)


C25_MODEL_MANIFEST_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "cycle25_shap_model_manifest.csv"
)


C25_SHAP_REGISTRY_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "cycle25_shap_registry.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_REGISTRY_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "overall_cross_cycle_shap_input_registry.csv"
)


OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "overall_cross_cycle_shap_inputs_contract.json"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def verify_exists(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def normalized_path(path) -> str:

    return Path(
        str(path)
    ).as_posix()


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.3.1 — "
    "FREEZE OVERALL CROSS-CYCLE SHAP INPUTS"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    CHAPTER9_CONTRACT_FILE,
    CHAPTER9_INVENTORY_FILE,
    C24_SHAP_CONTRACT_FILE,
    C25_EXECUTION_CONTRACT_FILE,
    C25_SHAP_CONTRACT_FILE,
    C25_MODEL_MANIFEST_FILE,
    C25_SHAP_REGISTRY_FILE,
    *C24_SHAP_FILES.values(),
]


for path in required_files:

    verify_exists(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    OUTPUT_REGISTRY_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "10.3.1 comparison inputs."
        )


# ============================================================
# VERIFY CHAPTER 9 FREEZE
# ============================================================

chapter9_contract = load_json(
    CHAPTER9_CONTRACT_FILE
)


if (
    chapter9_contract.get(
        "status"
    )
    !=
    "FROZEN_CHAPTER9_SHIFT_NULL_CONTROLS"
):

    raise RuntimeError(
        "Chapter 9 controls are not frozen."
    )


chapter9_inventory_hash = sha256_file(
    CHAPTER9_INVENTORY_FILE
)


if (
    chapter9_inventory_hash
    !=
    chapter9_contract[
        "artifact_inventory"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Chapter 9 frozen inventory changed."
    )


chapter9_inventory = pd.read_csv(
    CHAPTER9_INVENTORY_FILE,
    keep_default_na=False,
)


if len(
    chapter9_inventory
) != 48:

    raise RuntimeError(
        "Expected exactly 48 frozen "
        "Chapter-9 artifacts."
    )


print(
    "Chapter 9 frozen controls:",
    "PASS"
)


# ============================================================
# INVENTORY LOOKUP
# ============================================================

inventory_path_map = {
    normalized_path(
        row[
            "path"
        ]
    ):
        str(
            row[
                "sha256"
            ]
        )
    for _, row
    in chapter9_inventory.iterrows()
}


# ============================================================
# VERIFY C24 SHAP CONTRACT
# ============================================================

c24_contract = load_json(
    C24_SHAP_CONTRACT_FILE
)


if (
    c24_contract.get(
        "status"
    )
    !=
    "FROZEN_CYCLE24_HOLDOUT_SHAP"
):

    raise RuntimeError(
        "Cycle24 HOLDOUT SHAP is not frozen."
    )


c24_contract_path_key = normalized_path(
    C24_SHAP_CONTRACT_FILE
)


if (
    c24_contract_path_key
    not in
    inventory_path_map
):

    raise RuntimeError(
        "Cycle24 SHAP contract is not "
        "present in frozen Chapter-9 inventory."
    )


if (
    sha256_file(
        C24_SHAP_CONTRACT_FILE
    )
    !=
    inventory_path_map[
        c24_contract_path_key
    ]
):

    raise RuntimeError(
        "Cycle24 SHAP contract changed "
        "after Chapter 9 freeze."
    )


print(
    "Frozen Cycle24 SHAP contract:",
    "PASS"
)


# ============================================================
# VERIFY C25 EXECUTION CONTRACT
# ============================================================

c25_execution_contract = load_json(
    C25_EXECUTION_CONTRACT_FILE
)


if (
    c25_execution_contract.get(
        "status"
    )
    !=
    "FROZEN_CYCLE25_SHAP_EXECUTION_INPUTS"
):

    raise RuntimeError(
        "Cycle25 SHAP execution inputs "
        "are not frozen."
    )


# ============================================================
# VERIFY C25 SHAP FINAL CONTRACT
# ============================================================

c25_contract = load_json(
    C25_SHAP_CONTRACT_FILE
)


if (
    c25_contract.get(
        "status"
    )
    !=
    "FROZEN_CYCLE25_SHAP"
):

    raise RuntimeError(
        "Cycle25 SHAP artifacts are not frozen."
    )


if (
    c25_contract[
        "scientific_guardrails"
    ][
        "drift_analysis_performed"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected previous drift-analysis state."
    )


print(
    "Frozen Cycle25 SHAP contract:",
    "PASS"
)


# ============================================================
# VERIFY C25 MODEL MANIFEST HASH
# ============================================================

c25_model_manifest_hash = sha256_file(
    C25_MODEL_MANIFEST_FILE
)


if (
    c25_model_manifest_hash
    !=
    c25_execution_contract[
        "frozen_models"
    ][
        "model_manifest"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle25 frozen model manifest changed."
    )


c25_model_manifest = pd.read_csv(
    C25_MODEL_MANIFEST_FILE,
    keep_default_na=False,
)


if len(
    c25_model_manifest
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 frozen models."
    )


# ============================================================
# VERIFY C25 SHAP REGISTRY HASH
# ============================================================

c25_registry_hash = sha256_file(
    C25_SHAP_REGISTRY_FILE
)


if (
    c25_registry_hash
    !=
    c25_contract[
        "models"
    ][
        "registry_sha256"
    ]
):

    raise RuntimeError(
        "Cycle25 SHAP registry changed "
        "after 10.2.4 freeze."
    )


c25_registry = pd.read_csv(
    C25_SHAP_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    c25_registry
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 Cycle25 "
        "SHAP registry rows."
    )


print(
    "Cycle25 frozen registries:",
    "PASS"
)


# ============================================================
# C24 CONTRACT MODEL MAP
# ============================================================

c24_models = {
    row[
        "candidate_id"
    ]:
        row
    for row
    in c24_contract[
        "models"
    ]
}


if set(
    c24_models.keys()
) != set(
    MODEL_ORDER
):

    raise RuntimeError(
        "Cycle24 frozen SHAP model set mismatch."
    )


# ============================================================
# VERIFY EACH MATCHED MODEL PAIR
# ============================================================

registry_rows = []


for candidate_id in MODEL_ORDER:

    print()

    print(
        "PAIR:",
        candidate_id
    )


    # --------------------------------------------------------
    # C24 CONTRACT MODEL
    # --------------------------------------------------------

    c24_model = c24_models[
        candidate_id
    ]


    # --------------------------------------------------------
    # C25 MODEL MANIFEST
    # --------------------------------------------------------

    c25_model_match = c25_model_manifest[
        c25_model_manifest[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(
        c25_model_match
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: Cycle25 model "
            "manifest lookup failed."
        )


    c25_model = c25_model_match.iloc[
        0
    ]


    # --------------------------------------------------------
    # SAME FROZEN MODEL BINARY
    # --------------------------------------------------------

    c24_model_hash = str(
        c24_model[
            "model_sha256"
        ]
    )


    c25_model_hash = str(
        c25_model[
            "model_sha256"
        ]
    )


    if c24_model_hash != c25_model_hash:

        raise RuntimeError(
            f"{candidate_id}: C24 and C25 "
            "do not reference the same "
            "frozen model binary."
        )


    # --------------------------------------------------------
    # SAME EXPLAINER
    # --------------------------------------------------------

    c24_explainer = str(
        c24_model[
            "explainer"
        ]
    )


    c25_explainer = str(
        c25_model[
            "explainer"
        ]
    )


    if c24_explainer != c25_explainer:

        raise RuntimeError(
            f"{candidate_id}: explainer mismatch "
            "between C24 and C25."
        )


    # --------------------------------------------------------
    # SAME INPUT SPACE
    # --------------------------------------------------------

    c24_input_space = str(
        c24_model[
            "input_space"
        ]
    )


    c25_input_space = str(
        c25_model[
            "feature_input_space"
        ]
    )


    if c24_input_space != c25_input_space:

        raise RuntimeError(
            f"{candidate_id}: SHAP input-space "
            "mismatch."
        )


    # --------------------------------------------------------
    # SAME OUTPUT SPACE
    # --------------------------------------------------------

    c24_output_space = str(
        c24_model[
            "output_space"
        ]
    )


    c25_output_space = str(
        c25_model[
            "output_space"
        ]
    )


    if c24_output_space != c25_output_space:

        raise RuntimeError(
            f"{candidate_id}: SHAP output-space "
            "mismatch."
        )


    # --------------------------------------------------------
    # C24 ARTIFACT — VERIFY AGAINST FROZEN CH9 INVENTORY
    # --------------------------------------------------------

    c24_file = C24_SHAP_FILES[
        candidate_id
    ]


    c24_path_key = normalized_path(
        c24_file
    )


    if (
        c24_path_key
        not in
        inventory_path_map
    ):

        raise RuntimeError(
            f"{candidate_id}: Cycle24 SHAP file "
            "not present in Chapter-9 inventory."
        )


    c24_hash = sha256_file(
        c24_file
    )


    if (
        c24_hash
        !=
        inventory_path_map[
            c24_path_key
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: frozen Cycle24 "
            "SHAP artifact changed."
        )


    # --------------------------------------------------------
    # C25 ARTIFACT — VERIFY AGAINST FROZEN 10.2.4 REGISTRY
    # --------------------------------------------------------

    c25_registry_match = c25_registry[
        c25_registry[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(
        c25_registry_match
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: Cycle25 SHAP "
            "registry lookup failed."
        )


    c25_registry_row = c25_registry_match.iloc[
        0
    ]


    c25_file = Path(
        str(
            c25_registry_row[
                "shap_file"
            ]
        )
    )


    verify_exists(
        c25_file
    )


    c25_hash = sha256_file(
        c25_file
    )


    if (
        c25_hash
        !=
        str(
            c25_registry_row[
                "shap_sha256"
            ]
        )
    ):

        raise RuntimeError(
            f"{candidate_id}: frozen Cycle25 "
            "SHAP artifact changed."
        )


    # --------------------------------------------------------
    # LOAD ONLY SHAP COLUMNS
    # --------------------------------------------------------

    c24_shap = pd.read_parquet(
        c24_file,
        columns=SHAP_COLUMNS,
    )


    c25_shap = pd.read_parquet(
        c25_file,
        columns=SHAP_COLUMNS,
    )


    # --------------------------------------------------------
    # ROW COUNTS
    # --------------------------------------------------------

    if len(
        c24_shap
    ) != C24_ROWS:

        raise RuntimeError(
            f"{candidate_id}: Cycle24 SHAP "
            "row-count mismatch."
        )


    if len(
        c25_shap
    ) != C25_ROWS:

        raise RuntimeError(
            f"{candidate_id}: Cycle25 SHAP "
            "row-count mismatch."
        )


    # --------------------------------------------------------
    # FEATURE MATRIX
    # --------------------------------------------------------

    c24_matrix = c24_shap.to_numpy(
        dtype=np.float64
    )


    c25_matrix = c25_shap.to_numpy(
        dtype=np.float64
    )


    if c24_matrix.shape != (
        C24_ROWS,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            f"{candidate_id}: invalid C24 "
            "SHAP matrix shape."
        )


    if c25_matrix.shape != (
        C25_ROWS,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            f"{candidate_id}: invalid C25 "
            "SHAP matrix shape."
        )


    if not np.isfinite(
        c24_matrix
    ).all():

        raise RuntimeError(
            f"{candidate_id}: C24 SHAP "
            "contains non-finite values."
        )


    if not np.isfinite(
        c25_matrix
    ).all():

        raise RuntimeError(
            f"{candidate_id}: C25 SHAP "
            "contains non-finite values."
        )


    # --------------------------------------------------------
    # BACKGROUND CONSISTENCY
    # --------------------------------------------------------

    c25_background_hash = str(
        c25_model[
            "background_sha256"
        ]
    )


    frozen_background_hash = str(
        c25_execution_contract[
            "shap_background"
        ][
            "sha256"
        ]
    )


    if (
        c25_background_hash
        !=
        frozen_background_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: frozen SHAP "
            "background mismatch."
        )


    # --------------------------------------------------------
    # SAVE MATCHED PAIR
    # --------------------------------------------------------

    registry_rows.append(
        {
            "candidate_id":
                candidate_id,

            "model_family":
                str(
                    c25_model[
                        "model_family"
                    ]
                ),

            "model_sha256":
                c25_model_hash,

            "same_frozen_model":
                True,

            "explainer":
                c25_explainer,

            "same_explainer":
                True,

            "feature_input_space":
                c25_input_space,

            "same_input_space":
                True,

            "output_space":
                c25_output_space,

            "same_output_space":
                True,

            "background_sha256":
                frozen_background_hash,

            "same_frozen_background":
                True,

            "feature_count":
                EXPECTED_FEATURES,

            "same_feature_order":
                True,

            "cycle24_population":
                "CYCLE24_HOLDOUT",

            "cycle24_rows":
                C24_ROWS,

            "cycle24_shap_file":
                c24_file.as_posix(),

            "cycle24_shap_sha256":
                c24_hash,

            "cycle25_population":
                "CYCLE25_POOLED",

            "cycle25_rows":
                C25_ROWS,

            "cycle25_shap_file":
                c25_file.as_posix(),

            "cycle25_shap_sha256":
                c25_hash,

            "cross_cycle_comparison_allowed":
                True,

            "cross_model_raw_magnitude_comparison_allowed":
                False,
        }
    )


    print(
        "Same frozen model:",
        "PASS"
    )

    print(
        "Same explainer:",
        "PASS"
    )

    print(
        "Same input space:",
        "PASS"
    )

    print(
        "Same output space:",
        "PASS"
    )

    print(
        "Same frozen background:",
        "PASS"
    )

    print(
        "Cycle24 SHAP:",
        c24_matrix.shape,
        "PASS"
    )

    print(
        "Cycle25 SHAP:",
        c25_matrix.shape,
        "PASS"
    )


# ============================================================
# BUILD COMPARISON REGISTRY
# ============================================================

comparison_registry = pd.DataFrame(
    registry_rows
)


if len(
    comparison_registry
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 matched "
        "cross-cycle model pairs."
    )


# ============================================================
# VERIFY MODEL SET
# ============================================================

if (
    comparison_registry[
        "candidate_id"
    ].tolist()
    !=
    MODEL_ORDER
):

    raise RuntimeError(
        "Cross-cycle model ordering mismatch."
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_REGISTRY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAVE REGISTRY ATOMICALLY
# ============================================================

registry_temp = OUTPUT_REGISTRY_FILE.with_name(
    OUTPUT_REGISTRY_FILE.stem
    +
    ".tmp.csv"
)


if registry_temp.exists():

    registry_temp.unlink()


comparison_registry.to_csv(
    registry_temp,
    index=False,
)


registry_rt = pd.read_csv(
    registry_temp,
    keep_default_na=False,
)


if len(
    registry_rt
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Comparison registry round-trip failed."
    )


registry_temp.replace(
    OUTPUT_REGISTRY_FILE
)


registry_hash = sha256_file(
    OUTPUT_REGISTRY_FILE
)


print()

print(
    "Comparison registry round-trip:",
    "PASS"
)


# ============================================================
# FINAL COMPARISON INPUT CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Frozen Overall Cross-Cycle SHAP Comparison Inputs",

    "version":
        "1.0",

    "status":
        "FROZEN_OVERALL_CROSS_CYCLE_SHAP_INPUTS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_3_1",

    "comparison":
        {
            "reference_population":
                "CYCLE24_HOLDOUT",

            "transfer_population":
                "CYCLE25_POOLED",

            "cycle24_rows":
                C24_ROWS,

            "cycle24_positive_rows":
                C24_POSITIVES,

            "cycle24_entity_groups":
                C24_ENTITY_GROUPS,

            "cycle25_rows":
                C25_ROWS,

            "cycle25_positive_rows":
                C25_POSITIVES,

            "cycle25_entity_groups":
                C25_ENTITY_GROUPS,

            "models":
                EXPECTED_MODELS,

            "features":
                EXPECTED_FEATURES,
        },

    "frozen_sources":
        {
            "chapter9_contract":
                {
                    "file":
                        CHAPTER9_CONTRACT_FILE.as_posix(),

                    "sha256":
                        sha256_file(
                            CHAPTER9_CONTRACT_FILE
                        ),

                    "status":
                        "FROZEN_CHAPTER9_SHIFT_NULL_CONTROLS",
                },

            "cycle24_shap_contract":
                {
                    "file":
                        C24_SHAP_CONTRACT_FILE.as_posix(),

                    "sha256":
                        sha256_file(
                            C24_SHAP_CONTRACT_FILE
                        ),

                    "status":
                        "FROZEN_CYCLE24_HOLDOUT_SHAP",
                },

            "cycle25_execution_contract":
                {
                    "file":
                        C25_EXECUTION_CONTRACT_FILE.as_posix(),

                    "sha256":
                        sha256_file(
                            C25_EXECUTION_CONTRACT_FILE
                        ),

                    "status":
                        "FROZEN_CYCLE25_SHAP_EXECUTION_INPUTS",
                },

            "cycle25_shap_contract":
                {
                    "file":
                        C25_SHAP_CONTRACT_FILE.as_posix(),

                    "sha256":
                        sha256_file(
                            C25_SHAP_CONTRACT_FILE
                        ),

                    "status":
                        "FROZEN_CYCLE25_SHAP",
                },
        },

    "input_registry":
        {
            "file":
                OUTPUT_REGISTRY_FILE.as_posix(),

            "sha256":
                registry_hash,

            "rows":
                EXPECTED_MODELS,
        },

    "feature_order":
        FEATURES,

    "comparison_rules":
        {
            "comparison_unit":
                "WITHIN_SAME_FROZEN_MODEL",

            "lr":
                (
                    "CYCLE24_HOLDOUT_LR "
                    "VS CYCLE25_POOLED_LR"
                ),

            "rf":
                (
                    "CYCLE24_HOLDOUT_RF "
                    "VS CYCLE25_POOLED_RF"
                ),

            "xgb":
                (
                    "CYCLE24_HOLDOUT_XGB "
                    "VS CYCLE25_POOLED_XGB"
                ),

            "cross_model_raw_shap_magnitude_comparison":
                False,

            "same_frozen_models":
                True,

            "same_feature_order":
                True,

            "same_model_specific_explanation_spaces":
                True,

            "same_frozen_cycle24_train_background":
                True,
        },

    "explanation_spaces":
        {
            "LR_02_C0p01_BALANCED":
                "LOG_ODDS_DECISION_FUNCTION",

            "RF_04_D12_L5_BALSUB":
                "POSITIVE_CLASS_PROBABILITY",

            "XGB_02_LR0p03_D3_BAL":
                "XGBOOST_RAW_MARGIN",
        },

    "scientific_guardrails":
        {
            "cycle25_used_for_training":
                False,

            "models_retrained":
                False,

            "models_retuned":
                False,

            "preprocessing_refit":
                False,

            "background_changed":
                False,

            "cycle24_null_changed":
                False,

            "overall_drift_metrics_computed":
                False,

            "null_thresholds_applied":
                False,

            "hypothesis_test_performed":
                False,

            "drift_conclusion_made":
                False,

            "stability_conclusion_made":
                False,

            "shap_is_causal":
                False,
        },

    "immutability_rule":
        (
            "After Chapter 10.3.1, the matched "
            "Cycle24 HOLDOUT and Cycle25 POOLED SHAP "
            "artifacts, model mapping, feature order, "
            "model-specific explanation spaces and "
            "frozen background must not be changed "
            "in response to subsequent cross-cycle "
            "comparison results."
        ),

    "next_stage":
        "MODEL_CHAPTER_10_3_2_OVERALL_SHAP_STRUCTURE_POINT_ESTIMATES",
}


# ============================================================
# SAVE CONTRACT ATOMICALLY
# ============================================================

contract_temp = OUTPUT_CONTRACT_FILE.with_name(
    OUTPUT_CONTRACT_FILE.stem
    +
    ".tmp.json"
)


if contract_temp.exists():

    contract_temp.unlink()


with contract_temp.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


contract_rt = load_json(
    contract_temp
)


if (
    contract_rt.get(
        "status"
    )
    !=
    "FROZEN_OVERALL_CROSS_CYCLE_SHAP_INPUTS"
):

    raise RuntimeError(
        "10.3.1 contract round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "overall_drift_metrics_computed"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected metric-computation state."
    )


contract_temp.replace(
    OUTPUT_CONTRACT_FILE
)


contract_hash = sha256_file(
    OUTPUT_CONTRACT_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 10.3.1 — COMPLETE"
)

print("=" * 78)


print(
    "Matched models:",
    EXPECTED_MODELS
)

print(
    "Features:",
    EXPECTED_FEATURES
)


print()

print(
    "Cycle24 reference:",
    "HOLDOUT"
)

print(
    "Cycle24 rows:",
    C24_ROWS
)

print(
    "Cycle25 transfer:",
    "POOLED"
)

print(
    "Cycle25 rows:",
    C25_ROWS
)


print()

print(
    "Same frozen models:",
    "PASS"
)

print(
    "Same feature order:",
    "PASS"
)

print(
    "Same model-specific output spaces:",
    "PASS"
)

print(
    "Same frozen SHAP background:",
    "PASS"
)


print()

print(
    "Drift metrics computed:",
    False
)

print(
    "Null thresholds applied:",
    False
)

print(
    "Drift conclusion made:",
    False
)


print()

print(
    "Status:",
    "FROZEN_OVERALL_CROSS_CYCLE_SHAP_INPUTS"
)


print()

print(
    "Registry:",
    OUTPUT_REGISTRY_FILE
)

print(
    "Contract:",
    OUTPUT_CONTRACT_FILE
)

print(
    "Contract SHA256:",
    contract_hash
)


print()

print(
    "NEXT:",
    (
        "Chapter 10.3.2 — "
        "Overall SHAP Structure Point Estimates"
    )
)