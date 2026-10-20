# ============================================================
# MODEL CHAPTER 10.5.1B
# FREEZE CLASS-CONDITIONAL CROSS-CYCLE SHAP INPUTS
#
# Comparisons to be performed later:
#
#   C24 HOLDOUT TARGET_0 vs C25 POOLED TARGET_0
#   C24 HOLDOUT TARGET_1 vs C25 POOLED TARGET_1
#
# Within the same frozen model only.
#
# THIS STEP DOES NOT:
#   - calculate SHAP drift metrics
#   - apply null thresholds
#   - perform featurewise inference
#   - apply FDR
#   - make drift/stability conclusions
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

EXPECTED_MODELS = 3
EXPECTED_FEATURES = 25

EXPECTED_C24_ROWS = 13072
EXPECTED_C25_ROWS = 79361

EXPECTED_C24_TARGET_0 = 12545
EXPECTED_C24_TARGET_1 = 527

EXPECTED_C25_TARGET_0 = 74525
EXPECTED_C25_TARGET_1 = 4836

EXPECTED_REGISTRY_ROWS = 6

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TARGET_COLUMN = "target"
TIME_COLUMN = "observation_time_utc"


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


STRATUM_SPECS = {

    "TARGET_0": {
        "target_value": 0,
        "cycle24_rows": EXPECTED_C24_TARGET_0,
        "cycle25_rows": EXPECTED_C25_TARGET_0,
    },

    "TARGET_1": {
        "target_value": 1,
        "cycle24_rows": EXPECTED_C24_TARGET_1,
        "cycle25_rows": EXPECTED_C25_TARGET_1,
    },
}


# ============================================================
# FROZEN INPUTS
# ============================================================

OVERALL_INPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "overall_cross_cycle_shap_inputs_contract.json"
)


OVERALL_INPUT_REGISTRY_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "overall_cross_cycle_shap_input_registry.csv"
)


OVERALL_POINT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "overall_cross_cycle_shap_point_estimates_contract.json"
)


TEMPORAL_POINT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "temporal_block_cross_cycle_shap_point_estimates_contract.json"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_REGISTRY_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "class_conditional_cross_cycle_shap_input_registry.csv"
)


OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "class_conditional_cross_cycle_shap_inputs_contract.json"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(
    path: Path,
) -> str:

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


def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def verify_exists(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def membership_fingerprint(
    frame: pd.DataFrame,
    include_block: bool,
) -> str:

    digest = hashlib.sha256()


    timestamps = pd.to_datetime(
        frame[
            TIME_COLUMN
        ],
        utc=True,
        errors="raise",
    )


    if include_block:

        blocks = frame[
            "cycle25_block"
        ].astype(str)

    else:

        blocks = pd.Series(
            [
                "CYCLE24_HOLDOUT"
            ]
            *
            len(frame)
        )


    for (
        source_row,
        block,
        group,
        target,
        timestamp,
    ) in zip(
        frame[
            "source_row_number"
        ].astype(
            np.int64
        ),
        blocks,
        frame[
            ENTITY_GROUP_COLUMN
        ].astype(str),
        frame[
            TARGET_COLUMN
        ].astype(
            np.int8
        ),
        timestamps,
    ):

        line = (
            f"{int(source_row)}|"
            f"{block}|"
            f"{group}|"
            f"{int(target)}|"
            f"{timestamp.isoformat()}\n"
        )


        digest.update(
            line.encode(
                "utf-8"
            )
        )


    return digest.hexdigest()


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 10.5.1B — "
    "FREEZE CLASS-CONDITIONAL CROSS-CYCLE SHAP INPUTS"
)

print("=" * 78)


# ============================================================
# REQUIRED INPUTS
# ============================================================

for path in [
    OVERALL_INPUT_CONTRACT_FILE,
    OVERALL_INPUT_REGISTRY_FILE,
    OVERALL_POINT_CONTRACT_FILE,
    TEMPORAL_POINT_CONTRACT_FILE,
]:

    verify_exists(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for output_path in [
    OUTPUT_REGISTRY_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if output_path.exists():

        raise RuntimeError(
            f"Output already exists: {output_path}\n"
            "Refusing to overwrite frozen "
            "Chapter 10.5.1B outputs."
        )


# ============================================================
# VERIFY OVERALL INPUT CONTRACT
# ============================================================

overall_input_contract = load_json(
    OVERALL_INPUT_CONTRACT_FILE
)


if (
    overall_input_contract.get(
        "status"
    )
    !=
    "FROZEN_OVERALL_CROSS_CYCLE_SHAP_INPUTS"
):

    raise RuntimeError(
        "Overall cross-cycle SHAP inputs "
        "are not frozen."
    )


overall_input_contract_hash = sha256_file(
    OVERALL_INPUT_CONTRACT_FILE
)


overall_registry_hash = sha256_file(
    OVERALL_INPUT_REGISTRY_FILE
)


if (
    overall_registry_hash
    !=
    overall_input_contract[
        "input_registry"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Frozen overall comparison registry "
        "hash mismatch."
    )


print(
    "Frozen overall comparison inputs:",
    "PASS"
)


# ============================================================
# VERIFY OVERALL POINT ESTIMATES
# ============================================================

overall_point_contract = load_json(
    OVERALL_POINT_CONTRACT_FILE
)


if (
    overall_point_contract.get(
        "status"
    )
    !=
    "FROZEN_OVERALL_CROSS_CYCLE_SHAP_POINT_ESTIMATES"
):

    raise RuntimeError(
        "Overall SHAP point estimates "
        "are not frozen."
    )


if (
    overall_point_contract[
        "scientific_guardrails"
    ][
        "drift_conclusion_made"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected prior drift conclusion."
    )


print(
    "Frozen overall point estimates:",
    "PASS"
)


# ============================================================
# VERIFY TEMPORAL POINT ESTIMATES
# ============================================================

temporal_point_contract = load_json(
    TEMPORAL_POINT_CONTRACT_FILE
)


if (
    temporal_point_contract.get(
        "status"
    )
    !=
    "FROZEN_TEMPORAL_BLOCK_CROSS_CYCLE_SHAP_POINT_ESTIMATES"
):

    raise RuntimeError(
        "Temporal-block SHAP point estimates "
        "are not frozen."
    )


if (
    temporal_point_contract[
        "scientific_guardrails"
    ][
        "drift_conclusion_made"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected temporal drift conclusion."
    )


print(
    "Frozen temporal point estimates:",
    "PASS"
)


# ============================================================
# LOAD OVERALL REGISTRY
# ============================================================

overall_registry = pd.read_csv(
    OVERALL_INPUT_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    overall_registry
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 frozen model pairs."
    )


if (
    overall_registry[
        "candidate_id"
    ].astype(str).tolist()
    !=
    MODEL_ORDER
):

    raise RuntimeError(
        "Frozen model order changed."
    )


# ============================================================
# RESULTS
# ============================================================

registry_rows = []

reference_fingerprints = {
    "C24": {},
    "C25": {},
}


# ============================================================
# EACH MODEL
# ============================================================

for candidate_id in MODEL_ORDER:

    print()

    print(
        "VERIFY MODEL:",
        candidate_id
    )


    match = overall_registry[
        overall_registry[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(
        match
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: overall registry "
            "lookup failed."
        )


    row = match.iloc[
        0
    ]


    # --------------------------------------------------------
    # FROZEN PAIR GUARDRAILS
    # --------------------------------------------------------

    for column in [
        "same_frozen_model",
        "same_explainer",
        "same_input_space",
        "same_output_space",
        "same_frozen_background",
        "same_feature_order",
        "cross_cycle_comparison_allowed",
    ]:

        value = str(
            row[
                column
            ]
        ).strip().lower()


        if value not in {
            "true",
            "1",
        }:

            raise RuntimeError(
                f"{candidate_id}: frozen "
                f"guardrail failed: {column}"
            )


    # --------------------------------------------------------
    # FILES + HASHES
    # --------------------------------------------------------

    c24_file = Path(
        str(
            row[
                "cycle24_shap_file"
            ]
        )
    )


    c25_file = Path(
        str(
            row[
                "cycle25_shap_file"
            ]
        )
    )


    verify_exists(
        c24_file
    )


    verify_exists(
        c25_file
    )


    c24_hash = sha256_file(
        c24_file
    )


    c25_hash = sha256_file(
        c25_file
    )


    if (
        c24_hash
        !=
        str(
            row[
                "cycle24_shap_sha256"
            ]
        )
    ):

        raise RuntimeError(
            f"{candidate_id}: frozen Cycle24 "
            "SHAP artifact changed."
        )


    if (
        c25_hash
        !=
        str(
            row[
                "cycle25_shap_sha256"
            ]
        )
    ):

        raise RuntimeError(
            f"{candidate_id}: frozen Cycle25 "
            "SHAP artifact changed."
        )


    # --------------------------------------------------------
    # LOAD C24
    # --------------------------------------------------------

    c24 = pd.read_parquet(
        c24_file,
        columns=[
            "source_row_number",
            ENTITY_GROUP_COLUMN,
            TARGET_COLUMN,
            TIME_COLUMN,
            *SHAP_COLUMNS,
        ],
    )


    if len(
        c24
    ) != EXPECTED_C24_ROWS:

        raise RuntimeError(
            f"{candidate_id}: C24 row-count mismatch."
        )


    # --------------------------------------------------------
    # LOAD C25
    # --------------------------------------------------------

    c25 = pd.read_parquet(
        c25_file,
        columns=[
            "source_row_number",
            "cycle25_block",
            ENTITY_GROUP_COLUMN,
            TARGET_COLUMN,
            TIME_COLUMN,
            *SHAP_COLUMNS,
        ],
    )


    if len(
        c25
    ) != EXPECTED_C25_ROWS:

        raise RuntimeError(
            f"{candidate_id}: C25 row-count mismatch."
        )


    # --------------------------------------------------------
    # TARGET DOMAIN
    # --------------------------------------------------------

    if set(
        c24[
            TARGET_COLUMN
        ].astype(int).unique().tolist()
    ) != {
        0,
        1,
    }:

        raise RuntimeError(
            f"{candidate_id}: invalid C24 target domain."
        )


    if set(
        c25[
            TARGET_COLUMN
        ].astype(int).unique().tolist()
    ) != {
        0,
        1,
    }:

        raise RuntimeError(
            f"{candidate_id}: invalid C25 target domain."
        )


    # ========================================================
    # EACH CLASS STRATUM
    # ========================================================

    for (
        stratum_name,
        spec,
    ) in STRATUM_SPECS.items():

        target_value = spec[
            "target_value"
        ]


        c24_subset = c24[
            c24[
                TARGET_COLUMN
            ]
            ==
            target_value
        ].copy()


        c25_subset = c25[
            c25[
                TARGET_COLUMN
            ]
            ==
            target_value
        ].copy()


        c24_subset = c24_subset.sort_values(
            "source_row_number",
            kind="stable",
        ).reset_index(
            drop=True
        )


        c25_subset = c25_subset.sort_values(
            "source_row_number",
            kind="stable",
        ).reset_index(
            drop=True
        )


        # ----------------------------------------------------
        # ROW COUNTS
        # ----------------------------------------------------

        if len(
            c24_subset
        ) != spec[
            "cycle24_rows"
        ]:

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "C24 class row-count mismatch."
            )


        if len(
            c25_subset
        ) != spec[
            "cycle25_rows"
        ]:

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "C25 class row-count mismatch."
            )


        # ----------------------------------------------------
        # TARGET PURITY
        # ----------------------------------------------------

        if not (
            c24_subset[
                TARGET_COLUMN
            ]
            ==
            target_value
        ).all():

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "C24 target purity failed."
            )


        if not (
            c25_subset[
                TARGET_COLUMN
            ]
            ==
            target_value
        ).all():

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "C25 target purity failed."
            )


        # ----------------------------------------------------
        # SHAP NUMERIC INTEGRITY
        # ----------------------------------------------------

        c24_shap = c24_subset[
            SHAP_COLUMNS
        ].to_numpy(
            dtype=np.float64
        )


        c25_shap = c25_subset[
            SHAP_COLUMNS
        ].to_numpy(
            dtype=np.float64
        )


        if c24_shap.shape != (
            spec[
                "cycle24_rows"
            ],
            EXPECTED_FEATURES,
        ):

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "C24 SHAP shape mismatch."
            )


        if c25_shap.shape != (
            spec[
                "cycle25_rows"
            ],
            EXPECTED_FEATURES,
        ):

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "C25 SHAP shape mismatch."
            )


        if not np.isfinite(
            c24_shap
        ).all():

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "non-finite C24 SHAP."
            )


        if not np.isfinite(
            c25_shap
        ).all():

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "non-finite C25 SHAP."
            )


        # ----------------------------------------------------
        # GROUP COUNTS
        # ----------------------------------------------------

        c24_groups = int(
            c24_subset[
                ENTITY_GROUP_COLUMN
            ]
            .astype(str)
            .nunique()
        )


        c25_groups = int(
            c25_subset[
                ENTITY_GROUP_COLUMN
            ]
            .astype(str)
            .nunique()
        )


        # ----------------------------------------------------
        # MEMBERSHIP FINGERPRINTS
        # ----------------------------------------------------

        c24_fingerprint = membership_fingerprint(
            c24_subset[
                [
                    "source_row_number",
                    ENTITY_GROUP_COLUMN,
                    TARGET_COLUMN,
                    TIME_COLUMN,
                ]
            ].copy(),
            include_block=False,
        )


        c25_fingerprint = membership_fingerprint(
            c25_subset[
                [
                    "source_row_number",
                    "cycle25_block",
                    ENTITY_GROUP_COLUMN,
                    TARGET_COLUMN,
                    TIME_COLUMN,
                ]
            ].copy(),
            include_block=True,
        )


        # ----------------------------------------------------
        # SAME MEMBERSHIP ACROSS MODELS
        # ----------------------------------------------------

        if (
            stratum_name
            not in
            reference_fingerprints[
                "C24"
            ]
        ):

            reference_fingerprints[
                "C24"
            ][
                stratum_name
            ] = c24_fingerprint


        elif (
            reference_fingerprints[
                "C24"
            ][
                stratum_name
            ]
            !=
            c24_fingerprint
        ):

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "C24 class membership differs "
                "across models."
            )


        if (
            stratum_name
            not in
            reference_fingerprints[
                "C25"
            ]
        ):

            reference_fingerprints[
                "C25"
            ][
                stratum_name
            ] = c25_fingerprint


        elif (
            reference_fingerprints[
                "C25"
            ][
                stratum_name
            ]
            !=
            c25_fingerprint
        ):

            raise RuntimeError(
                f"{candidate_id} {stratum_name}: "
                "C25 class membership differs "
                "across models."
            )


        # ----------------------------------------------------
        # REGISTRY ROW
        # ----------------------------------------------------

        registry_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "model_family":
                    str(
                        row[
                            "model_family"
                        ]
                    ),

                "output_space":
                    str(
                        row[
                            "output_space"
                        ]
                    ),

                "stratum":
                    stratum_name,

                "target_value":
                    target_value,

                "cycle24_population":
                    "CYCLE24_HOLDOUT",

                "cycle24_rows":
                    len(
                        c24_subset
                    ),

                "cycle24_entity_groups":
                    c24_groups,

                "cycle24_membership_sha256":
                    c24_fingerprint,

                "cycle24_shap_file":
                    c24_file.as_posix(),

                "cycle24_shap_sha256":
                    c24_hash,

                "cycle25_population":
                    "CYCLE25_POOLED",

                "cycle25_rows":
                    len(
                        c25_subset
                    ),

                "cycle25_entity_groups":
                    c25_groups,

                "cycle25_membership_sha256":
                    c25_fingerprint,

                "cycle25_shap_file":
                    c25_file.as_posix(),

                "cycle25_shap_sha256":
                    c25_hash,

                "same_frozen_model":
                    True,

                "same_feature_order":
                    True,

                "same_output_space":
                    True,

                "same_frozen_background":
                    True,

                "class_conditional_metrics_computed":
                    False,

                "null_threshold_applied":
                    False,

                "drift_classification":
                    "NOT_EVALUATED",
            }
        )


        print(
            f"{stratum_name}: "
            f"C24 rows={len(c24_subset)}, "
            f"groups={c24_groups} | "
            f"C25 rows={len(c25_subset)}, "
            f"groups={c25_groups} — PASS"
        )


    # --------------------------------------------------------
    # PARTITION AUDIT
    # --------------------------------------------------------

    if (
        len(
            c24[
                c24[
                    TARGET_COLUMN
                ]
                ==
                0
            ]
        )
        +
        len(
            c24[
                c24[
                    TARGET_COLUMN
                ]
                ==
                1
            ]
        )
        !=
        EXPECTED_C24_ROWS
    ):

        raise RuntimeError(
            f"{candidate_id}: C24 class partition "
            "does not cover all rows."
        )


    if (
        len(
            c25[
                c25[
                    TARGET_COLUMN
                ]
                ==
                0
            ]
        )
        +
        len(
            c25[
                c25[
                    TARGET_COLUMN
                ]
                ==
                1
            ]
        )
        !=
        EXPECTED_C25_ROWS
    ):

        raise RuntimeError(
            f"{candidate_id}: C25 class partition "
            "does not cover all rows."
        )


    print(
        "Class partition coverage:",
        "PASS"
    )


# ============================================================
# FINAL REGISTRY
# ============================================================

registry = pd.DataFrame(
    registry_rows
)


if len(
    registry
) != EXPECTED_REGISTRY_ROWS:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_REGISTRY_ROWS} class-conditional "
        f"registry rows, got {len(registry)}."
    )


# ============================================================
# VERIFY ORDER
# ============================================================

expected_order = [
    (
        candidate_id,
        stratum_name,
    )
    for candidate_id in MODEL_ORDER
    for stratum_name in [
        "TARGET_0",
        "TARGET_1",
    ]
]


actual_order = list(
    zip(
        registry[
            "candidate_id"
        ].astype(str),

        registry[
            "stratum"
        ].astype(str),
    )
)


if actual_order != expected_order:

    raise RuntimeError(
        "Class-conditional registry "
        "ordering mismatch."
    )


print()

print(
    "Cross-model class membership alignment:",
    "PASS"
)


# ============================================================
# SAVE REGISTRY
# ============================================================

OUTPUT_REGISTRY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


registry_temp = OUTPUT_REGISTRY_FILE.with_name(
    OUTPUT_REGISTRY_FILE.stem
    +
    ".tmp.csv"
)


if registry_temp.exists():

    registry_temp.unlink()


registry.to_csv(
    registry_temp,
    index=False,
)


registry_rt = pd.read_csv(
    registry_temp,
    keep_default_na=False,
)


if len(
    registry_rt
) != EXPECTED_REGISTRY_ROWS:

    raise RuntimeError(
        "Class-conditional registry "
        "round-trip failed."
    )


registry_temp.replace(
    OUTPUT_REGISTRY_FILE
)


registry_hash = sha256_file(
    OUTPUT_REGISTRY_FILE
)


print(
    "Class-conditional registry round-trip:",
    "PASS"
)


# ============================================================
# FINAL CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Class-Conditional "
            "Cross-Cycle SHAP Inputs"
        ),

    "version":
        "1.0",

    "status":
        (
            "FROZEN_CLASS_CONDITIONAL_"
            "CROSS_CYCLE_SHAP_INPUTS"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_5_1B",

    "comparison":
        {
            "reference_population":
                "CYCLE24_HOLDOUT",

            "transfer_population":
                "CYCLE25_POOLED",

            "strata":
                [
                    "TARGET_0",
                    "TARGET_1",
                ],

            "models":
                EXPECTED_MODELS,

            "features":
                EXPECTED_FEATURES,

            "cycle24_rows":
                EXPECTED_C24_ROWS,

            "cycle25_rows":
                EXPECTED_C25_ROWS,

            "cycle24_target_0_rows":
                EXPECTED_C24_TARGET_0,

            "cycle24_target_1_rows":
                EXPECTED_C24_TARGET_1,

            "cycle25_target_0_rows":
                EXPECTED_C25_TARGET_0,

            "cycle25_target_1_rows":
                EXPECTED_C25_TARGET_1,
        },

    "frozen_sources":
        {
            "overall_input_contract":
                {
                    "file":
                        OVERALL_INPUT_CONTRACT_FILE.as_posix(),

                    "sha256":
                        overall_input_contract_hash,

                    "status":
                        "FROZEN_OVERALL_CROSS_CYCLE_SHAP_INPUTS",
                },

            "overall_point_contract":
                {
                    "file":
                        OVERALL_POINT_CONTRACT_FILE.as_posix(),

                    "sha256":
                        sha256_file(
                            OVERALL_POINT_CONTRACT_FILE
                        ),

                    "status":
                        (
                            "FROZEN_OVERALL_CROSS_CYCLE_"
                            "SHAP_POINT_ESTIMATES"
                        ),
                },

            "temporal_point_contract":
                {
                    "file":
                        TEMPORAL_POINT_CONTRACT_FILE.as_posix(),

                    "sha256":
                        sha256_file(
                            TEMPORAL_POINT_CONTRACT_FILE
                        ),

                    "status":
                        (
                            "FROZEN_TEMPORAL_BLOCK_"
                            "CROSS_CYCLE_SHAP_POINT_ESTIMATES"
                        ),
                },
        },

    "membership_fingerprints":
        reference_fingerprints,

    "input_registry":
        {
            "file":
                OUTPUT_REGISTRY_FILE.as_posix(),

            "sha256":
                registry_hash,

            "rows":
                EXPECTED_REGISTRY_ROWS,
        },

    "comparison_rules":
        {
            "within_same_frozen_model_only":
                True,

            "class_conditioning_variable":
                TARGET_COLUMN,

            "target_0_comparison":
                (
                    "CYCLE24_HOLDOUT_TARGET_0 "
                    "VS CYCLE25_POOLED_TARGET_0"
                ),

            "target_1_comparison":
                (
                    "CYCLE24_HOLDOUT_TARGET_1 "
                    "VS CYCLE25_POOLED_TARGET_1"
                ),

            "same_feature_order":
                True,

            "same_model_specific_output_space":
                True,

            "same_frozen_background":
                True,

            "cross_model_raw_shap_magnitude_comparison":
                False,
        },

    "scientific_guardrails":
        {
            "cycle25_used_for_training":
                False,

            "models_retrained":
                False,

            "shap_recomputed":
                False,

            "background_changed":
                False,

            "cycle24_null_changed":
                False,

            "class_conditional_populations_frozen":
                True,

            "class_conditional_metrics_computed":
                False,

            "null_thresholds_applied":
                False,

            "featurewise_inference_performed":
                False,

            "fdr_applied":
                False,

            "common_support_sensitivity_applied":
                False,

            "correlated_group_robustness_applied":
                False,

            "drift_conclusion_made":
                False,

            "stability_conclusion_made":
                False,

            "shap_is_causal":
                False,
        },

    "interpretation_rule":
        (
            "Class-conditioned comparisons are frozen "
            "before their SHAP structure metrics are "
            "computed. Their purpose is to distinguish "
            "cross-cycle explanation changes from changes "
            "that may be associated with differing class "
            "prevalence/composition. No drift or stability "
            "claim is made at this stage."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_5_2_"
            "CLASS_CONDITIONAL_SHAP_POINT_ESTIMATES"
        ),
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
    "FROZEN_CLASS_CONDITIONAL_CROSS_CYCLE_SHAP_INPUTS"
):

    raise RuntimeError(
        "10.5.1B contract round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "class_conditional_metrics_computed"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected class-conditional metric state."
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
    "MODEL CHAPTER 10.5.1B — COMPLETE"
)

print("=" * 78)


print(
    "Models:",
    EXPECTED_MODELS
)

print(
    "Class strata:",
    2
)

print(
    "Registry rows:",
    len(
        registry
    )
)


print()

print(
    "C24 TARGET_0 rows:",
    EXPECTED_C24_TARGET_0
)

print(
    "C24 TARGET_1 rows:",
    EXPECTED_C24_TARGET_1
)

print(
    "C25 TARGET_0 rows:",
    EXPECTED_C25_TARGET_0
)

print(
    "C25 TARGET_1 rows:",
    EXPECTED_C25_TARGET_1
)


print()

print(
    "Cross-model class alignment:",
    "PASS"
)

print(
    "Class-conditional metrics computed:",
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
    (
        "FROZEN_CLASS_CONDITIONAL_"
        "CROSS_CYCLE_SHAP_INPUTS"
    )
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
        "Chapter 10.5.2 — "
        "Class-Conditional SHAP Point Estimates"
    )
)