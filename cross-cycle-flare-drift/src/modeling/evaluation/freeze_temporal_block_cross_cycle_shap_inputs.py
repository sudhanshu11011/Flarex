# ============================================================
# MODEL CHAPTER 10.4.1
# FREEZE TEMPORAL-BLOCK CROSS-CYCLE SHAP INPUTS
#
# Comparisons to be performed later:
#
#   Cycle24 HOLDOUT vs Cycle25 B1 (2020-2021)
#   Cycle24 HOLDOUT vs Cycle25 B2 (2022-2023)
#   Cycle24 HOLDOUT vs Cycle25 B3 (2024-2025)
#
# Within same frozen model only:
#
#   LR  -> LR
#   RF  -> RF
#   XGB -> XGB
#
# THIS STEP DOES NOT:
#   - compute temporal SHAP metrics
#   - apply Cycle24 null thresholds
#   - perform inference
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

EXPECTED_C24_ROWS = 13072
EXPECTED_FEATURES = 25
EXPECTED_MODELS = 3
EXPECTED_REGISTRY_ROWS = 9


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


BLOCK_SPECS = {

    "B1": {
        "year_definition": "2020-2021",
        "rows": 10411,
        "positives": 133,
        "groups": 146,
    },

    "B2": {
        "year_definition": "2022-2023",
        "rows": 33132,
        "positives": 1801,
        "groups": 414,
    },

    "B3": {
        "year_definition": "2024-2025",
        "rows": 35818,
        "positives": 2902,
        "groups": 458,
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


CYCLE25_SHAP_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "cycle25_shap_contract.json"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_REGISTRY_FILE = Path(
    "artifacts/models/cross_cycle_shap/manifests/"
    "temporal_block_cross_cycle_shap_input_registry.csv"
)


OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/cross_cycle_shap/contracts/"
    "temporal_block_cross_cycle_shap_inputs_contract.json"
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

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
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


def block_membership_fingerprint(
    frame: pd.DataFrame,
) -> str:

    digest = hashlib.sha256()


    timestamps = pd.to_datetime(
        frame[
            TIME_COLUMN
        ],
        utc=True,
        errors="raise",
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
        frame[
            "cycle25_block"
        ].astype(str),
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
    "MODEL CHAPTER 10.4.1 — "
    "FREEZE TEMPORAL-BLOCK CROSS-CYCLE SHAP INPUTS"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

for path in [
    OVERALL_INPUT_CONTRACT_FILE,
    OVERALL_INPUT_REGISTRY_FILE,
    OVERALL_POINT_CONTRACT_FILE,
    CYCLE25_SHAP_CONTRACT_FILE,
]:

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
            "Chapter 10.4.1 outputs."
        )


# ============================================================
# VERIFY 10.3.1
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
        "Chapter 10.3.1 inputs are not frozen."
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
        "10.3.1 comparison registry changed."
    )


print(
    "Chapter 10.3.1 frozen inputs:",
    "PASS"
)


# ============================================================
# VERIFY 10.3.2B
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
        "Chapter 10.3.2B point estimates "
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
        "Unexpected previous drift conclusion."
    )


print(
    "Chapter 10.3.2B frozen point estimates:",
    "PASS"
)


# ============================================================
# VERIFY CYCLE25 SHAP FREEZE
# ============================================================

cycle25_contract = load_json(
    CYCLE25_SHAP_CONTRACT_FILE
)


if (
    cycle25_contract.get(
        "status"
    )
    !=
    "FROZEN_CYCLE25_SHAP"
):

    raise RuntimeError(
        "Cycle25 SHAP is not frozen."
    )


print(
    "Frozen Cycle25 SHAP:",
    "PASS"
)


# ============================================================
# LOAD OVERALL MODEL PAIR REGISTRY
# ============================================================

overall_registry = pd.read_csv(
    OVERALL_INPUT_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    overall_registry
) != EXPECTED_MODELS:

    raise RuntimeError(
        "Expected exactly 3 overall model pairs."
    )


if (
    overall_registry[
        "candidate_id"
    ].astype(str).tolist()
    !=
    MODEL_ORDER
):

    raise RuntimeError(
        "Overall registry model ordering changed."
    )


# ============================================================
# RESULTS
# ============================================================

registry_rows = []

reference_cycle25_metadata = None

reference_block_fingerprints = {}


# ============================================================
# VERIFY EACH MODEL
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
    # VERIFY COMPARISON GUARDRAILS
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
    # FILES
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


    # --------------------------------------------------------
    # HASHES
    # --------------------------------------------------------

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
            "SHAP changed."
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
            "SHAP changed."
        )


    # --------------------------------------------------------
    # VERIFY C24 SHAP MATRIX
    # --------------------------------------------------------

    c24_shap = pd.read_parquet(
        c24_file,
        columns=SHAP_COLUMNS,
    )


    if len(
        c24_shap
    ) != EXPECTED_C24_ROWS:

        raise RuntimeError(
            f"{candidate_id}: Cycle24 row "
            "count mismatch."
        )


    c24_matrix = c24_shap.to_numpy(
        dtype=np.float64
    )


    if c24_matrix.shape != (
        EXPECTED_C24_ROWS,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            f"{candidate_id}: Cycle24 SHAP "
            "matrix shape mismatch."
        )


    if not np.isfinite(
        c24_matrix
    ).all():

        raise RuntimeError(
            f"{candidate_id}: Cycle24 SHAP "
            "contains non-finite values."
        )


    # --------------------------------------------------------
    # LOAD C25 TEMPORAL METADATA
    # --------------------------------------------------------

    c25_metadata = pd.read_parquet(
        c25_file,
        columns=[
            "source_row_number",
            "cycle25_block",
            ENTITY_GROUP_COLUMN,
            TARGET_COLUMN,
            TIME_COLUMN,
        ],
    )


    if len(
        c25_metadata
    ) != 79361:

        raise RuntimeError(
            f"{candidate_id}: Cycle25 row "
            "count mismatch."
        )


    if c25_metadata[
        "source_row_number"
    ].duplicated().any():

        raise RuntimeError(
            f"{candidate_id}: duplicate "
            "Cycle25 source rows."
        )


    if not np.array_equal(
        c25_metadata[
            "source_row_number"
        ].to_numpy(
            dtype=np.int64
        ),
        np.arange(
            79361,
            dtype=np.int64,
        ),
    ):

        raise RuntimeError(
            f"{candidate_id}: Cycle25 "
            "source row ordering changed."
        )


    # --------------------------------------------------------
    # SAME TEMPORAL MEMBERSHIP ACROSS MODELS
    # --------------------------------------------------------

    comparison_metadata = pd.DataFrame(
        {
            "source_row_number":
                c25_metadata[
                    "source_row_number"
                ].to_numpy(
                    dtype=np.int64
                ),

            "cycle25_block":
                c25_metadata[
                    "cycle25_block"
                ].astype(
                    str
                ).to_numpy(),

            ENTITY_GROUP_COLUMN:
                c25_metadata[
                    ENTITY_GROUP_COLUMN
                ].astype(
                    str
                ).to_numpy(),

            TARGET_COLUMN:
                c25_metadata[
                    TARGET_COLUMN
                ].to_numpy(
                    dtype=np.int8
                ),

            TIME_COLUMN:
                pd.to_datetime(
                    c25_metadata[
                        TIME_COLUMN
                    ],
                    utc=True,
                ).astype(
                    "int64"
                ).to_numpy(),
        }
    )


    if reference_cycle25_metadata is None:

        reference_cycle25_metadata = (
            comparison_metadata
        )


    else:

        for column in [
            "source_row_number",
            "cycle25_block",
            ENTITY_GROUP_COLUMN,
            TARGET_COLUMN,
            TIME_COLUMN,
        ]:

            if not np.array_equal(
                comparison_metadata[
                    column
                ].to_numpy(),
                reference_cycle25_metadata[
                    column
                ].to_numpy(),
            ):

                raise RuntimeError(
                    f"{candidate_id}: Cycle25 "
                    f"temporal metadata differs "
                    f"across models: {column}"
                )


    # --------------------------------------------------------
    # VERIFY EACH BLOCK
    # --------------------------------------------------------

    for block_name, spec in (
        BLOCK_SPECS.items()
    ):

        block = c25_metadata[
            c25_metadata[
                "cycle25_block"
            ].astype(str)
            ==
            block_name
        ].copy()


        block = block.sort_values(
            "source_row_number",
            kind="stable",
        ).reset_index(
            drop=True
        )


        actual_rows = len(
            block
        )


        actual_positives = int(
            (
                block[
                    TARGET_COLUMN
                ]
                ==
                1
            ).sum()
        )


        actual_groups = int(
            block[
                ENTITY_GROUP_COLUMN
            ]
            .astype(str)
            .nunique()
        )


        if (
            actual_rows
            !=
            spec[
                "rows"
            ]
        ):

            raise RuntimeError(
                f"{candidate_id} {block_name}: "
                "row-count mismatch."
            )


        if (
            actual_positives
            !=
            spec[
                "positives"
            ]
        ):

            raise RuntimeError(
                f"{candidate_id} {block_name}: "
                "positive-count mismatch."
            )


        if (
            actual_groups
            !=
            spec[
                "groups"
            ]
        ):

            raise RuntimeError(
                f"{candidate_id} {block_name}: "
                "group-count mismatch."
            )


        fingerprint = (
            block_membership_fingerprint(
                block
            )
        )


        if (
            block_name
            not in
            reference_block_fingerprints
        ):

            reference_block_fingerprints[
                block_name
            ] = fingerprint


        elif (
            reference_block_fingerprints[
                block_name
            ]
            !=
            fingerprint
        ):

            raise RuntimeError(
                f"{candidate_id}: {block_name} "
                "row membership differs "
                "across models."
            )


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

                "reference_population":
                    "CYCLE24_HOLDOUT",

                "reference_rows":
                    EXPECTED_C24_ROWS,

                "reference_shap_file":
                    c24_file.as_posix(),

                "reference_shap_sha256":
                    c24_hash,

                "transfer_population":
                    f"CYCLE25_{block_name}",

                "cycle25_block":
                    block_name,

                "year_definition":
                    spec[
                        "year_definition"
                    ],

                "transfer_rows":
                    actual_rows,

                "transfer_positive_rows":
                    actual_positives,

                "transfer_negative_rows":
                    (
                        actual_rows
                        -
                        actual_positives
                    ),

                "transfer_entity_groups":
                    actual_groups,

                "block_membership_sha256":
                    fingerprint,

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

                "temporal_metrics_computed":
                    False,

                "null_threshold_applied":
                    False,

                "drift_classification":
                    "NOT_EVALUATED",
            }
        )


        print(
            f"{block_name}: "
            f"rows={actual_rows}, "
            f"positives={actual_positives}, "
            f"groups={actual_groups} — PASS"
        )


    print(
        "Temporal membership:",
        "PASS"
    )


# ============================================================
# FINAL CROSS-MODEL METADATA CHECK
# ============================================================

if reference_cycle25_metadata is None:

    raise RuntimeError(
        "No Cycle25 reference metadata created."
    )


print()

print(
    "Cross-model temporal row alignment:",
    "PASS"
)


# ============================================================
# BUILD REGISTRY
# ============================================================

registry = pd.DataFrame(
    registry_rows
)


if len(
    registry
) != EXPECTED_REGISTRY_ROWS:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_REGISTRY_ROWS} temporal "
        f"registry rows, got {len(registry)}."
    )


# ============================================================
# VERIFY 3 BLOCKS PER MODEL
# ============================================================

for candidate_id in MODEL_ORDER:

    subset = registry[
        registry[
            "candidate_id"
        ]
        ==
        candidate_id
    ]


    if (
        subset[
            "cycle25_block"
        ].tolist()
        !=
        [
            "B1",
            "B2",
            "B3",
        ]
    ):

        raise RuntimeError(
            f"{candidate_id}: temporal "
            "block ordering mismatch."
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
        "Temporal registry round-trip failed."
    )


registry_temp.replace(
    OUTPUT_REGISTRY_FILE
)


registry_hash = sha256_file(
    OUTPUT_REGISTRY_FILE
)


print(
    "Temporal registry round-trip:",
    "PASS"
)


# ============================================================
# FINAL CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Frozen Temporal-Block "
            "Cross-Cycle SHAP Inputs"
        ),

    "version":
        "1.0",

    "status":
        (
            "FROZEN_TEMPORAL_BLOCK_"
            "CROSS_CYCLE_SHAP_INPUTS"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_10_4_1",

    "reference_population":
        {
            "name":
                "CYCLE24_HOLDOUT",

            "rows":
                EXPECTED_C24_ROWS,
        },

    "transfer_blocks":
        {
            "B1":
                BLOCK_SPECS[
                    "B1"
                ],

            "B2":
                BLOCK_SPECS[
                    "B2"
                ],

            "B3":
                BLOCK_SPECS[
                    "B3"
                ],
        },

    "models":
        {
            "count":
                EXPECTED_MODELS,

            "candidate_ids":
                MODEL_ORDER,

            "features":
                EXPECTED_FEATURES,
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

            "overall_point_estimate_contract":
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

            "cycle25_shap_contract":
                {
                    "file":
                        CYCLE25_SHAP_CONTRACT_FILE.as_posix(),

                    "sha256":
                        sha256_file(
                            CYCLE25_SHAP_CONTRACT_FILE
                        ),

                    "status":
                        "FROZEN_CYCLE25_SHAP",
                },
        },

    "block_membership_sha256":
        reference_block_fingerprints,

    "input_registry":
        {
            "file":
                OUTPUT_REGISTRY_FILE.as_posix(),

            "sha256":
                registry_hash,

            "rows":
                EXPECTED_REGISTRY_ROWS,
        },

    "comparison_rule":
        {
            "within_same_frozen_model_only":
                True,

            "reference":
                "CYCLE24_HOLDOUT",

            "comparisons":
                [
                    "CYCLE24_HOLDOUT_VS_CYCLE25_B1",
                    "CYCLE24_HOLDOUT_VS_CYCLE25_B2",
                    "CYCLE24_HOLDOUT_VS_CYCLE25_B3",
                ],

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

            "temporal_metrics_computed":
                False,

            "null_thresholds_applied":
                False,

            "featurewise_inference_performed":
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
            "Temporal blocks are frozen before "
            "temporal SHAP metrics are computed. "
            "No temporal block may be redefined "
            "in response to observed SHAP results."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_10_4_2_"
            "TEMPORAL_BLOCK_SHAP_POINT_ESTIMATES"
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
    "FROZEN_TEMPORAL_BLOCK_CROSS_CYCLE_SHAP_INPUTS"
):

    raise RuntimeError(
        "10.4.1 contract round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "temporal_metrics_computed"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected temporal metric state."
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
    "MODEL CHAPTER 10.4.1 — COMPLETE"
)

print("=" * 78)


print(
    "Models:",
    EXPECTED_MODELS
)

print(
    "Temporal blocks:",
    3
)

print(
    "Registry rows:",
    len(
        registry
    )
)


print()

for block_name, spec in (
    BLOCK_SPECS.items()
):

    print(
        f"{block_name}: "
        f"{spec['year_definition']} | "
        f"rows={spec['rows']} | "
        f"positives={spec['positives']} | "
        f"groups={spec['groups']}"
    )


print()

print(
    "Cross-model temporal alignment:",
    "PASS"
)

print(
    "Temporal metrics computed:",
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
        "FROZEN_TEMPORAL_BLOCK_"
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
        "Chapter 10.4.2 — "
        "Temporal-Block SHAP Point Estimates"
    )
)