from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROTOCOL_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "shift_null_control_protocol.json"
)

CYCLE25_BLOCKS_FILE = Path(
    "artifacts/models/evaluation/manifests/"
    "cycle25_temporal_blocks.csv"
)

CYCLE24_HOLDOUT_FILE = Path(
    "artifacts/splitting/data/"
    "cycle24_holdout.parquet"
)

CYCLE25_FILE = Path(
    "data/final/dataset_v2/"
    "cycle25_dataset_v2.parquet"
)


OUTPUT_PREVALENCE_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "label_prevalence.csv"
)

OUTPUT_AR_POPULATION_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "active_region_population.csv"
)

OUTPUT_GROUP_SUMMARY_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "active_region_group_summary.csv"
)

OUTPUT_BOOTSTRAP_FILE = Path(
    "artifacts/models/shift_controls/bootstrap/"
    "label_prevalence_group_bootstrap.parquet"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "prevalence_ar_population_contract.json"
)


# ============================================================
# FROZEN CONSTANTS
# ============================================================

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TARGET_COLUMN = "target"
TIME_COLUMN = "observation_time_utc"

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 29031

LOWER_Q = 0.025
UPPER_Q = 0.975


PRIMARY_BLOCKS = [
    "C25_B1_2020_2021",
    "C25_B2_2022_2023",
    "C25_B3_2024_2025",
]

POOLED_SCOPE = "C25_ALL_2020_2025"

SCOPE_ORDER = [
    "C24_HOLDOUT",
    *PRIMARY_BLOCKS,
    POOLED_SCOPE,
]


# Known frozen population checks
EXPECTED_COUNTS = {
    "C24_HOLDOUT": {
        "rows": 13072,
        "positive": 527,
        "negative": 12545,
        "groups": 176,
    },

    "C25_B1_2020_2021": {
        "rows": 10411,
        "positive": 133,
        "negative": 10278,
        "groups": 146,
    },

    "C25_B2_2022_2023": {
        "rows": 33132,
        "positive": 1801,
        "negative": 31331,
        "groups": 414,
    },

    "C25_B3_2024_2025": {
        "rows": 35818,
        "positive": 2902,
        "negative": 32916,
        "groups": 458,
    },

    "C25_ALL_2020_2025": {
        "rows": 79361,
        "positive": 4836,
        "negative": 74525,
        "groups": 1011,
    },
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


def q(values, probability):

    return float(
        np.quantile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            probability,
            method="linear",
        )
    )


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 9.3 — LABEL PREVALENCE + ACTIVE-REGION POPULATION AUDIT"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    PROTOCOL_FILE,
    CYCLE25_BLOCKS_FILE,
    CYCLE24_HOLDOUT_FILE,
    CYCLE25_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# PROTECT OUTPUTS
# ============================================================

for path in [
    OUTPUT_PREVALENCE_FILE,
    OUTPUT_AR_POPULATION_FILE,
    OUTPUT_GROUP_SUMMARY_FILE,
    OUTPUT_BOOTSTRAP_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter 9.3 results."
        )


# ============================================================
# LOAD + VERIFY FROZEN PROTOCOL
# ============================================================

protocol = load_json(
    PROTOCOL_FILE
)


if protocol.get(
    "status"
) != "FROZEN_BEFORE_SHIFT_AND_SHAP_ANALYSIS":

    raise RuntimeError(
        "Chapter 9.1 protocol is not frozen."
    )


prevalence_protocol = protocol[
    "label_prevalence"
]


ar_protocol = protocol[
    "active_region_population"
]


if prevalence_protocol[
    "uncertainty"
][
    "unit"
] != "ENTITY_GROUP":

    raise RuntimeError(
        "Frozen prevalence bootstrap unit mismatch."
    )


if int(
    prevalence_protocol[
        "uncertainty"
    ][
        "replicates"
    ]
) != BOOTSTRAP_REPLICATES:

    raise RuntimeError(
        "Frozen prevalence bootstrap "
        "replicate count mismatch."
    )


if int(
    prevalence_protocol[
        "uncertainty"
    ][
        "seed"
    ]
) != BOOTSTRAP_SEED:

    raise RuntimeError(
        "Frozen prevalence bootstrap seed mismatch."
    )


if ar_protocol[
    "unit"
] != "ENTITY_GROUP":

    raise RuntimeError(
        "Frozen AR-population unit mismatch."
    )


print(
    "Chapter 9.1 protocol:",
    "FROZEN"
)

print(
    "Prevalence bootstrap unit:",
    "ENTITY_GROUP"
)

print(
    "Prevalence bootstrap replicates:",
    BOOTSTRAP_REPLICATES
)

print(
    "AR population unit:",
    "ENTITY_GROUP"
)


# ============================================================
# HASH VERIFICATION
# ============================================================

c24_hash = sha256_file(
    CYCLE24_HOLDOUT_FILE
)

c25_hash = sha256_file(
    CYCLE25_FILE
)


expected_c24_hash = protocol[
    "dataset_expectations"
][
    "cycle24_holdout"
][
    "sha256"
]


expected_c25_hash = protocol[
    "dataset_expectations"
][
    "cycle25"
][
    "sha256"
]


if c24_hash != expected_c24_hash:

    raise RuntimeError(
        "Cycle24 holdout hash mismatch."
    )


if c25_hash != expected_c25_hash:

    raise RuntimeError(
        "Cycle25 dataset hash mismatch."
    )


print(
    "Cycle24 holdout hash:",
    "PASS"
)

print(
    "Cycle25 dataset hash:",
    "PASS"
)


# ============================================================
# READ DATA
# ============================================================

c24 = pd.read_parquet(
    CYCLE24_HOLDOUT_FILE
)

c25 = pd.read_parquet(
    CYCLE25_FILE
)


required_columns = {
    ENTITY_GROUP_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
}


for name, dataframe in [
    ("C24_HOLDOUT", c24),
    ("CYCLE25", c25),
]:

    missing = (
        required_columns
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{name}: missing required columns "
            f"{sorted(missing)}"
        )


    if dataframe[
        ENTITY_GROUP_COLUMN
    ].isna().any():

        raise RuntimeError(
            f"{name}: missing ENTITY_GROUP_ID."
        )


    target_values = set(
        dataframe[
            TARGET_COLUMN
        ].astype(
            int
        ).unique()
    )


    if target_values != {0, 1}:

        raise RuntimeError(
            f"{name}: target is not binary."
        )


print(
    "Cycle24 holdout opened:",
    True
)

print(
    "Cycle25 opened:",
    True
)


# ============================================================
# VERIFY CROSS-CYCLE ENTITY-GROUP ISOLATION
# ============================================================

c24_groups = set(
    c24[
        ENTITY_GROUP_COLUMN
    ].astype(
        str
    )
)

c25_groups = set(
    c25[
        ENTITY_GROUP_COLUMN
    ].astype(
        str
    )
)


cross_cycle_overlap = (
    c24_groups
    &
    c25_groups
)


expected_overlap = int(
    ar_protocol[
        "cycle24_cycle25_group_overlap_expected"
    ]
)


if len(
    cross_cycle_overlap
) != expected_overlap:

    raise RuntimeError(
        "Unexpected Cycle24/Cycle25 "
        "ENTITY_GROUP overlap."
    )


print(
    "Cross-cycle ENTITY_GROUP overlap:",
    len(
        cross_cycle_overlap
    )
)

print(
    "Cross-cycle group isolation:",
    "PASS"
)


# ============================================================
# ASSIGN FROZEN CYCLE25 TEMPORAL BLOCKS
# ============================================================

blocks = pd.read_csv(
    CYCLE25_BLOCKS_FILE,
    keep_default_na=False,
)


primary_blocks = blocks[
    blocks[
        "primary_block"
    ].astype(
        str
    ).str.lower()
    ==
    "true"
].copy()


if len(
    primary_blocks
) != 3:

    raise RuntimeError(
        "Expected exactly three "
        "Cycle25 primary blocks."
    )


observation_time = pd.to_datetime(
    c25[
        TIME_COLUMN
    ],
    utc=True,
    errors="raise",
)


block_id = pd.Series(
    pd.NA,
    index=c25.index,
    dtype="string",
)


for _, block in primary_blocks.iterrows():

    current_id = str(
        block[
            "block_id"
        ]
    )


    start = pd.Timestamp(
        block[
            "start_utc"
        ]
    )


    end_exclusive = pd.Timestamp(
        block[
            "end_exclusive_utc"
        ]
    )


    mask = (
        observation_time
        >=
        start
    ) & (
        observation_time
        <
        end_exclusive
    )


    if block_id.loc[
        mask
    ].notna().any():

        raise RuntimeError(
            "Cycle25 temporal-block overlap."
        )


    block_id.loc[
        mask
    ] = current_id


if block_id.isna().any():

    raise RuntimeError(
        "Some Cycle25 rows were not "
        "assigned to a frozen block."
    )


c25 = c25.copy()

c25[
    "_C25_BLOCK_ID"
] = block_id.astype(
    str
)


if set(
    c25[
        "_C25_BLOCK_ID"
    ].unique()
) != set(
    PRIMARY_BLOCKS
):

    raise RuntimeError(
        "Cycle25 block IDs mismatch."
    )


print(
    "Frozen Cycle25 block assignment:",
    "PASS"
)


# ============================================================
# BUILD SCOPE DATAFRAMES
# ============================================================

scope_frames = {
    "C24_HOLDOUT":
        c24,
}


for scope_id in PRIMARY_BLOCKS:

    scope_frames[
        scope_id
    ] = c25[
        c25[
            "_C25_BLOCK_ID"
        ]
        ==
        scope_id
    ].copy()


scope_frames[
    POOLED_SCOPE
] = c25


# ============================================================
# VERIFY KNOWN POPULATIONS
# ============================================================

for scope_id in SCOPE_ORDER:

    dataframe = scope_frames[
        scope_id
    ]


    rows = int(
        len(
            dataframe
        )
    )


    positive = int(
        dataframe[
            TARGET_COLUMN
        ].sum()
    )


    negative = int(
        (
            dataframe[
                TARGET_COLUMN
            ]
            ==
            0
        ).sum()
    )


    groups = int(
        dataframe[
            ENTITY_GROUP_COLUMN
        ].nunique()
    )


    expected = EXPECTED_COUNTS[
        scope_id
    ]


    if rows != expected[
        "rows"
    ]:

        raise RuntimeError(
            f"{scope_id}: row-count mismatch."
        )


    if positive != expected[
        "positive"
    ]:

        raise RuntimeError(
            f"{scope_id}: positive-count mismatch."
        )


    if negative != expected[
        "negative"
    ]:

        raise RuntimeError(
            f"{scope_id}: negative-count mismatch."
        )


    if groups != expected[
        "groups"
    ]:

        raise RuntimeError(
            f"{scope_id}: ENTITY_GROUP "
            "count mismatch."
        )


print(
    "Frozen scope populations:",
    "PASS"
)


# ============================================================
# CHILD RNG STREAMS
# ============================================================

seed_sequence = np.random.SeedSequence(
    BOOTSTRAP_SEED
)


child_sequences = seed_sequence.spawn(
    len(
        SCOPE_ORDER
    )
)


seed_registry = {}


for index, scope_id in enumerate(
    SCOPE_ORDER
):

    seed_registry[
        scope_id
    ] = {
        "scope_index":
            index,

        "spawn_key":
            list(
                child_sequences[
                    index
                ].spawn_key
            ),

        "derived_seed_uint32":
            int(
                child_sequences[
                    index
                ].generate_state(
                    1,
                    dtype=np.uint32,
                )[0]
            ),
    }


# ============================================================
# GROUP SUMMARIES + PREVALENCE BOOTSTRAP
# ============================================================

group_summary_frames = []
population_rows = []
prevalence_rows = []
bootstrap_frames = []


print(
    "\n===================================="
)

print(
    "COMPUTING PREVALENCE + AR POPULATION"
)

print(
    "===================================="
)


for scope_index, scope_id in enumerate(
    SCOPE_ORDER
):

    dataframe = scope_frames[
        scope_id
    ].copy()


    # --------------------------------------------------------
    # PER-ENTITY-GROUP SUMMARY
    # --------------------------------------------------------

    group_summary = (
        dataframe.groupby(
            ENTITY_GROUP_COLUMN,
            sort=True,
            observed=True,
        )
        .agg(
            row_count=(
                TARGET_COLUMN,
                "size",
            ),

            positive_count=(
                TARGET_COLUMN,
                "sum",
            ),
        )
        .reset_index()
    )


    group_summary[
        "negative_count"
    ] = (
        group_summary[
            "row_count"
        ]
        -
        group_summary[
            "positive_count"
        ]
    )


    group_summary[
        "positive_rate"
    ] = (
        group_summary[
            "positive_count"
        ]
        /
        group_summary[
            "row_count"
        ]
    )


    group_summary[
        "has_positive"
    ] = (
        group_summary[
            "positive_count"
        ]
        >
        0
    )


    group_summary.insert(
        0,
        "scope_id",
        scope_id,
    )


    group_summary_frames.append(
        group_summary
    )


    # --------------------------------------------------------
    # BASIC POINT PREVALENCE
    # --------------------------------------------------------

    rows = int(
        len(
            dataframe
        )
    )


    positive = int(
        dataframe[
            TARGET_COLUMN
        ].sum()
    )


    negative = int(
        rows
        -
        positive
    )


    prevalence = float(
        positive
        /
        rows
    )


    entity_groups = int(
        len(
            group_summary
        )
    )


    # --------------------------------------------------------
    # ENTITY_GROUP BOOTSTRAP
    # --------------------------------------------------------

    group_row_counts = (
        group_summary[
            "row_count"
        ].to_numpy(
            dtype=np.int64
        )
    )


    group_positive_counts = (
        group_summary[
            "positive_count"
        ].to_numpy(
            dtype=np.int64
        )
    )


    G = entity_groups


    rng = np.random.default_rng(
        child_sequences[
            scope_index
        ]
    )


    multiplicity = rng.multinomial(
        G,
        np.full(
            G,
            1.0 / G,
            dtype=np.float64,
        ),
        size=BOOTSTRAP_REPLICATES,
    ).astype(
        np.int16
    )


    if not (
        multiplicity.sum(
            axis=1
        )
        ==
        G
    ).all():

        raise RuntimeError(
            f"{scope_id}: bootstrap "
            "group-draw integrity failed."
        )


    bootstrap_rows = (
        multiplicity
        @
        group_row_counts
    )


    bootstrap_positive = (
        multiplicity
        @
        group_positive_counts
    )


    if (
        bootstrap_rows
        <=
        0
    ).any():

        raise RuntimeError(
            f"{scope_id}: zero-row "
            "bootstrap replicate."
        )


    bootstrap_prevalence = (
        bootstrap_positive
        /
        bootstrap_rows
    )


    if not np.isfinite(
        bootstrap_prevalence
    ).all():

        raise RuntimeError(
            f"{scope_id}: non-finite "
            "bootstrap prevalence."
        )


    prevalence_ci_lower = q(
        bootstrap_prevalence,
        LOWER_Q,
    )


    prevalence_ci_upper = q(
        bootstrap_prevalence,
        UPPER_Q,
    )


    bootstrap_frames.append(
        pd.DataFrame(
            {
                "scope_id":
                    scope_id,

                "replicate_id":
                    np.arange(
                        1,
                        BOOTSTRAP_REPLICATES + 1,
                        dtype=np.int32,
                    ),

                "entity_group_draw_count":
                    np.full(
                        BOOTSTRAP_REPLICATES,
                        G,
                        dtype=np.int32,
                    ),

                "weighted_row_count":
                    bootstrap_rows,

                "weighted_positive_count":
                    bootstrap_positive,

                "positive_prevalence":
                    bootstrap_prevalence,
            }
        )
    )


    prevalence_rows.append(
        {
            "scope_id":
                scope_id,

            "row_count":
                rows,

            "positive_count":
                positive,

            "negative_count":
                negative,

            "positive_prevalence":
                prevalence,

            "entity_group_count":
                entity_groups,

            "bootstrap_mean":
                float(
                    bootstrap_prevalence.mean()
                ),

            "bootstrap_median":
                float(
                    np.median(
                        bootstrap_prevalence
                    )
                ),

            "ci_lower_2p5":
                prevalence_ci_lower,

            "ci_upper_97p5":
                prevalence_ci_upper,

            "confidence_level":
                0.95,

            "ci_method":
                "ENTITY_GROUP_PERCENTILE_BOOTSTRAP",

            "valid_replicates":
                BOOTSTRAP_REPLICATES,

            "point_inside_ci":
                bool(
                    prevalence_ci_lower
                    <=
                    prevalence
                    <=
                    prevalence_ci_upper
                ),
        }
    )


    # --------------------------------------------------------
    # ACTIVE-REGION POPULATION SUMMARY
    # --------------------------------------------------------

    rows_per_group = (
        group_summary[
            "row_count"
        ].to_numpy(
            dtype=np.float64
        )
    )


    group_positive_rate = (
        group_summary[
            "positive_rate"
        ].to_numpy(
            dtype=np.float64
        )
    )


    positive_groups = int(
        group_summary[
            "has_positive"
        ].sum()
    )


    positive_group_fraction = float(
        positive_groups
        /
        entity_groups
    )


    rows_q25 = q(
        rows_per_group,
        0.25,
    )

    rows_median = q(
        rows_per_group,
        0.50,
    )

    rows_q75 = q(
        rows_per_group,
        0.75,
    )


    rate_q25 = q(
        group_positive_rate,
        0.25,
    )

    rate_median = q(
        group_positive_rate,
        0.50,
    )

    rate_q75 = q(
        group_positive_rate,
        0.75,
    )


    population_rows.append(
        {
            "scope_id":
                scope_id,

            "row_count":
                rows,

            "entity_group_count":
                entity_groups,

            "rows_per_group_mean":
                float(
                    rows_per_group.mean()
                ),

            "rows_per_group_q25":
                rows_q25,

            "rows_per_group_median":
                rows_median,

            "rows_per_group_q75":
                rows_q75,

            "rows_per_group_iqr":
                float(
                    rows_q75
                    -
                    rows_q25
                ),

            "rows_per_group_min":
                float(
                    rows_per_group.min()
                ),

            "rows_per_group_max":
                float(
                    rows_per_group.max()
                ),

            "positive_groups":
                positive_groups,

            "zero_positive_groups":
                int(
                    entity_groups
                    -
                    positive_groups
                ),

            "positive_group_fraction":
                positive_group_fraction,

            "group_positive_rate_mean":
                float(
                    group_positive_rate.mean()
                ),

            "group_positive_rate_q25":
                rate_q25,

            "group_positive_rate_median":
                rate_median,

            "group_positive_rate_q75":
                rate_q75,

            "group_positive_rate_iqr":
                float(
                    rate_q75
                    -
                    rate_q25
                ),

            "group_positive_rate_min":
                float(
                    group_positive_rate.min()
                ),

            "group_positive_rate_max":
                float(
                    group_positive_rate.max()
                ),
        }
    )


    print(
        f"{scope_id}: "
        f"rows={rows}, "
        f"positive={positive}, "
        f"prevalence={prevalence:.6f}, "
        f"95% CI=["
        f"{prevalence_ci_lower:.6f}, "
        f"{prevalence_ci_upper:.6f}], "
        f"groups={entity_groups}, "
        f"positive_groups={positive_groups}"
    )


# ============================================================
# BUILD OUTPUT TABLES
# ============================================================

prevalence_df = pd.DataFrame(
    prevalence_rows
)


population_df = pd.DataFrame(
    population_rows
)


group_summary_df = pd.concat(
    group_summary_frames,
    ignore_index=True,
)


bootstrap_df = pd.concat(
    bootstrap_frames,
    ignore_index=True,
)


# ============================================================
# STRUCTURAL CHECKS
# ============================================================

EXPECTED_PREVALENCE_ROWS = 5

EXPECTED_BOOTSTRAP_ROWS = (
    5
    *
    BOOTSTRAP_REPLICATES
)

EXPECTED_GROUP_SUMMARY_ROWS = (
    EXPECTED_COUNTS[
        "C24_HOLDOUT"
    ][
        "groups"
    ]
    +
    EXPECTED_COUNTS[
        "C25_B1_2020_2021"
    ][
        "groups"
    ]
    +
    EXPECTED_COUNTS[
        "C25_B2_2022_2023"
    ][
        "groups"
    ]
    +
    EXPECTED_COUNTS[
        "C25_B3_2024_2025"
    ][
        "groups"
    ]
    +
    EXPECTED_COUNTS[
        "C25_ALL_2020_2025"
    ][
        "groups"
    ]
)


if len(
    prevalence_df
) != EXPECTED_PREVALENCE_ROWS:

    raise RuntimeError(
        "Prevalence table row-count mismatch."
    )


if len(
    population_df
) != EXPECTED_PREVALENCE_ROWS:

    raise RuntimeError(
        "AR-population table row-count mismatch."
    )


if len(
    bootstrap_df
) != EXPECTED_BOOTSTRAP_ROWS:

    raise RuntimeError(
        "Prevalence bootstrap row-count mismatch."
    )


if len(
    group_summary_df
) != EXPECTED_GROUP_SUMMARY_ROWS:

    raise RuntimeError(
        "Group-summary row-count mismatch."
    )


if prevalence_df[
    "scope_id"
].tolist() != SCOPE_ORDER:

    raise RuntimeError(
        "Prevalence scope order mismatch."
    )


if population_df[
    "scope_id"
].tolist() != SCOPE_ORDER:

    raise RuntimeError(
        "AR-population scope order mismatch."
    )


if prevalence_df.duplicated(
    subset=[
        "scope_id"
    ]
).any():

    raise RuntimeError(
        "Duplicate prevalence scope."
    )


if population_df.duplicated(
    subset=[
        "scope_id"
    ]
).any():

    raise RuntimeError(
        "Duplicate AR-population scope."
    )


if group_summary_df.duplicated(
    subset=[
        "scope_id",
        ENTITY_GROUP_COLUMN,
    ]
).any():

    raise RuntimeError(
        "Duplicate group-summary row."
    )


if bootstrap_df.duplicated(
    subset=[
        "scope_id",
        "replicate_id",
    ]
).any():

    raise RuntimeError(
        "Duplicate bootstrap replicate."
    )


print(
    "\nPrevalence rows:",
    len(
        prevalence_df
    )
)

print(
    "AR-population rows:",
    len(
        population_df
    )
)

print(
    "Group-summary rows:",
    len(
        group_summary_df
    )
)

print(
    "Bootstrap rows:",
    len(
        bootstrap_df
    )
)

print(
    "Output structure:",
    "PASS"
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_PREVALENCE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_AR_POPULATION_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_GROUP_SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_BOOTSTRAP_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


prevalence_df.to_csv(
    OUTPUT_PREVALENCE_FILE,
    index=False,
)


population_df.to_csv(
    OUTPUT_AR_POPULATION_FILE,
    index=False,
)


group_summary_df.to_csv(
    OUTPUT_GROUP_SUMMARY_FILE,
    index=False,
)


bootstrap_df.to_parquet(
    OUTPUT_BOOTSTRAP_FILE,
    index=False,
)


# ============================================================
# HASH OUTPUTS
# ============================================================

prevalence_hash = sha256_file(
    OUTPUT_PREVALENCE_FILE
)

population_hash = sha256_file(
    OUTPUT_AR_POPULATION_FILE
)

group_summary_hash = sha256_file(
    OUTPUT_GROUP_SUMMARY_FILE
)

bootstrap_hash = sha256_file(
    OUTPUT_BOOTSTRAP_FILE
)


# ============================================================
# ROUND-TRIP VERIFICATION
# ============================================================

prevalence_roundtrip = pd.read_csv(
    OUTPUT_PREVALENCE_FILE,
    keep_default_na=False,
)

population_roundtrip = pd.read_csv(
    OUTPUT_AR_POPULATION_FILE,
    keep_default_na=False,
)

group_roundtrip = pd.read_csv(
    OUTPUT_GROUP_SUMMARY_FILE,
    keep_default_na=False,
)

bootstrap_roundtrip = pd.read_parquet(
    OUTPUT_BOOTSTRAP_FILE
)


if len(
    prevalence_roundtrip
) != len(
    prevalence_df
):

    raise RuntimeError(
        "Prevalence CSV round-trip failed."
    )


if len(
    population_roundtrip
) != len(
    population_df
):

    raise RuntimeError(
        "AR-population CSV round-trip failed."
    )


if len(
    group_roundtrip
) != len(
    group_summary_df
):

    raise RuntimeError(
        "Group-summary CSV round-trip failed."
    )


if len(
    bootstrap_roundtrip
) != len(
    bootstrap_df
):

    raise RuntimeError(
        "Bootstrap parquet round-trip failed."
    )


print(
    "Prevalence round-trip:",
    "PASS"
)

print(
    "AR-population round-trip:",
    "PASS"
)

print(
    "Group-summary round-trip:",
    "PASS"
)

print(
    "Bootstrap round-trip:",
    "PASS"
)


# ============================================================
# CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Label Prevalence and "
            "Active-Region Population Audit"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_PREVALENCE_AR_POPULATION_RESULTS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_9_3",

    "scope_order":
        SCOPE_ORDER,

    "reference_scope":
        "C24_HOLDOUT",

    "transfer_scopes":
        [
            *PRIMARY_BLOCKS,
            POOLED_SCOPE,
        ],

    "label_prevalence":
        {
            "definition":
                "positive rows / total rows",

            "target":
                TARGET_COLUMN,

            "bootstrap_unit":
                "ENTITY_GROUP",

            "bootstrap_replicates":
                BOOTSTRAP_REPLICATES,

            "base_seed":
                BOOTSTRAP_SEED,

            "rng":
                (
                    "numpy.random.SeedSequence "
                    "+ default_rng"
                ),

            "domains_resampled":
                (
                    "Each scope independently "
                    "resampled at ENTITY_GROUP level."
                ),

            "group_resampling":
                (
                    "Draw G ENTITY_GROUP clusters "
                    "with replacement from a scope "
                    "containing G unique groups."
                ),

            "confidence_interval":
                {
                    "method":
                        "percentile",

                    "confidence_level":
                        0.95,

                    "lower_quantile":
                        LOWER_Q,

                    "upper_quantile":
                        UPPER_Q,

                    "numpy_quantile_method":
                        "linear",
                },

            "scope_rng_streams":
                seed_registry,
        },

    "active_region_population":
        {
            "unit":
                "ENTITY_GROUP",

            "row_frequency_metrics":
                [
                    "mean",
                    "q25",
                    "median",
                    "q75",
                    "iqr",
                    "min",
                    "max",
                ],

            "positive_group_metrics":
                [
                    "positive_groups",
                    "zero_positive_groups",
                    "positive_group_fraction",
                ],

            "group_positive_rate_metrics":
                [
                    "mean",
                    "q25",
                    "median",
                    "q75",
                    "iqr",
                    "min",
                    "max",
                ],

            "cross_cycle_entity_group_overlap":
                len(
                    cross_cycle_overlap
                ),

            "cross_cycle_overlap_expected":
                expected_overlap,
        },

    "artifacts":
        {
            "label_prevalence":
                {
                    "file":
                        str(
                            OUTPUT_PREVALENCE_FILE
                        ),

                    "sha256":
                        prevalence_hash,

                    "rows":
                        len(
                            prevalence_df
                        ),
                },

            "active_region_population":
                {
                    "file":
                        str(
                            OUTPUT_AR_POPULATION_FILE
                        ),

                    "sha256":
                        population_hash,

                    "rows":
                        len(
                            population_df
                        ),
                },

            "active_region_group_summary":
                {
                    "file":
                        str(
                            OUTPUT_GROUP_SUMMARY_FILE
                        ),

                    "sha256":
                        group_summary_hash,

                    "rows":
                        len(
                            group_summary_df
                        ),
                },

            "prevalence_bootstrap_replicates":
                {
                    "file":
                        str(
                            OUTPUT_BOOTSTRAP_FILE
                        ),

                    "sha256":
                        bootstrap_hash,

                    "rows":
                        len(
                            bootstrap_df
                        ),
                },
        },

    "data_access":
        {
            "cycle24_holdout_read":
                True,

            "cycle25_read":
                True,

            "cycle24_train_read":
                False,

            "models_loaded":
                False,

            "models_retrained":
                False,

            "prediction_artifacts_read":
                False,

            "shap_computed":
                False,
        },

    "evaluation_state":
        {
            "label_prevalence_audit":
                True,

            "active_region_population_audit":
                True,

            "marginal_shift_results_altered":
                False,

            "joint_domain_classifier":
                False,

            "common_support":
                False,

            "correlated_feature_groups":
                False,

            "shap_null":
                False,

            "cross_cycle_shap":
                False,
        },

    "provenance":
        {
            "chapter_9_1_protocol":
                {
                    "file":
                        str(
                            PROTOCOL_FILE
                        ),

                    "sha256":
                        sha256_file(
                            PROTOCOL_FILE
                        ),
                },

            "cycle25_temporal_blocks":
                {
                    "file":
                        str(
                            CYCLE25_BLOCKS_FILE
                        ),

                    "sha256":
                        sha256_file(
                            CYCLE25_BLOCKS_FILE
                        ),
                },

            "cycle24_holdout":
                {
                    "file":
                        str(
                            CYCLE24_HOLDOUT_FILE
                        ),

                    "sha256":
                        c24_hash,
                },

            "cycle25_dataset":
                {
                    "file":
                        str(
                            CYCLE25_FILE
                        ),

                    "sha256":
                        c25_hash,
                },
        },

    "interpretation_rule":
        (
            "Differences in label prevalence and "
            "ENTITY_GROUP population composition "
            "are treated as potential contextual "
            "or confounding differences between "
            "cycles. They are not by themselves "
            "evidence of predictive drift, "
            "explanation drift, or altered "
            "solar-physics causality."
        ),

    "ci_interpretation_rule":
        (
            "Scope-specific prevalence bootstrap "
            "confidence intervals quantify "
            "ENTITY_GROUP-level sampling "
            "uncertainty. Overlap or non-overlap "
            "between marginal scope CIs is not used "
            "as a formal hypothesis test of "
            "cross-cycle prevalence differences."
        ),

    "immutability_rule":
        (
            "Observed prevalence or active-region "
            "population differences must not alter "
            "the frozen model, feature contract, "
            "Cycle25 temporal blocks, shift metrics, "
            "common-support design, correlated-"
            "feature grouping rule, SHAP background, "
            "or within-Cycle24 SHAP null design."
        ),
}


with OUTPUT_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "CHAPTER 9.3 SUMMARY"
)

print(
    "===================================="
)


print(
    "Scopes:",
    len(
        SCOPE_ORDER
    )
)

print(
    "Prevalence rows:",
    len(
        prevalence_df
    )
)

print(
    "AR population rows:",
    len(
        population_df
    )
)

print(
    "Group summary rows:",
    len(
        group_summary_df
    )
)

print(
    "Bootstrap replicates/scope:",
    BOOTSTRAP_REPLICATES
)

print(
    "Total prevalence bootstrap rows:",
    len(
        bootstrap_df
    )
)

print(
    "Cross-cycle ENTITY_GROUP overlap:",
    len(
        cross_cycle_overlap
    )
)

print(
    "Cycle24 TRAIN read:",
    False
)

print(
    "Models loaded/retrained:",
    False
)

print(
    "SHAP computed:",
    False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_PREVALENCE_FILE
)

print(
    OUTPUT_AR_POPULATION_FILE
)

print(
    OUTPUT_GROUP_SUMMARY_FILE
)

print(
    OUTPUT_BOOTSTRAP_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 9.3 COMPLETE — PREVALENCE + AR POPULATION FROZEN"
)

print(
    "===================================="
)