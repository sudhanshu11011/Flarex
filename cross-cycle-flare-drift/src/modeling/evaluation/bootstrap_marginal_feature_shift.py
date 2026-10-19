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

POINT_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "marginal_feature_shift_point.csv"
)

POINT_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "marginal_feature_shift_point_contract.json"
)

FEATURE_REGISTRY_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "shift_feature_registry.csv"
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


OUTPUT_REPLICATES_FILE = Path(
    "artifacts/models/shift_controls/bootstrap/"
    "marginal_shift_group_bootstrap_replicates.parquet"
)

OUTPUT_CI_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "marginal_feature_shift_bootstrap_ci.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "marginal_feature_shift_bootstrap_contract.json"
)


# ============================================================
# FROZEN EXPECTATIONS
# ============================================================

EXPECTED_C24_SHA256 = (
    "8abb3929febcde7d778c4781ac890ff5e"
    "658933c2ef2a06aa19da085e7e255fc"
)

EXPECTED_C25_SHA256 = (
    "0bd6a364e152685d13c7eb955da074a4"
    "b9f019f1851c6b5dafebf295c406cd4d"
)

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TARGET_COLUMN = "target"
TIME_COLUMN = "observation_time_utc"

VALID_REPLICATES = 2000
LOWER_Q = 0.025
UPPER_Q = 0.975

BASE_BOOTSTRAP_SEED = 29021
BATCH_SIZE = 25


PRIMARY_BLOCKS = [
    "C25_B1_2020_2021",
    "C25_B2_2022_2023",
    "C25_B3_2024_2025",
]

POOLED_SCOPE = "C25_ALL_2020_2025"

TRANSFER_SCOPES = (
    PRIMARY_BLOCKS
    +
    [POOLED_SCOPE]
)

STRATA = [
    "OVERALL",
    "TARGET_0",
    "TARGET_1",
]

METRICS = [
    "ks_statistic",
    "wasserstein_distance",
    "normalized_wasserstein",
    "median_shift_iqr_units",
]


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


def get_stratum_mask(
    dataframe,
    stratum,
):

    if stratum == "OVERALL":

        return np.ones(
            len(dataframe),
            dtype=bool,
        )

    target = dataframe[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.int64
    )

    if stratum == "TARGET_0":
        return target == 0

    if stratum == "TARGET_1":
        return target == 1

    raise RuntimeError(
        f"Unknown stratum: {stratum}"
    )


def exact_weighted_median_from_sorted(
    sorted_values,
    sorted_row_weights,
):

    """
    Exact equivalent of:
        np.quantile(expanded_integer_weight_sample, 0.5,
                    method='linear')

    without physically expanding duplicated rows.
    """

    cumulative = np.cumsum(
        sorted_row_weights,
        axis=1,
    )

    totals = cumulative[
        :,
        -1,
    ].astype(
        np.int64
    )

    h = (
        totals.astype(
            np.float64
        )
        -
        1.0
    ) * 0.5

    lower_rank = np.floor(
        h
    ).astype(
        np.int64
    )

    upper_rank = np.ceil(
        h
    ).astype(
        np.int64
    )

    fraction = (
        h
        -
        lower_rank
    )

    # 0-indexed rank r corresponds to
    # first cumulative count > r.

    lower_index = np.argmax(
        cumulative
        >
        lower_rank[
            :,
            None
        ],
        axis=1,
    )

    upper_index = np.argmax(
        cumulative
        >
        upper_rank[
            :,
            None
        ],
        axis=1,
    )

    lower_value = sorted_values[
        lower_index
    ]

    upper_value = sorted_values[
        upper_index
    ]

    return (
        lower_value
        +
        fraction
        *
        (
            upper_value
            -
            lower_value
        )
    )


def bootstrap_feature_batch(
    reference_values,
    reference_group_codes,
    transfer_values,
    transfer_group_codes,
    reference_multiplicity,
    transfer_multiplicity,
    frozen_reference_iqr,
):

    reference_values = np.asarray(
        reference_values,
        dtype=np.float64,
    )

    transfer_values = np.asarray(
        transfer_values,
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # SORT ONCE — BOOTSTRAP ONLY CHANGES GROUP WEIGHTS
    # --------------------------------------------------------

    ref_order = np.argsort(
        reference_values,
        kind="mergesort",
    )

    tr_order = np.argsort(
        transfer_values,
        kind="mergesort",
    )

    ref_sorted = reference_values[
        ref_order
    ]

    tr_sorted = transfer_values[
        tr_order
    ]

    ref_codes_sorted = reference_group_codes[
        ref_order
    ]

    tr_codes_sorted = transfer_group_codes[
        tr_order
    ]


    # --------------------------------------------------------
    # FIXED COMBINED SUPPORT FOR KS + WASSERSTEIN
    # --------------------------------------------------------

    support = np.unique(
        np.concatenate(
            [
                ref_sorted,
                tr_sorted,
            ]
        )
    )


    if len(
        support
    ) == 0:

        raise RuntimeError(
            "Empty feature support."
        )


    ref_support_index = (
        np.searchsorted(
            ref_sorted,
            support,
            side="right",
        )
        -
        1
    )

    tr_support_index = (
        np.searchsorted(
            tr_sorted,
            support,
            side="right",
        )
        -
        1
    )


    support_widths = np.diff(
        support
    )


    replicate_count = int(
        reference_multiplicity.shape[
            0
        ]
    )


    ks_output = np.empty(
        replicate_count,
        dtype=np.float64,
    )

    wasserstein_output = np.empty(
        replicate_count,
        dtype=np.float64,
    )

    norm_wasserstein_output = np.empty(
        replicate_count,
        dtype=np.float64,
    )

    median_shift_output = np.empty(
        replicate_count,
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # BATCH TO CONTROL MEMORY
    # --------------------------------------------------------

    for batch_start in range(
        0,
        replicate_count,
        BATCH_SIZE,
    ):

        batch_end = min(
            batch_start
            +
            BATCH_SIZE,
            replicate_count,
        )


        ref_mult = reference_multiplicity[
            batch_start:
            batch_end
        ]

        tr_mult = transfer_multiplicity[
            batch_start:
            batch_end
        ]


        ref_weights = ref_mult[
            :,
            ref_codes_sorted
        ].astype(
            np.int32,
            copy=False,
        )

        tr_weights = tr_mult[
            :,
            tr_codes_sorted
        ].astype(
            np.int32,
            copy=False,
        )


        ref_total = ref_weights.sum(
            axis=1
        )

        tr_total = tr_weights.sum(
            axis=1
        )


        if (
            ref_total
            <=
            0
        ).any():

            raise RuntimeError(
                "Zero reference row weight "
                "in bootstrap replicate."
            )


        if (
            tr_total
            <=
            0
        ).any():

            raise RuntimeError(
                "Zero transfer row weight "
                "in bootstrap replicate."
            )


        ref_cumulative_counts = np.cumsum(
            ref_weights,
            axis=1,
        )

        tr_cumulative_counts = np.cumsum(
            tr_weights,
            axis=1,
        )


        ref_cdf_sorted = (
            ref_cumulative_counts
            /
            ref_total[
                :,
                None
            ]
        )

        tr_cdf_sorted = (
            tr_cumulative_counts
            /
            tr_total[
                :,
                None
            ]
        )


        batch_n = (
            batch_end
            -
            batch_start
        )


        ref_cdf_support = np.zeros(
            (
                batch_n,
                len(
                    support
                ),
            ),
            dtype=np.float64,
        )

        tr_cdf_support = np.zeros(
            (
                batch_n,
                len(
                    support
                ),
            ),
            dtype=np.float64,
        )


        ref_valid = (
            ref_support_index
            >=
            0
        )

        tr_valid = (
            tr_support_index
            >=
            0
        )


        ref_cdf_support[
            :,
            ref_valid
        ] = ref_cdf_sorted[
            :,
            ref_support_index[
                ref_valid
            ]
        ]


        tr_cdf_support[
            :,
            tr_valid
        ] = tr_cdf_sorted[
            :,
            tr_support_index[
                tr_valid
            ]
        ]


        absolute_cdf_difference = np.abs(
            ref_cdf_support
            -
            tr_cdf_support
        )


        ks_batch = np.max(
            absolute_cdf_difference,
            axis=1,
        )


        if len(
            support
        ) > 1:

            wasserstein_batch = np.sum(
                absolute_cdf_difference[
                    :,
                    :-1
                ]
                *
                support_widths[
                    None,
                    :
                ],
                axis=1,
            )

        else:

            wasserstein_batch = np.zeros(
                batch_n,
                dtype=np.float64,
            )


        ref_median = exact_weighted_median_from_sorted(
            ref_sorted,
            ref_weights,
        )


        tr_median = exact_weighted_median_from_sorted(
            tr_sorted,
            tr_weights,
        )


        normalized_wasserstein_batch = (
            wasserstein_batch
            /
            frozen_reference_iqr
        )


        median_shift_batch = (
            tr_median
            -
            ref_median
        ) / frozen_reference_iqr


        ks_output[
            batch_start:
            batch_end
        ] = ks_batch

        wasserstein_output[
            batch_start:
            batch_end
        ] = wasserstein_batch

        norm_wasserstein_output[
            batch_start:
            batch_end
        ] = normalized_wasserstein_batch

        median_shift_output[
            batch_start:
            batch_end
        ] = median_shift_batch


    return {
        "ks_statistic":
            ks_output,

        "wasserstein_distance":
            wasserstein_output,

        "normalized_wasserstein":
            norm_wasserstein_output,

        "median_shift_iqr_units":
            median_shift_output,
    }


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 9.2B — MARGINAL SHIFT ENTITY_GROUP BOOTSTRAP"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    PROTOCOL_FILE,
    POINT_FILE,
    POINT_CONTRACT_FILE,
    FEATURE_REGISTRY_FILE,
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
    OUTPUT_REPLICATES_FILE,
    OUTPUT_CI_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter 9.2B results."
        )


# ============================================================
# CONTRACTS
# ============================================================

protocol = load_json(
    PROTOCOL_FILE
)

point_contract = load_json(
    POINT_CONTRACT_FILE
)


if protocol.get(
    "status"
) != "FROZEN_BEFORE_SHIFT_AND_SHAP_ANALYSIS":

    raise RuntimeError(
        "Chapter 9.1 protocol "
        "is not frozen."
    )


if point_contract.get(
    "status"
) != "FROZEN_MARGINAL_SHIFT_POINT_RESULTS":

    raise RuntimeError(
        "Chapter 9.2A point results "
        "are not frozen."
    )


bootstrap_protocol = (
    protocol[
        "marginal_distribution_shift"
    ][
        "uncertainty"
    ]
)


if bootstrap_protocol[
    "unit"
] != "ENTITY_GROUP":

    raise RuntimeError(
        "Bootstrap unit mismatch."
    )


if int(
    bootstrap_protocol[
        "replicates"
    ]
) != VALID_REPLICATES:

    raise RuntimeError(
        "Frozen bootstrap replicate "
        "count mismatch."
    )


if int(
    bootstrap_protocol[
        "seed"
    ]
) != BASE_BOOTSTRAP_SEED:

    raise RuntimeError(
        "Frozen bootstrap seed mismatch."
    )


if not bootstrap_protocol[
    "same_group_draws_across_features"
]:

    raise RuntimeError(
        "Frozen protocol requires same "
        "group draws across features."
    )


print(
    "Chapter 9.1 protocol:",
    "FROZEN"
)

print(
    "Chapter 9.2A point results:",
    "FROZEN"
)

print(
    "Bootstrap unit:",
    "ENTITY_GROUP"
)

print(
    "Replicates/comparison:",
    VALID_REPLICATES
)

print(
    "Same group draws across features:",
    True
)


# ============================================================
# HASH VERIFICATION
# ============================================================

point_hash = sha256_file(
    POINT_FILE
)


if (
    point_hash
    !=
    point_contract[
        "artifacts"
    ][
        "point_shift_results"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "9.2A point-result hash mismatch."
    )


c24_hash = sha256_file(
    CYCLE24_HOLDOUT_FILE
)

c25_hash = sha256_file(
    CYCLE25_FILE
)


if c24_hash != EXPECTED_C24_SHA256:

    raise RuntimeError(
        "Cycle24 holdout hash mismatch."
    )


if c25_hash != EXPECTED_C25_SHA256:

    raise RuntimeError(
        "Cycle25 hash mismatch."
    )


print(
    "9.2A point-result hash:",
    "PASS"
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
# LOAD DATA
# ============================================================

point_results = pd.read_csv(
    POINT_FILE,
    keep_default_na=False,
)

c24 = pd.read_parquet(
    CYCLE24_HOLDOUT_FILE
)

c25 = pd.read_parquet(
    CYCLE25_FILE
)


if len(
    point_results
) != 300:

    raise RuntimeError(
        "Expected 300 frozen "
        "point comparisons."
    )


# ============================================================
# ASSIGN FROZEN CYCLE25 BLOCKS
# ============================================================

blocks = pd.read_csv(
    CYCLE25_BLOCKS_FILE,
    keep_default_na=False,
)


primary_blocks = blocks[
    blocks[
        "primary_block"
    ].astype(str).str.lower()
    ==
    "true"
].copy()


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


    block_id.loc[
        mask
    ] = current_id


if block_id.isna().any():

    raise RuntimeError(
        "Cycle25 temporal assignment failed."
    )


c25 = c25.copy()

c25[
    "_C25_BLOCK_ID"
] = block_id.astype(str)


scope_frames = {}


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
# FIXED COMPARISON ORDER + CHILD RNG STREAMS
# ============================================================

comparison_keys = [
    (
        scope_id,
        stratum,
    )
    for scope_id in TRANSFER_SCOPES
    for stratum in STRATA
]


if len(
    comparison_keys
) != 12:

    raise RuntimeError(
        "Expected exactly 12 "
        "scope-stratum comparisons."
    )


seed_sequence = np.random.SeedSequence(
    BASE_BOOTSTRAP_SEED
)


child_sequences = seed_sequence.spawn(
    len(
        comparison_keys
    )
)


child_seed_registry = {}


for index, key in enumerate(
    comparison_keys
):

    generated_seed = int(
        child_sequences[
            index
        ].generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


    child_seed_registry[
        f"{key[0]}__{key[1]}"
    ] = {
        "comparison_index":
            index,

        "derived_seed_uint32":
            generated_seed,

        "spawn_key":
            list(
                child_sequences[
                    index
                ].spawn_key
            ),
    }


# ============================================================
# BOOTSTRAP
# ============================================================

replicate_frames = []
ci_rows = []
comparison_summary = []


print(
    "\n===================================="
)

print(
    "RUNNING MARGINAL-SHIFT BOOTSTRAP"
)

print(
    "===================================="
)


for comparison_index, (
    transfer_scope,
    stratum,
) in enumerate(
    comparison_keys
):

    reference_mask = get_stratum_mask(
        c24,
        stratum,
    )

    transfer_frame = scope_frames[
        transfer_scope
    ]


    transfer_mask = get_stratum_mask(
        transfer_frame,
        stratum,
    )


    reference = c24.loc[
        reference_mask
    ].reset_index(
        drop=True
    )


    transfer = transfer_frame.loc[
        transfer_mask
    ].reset_index(
        drop=True
    )


    if len(reference) == 0:

        raise RuntimeError(
            f"{stratum}: empty Cycle24 "
            "reference population."
        )


    if len(transfer) == 0:

        raise RuntimeError(
            f"{transfer_scope}/{stratum}: "
            "empty Cycle25 population."
        )


    # --------------------------------------------------------
    # GROUP CODES
    # --------------------------------------------------------

    reference_group_codes, reference_groups = (
        pd.factorize(
            reference[
                ENTITY_GROUP_COLUMN
            ],
            sort=True,
        )
    )


    transfer_group_codes, transfer_groups = (
        pd.factorize(
            transfer[
                ENTITY_GROUP_COLUMN
            ],
            sort=True,
        )
    )


    reference_group_codes = np.asarray(
        reference_group_codes,
        dtype=np.int64,
    )

    transfer_group_codes = np.asarray(
        transfer_group_codes,
        dtype=np.int64,
    )


    G_reference = int(
        len(
            reference_groups
        )
    )

    G_transfer = int(
        len(
            transfer_groups
        )
    )


    if G_reference <= 0:

        raise RuntimeError(
            "Reference ENTITY_GROUP count "
            "must be positive."
        )


    if G_transfer <= 0:

        raise RuntimeError(
            "Transfer ENTITY_GROUP count "
            "must be positive."
        )


    # --------------------------------------------------------
    # ONE RNG STREAM PER SCOPE-STRATUM
    # --------------------------------------------------------

    rng = np.random.default_rng(
        child_sequences[
            comparison_index
        ]
    )


    # Multinomial(G, 1/G) is exactly
    # the count representation of
    # drawing G groups with replacement.

    reference_multiplicity = rng.multinomial(
        G_reference,
        np.full(
            G_reference,
            1.0 / G_reference,
            dtype=np.float64,
        ),
        size=VALID_REPLICATES,
    ).astype(
        np.int16
    )


    transfer_multiplicity = rng.multinomial(
        G_transfer,
        np.full(
            G_transfer,
            1.0 / G_transfer,
            dtype=np.float64,
        ),
        size=VALID_REPLICATES,
    ).astype(
        np.int16
    )


    if not (
        reference_multiplicity.sum(
            axis=1
        )
        ==
        G_reference
    ).all():

        raise RuntimeError(
            "Reference group-draw count "
            "integrity failed."
        )


    if not (
        transfer_multiplicity.sum(
            axis=1
        )
        ==
        G_transfer
    ).all():

        raise RuntimeError(
            "Transfer group-draw count "
            "integrity failed."
        )


    print(
        f"\n{transfer_scope} | {stratum}: "
        f"C24 rows={len(reference)}, "
        f"C24 groups={G_reference}, "
        f"C25 rows={len(transfer)}, "
        f"C25 groups={G_transfer}"
    )


    # ========================================================
    # ALL FEATURES — SAME GROUP DRAWS
    # ========================================================

    for feature_index, feature in enumerate(
        FEATURES,
        start=1,
    ):

        official = point_results[
            (
                point_results[
                    "transfer_scope"
                ].astype(str)
                ==
                transfer_scope
            )
            &
            (
                point_results[
                    "stratum"
                ].astype(str)
                ==
                stratum
            )
            &
            (
                point_results[
                    "feature"
                ].astype(str)
                ==
                feature
            )
        ]


        if len(
            official
        ) != 1:

            raise RuntimeError(
                f"{transfer_scope}/"
                f"{stratum}/{feature}: "
                "point-result row missing."
            )


        official = official.iloc[
            0
        ]


        frozen_reference_iqr = float(
            official[
                "reference_iqr"
            ]
        )


        if frozen_reference_iqr <= 0.0:

            raise RuntimeError(
                f"{stratum}/{feature}: "
                "frozen Cycle24 reference "
                "IQR is not positive."
            )


        reference_values = reference[
            feature
        ].to_numpy(
            dtype=np.float64
        )


        transfer_values = transfer[
            feature
        ].to_numpy(
            dtype=np.float64
        )


        bootstrap_metrics = bootstrap_feature_batch(
            reference_values,
            reference_group_codes,
            transfer_values,
            transfer_group_codes,
            reference_multiplicity,
            transfer_multiplicity,
            frozen_reference_iqr,
        )


        replicate_frame = pd.DataFrame(
            {
                "comparison_index":
                    np.full(
                        VALID_REPLICATES,
                        comparison_index,
                        dtype=np.int16,
                    ),

                "transfer_scope":
                    transfer_scope,

                "stratum":
                    stratum,

                "feature_index":
                    np.full(
                        VALID_REPLICATES,
                        feature_index,
                        dtype=np.int16,
                    ),

                "feature":
                    feature,

                "replicate_id":
                    np.arange(
                        1,
                        VALID_REPLICATES + 1,
                        dtype=np.int32,
                    ),

                "reference_group_draw_count":
                    np.full(
                        VALID_REPLICATES,
                        G_reference,
                        dtype=np.int16,
                    ),

                "transfer_group_draw_count":
                    np.full(
                        VALID_REPLICATES,
                        G_transfer,
                        dtype=np.int16,
                    ),

                "ks_statistic":
                    bootstrap_metrics[
                        "ks_statistic"
                    ],

                "wasserstein_distance":
                    bootstrap_metrics[
                        "wasserstein_distance"
                    ],

                "normalized_wasserstein":
                    bootstrap_metrics[
                        "normalized_wasserstein"
                    ],

                "median_shift_iqr_units":
                    bootstrap_metrics[
                        "median_shift_iqr_units"
                    ],
            }
        )


        replicate_frames.append(
            replicate_frame
        )


        # ----------------------------------------------------
        # CI FOR EACH METRIC
        # ----------------------------------------------------

        for metric in METRICS:

            values = bootstrap_metrics[
                metric
            ]


            if not np.isfinite(
                values
            ).all():

                raise RuntimeError(
                    f"{transfer_scope}/"
                    f"{stratum}/{feature}/"
                    f"{metric}: non-finite "
                    "bootstrap values."
                )


            lower = float(
                np.quantile(
                    values,
                    LOWER_Q,
                    method="linear",
                )
            )


            upper = float(
                np.quantile(
                    values,
                    UPPER_Q,
                    method="linear",
                )
            )


            point_estimate = float(
                official[
                    metric
                ]
            )


            ci_rows.append(
                {
                    "reference_scope":
                        "C24_HOLDOUT",

                    "transfer_scope":
                        transfer_scope,

                    "stratum":
                        stratum,

                    "feature":
                        feature,

                    "metric":
                        metric,

                    "point_estimate":
                        point_estimate,

                    "bootstrap_mean":
                        float(
                            values.mean()
                        ),

                    "bootstrap_median":
                        float(
                            np.median(
                                values
                            )
                        ),

                    "ci_lower_2p5":
                        lower,

                    "ci_upper_97p5":
                        upper,

                    "confidence_level":
                        0.95,

                    "ci_method":
                        "percentile",

                    "valid_replicates":
                        VALID_REPLICATES,

                    "bootstrap_unit":
                        "ENTITY_GROUP",

                    "reference_group_count":
                        G_reference,

                    "transfer_group_count":
                        G_transfer,

                    "normalization_reference_iqr":
                        frozen_reference_iqr,

                    "point_inside_ci":
                        bool(
                            lower
                            <=
                            point_estimate
                            <=
                            upper
                        ),
                }
            )


    comparison_summary.append(
        {
            "comparison_index":
                comparison_index,

            "transfer_scope":
                transfer_scope,

            "stratum":
                stratum,

            "reference_rows":
                int(
                    len(
                        reference
                    )
                ),

            "transfer_rows":
                int(
                    len(
                        transfer
                    )
                ),

            "reference_groups":
                G_reference,

            "transfer_groups":
                G_transfer,

            "replicates":
                VALID_REPLICATES,

            "derived_seed_uint32":
                child_seed_registry[
                    f"{transfer_scope}__{stratum}"
                ][
                    "derived_seed_uint32"
                ],
        }
    )


    print(
        f"{transfer_scope} | {stratum}: "
        f"25 features × {VALID_REPLICATES} "
        f"replicates COMPLETE"
    )


# ============================================================
# BUILD FINAL TABLES
# ============================================================

replicates_df = pd.concat(
    replicate_frames,
    ignore_index=True,
)


ci_df = pd.DataFrame(
    ci_rows
)


EXPECTED_REPLICATE_ROWS = (
    4
    *
    3
    *
    25
    *
    VALID_REPLICATES
)


EXPECTED_CI_ROWS = (
    4
    *
    3
    *
    25
    *
    4
)


if len(
    replicates_df
) != EXPECTED_REPLICATE_ROWS:

    raise RuntimeError(
        "Bootstrap replicate table "
        "row-count mismatch."
    )


if len(
    ci_df
) != EXPECTED_CI_ROWS:

    raise RuntimeError(
        "Bootstrap CI table "
        "row-count mismatch."
    )


if ci_df.duplicated(
    subset=[
        "transfer_scope",
        "stratum",
        "feature",
        "metric",
    ]
).any():

    raise RuntimeError(
        "Duplicate bootstrap CI row."
    )


print(
    "\nBootstrap replicate rows:",
    len(
        replicates_df
    )
)

print(
    "Bootstrap CI rows:",
    len(
        ci_df
    )
)

print(
    "Bootstrap structure:",
    "PASS"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_REPLICATES_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CI_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


replicates_df.to_parquet(
    OUTPUT_REPLICATES_FILE,
    index=False,
)


ci_df.to_csv(
    OUTPUT_CI_FILE,
    index=False,
)


replicate_hash = sha256_file(
    OUTPUT_REPLICATES_FILE
)

ci_hash = sha256_file(
    OUTPUT_CI_FILE
)


# ============================================================
# ROUND-TRIP
# ============================================================

replicate_roundtrip = pd.read_parquet(
    OUTPUT_REPLICATES_FILE
)

ci_roundtrip = pd.read_csv(
    OUTPUT_CI_FILE,
    keep_default_na=False,
)


if len(
    replicate_roundtrip
) != EXPECTED_REPLICATE_ROWS:

    raise RuntimeError(
        "Bootstrap parquet round-trip "
        "row-count mismatch."
    )


if len(
    ci_roundtrip
) != EXPECTED_CI_ROWS:

    raise RuntimeError(
        "Bootstrap CI round-trip "
        "row-count mismatch."
    )


print(
    "Bootstrap parquet round-trip:",
    "PASS"
)

print(
    "Bootstrap CI round-trip:",
    "PASS"
)


# ============================================================
# PRINT MAX NORMALIZED-WASSERSTEIN FEATURE
# ============================================================

print(
    "\n===================================="
)

print(
    "MAX POINT NORMALIZED WASSERSTEIN + 95% CI"
)

print(
    "===================================="
)


for transfer_scope in TRANSFER_SCOPES:

    for stratum in STRATA:

        subset = point_results[
            (
                point_results[
                    "transfer_scope"
                ].astype(str)
                ==
                transfer_scope
            )
            &
            (
                point_results[
                    "stratum"
                ].astype(str)
                ==
                stratum
            )
        ].copy()


        top_index = subset[
            "normalized_wasserstein"
        ].astype(
            float
        ).idxmax()


        top_row = subset.loc[
            top_index
        ]


        top_feature = str(
            top_row[
                "feature"
            ]
        )


        ci_row = ci_df[
            (
                ci_df[
                    "transfer_scope"
                ]
                ==
                transfer_scope
            )
            &
            (
                ci_df[
                    "stratum"
                ]
                ==
                stratum
            )
            &
            (
                ci_df[
                    "feature"
                ]
                ==
                top_feature
            )
            &
            (
                ci_df[
                    "metric"
                ]
                ==
                "normalized_wasserstein"
            )
        ]


        if len(
            ci_row
        ) != 1:

            raise RuntimeError(
                "Top-feature CI lookup failed."
            )


        ci_row = ci_row.iloc[
            0
        ]


        print(
            f"{transfer_scope} | "
            f"{stratum} | "
            f"{top_feature}: "
            f"norm_W="
            f"{float(top_row['normalized_wasserstein']):.6f}, "
            f"95% CI=["
            f"{float(ci_row['ci_lower_2p5']):.6f}, "
            f"{float(ci_row['ci_upper_97p5']):.6f}]"
        )


# ============================================================
# CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Cycle24 Holdout to Cycle25 "
            "Marginal Feature Shift "
            "ENTITY_GROUP Bootstrap"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_MARGINAL_SHIFT_BOOTSTRAP_CI",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_9_2B",

    "reference_domain":
        "CYCLE24_HOLDOUT",

    "transfer_domain":
        "CYCLE25",

    "transfer_scopes":
        TRANSFER_SCOPES,

    "strata":
        STRATA,

    "features":
        FEATURES,

    "feature_count":
        25,

    "point_comparisons":
        300,

    "bootstrap_design":
        {
            "unit":
                "ENTITY_GROUP",

            "domains_resampled_independently":
                True,

            "reference_draw_rule":
                (
                    "Within each stratum, draw "
                    "G_ref Cycle24-holdout "
                    "ENTITY_GROUP clusters with "
                    "replacement."
                ),

            "transfer_draw_rule":
                (
                    "Within each Cycle25 scope "
                    "and stratum, draw G_transfer "
                    "ENTITY_GROUP clusters with "
                    "replacement."
                ),

            "group_count_distribution":
                (
                    "Multinomial count representation "
                    "of G independent group draws "
                    "with replacement."
                ),

            "same_group_draws_across_25_features":
                True,

            "replicates":
                VALID_REPLICATES,

            "base_seed":
                BASE_BOOTSTRAP_SEED,

            "rng":
                "numpy.random.SeedSequence + default_rng",

            "comparison_child_streams":
                child_seed_registry,

            "comparison_order":
                [
                    {
                        "comparison_index":
                            index,

                        "transfer_scope":
                            key[
                                0
                            ],

                        "stratum":
                            key[
                                1
                            ],
                    }

                    for index, key
                    in enumerate(
                        comparison_keys
                    )
                ],
        },

    "metric_implementation":
        {
            "ks_statistic":
                (
                    "Maximum absolute difference "
                    "between cluster-bootstrap "
                    "weighted empirical CDFs."
                ),

            "wasserstein_distance":
                (
                    "Integral of absolute "
                    "difference between weighted "
                    "empirical CDFs over the "
                    "combined fixed feature support."
                ),

            "normalized_wasserstein":
                (
                    "Bootstrap Wasserstein distance "
                    "divided by the corresponding "
                    "full-sample frozen Cycle24-"
                    "holdout stratum IQR."
                ),

            "median_shift_iqr_units":
                (
                    "Bootstrap Cycle25 median minus "
                    "bootstrap Cycle24 median, divided "
                    "by the corresponding full-sample "
                    "frozen Cycle24-holdout stratum IQR."
                ),

            "normalization_denominator_reestimated_in_bootstrap":
                False,
        },

    "confidence_intervals":
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

    "comparison_summary":
        comparison_summary,

    "artifacts":
        {
            "bootstrap_replicates":
                {
                    "file":
                        str(
                            OUTPUT_REPLICATES_FILE
                        ),

                    "sha256":
                        replicate_hash,

                    "rows":
                        len(
                            replicates_df
                        ),
                },

            "bootstrap_ci":
                {
                    "file":
                        str(
                            OUTPUT_CI_FILE
                        ),

                    "sha256":
                        ci_hash,

                    "rows":
                        len(
                            ci_df
                        ),
                },
        },

    "evaluation_state":
        {
            "marginal_point_results":
                True,

            "marginal_bootstrap_ci":
                True,

            "binary_feature_shift_declaration":
                False,

            "joint_domain_classifier":
                False,

            "common_support":
                False,

            "correlated_feature_groups":
                False,

            "shap_values":
                False,

            "shap_null":
                False,

            "explanation_drift":
                False,
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

            "shap_computed":
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

            "chapter_9_2a_point_results":
                {
                    "file":
                        str(
                            POINT_FILE
                        ),

                    "sha256":
                        point_hash,
                },

            "chapter_9_2a_contract":
                {
                    "file":
                        str(
                            POINT_CONTRACT_FILE
                        ),

                    "sha256":
                        sha256_file(
                            POINT_CONTRACT_FILE
                        ),
                },

            "feature_registry":
                {
                    "file":
                        str(
                            FEATURE_REGISTRY_FILE
                        ),

                    "sha256":
                        sha256_file(
                            FEATURE_REGISTRY_FILE
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
            "Bootstrap confidence intervals quantify "
            "ENTITY_GROUP-level sampling uncertainty "
            "for the frozen marginal distribution "
            "discrepancy metrics. Because KS and "
            "Wasserstein distances are non-negative, "
            "their percentile intervals are not "
            "interpreted as null-hypothesis tests of "
            "identical distributions. No binary "
            "feature-shift declaration is made here."
        ),

    "immutability_rule":
        (
            "Observed bootstrap intervals must not "
            "change the frozen shift metrics, "
            "strata, temporal blocks, common-support "
            "design, correlated-feature grouping "
            "rule, SHAP background, or within-Cycle24 "
            "SHAP null-control design."
        ),
}


# ============================================================
# SAVE CONTRACT
# ============================================================

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
    "CHAPTER 9.2B SUMMARY"
)

print(
    "===================================="
)


print(
    "Scope-stratum comparisons:",
    12
)

print(
    "Features/comparison:",
    25
)

print(
    "Replicates/comparison:",
    2000
)

print(
    "Total bootstrap rows:",
    len(
        replicates_df
    )
)

print(
    "Bootstrap CI rows:",
    len(
        ci_df
    )
)

print(
    "Bootstrap unit:",
    "ENTITY_GROUP"
)

print(
    "Reference + transfer independently resampled:",
    True
)

print(
    "Same draws across features:",
    True
)

print(
    "Binary feature-shift declaration:",
    False
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
    OUTPUT_REPLICATES_FILE
)

print(
    OUTPUT_CI_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 9.2B COMPLETE — MARGINAL SHIFT BOOTSTRAP FROZEN"
)

print(
    "===================================="
)