from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

V1_DIR = Path(
    "data/final/dataset_v1"
)

V2_DIR = Path(
    "data/processed/dataset_v2_candidate"
)

MANIFEST_DIR = Path(
    "data_manifests"
)


FILES = {

    "cycle24": {
        "v1":
            V1_DIR
            /
            "cycle24_dataset_v1.parquet",

        "v2":
            V2_DIR
            /
            "cycle24_dataset_v2_candidate.parquet",
    },

    "cycle25": {
        "v1":
            V1_DIR
            /
            "cycle25_dataset_v1.parquet",

        "v2":
            V2_DIR
            /
            "cycle25_dataset_v2_candidate.parquet",
    },
}


CORRECTIONS_FILE = (
    MANIFEST_DIR
    /
    "pre_model_dataset_v2_label_corrections.csv"
)

CANDIDATE_SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "pre_model_dataset_v2_candidate_summary.csv"
)

TEMPORAL_CONTRACT_FILE = (
    MANIFEST_DIR
    /
    "temporal_data_contract.json"
)


AUDIT_OUTPUT = (
    MANIFEST_DIR
    /
    "pre_model_dataset_v2_full_audit.csv"
)

YEAR_OUTPUT = (
    MANIFEST_DIR
    /
    "pre_model_dataset_v2_year_summary.csv"
)

OVERLAP_OUTPUT = (
    MANIFEST_DIR
    /
    "pre_model_dataset_v2_overlap_audit.csv"
)


# ============================================================
# MODEL FEATURE CONTRACT
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


ALL_FEATURES = (
    SHARP_FEATURES
    +
    LORENTZ_FEATURES
)


EXPECTED_FEATURE_COUNT = 25


# ============================================================
# HELPERS
# ============================================================

def parse_trec(value):

    if pd.isna(value):
        return pd.NaT

    text = str(
        value
    ).strip()

    text = text.replace(
        "_TAI",
        ""
    )

    text = (
        text
        .replace(
            ".",
            "-",
            2
        )
        .replace(
            "_",
            "T",
            1
        )
    )

    return pd.to_datetime(
        text,
        format="%Y-%m-%dT%H:%M:%S",
        errors="coerce",
    )


def build_key_frame(df):

    result = df[
        [
            "HARPNUM",
            "T_REC",
        ]
    ].copy()

    result[
        "HARPNUM"
    ] = pd.to_numeric(
        result[
            "HARPNUM"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    return result


def key_set(df):

    keys = build_key_frame(
        df
    )

    return set(
        zip(
            keys[
                "HARPNUM"
            ],
            keys[
                "T_REC"
            ],
        )
    )


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    CORRECTIONS_FILE,
    CANDIDATE_SUMMARY_FILE,
    TEMPORAL_CONTRACT_FILE,
]


for cycle in FILES:

    required_files.extend(
        [
            FILES[
                cycle
            ][
                "v1"
            ],

            FILES[
                cycle
            ][
                "v2"
            ],
        ]
    )


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: "
            f"{path}"
        )


# ============================================================
# LOAD MANIFESTS
# ============================================================

corrections = pd.read_csv(
    CORRECTIONS_FILE
)


candidate_summary = pd.read_csv(
    CANDIDATE_SUMMARY_FILE
)


with TEMPORAL_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    temporal_contract = json.load(
        file
    )


if (
    temporal_contract.get(
        "status"
    )
    !=
    "FROZEN"
):

    raise RuntimeError(
        "Temporal contract is not frozen."
    )


# ============================================================
# CORRECTION MANIFEST VALIDATION
# ============================================================

required_correction_columns = {
    "cycle",
    "HARPNUM",
    "T_REC",
    "original_target",
    "srs_decision",
    "dataset_v2_action",
}


missing = (
    required_correction_columns
    -
    set(
        corrections.columns
    )
)


if missing:

    raise RuntimeError(
        "Correction manifest missing columns: "
        f"{sorted(missing)}"
    )


corrections[
    "HARPNUM"
] = pd.to_numeric(
    corrections[
        "HARPNUM"
    ],
    errors="coerce",
).astype(
    "Int64"
)


correction_duplicates = int(
    corrections[
        [
            "cycle",
            "HARPNUM",
            "T_REC",
        ]
    ]
    .duplicated()
    .sum()
)


if correction_duplicates != 0:

    raise RuntimeError(
        "Duplicate correction keys found."
    )


allowed_actions = {
    "CENSOR_TEMPORAL_AMBIGUITY",
    "RELABEL_NEGATIVE_STALE_ASSOCIATION",
    "CENSOR_SRS_SOURCE_MISSING",
    "CENSOR_UNASSIGNED_M1_AFTER_STALE_ASSOCIATION",
}


unexpected_actions = (
    set(
        corrections[
            "dataset_v2_action"
        ]
        .dropna()
        .unique()
    )
    -
    allowed_actions
)


if unexpected_actions:

    raise RuntimeError(
        "Unexpected correction actions: "
        f"{sorted(unexpected_actions)}"
    )


print(
    "===================================="
)

print(
    "PRE-MODEL FIX J — DATASET v2 FULL AUDIT"
)

print(
    "===================================="
)


print(
    "Correction rows:",
    len(
        corrections
    )
)


print(
    "Correction duplicate keys:",
    correction_duplicates
)


# ============================================================
# DATASET CONFIG
# ============================================================

configs = {

    "cycle24": {

        "expected_cycle_tag":
            "Cycle24",

        "domain_start":
            pd.Timestamp(
                temporal_contract[
                    "cycle24"
                ][
                    "domain_start_utc"
                ]
            ),

        "domain_end":
            pd.Timestamp(
                temporal_contract[
                    "cycle24"
                ][
                    "domain_end_exclusive_utc"
                ]
            ),
    },

    "cycle25": {

        "expected_cycle_tag":
            "Cycle25",

        "domain_start":
            pd.Timestamp(
                temporal_contract[
                    "cycle25"
                ][
                    "domain_start_utc"
                ]
            ),

        "domain_end":
            pd.Timestamp(
                temporal_contract[
                    "cycle25"
                ][
                    "domain_end_exclusive_utc"
                ]
            ),
    },
}


# ============================================================
# AUDIT EACH CYCLE
# ============================================================

audit_rows = []
year_rows = []

loaded_v2 = {}

overall_pass = True


for cycle in [
    "cycle24",
    "cycle25",
]:

    v1 = pd.read_parquet(
        FILES[
            cycle
        ][
            "v1"
        ]
    )


    v2 = pd.read_parquet(
        FILES[
            cycle
        ][
            "v2"
        ]
    )


    loaded_v2[
        cycle
    ] = v2


    print(
        "\n===================================="
    )

    print(
        cycle.upper()
    )

    print(
        "===================================="
    )


    # ========================================================
    # SCHEMA
    # ========================================================

    schema_identical_to_v1 = (
        list(
            v1.columns
        )
        ==
        list(
            v2.columns
        )
    )


    dtype_identical_to_v1 = (
        [
            str(
                v1[
                    column
                ].dtype
            )

            for column
            in v1.columns
        ]
        ==
        [
            str(
                v2[
                    column
                ].dtype
            )

            for column
            in v2.columns
        ]
    )


    feature_count = sum(
        feature
        in
        v2.columns

        for feature
        in ALL_FEATURES
    )


    feature_contract_ok = (
        feature_count
        ==
        EXPECTED_FEATURE_COUNT
    )


    # ========================================================
    # BASIC INTEGRITY
    # ========================================================

    duplicate_keys = int(
        v2[
            [
                "HARPNUM",
                "T_REC",
            ]
        ]
        .duplicated()
        .sum()
    )


    invalid_targets = int(
        (
            ~v2[
                "target"
            ]
            .isin(
                [
                    0,
                    1,
                ]
            )
        )
        .sum()
    )


    invalid_label_status = int(
        (
            ~v2[
                "label_status"
            ]
            .isin(
                [
                    "POSITIVE",
                    "NEGATIVE",
                ]
            )
        )
        .sum()
    )


    target_status_mismatch = int(
        (
            (
                v2[
                    "target"
                ]
                ==
                1
            )
            &
            (
                v2[
                    "label_status"
                ]
                !=
                "POSITIVE"
            )
        )
        .sum()
        +
        (
            (
                v2[
                    "target"
                ]
                ==
                0
            )
            &
            (
                v2[
                    "label_status"
                ]
                !=
                "NEGATIVE"
            )
        )
        .sum()
    )


    missing_groups = int(
        v2[
            "ENTITY_GROUP_ID"
        ]
        .isna()
        .sum()
    )


    blank_groups = int(
        (
            v2[
                "ENTITY_GROUP_ID"
            ]
            .astype(
                "string"
            )
            .str.strip()
            ==
            ""
        )
        .sum()
    )


    numeric_features = (
        v2[
            ALL_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )


    finite_array = np.isfinite(
        numeric_features.to_numpy(
            dtype=float
        )
    )


    nonfinite_rows = int(
        (
            ~finite_array.all(
                axis=1
            )
        )
        .sum()
    )


    # ========================================================
    # FIXED 120-MINUTE GRID
    # ========================================================

    trec = v2[
        "T_REC"
    ].apply(
        parse_trec
    )


    invalid_trec = int(
        trec.isna().sum()
    )


    valid_trec = trec.dropna()


    off_grid_rows = int(
        (
            (
                valid_trec.dt.minute
                !=
                0
            )
            |
            (
                valid_trec.dt.second
                !=
                0
            )
            |
            (
                valid_trec.dt.hour
                %
                2
                !=
                0
            )
        )
        .sum()
    )


    # ========================================================
    # UTC DOMAIN + 24H HORIZON
    # ========================================================

    observation_utc = pd.to_datetime(
        v2[
            "observation_time_utc"
        ],
        utc=True,
        errors="coerce",
    )


    invalid_utc = int(
        observation_utc
        .isna()
        .sum()
    )


    domain_start = configs[
        cycle
    ][
        "domain_start"
    ]


    domain_end = configs[
        cycle
    ][
        "domain_end"
    ]


    outside_start = int(
        (
            observation_utc
            <
            domain_start
        )
        .sum()
    )


    outside_end = int(
        (
            observation_utc
            >=
            domain_end
        )
        .sum()
    )


    horizon_end = (
        observation_utc
        +
        pd.Timedelta(
            hours=24
        )
    )


    invalid_horizon = int(
        (
            horizon_end
            >=
            domain_end
        )
        .sum()
    )


    # ========================================================
    # CYCLE TAG
    # ========================================================

    wrong_cycle_tag = int(
        (
            v2[
                "cycle"
            ]
            !=
            configs[
                cycle
            ][
                "expected_cycle_tag"
            ]
        )
        .sum()
    )


    # ========================================================
    # CANDIDATE SUMMARY RECONCILIATION
    # ========================================================

    summary_row = candidate_summary[
        candidate_summary[
            "cycle"
        ]
        ==
        cycle
    ]


    if len(
        summary_row
    ) != 1:

        raise RuntimeError(
            f"{cycle}: candidate summary "
            "row missing or duplicated."
        )


    summary_row = summary_row.iloc[
        0
    ]


    actual_rows = len(
        v2
    )


    actual_positive = int(
        (
            v2[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    actual_negative = int(
        (
            v2[
                "target"
            ]
            ==
            0
        )
        .sum()
    )


    expected_rows = int(
        summary_row[
            "dataset_v2_candidate_rows"
        ]
    )


    expected_positive = int(
        summary_row[
            "dataset_v2_candidate_positive"
        ]
    )


    expected_negative = int(
        summary_row[
            "dataset_v2_candidate_negative"
        ]
    )


    summary_reconciles = (
        actual_rows
        ==
        expected_rows
        and
        actual_positive
        ==
        expected_positive
        and
        actual_negative
        ==
        expected_negative
    )


    # ========================================================
    # V1 → V2 LINEAGE CHECK
    # ========================================================

    v1_keys = key_set(
        v1
    )


    v2_keys = key_set(
        v2
    )


    new_keys = (
        v2_keys
        -
        v1_keys
    )


    removed_keys = (
        v1_keys
        -
        v2_keys
    )


    cycle_corrections = corrections[
        corrections[
            "cycle"
        ]
        ==
        cycle
    ].copy()


    censor_actions = {
        "CENSOR_TEMPORAL_AMBIGUITY",
        "CENSOR_SRS_SOURCE_MISSING",
        "CENSOR_UNASSIGNED_M1_AFTER_STALE_ASSOCIATION",
    }


    censored = cycle_corrections[
        cycle_corrections[
            "dataset_v2_action"
        ]
        .isin(
            censor_actions
        )
    ].copy()


    relabeled = cycle_corrections[
        cycle_corrections[
            "dataset_v2_action"
        ]
        ==
        "RELABEL_NEGATIVE_STALE_ASSOCIATION"
    ].copy()


    censored_keys = set(
        zip(
            censored[
                "HARPNUM"
            ],
            censored[
                "T_REC"
            ],
        )
    )


    relabeled_keys = set(
        zip(
            relabeled[
                "HARPNUM"
            ],
            relabeled[
                "T_REC"
            ],
        )
    )


    removed_keys_match_censor = (
        removed_keys
        ==
        censored_keys
    )


    no_new_keys = (
        len(
            new_keys
        )
        ==
        0
    )


    # ========================================================
    # VERIFY RELABELED ROWS
    # ========================================================

    v2_indexed = v2.copy()


    v2_indexed[
        "HARPNUM"
    ] = pd.to_numeric(
        v2_indexed[
            "HARPNUM"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )


    relabel_target_failures = 0


    for (
        harpnum,
        t_rec,
    ) in relabeled_keys:

        row = v2_indexed[
            (
                v2_indexed[
                    "HARPNUM"
                ]
                ==
                harpnum
            )
            &
            (
                v2_indexed[
                    "T_REC"
                ]
                ==
                t_rec
            )
        ]


        if len(
            row
        ) != 1:

            relabel_target_failures += 1
            continue


        if (
            int(
                row.iloc[
                    0
                ][
                    "target"
                ]
            )
            !=
            0
        ):

            relabel_target_failures += 1
            continue


        if (
            row.iloc[
                0
            ][
                "label_status"
            ]
            !=
            "NEGATIVE"
        ):

            relabel_target_failures += 1


    # ========================================================
    # VERIFY ORIGINAL NEGATIVES ALL SURVIVE
    # ========================================================

    v1_negative_keys = key_set(
        v1[
            v1[
                "target"
            ]
            ==
            0
        ]
    )


    missing_original_negatives = (
        v1_negative_keys
        -
        v2_keys
    )


    # ========================================================
    # YEAR SUMMARY
    # ========================================================

    year_df = v2.copy()


    year_df[
        "_year"
    ] = observation_utc.dt.year


    for year, group in year_df.groupby(
        "_year"
    ):

        positive = int(
            (
                group[
                    "target"
                ]
                ==
                1
            )
            .sum()
        )


        negative = int(
            (
                group[
                    "target"
                ]
                ==
                0
            )
            .sum()
        )


        year_rows.append(
            {
                "cycle":
                    cycle,

                "year":
                    int(
                        year
                    ),

                "rows":
                    len(
                        group
                    ),

                "positive":
                    positive,

                "negative":
                    negative,

                "positive_rate_pct":
                    (
                        positive
                        /
                        len(
                            group
                        )
                        *
                        100
                    ),

                "unique_entity_groups":
                    int(
                        group[
                            "ENTITY_GROUP_ID"
                        ]
                        .nunique()
                    ),

                "unique_harpnums":
                    int(
                        group[
                            "HARPNUM"
                        ]
                        .nunique()
                    ),
            }
        )


    # ========================================================
    # FINAL CYCLE PASS
    # ========================================================

    cycle_pass = (

        schema_identical_to_v1
        and
        dtype_identical_to_v1
        and
        feature_contract_ok
        and
        duplicate_keys == 0
        and
        invalid_targets == 0
        and
        invalid_label_status == 0
        and
        target_status_mismatch == 0
        and
        missing_groups == 0
        and
        blank_groups == 0
        and
        nonfinite_rows == 0
        and
        invalid_trec == 0
        and
        off_grid_rows == 0
        and
        invalid_utc == 0
        and
        outside_start == 0
        and
        outside_end == 0
        and
        invalid_horizon == 0
        and
        wrong_cycle_tag == 0
        and
        summary_reconciles
        and
        no_new_keys
        and
        removed_keys_match_censor
        and
        relabel_target_failures == 0
        and
        len(
            missing_original_negatives
        )
        ==
        0
    )


    if not cycle_pass:

        overall_pass = False


    audit_rows.append(
        {
            "cycle":
                cycle,

            "rows":
                actual_rows,

            "positive":
                actual_positive,

            "negative":
                actual_negative,

            "positive_rate_pct":
                (
                    actual_positive
                    /
                    actual_rows
                    *
                    100
                ),

            "feature_count":
                feature_count,

            "schema_identical_to_v1":
                schema_identical_to_v1,

            "dtype_identical_to_v1":
                dtype_identical_to_v1,

            "duplicate_keys":
                duplicate_keys,

            "invalid_targets":
                invalid_targets,

            "invalid_label_status":
                invalid_label_status,

            "target_status_mismatch":
                target_status_mismatch,

            "missing_entity_groups":
                missing_groups,

            "blank_entity_groups":
                blank_groups,

            "nonfinite_feature_rows":
                nonfinite_rows,

            "invalid_trec":
                invalid_trec,

            "off_120min_grid_rows":
                off_grid_rows,

            "invalid_observation_utc":
                invalid_utc,

            "outside_domain_start":
                outside_start,

            "outside_domain_end":
                outside_end,

            "invalid_24h_horizon":
                invalid_horizon,

            "wrong_cycle_tag":
                wrong_cycle_tag,

            "expected_rows":
                expected_rows,

            "expected_positive":
                expected_positive,

            "expected_negative":
                expected_negative,

            "candidate_summary_reconciles":
                summary_reconciles,

            "new_keys":
                len(
                    new_keys
                ),

            "removed_keys":
                len(
                    removed_keys
                ),

            "expected_censored_keys":
                len(
                    censored_keys
                ),

            "removed_keys_match_censor":
                removed_keys_match_censor,

            "relabeled_rows":
                len(
                    relabeled_keys
                ),

            "relabel_target_failures":
                relabel_target_failures,

            "missing_original_negative_rows":
                len(
                    missing_original_negatives
                ),

            "cycle_audit_pass":
                cycle_pass,
        }
    )


    print(
        "Rows:",
        actual_rows
    )


    print(
        "Positive:",
        actual_positive
    )


    print(
        "Negative:",
        actual_negative
    )


    print(
        "Schema identical to v1:",
        schema_identical_to_v1
    )


    print(
        "Feature count:",
        feature_count
    )


    print(
        "Duplicate keys:",
        duplicate_keys
    )


    print(
        "Non-finite feature rows:",
        nonfinite_rows
    )


    print(
        "Off-grid rows:",
        off_grid_rows
    )


    print(
        "Invalid 24h horizons:",
        invalid_horizon
    )


    print(
        "New keys:",
        len(
            new_keys
        )
    )


    print(
        "Removed keys:",
        len(
            removed_keys
        )
    )


    print(
        "Expected censored keys:",
        len(
            censored_keys
        )
    )


    print(
        "Removed keys match censor:",
        removed_keys_match_censor
    )


    print(
        "Relabel failures:",
        relabel_target_failures
    )


    print(
        "Missing original negatives:",
        len(
            missing_original_negatives
        )
    )


    print(
        "Cycle audit:",
        (
            "PASS"
            if cycle_pass
            else
            "FAIL"
        )
    )


# ============================================================
# CROSS-CYCLE OVERLAP
# ============================================================

c24 = loaded_v2[
    "cycle24"
]

c25 = loaded_v2[
    "cycle25"
]


entity_overlap = sorted(
    set(
        c24[
            "ENTITY_GROUP_ID"
        ].astype(str)
    )
    &
    set(
        c25[
            "ENTITY_GROUP_ID"
        ].astype(str)
    )
)


c24_harps = set(
    pd.to_numeric(
        c24[
            "HARPNUM"
        ],
        errors="coerce",
    )
    .dropna()
    .astype(int)
)


c25_harps = set(
    pd.to_numeric(
        c25[
            "HARPNUM"
        ],
        errors="coerce",
    )
    .dropna()
    .astype(int)
)


harp_overlap = sorted(
    c24_harps
    &
    c25_harps
)


cross_cycle_pass = (
    len(
        entity_overlap
    )
    ==
    0
    and
    len(
        harp_overlap
    )
    ==
    0
)


if not cross_cycle_pass:

    overall_pass = False


overlap_rows = []


for value in entity_overlap:

    overlap_rows.append(
        {
            "overlap_type":
                "ENTITY_GROUP_ID",

            "value":
                value,
        }
    )


for value in harp_overlap:

    overlap_rows.append(
        {
            "overlap_type":
                "HARPNUM",

            "value":
                str(
                    value
                ),
        }
    )


overlap_df = pd.DataFrame(
    overlap_rows,
    columns=[
        "overlap_type",
        "value",
    ],
)


# ============================================================
# SAVE
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


year_df = pd.DataFrame(
    year_rows
)


audit_df.to_csv(
    AUDIT_OUTPUT,
    index=False,
)


year_df.to_csv(
    YEAR_OUTPUT,
    index=False,
)


overlap_df.to_csv(
    OVERLAP_OUTPUT,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "CROSS-CYCLE AUDIT"
)

print(
    "===================================="
)


print(
    "ENTITY_GROUP_ID overlap:",
    len(
        entity_overlap
    )
)


print(
    "HARPNUM overlap:",
    len(
        harp_overlap
    )
)


print(
    "Cross-cycle audit:",
    (
        "PASS"
        if cross_cycle_pass
        else
        "FAIL"
    )
)


print(
    "\n===================================="
)

print(
    "DATASET v2 FULL AUDIT SUMMARY"
)

print(
    "===================================="
)


print(
    audit_df.to_string(
        index=False
    )
)


print(
    "\n===================================="
)

print(
    "YEAR-WISE CLASS BALANCE"
)

print(
    "===================================="
)


print(
    year_df.to_string(
        index=False
    )
)


print(
    "\nOverall Dataset v2 candidate audit:",
    (
        "PASS"
        if overall_pass
        else
        "FAIL"
    )
)


print(
    "\nSaved:"
)

print(
    AUDIT_OUTPUT
)

print(
    YEAR_OUTPUT
)

print(
    OVERLAP_OUTPUT
)


if not overall_pass:

    raise RuntimeError(
        "Dataset v2 candidate audit failed."
    )


print(
    "\nFIX J COMPLETE — DATASET v2 CANDIDATE VERIFIED."
)