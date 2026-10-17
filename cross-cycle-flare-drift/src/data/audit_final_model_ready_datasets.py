from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path(
    "data/processed"
)

MANIFEST_DIR = Path(
    "data_manifests"
)

PHASE10_SUMMARY_FILE = (
    MANIFEST_DIR
    / "phase10_final_dataset_summary.csv"
)

TEMPORAL_CONTRACT_FILE = (
    MANIFEST_DIR
    / "temporal_data_contract.json"
)

AUDIT_OUTPUT_FILE = (
    MANIFEST_DIR
    / "phase11_final_dataset_audit.csv"
)

YEAR_OUTPUT_FILE = (
    MANIFEST_DIR
    / "phase11_year_summary.csv"
)

OVERLAP_OUTPUT_FILE = (
    MANIFEST_DIR
    / "phase11_cross_cycle_overlap.csv"
)


# ============================================================
# FEATURE CONTRACT
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
EXPECTED_STRIDE_MINUTES = 120
FORECAST_HOURS = 24


# ============================================================
# HELPERS
# ============================================================

def parse_trec_tai(value):

    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()

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
        errors="coerce"
    )


def check_tai_grid(series):

    parsed = series.apply(
        parse_trec_tai
    )

    invalid = int(
        parsed.isna().sum()
    )

    valid = parsed.dropna()

    off_grid = int(
        (
            (
                valid.dt.minute != 0
            )
            |
            (
                valid.dt.second != 0
            )
            |
            (
                valid.dt.hour % 2 != 0
            )
        )
        .sum()
    )

    return (
        parsed,
        invalid,
        off_grid,
    )


def safe_bool(value):

    return bool(
        value
    )


# ============================================================
# LOAD CONTRACTS
# ============================================================

if not PHASE10_SUMMARY_FILE.exists():

    raise FileNotFoundError(
        f"Missing: {PHASE10_SUMMARY_FILE}"
    )


if not TEMPORAL_CONTRACT_FILE.exists():

    raise FileNotFoundError(
        f"Missing: {TEMPORAL_CONTRACT_FILE}"
    )


phase10_summary = pd.read_csv(
    PHASE10_SUMMARY_FILE
)


with TEMPORAL_CONTRACT_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    temporal_contract = json.load(
        file
    )


if temporal_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Temporal contract is not frozen."
    )


contract_stride = int(
    temporal_contract[
        "sampling"
    ][
        "primary_stride_minutes"
    ]
)


if contract_stride != EXPECTED_STRIDE_MINUTES:

    raise RuntimeError(
        "Frozen temporal contract does not "
        "use 120-minute stride."
    )


# ============================================================
# CONFIG
# ============================================================

cycle_configs = {

    "cycle24": {
        "file":
            PROCESSED_DIR
            /
            "cycle24_model_ready.parquet",

        "cycle_label":
            "Cycle24",

        "start_utc":
            pd.Timestamp(
                temporal_contract[
                    "cycle24"
                ][
                    "domain_start_utc"
                ]
            ),

        "end_exclusive_utc":
            pd.Timestamp(
                temporal_contract[
                    "cycle24"
                ][
                    "domain_end_exclusive_utc"
                ]
            ),

        "first_tai":
            parse_trec_tai(
                temporal_contract[
                    "cycle24"
                ][
                    "first_valid_sample_tai"
                ]
            ),

        "last_tai":
            parse_trec_tai(
                temporal_contract[
                    "cycle24"
                ][
                    "last_valid_sample_tai"
                ]
            ),
    },

    "cycle25": {
        "file":
            PROCESSED_DIR
            /
            "cycle25_model_ready.parquet",

        "cycle_label":
            "Cycle25",

        "start_utc":
            pd.Timestamp(
                temporal_contract[
                    "cycle25"
                ][
                    "domain_start_utc"
                ]
            ),

        "end_exclusive_utc":
            pd.Timestamp(
                temporal_contract[
                    "cycle25"
                ][
                    "domain_end_exclusive_utc"
                ]
            ),

        "first_tai":
            parse_trec_tai(
                temporal_contract[
                    "cycle25"
                ][
                    "first_valid_sample_tai"
                ]
            ),

        "last_tai":
            parse_trec_tai(
                temporal_contract[
                    "cycle25"
                ][
                    "last_valid_sample_tai"
                ]
            ),
    },
}


# ============================================================
# LOAD DATASETS
# ============================================================

datasets = {}


for cycle_key, config in cycle_configs.items():

    file = config[
        "file"
    ]


    if not file.exists():

        raise FileNotFoundError(
            f"Missing model-ready dataset: "
            f"{file}"
        )


    datasets[
        cycle_key
    ] = pd.read_parquet(
        file
    )


# ============================================================
# SHARED SCHEMA CHECK
# ============================================================

c24_columns = list(
    datasets[
        "cycle24"
    ].columns
)

c25_columns = list(
    datasets[
        "cycle25"
    ].columns
)


schema_equal = (
    c24_columns
    ==
    c25_columns
)


print(
    "===================================="
)

print(
    "PHASE 11 — FINAL DATASET AUDIT"
)

print(
    "===================================="
)


print(
    "Cycle24 columns:",
    len(
        c24_columns
    )
)


print(
    "Cycle25 columns:",
    len(
        c25_columns
    )
)


print(
    "Cross-cycle schema identical:",
    schema_equal
)


# ============================================================
# PER-CYCLE AUDIT
# ============================================================

audit_rows = []
year_rows = []


overall_pass = True


for cycle_key, config in cycle_configs.items():

    print(
        "\n===================================="
    )

    print(
        cycle_key.upper()
    )

    print(
        "===================================="
    )


    df = datasets[
        cycle_key
    ].copy()


    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = {

        "T_REC",
        "observation_time_utc",
        "HARPNUM",
        "ENTITY_GROUP_ID",
        "entity_group_source",
        "cycle",
        "target",
        "label_status",
    }.union(
        ALL_FEATURES
    )


    missing_columns = sorted(
        required_columns
        -
        set(
            df.columns
        )
    )


    required_columns_ok = (
        len(
            missing_columns
        )
        ==
        0
    )


    # ========================================================
    # FEATURE CONTRACT
    # ========================================================

    feature_columns_present = [
        feature

        for feature in ALL_FEATURES

        if feature
        in
        df.columns
    ]


    feature_contract_ok = (
        len(
            feature_columns_present
        )
        ==
        EXPECTED_FEATURE_COUNT
    )


    # ========================================================
    # DUPLICATES
    # ========================================================

    duplicate_keys = int(
        df[
            [
                "HARPNUM",
                "T_REC",
            ]
        ]
        .duplicated()
        .sum()
    )


    # ========================================================
    # TARGET
    # ========================================================

    invalid_targets = int(
        (
            ~df[
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


    # ========================================================
    # LABEL STATUS
    # ========================================================

    invalid_label_status = int(
        (
            ~df[
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


    # ========================================================
    # ENTITY GROUP
    # ========================================================

    missing_entity_groups = int(
        df[
            "ENTITY_GROUP_ID"
        ]
        .isna()
        .sum()
    )


    blank_entity_groups = int(
        (
            df[
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


    unique_entity_groups = int(
        df[
            "ENTITY_GROUP_ID"
        ]
        .nunique()
    )


    # ========================================================
    # FEATURE FINITENESS
    # ========================================================

    feature_numeric = (
        df[
            ALL_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    finite_matrix = np.isfinite(
        feature_numeric
        .to_numpy(
            dtype=float
        )
    )


    nonfinite_feature_rows = int(
        (
            ~finite_matrix.all(
                axis=1
            )
        )
        .sum()
    )


    # ========================================================
    # TAI GRID
    # ========================================================

    (
        parsed_tai,
        invalid_trec,
        off_grid_rows,
    ) = check_tai_grid(
        df[
            "T_REC"
        ]
    )


    # ========================================================
    # TEMPORAL CONTRACT — TAI
    # ========================================================

    before_tai_start = int(
        (
            parsed_tai
            <
            config[
                "first_tai"
            ]
        )
        .sum()
    )


    after_tai_end = int(
        (
            parsed_tai
            >
            config[
                "last_tai"
            ]
        )
        .sum()
    )


    # ========================================================
    # TEMPORAL CONTRACT — UTC
    # ========================================================

    observation_utc = pd.to_datetime(
        df[
            "observation_time_utc"
        ],
        utc=True,
        errors="coerce"
    )


    invalid_observation_utc = int(
        observation_utc
        .isna()
        .sum()
    )


    before_utc_start = int(
        (
            observation_utc
            <
            config[
                "start_utc"
            ]
        )
        .sum()
    )


    after_utc_end = int(
        (
            observation_utc
            >=
            config[
                "end_exclusive_utc"
            ]
        )
        .sum()
    )


    # ========================================================
    # 24-HOUR LABEL HORIZON BOUNDARY
    # ========================================================

    horizon_end = (
        observation_utc
        +
        pd.Timedelta(
            hours=FORECAST_HOURS
        )
    )


    invalid_label_horizon = int(
        (
            horizon_end
            >=
            config[
                "end_exclusive_utc"
            ]
        )
        .sum()
    )


    # ========================================================
    # CYCLE TAG
    # ========================================================

    wrong_cycle_tag = int(
        (
            df[
                "cycle"
            ]
            !=
            config[
                "cycle_label"
            ]
        )
        .sum()
    )


    # ========================================================
    # PHASE 10 ROW RECONCILIATION
    # ========================================================

    phase10_row = phase10_summary[
        phase10_summary[
            "cycle"
        ]
        ==
        cycle_key
    ]


    if len(
        phase10_row
    ) != 1:

        raise RuntimeError(
            f"Could not find unique "
            f"Phase10 summary for {cycle_key}"
        )


    expected_rows = int(
        phase10_row.iloc[
            0
        ][
            "final_model_rows"
        ]
    )


    expected_positive = int(
        phase10_row.iloc[
            0
        ][
            "positive"
        ]
    )


    expected_negative = int(
        phase10_row.iloc[
            0
        ][
            "negative"
        ]
    )


    actual_rows = len(
        df
    )


    actual_positive = int(
        (
            df[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    actual_negative = int(
        (
            df[
                "target"
            ]
            ==
            0
        )
        .sum()
    )


    row_count_matches_phase10 = (
        actual_rows
        ==
        expected_rows
    )


    positive_matches_phase10 = (
        actual_positive
        ==
        expected_positive
    )


    negative_matches_phase10 = (
        actual_negative
        ==
        expected_negative
    )


    # ========================================================
    # CLASS BALANCE
    # ========================================================

    positive_rate_pct = (
        actual_positive
        /
        actual_rows
        *
        100
        if actual_rows > 0
        else np.nan
    )


    # ========================================================
    # YEAR SUMMARY
    # ========================================================

    temp = df.copy()


    temp[
        "_year"
    ] = (
        observation_utc
        .dt
        .year
    )


    for year, group in temp.groupby(
        "_year"
    ):

        year_total = len(
            group
        )


        year_positive = int(
            (
                group[
                    "target"
                ]
                ==
                1
            )
            .sum()
        )


        year_negative = int(
            (
                group[
                    "target"
                ]
                ==
                0
            )
            .sum()
        )


        year_positive_rate = (
            year_positive
            /
            year_total
            *
            100
            if year_total
            else np.nan
        )


        year_rows.append(
            {
                "cycle":
                    cycle_key,

                "year":
                    int(
                        year
                    ),

                "rows":
                    year_total,

                "positive":
                    year_positive,

                "negative":
                    year_negative,

                "positive_rate_pct":
                    year_positive_rate,

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

        schema_equal

        and

        required_columns_ok

        and

        feature_contract_ok

        and

        duplicate_keys == 0

        and

        invalid_targets == 0

        and

        invalid_label_status == 0

        and

        missing_entity_groups == 0

        and

        blank_entity_groups == 0

        and

        nonfinite_feature_rows == 0

        and

        invalid_trec == 0

        and

        off_grid_rows == 0

        and

        before_tai_start == 0

        and

        after_tai_end == 0

        and

        invalid_observation_utc == 0

        and

        before_utc_start == 0

        and

        after_utc_end == 0

        and

        invalid_label_horizon == 0

        and

        wrong_cycle_tag == 0

        and

        row_count_matches_phase10

        and

        positive_matches_phase10

        and

        negative_matches_phase10
    )


    if not cycle_pass:

        overall_pass = False


    summary = {

        "cycle":
            cycle_key,

        "rows":
            actual_rows,

        "positive":
            actual_positive,

        "negative":
            actual_negative,

        "positive_rate_pct":
            positive_rate_pct,

        "unique_entity_groups":
            unique_entity_groups,

        "required_columns_ok":
            required_columns_ok,

        "feature_count":
            len(
                feature_columns_present
            ),

        "feature_contract_ok":
            feature_contract_ok,

        "duplicate_keys":
            duplicate_keys,

        "invalid_targets":
            invalid_targets,

        "invalid_label_status":
            invalid_label_status,

        "missing_entity_groups":
            missing_entity_groups,

        "blank_entity_groups":
            blank_entity_groups,

        "nonfinite_feature_rows":
            nonfinite_feature_rows,

        "invalid_trec":
            invalid_trec,

        "off_120min_grid_rows":
            off_grid_rows,

        "before_tai_start":
            before_tai_start,

        "after_tai_end":
            after_tai_end,

        "invalid_observation_utc":
            invalid_observation_utc,

        "before_utc_start":
            before_utc_start,

        "after_utc_end":
            after_utc_end,

        "invalid_24h_horizon":
            invalid_label_horizon,

        "wrong_cycle_tag":
            wrong_cycle_tag,

        "phase10_expected_rows":
            expected_rows,

        "row_count_matches_phase10":
            row_count_matches_phase10,

        "positive_matches_phase10":
            positive_matches_phase10,

        "negative_matches_phase10":
            negative_matches_phase10,

        "cycle_audit_pass":
            cycle_pass,
    }


    audit_rows.append(
        summary
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
        "Positive rate:",
        f"{positive_rate_pct:.4f}%"
    )


    print(
        "Unique entity groups:",
        unique_entity_groups
    )


    print(
        "Feature count:",
        len(
            feature_columns_present
        )
    )


    print(
        "Duplicate keys:",
        duplicate_keys
    )


    print(
        "Non-finite feature rows:",
        nonfinite_feature_rows
    )


    print(
        "Invalid targets:",
        invalid_targets
    )


    print(
        "Missing entity groups:",
        missing_entity_groups
    )


    print(
        "Off 120-min grid rows:",
        off_grid_rows
    )


    print(
        "Rows outside TAI bounds:",
        before_tai_start
        +
        after_tai_end
    )


    print(
        "Rows outside UTC bounds:",
        before_utc_start
        +
        after_utc_end
    )


    print(
        "Invalid 24h horizons:",
        invalid_label_horizon
    )


    print(
        "Phase10 row reconciliation:",
        row_count_matches_phase10
    )


    print(
        "Cycle audit:",
        (
            "PASS"
            if cycle_pass
            else "FAIL"
        )
    )


# ============================================================
# CROSS-CYCLE LEAKAGE AUDIT
# ============================================================

c24 = datasets[
    "cycle24"
]

c25 = datasets[
    "cycle25"
]


c24_groups = set(
    c24[
        "ENTITY_GROUP_ID"
    ]
    .astype(str)
)


c25_groups = set(
    c25[
        "ENTITY_GROUP_ID"
    ]
    .astype(str)
)


overlap_groups = sorted(
    c24_groups
    &
    c25_groups
)


c24_harps = set(
    pd.to_numeric(
        c24[
            "HARPNUM"
        ],
        errors="coerce"
    )
    .dropna()
    .astype(int)
)


c25_harps = set(
    pd.to_numeric(
        c25[
            "HARPNUM"
        ],
        errors="coerce"
    )
    .dropna()
    .astype(int)
)


overlap_harps = sorted(
    c24_harps
    &
    c25_harps
)


overlap_rows = []


for group in overlap_groups:

    overlap_rows.append(
        {
            "overlap_type":
                "ENTITY_GROUP_ID",

            "value":
                group,
        }
    )


for harpnum in overlap_harps:

    overlap_rows.append(
        {
            "overlap_type":
                "HARPNUM",

            "value":
                str(
                    harpnum
                ),
        }
    )


overlap_df = pd.DataFrame(
    overlap_rows,
    columns=[
        "overlap_type",
        "value",
    ]
)


overlap_df.to_csv(
    OVERLAP_OUTPUT_FILE,
    index=False
)


entity_group_overlap_count = len(
    overlap_groups
)


harp_overlap_count = len(
    overlap_harps
)


cross_cycle_leakage_pass = (
    entity_group_overlap_count
    ==
    0

    and

    harp_overlap_count
    ==
    0
)


if not cross_cycle_leakage_pass:

    overall_pass = False


# ============================================================
# SAVE OUTPUTS
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


year_df = pd.DataFrame(
    year_rows
)


audit_df.to_csv(
    AUDIT_OUTPUT_FILE,
    index=False
)


year_df.to_csv(
    YEAR_OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "CROSS-CYCLE LEAKAGE AUDIT"
)

print(
    "===================================="
)


print(
    "ENTITY_GROUP_ID overlap:",
    entity_group_overlap_count
)


print(
    "HARPNUM overlap:",
    harp_overlap_count
)


print(
    "Cross-cycle leakage audit:",
    (
        "PASS"
        if cross_cycle_leakage_pass
        else "FAIL"
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
    "\n===================================="
)

print(
    "PHASE 11 FINAL SUMMARY"
)

print(
    "===================================="
)


print(
    audit_df[
        [
            "cycle",
            "rows",
            "positive",
            "negative",
            "positive_rate_pct",
            "unique_entity_groups",
            "feature_count",
            "duplicate_keys",
            "nonfinite_feature_rows",
            "invalid_targets",
            "missing_entity_groups",
            "off_120min_grid_rows",
            "invalid_24h_horizon",
            "cycle_audit_pass",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nSchema identical:",
    schema_equal
)


print(
    "Cross-cycle entity overlap:",
    entity_group_overlap_count
)


print(
    "Cross-cycle HARP overlap:",
    harp_overlap_count
)


print(
    "Overall Phase 11 audit:",
    (
        "PASS"
        if overall_pass
        else "FAIL"
    )
)


print(
    "\nSaved:"
)


print(
    AUDIT_OUTPUT_FILE
)


print(
    YEAR_OUTPUT_FILE
)


print(
    OVERLAP_OUTPUT_FILE
)


if not overall_pass:

    raise RuntimeError(
        "Phase 11 final dataset audit failed."
    )


print(
    "\nPHASE 11 COMPLETE."
)