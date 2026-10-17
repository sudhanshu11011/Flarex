from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")
MANIFEST_DIR = Path("data_manifests")


C24_FILE = (
    PROCESSED_DIR
    /
    "cycle24_model_ready_leakage_safe.parquet"
)

C25_FILE = (
    PROCESSED_DIR
    /
    "cycle25_model_ready_leakage_safe.parquet"
)


PHASE10_SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "phase10_final_dataset_summary.csv"
)

QUARANTINE_SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "phase11_cross_cycle_quarantine_summary.csv"
)

TEMPORAL_CONTRACT_FILE = (
    MANIFEST_DIR
    /
    "temporal_data_contract.json"
)


AUDIT_OUTPUT_FILE = (
    MANIFEST_DIR
    /
    "phase11_leakage_safe_final_audit.csv"
)

YEAR_OUTPUT_FILE = (
    MANIFEST_DIR
    /
    "phase11_leakage_safe_year_summary.csv"
)

OVERLAP_OUTPUT_FILE = (
    MANIFEST_DIR
    /
    "phase11_leakage_safe_overlap.csv"
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
        .replace(".", "-", 2)
        .replace("_", "T", 1)
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


# ============================================================
# LOAD CONTRACTS
# ============================================================

for required_file in [
    C24_FILE,
    C25_FILE,
    PHASE10_SUMMARY_FILE,
    QUARANTINE_SUMMARY_FILE,
    TEMPORAL_CONTRACT_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Missing: {required_file}"
        )


phase10 = pd.read_csv(
    PHASE10_SUMMARY_FILE
)


quarantine = pd.read_csv(
    QUARANTINE_SUMMARY_FILE
)


with TEMPORAL_CONTRACT_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    contract = json.load(
        file
    )


if contract.get("status") != "FROZEN":

    raise RuntimeError(
        "Temporal contract is not frozen."
    )


contract_stride = int(
    contract[
        "sampling"
    ][
        "primary_stride_minutes"
    ]
)


if contract_stride != EXPECTED_STRIDE_MINUTES:

    raise RuntimeError(
        "Frozen stride is not 120 minutes."
    )


# ============================================================
# LOAD DATASETS
# ============================================================

datasets = {
    "cycle24":
        pd.read_parquet(
            C24_FILE
        ),

    "cycle25":
        pd.read_parquet(
            C25_FILE
        ),
}


configs = {

    "cycle24": {

        "cycle_label":
            "Cycle24",

        "start_utc":
            pd.Timestamp(
                contract[
                    "cycle24"
                ][
                    "domain_start_utc"
                ]
            ),

        "end_exclusive_utc":
            pd.Timestamp(
                contract[
                    "cycle24"
                ][
                    "domain_end_exclusive_utc"
                ]
            ),

        "first_tai":
            parse_trec_tai(
                contract[
                    "cycle24"
                ][
                    "first_valid_sample_tai"
                ]
            ),

        "last_tai":
            parse_trec_tai(
                contract[
                    "cycle24"
                ][
                    "last_valid_sample_tai"
                ]
            ),
    },

    "cycle25": {

        "cycle_label":
            "Cycle25",

        "start_utc":
            pd.Timestamp(
                contract[
                    "cycle25"
                ][
                    "domain_start_utc"
                ]
            ),

        "end_exclusive_utc":
            pd.Timestamp(
                contract[
                    "cycle25"
                ][
                    "domain_end_exclusive_utc"
                ]
            ),

        "first_tai":
            parse_trec_tai(
                contract[
                    "cycle25"
                ][
                    "first_valid_sample_tai"
                ]
            ),

        "last_tai":
            parse_trec_tai(
                contract[
                    "cycle25"
                ][
                    "last_valid_sample_tai"
                ]
            ),
    },
}


# ============================================================
# EXPECTED COUNTS AFTER QUARANTINE
# ============================================================

expected = {}


for cycle in [
    "cycle24",
    "cycle25",
]:

    phase10_row = phase10[
        phase10[
            "cycle"
        ]
        ==
        cycle
    ]


    if len(
        phase10_row
    ) != 1:

        raise RuntimeError(
            f"Missing Phase10 summary for {cycle}"
        )


    phase10_row = phase10_row.iloc[
        0
    ]


    if cycle == "cycle24":

        removed_rows = int(
            quarantine[
                "cycle24_rows_removed"
            ]
            .sum()
        )

        removed_positive = int(
            quarantine[
                "cycle24_positive_removed"
            ]
            .sum()
        )

        removed_negative = int(
            quarantine[
                "cycle24_negative_removed"
            ]
            .sum()
        )


    else:

        removed_rows = int(
            quarantine[
                "cycle25_rows_removed"
            ]
            .sum()
        )

        removed_positive = int(
            quarantine[
                "cycle25_positive_removed"
            ]
            .sum()
        )

        removed_negative = int(
            quarantine[
                "cycle25_negative_removed"
            ]
            .sum()
        )


    expected[
        cycle
    ] = {

        "rows":
            int(
                phase10_row[
                    "final_model_rows"
                ]
            )
            -
            removed_rows,

        "positive":
            int(
                phase10_row[
                    "positive"
                ]
            )
            -
            removed_positive,

        "negative":
            int(
                phase10_row[
                    "negative"
                ]
            )
            -
            removed_negative,
    }


# ============================================================
# SHARED SCHEMA
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


schema_identical = (
    c24_columns
    ==
    c25_columns
)


print(
    "===================================="
)

print(
    "PHASE 11G — LEAKAGE-SAFE FINAL AUDIT"
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
    "Schema identical:",
    schema_identical
)


# ============================================================
# PER-CYCLE AUDIT
# ============================================================

audit_rows = []
year_rows = []

overall_pass = True


for cycle, df in datasets.items():

    config = configs[
        cycle
    ]


    print(
        "\n===================================="
    )

    print(
        cycle.upper()
    )

    print(
        "===================================="
    )


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


    feature_count = sum(
        feature
        in
        df.columns

        for feature
        in
        ALL_FEATURES
    )


    feature_contract_ok = (
        feature_count
        ==
        EXPECTED_FEATURE_COUNT
    )


    # ========================================================
    # DUPLICATE KEYS
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
    # TARGET INTEGRITY
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
    # ENTITY GROUP INTEGRITY
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

    numeric_features = (
        df[
            ALL_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    finite_matrix = np.isfinite(
        numeric_features.to_numpy(
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
    # FROZEN TAI GRID
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
    # UTC TEMPORAL CONTRACT
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
    # 24-HOUR LABEL HORIZON
    # ========================================================

    horizon_end = (
        observation_utc
        +
        pd.Timedelta(
            hours=FORECAST_HOURS
        )
    )


    invalid_horizon = int(
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
    # EXPECTED COUNT RECONCILIATION
    # ========================================================

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


    expected_rows = expected[
        cycle
    ][
        "rows"
    ]


    expected_positive = expected[
        cycle
    ][
        "positive"
    ]


    expected_negative = expected[
        cycle
    ][
        "negative"
    ]


    rows_match = (
        actual_rows
        ==
        expected_rows
    )


    positives_match = (
        actual_positive
        ==
        expected_positive
    )


    negatives_match = (
        actual_negative
        ==
        expected_negative
    )


    positive_rate = (
        actual_positive
        /
        actual_rows
        *
        100
        if actual_rows
        else np.nan
    )


    # ========================================================
    # YEAR-WISE SUMMARY
    # ========================================================

    temp = df.copy()


    temp[
        "_year"
    ] = observation_utc.dt.year


    for year, group in temp.groupby(
        "_year"
    ):

        rows = len(
            group
        )


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


        positive_rate_year = (
            positive
            /
            rows
            *
            100
            if rows
            else np.nan
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
                    rows,

                "positive":
                    positive,

                "negative":
                    negative,

                "positive_rate_pct":
                    positive_rate_year,

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
    # FINAL CYCLE DECISION
    # ========================================================

    cycle_pass = (

        schema_identical

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
        invalid_horizon == 0

        and
        wrong_cycle_tag == 0

        and
        rows_match

        and
        positives_match

        and
        negatives_match
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
                positive_rate,

            "unique_entity_groups":
                unique_entity_groups,

            "feature_count":
                feature_count,

            "duplicate_keys":
                duplicate_keys,

            "nonfinite_feature_rows":
                nonfinite_feature_rows,

            "invalid_targets":
                invalid_targets,

            "invalid_label_status":
                invalid_label_status,

            "missing_entity_groups":
                missing_entity_groups,

            "blank_entity_groups":
                blank_entity_groups,

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
                invalid_horizon,

            "wrong_cycle_tag":
                wrong_cycle_tag,

            "expected_rows":
                expected_rows,

            "expected_positive":
                expected_positive,

            "expected_negative":
                expected_negative,

            "rows_match_expected":
                rows_match,

            "positives_match_expected":
                positives_match,

            "negatives_match_expected":
                negatives_match,

            "cycle_audit_pass":
                cycle_pass,
        }
    )


    print(
        "Rows:",
        actual_rows
    )


    print(
        "Expected rows:",
        expected_rows
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
        f"{positive_rate:.4f}%"
    )


    print(
        "Unique entity groups:",
        unique_entity_groups
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
        "Off-grid rows:",
        off_grid_rows
    )


    print(
        "Invalid 24h horizons:",
        invalid_horizon
    )


    print(
        "Count reconciliation:",
        (
            rows_match
            and
            positives_match
            and
            negatives_match
        )
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
# CROSS-CYCLE LEAKAGE CHECK
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
    ].astype(str)
)


c25_groups = set(
    c25[
        "ENTITY_GROUP_ID"
    ].astype(str)
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


cross_cycle_pass = (
    len(
        overlap_groups
    )
    ==
    0

    and

    len(
        overlap_harps
    )
    ==
    0
)


if not cross_cycle_pass:

    overall_pass = False


# ============================================================
# SAVE
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


year_df = pd.DataFrame(
    year_rows
)


AUDIT_OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


audit_df.to_csv(
    AUDIT_OUTPUT_FILE,
    index=False
)


year_df.to_csv(
    YEAR_OUTPUT_FILE,
    index=False
)


overlap_df.to_csv(
    OVERLAP_OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
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
    len(
        overlap_groups
    )
)


print(
    "HARPNUM overlap:",
    len(
        overlap_harps
    )
)


print(
    "Cross-cycle leakage audit:",
    (
        "PASS"
        if cross_cycle_pass
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
    schema_identical
)


print(
    "Cross-cycle entity overlap:",
    len(
        overlap_groups
    )
)


print(
    "Cross-cycle HARP overlap:",
    len(
        overlap_harps
    )
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
        "Leakage-safe Phase 11 audit failed."
    )


print(
    "\nPHASE 11 COMPLETE."
)