from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from scipy.stats import (
    ks_2samp,
    wasserstein_distance,
)


# ============================================================
# PATHS
# ============================================================

PROTOCOL_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "shift_null_control_protocol.json"
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


OUTPUT_POINT_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "marginal_feature_shift_point.csv"
)

OUTPUT_MISSINGNESS_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "missingness_audit.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "marginal_feature_shift_point_contract.json"
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

EXPECTED_C24_ROWS = 13072
EXPECTED_C24_POSITIVES = 527
EXPECTED_C24_NEGATIVES = 12545
EXPECTED_C24_GROUPS = 176

EXPECTED_C25_ROWS = 79361
EXPECTED_C25_POSITIVES = 4836
EXPECTED_C25_NEGATIVES = 74525
EXPECTED_C25_GROUPS = 1011

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TARGET_COLUMN = "target"
TIME_COLUMN = "observation_time_utc"


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


# ============================================================
# FEATURES
# ============================================================

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


def percentile(
    values,
    q,
):

    return float(
        np.quantile(
            values,
            q,
            method="linear",
        )
    )


def get_stratum_mask(
    dataframe,
    stratum,
):

    if stratum == "OVERALL":

        return np.ones(
            len(dataframe),
            dtype=bool,
        )


    if stratum == "TARGET_0":

        return (
            dataframe[
                TARGET_COLUMN
            ].to_numpy(
                dtype=np.int64
            )
            ==
            0
        )


    if stratum == "TARGET_1":

        return (
            dataframe[
                TARGET_COLUMN
            ].to_numpy(
                dtype=np.int64
            )
            ==
            1
        )


    raise RuntimeError(
        f"Unknown stratum: {stratum}"
    )


def build_feature_summary(
    reference_values,
    transfer_values,
):

    reference_values = np.asarray(
        reference_values,
        dtype=np.float64,
    )

    transfer_values = np.asarray(
        transfer_values,
        dtype=np.float64,
    )


    if len(reference_values) == 0:

        raise RuntimeError(
            "Reference feature population is empty."
        )


    if len(transfer_values) == 0:

        raise RuntimeError(
            "Transfer feature population is empty."
        )


    if not np.isfinite(
        reference_values
    ).all():

        raise RuntimeError(
            "Reference feature contains "
            "non-finite values."
        )


    if not np.isfinite(
        transfer_values
    ).all():

        raise RuntimeError(
            "Transfer feature contains "
            "non-finite values."
        )


    ref_q25 = percentile(
        reference_values,
        0.25,
    )

    ref_median = percentile(
        reference_values,
        0.50,
    )

    ref_q75 = percentile(
        reference_values,
        0.75,
    )

    ref_iqr = float(
        ref_q75
        -
        ref_q25
    )


    transfer_q25 = percentile(
        transfer_values,
        0.25,
    )

    transfer_median = percentile(
        transfer_values,
        0.50,
    )

    transfer_q75 = percentile(
        transfer_values,
        0.75,
    )

    transfer_iqr = float(
        transfer_q75
        -
        transfer_q25
    )


    ks_statistic = float(
        ks_2samp(
            reference_values,
            transfer_values,
            alternative="two-sided",
            method="asymp",
        ).statistic
    )


    wasserstein = float(
        wasserstein_distance(
            reference_values,
            transfer_values,
        )
    )


    if ref_iqr > 0.0:

        normalized_wasserstein = float(
            wasserstein
            /
            ref_iqr
        )

        median_shift_iqr_units = float(
            (
                transfer_median
                -
                ref_median
            )
            /
            ref_iqr
        )

        reference_iqr_zero = False

    else:

        normalized_wasserstein = np.nan
        median_shift_iqr_units = np.nan
        reference_iqr_zero = True


    return {
        "reference_q25":
            ref_q25,

        "reference_median":
            ref_median,

        "reference_q75":
            ref_q75,

        "reference_iqr":
            ref_iqr,

        "transfer_q25":
            transfer_q25,

        "transfer_median":
            transfer_median,

        "transfer_q75":
            transfer_q75,

        "transfer_iqr":
            transfer_iqr,

        "ks_statistic":
            ks_statistic,

        "wasserstein_distance":
            wasserstein,

        "normalized_wasserstein":
            normalized_wasserstein,

        "median_shift_iqr_units":
            median_shift_iqr_units,

        "reference_iqr_zero":
            reference_iqr_zero,
    }


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 9.2A — MARGINAL FEATURE-DISTRIBUTION POINT AUDIT"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    PROTOCOL_FILE,
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
    OUTPUT_POINT_FILE,
    OUTPUT_MISSINGNESS_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite Chapter "
            "9.2A frozen results."
        )


# ============================================================
# LOAD / VERIFY PROTOCOL
# ============================================================

protocol = load_json(
    PROTOCOL_FILE
)


if protocol.get(
    "status"
) != "FROZEN_BEFORE_SHIFT_AND_SHAP_ANALYSIS":

    raise RuntimeError(
        "Chapter 9.1 protocol "
        "is not frozen."
    )


if protocol[
    "feature_contract"
][
    "feature_order"
] != FEATURES:

    raise RuntimeError(
        "Frozen feature order mismatch."
    )


if int(
    protocol[
        "feature_contract"
    ][
        "feature_count"
    ]
) != 25:

    raise RuntimeError(
        "Frozen feature count mismatch."
    )


marginal_protocol = protocol[
    "marginal_distribution_shift"
]


if marginal_protocol[
    "strata"
] != STRATA:

    raise RuntimeError(
        "Frozen marginal-shift "
        "strata mismatch."
    )


print(
    "Chapter 9.1 protocol:",
    "FROZEN"
)

print(
    "Frozen feature order:",
    "PASS"
)

print(
    "Marginal shift strata:",
    "PASS"
)


# ============================================================
# VERIFY FEATURE REGISTRY HASH
# ============================================================

feature_registry_hash = sha256_file(
    FEATURE_REGISTRY_FILE
)


expected_feature_registry_hash = (
    protocol[
        "artifacts"
    ][
        "feature_registry"
    ][
        "sha256"
    ]
)


if (
    feature_registry_hash
    !=
    expected_feature_registry_hash
):

    raise RuntimeError(
        "Shift feature-registry "
        "hash mismatch."
    )


feature_registry = pd.read_csv(
    FEATURE_REGISTRY_FILE,
    keep_default_na=False,
)


if (
    feature_registry[
        "feature_name"
    ].astype(str).tolist()
    !=
    FEATURES
):

    raise RuntimeError(
        "Shift feature-registry "
        "order mismatch."
    )


print(
    "Shift feature registry:",
    "PASS"
)


# ============================================================
# VERIFY DATASET HASHES BEFORE CONTENT READ
# ============================================================

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
        "Cycle25 Dataset v2 hash mismatch."
    )


if (
    c24_hash
    !=
    protocol[
        "dataset_expectations"
    ][
        "cycle24_holdout"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle24 holdout hash differs "
        "from Chapter 9.1 protocol."
    )


if (
    c25_hash
    !=
    protocol[
        "dataset_expectations"
    ][
        "cycle25"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle25 hash differs from "
        "Chapter 9.1 protocol."
    )


print(
    "Cycle24 holdout hash:",
    "PASS"
)

print(
    "Cycle25 Dataset v2 hash:",
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


print(
    "Cycle24 holdout opened:",
    True
)

print(
    "Cycle25 opened:",
    True
)


# ============================================================
# SCHEMA
# ============================================================

required_columns = set(
    FEATURES
    +
    [
        ENTITY_GROUP_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN,
    ]
)


for name, dataframe in [
    ("CYCLE24_HOLDOUT", c24),
    ("CYCLE25", c25),
]:

    missing_columns = (
        required_columns
        -
        set(
            dataframe.columns
        )
    )


    if missing_columns:

        raise RuntimeError(
            f"{name}: missing columns: "
            f"{sorted(missing_columns)}"
        )


# ============================================================
# POPULATION INTEGRITY
# ============================================================

if len(c24) != EXPECTED_C24_ROWS:

    raise RuntimeError(
        "Cycle24 holdout row-count mismatch."
    )


if len(c25) != EXPECTED_C25_ROWS:

    raise RuntimeError(
        "Cycle25 row-count mismatch."
    )


c24_target = c24[
    TARGET_COLUMN
].to_numpy(
    dtype=np.int64
)

c25_target = c25[
    TARGET_COLUMN
].to_numpy(
    dtype=np.int64
)


if set(
    np.unique(
        c24_target
    ).tolist()
) != {0, 1}:

    raise RuntimeError(
        "Cycle24 target is not binary."
    )


if set(
    np.unique(
        c25_target
    ).tolist()
) != {0, 1}:

    raise RuntimeError(
        "Cycle25 target is not binary."
    )


if int(
    c24_target.sum()
) != EXPECTED_C24_POSITIVES:

    raise RuntimeError(
        "Cycle24 positive-count mismatch."
    )


if int(
    (c24_target == 0).sum()
) != EXPECTED_C24_NEGATIVES:

    raise RuntimeError(
        "Cycle24 negative-count mismatch."
    )


if int(
    c25_target.sum()
) != EXPECTED_C25_POSITIVES:

    raise RuntimeError(
        "Cycle25 positive-count mismatch."
    )


if int(
    (c25_target == 0).sum()
) != EXPECTED_C25_NEGATIVES:

    raise RuntimeError(
        "Cycle25 negative-count mismatch."
    )


c24_group_count = int(
    c24[
        ENTITY_GROUP_COLUMN
    ].nunique(
        dropna=False
    )
)

c25_group_count = int(
    c25[
        ENTITY_GROUP_COLUMN
    ].nunique(
        dropna=False
    )
)


if c24_group_count != EXPECTED_C24_GROUPS:

    raise RuntimeError(
        "Cycle24 ENTITY_GROUP count mismatch."
    )


if c25_group_count != EXPECTED_C25_GROUPS:

    raise RuntimeError(
        "Cycle25 ENTITY_GROUP count mismatch."
    )


print(
    "Cycle24 holdout population:",
    "PASS"
)

print(
    "Cycle25 population:",
    "PASS"
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


if len(
    primary_blocks
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen "
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
            "Frozen temporal-block "
            "overlap detected."
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
] = block_id.astype(str)


if set(
    c25[
        "_C25_BLOCK_ID"
    ].unique()
) != set(
    PRIMARY_BLOCKS
):

    raise RuntimeError(
        "Cycle25 assigned block-set mismatch."
    )


print(
    "Frozen Cycle25 block assignment:",
    "PASS"
)


# ============================================================
# BUILD SCOPE TABLES
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
# MISSINGNESS / NONFINITE AUDIT
# ============================================================

missingness_rows = []


for scope_id, dataframe in scope_frames.items():

    total_rows = len(
        dataframe
    )


    for feature in FEATURES:

        original = dataframe[
            feature
        ]


        missing_mask = (
            original.isna()
        )


        numeric = pd.to_numeric(
            original,
            errors="coerce",
        ).to_numpy(
            dtype=np.float64
        )


        original_missing = missing_mask.to_numpy()


        nonfinite_nonmissing = (
            (~np.isfinite(numeric))
            &
            (~original_missing)
        )


        missing_count = int(
            missing_mask.sum()
        )


        nonfinite_count = int(
            nonfinite_nonmissing.sum()
        )


        missingness_rows.append(
            {
                "scope_id":
                    scope_id,

                "feature":
                    feature,

                "rows":
                    total_rows,

                "missing_count":
                    missing_count,

                "missing_fraction":
                    float(
                        missing_count
                        /
                        total_rows
                    ),

                "nonfinite_count":
                    nonfinite_count,

                "nonfinite_fraction":
                    float(
                        nonfinite_count
                        /
                        total_rows
                    ),

                "clean":
                    bool(
                        missing_count == 0
                        and
                        nonfinite_count == 0
                    ),
            }
        )


missingness_df = pd.DataFrame(
    missingness_rows
)


expected_missingness_rows = (
    5
    *
    25
)


if len(
    missingness_df
) != expected_missingness_rows:

    raise RuntimeError(
        "Missingness-audit row-count mismatch."
    )


OUTPUT_MISSINGNESS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


missingness_df.to_csv(
    OUTPUT_MISSINGNESS_FILE,
    index=False,
)


if not missingness_df[
    "clean"
].all():

    raise RuntimeError(
        "Missing/non-finite frozen feature "
        "values detected. Missingness audit "
        "was saved; marginal-shift analysis "
        "has been stopped without imputation."
    )


print(
    "Missingness/nonfinite audit:",
    "PASS"
)

print(
    "Missingness audit rows:",
    len(
        missingness_df
    )
)


# ============================================================
# POINT MARGINAL-SHIFT ANALYSIS
# ============================================================

result_rows = []


print(
    "\n===================================="
)

print(
    "COMPUTING MARGINAL DISTRIBUTION SHIFT"
)

print(
    "===================================="
)


reference_frame = scope_frames[
    "C24_HOLDOUT"
]


for transfer_scope in TRANSFER_SCOPES:

    transfer_frame = scope_frames[
        transfer_scope
    ]


    for stratum in STRATA:

        ref_mask = get_stratum_mask(
            reference_frame,
            stratum,
        )


        transfer_mask = get_stratum_mask(
            transfer_frame,
            stratum,
        )


        ref_subset = reference_frame.loc[
            ref_mask
        ]


        transfer_subset = transfer_frame.loc[
            transfer_mask
        ]


        ref_rows = int(
            len(
                ref_subset
            )
        )


        transfer_rows = int(
            len(
                transfer_subset
            )
        )


        ref_groups = int(
            ref_subset[
                ENTITY_GROUP_COLUMN
            ].nunique(
                dropna=False
            )
        )


        transfer_groups = int(
            transfer_subset[
                ENTITY_GROUP_COLUMN
            ].nunique(
                dropna=False
            )
        )


        if ref_rows == 0:

            raise RuntimeError(
                f"{stratum}: empty "
                "Cycle24 reference."
            )


        if transfer_rows == 0:

            raise RuntimeError(
                f"{transfer_scope}/{stratum}: "
                "empty transfer population."
            )


        for feature in FEATURES:

            reference_values = (
                ref_subset[
                    feature
                ].to_numpy(
                    dtype=np.float64
                )
            )


            transfer_values = (
                transfer_subset[
                    feature
                ].to_numpy(
                    dtype=np.float64
                )
            )


            summary = build_feature_summary(
                reference_values,
                transfer_values,
            )


            result_rows.append(
                {
                    "reference_scope":
                        "C24_HOLDOUT",

                    "transfer_scope":
                        transfer_scope,

                    "stratum":
                        stratum,

                    "feature":
                        feature,

                    "reference_rows":
                        ref_rows,

                    "transfer_rows":
                        transfer_rows,

                    "reference_entity_groups":
                        ref_groups,

                    "transfer_entity_groups":
                        transfer_groups,

                    **summary,

                    "ks_pvalue_used":
                        False,

                    "bootstrap_ci_computed":
                        False,
                }
            )


        subset_results = result_rows[
            -25:
        ]


        finite_normalized = [
            item[
                "normalized_wasserstein"
            ]
            for item in subset_results
            if np.isfinite(
                item[
                    "normalized_wasserstein"
                ]
            )
        ]


        max_normalized = (
            max(
                finite_normalized
            )
            if finite_normalized
            else
            np.nan
        )


        print(
            f"{transfer_scope} | "
            f"{stratum}: "
            f"C24 rows={ref_rows}, "
            f"C25 rows={transfer_rows}, "
            f"C24 groups={ref_groups}, "
            f"C25 groups={transfer_groups}, "
            f"features=25, "
            f"max_norm_W={max_normalized:.6f}"
        )


results_df = pd.DataFrame(
    result_rows
)


# ============================================================
# STRUCTURAL VERIFICATION
# ============================================================

EXPECTED_RESULT_ROWS = (
    4
    *
    3
    *
    25
)


if len(
    results_df
) != EXPECTED_RESULT_ROWS:

    raise RuntimeError(
        "Expected exactly 300 marginal "
        "feature-shift rows."
    )


if results_df.duplicated(
    subset=[
        "transfer_scope",
        "stratum",
        "feature",
    ]
).any():

    raise RuntimeError(
        "Duplicate marginal-shift "
        "comparison detected."
    )


if not np.isfinite(
    results_df[
        "ks_statistic"
    ].to_numpy(
        dtype=np.float64
    )
).all():

    raise RuntimeError(
        "Non-finite KS statistic detected."
    )


if not np.isfinite(
    results_df[
        "wasserstein_distance"
    ].to_numpy(
        dtype=np.float64
    )
).all():

    raise RuntimeError(
        "Non-finite Wasserstein "
        "distance detected."
    )


if (
    (
        results_df[
            "ks_statistic"
        ]
        <
        0.0
    ).any()
    or
    (
        results_df[
            "ks_statistic"
        ]
        >
        1.0
    ).any()
):

    raise RuntimeError(
        "KS statistic outside [0,1]."
    )


print(
    "\nMarginal result rows:",
    len(
        results_df
    )
)

print(
    "Marginal result structure:",
    "PASS"
)


# ============================================================
# SAVE POINT RESULTS
# ============================================================

OUTPUT_POINT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


results_df.to_csv(
    OUTPUT_POINT_FILE,
    index=False,
)


# ============================================================
# HASHES
# ============================================================

point_hash = sha256_file(
    OUTPUT_POINT_FILE
)

missingness_hash = sha256_file(
    OUTPUT_MISSINGNESS_FILE
)


# ============================================================
# ROUND TRIP
# ============================================================

point_roundtrip = pd.read_csv(
    OUTPUT_POINT_FILE,
    keep_default_na=False,
)


missingness_roundtrip = pd.read_csv(
    OUTPUT_MISSINGNESS_FILE,
    keep_default_na=False,
)


if len(
    point_roundtrip
) != EXPECTED_RESULT_ROWS:

    raise RuntimeError(
        "Point-result CSV round-trip "
        "row-count mismatch."
    )


if len(
    missingness_roundtrip
) != expected_missingness_rows:

    raise RuntimeError(
        "Missingness CSV round-trip "
        "row-count mismatch."
    )


print(
    "Point-result round-trip:",
    "PASS"
)

print(
    "Missingness round-trip:",
    "PASS"
)


# ============================================================
# CONTRACT
# ============================================================

scope_summary = []


for scope_id, dataframe in scope_frames.items():

    scope_summary.append(
        {
            "scope_id":
                scope_id,

            "rows":
                int(
                    len(
                        dataframe
                    )
                ),

            "positive":
                int(
                    dataframe[
                        TARGET_COLUMN
                    ].sum()
                ),

            "negative":
                int(
                    (
                        dataframe[
                            TARGET_COLUMN
                        ]
                        ==
                        0
                    ).sum()
                ),

            "entity_groups":
                int(
                    dataframe[
                        ENTITY_GROUP_COLUMN
                    ].nunique(
                        dropna=False
                    )
                ),
        }
    )


iqr_zero_rows = int(
    results_df[
        "reference_iqr_zero"
    ].sum()
)


contract = {

    "contract_name":
        (
            "Cycle24 Holdout to Cycle25 "
            "Marginal Feature Shift Point Audit"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_MARGINAL_SHIFT_POINT_RESULTS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_9_2A",

    "reference_domain":
        "CYCLE24_HOLDOUT",

    "transfer_domain":
        "CYCLE25",

    "features":
        FEATURES,

    "feature_count":
        25,

    "transfer_scopes":
        TRANSFER_SCOPES,

    "strata":
        STRATA,

    "comparison_count":
        300,

    "scope_populations":
        scope_summary,

    "marginal_metrics":
        {
            "ks_statistic":
                {
                    "definition":
                        (
                            "Two-sample "
                            "Kolmogorov-Smirnov "
                            "D statistic"
                        ),

                    "pvalue_used_for_claims":
                        False,
                },

            "wasserstein_distance":
                (
                    "1D Wasserstein distance "
                    "on raw frozen feature values"
                ),

            "normalized_wasserstein":
                (
                    "Raw Wasserstein distance "
                    "divided by the corresponding "
                    "Cycle24-holdout stratum IQR"
                ),

            "median_shift_iqr_units":
                (
                    "(Cycle25 median - "
                    "Cycle24-holdout median) / "
                    "Cycle24-holdout stratum IQR"
                ),
        },

    "reference_iqr_zero_rows":
        iqr_zero_rows,

    "missingness":
        {
            "scopes":
                list(
                    scope_frames.keys()
                ),

            "feature_rows":
                expected_missingness_rows,

            "all_clean":
                bool(
                    missingness_df[
                        "clean"
                    ].all()
                ),

            "silent_imputation":
                False,
        },

    "artifacts":
        {
            "point_shift_results":
                {
                    "file":
                        str(
                            OUTPUT_POINT_FILE
                        ),

                    "sha256":
                        point_hash,

                    "rows":
                        len(
                            results_df
                        ),
                },

            "missingness_audit":
                {
                    "file":
                        str(
                            OUTPUT_MISSINGNESS_FILE
                        ),

                    "sha256":
                        missingness_hash,

                    "rows":
                        len(
                            missingness_df
                        ),
                },
        },

    "evaluation_state":
        {
            "point_distribution_metrics":
                True,

            "missingness_audit":
                True,

            "entity_group_bootstrap_ci":
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

            "prediction_artifacts_read":
                False,

            "shap_artifacts_read":
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

            "feature_registry":
                {
                    "file":
                        str(
                            FEATURE_REGISTRY_FILE
                        ),

                    "sha256":
                        feature_registry_hash,
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
            "These are descriptive marginal "
            "distribution-shift point estimates. "
            "No feature is declared significantly "
            "shifted from these point values alone. "
            "The frozen ENTITY_GROUP bootstrap "
            "uncertainty analysis must be completed "
            "in Chapter 9.2B."
        ),

    "immutability_rule":
        (
            "Observed point-shift results must "
            "not change the frozen metrics, "
            "strata, Cycle25 temporal blocks, "
            "bootstrap replicate count, bootstrap "
            "unit, or later SHAP-control design."
        ),
}


# ============================================================
# SAVE CONTRACT
# ============================================================

OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


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
# SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "CHAPTER 9.2A SUMMARY"
)

print(
    "===================================="
)


print(
    "Features:",
    25
)

print(
    "Cycle25 scopes:",
    4
)

print(
    "Strata/scope:",
    3
)

print(
    "Marginal comparisons:",
    len(
        results_df
    )
)

print(
    "Missingness audit rows:",
    len(
        missingness_df
    )
)

print(
    "All feature values clean:",
    bool(
        missingness_df[
            "clean"
        ].all()
    )
)

print(
    "Reference-IQR-zero comparisons:",
    iqr_zero_rows
)

print(
    "KS p-values used:",
    False
)

print(
    "Bootstrap CIs computed:",
    False
)

print(
    "Cycle24 TRAIN read:",
    False
)

print(
    "Models loaded:",
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
    OUTPUT_POINT_FILE
)

print(
    OUTPUT_MISSINGNESS_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 9.2A COMPLETE — MARGINAL POINT SHIFT FROZEN"
)

print(
    "===================================="
)