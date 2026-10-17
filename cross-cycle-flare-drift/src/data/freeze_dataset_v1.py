from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path(
    "data/processed"
)

FINAL_DIR = Path(
    "data/final/dataset_v1"
)

MANIFEST_DIR = Path(
    "data_manifests"
)


C24_SOURCE = (
    PROCESSED_DIR
    /
    "cycle24_model_ready_leakage_safe.parquet"
)

C25_SOURCE = (
    PROCESSED_DIR
    /
    "cycle25_model_ready_leakage_safe.parquet"
)


C24_FINAL = (
    FINAL_DIR
    /
    "cycle24_dataset_v1.parquet"
)

C25_FINAL = (
    FINAL_DIR
    /
    "cycle25_dataset_v1.parquet"
)


PHASE11_AUDIT_FILE = (
    MANIFEST_DIR
    /
    "phase11_leakage_safe_final_audit.csv"
)

QUARANTINE_FILE = (
    MANIFEST_DIR
    /
    "phase11_cross_cycle_quarantine_summary.csv"
)

TEMPORAL_CONTRACT_FILE = (
    MANIFEST_DIR
    /
    "temporal_data_contract.json"
)

SAMPLING_CONTRACT_FILE = (
    MANIFEST_DIR
    /
    "sampling_stride_contract.json"
)


FILES_MANIFEST = (
    MANIFEST_DIR
    /
    "dataset_v1_files.csv"
)

SCHEMA_MANIFEST = (
    MANIFEST_DIR
    /
    "dataset_v1_schema.csv"
)

FREEZE_CONTRACT = (
    MANIFEST_DIR
    /
    "dataset_v1_contract.json"
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


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

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


def schema_fingerprint(
    dataframe
):

    schema = [
        {
            "column":
                str(column),

            "dtype":
                str(
                    dataframe[
                        column
                    ].dtype
                ),
        }

        for column in dataframe.columns
    ]


    encoded = json.dumps(
        schema,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )


    return hashlib.sha256(
        encoded
    ).hexdigest()


def immutable_copy(
    source: Path,
    destination: Path
):

    source_hash = sha256_file(
        source
    )


    if destination.exists():

        destination_hash = sha256_file(
            destination
        )


        if (
            source_hash
            !=
            destination_hash
        ):

            raise RuntimeError(
                "\nIMMUTABILITY VIOLATION:\n"
                f"{destination} already exists "
                "but differs from the verified source.\n"
                "Dataset v1 will NOT be overwritten."
            )


        print(
            "Existing frozen file verified:",
            destination
        )


        return source_hash


    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    temporary = destination.with_suffix(
        ".tmp.parquet"
    )


    shutil.copy2(
        source,
        temporary
    )


    copied_hash = sha256_file(
        temporary
    )


    if copied_hash != source_hash:

        if temporary.exists():

            temporary.unlink()

        raise RuntimeError(
            "Dataset copy checksum mismatch."
        )


    temporary.replace(
        destination
    )


    final_hash = sha256_file(
        destination
    )


    if final_hash != source_hash:

        raise RuntimeError(
            "Final frozen dataset checksum mismatch."
        )


    return final_hash


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    C24_SOURCE,
    C25_SOURCE,
    PHASE11_AUDIT_FILE,
    QUARANTINE_FILE,
    TEMPORAL_CONTRACT_FILE,
    SAMPLING_CONTRACT_FILE,
]


for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"Missing required freeze input: "
            f"{file}"
        )


# ============================================================
# LOAD AUDIT CONTRACTS
# ============================================================

phase11 = pd.read_csv(
    PHASE11_AUDIT_FILE
)


quarantine = pd.read_csv(
    QUARANTINE_FILE
)


with TEMPORAL_CONTRACT_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    temporal_contract = json.load(
        file
    )


with SAMPLING_CONTRACT_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    sampling_contract = json.load(
        file
    )


# ============================================================
# VERIFY UPSTREAM FREEZE STATE
# ============================================================

print(
    "===================================="
)

print(
    "PHASE 12 — DATASET v1 FREEZE"
)

print(
    "===================================="
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


if (
    sampling_contract.get(
        "status"
    )
    !=
    "FROZEN"
):

    raise RuntimeError(
        "Sampling contract is not frozen."
    )


if int(
    sampling_contract[
        "primary_sampling_stride_minutes"
    ]
) != 120:

    raise RuntimeError(
        "Sampling stride is not 120 minutes."
    )


if len(
    phase11
) != 2:

    raise RuntimeError(
        "Phase 11 audit does not contain "
        "exactly two cycle rows."
    )


if (
    not phase11[
        "cycle_audit_pass"
    ].astype(bool).all()
):

    raise RuntimeError(
        "Phase 11 contains a failed cycle audit."
    )


print(
    "Temporal contract: FROZEN"
)

print(
    "Sampling contract: FROZEN"
)

print(
    "Phase 11 cycle audits: PASS"
)


# ============================================================
# LOAD FINAL VERIFIED SOURCES
# ============================================================

c24 = pd.read_parquet(
    C24_SOURCE
)


c25 = pd.read_parquet(
    C25_SOURCE
)


print(
    "\n------------------------------------"
)

print(
    "SOURCE DATASETS"
)

print(
    "------------------------------------"
)


print(
    "Cycle24 rows:",
    len(
        c24
    )
)


print(
    "Cycle25 rows:",
    len(
        c25
    )
)


# ============================================================
# SCHEMA VALIDATION
# ============================================================

if list(
    c24.columns
) != list(
    c25.columns
):

    raise RuntimeError(
        "Cycle24 and Cycle25 column schemas differ."
    )


c24_dtypes = [
    str(
        c24[
            column
        ].dtype
    )

    for column in c24.columns
]


c25_dtypes = [
    str(
        c25[
            column
        ].dtype
    )

    for column in c25.columns
]


if (
    c24_dtypes
    !=
    c25_dtypes
):

    raise RuntimeError(
        "Cycle24 and Cycle25 dtypes differ."
    )


missing_features = [
    feature

    for feature in ALL_FEATURES

    if (
        feature
        not in
        c24.columns
        or
        feature
        not in
        c25.columns
    )
]


if missing_features:

    raise RuntimeError(
        "Missing model features: "
        f"{missing_features}"
    )


if len(
    ALL_FEATURES
) != EXPECTED_FEATURE_COUNT:

    raise RuntimeError(
        "Feature contract does not contain 25 features."
    )


print(
    "Cross-cycle schema identical: PASS"
)

print(
    "Feature count:",
    len(
        ALL_FEATURES
    )
)


# ============================================================
# BASIC FINAL INTEGRITY
# ============================================================

for cycle, dataframe in [
    (
        "cycle24",
        c24,
    ),
    (
        "cycle25",
        c25,
    ),
]:

    duplicates = int(
        dataframe[
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
            ~dataframe[
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


    missing_groups = int(
        dataframe[
            "ENTITY_GROUP_ID"
        ]
        .isna()
        .sum()
    )


    features = (
        dataframe[
            ALL_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=float
        )
    )


    nonfinite = int(
        (
            ~np.isfinite(
                features
            )
            .all(
                axis=1
            )
        )
        .sum()
    )


    if (
        duplicates != 0
        or
        invalid_targets != 0
        or
        missing_groups != 0
        or
        nonfinite != 0
    ):

        raise RuntimeError(
            f"{cycle}: final integrity failure."
        )


# ============================================================
# CROSS-CYCLE LEAKAGE CHECK
# ============================================================

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


group_overlap = sorted(
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


harp_overlap = sorted(
    c24_harps
    &
    c25_harps
)


if group_overlap:

    raise RuntimeError(
        "Cross-cycle ENTITY_GROUP_ID leakage remains."
    )


if harp_overlap:

    raise RuntimeError(
        "Cross-cycle HARPNUM overlap remains."
    )


print(
    "Cross-cycle ENTITY_GROUP_ID overlap: 0"
)

print(
    "Cross-cycle HARPNUM overlap: 0"
)


# ============================================================
# RECONCILE AGAINST PHASE 11
# ============================================================

expected = {}


for _, row in phase11.iterrows():

    cycle = str(
        row[
            "cycle"
        ]
    )


    expected[
        cycle
    ] = {
        "rows":
            int(
                row[
                    "rows"
                ]
            ),

        "positive":
            int(
                row[
                    "positive"
                ]
            ),

        "negative":
            int(
                row[
                    "negative"
                ]
            ),
    }


actual = {}


for cycle, dataframe in [
    (
        "cycle24",
        c24,
    ),
    (
        "cycle25",
        c25,
    ),
]:

    actual[
        cycle
    ] = {
        "rows":
            len(
                dataframe
            ),

        "positive":
            int(
                (
                    dataframe[
                        "target"
                    ]
                    ==
                    1
                )
                .sum()
            ),

        "negative":
            int(
                (
                    dataframe[
                        "target"
                    ]
                    ==
                    0
                )
                .sum()
            ),
    }


    if (
        actual[
            cycle
        ]
        !=
        expected[
            cycle
        ]
    ):

        raise RuntimeError(
            f"{cycle}: does not reconcile "
            "with Phase 11 final audit."
        )


print(
    "Phase 11 count reconciliation: PASS"
)


# ============================================================
# SCHEMA FINGERPRINT
# ============================================================

c24_schema_hash = schema_fingerprint(
    c24
)


c25_schema_hash = schema_fingerprint(
    c25
)


if (
    c24_schema_hash
    !=
    c25_schema_hash
):

    raise RuntimeError(
        "Cross-cycle schema fingerprints differ."
    )


print(
    "Schema fingerprint:",
    c24_schema_hash
)


# ============================================================
# IMMUTABLE COPY
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "FREEZING DATASET FILES"
)

print(
    "------------------------------------"
)


c24_hash = immutable_copy(
    C24_SOURCE,
    C24_FINAL
)


print(
    "Cycle24 SHA256:",
    c24_hash
)


c25_hash = immutable_copy(
    C25_SOURCE,
    C25_FINAL
)


print(
    "Cycle25 SHA256:",
    c25_hash
)


# ============================================================
# VERIFY FROZEN COPY CONTENT
# ============================================================

frozen_c24 = pd.read_parquet(
    C24_FINAL
)


frozen_c25 = pd.read_parquet(
    C25_FINAL
)


if len(
    frozen_c24
) != len(
    c24
):

    raise RuntimeError(
        "Frozen Cycle24 row count mismatch."
    )


if len(
    frozen_c25
) != len(
    c25
):

    raise RuntimeError(
        "Frozen Cycle25 row count mismatch."
    )


if sha256_file(
    C24_FINAL
) != c24_hash:

    raise RuntimeError(
        "Frozen Cycle24 checksum changed."
    )


if sha256_file(
    C25_FINAL
) != c25_hash:

    raise RuntimeError(
        "Frozen Cycle25 checksum changed."
    )


print(
    "Frozen copy verification: PASS"
)


# ============================================================
# SCHEMA MANIFEST
# ============================================================

schema_rows = []


for position, column in enumerate(
    c24.columns,
    start=1
):

    if column in SHARP_FEATURES:

        role = "MODEL_FEATURE_SHARP"


    elif column in LORENTZ_FEATURES:

        role = "MODEL_FEATURE_LORENTZ"


    elif column == "target":

        role = "TARGET"


    elif column == "ENTITY_GROUP_ID":

        role = "LEAKAGE_GROUP"


    elif column in [
        "T_REC",
        "observation_time_utc",
    ]:

        role = "TIME_METADATA"


    else:

        role = "AUDIT_METADATA"


    schema_rows.append(
        {
            "position":
                position,

            "column":
                column,

            "dtype":
                str(
                    c24[
                        column
                    ].dtype
                ),

            "role":
                role,

            "is_model_feature":
                column
                in
                ALL_FEATURES,
        }
    )


schema_df = pd.DataFrame(
    schema_rows
)


schema_df.to_csv(
    SCHEMA_MANIFEST,
    index=False
)


# ============================================================
# FILE MANIFEST
# ============================================================

file_rows = []


for cycle, path, dataframe, digest in [

    (
        "cycle24",
        C24_FINAL,
        frozen_c24,
        c24_hash,
    ),

    (
        "cycle25",
        C25_FINAL,
        frozen_c25,
        c25_hash,
    ),
]:

    positives = int(
        (
            dataframe[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    negatives = int(
        (
            dataframe[
                "target"
            ]
            ==
            0
        )
        .sum()
    )


    file_rows.append(
        {
            "dataset_version":
                "1.0",

            "cycle":
                cycle,

            "file":
                str(
                    path
                ),

            "rows":
                len(
                    dataframe
                ),

            "columns":
                len(
                    dataframe.columns
                ),

            "model_features":
                EXPECTED_FEATURE_COUNT,

            "positive":
                positives,

            "negative":
                negatives,

            "positive_rate_pct":
                (
                    positives
                    /
                    len(
                        dataframe
                    )
                    *
                    100
                ),

            "unique_harpnums":
                int(
                    dataframe[
                        "HARPNUM"
                    ]
                    .nunique()
                ),

            "unique_entity_groups":
                int(
                    dataframe[
                        "ENTITY_GROUP_ID"
                    ]
                    .nunique()
                ),

            "schema_sha256":
                c24_schema_hash,

            "file_sha256":
                digest,

            "file_size_bytes":
                path.stat().st_size,
        }
    )


files_df = pd.DataFrame(
    file_rows
)


files_df.to_csv(
    FILES_MANIFEST,
    index=False
)


# ============================================================
# QUARANTINE INFORMATION
# ============================================================

quarantined_groups = sorted(
    quarantine[
        "ENTITY_GROUP_ID"
    ]
    .astype(str)
    .unique()
    .tolist()
)


cycle24_removed = int(
    quarantine[
        "cycle24_rows_removed"
    ].sum()
)


cycle25_removed = int(
    quarantine[
        "cycle25_rows_removed"
    ].sum()
)


# ============================================================
# PROVENANCE HASHES
# ============================================================

provenance_files = {

    "phase11_final_audit":
        PHASE11_AUDIT_FILE,

    "cross_cycle_quarantine":
        QUARANTINE_FILE,

    "temporal_contract":
        TEMPORAL_CONTRACT_FILE,

    "sampling_contract":
        SAMPLING_CONTRACT_FILE,
}


provenance_hashes = {}


for name, path in provenance_files.items():

    provenance_hashes[
        name
    ] = {
        "file":
            str(
                path
            ),

        "sha256":
            sha256_file(
                path
            ),
    }


# ============================================================
# DATASET v1 CONTRACT
# ============================================================

contract = {

    "dataset_name":
        "Cross-Cycle Solar Flare Dataset",

    "dataset_version":
        "1.0",

    "status":
        "FROZEN",

    "frozen_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "research_role": {

        "cycle24":
            (
                "Development domain. "
                "Future train/validation/independent-holdout "
                "splits must respect ENTITY_GROUP_ID."
            ),

        "cycle25":
            (
                "Frozen temporal transfer/test domain. "
                "Must not be used for model fitting, "
                "preprocessing fitting, feature selection, "
                "hyperparameter tuning, or threshold tuning."
            ),
    },

    "feature_contract": {

        "total_model_features":
            25,

        "sharp_feature_count":
            18,

        "lorentz_feature_count":
            7,

        "sharp_features":
            SHARP_FEATURES,

        "lorentz_features":
            LORENTZ_FEATURES,
    },

    "sampling_contract": {

        "native_cadence_minutes":
            12,

        "primary_stride_minutes":
            120,

        "sampling_anchor":
            "00:00 TAI",

        "sampling_method":
            (
                "Fixed absolute 2-hour TAI timestamp grid; "
                "never every-Nth-available-row sampling."
            ),
    },

    "label_contract": {

        "target":
            (
                "At least one GOES M1.0+ flare associated "
                "with the SHARP row NOAA metadata "
                "within the next 24 hours."
            ),

        "event_timestamp":
            "GOES flare start_time",

        "forecast_window":
            "(t, t+24h]",

        "positive":
            (
                "At least one matching region-assigned "
                "M1.0+ event in the forecast window."
            ),

        "negative":
            (
                "No matching assigned M1.0+ event and "
                "no region-unassigned M1.0+ event "
                "in the forecast window."
            ),

        "unassigned_m1_policy":
            "CENSOR",

        "no_noaa_association_policy":
            "EXCLUDE",

        "boundary_policy":
            (
                "Require complete 24-hour outcome horizon "
                "inside the frozen cycle domain."
            ),

        "goes_region_normalization":
            (
                "For audited GOES catalogs: "
                "active_region < 10000 -> +10000; "
                "already-full NOAA values retained. "
                "No generic modulo normalization."
            ),
    },

    "qc_contract": {

        "QC1":
            "Valid HARPNUM and T_REC; unique key.",

        "QC2":
            (
                "Exact one-to-one SHARP/Lorentz merge "
                "on HARPNUM + T_REC."
            ),

        "QC3":
            "QUALITY == 0",

        "QC4":
            "CMASK > 0",

        "QC5":
            (
                "Position available and "
                "|LON_FWT| <= 68 degrees."
            ),

        "QC6":
            (
                "All 25 model features finite; "
                "no blind imputation."
            ),
    },

    "leakage_contract": {

        "primary_leakage_unit":
            "connected HARP-NOAA ENTITY_GROUP_ID",

        "cross_cycle_entity_overlap_allowed":
            False,

        "cross_cycle_harp_overlap_allowed":
            False,

        "quarantined_entity_groups":
            quarantined_groups,

        "cycle24_rows_quarantined":
            cycle24_removed,

        "cycle25_rows_quarantined":
            cycle25_removed,

        "quarantine_rule":
            (
                "Any ENTITY_GROUP_ID occurring in both "
                "Cycle24 and Cycle25 primary datasets is "
                "removed from both domains. "
                "Rule is label-blind."
            ),
    },

    "cycle24": {

        "file":
            str(
                C24_FINAL
            ),

        "rows":
            len(
                frozen_c24
            ),

        "positive":
            int(
                (
                    frozen_c24[
                        "target"
                    ]
                    ==
                    1
                )
                .sum()
            ),

        "negative":
            int(
                (
                    frozen_c24[
                        "target"
                    ]
                    ==
                    0
                )
                .sum()
            ),

        "unique_entity_groups":
            int(
                frozen_c24[
                    "ENTITY_GROUP_ID"
                ]
                .nunique()
            ),

        "sha256":
            c24_hash,
    },

    "cycle25": {

        "file":
            str(
                C25_FINAL
            ),

        "rows":
            len(
                frozen_c25
            ),

        "positive":
            int(
                (
                    frozen_c25[
                        "target"
                    ]
                    ==
                    1
                )
                .sum()
            ),

        "negative":
            int(
                (
                    frozen_c25[
                        "target"
                    ]
                    ==
                    0
                )
                .sum()
            ),

        "unique_entity_groups":
            int(
                frozen_c25[
                    "ENTITY_GROUP_ID"
                ]
                .nunique()
            ),

        "sha256":
            c25_hash,
    },

    "schema_sha256":
        c24_schema_hash,

    "provenance":
        provenance_hashes,

    "immutability_rule":
        (
            "Dataset v1 files must not be modified in place. "
            "Any methodological or data change requires "
            "a new dataset version."
        ),

    "modeling_gate": {

        "dataset_frozen":
            True,

        "cycle25_tuning_allowed":
            False,

        "modeling_may_begin":
            True,
    },
}


# ============================================================
# WRITE CONTRACT SAFELY
# ============================================================

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


if FREEZE_CONTRACT.exists():

    with FREEZE_CONTRACT.open(
        "r",
        encoding="utf-8"
    ) as file:

        existing_contract = json.load(
            file
        )


    existing_c24_hash = (
        existing_contract
        .get(
            "cycle24",
            {}
        )
        .get(
            "sha256"
        )
    )


    existing_c25_hash = (
        existing_contract
        .get(
            "cycle25",
            {}
        )
        .get(
            "sha256"
        )
    )


    if (
        existing_c24_hash
        !=
        c24_hash
        or
        existing_c25_hash
        !=
        c25_hash
    ):

        raise RuntimeError(
            "dataset_v1_contract.json already exists "
            "for different dataset hashes. "
            "Do not overwrite Dataset v1."
        )


    print(
        "Existing Dataset v1 contract "
        "matches frozen dataset hashes."
    )


else:

    with FREEZE_CONTRACT.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            contract,
            file,
            indent=2,
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

print(
    "\n===================================="
)

print(
    "DATASET v1 FINAL VALIDATION"
)

print(
    "===================================="
)


print(
    "Dataset version: 1.0"
)


print(
    "Status: FROZEN"
)


print(
    "\nCycle24"
)


print(
    "Rows:",
    len(
        frozen_c24
    )
)


print(
    "Positive:",
    int(
        (
            frozen_c24[
                "target"
            ]
            ==
            1
        )
        .sum()
    )
)


print(
    "Negative:",
    int(
        (
            frozen_c24[
                "target"
            ]
            ==
            0
        )
        .sum()
    )
)


print(
    "Entity groups:",
    int(
        frozen_c24[
            "ENTITY_GROUP_ID"
        ]
        .nunique()
    )
)


print(
    "\nCycle25"
)


print(
    "Rows:",
    len(
        frozen_c25
    )
)


print(
    "Positive:",
    int(
        (
            frozen_c25[
                "target"
            ]
            ==
            1
        )
        .sum()
    )
)


print(
    "Negative:",
    int(
        (
            frozen_c25[
                "target"
            ]
            ==
            0
        )
        .sum()
    )
)


print(
    "Entity groups:",
    int(
        frozen_c25[
            "ENTITY_GROUP_ID"
        ]
        .nunique()
    )
)


print(
    "\nModel features:",
    EXPECTED_FEATURE_COUNT
)


print(
    "Schema identical: PASS"
)


print(
    "Cross-cycle entity overlap: 0"
)


print(
    "Cross-cycle HARP overlap: 0"
)


print(
    "Phase11 reconciliation: PASS"
)


print(
    "Frozen file checksum validation: PASS"
)


print(
    "\nSaved:"
)


print(
    C24_FINAL
)


print(
    C25_FINAL
)


print(
    FILES_MANIFEST
)


print(
    SCHEMA_MANIFEST
)


print(
    FREEZE_CONTRACT
)


print(
    "\n===================================="
)

print(
    "PHASE 12 COMPLETE — DATASET v1 FROZEN"
)

print(
    "===================================="
)