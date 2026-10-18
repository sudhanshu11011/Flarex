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

CANDIDATE_DIR = Path(
    "data/processed/dataset_v2_candidate"
)

FINAL_DIR = Path(
    "data/final/dataset_v2"
)

MANIFEST_DIR = Path(
    "data_manifests"
)


C24_SOURCE = (
    CANDIDATE_DIR
    /
    "cycle24_dataset_v2_candidate.parquet"
)

C25_SOURCE = (
    CANDIDATE_DIR
    /
    "cycle25_dataset_v2_candidate.parquet"
)


C24_FINAL = (
    FINAL_DIR
    /
    "cycle24_dataset_v2.parquet"
)

C25_FINAL = (
    FINAL_DIR
    /
    "cycle25_dataset_v2.parquet"
)


FULL_AUDIT_FILE = (
    MANIFEST_DIR
    /
    "pre_model_dataset_v2_full_audit.csv"
)

CANDIDATE_SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "pre_model_dataset_v2_candidate_summary.csv"
)

CORRECTIONS_FILE = (
    MANIFEST_DIR
    /
    "pre_model_dataset_v2_label_corrections.csv"
)

SRS_SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "pre_model_srs_all_positive_summary.csv"
)

SRS_ROW_AUDIT_FILE = (
    MANIFEST_DIR
    /
    "pre_model_srs_all_positive_row_audit.csv"
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

DATASET_V1_CONTRACT_FILE = (
    MANIFEST_DIR
    /
    "dataset_v1_contract.json"
)


FILES_MANIFEST = (
    MANIFEST_DIR
    /
    "dataset_v2_files.csv"
)

SCHEMA_MANIFEST = (
    MANIFEST_DIR
    /
    "dataset_v2_schema.csv"
)

FREEZE_CONTRACT = (
    MANIFEST_DIR
    /
    "dataset_v2_contract.json"
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


def schema_fingerprint(
    dataframe
):

    schema = [
        {
            "column":
                str(column),

            "dtype":
                str(
                    dataframe[column].dtype
                ),
        }

        for column in dataframe.columns
    ]


    encoded = json.dumps(
        schema,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


    return hashlib.sha256(
        encoded
    ).hexdigest()


def immutable_copy(
    source: Path,
    destination: Path,
):

    source_hash = sha256_file(
        source
    )


    if destination.exists():

        destination_hash = sha256_file(
            destination
        )


        if destination_hash != source_hash:

            raise RuntimeError(
                "\nIMMUTABILITY VIOLATION:\n"
                f"{destination} already exists "
                "but differs from the verified candidate.\n"
                "Dataset v2 will NOT be overwritten."
            )


        print(
            "Existing frozen file verified:",
            destination
        )

        return source_hash


    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temporary = destination.with_suffix(
        ".tmp.parquet"
    )


    shutil.copy2(
        source,
        temporary,
    )


    copied_hash = sha256_file(
        temporary
    )


    if copied_hash != source_hash:

        temporary.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Temporary copy checksum mismatch."
        )


    temporary.replace(
        destination
    )


    final_hash = sha256_file(
        destination
    )


    if final_hash != source_hash:

        raise RuntimeError(
            "Frozen file checksum mismatch."
        )


    return final_hash


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    C24_SOURCE,
    C25_SOURCE,
    FULL_AUDIT_FILE,
    CANDIDATE_SUMMARY_FILE,
    CORRECTIONS_FILE,
    SRS_SUMMARY_FILE,
    SRS_ROW_AUDIT_FILE,
    TEMPORAL_CONTRACT_FILE,
    SAMPLING_CONTRACT_FILE,
    DATASET_V1_CONTRACT_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required freeze input: "
            f"{path}"
        )


# ============================================================
# LOAD CONTRACTS / AUDITS
# ============================================================

full_audit = pd.read_csv(
    FULL_AUDIT_FILE
)


candidate_summary = pd.read_csv(
    CANDIDATE_SUMMARY_FILE
)


corrections = pd.read_csv(
    CORRECTIONS_FILE
)


srs_summary = pd.read_csv(
    SRS_SUMMARY_FILE
)


srs_rows = pd.read_csv(
    SRS_ROW_AUDIT_FILE
)


with TEMPORAL_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    temporal_contract = json.load(
        file
    )


with SAMPLING_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    sampling_contract = json.load(
        file
    )


with DATASET_V1_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    dataset_v1_contract = json.load(
        file
    )


# ============================================================
# VERIFY UPSTREAM STATE
# ============================================================

print(
    "===================================="
)

print(
    "DATASET v2 FREEZE"
)

print(
    "===================================="
)


if temporal_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Temporal contract is not frozen."
    )


if sampling_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Sampling contract is not frozen."
    )


if len(full_audit) != 2:

    raise RuntimeError(
        "Dataset v2 full audit should "
        "contain exactly two cycle rows."
    )


if (
    not full_audit[
        "cycle_audit_pass"
    ]
    .astype(bool)
    .all()
):

    raise RuntimeError(
        "Dataset v2 full audit contains failure."
    )


if len(srs_rows) != 8836:

    raise RuntimeError(
        "SRS all-positive audit row count "
        "does not equal 8836."
    )


print(
    "Temporal contract: FROZEN"
)

print(
    "Sampling contract: FROZEN"
)

print(
    "Dataset v2 full audit: PASS"
)

print(
    "SRS positive audit rows:",
    len(srs_rows)
)


# ============================================================
# LOAD CANDIDATES
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
    "SOURCE CANDIDATES"
)

print(
    "------------------------------------"
)


print(
    "Cycle24 rows:",
    len(c24)
)

print(
    "Cycle25 rows:",
    len(c25)
)


# ============================================================
# SCHEMA + FEATURE CONTRACT
# ============================================================

if list(c24.columns) != list(
    c25.columns
):

    raise RuntimeError(
        "Cross-cycle column schema differs."
    )


c24_dtypes = [
    str(c24[column].dtype)
    for column in c24.columns
]


c25_dtypes = [
    str(c25[column].dtype)
    for column in c25.columns
]


if c24_dtypes != c25_dtypes:

    raise RuntimeError(
        "Cross-cycle dtype schema differs."
    )


missing_features = [
    feature

    for feature in ALL_FEATURES

    if feature not in c24.columns
    or feature not in c25.columns
]


if missing_features:

    raise RuntimeError(
        "Missing features: "
        f"{missing_features}"
    )


if len(ALL_FEATURES) != 25:

    raise RuntimeError(
        "Feature contract is not 25."
    )


schema_hash = schema_fingerprint(
    c24
)


if schema_hash != schema_fingerprint(
    c25
):

    raise RuntimeError(
        "Schema fingerprints differ."
    )


print(
    "Schema identical: PASS"
)

print(
    "Feature count:",
    len(ALL_FEATURES)
)

print(
    "Schema fingerprint:",
    schema_hash
)


# ============================================================
# FINAL DATA INTEGRITY
# ============================================================

for cycle, dataframe in [
    ("cycle24", c24),
    ("cycle25", c25),
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


    invalid_target = int(
        (
            ~dataframe["target"]
            .isin([0, 1])
        ).sum()
    )


    status_mismatch = int(
        (
            (
                dataframe["target"] == 1
            )
            &
            (
                dataframe["label_status"]
                != "POSITIVE"
            )
        ).sum()
        +
        (
            (
                dataframe["target"] == 0
            )
            &
            (
                dataframe["label_status"]
                != "NEGATIVE"
            )
        ).sum()
    )


    missing_groups = int(
        dataframe[
            "ENTITY_GROUP_ID"
        ]
        .isna()
        .sum()
    )


    numeric = (
        dataframe[
            ALL_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )


    nonfinite = int(
        (
            ~np.isfinite(
                numeric
            )
            .all(
                axis=1
            )
        )
        .sum()
    )


    if (
        duplicates != 0
        or invalid_target != 0
        or status_mismatch != 0
        or missing_groups != 0
        or nonfinite != 0
    ):

        raise RuntimeError(
            f"{cycle}: final integrity failure."
        )


# ============================================================
# CROSS-CYCLE LEAKAGE
# ============================================================

entity_overlap = (
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


harp_overlap = (
    set(
        pd.to_numeric(
            c24["HARPNUM"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    &
    set(
        pd.to_numeric(
            c25["HARPNUM"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
)


if entity_overlap:

    raise RuntimeError(
        "Cross-cycle ENTITY_GROUP_ID overlap."
    )


if harp_overlap:

    raise RuntimeError(
        "Cross-cycle HARPNUM overlap."
    )


print(
    "Cross-cycle ENTITY_GROUP_ID overlap: 0"
)

print(
    "Cross-cycle HARPNUM overlap: 0"
)


# ============================================================
# RECONCILE WITH AUDIT
# ============================================================

for cycle, dataframe in [
    ("cycle24", c24),
    ("cycle25", c25),
]:

    row = full_audit[
        full_audit["cycle"]
        ==
        cycle
    ]


    if len(row) != 1:

        raise RuntimeError(
            f"{cycle}: audit row unavailable."
        )


    row = row.iloc[0]


    actual = {
        "rows":
            len(dataframe),

        "positive":
            int(
                (
                    dataframe["target"]
                    ==
                    1
                ).sum()
            ),

        "negative":
            int(
                (
                    dataframe["target"]
                    ==
                    0
                ).sum()
            ),
    }


    expected = {
        "rows":
            int(row["rows"]),

        "positive":
            int(row["positive"]),

        "negative":
            int(row["negative"]),
    }


    if actual != expected:

        raise RuntimeError(
            f"{cycle}: final audit "
            "reconciliation failed."
        )


print(
    "Fix J reconciliation: PASS"
)


# ============================================================
# CORRECTION ACCOUNTING
# ============================================================

action_counts = (
    corrections[
        "dataset_v2_action"
    ]
    .value_counts()
)


temporal_censor = int(
    action_counts.get(
        "CENSOR_TEMPORAL_AMBIGUITY",
        0
    )
)


source_missing_censor = int(
    action_counts.get(
        "CENSOR_SRS_SOURCE_MISSING",
        0
    )
)


unassigned_censor = int(
    action_counts.get(
        "CENSOR_UNASSIGNED_M1_AFTER_STALE_ASSOCIATION",
        0
    )
)


stale_relabeled = int(
    action_counts.get(
        "RELABEL_NEGATIVE_STALE_ASSOCIATION",
        0
    )
)


total_censored = (
    temporal_censor
    +
    source_missing_censor
    +
    unassigned_censor
)


if (
    temporal_censor != 393
    or
    source_missing_censor != 42
    or
    unassigned_censor != 8
    or
    stale_relabeled != 95
):

    raise RuntimeError(
        "Correction counts differ from "
        "verified Fix I output."
    )


print(
    "Correction accounting: PASS"
)


# ============================================================
# IMMUTABLE FREEZE
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "FREEZING DATASET v2"
)

print(
    "------------------------------------"
)


c24_hash = immutable_copy(
    C24_SOURCE,
    C24_FINAL,
)


c25_hash = immutable_copy(
    C25_SOURCE,
    C25_FINAL,
)


print(
    "Cycle24 SHA256:",
    c24_hash
)


print(
    "Cycle25 SHA256:",
    c25_hash
)


# ============================================================
# VERIFY COPIES
# ============================================================

frozen_c24 = pd.read_parquet(
    C24_FINAL
)

frozen_c25 = pd.read_parquet(
    C25_FINAL
)


if len(frozen_c24) != len(c24):

    raise RuntimeError(
        "Cycle24 frozen row mismatch."
    )


if len(frozen_c25) != len(c25):

    raise RuntimeError(
        "Cycle25 frozen row mismatch."
    )


if sha256_file(
    C24_FINAL
) != c24_hash:

    raise RuntimeError(
        "Cycle24 frozen hash mismatch."
    )


if sha256_file(
    C25_FINAL
) != c25_hash:

    raise RuntimeError(
        "Cycle25 frozen hash mismatch."
    )


print(
    "Frozen copy verification: PASS"
)


# ============================================================
# SCHEMA MANIFEST
# ============================================================

schema_rows = []


for position, column in enumerate(
    frozen_c24.columns,
    start=1,
):

    if column in SHARP_FEATURES:
        role = "MODEL_FEATURE_SHARP"

    elif column in LORENTZ_FEATURES:
        role = "MODEL_FEATURE_LORENTZ"

    elif column == "target":
        role = "TARGET"

    elif column == "ENTITY_GROUP_ID":
        role = "LEAKAGE_GROUP"

    elif column in {
        "T_REC",
        "observation_time_utc",
    }:
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
                    frozen_c24[
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
    index=False,
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

    positive = int(
        (
            dataframe["target"] == 1
        ).sum()
    )


    negative = int(
        (
            dataframe["target"] == 0
        ).sum()
    )


    file_rows.append(
        {
            "dataset_version":
                "2.0",

            "cycle":
                cycle,

            "file":
                str(path),

            "rows":
                len(dataframe),

            "columns":
                len(
                    dataframe.columns
                ),

            "model_features":
                25,

            "positive":
                positive,

            "negative":
                negative,

            "positive_rate_pct":
                positive
                /
                len(dataframe)
                *
                100,

            "unique_harpnums":
                int(
                    dataframe[
                        "HARPNUM"
                    ].nunique()
                ),

            "unique_entity_groups":
                int(
                    dataframe[
                        "ENTITY_GROUP_ID"
                    ].nunique()
                ),

            "schema_sha256":
                schema_hash,

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
    index=False,
)


# ============================================================
# PROVENANCE HASHES
# ============================================================

provenance_paths = {
    "dataset_v1_contract":
        DATASET_V1_CONTRACT_FILE,

    "dataset_v2_candidate_audit":
        FULL_AUDIT_FILE,

    "dataset_v2_candidate_summary":
        CANDIDATE_SUMMARY_FILE,

    "dataset_v2_label_corrections":
        CORRECTIONS_FILE,

    "srs_all_positive_summary":
        SRS_SUMMARY_FILE,

    "srs_all_positive_row_audit":
        SRS_ROW_AUDIT_FILE,

    "temporal_contract":
        TEMPORAL_CONTRACT_FILE,

    "sampling_contract":
        SAMPLING_CONTRACT_FILE,
}


provenance = {}


for name, path in provenance_paths.items():

    provenance[name] = {
        "file":
            str(path),

        "sha256":
            sha256_file(
                path
            ),
    }


# ============================================================
# DATASET v2 CONTRACT
# ============================================================

contract = {

    "dataset_name":
        "Cross-Cycle Solar Flare Dataset",

    "dataset_version":
        "2.0",

    "status":
        "FROZEN",

    "frozen_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "supersedes_for_primary_modeling":
        "Dataset v1.0",

    "dataset_v1_preserved":
        True,

    "reason_for_v2":
        (
            "Dataset v1 used static SHARP NOAA association "
            "metadata for labels. Pre-model audits showed "
            "NOAA_ARS was effectively lifetime-level rather "
            "than time-resolved. Dataset v2 validates positive "
            "region associations using daily NOAA/NCEI Solar "
            "Region Summary temporal bracketing."
        ),

    "research_role": {

        "cycle24":
            (
                "Development domain. Future train, validation, "
                "and independent holdout partitions must remain "
                "ENTITY_GROUP_ID-safe."
            ),

        "cycle25":
            (
                "Frozen temporal transfer domain. "
                "No model fitting, preprocessing fitting, "
                "feature selection, hyperparameter tuning, "
                "or threshold tuning is permitted."
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

        "primary_stride_minutes":
            120,

        "sampling_anchor":
            "00:00 TAI",

        "sampling_method":
            (
                "Fixed absolute two-hour TAI timestamp grid."
            ),
    },

    "label_contract": {

        "threshold":
            "GOES M1.0+",

        "event_timestamp":
            "GOES start_time",

        "forecast_window":
            "(t, t+24h]",

        "srs_temporal_validation":
            True,

        "srs_rule":
            (
                "Positive region association is supported when "
                "at least one matched flare NOAA region is "
                "present in both Solar Region Summary snapshots "
                "bracketing the UTC observation date."
            ),

        "temporal_ambiguity_policy":
            "CENSOR",

        "missing_srs_source_policy":
            "CENSOR",

        "stale_association_policy":
            (
                "If all matched associations are absent from "
                "both SRS bracketing reports, the former "
                "positive is relabeled NEGATIVE unless a "
                "region-unassigned M1.0+ flare occurs in the "
                "forecast window, in which case it is censored."
            ),

        "region_unassigned_m1_policy":
            "CENSOR",
    },

    "correction_accounting": {

        "censor_temporal_ambiguity":
            temporal_censor,

        "censor_srs_source_missing":
            source_missing_censor,

        "censor_unassigned_m1_after_stale":
            unassigned_censor,

        "relabeled_negative_stale_association":
            stale_relabeled,

        "total_rows_removed":
            total_censored,
    },

    "leakage_contract": {

        "unit":
            "connected HARP-NOAA ENTITY_GROUP_ID",

        "cross_cycle_entity_overlap":
            0,

        "cross_cycle_harp_overlap":
            0,
    },

    "cycle24": {

        "file":
            str(C24_FINAL),

        "rows":
            len(frozen_c24),

        "positive":
            int(
                (
                    frozen_c24[
                        "target"
                    ]
                    ==
                    1
                ).sum()
            ),

        "negative":
            int(
                (
                    frozen_c24[
                        "target"
                    ]
                    ==
                    0
                ).sum()
            ),

        "unique_entity_groups":
            int(
                frozen_c24[
                    "ENTITY_GROUP_ID"
                ].nunique()
            ),

        "sha256":
            c24_hash,
    },

    "cycle25": {

        "file":
            str(C25_FINAL),

        "rows":
            len(frozen_c25),

        "positive":
            int(
                (
                    frozen_c25[
                        "target"
                    ]
                    ==
                    1
                ).sum()
            ),

        "negative":
            int(
                (
                    frozen_c25[
                        "target"
                    ]
                    ==
                    0
                ).sum()
            ),

        "unique_entity_groups":
            int(
                frozen_c25[
                    "ENTITY_GROUP_ID"
                ].nunique()
            ),

        "sha256":
            c25_hash,
    },

    "schema_sha256":
        schema_hash,

    "provenance":
        provenance,

    "immutability_rule":
        (
            "Dataset v2 files must never be modified "
            "in place. Any further methodological or "
            "data change requires Dataset v3."
        ),

    "modeling_gate": {

        "dataset_frozen":
            True,

        "cycle25_tuning_allowed":
            False,

        "modeling_pending_methodology_closure":
            True,
    },
}


# ============================================================
# SAFE CONTRACT WRITE
# ============================================================

if FREEZE_CONTRACT.exists():

    with FREEZE_CONTRACT.open(
        "r",
        encoding="utf-8",
    ) as file:

        existing = json.load(
            file
        )


    existing_c24 = (
        existing.get(
            "cycle24",
            {}
        ).get(
            "sha256"
        )
    )


    existing_c25 = (
        existing.get(
            "cycle25",
            {}
        ).get(
            "sha256"
        )
    )


    if (
        existing_c24 != c24_hash
        or
        existing_c25 != c25_hash
    ):

        raise RuntimeError(
            "Existing dataset_v2_contract.json "
            "belongs to different dataset hashes. "
            "Do not overwrite Dataset v2."
        )


    print(
        "Existing Dataset v2 contract "
        "matches dataset hashes."
    )


else:

    with FREEZE_CONTRACT.open(
        "w",
        encoding="utf-8",
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
    "DATASET v2 FINAL VALIDATION"
)

print(
    "===================================="
)


print(
    "Dataset version: 2.0"
)

print(
    "Status: FROZEN"
)


print(
    "\nCycle24"
)

print(
    "Rows:",
    len(frozen_c24)
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
        ).sum()
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
        ).sum()
    )
)


print(
    "\nCycle25"
)

print(
    "Rows:",
    len(frozen_c25)
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
        ).sum()
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
        ).sum()
    )
)


print(
    "\nModel features: 25"
)

print(
    "Schema identical: PASS"
)

print(
    "Cross-cycle ENTITY overlap: 0"
)

print(
    "Cross-cycle HARP overlap: 0"
)

print(
    "Fix J reconciliation: PASS"
)

print(
    "Frozen checksum validation: PASS"
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
    "DATASET v2 FROZEN SUCCESSFULLY"
)

print(
    "===================================="
)