from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

SOURCE_DATASET = Path(
    "data/final/dataset_v2/cycle24_dataset_v2.parquet"
)

DATASET_CONTRACT = Path(
    "data_manifests/dataset_v2_contract.json"
)

PRE_MODEL_GATE = Path(
    "data_manifests/pre_model_gate.json"
)


ASSIGNMENT_FILE = Path(
    "data_manifests/model_ch1_split_assignments.csv"
)

SEARCH_SUMMARY_FILE = Path(
    "data_manifests/model_ch1_split_search_summary.csv"
)

ASSIGNMENT_SUMMARY_FILE = Path(
    "data_manifests/model_ch1_split_assignment_summary.csv"
)


AUDIT_FILE = Path(
    "artifacts/splitting/manifests/model_ch1_split_audit_summary.csv"
)

YEAR_FILE = Path(
    "artifacts/splitting/manifests/model_ch1_split_year_balance.csv"
)

OVERLAP_FILE = Path(
    "artifacts/splitting/manifests/model_ch1_split_overlap_audit.csv"
)

MATERIALIZATION_FILE = Path(
    "artifacts/splitting/manifests/model_ch1_materialization_summary.csv"
)


TRAIN_FILE = Path(
    "artifacts/splitting/data/cycle24_train.parquet"
)

VALIDATION_FILE = Path(
    "artifacts/splitting/data/cycle24_validation.parquet"
)

HOLDOUT_FILE = Path(
    "artifacts/splitting/data/cycle24_holdout.parquet"
)


CONTRACT_DIR = Path(
    "artifacts/splitting/contracts"
)

MANIFEST_DIR = Path(
    "artifacts/splitting/manifests"
)


CONTRACT_FILE = (
    CONTRACT_DIR
    /
    "cycle24_split_contract.json"
)

FILES_MANIFEST = (
    MANIFEST_DIR
    /
    "model_ch1_split_files.csv"
)


# ============================================================
# FROZEN DESIGN
# ============================================================

SPLIT_VERSION = "1.0"

BASE_SEED = 24025
SELECTED_SEED = 26311
CANDIDATE_COUNT = 10000


TARGET_RATIOS = {
    "TRAIN": 0.70,
    "VALIDATION": 0.15,
    "HOLDOUT": 0.15,
}


EXPECTED = {

    "TRAIN": {
        "rows": 62064,
        "positive": 2420,
        "negative": 59644,
        "entity_groups": 822,
    },

    "VALIDATION": {
        "rows": 13299,
        "positive": 515,
        "negative": 12784,
        "entity_groups": 176,
    },

    "HOLDOUT": {
        "rows": 13072,
        "positive": 527,
        "negative": 12545,
        "entity_groups": 176,
    },
}


FILES = {
    "TRAIN": TRAIN_FILE,
    "VALIDATION": VALIDATION_FILE,
    "HOLDOUT": HOLDOUT_FILE,
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path):

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


def schema_fingerprint(df):

    schema = [
        {
            "column":
                str(column),

            "dtype":
                str(
                    df[column].dtype
                ),
        }

        for column in df.columns
    ]


    encoded = json.dumps(
        schema,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


    return hashlib.sha256(
        encoded
    ).hexdigest()


def sample_keys(df):

    return set(
        zip(
            pd.to_numeric(
                df["HARPNUM"],
                errors="raise",
            ).astype(int),

            df["T_REC"].astype(str),
        )
    )


# ============================================================
# INPUT CHECK
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 1.5 — FREEZE SPLIT CONTRACT"
)

print(
    "===================================="
)


required_files = [
    SOURCE_DATASET,
    DATASET_CONTRACT,
    PRE_MODEL_GATE,
    ASSIGNMENT_FILE,
    SEARCH_SUMMARY_FILE,
    ASSIGNMENT_SUMMARY_FILE,
    AUDIT_FILE,
    YEAR_FILE,
    OVERLAP_FILE,
    MATERIALIZATION_FILE,
    TRAIN_FILE,
    VALIDATION_FILE,
    HOLDOUT_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# VERIFY UPSTREAM CONTRACTS
# ============================================================

with DATASET_CONTRACT.open(
    "r",
    encoding="utf-8",
) as file:

    dataset_contract = json.load(
        file
    )


with PRE_MODEL_GATE.open(
    "r",
    encoding="utf-8",
) as file:

    pre_model_gate = json.load(
        file
    )


if dataset_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Dataset v2 is not frozen."
    )


if dataset_contract.get(
    "dataset_version"
) != "2.0":

    raise RuntimeError(
        "Unexpected dataset version."
    )


if pre_model_gate.get(
    "status"
) != "OPEN_FOR_MODELING":

    raise RuntimeError(
        "Pre-model gate is not open."
    )


print(
    "Dataset v2: FROZEN"
)

print(
    "Pre-model gate: OPEN"
)


# ============================================================
# VERIFY CHAPTER 1.3 AUDIT
# ============================================================

audit = pd.read_csv(
    AUDIT_FILE
)


if len(audit) != 3:

    raise RuntimeError(
        "Split audit should contain 3 rows."
    )


if not audit[
    "split_audit_pass"
].astype(bool).all():

    raise RuntimeError(
        "Split audit is not fully PASS."
    )


overlap = pd.read_csv(
    OVERLAP_FILE
)


if not overlap[
    "count"
].eq(0).all():

    raise RuntimeError(
        "Split overlap audit contains leakage."
    )


print(
    "Chapter 1.3 audit: PASS"
)

print(
    "Split leakage audit: PASS"
)


# ============================================================
# LOAD SOURCE + SPLITS
# ============================================================

source = pd.read_parquet(
    SOURCE_DATASET
)


split_frames = {

    split:
        pd.read_parquet(
            path
        )

    for split, path
    in FILES.items()
}


source_schema_hash = schema_fingerprint(
    source
)


# ============================================================
# VERIFY EACH SPLIT
# ============================================================

file_rows = []


for split in [
    "TRAIN",
    "VALIDATION",
    "HOLDOUT",
]:

    df = split_frames[
        split
    ]


    expected = EXPECTED[
        split
    ]


    rows = len(
        df
    )


    positive = int(
        (
            df["target"] == 1
        ).sum()
    )


    negative = int(
        (
            df["target"] == 0
        ).sum()
    )


    groups = int(
        df[
            "ENTITY_GROUP_ID"
        ].nunique()
    )


    if rows != expected[
        "rows"
    ]:

        raise RuntimeError(
            f"{split}: row mismatch."
        )


    if positive != expected[
        "positive"
    ]:

        raise RuntimeError(
            f"{split}: positive mismatch."
        )


    if negative != expected[
        "negative"
    ]:

        raise RuntimeError(
            f"{split}: negative mismatch."
        )


    if groups != expected[
        "entity_groups"
    ]:

        raise RuntimeError(
            f"{split}: entity-group mismatch."
        )


    if list(
        df.columns
    ) != list(
        source.columns
    ):

        raise RuntimeError(
            f"{split}: schema mismatch."
        )


    if schema_fingerprint(
        df
    ) != source_schema_hash:

        raise RuntimeError(
            f"{split}: schema fingerprint mismatch."
        )


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


    if duplicate_keys != 0:

        raise RuntimeError(
            f"{split}: duplicate sample keys."
        )


    digest = sha256_file(
        FILES[
            split
        ]
    )


    file_rows.append(
        {
            "split":
                split,

            "file":
                str(
                    FILES[
                        split
                    ]
                ),

            "rows":
                rows,

            "positive":
                positive,

            "negative":
                negative,

            "positive_rate_pct":
                positive
                /
                rows
                *
                100,

            "entity_groups":
                groups,

            "unique_harpnums":
                int(
                    df[
                        "HARPNUM"
                    ].nunique()
                ),

            "columns":
                len(
                    df.columns
                ),

            "schema_sha256":
                source_schema_hash,

            "file_sha256":
                digest,

            "file_size_bytes":
                FILES[
                    split
                ].stat().st_size,
        }
    )


# ============================================================
# EXACT PARTITION CHECK
# ============================================================

source_keys = sample_keys(
    source
)


train_keys = sample_keys(
    split_frames[
        "TRAIN"
    ]
)

validation_keys = sample_keys(
    split_frames[
        "VALIDATION"
    ]
)

holdout_keys = sample_keys(
    split_frames[
        "HOLDOUT"
    ]
)


if train_keys & validation_keys:

    raise RuntimeError(
        "TRAIN/VALIDATION sample overlap."
    )


if train_keys & holdout_keys:

    raise RuntimeError(
        "TRAIN/HOLDOUT sample overlap."
    )


if validation_keys & holdout_keys:

    raise RuntimeError(
        "VALIDATION/HOLDOUT sample overlap."
    )


combined_keys = (
    train_keys
    |
    validation_keys
    |
    holdout_keys
)


if combined_keys != source_keys:

    raise RuntimeError(
        "Splits do not exactly partition Cycle24."
    )


# ============================================================
# ENTITY-GROUP OVERLAP
# ============================================================

group_sets = {

    split:
        set(
            frame[
                "ENTITY_GROUP_ID"
            ].astype(str)
        )

    for split, frame
    in split_frames.items()
}


if group_sets[
    "TRAIN"
] & group_sets[
    "VALIDATION"
]:

    raise RuntimeError(
        "TRAIN/VALIDATION group leakage."
    )


if group_sets[
    "TRAIN"
] & group_sets[
    "HOLDOUT"
]:

    raise RuntimeError(
        "TRAIN/HOLDOUT group leakage."
    )


if group_sets[
    "VALIDATION"
] & group_sets[
    "HOLDOUT"
]:

    raise RuntimeError(
        "VALIDATION/HOLDOUT group leakage."
    )


# ============================================================
# HARP OVERLAP
# ============================================================

harp_sets = {

    split:
        set(
            pd.to_numeric(
                frame[
                    "HARPNUM"
                ],
                errors="raise",
            ).astype(int)
        )

    for split, frame
    in split_frames.items()
}


for left, right in [
    ("TRAIN", "VALIDATION"),
    ("TRAIN", "HOLDOUT"),
    ("VALIDATION", "HOLDOUT"),
]:

    if harp_sets[
        left
    ] & harp_sets[
        right
    ]:

        raise RuntimeError(
            f"{left}/{right} HARPNUM leakage."
        )


# ============================================================
# VERIFY ASSIGNMENT MANIFEST
# ============================================================

assignments = pd.read_csv(
    ASSIGNMENT_FILE
)


if len(
    assignments
) != 1174:

    raise RuntimeError(
        "Assignment manifest does not contain "
        "1174 entity groups."
    )


if assignments[
    "ENTITY_GROUP_ID"
].duplicated().any():

    raise RuntimeError(
        "Duplicate group assignments."
    )


if set(
    assignments[
        "split"
    ].unique()
) != {
    "TRAIN",
    "VALIDATION",
    "HOLDOUT",
}:

    raise RuntimeError(
        "Unexpected split names."
    )


# ============================================================
# VERIFY SELECTED SEED EXISTS IN SEARCH MANIFEST
# ============================================================

search_summary = pd.read_csv(
    SEARCH_SUMMARY_FILE
)


selected_seed_rows = search_summary[
    search_summary[
        "seed"
    ]
    ==
    SELECTED_SEED
]


if len(
    selected_seed_rows
) != 1:

    raise RuntimeError(
        "Selected seed is not uniquely present "
        "in search summary."
    )


selected_score = float(
    selected_seed_rows.iloc[
        0
    ][
        "score"
    ]
)


# ============================================================
# FILE MANIFEST
# ============================================================

files_df = pd.DataFrame(
    file_rows
)


MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


files_df.to_csv(
    FILES_MANIFEST,
    index=False,
)


# ============================================================
# PROVENANCE
# ============================================================

provenance_files = {

    "cycle24_dataset_v2":
        SOURCE_DATASET,

    "dataset_v2_contract":
        DATASET_CONTRACT,

    "pre_model_gate":
        PRE_MODEL_GATE,

    "split_assignments":
        ASSIGNMENT_FILE,

    "split_search_summary":
        SEARCH_SUMMARY_FILE,

    "split_assignment_summary":
        ASSIGNMENT_SUMMARY_FILE,

    "split_audit_summary":
        AUDIT_FILE,

    "split_year_balance":
        YEAR_FILE,

    "split_overlap_audit":
        OVERLAP_FILE,

    "materialization_summary":
        MATERIALIZATION_FILE,
}


provenance = {}


for name, path in provenance_files.items():

    provenance[
        name
    ] = {

        "file":
            str(path),

        "sha256":
            sha256_file(
                path
            ),
    }


# ============================================================
# CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Cycle24 Model Development Split",

    "split_version":
        SPLIT_VERSION,

    "status":
        "FROZEN",

    "frozen_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source_dataset": {

        "dataset_version":
            "2.0",

        "file":
            str(
                SOURCE_DATASET
            ),

        "rows":
            len(
                source
            ),

        "positive":
            int(
                source[
                    "target"
                ].sum()
            ),

        "negative":
            int(
                (
                    source[
                        "target"
                    ]
                    ==
                    0
                ).sum()
            ),

        "entity_groups":
            int(
                source[
                    "ENTITY_GROUP_ID"
                ].nunique()
            ),

        "file_sha256":
            sha256_file(
                SOURCE_DATASET
            ),

        "schema_sha256":
            source_schema_hash,
    },

    "split_design": {

        "unit":
            "ENTITY_GROUP_ID",

        "target_ratios":
            TARGET_RATIOS,

        "method":
            (
                "Deterministic candidate search over "
                "ENTITY_GROUP_ID-level assignments using "
                "Cycle24 row count, target count, "
                "positive-group count, year-wise rows, "
                "and year-wise positive balance."
            ),

        "base_seed":
            BASE_SEED,

        "candidate_count":
            CANDIDATE_COUNT,

        "selected_seed":
            SELECTED_SEED,

        "selected_score":
            selected_score,

        "cycle25_used":
            False,

        "feature_values_used":
            False,

        "model_performance_used":
            False,
    },

    "train": {

        "file":
            str(
                TRAIN_FILE
            ),

        "rows":
            EXPECTED[
                "TRAIN"
            ][
                "rows"
            ],

        "positive":
            EXPECTED[
                "TRAIN"
            ][
                "positive"
            ],

        "negative":
            EXPECTED[
                "TRAIN"
            ][
                "negative"
            ],

        "entity_groups":
            EXPECTED[
                "TRAIN"
            ][
                "entity_groups"
            ],

        "sha256":
            sha256_file(
                TRAIN_FILE
            ),
    },

    "validation": {

        "file":
            str(
                VALIDATION_FILE
            ),

        "rows":
            EXPECTED[
                "VALIDATION"
            ][
                "rows"
            ],

        "positive":
            EXPECTED[
                "VALIDATION"
            ][
                "positive"
            ],

        "negative":
            EXPECTED[
                "VALIDATION"
            ][
                "negative"
            ],

        "entity_groups":
            EXPECTED[
                "VALIDATION"
            ][
                "entity_groups"
            ],

        "sha256":
            sha256_file(
                VALIDATION_FILE
            ),
    },

    "holdout": {

        "file":
            str(
                HOLDOUT_FILE
            ),

        "rows":
            EXPECTED[
                "HOLDOUT"
            ][
                "rows"
            ],

        "positive":
            EXPECTED[
                "HOLDOUT"
            ][
                "positive"
            ],

        "negative":
            EXPECTED[
                "HOLDOUT"
            ][
                "negative"
            ],

        "entity_groups":
            EXPECTED[
                "HOLDOUT"
            ][
                "entity_groups"
            ],

        "sha256":
            sha256_file(
                HOLDOUT_FILE
            ),
    },

    "integrity": {

        "entity_group_overlap":
            0,

        "harp_overlap":
            0,

        "sample_key_overlap":
            0,

        "exact_source_partition":
            True,

        "chapter_1_3_audit_pass":
            True,

        "chapter_1_4_materialization_pass":
            True,
    },

    "usage_rules": {

        "train":
            (
                "May be used to fit preprocessing "
                "and model parameters."
            ),

        "validation":
            (
                "May be used for hyperparameter selection, "
                "model selection, and decision-threshold "
                "selection."
            ),

        "holdout":
            (
                "Must remain untouched until the complete "
                "Cycle24 development pipeline is frozen."
            ),

        "cycle25":
            (
                "Must remain untouched for model development "
                "and may only be used after the full model "
                "pipeline is frozen."
            ),
    },

    "immutability_rule":
        (
            "This split is frozen. Train, validation, and "
            "holdout memberships must not be modified in "
            "response to model performance. Any scientifically "
            "necessary split change requires a new split "
            "version and explicit justification."
        ),

    "provenance":
        provenance,
}


# ============================================================
# SAFE CONTRACT WRITE
# ============================================================

CONTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if CONTRACT_FILE.exists():

    with CONTRACT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        existing = json.load(
            file
        )


    expected_hashes = {
        "train":
            contract[
                "train"
            ][
                "sha256"
            ],

        "validation":
            contract[
                "validation"
            ][
                "sha256"
            ],

        "holdout":
            contract[
                "holdout"
            ][
                "sha256"
            ],
    }


    existing_hashes = {

        "train":
            existing.get(
                "train",
                {}
            ).get(
                "sha256"
            ),

        "validation":
            existing.get(
                "validation",
                {}
            ).get(
                "sha256"
            ),

        "holdout":
            existing.get(
                "holdout",
                {}
            ).get(
                "sha256"
            ),
    }


    if (
        existing_hashes
        !=
        expected_hashes
    ):

        raise RuntimeError(
            "Existing split contract belongs to "
            "different split files. "
            "Frozen split will NOT be overwritten."
        )


    print(
        "Existing split contract matches "
        "current frozen hashes."
    )


else:

    with CONTRACT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            contract,
            file,
            indent=2,
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "FROZEN SPLIT SUMMARY"
)

print(
    "===================================="
)


print(
    files_df[
        [
            "split",
            "rows",
            "positive",
            "negative",
            "positive_rate_pct",
            "entity_groups",
            "file_sha256",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n===================================="
)

print(
    "SPLIT CONTRACT"
)

print(
    "===================================="
)


print(
    "Split version:",
    SPLIT_VERSION
)

print(
    "Status: FROZEN"
)

print(
    "Base seed:",
    BASE_SEED
)

print(
    "Selected seed:",
    SELECTED_SEED
)

print(
    "Candidate count:",
    CANDIDATE_COUNT
)

print(
    "Selected score:",
    selected_score
)

print(
    "Cycle25 used:",
    False
)

print(
    "Feature values used:",
    False
)

print(
    "Model performance used:",
    False
)


print(
    "\n===================================="
)

print(
    "LEAKAGE / PARTITION"
)

print(
    "===================================="
)


print(
    "ENTITY_GROUP overlap: 0"
)

print(
    "HARPNUM overlap: 0"
)

print(
    "Sample-key overlap: 0"
)

print(
    "Exact Cycle24 partition: PASS"
)


print(
    "\nSaved:"
)

print(
    FILES_MANIFEST
)

print(
    CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 1.5 COMPLETE — SPLIT FROZEN"
)

print(
    "===================================="
)