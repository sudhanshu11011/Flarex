from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATASET_CONTRACT_FILE = Path(
    "data_manifests/dataset_v2_contract.json"
)

SPLIT_CONTRACT_FILE = Path(
    "artifacts/splitting/contracts/cycle24_split_contract.json"
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


MANIFEST_DIR = Path(
    "artifacts/preprocessing/manifests"
)

CONTRACT_DIR = Path(
    "artifacts/preprocessing/contracts"
)


AUDIT_OUTPUT = (
    MANIFEST_DIR
    /
    "model_ch2_feature_contract_audit.csv"
)

CONTRACT_OUTPUT = (
    CONTRACT_DIR
    /
    "model_feature_contract.json"
)


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


# ============================================================
# VERIFY INPUTS
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 2.1 — FEATURE CONTRACT VERIFICATION"
)

print(
    "===================================="
)


required_files = [
    DATASET_CONTRACT_FILE,
    SPLIT_CONTRACT_FILE,
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
# LOAD CONTRACTS
# ============================================================

with DATASET_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    dataset_contract = json.load(
        file
    )


with SPLIT_CONTRACT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    split_contract = json.load(
        file
    )


if dataset_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Dataset v2 contract is not frozen."
    )


if split_contract.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Cycle24 split contract is not frozen."
    )


print(
    "Dataset v2 contract: FROZEN"
)

print(
    "Cycle24 split contract: FROZEN"
)


# ============================================================
# READ FROZEN FEATURE CONTRACT
# ============================================================

feature_contract = dataset_contract.get(
    "feature_contract",
    {}
)


sharp_features = feature_contract.get(
    "sharp_features",
    []
)


lorentz_features = feature_contract.get(
    "lorentz_features",
    []
)


features = (
    sharp_features
    +
    lorentz_features
)


expected_feature_count = feature_contract.get(
    "total_model_features"
)


if expected_feature_count != 25:

    raise RuntimeError(
        "Frozen Dataset v2 contract does not "
        "specify exactly 25 model features."
    )


if len(sharp_features) != 18:

    raise RuntimeError(
        "Expected exactly 18 SHARP features."
    )


if len(lorentz_features) != 7:

    raise RuntimeError(
        "Expected exactly 7 Lorentz features."
    )


if len(features) != 25:

    raise RuntimeError(
        "Combined model feature count is not 25."
    )


if len(set(features)) != 25:

    raise RuntimeError(
        "Duplicate model features detected."
    )


# ============================================================
# FORBIDDEN MODEL INPUTS
# ============================================================

forbidden_inputs = {
    "target",
    "label_status",
    "HARPNUM",
    "ENTITY_GROUP_ID",
    "T_REC",
    "observation_time_utc",
    "NOAA_AR",
    "NOAA_ARS",
    "cycle",
    "QUALITY",
    "CMASK",
    "LON_FWT",
    "LAT_FWT",
}


forbidden_feature_overlap = (
    set(features)
    &
    forbidden_inputs
)


if forbidden_feature_overlap:

    raise RuntimeError(
        "Forbidden metadata leaked into feature contract: "
        f"{sorted(forbidden_feature_overlap)}"
    )


# ============================================================
# LOAD SPLITS
# ============================================================

datasets = {

    "TRAIN":
        pd.read_parquet(
            TRAIN_FILE
        ),

    "VALIDATION":
        pd.read_parquet(
            VALIDATION_FILE
        ),

    "HOLDOUT":
        pd.read_parquet(
            HOLDOUT_FILE
        ),
}


# ============================================================
# EXPECTED SPLIT HASH CHECK
# ============================================================

expected_hashes = {

    "TRAIN":
        split_contract[
            "train"
        ][
            "sha256"
        ],

    "VALIDATION":
        split_contract[
            "validation"
        ][
            "sha256"
        ],

    "HOLDOUT":
        split_contract[
            "holdout"
        ][
            "sha256"
        ],
}


actual_hashes = {

    "TRAIN":
        sha256_file(
            TRAIN_FILE
        ),

    "VALIDATION":
        sha256_file(
            VALIDATION_FILE
        ),

    "HOLDOUT":
        sha256_file(
            HOLDOUT_FILE
        ),
}


for split in datasets:

    if (
        actual_hashes[split]
        !=
        expected_hashes[split]
    ):

        raise RuntimeError(
            f"{split}: split file hash differs "
            "from frozen Chapter 1 contract."
        )


print(
    "Frozen split hash verification: PASS"
)


# ============================================================
# FEATURE AUDIT
# ============================================================

audit_rows = []


reference_dtypes = None


for split, df in datasets.items():

    print(
        "\n===================================="
    )

    print(
        split
    )

    print(
        "===================================="
    )


    missing_features = [
        feature

        for feature in features

        if feature not in df.columns
    ]


    if missing_features:

        raise RuntimeError(
            f"{split}: missing features: "
            f"{missing_features}"
        )


    # --------------------------------------------------------
    # Exact feature frame in frozen order
    # --------------------------------------------------------

    X = df[
        features
    ].copy()


    # --------------------------------------------------------
    # Numeric conversion test
    # --------------------------------------------------------

    numeric_X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )


    missing_value_count = int(
        numeric_X
        .isna()
        .sum()
        .sum()
    )


    array = numeric_X.to_numpy(
        dtype=float
    )


    nonfinite_count = int(
        (
            ~np.isfinite(
                array
            )
        )
        .sum()
    )


    nonfinite_rows = int(
        (
            ~np.isfinite(
                array
            )
            .all(
                axis=1
            )
        )
        .sum()
    )


    constant_features = [

        feature

        for feature
        in features

        if numeric_X[
            feature
        ].nunique(
            dropna=False
        )
        <=
        1
    ]


    feature_dtypes = {
        feature:
            str(
                X[
                    feature
                ].dtype
            )

        for feature
        in features
    }


    if reference_dtypes is None:

        reference_dtypes = (
            feature_dtypes
        )

    elif (
        feature_dtypes
        !=
        reference_dtypes
    ):

        raise RuntimeError(
            f"{split}: feature dtypes differ "
            "from TRAIN."
        )


    # --------------------------------------------------------
    # Target separation
    # --------------------------------------------------------

    target_present = (
        "target"
        in
        df.columns
    )


    target_in_features = (
        "target"
        in
        features
    )


    # --------------------------------------------------------
    # Contract pass
    # --------------------------------------------------------

    split_pass = (

        len(
            missing_features
        )
        ==
        0

        and
        missing_value_count
        ==
        0

        and
        nonfinite_count
        ==
        0

        and
        nonfinite_rows
        ==
        0

        and
        len(
            constant_features
        )
        ==
        0

        and
        target_present

        and
        not target_in_features
    )


    audit_rows.append(
        {
            "split":
                split,

            "rows":
                len(
                    df
                ),

            "feature_count":
                len(
                    features
                ),

            "missing_features":
                len(
                    missing_features
                ),

            "missing_values":
                missing_value_count,

            "nonfinite_values":
                nonfinite_count,

            "nonfinite_rows":
                nonfinite_rows,

            "constant_features":
                len(
                    constant_features
                ),

            "target_present":
                target_present,

            "target_in_feature_list":
                target_in_features,

            "feature_contract_pass":
                split_pass,
        }
    )


    print(
        "Rows:",
        len(
            df
        )
    )


    print(
        "Feature count:",
        len(
            features
        )
    )


    print(
        "Missing features:",
        len(
            missing_features
        )
    )


    print(
        "Missing values:",
        missing_value_count
    )


    print(
        "Non-finite values:",
        nonfinite_count
    )


    print(
        "Constant features:",
        len(
            constant_features
        )
    )


    print(
        "Feature contract:",
        (
            "PASS"
            if split_pass
            else
            "FAIL"
        )
    )


# ============================================================
# CROSS-SPLIT FEATURE ORDER
# ============================================================

train_columns = list(
    datasets[
        "TRAIN"
    ][
        features
    ].columns
)


validation_columns = list(
    datasets[
        "VALIDATION"
    ][
        features
    ].columns
)


holdout_columns = list(
    datasets[
        "HOLDOUT"
    ][
        features
    ].columns
)


feature_order_identical = (
    train_columns
    ==
    validation_columns
    ==
    holdout_columns
    ==
    features
)


if not feature_order_identical:

    raise RuntimeError(
        "Feature order mismatch across splits."
    )


# ============================================================
# SAVE AUDIT
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


overall_pass = bool(
    audit_df[
        "feature_contract_pass"
    ].all()
    and
    feature_order_identical
    and
    len(
        forbidden_feature_overlap
    )
    ==
    0
)


MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CONTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


audit_df.to_csv(
    AUDIT_OUTPUT,
    index=False,
)


# ============================================================
# CREATE MODEL FEATURE CONTRACT
# ============================================================

contract = {

    "contract_name":
        "Frozen Model Feature Contract",

    "version":
        "1.0",

    "status":
        (
            "VERIFIED"
            if overall_pass
            else
            "FAILED"
        ),

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source_dataset_version":
        "2.0",

    "source_split_version":
        split_contract.get(
            "split_version"
        ),

    "total_features":
        25,

    "sharp_feature_count":
        18,

    "lorentz_feature_count":
        7,

    "feature_order":
        features,

    "sharp_features":
        sharp_features,

    "lorentz_features":
        lorentz_features,

    "feature_dtypes":
        reference_dtypes,

    "forbidden_model_inputs":
        sorted(
            forbidden_inputs
        ),

    "target_column":
        "target",

    "rules": {

        "feature_order_fixed":
            True,

        "target_never_model_input":
            True,

        "metadata_never_model_input":
            True,

        "no_feature_selection_using_cycle25":
            True,

        "no_preprocessing_fit_using_cycle25":
            True,

        "no_preprocessing_fit_using_validation":
            True,

        "no_preprocessing_fit_using_holdout":
            True,
    },

    "split_hashes": {

        "train":
            actual_hashes[
                "TRAIN"
            ],

        "validation":
            actual_hashes[
                "VALIDATION"
            ],

        "holdout":
            actual_hashes[
                "HOLDOUT"
            ],
    },

    "dataset_v2_contract_sha256":
        sha256_file(
            DATASET_CONTRACT_FILE
        ),

    "cycle24_split_contract_sha256":
        sha256_file(
            SPLIT_CONTRACT_FILE
        ),
}


with CONTRACT_OUTPUT.open(
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
    "FEATURE CONTRACT"
)

print(
    "===================================="
)


print(
    "SHARP features:",
    len(
        sharp_features
    )
)

print(
    "Lorentz features:",
    len(
        lorentz_features
    )
)

print(
    "Total model features:",
    len(
        features
    )
)

print(
    "Forbidden metadata in feature list:",
    len(
        forbidden_feature_overlap
    )
)

print(
    "Feature order identical:",
    feature_order_identical
)


print(
    "\nFeature order:"
)


for index, feature in enumerate(
    features,
    start=1,
):

    print(
        f"{index:02d}. {feature}"
    )


print(
    "\n===================================="
)

print(
    "FEATURE CONTRACT AUDIT SUMMARY"
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
    "\nOverall feature contract:",
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
    CONTRACT_OUTPUT
)


if not overall_pass:

    raise RuntimeError(
        "Feature contract verification failed."
    )


print(
    "\nMODEL CHAPTER 2.1 COMPLETE."
)