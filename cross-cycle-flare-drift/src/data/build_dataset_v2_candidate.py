from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATASET_V1_DIR = Path(
    "data/final/dataset_v1"
)

GOES_DIR = Path(
    "data/raw/goes"
)

OUTPUT_DIR = Path(
    "data/processed/dataset_v2_candidate"
)

MANIFEST_DIR = Path(
    "data_manifests"
)


C24_INPUT = (
    DATASET_V1_DIR
    /
    "cycle24_dataset_v1.parquet"
)

C25_INPUT = (
    DATASET_V1_DIR
    /
    "cycle25_dataset_v1.parquet"
)


SRS_AUDIT_FILE = (
    MANIFEST_DIR
    /
    "pre_model_srs_all_positive_row_audit.csv"
)


C24_OUTPUT = (
    OUTPUT_DIR
    /
    "cycle24_dataset_v2_candidate.parquet"
)

C25_OUTPUT = (
    OUTPUT_DIR
    /
    "cycle25_dataset_v2_candidate.parquet"
)


CORRECTION_FILE = (
    MANIFEST_DIR
    /
    "pre_model_dataset_v2_label_corrections.csv"
)

SUMMARY_FILE = (
    MANIFEST_DIR
    /
    "pre_model_dataset_v2_candidate_summary.csv"
)


FORECAST_HOURS = 24


# ============================================================
# HELPERS
# ============================================================

def flare_class_to_value(value):

    if pd.isna(value):
        return np.nan

    text = str(
        value
    ).strip().upper()

    match = re.match(
        r"^([ABCMX])\s*([0-9.]+)",
        text,
    )

    if not match:
        return np.nan

    letter = match.group(1)

    try:
        magnitude = float(
            match.group(2)
        )

    except ValueError:
        return np.nan


    scales = {
        "A": 1e-8,
        "B": 1e-7,
        "C": 1e-6,
        "M": 1e-5,
        "X": 1e-4,
    }


    return (
        scales[letter]
        *
        magnitude
    )


def normalize_goes_region(value):

    if pd.isna(value):
        return None

    try:
        region = int(
            float(value)
        )

    except Exception:
        return None


    if region <= 0:
        return None


    if region < 10000:
        return region + 10000


    return region


# ============================================================
# LOAD GOES
# ============================================================

print(
    "===================================="
)

print(
    "PRE-MODEL FIX I — DATASET v2 LABEL REBUILD"
)

print(
    "===================================="
)


goes_frames = []


for year in range(
    2010,
    2026,
):

    path = (
        GOES_DIR
        /
        f"goes_flare_{year}.csv"
    )


    if not path.exists():

        raise FileNotFoundError(
            f"Missing GOES catalog: {path}"
        )


    df = pd.read_csv(
        path
    )


    required = {
        "start_time",
        "flare_class",
        "active_region",
    }


    missing = (
        required
        -
        set(df.columns)
    )


    if missing:

        raise RuntimeError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )


    goes_frames.append(
        df
    )


goes = pd.concat(
    goes_frames,
    ignore_index=True,
)


goes["_start_time"] = pd.to_datetime(
    goes["start_time"],
    utc=True,
    errors="coerce",
)


goes["_class_value"] = (
    goes["flare_class"]
    .apply(
        flare_class_to_value
    )
)


goes["_region"] = (
    goes["active_region"]
    .apply(
        normalize_goes_region
    )
)


M1_THRESHOLD = flare_class_to_value(
    "M1.0"
)


m1plus = goes[
    goes["_start_time"].notna()
    &
    (
        goes["_class_value"]
        >=
        M1_THRESHOLD
    )
].copy()


unassigned = m1plus[
    m1plus["_region"].isna()
].copy()


unassigned = unassigned.sort_values(
    "_start_time"
).reset_index(
    drop=True
)


# IMPORTANT:
# Use Timestamp.value explicitly so all comparisons
# use nanoseconds.
unassigned_times = np.array(
    [
        timestamp.value

        for timestamp
        in unassigned["_start_time"]
    ],
    dtype=np.int64,
)


print(
    "GOES M1+ events:",
    len(m1plus)
)


print(
    "Region-unassigned M1+ events:",
    len(unassigned)
)


# ============================================================
# LOAD SRS POSITIVE AUDIT
# ============================================================

if not SRS_AUDIT_FILE.exists():

    raise FileNotFoundError(
        f"Missing Fix H output: "
        f"{SRS_AUDIT_FILE}"
    )


srs_audit = pd.read_csv(
    SRS_AUDIT_FILE
)


required_srs = {
    "cycle",
    "HARPNUM",
    "T_REC",
    "row_decision",
}


missing = (
    required_srs
    -
    set(srs_audit.columns)
)


if missing:

    raise RuntimeError(
        "Fix H audit missing columns: "
        f"{sorted(missing)}"
    )


if len(srs_audit) != 8836:

    raise RuntimeError(
        "Fix H audit should contain exactly "
        "8836 Dataset v1 positive rows."
    )


duplicate_srs = int(
    srs_audit[
        [
            "cycle",
            "HARPNUM",
            "T_REC",
        ]
    ]
    .duplicated()
    .sum()
)


if duplicate_srs != 0:

    raise RuntimeError(
        "Duplicate rows found in Fix H audit."
    )


allowed_decisions = {
    "POSITIVE_SUPPORTED",
    "CENSOR_TEMPORAL_AMBIGUITY",
    "STALE_ALL_MATCHED_ASSOCIATIONS",
    "CENSOR_SOURCE_MISSING",
}


unexpected = set(
    srs_audit["row_decision"]
    .dropna()
    .unique()
) - allowed_decisions


if unexpected:

    raise RuntimeError(
        "Unexpected SRS decisions: "
        f"{sorted(unexpected)}"
    )


# ============================================================
# UNASSIGNED FLARE WINDOW CHECK
# ============================================================

def has_unassigned_m1plus(
    observation_time
):

    if pd.isna(observation_time):
        return False


    observation_time = pd.Timestamp(
        observation_time
    )


    start_ns = observation_time.value


    end_ns = (
        observation_time
        +
        pd.Timedelta(
            hours=FORECAST_HOURS
        )
    ).value


    left = np.searchsorted(
        unassigned_times,
        start_ns,
        side="right",
    )


    right = np.searchsorted(
        unassigned_times,
        end_ns,
        side="right",
    )


    return right > left


# ============================================================
# BUILD EACH CYCLE
# ============================================================

datasets = {
    "cycle24": (
        C24_INPUT,
        C24_OUTPUT,
    ),

    "cycle25": (
        C25_INPUT,
        C25_OUTPUT,
    ),
}


summary_rows = []
correction_rows = []


for cycle, (
    input_path,
    output_path,
) in datasets.items():

    print(
        "\n===================================="
    )

    print(
        cycle.upper()
    )

    print(
        "===================================="
    )


    if not input_path.exists():

        raise FileNotFoundError(
            f"Missing Dataset v1: "
            f"{input_path}"
        )


    df = pd.read_parquet(
        input_path
    )


    original_columns = list(
        df.columns
    )


    original_rows = len(
        df
    )


    original_positive = int(
        (
            df["target"] == 1
        ).sum()
    )


    original_negative = int(
        (
            df["target"] == 0
        ).sum()
    )


    positive_mask = (
        df["target"] == 1
    )


    positives = df[
        positive_mask
    ].copy()


    negatives = df[
        ~positive_mask
    ].copy()


    cycle_audit = srs_audit[
        srs_audit["cycle"]
        ==
        cycle
    ].copy()


    if len(cycle_audit) != len(
        positives
    ):

        raise RuntimeError(
            f"{cycle}: SRS positive audit count "
            "does not match Dataset v1 positives."
        )


    cycle_audit["HARPNUM"] = pd.to_numeric(
        cycle_audit["HARPNUM"],
        errors="coerce",
    ).astype(
        "Int64"
    )


    positives["HARPNUM"] = pd.to_numeric(
        positives["HARPNUM"],
        errors="coerce",
    ).astype(
        "Int64"
    )


    # ========================================================
    # MERGE SRS DECISION ON POSITIVE ROWS
    # ========================================================

    positives = positives.merge(
        cycle_audit[
            [
                "HARPNUM",
                "T_REC",
                "row_decision",
            ]
        ],
        on=[
            "HARPNUM",
            "T_REC",
        ],
        how="left",
        validate="one_to_one",
    )


    missing_decision = int(
        positives[
            "row_decision"
        ]
        .isna()
        .sum()
    )


    if missing_decision != 0:

        raise RuntimeError(
            f"{cycle}: {missing_decision} "
            "positive rows have no SRS decision."
        )


    retained_positive_rows = []
    relabeled_negative_rows = []
    censored_rows = []


    # ========================================================
    # APPLY CORRECTION POLICY
    # ========================================================

    for _, row in positives.iterrows():

        decision = row[
            "row_decision"
        ]


        observation_time = pd.to_datetime(
            row[
                "observation_time_utc"
            ],
            utc=True,
            errors="coerce",
        )


        correction = None
        unassigned_in_window = False


        if decision == "POSITIVE_SUPPORTED":

            retained_positive_rows.append(
                row
            )

            correction = (
                "KEEP_POSITIVE_SRS_SUPPORTED"
            )


        elif decision == (
            "CENSOR_TEMPORAL_AMBIGUITY"
        ):

            censored_rows.append(
                row
            )

            correction = (
                "CENSOR_TEMPORAL_AMBIGUITY"
            )


        elif decision == (
            "CENSOR_SOURCE_MISSING"
        ):

            censored_rows.append(
                row
            )

            correction = (
                "CENSOR_SRS_SOURCE_MISSING"
            )


        elif decision == (
            "STALE_ALL_MATCHED_ASSOCIATIONS"
        ):

            unassigned_in_window = (
                has_unassigned_m1plus(
                    observation_time
                )
            )


            if unassigned_in_window:

                censored_rows.append(
                    row
                )

                correction = (
                    "CENSOR_UNASSIGNED_M1_AFTER_"
                    "STALE_ASSOCIATION"
                )


            else:

                corrected = row.copy()

                corrected[
                    "target"
                ] = 0

                corrected[
                    "label_status"
                ] = "NEGATIVE"

                relabeled_negative_rows.append(
                    corrected
                )

                correction = (
                    "RELABEL_NEGATIVE_"
                    "STALE_ASSOCIATION"
                )


        else:

            raise RuntimeError(
                f"Unexpected decision: "
                f"{decision}"
            )


        if correction != (
            "KEEP_POSITIVE_SRS_SUPPORTED"
        ):

            correction_rows.append(
                {
                    "cycle":
                        cycle,

                    "HARPNUM":
                        int(
                            row[
                                "HARPNUM"
                            ]
                        ),

                    "T_REC":
                        row[
                            "T_REC"
                        ],

                    "ENTITY_GROUP_ID":
                        row[
                            "ENTITY_GROUP_ID"
                        ],

                    "observation_time_utc":
                        row[
                            "observation_time_utc"
                        ],

                    "original_target":
                        1,

                    "srs_decision":
                        decision,

                    "unassigned_m1plus_in_window":
                        unassigned_in_window,

                    "dataset_v2_action":
                        correction,
                }
            )


    # ========================================================
    # CONSTRUCT CANDIDATE
    # ========================================================

    if retained_positive_rows:

        positive_keep = pd.DataFrame(
            retained_positive_rows
        )

    else:

        positive_keep = positives.iloc[
            0:0
        ].copy()


    if relabeled_negative_rows:

        relabeled = pd.DataFrame(
            relabeled_negative_rows
        )

    else:

        relabeled = positives.iloc[
            0:0
        ].copy()


    # Remove temporary audit column.
    for frame in [
        positive_keep,
        relabeled,
    ]:

        if "row_decision" in frame.columns:

            frame.drop(
                columns=[
                    "row_decision"
                ],
                inplace=True,
            )


    candidate = pd.concat(
        [
            negatives,
            positive_keep,
            relabeled,
        ],
        ignore_index=True,
    )


    # ========================================================
    # RESTORE ORIGINAL SCHEMA / ORDER
    # ========================================================

    extra_columns = (
        set(candidate.columns)
        -
        set(original_columns)
    )


    if extra_columns:

        raise RuntimeError(
            f"{cycle}: unexpected columns: "
            f"{sorted(extra_columns)}"
        )


    candidate = candidate[
        original_columns
    ]


    # ========================================================
    # SORT DETERMINISTICALLY
    # ========================================================

    candidate = candidate.sort_values(
        [
            "T_REC",
            "HARPNUM",
        ]
    ).reset_index(
        drop=True
    )


    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    invalid_targets = int(
        (
            ~candidate[
                "target"
            ].isin(
                [
                    0,
                    1,
                ]
            )
        )
        .sum()
    )


    duplicates = int(
        candidate[
            [
                "HARPNUM",
                "T_REC",
            ]
        ]
        .duplicated()
        .sum()
    )


    if invalid_targets != 0:

        raise RuntimeError(
            f"{cycle}: invalid targets."
        )


    if duplicates != 0:

        raise RuntimeError(
            f"{cycle}: duplicate sample keys."
        )


    final_rows = len(
        candidate
    )


    final_positive = int(
        (
            candidate[
                "target"
            ]
            ==
            1
        )
        .sum()
    )


    final_negative = int(
        (
            candidate[
                "target"
            ]
            ==
            0
        )
        .sum()
    )


    relabeled_count = len(
        relabeled_negative_rows
    )


    censored_count = len(
        censored_rows
    )


    expected_rows = (
        original_rows
        -
        censored_count
    )


    if final_rows != expected_rows:

        raise RuntimeError(
            f"{cycle}: candidate row accounting failed."
        )


    if (
        final_positive
        +
        final_negative
        !=
        final_rows
    ):

        raise RuntimeError(
            f"{cycle}: class accounting failed."
        )


    # ========================================================
    # SAVE CANDIDATE
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    candidate.to_parquet(
        output_path,
        index=False,
    )


    summary_rows.append(
        {
            "cycle":
                cycle,

            "dataset_v1_rows":
                original_rows,

            "dataset_v1_positive":
                original_positive,

            "dataset_v1_negative":
                original_negative,

            "srs_supported_positive":
                len(
                    retained_positive_rows
                ),

            "stale_relabeled_negative":
                relabeled_count,

            "censored_removed":
                censored_count,

            "dataset_v2_candidate_rows":
                final_rows,

            "dataset_v2_candidate_positive":
                final_positive,

            "dataset_v2_candidate_negative":
                final_negative,

            "dataset_v2_positive_rate_pct":
                (
                    final_positive
                    /
                    final_rows
                    *
                    100
                ),
        }
    )


    print(
        "Dataset v1 rows:",
        original_rows
    )


    print(
        "Original positives:",
        original_positive
    )


    print(
        "SRS-supported positives:",
        len(
            retained_positive_rows
        )
    )


    print(
        "Stale positives relabeled negative:",
        relabeled_count
    )


    print(
        "Rows censored/removed:",
        censored_count
    )


    print(
        "Dataset v2 candidate rows:",
        final_rows
    )


    print(
        "Candidate positives:",
        final_positive
    )


    print(
        "Candidate negatives:",
        final_negative
    )


# ============================================================
# SAVE MANIFESTS
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


corrections = pd.DataFrame(
    correction_rows
)


MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


corrections.to_csv(
    CORRECTION_FILE,
    index=False,
)


# ============================================================
# CROSS-CYCLE LEAKAGE CHECK
# ============================================================

c24 = pd.read_parquet(
    C24_OUTPUT
)

c25 = pd.read_parquet(
    C25_OUTPUT
)


group_overlap = (
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
            c24[
                "HARPNUM"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    &
    set(
        pd.to_numeric(
            c25[
                "HARPNUM"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
)


if group_overlap:

    raise RuntimeError(
        "Cross-cycle ENTITY_GROUP_ID overlap "
        "introduced."
    )


if harp_overlap:

    raise RuntimeError(
        "Cross-cycle HARPNUM overlap introduced."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n===================================="
)

print(
    "DATASET v2 CANDIDATE SUMMARY"
)

print(
    "===================================="
)


print(
    summary.to_string(
        index=False
    )
)


print(
    "\n===================================="
)

print(
    "CORRECTION ACTION COUNTS"
)

print(
    "===================================="
)


if len(corrections) > 0:

    action_counts = (
        corrections[
            "dataset_v2_action"
        ]
        .value_counts()
    )


    print(
        action_counts.to_string()
    )


else:

    print(
        "No corrections."
    )


print(
    "\nCross-cycle ENTITY_GROUP_ID overlap:",
    len(
        group_overlap
    )
)


print(
    "Cross-cycle HARPNUM overlap:",
    len(
        harp_overlap
    )
)


print(
    "\nSaved:"
)

print(
    C24_OUTPUT
)

print(
    C25_OUTPUT
)

print(
    CORRECTION_FILE
)

print(
    SUMMARY_FILE
)


print(
    "\nPre-model Fix I complete."
)