from pathlib import Path
from datetime import timezone

import pandas as pd
from astropy.time import Time


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data_manifests/cycle24_label_rule_audit.csv"
)

OUTPUT_FILE = Path(
    "data_manifests/cycle24_label_horizon_boundary_audit.csv"
)

SUMMARY_FILE = Path(
    "data_manifests/cycle24_label_horizon_boundary_summary.csv"
)

PROBE_FILE = Path(
    "data_manifests/cycle24_label_horizon_boundary_probes.csv"
)


# ============================================================
# TEMPORAL CONTRACT
#
# Cycle 24 observations belong to the pre-2020 domain.
#
# IMPORTANT:
# Boundary is defined in UTC after converting SHARP T_REC
# from TAI.
#
# We deliberately DO NOT allow a Cycle-24 training label
# to depend on a flare beginning in Cycle 25.
# ============================================================

C24_END_EXCLUSIVE_UTC = pd.Timestamp(
    "2020-01-01T00:00:00Z"
)

FORECAST_HORIZON = pd.Timedelta(
    hours=24
)


# ============================================================
# HELPER:
# SHARP T_REC (TAI) -> UTC
# ============================================================

def sharp_trec_to_utc(t_rec):

    """
    Example:

    2011.03.15_12:00:00_TAI

    becomes approximately:

    2011-03-15 11:59:26 UTC

    Astropy handles the TAI-UTC offset.
    """

    text = str(
        t_rec
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


    tai_time = Time(
        text,
        format="isot",
        scale="tai"
    )


    utc_datetime = (
        tai_time
        .utc
        .to_datetime(
            timezone=timezone.utc
        )
    )


    return pd.Timestamp(
        utc_datetime
    )


# ============================================================
# LOAD PREVIOUS QC-8C AUDIT
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Missing QC-8C audit file: {INPUT_FILE}"
    )


audit = pd.read_csv(
    INPUT_FILE
)


required_columns = {
    "T_REC",
    "HARPNUM",
    "label_status",
}


missing_columns = (
    required_columns
    -
    set(audit.columns)
)


if missing_columns:

    raise RuntimeError(
        "QC-8C file missing required columns: "
        f"{sorted(missing_columns)}"
    )


print(
    "QC-8C rows loaded:",
    len(audit)
)


# ============================================================
# CONVERT ALL T_REC VALUES
#
# Recalculate from T_REC instead of trusting previously saved
# sample_time_utc so even EXCLUDE_NOAA rows get audited.
# ============================================================

converted_times = []

conversion_errors = []


for index, row in audit.iterrows():

    try:

        converted = sharp_trec_to_utc(
            row["T_REC"]
        )


        converted_times.append(
            converted
        )


    except Exception as error:

        converted_times.append(
            pd.NaT
        )


        conversion_errors.append(
            {
                "row_index":
                    index,

                "T_REC":
                    row["T_REC"],

                "error":
                    str(error),
            }
        )


audit[
    "sample_time_utc_recomputed"
] = converted_times


print(
    "TAI -> UTC conversion errors:",
    len(conversion_errors)
)


# ============================================================
# COMPUTE FORECAST HORIZON
# ============================================================

audit[
    "window_end_utc"
] = (
    audit[
        "sample_time_utc_recomputed"
    ]
    +
    FORECAST_HORIZON
)


# ============================================================
# STRICT CYCLE-24 HORIZON RULE
#
# Required:
#
# window_end < 2020-01-01 00:00 UTC
#
# NOT <=
#
# because an event at exactly
# 2020-01-01 00:00 UTC
# belongs to the Cycle-25 temporal domain.
# ============================================================

audit[
    "horizon_fully_inside_c24"
] = (
    audit[
        "window_end_utc"
    ]
    <
    C24_END_EXCLUSIVE_UTC
)


audit[
    "crosses_cycle_boundary"
] = (
    ~audit[
        "horizon_fully_inside_c24"
    ]
)


# Invalid conversion should never count as valid horizon.
audit.loc[
    audit[
        "sample_time_utc_recomputed"
    ].isna(),
    "horizon_fully_inside_c24"
] = False


audit.loc[
    audit[
        "sample_time_utc_recomputed"
    ].isna(),
    "crosses_cycle_boundary"
] = True


# ============================================================
# ORIGINAL USABLE LABELS
# ============================================================

audit[
    "was_qc8c_usable"
] = (
    audit[
        "label_status"
    ]
    .isin(
        [
            "POSITIVE",
            "NEGATIVE",
        ]
    )
)


# ============================================================
# APPLY BOUNDARY POLICY
#
# This is only an audit/status column.
# We are not rewriting the old QC-8C label file.
# ============================================================

def final_boundary_status(row):

    if pd.isna(
        row[
            "sample_time_utc_recomputed"
        ]
    ):

        return (
            "EXCLUDE_TIME_ERROR"
        )


    if row[
        "crosses_cycle_boundary"
    ]:

        return (
            "EXCLUDE_HORIZON_BOUNDARY"
        )


    return row[
        "label_status"
    ]


audit[
    "qc8d_status"
] = audit.apply(
    final_boundary_status,
    axis=1
)


audit[
    "qc8d_usable"
] = (
    audit[
        "qc8d_status"
    ]
    .isin(
        [
            "POSITIVE",
            "NEGATIVE",
        ]
    )
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "QC-8D LABEL HORIZON BOUNDARY AUDIT"
)

print(
    "===================================="
)


print(
    "Cycle24 end boundary:",
    C24_END_EXCLUSIVE_UTC
)


print(
    "Forecast horizon:",
    FORECAST_HORIZON
)


print(
    "Total audit rows:",
    len(audit)
)


print(
    "TAI -> UTC conversion errors:",
    len(conversion_errors)
)


inside_count = int(
    audit[
        "horizon_fully_inside_c24"
    ].sum()
)


boundary_count = int(
    audit[
        "crosses_cycle_boundary"
    ].sum()
)


print(
    "Rows with complete horizon inside Cycle24:",
    inside_count
)


print(
    "Rows crossing Cycle24 -> Cycle25 boundary:",
    boundary_count
)


original_usable = int(
    audit[
        "was_qc8c_usable"
    ].sum()
)


final_usable = int(
    audit[
        "qc8d_usable"
    ].sum()
)


usable_removed_boundary = int(
    (
        audit[
            "was_qc8c_usable"
        ]
        &
        audit[
            "crosses_cycle_boundary"
        ]
    )
    .sum()
)


print(
    "\nQC-8C usable labels:",
    original_usable
)


print(
    "Usable labels removed by boundary rule:",
    usable_removed_boundary
)


print(
    "QC-8D usable labels:",
    final_usable
)


# ============================================================
# FINAL STATUS COUNTS
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "QC-8D STATUS"
)

print(
    "------------------------------------"
)


status_counts = (
    audit[
        "qc8d_status"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    status_counts.to_string()
)


# ============================================================
# BOUNDARY ROW EXAMPLES
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "BOUNDARY-EXCLUDED AUDIT ROWS"
)

print(
    "------------------------------------"
)


boundary_rows = audit[
    audit[
        "crosses_cycle_boundary"
    ]
]


if len(
    boundary_rows
) == 0:

    print(
        "None in current sparse audit grid."
    )

else:

    print(
        boundary_rows[
            [
                "T_REC",
                "HARPNUM",
                "sample_time_utc_recomputed",
                "window_end_utc",
                "label_status",
                "qc8d_status",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# SYNTHETIC BOUNDARY PROBES
#
# Current quarterly grid may not contain observations near
# Dec-31-2019. Therefore explicitly test the rule around the
# cycle boundary.
# ============================================================

probe_times = [
    "2019-12-30T23:59:58Z",
    "2019-12-30T23:59:59Z",
    "2019-12-31T00:00:00Z",
    "2019-12-31T00:00:01Z",
    "2019-12-31T12:00:00Z",
    "2019-12-31T23:59:59Z",
]


probe_rows = []


for value in probe_times:

    sample_time = pd.Timestamp(
        value
    )


    window_end = (
        sample_time
        +
        FORECAST_HORIZON
    )


    eligible = (
        window_end
        <
        C24_END_EXCLUSIVE_UTC
    )


    probe_rows.append(
        {
            "sample_time_utc":
                sample_time,

            "window_end_utc":
                window_end,

            "c24_label_allowed":
                eligible,

            "expected_action":
                (
                    "ALLOW_LABEL"
                    if eligible
                    else
                    "EXCLUDE_HORIZON_BOUNDARY"
                ),
        }
    )


probes = pd.DataFrame(
    probe_rows
)


print(
    "\n------------------------------------"
)

print(
    "BOUNDARY POLICY PROBES"
)

print(
    "------------------------------------"
)


print(
    probes.to_string(
        index=False
    )
)


# ============================================================
# AUTOMATED POLICY TESTS
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "BOUNDARY POLICY VALIDATION"
)

print(
    "------------------------------------"
)


# Before 24h boundary should pass.
probe_before = probes.iloc[0]


test_1 = bool(
    probe_before[
        "c24_label_allowed"
    ]
)


# Exactly 24h before exclusive boundary:
#
# 2019-12-31 00:00 + 24h
# =
# 2020-01-01 00:00
#
# must NOT pass.
probe_exact = probes[
    probes[
        "sample_time_utc"
    ]
    ==
    pd.Timestamp(
        "2019-12-31T00:00:00Z"
    )
].iloc[0]


test_2 = (
    not bool(
        probe_exact[
            "c24_label_allowed"
        ]
    )
)


# Late Dec31 sample must not pass.
probe_late = probes.iloc[-1]


test_3 = (
    not bool(
        probe_late[
            "c24_label_allowed"
        ]
    )
)


print(
    "Probe before boundary:",
    "PASS"
    if test_1
    else
    "FAIL"
)


print(
    "Exact 24h boundary excluded:",
    "PASS"
    if test_2
    else
    "FAIL"
)


print(
    "Late Dec-31 sample excluded:",
    "PASS"
    if test_3
    else
    "FAIL"
)


all_policy_tests_pass = (
    test_1
    and
    test_2
    and
    test_3
)


print(
    "Overall boundary policy:",
    "PASS"
    if all_policy_tests_pass
    else
    "FAIL"
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


audit.to_csv(
    OUTPUT_FILE,
    index=False
)


probes.to_csv(
    PROBE_FILE,
    index=False
)


summary = pd.DataFrame(
    [
        {
            "cycle":
                "Cycle24",

            "forecast_horizon_hours":
                24,

            "cycle_end_exclusive_utc":
                str(
                    C24_END_EXCLUSIVE_UTC
                ),

            "total_audit_rows":
                len(audit),

            "conversion_errors":
                len(conversion_errors),

            "inside_cycle_horizon":
                inside_count,

            "cross_boundary_rows":
                boundary_count,

            "qc8c_usable_labels":
                original_usable,

            "usable_removed_by_boundary":
                usable_removed_boundary,

            "qc8d_usable_labels":
                final_usable,

            "boundary_policy_tests_pass":
                all_policy_tests_pass,
        }
    ]
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)

print(
    PROBE_FILE
)


print(
    "\nQC-8D horizon-boundary audit finished."
)