from pathlib import Path
import json

import pandas as pd


MANIFEST_FILE = Path(
    "data_manifests/phase9_jsoc_extraction_manifest.csv"
)

CONTRACT_FILE = Path(
    "data_manifests/temporal_data_contract.json"
)

OUTPUT_FILE = Path(
    "data_manifests/phase9_extraction_completeness_summary.csv"
)


EXPECTED = {
    "cycle24": 116,
    "cycle25": 72,
}


# ============================================================
# LOAD FILES
# ============================================================

if not MANIFEST_FILE.exists():

    raise FileNotFoundError(
        f"Missing manifest: {MANIFEST_FILE}"
    )


if not CONTRACT_FILE.exists():

    raise FileNotFoundError(
        f"Missing contract: {CONTRACT_FILE}"
    )


manifest = pd.read_csv(
    MANIFEST_FILE
)


with CONTRACT_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    contract = json.load(file)


if contract.get("status") != "FROZEN":

    raise RuntimeError(
        "Temporal contract is not frozen."
    )


# ============================================================
# REQUIRED MANIFEST COLUMNS
# ============================================================

required_columns = {
    "cycle",
    "chunk_id",
    "status",
    "sharp_rows",
    "lorentz_rows",
    "sharp_duplicate_keys",
    "lorentz_duplicate_keys",
    "sharp_file",
    "lorentz_file",
    "sharp_sha256",
    "lorentz_sha256",
    "stride_minutes",
}


missing_columns = (
    required_columns
    -
    set(manifest.columns)
)


if missing_columns:

    raise RuntimeError(
        f"Manifest missing columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# AUDIT
# ============================================================

summary_rows = []

overall_pass = True


print(
    "===================================="
)

print(
    "PHASE 9C — EXTRACTION COMPLETENESS"
)

print(
    "===================================="
)


for cycle in [
    "cycle24",
    "cycle25",
]:

    subset = (
        manifest[
            manifest["cycle"] == cycle
        ]
        .copy()
        .sort_values("chunk_id")
    )


    expected_chunks = EXPECTED[
        cycle
    ]


    manifest_chunks = len(
        subset
    )


    unique_chunks = subset[
        "chunk_id"
    ].nunique()


    complete_chunks = int(
        (
            subset["status"]
            ==
            "COMPLETE"
        )
        .sum()
    )


    failed_chunks = int(
        (
            subset["status"]
            !=
            "COMPLETE"
        )
        .sum()
    )


    wrong_stride = int(
        (
            pd.to_numeric(
                subset["stride_minutes"],
                errors="coerce"
            )
            !=
            120
        )
        .sum()
    )


    sharp_duplicates = int(
        pd.to_numeric(
            subset[
                "sharp_duplicate_keys"
            ],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


    lorentz_duplicates = int(
        pd.to_numeric(
            subset[
                "lorentz_duplicate_keys"
            ],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


    sharp_rows = int(
        pd.to_numeric(
            subset["sharp_rows"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


    lorentz_rows = int(
        pd.to_numeric(
            subset["lorentz_rows"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


    sharp_missing_files = 0
    lorentz_missing_files = 0

    sharp_missing_hash = 0
    lorentz_missing_hash = 0


    for _, row in subset.iterrows():

        sharp_path = Path(
            str(
                row["sharp_file"]
            )
        )


        lorentz_path = Path(
            str(
                row["lorentz_file"]
            )
        )


        if not sharp_path.exists():

            sharp_missing_files += 1


        if not lorentz_path.exists():

            lorentz_missing_files += 1


        if (
            pd.isna(
                row["sharp_sha256"]
            )
            or
            not str(
                row["sharp_sha256"]
            ).strip()
        ):

            sharp_missing_hash += 1


        if (
            pd.isna(
                row["lorentz_sha256"]
            )
            or
            not str(
                row["lorentz_sha256"]
            ).strip()
        ):

            lorentz_missing_hash += 1


    cycle_pass = (

        manifest_chunks
        ==
        expected_chunks

        and

        unique_chunks
        ==
        expected_chunks

        and

        complete_chunks
        ==
        expected_chunks

        and

        failed_chunks
        ==
        0

        and

        wrong_stride
        ==
        0

        and

        sharp_duplicates
        ==
        0

        and

        lorentz_duplicates
        ==
        0

        and

        sharp_missing_files
        ==
        0

        and

        lorentz_missing_files
        ==
        0

        and

        sharp_missing_hash
        ==
        0

        and

        lorentz_missing_hash
        ==
        0
    )


    if not cycle_pass:

        overall_pass = False


    print(
        f"\n{cycle.upper()}"
    )


    print(
        "Expected chunks:",
        expected_chunks
    )


    print(
        "Manifest chunks:",
        manifest_chunks
    )


    print(
        "Unique chunks:",
        unique_chunks
    )


    print(
        "Complete chunks:",
        complete_chunks
    )


    print(
        "Failed chunks:",
        failed_chunks
    )


    print(
        "Wrong stride chunks:",
        wrong_stride
    )


    print(
        "SHARP rows:",
        sharp_rows
    )


    print(
        "Lorentz rows:",
        lorentz_rows
    )


    print(
        "SHARP duplicate keys:",
        sharp_duplicates
    )


    print(
        "Lorentz duplicate keys:",
        lorentz_duplicates
    )


    print(
        "Missing SHARP files:",
        sharp_missing_files
    )


    print(
        "Missing Lorentz files:",
        lorentz_missing_files
    )


    print(
        "Missing SHARP hashes:",
        sharp_missing_hash
    )


    print(
        "Missing Lorentz hashes:",
        lorentz_missing_hash
    )


    print(
        "Cycle audit:",
        (
            "PASS"
            if cycle_pass
            else "FAIL"
        )
    )


    summary_rows.append(
        {
            "cycle":
                cycle,

            "expected_chunks":
                expected_chunks,

            "manifest_chunks":
                manifest_chunks,

            "unique_chunks":
                unique_chunks,

            "complete_chunks":
                complete_chunks,

            "failed_chunks":
                failed_chunks,

            "wrong_stride_chunks":
                wrong_stride,

            "sharp_rows":
                sharp_rows,

            "lorentz_rows":
                lorentz_rows,

            "sharp_duplicate_keys":
                sharp_duplicates,

            "lorentz_duplicate_keys":
                lorentz_duplicates,

            "missing_sharp_files":
                sharp_missing_files,

            "missing_lorentz_files":
                lorentz_missing_files,

            "missing_sharp_hashes":
                sharp_missing_hash,

            "missing_lorentz_hashes":
                lorentz_missing_hash,

            "audit_pass":
                cycle_pass,
        }
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# GLOBAL SUMMARY
# ============================================================

total_expected = sum(
    EXPECTED.values()
)


total_manifest = len(
    manifest[
        manifest["cycle"].isin(
            EXPECTED.keys()
        )
    ]
)


total_sharp_rows = int(
    summary["sharp_rows"].sum()
)


total_lorentz_rows = int(
    summary["lorentz_rows"].sum()
)


print(
    "\n===================================="
)

print(
    "PHASE 9 FINAL SUMMARY"
)

print(
    "===================================="
)


print(
    "Expected total chunks:",
    total_expected
)


print(
    "Manifest total chunks:",
    total_manifest
)


print(
    "Total SHARP rows:",
    total_sharp_rows
)


print(
    "Total Lorentz rows:",
    total_lorentz_rows
)


print(
    "Overall Phase 9 audit:",
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
    OUTPUT_FILE
)


if not overall_pass:

    raise RuntimeError(
        "Phase 9 completeness audit failed."
    )


print(
    "\nPHASE 9 COMPLETE."
)