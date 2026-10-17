from pathlib import Path
import subprocess
import hashlib
import pandas as pd
from datetime import datetime, timezone


# ============================================================
# CONFIG
# ============================================================

JSOC_HTTP_URL = (
    "http://jsoc.stanford.edu/doc/data/hmi/"
    "harpnum_to_noaa/all_harps_with_noaa_ars.txt"
)

JSOC_HTTPS_URL = (
    "https://jsoc.stanford.edu/doc/data/hmi/"
    "harpnum_to_noaa/all_harps_with_noaa_ars.txt"
)


RAW_DIR = Path(
    "data/raw/mappings"
)

MANIFEST_DIR = Path(
    "data_manifests"
)


RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RAW_FILE = (
    RAW_DIR
    / "all_harps_with_noaa_ars.txt"
)


EDGE_FILE = (
    MANIFEST_DIR
    / "harp_noaa_edges.csv"
)


METADATA_FILE = (
    MANIFEST_DIR
    / "harp_noaa_mapping_metadata.csv"
)


# ============================================================
# HELPER: SHA256
# ============================================================

def sha256_file(path: Path) -> str:

    sha = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# HELPER: DOWNLOAD USING WINDOWS CURL
# ============================================================

def download_with_curl(
    url: str,
    output_file: Path
) -> bool:

    print("\nTrying:")
    print(url)

    command = [
        "curl.exe",

        "-L",

        "--fail",

        "--retry",
        "3",

        "--retry-delay",
        "2",

        "--connect-timeout",
        "30",

        "--max-time",
        "120",

        "-A",
        "Mozilla/5.0",

        url,

        "-o",
        str(output_file),
    ]


    try:

        subprocess.run(
            command,
            check=True
        )

    except subprocess.CalledProcessError as error:

        print(
            "Download failed:",
            error
        )

        return False


    if not output_file.exists():

        print(
            "Download command finished "
            "but file does not exist."
        )

        return False


    if output_file.stat().st_size == 0:

        print(
            "Downloaded file is empty."
        )

        return False


    print(
        "Download successful."
    )

    return True


# ============================================================
# DOWNLOAD CANONICAL MAPPING
# ============================================================

print(
    "Downloading canonical JSOC "
    "HARP-NOAA mapping..."
)


download_success = False
download_url_used = None


# First try HTTP.
# This sometimes avoids Python/OpenSSL TLS issues.
if download_with_curl(
    JSOC_HTTP_URL,
    RAW_FILE
):

    download_success = True
    download_url_used = JSOC_HTTP_URL


# If HTTP fails, try HTTPS via curl.exe.
if not download_success:

    print(
        "\nHTTP attempt failed."
    )

    print(
        "Trying HTTPS through curl.exe..."
    )

    if download_with_curl(
        JSOC_HTTPS_URL,
        RAW_FILE
    ):

        download_success = True
        download_url_used = JSOC_HTTPS_URL


# ============================================================
# FAIL SAFELY
# ============================================================

if not download_success:

    if (
        RAW_FILE.exists()
        and RAW_FILE.stat().st_size > 0
    ):

        print(
            "\nWARNING:"
        )

        print(
            "Fresh download failed, but an "
            "existing local mapping file exists."
        )

        print(
            "Using existing file:"
        )

        print(
            RAW_FILE
        )

        download_url_used = (
            "existing_local_copy"
        )

    else:

        raise RuntimeError(
            "\nJSOC mapping download failed.\n"
            "No usable local mapping file exists.\n"
            "Do NOT continue QC-7 with guessed mappings."
        )


print("\nRaw mapping file:")
print(RAW_FILE)


print(
    "File size:",
    RAW_FILE.stat().st_size,
    "bytes"
)


mapping_sha256 = sha256_file(
    RAW_FILE
)


print(
    "SHA256:",
    mapping_sha256
)


# ============================================================
# PARSE CANONICAL FILE
# ============================================================

edges = []

harp_to_noaa = {}


with RAW_FILE.open(
    "r",
    encoding="utf-8",
    errors="ignore"
) as file:

    for raw_line in file:

        line = raw_line.strip()


        # Skip blank lines
        if not line:
            continue


        # Skip comments
        if line.startswith("#"):
            continue


        # Split HARP number from NOAA list
        parts = line.split(
            maxsplit=1
        )


        if len(parts) < 1:
            continue


        harp_text = parts[0].strip()


        # Skip headers / malformed lines
        if not harp_text.isdigit():
            continue


        harpnum = int(
            harp_text
        )


        # If no NOAA portion exists
        if len(parts) == 1:

            noaa_list = []

        else:

            noaa_text = (
                parts[1]
                .strip()
                .replace(" ", "")
            )


            if (
                not noaa_text
                or
                noaa_text.upper()
                in {
                    "MISSING",
                    "NONE",
                    "NULL",
                    "N/A",
                    "NA",
                    "-"
                }
            ):

                noaa_list = []

            else:

                noaa_list = []


                # Expected format can contain comma-separated NOAA ARs
                for value in noaa_text.split(","):

                    value = value.strip()


                    if value.isdigit():

                        noaa_value = int(
                            value
                        )


                        if noaa_value > 0:

                            noaa_list.append(
                                noaa_value
                            )


                # Deduplicate NOAA ARs for same HARP
                noaa_list = sorted(
                    set(
                        noaa_list
                    )
                )


        harp_to_noaa[
            harpnum
        ] = noaa_list


        for noaa_ar in noaa_list:

            edges.append(
                {
                    "HARPNUM":
                        harpnum,

                    "NOAA_AR":
                        noaa_ar,
                }
            )


# ============================================================
# CREATE EDGE TABLE
# ============================================================

edges_df = pd.DataFrame(
    edges,
    columns=[
        "HARPNUM",
        "NOAA_AR",
    ]
)


if len(edges_df) > 0:

    edges_df = (
        edges_df
        .drop_duplicates(
            subset=[
                "HARPNUM",
                "NOAA_AR",
            ]
        )
        .sort_values(
            [
                "HARPNUM",
                "NOAA_AR",
            ]
        )
        .reset_index(
            drop=True
        )
    )


edges_df.to_csv(
    EDGE_FILE,
    index=False
)


# ============================================================
# BASIC AUDIT
# ============================================================

print(
    "\n===================================="
)

print(
    "HARP ↔ NOAA CANONICAL MAPPING AUDIT"
)

print(
    "===================================="
)


print(
    "HARPs in mapping file:",
    len(harp_to_noaa)
)


print(
    "Unique HARP-NOAA edges:",
    len(edges_df)
)


if len(edges_df) > 0:

    unique_noaa = int(
        edges_df[
            "NOAA_AR"
        ].nunique()
    )

else:

    unique_noaa = 0


print(
    "Unique NOAA ARs:",
    unique_noaa
)


# ============================================================
# HARP → NOAA MULTIPLICITY
# ============================================================

harp_counts = {
    harp:
    len(noaa_list)

    for harp, noaa_list
    in harp_to_noaa.items()
}


zero_noaa_harps = [
    harp

    for harp, count
    in harp_counts.items()

    if count == 0
]


single_noaa_harps = [
    harp

    for harp, count
    in harp_counts.items()

    if count == 1
]


multi_noaa_harps = [
    harp

    for harp, count
    in harp_counts.items()

    if count > 1
]


print(
    "\n------------------------------------"
)

print(
    "HARP → NOAA multiplicity"
)

print(
    "------------------------------------"
)


print(
    "HARPs with 0 NOAA AR:",
    len(zero_noaa_harps)
)


print(
    "HARPs with exactly 1 NOAA AR:",
    len(single_noaa_harps)
)


print(
    "HARPs with multiple NOAA ARs:",
    len(multi_noaa_harps)
)


print(
    "\nFirst multi-NOAA HARPs:"
)


if len(multi_noaa_harps) == 0:

    print(
        "None found."
    )

else:

    for harp in multi_noaa_harps[:20]:

        print(
            harp,
            "->",
            harp_to_noaa[
                harp
            ]
        )


# ============================================================
# NOAA → HARP MULTIPLICITY
# ============================================================

noaa_to_harps = {}


for row in edges:

    noaa = row[
        "NOAA_AR"
    ]

    harp = row[
        "HARPNUM"
    ]


    if noaa not in noaa_to_harps:

        noaa_to_harps[
            noaa
        ] = set()


    noaa_to_harps[
        noaa
    ].add(
        harp
    )


multi_harp_noaa = {
    noaa:
    sorted(
        harps
    )

    for noaa, harps
    in noaa_to_harps.items()

    if len(harps) > 1
}


print(
    "\n------------------------------------"
)

print(
    "NOAA → HARP multiplicity"
)

print(
    "------------------------------------"
)


print(
    "NOAA ARs connected to >1 HARP:",
    len(multi_harp_noaa)
)


print(
    "\nFirst NOAA ARs connected "
    "to multiple HARPs:"
)


if len(multi_harp_noaa) == 0:

    print(
        "None found."
    )

else:

    for noaa, harps in list(
        sorted(
            multi_harp_noaa.items()
        )
    )[:20]:

        print(
            noaa,
            "->",
            harps
        )


# ============================================================
# KNOWN SAMPLE VALIDATION
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "KNOWN SAMPLE CHECK"
)

print(
    "------------------------------------"
)


TEST_HARP = 401
EXPECTED_NOAA = 11166


sample_mapping = (
    harp_to_noaa.get(
        TEST_HARP
    )
)


print(
    "HARP 401 mapping:",
    sample_mapping
)


if (
    sample_mapping is not None
    and
    EXPECTED_NOAA in sample_mapping
):

    print(
        "PASS:"
        " HARP 401 ↔ NOAA 11166"
    )

else:

    print(
        "CHECK REQUIRED:"
        " expected NOAA 11166"
    )


# ============================================================
# DUPLICATE EDGE VALIDATION
# ============================================================

if len(edges_df) > 0:

    duplicate_edges = int(
        edges_df[
            [
                "HARPNUM",
                "NOAA_AR",
            ]
        ]
        .duplicated()
        .sum()
    )

else:

    duplicate_edges = 0


print(
    "\n------------------------------------"
)

print(
    "EDGE VALIDATION"
)

print(
    "------------------------------------"
)


print(
    "Duplicate HARP-NOAA edges:",
    duplicate_edges
)


# ============================================================
# ADDITIONAL SANITY CHECKS
# ============================================================

invalid_harp_edges = 0
invalid_noaa_edges = 0


if len(edges_df) > 0:

    invalid_harp_edges = int(
        (
            edges_df[
                "HARPNUM"
            ] <= 0
        ).sum()
    )


    invalid_noaa_edges = int(
        (
            edges_df[
                "NOAA_AR"
            ] <= 0
        ).sum()
    )


print(
    "Invalid HARPNUM edges:",
    invalid_harp_edges
)


print(
    "Invalid NOAA_AR edges:",
    invalid_noaa_edges
)


# ============================================================
# SAVE REPRODUCIBILITY METADATA
# ============================================================

metadata = pd.DataFrame(
    [
        {
            "downloaded_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "source":
                download_url_used,

            "raw_file":
                str(
                    RAW_FILE
                ),

            "file_size_bytes":
                RAW_FILE.stat().st_size,

            "sha256":
                mapping_sha256,

            "harps":
                len(
                    harp_to_noaa
                ),

            "unique_edges":
                len(
                    edges_df
                ),

            "unique_noaa_ars":
                unique_noaa,

            "zero_noaa_harps":
                len(
                    zero_noaa_harps
                ),

            "single_noaa_harps":
                len(
                    single_noaa_harps
                ),

            "multi_noaa_harps":
                len(
                    multi_noaa_harps
                ),

            "noaa_connected_to_multiple_harps":
                len(
                    multi_harp_noaa
                ),

            "duplicate_edges":
                duplicate_edges,
        }
    ]
)


metadata.to_csv(
    METADATA_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "FILES SAVED"
)

print(
    "------------------------------------"
)


print(
    "Raw mapping:"
)

print(
    RAW_FILE
)


print(
    "\nParsed edges:"
)

print(
    EDGE_FILE
)


print(
    "\nMetadata:"
)

print(
    METADATA_FILE
)


print(
    "\nHARP-NOAA mapping audit finished."
)