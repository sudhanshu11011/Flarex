from pathlib import Path

import drms


# --------------------------------------------------
# 1. JSOC SERIES
# --------------------------------------------------

SERIES = "hmi.sharp_cea_720s"


# --------------------------------------------------
# 2. CONNECT TO JSOC
# --------------------------------------------------

print("Connecting to JSOC...")

client = drms.Client()

print("Connected successfully.")
print("Series:", SERIES)


# --------------------------------------------------
# 3. SMALL TEST QUERY
# --------------------------------------------------

# We are NOT downloading the full dataset yet.
# First we download only a tiny sample.

query = (
    "hmi.sharp_cea_720s"
    "[401]"
    "[2011.03.09_23:00:00_TAI/1h@12m]"
)


# Columns we want for this test
keys = ",".join(
    [
        "T_REC",
        "HARPNUM",
        "NOAA_AR",
        "NOAA_ARS",
        "QUALITY",
        "USFLUX",
        "R_VALUE",
        "TOTUSJH",
    ]
)


print("\nDownloading small SHARP sample...")


# --------------------------------------------------
# 4. DOWNLOAD DATA
# --------------------------------------------------

df = client.query(
    query,
    key=keys
)


# --------------------------------------------------
# 5. SHOW RESULT
# --------------------------------------------------

print("\nFirst rows:")
print(df.head())

print("\nTotal rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 6. SAVE DATA
# --------------------------------------------------

output_folder = Path("data/raw/sharp")

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

output_file = output_folder / "sharp_sample.parquet"

df.to_parquet(
    output_file,
    index=False
)

print("\nSaved successfully:")
print(output_file)