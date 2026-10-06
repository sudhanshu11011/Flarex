from pathlib import Path

import pandas as pd
import requests


# --------------------------------------------------
# 1. NOAA GOES DATA URL
# --------------------------------------------------

URL = (
    "https://data.ngdc.noaa.gov/"
    "platforms/solar-space-observing-satellites/"
    "goes/multi/l2/data/"
    "xrsf-l2-flrpt_science/csv/"
    "sci_xrsf-l2-flrpt_geo_y2011_v1-0-1.csv"
)


# --------------------------------------------------
# 2. OUTPUT LOCATION
# --------------------------------------------------

output_folder = Path("data/raw/goes")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "goes_flare_2011.csv"


# --------------------------------------------------
# 3. DOWNLOAD FILE
# --------------------------------------------------

print("Downloading GOES flare dataset...")

response = requests.get(URL, timeout=60)

response.raise_for_status()

output_file.write_bytes(response.content)

print("Download successful.")
print("Saved:", output_file)


# --------------------------------------------------
# 4. READ DATA
# --------------------------------------------------

df = pd.read_csv(output_file)


# --------------------------------------------------
# 5. INSPECT DATA
# --------------------------------------------------

print("\nTotal rows:", len(df))

print("\nColumns:")

for column in df.columns:
    print(column)

print("\nFirst 5 rows:")

print(df.head())

# --------------------------------------------------
# 6. CHECK FLARE CLASSES
# --------------------------------------------------

print("\n--- FLARE CLASS CHECK ---")

df["flare_class"] = (
    df["flare_class"]
    .astype("string")
    .str.strip()
    .str.upper()
)

print("\nExample flare classes:")
print(df["flare_class"].dropna().head(20).tolist())


# --------------------------------------------------
# 7. FILTER M AND X FLARES
# --------------------------------------------------

mx_flares = df[
    df["flare_class"].str.startswith(("M", "X"), na=False)
].copy()

print("\nTotal M/X flares:", len(mx_flares))


# --------------------------------------------------
# 8. CHECK ACTIVE REGION INFORMATION
# --------------------------------------------------

mx_with_region = mx_flares[
    mx_flares["active_region"].notna()
].copy()

print(
    "M/X flares with active-region number:",
    len(mx_with_region)
)

print(
    "M/X flares without active-region number:",
    mx_flares["active_region"].isna().sum()
)


# --------------------------------------------------
# 9. SHOW USEFUL M/X EVENTS
# --------------------------------------------------

print("\nSample M/X flares:")

print(
    mx_with_region[
        [
            "start_time",
            "time",
            "flare_class",
            "active_region",
        ]
    ].head(10)
)


# --------------------------------------------------
# 10. CHECK NOAA REGION 11166
# --------------------------------------------------

# SHARP may store full NOAA number like 11166
# GOES/SWPC historical report may store it as 1166

sharp_noaa_region = 11166

goes_region = sharp_noaa_region % 10000

print("\nSHARP NOAA Region:", sharp_noaa_region)
print("GOES/SWPC Region used for matching:", goes_region)

region_events = df[
    df["active_region"] == goes_region
]

print("\nEvents for NOAA Active Region 11166 / SWPC 1166:")

print(
    region_events[
        [
            "start_time",
            "time",
            "flare_class",
            "active_region",
        ]
    ]
)

# --------------------------------------------------
# 10. CHECK NOAA REGION MATCHING
# --------------------------------------------------

sharp_noaa_region = 11166

# Historical GOES/SWPC reports may store 11166 as 1166
goes_region = sharp_noaa_region % 10000

print("\nSHARP NOAA Region:", sharp_noaa_region)
print("GOES/SWPC Region used for matching:", goes_region)

region_events = df[
    df["active_region"] == goes_region
]

print("\nEvents for NOAA Active Region 11166 / SWPC 1166:")

print(
    region_events[
        [
            "start_time",
            "time",
            "flare_class",
            "active_region",
        ]
    ].to_string(index=False)
)