from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from astropy.time import Time


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

sharp_file = Path("data/raw/sharp/sharp_sample.parquet")
goes_file = Path("data/raw/goes/goes_flare_2011.csv")


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

sharp_df = pd.read_parquet(sharp_file)
goes_df = pd.read_csv(goes_file)

print("SHARP rows:", len(sharp_df))
print("GOES rows:", len(goes_df))


# --------------------------------------------------
# 3. CONVERT SHARP TAI TIME TO UTC
# --------------------------------------------------

def tai_to_utc(value):

    cleaned = value.replace("_TAI", "")

    dt = datetime.strptime(
        cleaned,
        "%Y.%m.%d_%H:%M:%S"
    )

    tai_time = Time(
        dt,
        scale="tai"
    )

    utc_time = tai_time.utc.to_datetime(
        timezone=timezone.utc
    )

    return utc_time


sharp_df["time_utc"] = sharp_df["T_REC"].apply(
    tai_to_utc
)


# --------------------------------------------------
# 4. PREPARE GOES TIME
# --------------------------------------------------

goes_df["start_time"] = pd.to_datetime(
    goes_df["start_time"],
    utc=True
)


# --------------------------------------------------
# 5. CHECK IF FLARE IS M1.0+
# --------------------------------------------------

def is_m1_plus(flare_class):

    if pd.isna(flare_class):
        return False

    flare_class = str(flare_class).strip().upper()

    if flare_class.startswith("X"):
        return True

    if flare_class.startswith("M"):
        try:
            value = float(flare_class[1:])
            return value >= 1.0
        except ValueError:
            return False

    return False


goes_df["is_m1_plus"] = goes_df["flare_class"].apply(
    is_m1_plus
)


# --------------------------------------------------
# 6. TAKE FIRST SHARP SAMPLE
# --------------------------------------------------

sample = sharp_df.iloc[0]

sample_time = sample["time_utc"]
sharp_region = int(sample["NOAA_AR"])


# GOES historical region format
goes_region = sharp_region % 10000


print("\nSHARP sample:")
print("Time UTC:", sample_time)
print("SHARP NOAA region:", sharp_region)
print("GOES region:", goes_region)


# --------------------------------------------------
# 7. CREATE 24-HOUR WINDOW
# --------------------------------------------------

window_end = sample_time + timedelta(hours=24)


# --------------------------------------------------
# 8. FIND FUTURE M1+ FLARES
# --------------------------------------------------

future_flares = goes_df[
    (goes_df["active_region"] == goes_region)
    &
    (goes_df["is_m1_plus"])
    &
    (goes_df["start_time"] > sample_time)
    &
    (goes_df["start_time"] <= window_end)
].copy()


# --------------------------------------------------
# 9. CREATE TARGET
# --------------------------------------------------

target = 1 if len(future_flares) > 0 else 0


print("\n24-hour window:")
print(sample_time, "→", window_end)

print("\nFuture M1.0+ flares:")

print(
    future_flares[
        [
            "start_time",
            "flare_class",
            "active_region",
        ]
    ].to_string(index=False)
)

print("\nFINAL TARGET =", target)

# --------------------------------------------------
# 10. LABEL ALL SHARP SAMPLE ROWS
# --------------------------------------------------

def create_label(row):

    sample_time = row["time_utc"]

    sharp_region = int(row["NOAA_AR"])
    goes_region = sharp_region % 10000

    window_end = sample_time + timedelta(hours=24)

    matching_flares = goes_df[
        (goes_df["active_region"] == goes_region)
        &
        (goes_df["is_m1_plus"])
        &
        (goes_df["start_time"] > sample_time)
        &
        (goes_df["start_time"] <= window_end)
    ]

    if len(matching_flares) > 0:
        return 1

    return 0


sharp_df["target"] = sharp_df.apply(
    create_label,
    axis=1
)


print("\n--- ALL SHARP SAMPLE LABELS ---")

print(
    sharp_df[
        [
            "T_REC",
            "time_utc",
            "HARPNUM",
            "NOAA_AR",
            "target",
        ]
    ].to_string(index=False)
)