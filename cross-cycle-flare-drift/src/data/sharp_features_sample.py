from pathlib import Path
import drms


SERIES = "hmi.sharp_cea_720s"

client = drms.Client()


# Same HARP and same time window
query = (
    "hmi.sharp_cea_720s"
    "[401]"
    "[2011.03.09_23:00:00_TAI/1h@12m]"
)


# 18 features directly available in SHARP
FEATURES = [
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


keys = ",".join(
    [
        "T_REC",
        "HARPNUM",
        "NOAA_AR",
        "NOAA_ARS",
        "QUALITY",
    ]
    + FEATURES
)


print("Downloading 18 SHARP features...")

df = client.query(
    query,
    key=keys
)


print("\nFirst rows:")
print(df.head())

print("\nTotal rows:", len(df))

print("\nTotal columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())


# Save
output_folder = Path("data/raw/sharp")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "sharp_18_sample.parquet"

df.to_parquet(
    output_file,
    index=False
)

print("\nSaved successfully:")
print(output_file)