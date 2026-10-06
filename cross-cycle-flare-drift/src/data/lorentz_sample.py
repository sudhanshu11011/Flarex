from pathlib import Path
import drms


# JSOC Lorentz series
SERIES = "cgem.lorentz"

client = drms.Client()


# Same HARP and same time window as our SHARP sample
query = (
    "cgem.lorentz"
    "[401]"
    "[2011.03.09_23:00:00_TAI/1h@12m]"
)


# 7 missing Lorentz features
keys = ",".join(
    [
        "T_REC",
        "HARPNUM",
        "TOTBSQ",
        "TOTFX",
        "TOTFY",
        "TOTFZ",
        "EPSX",
        "EPSY",
        "EPSZ",
    ]
)


print("Downloading Lorentz sample...")

df = client.query(
    query,
    key=keys
)


print("\nFirst rows:")
print(df.head())

print("\nTotal rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())


# Save sample
output_folder = Path("data/raw/lorentz")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "lorentz_sample.parquet"

df.to_parquet(
    output_file,
    index=False
)

print("\nSaved successfully:")
print(output_file)