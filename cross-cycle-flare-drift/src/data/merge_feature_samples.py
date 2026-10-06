from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

sharp_file = Path(
    "data/raw/sharp/sharp_18_sample.parquet"
)

lorentz_file = Path(
    "data/raw/lorentz/lorentz_sample.parquet"
)


# --------------------------------------------------
# 2. LOAD BOTH DATASETS
# --------------------------------------------------

sharp_df = pd.read_parquet(sharp_file)
lorentz_df = pd.read_parquet(lorentz_file)

print("SHARP rows:", len(sharp_df))
print("Lorentz rows:", len(lorentz_df))


# --------------------------------------------------
# 3. CHECK DUPLICATES
# --------------------------------------------------

sharp_duplicates = sharp_df.duplicated(
    subset=["HARPNUM", "T_REC"]
).sum()

lorentz_duplicates = lorentz_df.duplicated(
    subset=["HARPNUM", "T_REC"]
).sum()

print("\nSHARP duplicate keys:", sharp_duplicates)
print("Lorentz duplicate keys:", lorentz_duplicates)


# --------------------------------------------------
# 4. MERGE DATA
# --------------------------------------------------

merged_df = sharp_df.merge(
    lorentz_df,
    on=["HARPNUM", "T_REC"],
    how="inner",
    validate="one_to_one"
)


# --------------------------------------------------
# 5. RESULT
# --------------------------------------------------

print("\nMerged rows:", len(merged_df))
print("Total columns:", len(merged_df.columns))

print("\nColumns:")
print(merged_df.columns.tolist())


# --------------------------------------------------
# 6. CHECK MISSING VALUES
# --------------------------------------------------

print("\nMissing values:")

print(
    merged_df.isna().sum()[
        merged_df.isna().sum() > 0
    ]
)


# --------------------------------------------------
# 7. SAVE
# --------------------------------------------------

output_folder = Path("data/interim")
output_folder.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_folder /
    "combined_25_sample.parquet"
)

merged_df.to_parquet(
    output_file,
    index=False
)

print("\nSaved successfully:")
print(output_file)