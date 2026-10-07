import drms
import pandas as pd


# --------------------------------------------------
# 1. JSOC CLIENT
# --------------------------------------------------

client = drms.Client()


# --------------------------------------------------
# 2. 2024 SAMPLE
# --------------------------------------------------

HARPNUM = 12519
TIME = "2024.12.30_22:24:00_TAI"


# --------------------------------------------------
# 3. SHARP FEATURES
# --------------------------------------------------

SHARP_FEATURES = [
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


# --------------------------------------------------
# 4. LORENTZ FEATURES
# --------------------------------------------------

LORENTZ_FEATURES = [
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
]


# --------------------------------------------------
# 5. QUERY SHARP
# --------------------------------------------------

sharp_query = (
    f"hmi.sharp_cea_720s"
    f"[{HARPNUM}]"
    f"[{TIME}/1h@12m]"
)

sharp_keys = ",".join(
    [
        "T_REC",
        "HARPNUM",
        "NOAA_AR",
        "QUALITY",
    ]
    + SHARP_FEATURES
)

sharp_df = client.query(
    sharp_query,
    key=sharp_keys
)


# --------------------------------------------------
# 6. QUERY LORENTZ
# --------------------------------------------------

lorentz_query = (
    f"cgem.lorentz"
    f"[{HARPNUM}]"
    f"[{TIME}/1h@12m]"
)

lorentz_keys = ",".join(
    [
        "T_REC",
        "HARPNUM",
    ]
    + LORENTZ_FEATURES
)

lorentz_df = client.query(
    lorentz_query,
    key=lorentz_keys
)


# --------------------------------------------------
# 7. BASIC INFO
# --------------------------------------------------

print("\n==============================")
print("2024 MISSING VALUE DIAGNOSIS")
print("==============================")

print("\nSHARP rows:", len(sharp_df))
print("Lorentz rows:", len(lorentz_df))


# --------------------------------------------------
# 8. SHARP MISSING VALUES
# --------------------------------------------------

print("\n--- SHARP FEATURE MISSING VALUES ---")

sharp_missing = sharp_df[
    SHARP_FEATURES
].isna().sum()

for feature, count in sharp_missing.items():

    if count > 0:
        print(feature, "->", count)


print(
    "\nTotal SHARP missing values:",
    sharp_missing.sum()
)


# --------------------------------------------------
# 9. LORENTZ MISSING VALUES
# --------------------------------------------------

print("\n--- LORENTZ FEATURE MISSING VALUES ---")

lorentz_missing = lorentz_df[
    LORENTZ_FEATURES
].isna().sum()

for feature, count in lorentz_missing.items():

    if count > 0:
        print(feature, "->", count)


print(
    "\nTotal Lorentz missing values:",
    lorentz_missing.sum()
)


# --------------------------------------------------
# 10. QUALITY VALUES
# --------------------------------------------------

print("\n--- SHARP QUALITY VALUES ---")

print(
    sharp_df[
        [
            "T_REC",
            "HARPNUM",
            "NOAA_AR",
            "QUALITY",
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# 11. MERGE BOTH
# --------------------------------------------------

merged_df = sharp_df.merge(
    lorentz_df,
    on=["HARPNUM", "T_REC"],
    how="inner",
    validate="one_to_one"
)


ALL_FEATURES = (
    SHARP_FEATURES
    + LORENTZ_FEATURES
)


# --------------------------------------------------
# 12. MISSING VALUES PER ROW
# --------------------------------------------------

merged_df["missing_feature_count"] = (
    merged_df[
        ALL_FEATURES
    ].isna().sum(axis=1)
)


print("\n--- MISSING VALUES PER TIMESTAMP ---")

print(
    merged_df[
        [
            "T_REC",
            "QUALITY",
            "missing_feature_count",
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# 13. EXACT MISSING FEATURE LIST PER ROW
# --------------------------------------------------

print("\n--- EXACT MISSING FEATURES ---")

for _, row in merged_df.iterrows():

    missing_features = [
        feature
        for feature in ALL_FEATURES
        if pd.isna(row[feature])
    ]

    print("\nTime:", row["T_REC"])

    if missing_features:
        print("Missing:", missing_features)

    else:
        print("Missing: NONE")


# --------------------------------------------------
# 14. FINAL SUMMARY
# --------------------------------------------------

total_missing = (
    merged_df[
        ALL_FEATURES
    ].isna().sum().sum()
)

print("\n==============================")
print("FINAL SUMMARY")
print("==============================")

print("Total rows:", len(merged_df))
print("Total features:", len(ALL_FEATURES))
print("Total missing feature values:", total_missing)

print("\nDiagnosis finished.")


# --------------------------------------------------
# 15. CHECK A WIDER 6-HOUR WINDOW
# --------------------------------------------------

print("\n==============================")
print("WIDER 2024 TIME CHECK")
print("==============================")


wide_query = (
    "hmi.sharp_cea_720s"
    "[12519]"
    "[2024.12.30_20:00:00_TAI/6h@12m]"
)


wide_keys = ",".join(
    [
        "T_REC",
        "HARPNUM",
        "NOAA_AR",
        "QUALITY",
        "CMASK",
    ]
    + SHARP_FEATURES
)


wide_df = client.query(
    wide_query,
    key=wide_keys
)


wide_df["missing_count"] = (
    wide_df[SHARP_FEATURES]
    .isna()
    .sum(axis=1)
)


print("\nTotal rows:", len(wide_df))

print("\nTime / Quality / CMASK / Missing:")

print(
    wide_df[
        [
            "T_REC",
            "QUALITY",
            "CMASK",
            "missing_count",
        ]
    ].to_string(index=False)
)


print("\nRows with missing features:")

problem_rows = wide_df[
    wide_df["missing_count"] > 0
]

print(
    problem_rows[
        [
            "T_REC",
            "QUALITY",
            "CMASK",
            "missing_count",
        ]
    ].to_string(index=False)
)