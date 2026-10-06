import drms


SERIES = "hmi.sharp_cea_720s"

client = drms.Client()


# 25 features used in the Bobra-Couvidat flare forecasting setup
FEATURES = [
    "TOTUSJH",
    "TOTBSQ",
    "TOTPOT",
    "TOTUSJZ",
    "ABSNJZH",
    "SAVNCPP",
    "USFLUX",
    "AREA_ACR",
    "TOTFZ",
    "MEANPOT",
    "R_VALUE",
    "EPSZ",
    "SHRGT45",
    "MEANSHR",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "MEANJZH",
    "TOTFY",
    "MEANJZD",
    "MEANALP",
    "TOTFX",
    "EPSY",
    "EPSX",
]


print("Checking SHARP features...\n")

available_keys = client.keys(SERIES)

missing_features = []


for feature in FEATURES:

    if feature in available_keys:
        print("[OK]", feature)

    else:
        print("[MISSING]", feature)
        missing_features.append(feature)


print("\nTotal features:", len(FEATURES))
print("Missing features:", len(missing_features))


if missing_features:
    print("\nMissing list:")
    print(missing_features)

else:
    print("\nSUCCESS: All 25 features are available.")