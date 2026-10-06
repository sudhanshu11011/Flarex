import drms


SERIES = "cgem.lorentz"

client = drms.Client()

FEATURES = [
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
]


print("Checking cgem.lorentz features...\n")

available_keys = client.keys(SERIES)

missing_features = []


for feature in FEATURES:

    if feature in available_keys:
        print("[OK]", feature)

    else:
        print("[MISSING]", feature)
        missing_features.append(feature)


print("\nTotal Lorentz features:", len(FEATURES))
print("Missing:", len(missing_features))


if not missing_features:
    print("\nSUCCESS: All 7 Lorentz features are available.")
else:
    print("\nMissing list:")
    print(missing_features)



print("\nPrime keys of cgem.lorentz:")
print(client.pkeys(SERIES))