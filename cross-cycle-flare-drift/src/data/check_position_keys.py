import drms


SERIES = "hmi.sharp_cea_720s"

client = drms.Client()

print("Checking position-related SHARP keywords...\n")

available_keys = client.keys(SERIES)

keywords = [
    key
    for key in available_keys
    if (
        "LAT" in key.upper()
        or "LON" in key.upper()
        or "CRLN" in key.upper()
        or "CRLT" in key.upper()
    )
]

for key in sorted(keywords):
    print(key)

print("\nTotal position-related keys:", len(keywords))