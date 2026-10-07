import drms


client = drms.Client()

SERIES = "hmi.sharp_cea_720s"


print("Searching SHARP keys related to MEANSHR / SHR...\n")

keys = client.keys(SERIES)

related_keys = [
    key
    for key in keys
    if (
        "SHR" in key.upper()
        or "MSH" in key.upper()
    )
]


for key in sorted(related_keys):
    print(key)


print("\nTotal related keys:", len(related_keys))