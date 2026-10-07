import drms


client = drms.Client()

SERIES = "hmi.sharp_cea_720s"


print("Finding active HARPs in 2024...\n")


query = (
    "hmi.sharp_cea_720s"
    "[][2024.12.30_12:00:00_TAI]"
)


try:

    df = client.query(
        query,
        key=[
            "T_REC",
            "HARPNUM",
            "NOAA_AR",
            "QUALITY",
            "CMASK",
            "LON_FWT",
            "LAT_FWT",
        ],
    )

    print("Total HARPs found:", len(df))

    print("\nAvailable HARPs:")

    print(
        df[
            [
                "HARPNUM",
                "NOAA_AR",
                "QUALITY",
                "CMASK",
                "LON_FWT",
                "LAT_FWT",
            ]
        ].to_string(index=False)
    )

except Exception as error:

    print("ERROR:")
    print(error)