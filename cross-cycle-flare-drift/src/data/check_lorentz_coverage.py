import drms


client = drms.Client()


# --------------------------------------------------
# TEST CASES
# --------------------------------------------------

tests = [
    {
        "name": "Cycle 24 - 2011",
        "harp": 401,
        "time": "2011.03.09_23:00:00_TAI",
    },

    {
        "name": "Cycle 25 - 2020",
        "harp": 7451,
        "time": "2020.09.27_00:00:00_TAI",
    },

    {
        "name": "Cycle 25 - 2024",
        "harp": 12519,
        "time": "2024.12.30_22:24:00_TAI",
    },
]


# --------------------------------------------------
# CHECK EACH TEST
# --------------------------------------------------

for test in tests:

    print("\n----------------------------------")
    print(test["name"])
    print("----------------------------------")

    query = (
        f"cgem.lorentz"
        f"[{test['harp']}]"
        f"[{test['time']}/1h@12m]"
    )

    try:

        df = client.query(
            query,
            key="T_REC,HARPNUM,TOTBSQ,TOTFX,TOTFY,TOTFZ,EPSX,EPSY,EPSZ"
        )

        print("HARPNUM:", test["harp"])
        print("Rows found:", len(df))

        if len(df) > 0:

            print("STATUS: AVAILABLE ✅")

            print(
                df[
                    [
                        "T_REC",
                        "HARPNUM",
                        "TOTBSQ",
                    ]
                ].head()
            )

        else:

            print("STATUS: NO DATA ⚠️")

    except Exception as error:

        print("STATUS: ERROR ❌")
        print(error)


print("\nCoverage test finished.")