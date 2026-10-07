import drms
import pandas as pd


client = drms.Client()

SERIES = "hmi.sharp_cea_720s"


TESTS = [
    {
        "harp": 12456,
        "date": "2024.12.30_00:00:00_TAI",
    },
    {
        "harp": 12471,
        "date": "2024.12.30_00:00:00_TAI",
    },
]


for test in TESTS:

    print("\n====================================")
    print("HARPNUM:", test["harp"])
    print("====================================")

    query = (
        f"{SERIES}"
        f"[{test['harp']}]"
        f"[{test['date']}/24h@12m]"
    )

    keys = ",".join(
        [
            "T_REC",
            "HARPNUM",
            "QUALITY",
            "CMASK",
            "MEANSHR",
            "ERRMSHA",
            "SHRGT45",
            "LON_FWT",
            "LAT_FWT",
        ]
    )

    df = client.query(
        query,
        key=keys
    )


    # Only rows where MEANSHR is missing
    problem_rows = df[
        df["MEANSHR"].isna()
    ]


    print("MEANSHR missing rows:", len(problem_rows))


    if len(problem_rows) == 0:
        print("No missing MEANSHR rows.")
        continue


    print(
        problem_rows[
            [
                "T_REC",
                "QUALITY",
                "CMASK",
                "MEANSHR",
                "ERRMSHA",
                "SHRGT45",
                "LON_FWT",
                "LAT_FWT",
            ]
        ].to_string(index=False)
    )


print("\nDiagnosis finished.")