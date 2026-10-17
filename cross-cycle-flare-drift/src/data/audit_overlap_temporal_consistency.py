from pathlib import Path
import re

import pandas as pd


RAW_ROOT = Path(
    "data/raw/jsoc"
)

EDGE_FILE = Path(
    "data_manifests/harp_noaa_edges.csv"
)

OUTPUT_FILE = Path(
    "data_manifests/phase11_overlap_temporal_consistency.csv"
)


TARGET_HARPS = [
    219,
    13354,
]


# ============================================================
# NOAA PARSER
# ============================================================

def parse_noaa_values(
    noaa_ar,
    noaa_ars
):

    values = set()

    for value in [
        noaa_ar,
        noaa_ars,
    ]:

        if pd.isna(value):
            continue

        for match in re.findall(
            r"\d+",
            str(value)
        ):

            number = int(
                match
            )

            if number > 0:

                values.add(
                    number
                )

    return sorted(
        values
    )


# ============================================================
# LOAD ALL SHARP RECORDS FOR TARGET HARPS
# ============================================================

records = []


for cycle in [
    "cycle24",
    "cycle25",
]:

    directory = (
        RAW_ROOT
        /
        cycle
        /
        "sharp"
    )


    files = sorted(
        directory.glob(
            "*.parquet"
        )
    )


    print(
        cycle,
        "files:",
        len(files)
    )


    for index, file in enumerate(
        files,
        start=1
    ):

        df = pd.read_parquet(
            file,
            columns=[
                "T_REC",
                "HARPNUM",
                "NOAA_AR",
                "NOAA_ARS",
            ],
        )


        df[
            "HARPNUM"
        ] = pd.to_numeric(
            df[
                "HARPNUM"
            ],
            errors="coerce"
        )


        target = df[
            df[
                "HARPNUM"
            ].isin(
                TARGET_HARPS
            )
        ].copy()


        if len(
            target
        ) > 0:

            target[
                "cycle"
            ] = cycle


            target[
                "source_file"
            ] = file.name


            records.append(
                target
            )


        if (
            index % 30 == 0
            or
            index == len(files)
        ):

            print(
                cycle,
                index,
                "/",
                len(files)
            )


if not records:

    raise RuntimeError(
        "No target HARP records found."
    )


data = pd.concat(
    records,
    ignore_index=True
)


# ============================================================
# PARSE ASSOCIATIONS
# ============================================================

data[
    "associated_noaa"
] = [

    parse_noaa_values(
        noaa_ar,
        noaa_ars
    )

    for noaa_ar, noaa_ars
    in zip(
        data[
            "NOAA_AR"
        ],
        data[
            "NOAA_ARS"
        ],
    )
]


data[
    "associated_noaa_string"
] = [

    ",".join(
        str(value)
        for value in values
    )

    for values
    in data[
        "associated_noaa"
    ]
]


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


print(
    "\n===================================="
)

print(
    "PHASE 11D — TEMPORAL CONSISTENCY"
)

print(
    "===================================="
)


for harpnum in TARGET_HARPS:

    subset = data[
        data[
            "HARPNUM"
        ]
        ==
        harpnum
    ].copy()


    if len(
        subset
    ) == 0:

        print(
            "\nHARP",
            harpnum,
            "NOT FOUND"
        )

        continue


    all_noaa = sorted(
        {
            region

            for values in subset[
                "associated_noaa"
            ]

            for region in values
        }
    )


    noaa_11114_rows = int(
        subset[
            "associated_noaa"
        ]
        .apply(
            lambda values:
                11114 in values
        )
        .sum()
    )


    noaa_14114_rows = int(
        subset[
            "associated_noaa"
        ]
        .apply(
            lambda values:
                14114 in values
        )
        .sum()
    )


    print(
        "\n------------------------------------"
    )

    print(
        "HARP:",
        harpnum
    )

    print(
        "------------------------------------"
    )


    print(
        "Rows:",
        len(
            subset
        )
    )


    print(
        "Cycles:",
        sorted(
            subset[
                "cycle"
            ].unique()
        )
    )


    print(
        "First T_REC:",
        subset[
            "T_REC"
        ].min()
    )


    print(
        "Last T_REC:",
        subset[
            "T_REC"
        ].max()
    )


    print(
        "Unique contemporaneous NOAA:",
        all_noaa
    )


    print(
        "Rows containing NOAA 11114:",
        noaa_11114_rows
    )


    print(
        "Rows containing NOAA 14114:",
        noaa_14114_rows
    )


    print(
        "\nUnique NOAA_AR values:"
    )


    print(
        sorted(
            subset[
                "NOAA_AR"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )


    print(
        "\nUnique NOAA_ARS values:"
    )


    print(
        sorted(
            subset[
                "NOAA_ARS"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )


    summary_rows.append(
        {
            "HARPNUM":
                harpnum,

            "rows":
                len(
                    subset
                ),

            "cycles":
                ",".join(
                    sorted(
                        subset[
                            "cycle"
                        ].unique()
                    )
                ),

            "first_trec":
                subset[
                    "T_REC"
                ].min(),

            "last_trec":
                subset[
                    "T_REC"
                ].max(),

            "unique_contemporaneous_noaa":
                ",".join(
                    map(
                        str,
                        all_noaa
                    )
                ),

            "rows_with_11114":
                noaa_11114_rows,

            "rows_with_14114":
                noaa_14114_rows,
        }
    )


# ============================================================
# CANONICAL EDGE COMPARISON
# ============================================================

edges = pd.read_csv(
    EDGE_FILE
)


edges[
    "HARPNUM"
] = pd.to_numeric(
    edges[
        "HARPNUM"
    ],
    errors="coerce"
)


edges[
    "NOAA_AR"
] = pd.to_numeric(
    edges[
        "NOAA_AR"
    ],
    errors="coerce"
)


target_edges = edges[
    edges[
        "HARPNUM"
    ].isin(
        TARGET_HARPS
    )
].copy()


print(
    "\n===================================="
)

print(
    "CANONICAL EDGE COMPARISON"
)

print(
    "===================================="
)


print(
    target_edges.to_string(
        index=False
    )
)


# ============================================================
# DECISION SIGNALS
# ============================================================

harp219 = data[
    data[
        "HARPNUM"
    ]
    ==
    219
]


harp13354 = data[
    data[
        "HARPNUM"
    ]
    ==
    13354
]


h219_has_11114 = bool(
    harp219[
        "associated_noaa"
    ]
    .apply(
        lambda values:
            11114 in values
    )
    .any()
)


h13354_has_11114 = bool(
    harp13354[
        "associated_noaa"
    ]
    .apply(
        lambda values:
            11114 in values
    )
    .any()
)


h13354_has_14114 = bool(
    harp13354[
        "associated_noaa"
    ]
    .apply(
        lambda values:
            14114 in values
    )
    .any()
)


print(
    "\n===================================="
)

print(
    "DECISION SIGNALS"
)

print(
    "===================================="
)


print(
    "HARP 219 contemporaneously has NOAA 11114:",
    h219_has_11114
)


print(
    "HARP 13354 contemporaneously has NOAA 11114:",
    h13354_has_11114
)


print(
    "HARP 13354 contemporaneously has NOAA 14114:",
    h13354_has_14114
)


if (
    h219_has_11114
    and
    not h13354_has_11114
    and
    h13354_has_14114
):

    decision = (
        "SUSPECT_CANONICAL_EDGE_13354_TO_11114"
    )

else:

    decision = (
        "REQUIRES_FURTHER_INVESTIGATION"
    )


print(
    "Decision signal:",
    decision
)


# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


summary[
    "decision_signal"
] = decision


summary.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\nPhase 11D complete."
)