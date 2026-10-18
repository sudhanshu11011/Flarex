from pathlib import Path

import pandas as pd


RAW_ROOT = Path("data/raw/jsoc")

DETAIL_OUTPUT = Path(
    "data_manifests/pre_model_noaa_keyword_temporal_behavior.csv"
)

SUMMARY_OUTPUT = Path(
    "data_manifests/pre_model_noaa_keyword_temporal_summary.csv"
)


print(
    "===================================="
)

print(
    "PRE-MODEL FIX B — NOAA KEYWORD TEMPORAL AUDIT"
)

print(
    "===================================="
)


frames = []


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


    if not files:

        raise RuntimeError(
            f"No SHARP files found for {cycle}"
        )


    for index, file in enumerate(
        files,
        start=1
    ):

        df = pd.read_parquet(
            file,
            columns=[
                "HARPNUM",
                "T_REC",
                "NOAA_AR",
                "NOAA_ARS",
            ],
        )


        df[
            "cycle"
        ] = cycle


        frames.append(
            df
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


data = pd.concat(
    frames,
    ignore_index=True
)


data[
    "HARPNUM"
] = pd.to_numeric(
    data[
        "HARPNUM"
    ],
    errors="coerce"
)


data = data[
    data[
        "HARPNUM"
    ]
    .notna()
].copy()


data[
    "HARPNUM"
] = data[
    "HARPNUM"
].astype(int)


# ============================================================
# NORMALIZE KEYWORD STRINGS
# ============================================================

def normalize_keyword(value):

    if pd.isna(value):
        return "<MISSING>"

    text = str(
        value
    ).strip()

    if (
        text == ""
        or
        text.lower()
        in {
            "nan",
            "none",
        }
    ):

        return "<MISSING>"

    return text


data[
    "_NOAA_AR"
] = data[
    "NOAA_AR"
].apply(
    normalize_keyword
)


data[
    "_NOAA_ARS"
] = data[
    "NOAA_ARS"
].apply(
    normalize_keyword
)


# ============================================================
# HARP-LEVEL TEMPORAL AUDIT
# ============================================================

detail_rows = []


for (
    cycle,
    harpnum,
), group in data.groupby(
    [
        "cycle",
        "HARPNUM",
    ]
):

    group = group.sort_values(
        "T_REC"
    )


    noaa_ar_values = (
        group[
            "_NOAA_AR"
        ]
        .drop_duplicates()
        .tolist()
    )


    noaa_ars_values = (
        group[
            "_NOAA_ARS"
        ]
        .drop_duplicates()
        .tolist()
    )


    noaa_ar_nonmissing = [
        value

        for value in noaa_ar_values

        if value != "<MISSING>"
    ]


    noaa_ars_nonmissing = [
        value

        for value in noaa_ars_values

        if value != "<MISSING>"
    ]


    detail_rows.append(
        {
            "cycle":
                cycle,

            "HARPNUM":
                harpnum,

            "rows":
                len(group),

            "first_trec":
                group[
                    "T_REC"
                ].iloc[0],

            "last_trec":
                group[
                    "T_REC"
                ].iloc[-1],

            "unique_NOAA_AR_values":
                len(
                    noaa_ar_nonmissing
                ),

            "unique_NOAA_ARS_values":
                len(
                    noaa_ars_nonmissing
                ),

            "NOAA_AR_changed_over_time":
                len(
                    noaa_ar_nonmissing
                )
                >
                1,

            "NOAA_ARS_changed_over_time":
                len(
                    noaa_ars_nonmissing
                )
                >
                1,

            "NOAA_AR_values":
                " | ".join(
                    noaa_ar_values
                ),

            "NOAA_ARS_values":
                " | ".join(
                    noaa_ars_values
                ),
        }
    )


detail = pd.DataFrame(
    detail_rows
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for cycle in [
    "cycle24",
    "cycle25",
]:

    subset = detail[
        detail[
            "cycle"
        ]
        ==
        cycle
    ].copy()


    total_harps = len(
        subset
    )


    ar_changed = int(
        subset[
            "NOAA_AR_changed_over_time"
        ]
        .sum()
    )


    ars_changed = int(
        subset[
            "NOAA_ARS_changed_over_time"
        ]
        .sum()
    )


    ar_static = (
        total_harps
        -
        ar_changed
    )


    ars_static = (
        total_harps
        -
        ars_changed
    )


    multi_noaa_ars = int(
        (
            subset[
                "NOAA_ARS_values"
            ]
            .str.contains(
                ",",
                regex=False,
                na=False
            )
        )
        .sum()
    )


    multi_and_static = int(
        (
            subset[
                "NOAA_ARS_values"
            ]
            .str.contains(
                ",",
                regex=False,
                na=False
            )
            &
            (
                ~subset[
                    "NOAA_ARS_changed_over_time"
                ]
            )
        )
        .sum()
    )


    summary_rows.append(
        {
            "cycle":
                cycle,

            "total_harps":
                total_harps,

            "NOAA_AR_changed_harps":
                ar_changed,

            "NOAA_AR_static_harps":
                ar_static,

            "NOAA_ARS_changed_harps":
                ars_changed,

            "NOAA_ARS_static_harps":
                ars_static,

            "multi_region_NOAA_ARS_harps":
                multi_noaa_ars,

            "multi_region_but_static_NOAA_ARS_harps":
                multi_and_static,

            "NOAA_AR_changed_pct":
                (
                    ar_changed
                    /
                    total_harps
                    *
                    100
                    if total_harps
                    else 0
                ),

            "NOAA_ARS_changed_pct":
                (
                    ars_changed
                    /
                    total_harps
                    *
                    100
                    if total_harps
                    else 0
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SPECIAL HARP 13354
# ============================================================

target = detail[
    detail[
        "HARPNUM"
    ]
    ==
    13354
]


print(
    "\n===================================="
)

print(
    "TEMPORAL KEYWORD SUMMARY"
)

print(
    "===================================="
)


print(
    summary.to_string(
        index=False
    )
)


print(
    "\n===================================="
)

print(
    "HARP 13354 CHECK"
)

print(
    "===================================="
)


if len(
    target
) > 0:

    print(
        target.to_string(
            index=False
        )
    )

else:

    print(
        "HARP 13354 not found."
    )


# ============================================================
# SAVE
# ============================================================

DETAIL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


detail.to_csv(
    DETAIL_OUTPUT,
    index=False
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


print(
    "\nSaved:"
)

print(
    DETAIL_OUTPUT
)

print(
    SUMMARY_OUTPUT
)


print(
    "\nPre-model Fix B audit complete."
)