from pathlib import Path
from collections import defaultdict, deque
import pandas as pd


# ============================================================
# PATHS
# ============================================================

EDGE_FILE = Path(
    "data_manifests/harp_noaa_edges.csv"
)

GROUP_FILE = Path(
    "data_manifests/harp_noaa_entity_groups.csv"
)

SUMMARY_FILE = Path(
    "data_manifests/harp_noaa_entity_group_summary.csv"
)


# ============================================================
# LOAD EDGES
# ============================================================

if not EDGE_FILE.exists():

    raise FileNotFoundError(
        f"Missing edge file: {EDGE_FILE}"
    )


edges = pd.read_csv(
    EDGE_FILE
)


edges["HARPNUM"] = pd.to_numeric(
    edges["HARPNUM"],
    errors="coerce"
)

edges["NOAA_AR"] = pd.to_numeric(
    edges["NOAA_AR"],
    errors="coerce"
)


edges = edges.dropna(
    subset=[
        "HARPNUM",
        "NOAA_AR",
    ]
)


edges["HARPNUM"] = (
    edges["HARPNUM"]
    .astype(int)
)

edges["NOAA_AR"] = (
    edges["NOAA_AR"]
    .astype(int)
)


edges = (
    edges
    .drop_duplicates()
    .reset_index(drop=True)
)


print(
    "HARP-NOAA edges:",
    len(edges)
)


# ============================================================
# BUILD BIPARTITE GRAPH
# ============================================================

graph = defaultdict(set)


for _, row in edges.iterrows():

    harp_node = (
        "HARP",
        int(row["HARPNUM"])
    )

    noaa_node = (
        "NOAA",
        int(row["NOAA_AR"])
    )


    graph[harp_node].add(
        noaa_node
    )

    graph[noaa_node].add(
        harp_node
    )


print(
    "Graph nodes:",
    len(graph)
)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

visited = set()

components = []


for start_node in graph:

    if start_node in visited:
        continue


    queue = deque(
        [start_node]
    )

    visited.add(
        start_node
    )


    component = []


    while queue:

        node = queue.popleft()

        component.append(
            node
        )


        for neighbor in graph[node]:

            if neighbor not in visited:

                visited.add(
                    neighbor
                )

                queue.append(
                    neighbor
                )


    components.append(
        component
    )


# ============================================================
# SORT COMPONENTS DETERMINISTICALLY
# ============================================================

def component_sort_key(component):

    harps = sorted(
        value
        for node_type, value
        in component
        if node_type == "HARP"
    )

    noaas = sorted(
        value
        for node_type, value
        in component
        if node_type == "NOAA"
    )

    return (
        harps[0]
        if harps
        else 10**12,

        noaas[0]
        if noaas
        else 10**12,
    )


components = sorted(
    components,
    key=component_sort_key
)


print(
    "Connected components:",
    len(components)
)


# ============================================================
# ASSIGN ENTITY GROUP IDS
# ============================================================

group_rows = []
summary_rows = []


for index, component in enumerate(
    components,
    start=1
):

    group_id = (
        f"ENTITY_{index:05d}"
    )


    harps = sorted(
        value
        for node_type, value
        in component
        if node_type == "HARP"
    )


    noaas = sorted(
        value
        for node_type, value
        in component
        if node_type == "NOAA"
    )


    for harp in harps:

        group_rows.append(
            {
                "ENTITY_GROUP_ID":
                    group_id,

                "node_type":
                    "HARP",

                "node_id":
                    harp,

                "HARPNUM":
                    harp,

                "NOAA_AR":
                    pd.NA,
            }
        )


    for noaa in noaas:

        group_rows.append(
            {
                "ENTITY_GROUP_ID":
                    group_id,

                "node_type":
                    "NOAA",

                "node_id":
                    noaa,

                "HARPNUM":
                    pd.NA,

                "NOAA_AR":
                    noaa,
            }
        )


    summary_rows.append(
        {
            "ENTITY_GROUP_ID":
                group_id,

            "harp_count":
                len(harps),

            "noaa_count":
                len(noaas),

            "node_count":
                len(component),

            "HARPNUMS":
                ",".join(
                    map(
                        str,
                        harps
                    )
                ),

            "NOAA_ARS":
                ",".join(
                    map(
                        str,
                        noaas
                    )
                ),
        }
    )


groups_df = pd.DataFrame(
    group_rows
)


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n===================================="
)

print(
    "ENTITY GROUP VALIDATION"
)

print(
    "===================================="
)


harp_rows = groups_df[
    groups_df[
        "node_type"
    ] == "HARP"
]


noaa_rows = groups_df[
    groups_df[
        "node_type"
    ] == "NOAA"
]


print(
    "Entity groups:",
    len(summary_df)
)


print(
    "HARPs assigned:",
    len(harp_rows)
)


print(
    "NOAA ARs assigned:",
    len(noaa_rows)
)


duplicate_harps = int(
    harp_rows[
        "node_id"
    ]
    .duplicated()
    .sum()
)


duplicate_noaas = int(
    noaa_rows[
        "node_id"
    ]
    .duplicated()
    .sum()
)


print(
    "HARPs assigned to >1 group:",
    duplicate_harps
)


print(
    "NOAA ARs assigned to >1 group:",
    duplicate_noaas
)


# ============================================================
# GROUP SIZE DISTRIBUTION
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "GROUP SIZE SUMMARY"
)

print(
    "------------------------------------"
)


print(
    "Single-HARP groups:",
    int(
        (
            summary_df[
                "harp_count"
            ] == 1
        ).sum()
    )
)


print(
    "Multi-HARP groups:",
    int(
        (
            summary_df[
                "harp_count"
            ] > 1
        ).sum()
    )
)


print(
    "Single-NOAA groups:",
    int(
        (
            summary_df[
                "noaa_count"
            ] == 1
        ).sum()
    )
)


print(
    "Multi-NOAA groups:",
    int(
        (
            summary_df[
                "noaa_count"
            ] > 1
        ).sum()
    )
)


print(
    "Largest HARP count:",
    int(
        summary_df[
            "harp_count"
        ].max()
    )
)


print(
    "Largest NOAA count:",
    int(
        summary_df[
            "noaa_count"
        ].max()
    )
)


print(
    "\nLargest entity groups:"
)


largest = (
    summary_df
    .sort_values(
        [
            "node_count",
            "harp_count",
            "noaa_count",
        ],
        ascending=False
    )
    .head(15)
)


print(
    largest.to_string(
        index=False
    )
)


# ============================================================
# KNOWN SAMPLE CHECK
# ============================================================

print(
    "\n------------------------------------"
)

print(
    "KNOWN CONNECTIVITY CHECK"
)

print(
    "------------------------------------"
)


def find_harp_group(harpnum):

    result = harp_rows[
        harp_rows[
            "HARPNUM"
        ] == harpnum
    ]


    if len(result) == 0:
        return None


    return result.iloc[0][
        "ENTITY_GROUP_ID"
    ]


for harp in [
    38,
    45,
    401,
    12519,
]:

    print(
        f"HARP {harp}:",
        find_harp_group(
            harp
        )
    )


group_38 = find_harp_group(
    38
)

group_45 = find_harp_group(
    45
)


if (
    group_38 is not None
    and
    group_38 == group_45
):

    print(
        "PASS: HARP 38 and HARP 45 "
        "are in the same entity group."
    )

else:

    print(
        "CHECK REQUIRED: "
        "HARP 38 and HARP 45 "
        "should connect through NOAA 11073."
    )


# ============================================================
# SAVE
# ============================================================

GROUP_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


groups_df.to_csv(
    GROUP_FILE,
    index=False
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


print(
    "\nSaved:"
)

print(
    GROUP_FILE
)

print(
    SUMMARY_FILE
)


print(
    "\nConnected entity-group build finished."
)