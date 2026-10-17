from pathlib import Path
from collections import defaultdict, deque
import re

import pandas as pd


TARGET_GROUP = "ENTITY_00049"

EDGE_FILE = Path(
    "data_manifests/harp_noaa_edges.csv"
)

GROUP_FILE = Path(
    "data_manifests/harp_noaa_entity_groups.csv"
)

RAW_MAPPING_FILE = Path(
    "data/raw/mappings/all_harps_with_noaa_ars.txt"
)

OUTPUT_FILE = Path(
    "data_manifests/phase11_entity_overlap_edge_trace.csv"
)


# ============================================================
# LOAD
# ============================================================

edges = pd.read_csv(
    EDGE_FILE
)

groups = pd.read_csv(
    GROUP_FILE
)


edges["HARPNUM"] = pd.to_numeric(
    edges["HARPNUM"],
    errors="coerce"
).astype("Int64")


edges["NOAA_AR"] = pd.to_numeric(
    edges["NOAA_AR"],
    errors="coerce"
).astype("Int64")


target = groups[
    groups["ENTITY_GROUP_ID"].astype(str)
    ==
    TARGET_GROUP
].copy()


if len(target) == 0:

    raise RuntimeError(
        f"{TARGET_GROUP} not found."
    )


# ============================================================
# MEMBERS
# ============================================================

harps = sorted(
    pd.to_numeric(
        target["HARPNUM"],
        errors="coerce"
    )
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)


noaas = sorted(
    pd.to_numeric(
        target["NOAA_AR"],
        errors="coerce"
    )
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)


print(
    "===================================="
)

print(
    "PHASE 11C — ENTITY EDGE TRACE"
)

print(
    "===================================="
)


print(
    "Entity:",
    TARGET_GROUP
)


print(
    "HARPs:",
    harps
)


print(
    "NOAA ARs:",
    noaas
)


# ============================================================
# ALL EDGES TOUCHING THESE MEMBERS
# ============================================================

touching = edges[
    edges["HARPNUM"].isin(harps)
    |
    edges["NOAA_AR"].isin(noaas)
].copy()


touching = (
    touching
    .sort_values(
        [
            "HARPNUM",
            "NOAA_AR",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "\n===================================="
)

print(
    "EDGES TOUCHING TARGET MEMBERS"
)

print(
    "===================================="
)


print(
    touching.to_string(
        index=False
    )
)


# ============================================================
# INTERNAL COMPONENT EDGES
# ============================================================

internal = touching[
    touching["HARPNUM"].isin(harps)
    &
    touching["NOAA_AR"].isin(noaas)
].copy()


print(
    "\n===================================="
)

print(
    "INTERNAL EDGES"
)

print(
    "===================================="
)


print(
    internal.to_string(
        index=False
    )
)


# ============================================================
# SHARED NOAA CHECK
# ============================================================

noaa_degree = (
    internal
    .groupby("NOAA_AR")[
        "HARPNUM"
    ]
    .nunique()
)


shared_noaa = (
    noaa_degree[
        noaa_degree > 1
    ]
)


print(
    "\n===================================="
)

print(
    "SHARED NOAA BRIDGES"
)

print(
    "===================================="
)


if len(shared_noaa) == 0:

    print(
        "No NOAA region links more than one HARP."
    )

else:

    print(
        shared_noaa.to_string()
    )


# ============================================================
# REBUILD NAMESPACED GRAPH
#
# Critical:
# HARP and NOAA IDs are different node namespaces.
#
# ("HARP", 11114)
# is NOT the same node as
# ("NOAA", 11114)
# ============================================================

graph = defaultdict(set)


for _, row in internal.iterrows():

    harp = int(
        row["HARPNUM"]
    )

    noaa = int(
        row["NOAA_AR"]
    )


    harp_node = (
        "HARP",
        harp
    )

    noaa_node = (
        "NOAA",
        noaa
    )


    graph[
        harp_node
    ].add(
        noaa_node
    )


    graph[
        noaa_node
    ].add(
        harp_node
    )


all_nodes = set()


for harp in harps:

    all_nodes.add(
        (
            "HARP",
            harp
        )
    )


for noaa in noaas:

    all_nodes.add(
        (
            "NOAA",
            noaa
        )
    )


visited = set()
components = []


for start in sorted(
    all_nodes
):

    if start in visited:
        continue


    queue = deque(
        [start]
    )

    visited.add(
        start
    )

    component = []


    while queue:

        node = queue.popleft()

        component.append(
            node
        )


        for neighbour in graph.get(
            node,
            []
        ):

            if neighbour not in visited:

                visited.add(
                    neighbour
                )

                queue.append(
                    neighbour
                )


    components.append(
        sorted(
            component
        )
    )


print(
    "\n===================================="
)

print(
    "NAMESPACED GRAPH RECONSTRUCTION"
)

print(
    "===================================="
)


print(
    "Connected components found:",
    len(
        components
    )
)


for index, component in enumerate(
    components,
    start=1
):

    print(
        f"\nComponent {index}:"
    )

    for node in component:

        print(
            " ",
            node
        )


# ============================================================
# SPECIFIC HARP CONNECTION
# ============================================================

if (
    ("HARP", 219)
    in
    all_nodes
    and
    ("HARP", 13354)
    in
    all_nodes
):

    start = (
        "HARP",
        219
    )

    target_node = (
        "HARP",
        13354
    )


    queue = deque(
        [start]
    )

    parent = {
        start: None
    }


    while queue:

        node = queue.popleft()


        if node == target_node:

            break


        for neighbour in graph.get(
            node,
            []
        ):

            if neighbour not in parent:

                parent[
                    neighbour
                ] = node

                queue.append(
                    neighbour
                )


    print(
        "\n===================================="
    )

    print(
        "PATH HARP 219 → HARP 13354"
    )

    print(
        "===================================="
    )


    if target_node not in parent:

        print(
            "NO PATH in correctly namespaced graph."
        )

    else:

        path = []

        current = target_node


        while current is not None:

            path.append(
                current
            )

            current = parent[
                current
            ]


        path.reverse()


        for node in path:

            print(
                node
            )


# ============================================================
# RAW SOURCE LINES
# ============================================================

print(
    "\n===================================="
)

print(
    "RAW MAPPING SOURCE MATCHES"
)

print(
    "===================================="
)


if RAW_MAPPING_FILE.exists():

    target_numbers = (
        harps
        +
        noaas
    )


    patterns = {
        number:
            re.compile(
                rf"(?<!\d){number}(?!\d)"
            )

        for number in target_numbers
    }


    matches = []


    with RAW_MAPPING_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            matched_numbers = [
                number

                for number, pattern
                in patterns.items()

                if pattern.search(
                    line
                )
            ]


            if matched_numbers:

                matches.append(
                    (
                        line_number,
                        matched_numbers,
                        line.rstrip()
                    )
                )


    for (
        line_number,
        matched_numbers,
        line
    ) in matches:

        print(
            f"Line {line_number} "
            f"{matched_numbers}: "
            f"{line}"
        )

else:

    print(
        "Raw mapping file not found."
    )


# ============================================================
# SAVE TRACE
# ============================================================

touching[
    "target_entity_group"
] = TARGET_GROUP


touching.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n===================================="
)

print(
    "TRACE DECISION DATA"
)

print(
    "===================================="
)


print(
    "Target HARP count:",
    len(
        harps
    )
)


print(
    "Target NOAA count:",
    len(
        noaas
    )
)


print(
    "Internal edge count:",
    len(
        internal
    )
)


print(
    "Shared NOAA bridge count:",
    len(
        shared_noaa
    )
)


print(
    "Correctly namespaced components:",
    len(
        components
    )
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\nPhase 11C trace complete."
)