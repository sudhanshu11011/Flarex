from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROTOCOL_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "shift_null_control_protocol.json"
)

FEATURE_REGISTRY_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "shift_feature_registry.csv"
)

CYCLE24_TRAIN_FILE = Path(
    "artifacts/splitting/data/"
    "cycle24_train.parquet"
)


OUTPUT_CORRELATION_FILE = Path(
    "artifacts/models/shift_controls/metrics/"
    "cycle24_train_spearman_correlation.csv"
)

OUTPUT_PAIRS_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "correlated_feature_pairs.csv"
)

OUTPUT_GROUPS_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "correlated_feature_groups.csv"
)

OUTPUT_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "correlated_feature_groups_contract.json"
)


# ============================================================
# FROZEN EXPECTATIONS
# ============================================================

EXPECTED_TRAIN_SHA256 = (
    "4801a6e3b11d39cef8d4b0ef0eea4ae1"
    "67c27862431662c3f43ff55e4c433def"
)

EXPECTED_TRAIN_ROWS = 62064

EXPECTED_TRAIN_GROUPS = 822

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"

CORRELATION_METHOD = "spearman"

ABS_RHO_THRESHOLD = 0.80


FEATURES = [
    "TOTUSJH",
    "TOTPOT",
    "TOTUSJZ",
    "ABSNJZH",
    "SAVNCPP",
    "USFLUX",
    "AREA_ACR",
    "MEANPOT",
    "R_VALUE",
    "SHRGT45",
    "MEANSHR",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "MEANJZH",
    "MEANJZD",
    "MEANALP",
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
]


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self, items):

        self.parent = {
            item: item
            for item in items
        }

        self.rank = {
            item: 0
            for item in items
        }


    def find(self, item):

        if self.parent[item] != item:

            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]


    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if (
            self.rank[root_a]
            <
            self.rank[root_b]
        ):

            self.parent[root_a] = root_b

        elif (
            self.rank[root_a]
            >
            self.rank[root_b]
        ):

            self.parent[root_b] = root_a

        else:

            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# START
# ============================================================

print(
    "===================================="
)

print(
    "MODEL CHAPTER 9.5 — CORRELATED SHARP FEATURE GROUPS"
)

print(
    "===================================="
)


# ============================================================
# REQUIRED INPUTS
# ============================================================

required_files = [
    PROTOCOL_FILE,
    FEATURE_REGISTRY_FILE,
    CYCLE24_TRAIN_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


# ============================================================
# PROTECT OUTPUTS
# ============================================================

for path in [
    OUTPUT_CORRELATION_FILE,
    OUTPUT_PAIRS_FILE,
    OUTPUT_GROUPS_FILE,
    OUTPUT_CONTRACT_FILE,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter 9.5 results."
        )


# ============================================================
# LOAD + VERIFY CHAPTER 9.1 PROTOCOL
# ============================================================

protocol = load_json(
    PROTOCOL_FILE
)


if protocol.get(
    "status"
) != "FROZEN_BEFORE_SHIFT_AND_SHAP_ANALYSIS":

    raise RuntimeError(
        "Chapter 9.1 protocol is not frozen."
    )


correlation_protocol = protocol[
    "correlated_feature_control"
]


if correlation_protocol[
    "source_population"
] != "CYCLE24_TRAIN_ONLY":

    raise RuntimeError(
        "Frozen correlation source "
        "population mismatch."
    )


if correlation_protocol[
    "correlation"
].lower() != CORRELATION_METHOD:

    raise RuntimeError(
        "Frozen correlation method mismatch."
    )


if correlation_protocol[
    "edge_rule"
] != "absolute_rho >= 0.80":

    raise RuntimeError(
        "Frozen correlation edge rule mismatch."
    )


if correlation_protocol[
    "cycle25_used_to_form_groups"
]:

    raise RuntimeError(
        "Frozen protocol incorrectly allows "
        "Cycle25 in correlation grouping."
    )


if protocol[
    "feature_contract"
][
    "feature_order"
] != FEATURES:

    raise RuntimeError(
        "Frozen feature order mismatch."
    )


print(
    "Chapter 9.1 protocol:",
    "FROZEN"
)

print(
    "Correlation source:",
    "CYCLE24_TRAIN_ONLY"
)

print(
    "Correlation method:",
    "Spearman"
)

print(
    "Edge rule:",
    "|rho| >= 0.80"
)

print(
    "Frozen feature order:",
    "PASS"
)


# ============================================================
# VERIFY FEATURE REGISTRY
# ============================================================

expected_registry_hash = protocol[
    "artifacts"
][
    "feature_registry"
][
    "sha256"
]


actual_registry_hash = sha256_file(
    FEATURE_REGISTRY_FILE
)


if (
    actual_registry_hash
    !=
    expected_registry_hash
):

    raise RuntimeError(
        "Shift feature-registry "
        "hash mismatch."
    )


feature_registry = pd.read_csv(
    FEATURE_REGISTRY_FILE,
    keep_default_na=False,
)


if (
    feature_registry[
        "feature_name"
    ].astype(str).tolist()
    !=
    FEATURES
):

    raise RuntimeError(
        "Feature registry order mismatch."
    )


print(
    "Shift feature registry:",
    "PASS"
)


# ============================================================
# VERIFY CYCLE24 TRAIN HASH
# ============================================================

train_hash = sha256_file(
    CYCLE24_TRAIN_FILE
)


if train_hash != EXPECTED_TRAIN_SHA256:

    raise RuntimeError(
        "Cycle24 TRAIN hash mismatch."
    )


protocol_train_hash = protocol[
    "dataset_expectations"
][
    "cycle24_train"
][
    "sha256"
]


if train_hash != protocol_train_hash:

    raise RuntimeError(
        "Cycle24 TRAIN hash differs "
        "from Chapter 9.1 protocol."
    )


print(
    "Cycle24 TRAIN hash:",
    "PASS"
)


# ============================================================
# READ CYCLE24 TRAIN ONLY
# ============================================================

train = pd.read_parquet(
    CYCLE24_TRAIN_FILE
)


if len(
    train
) != EXPECTED_TRAIN_ROWS:

    raise RuntimeError(
        "Cycle24 TRAIN row-count mismatch."
    )


required_columns = set(
    FEATURES
    +
    [
        ENTITY_GROUP_COLUMN
    ]
)


missing_columns = (
    required_columns
    -
    set(
        train.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "Cycle24 TRAIN missing columns: "
        f"{sorted(missing_columns)}"
    )


train_group_count = int(
    train[
        ENTITY_GROUP_COLUMN
    ].nunique(
        dropna=False
    )
)


if train_group_count != EXPECTED_TRAIN_GROUPS:

    raise RuntimeError(
        "Cycle24 TRAIN ENTITY_GROUP "
        "count mismatch."
    )


print(
    "Cycle24 TRAIN rows:",
    len(
        train
    )
)

print(
    "Cycle24 TRAIN ENTITY_GROUPs:",
    train_group_count
)

print(
    "Cycle25 read:",
    False
)


# ============================================================
# FEATURE FINITENESS
# ============================================================

feature_matrix = train[
    FEATURES
].to_numpy(
    dtype=np.float64
)


if not np.isfinite(
    feature_matrix
).all():

    raise RuntimeError(
        "Cycle24 TRAIN contains "
        "non-finite feature values."
    )


print(
    "Cycle24 TRAIN feature values:",
    "FINITE"
)


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

print(
    "\n===================================="
)

print(
    "COMPUTING CYCLE24 TRAIN SPEARMAN CORRELATION"
)

print(
    "===================================="
)


correlation_matrix = train[
    FEATURES
].corr(
    method="spearman"
)


if correlation_matrix.shape != (
    25,
    25,
):

    raise RuntimeError(
        "Correlation matrix shape mismatch."
    )


if list(
    correlation_matrix.index
) != FEATURES:

    raise RuntimeError(
        "Correlation matrix row order mismatch."
    )


if list(
    correlation_matrix.columns
) != FEATURES:

    raise RuntimeError(
        "Correlation matrix column "
        "order mismatch."
    )


correlation_values = (
    correlation_matrix.to_numpy(
        dtype=np.float64
    )
)


if not np.isfinite(
    correlation_values
).all():

    raise RuntimeError(
        "Correlation matrix contains "
        "non-finite values."
    )


if not np.allclose(
    correlation_values,
    correlation_values.T,
    rtol=0.0,
    atol=1e-12,
):

    raise RuntimeError(
        "Correlation matrix is not symmetric."
    )


if not np.allclose(
    np.diag(
        correlation_values
    ),
    1.0,
    rtol=0.0,
    atol=1e-12,
):

    raise RuntimeError(
        "Correlation matrix diagonal "
        "is not exactly 1."
    )


print(
    "Correlation matrix shape:",
    correlation_matrix.shape
)

print(
    "Correlation matrix integrity:",
    "PASS"
)


# ============================================================
# SAVE MATRIX
# ============================================================

correlation_output = (
    correlation_matrix
    .reset_index()
    .rename(
        columns={
            "index":
                "feature"
        }
    )
)


OUTPUT_CORRELATION_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


correlation_output.to_csv(
    OUTPUT_CORRELATION_FILE,
    index=False,
)


# ============================================================
# BUILD ALL UNIQUE FEATURE PAIRS
# ============================================================

pair_rows = []

union_find = UnionFind(
    FEATURES
)


for i in range(
    len(
        FEATURES
    )
):

    for j in range(
        i + 1,
        len(
            FEATURES
        )
    ):

        feature_a = FEATURES[
            i
        ]

        feature_b = FEATURES[
            j
        ]


        rho = float(
            correlation_matrix.loc[
                feature_a,
                feature_b,
            ]
        )


        absolute_rho = abs(
            rho
        )


        correlated_edge = bool(
            absolute_rho
            >=
            ABS_RHO_THRESHOLD
        )


        if correlated_edge:

            union_find.union(
                feature_a,
                feature_b,
            )


        pair_rows.append(
            {
                "feature_a_index":
                    i + 1,

                "feature_a":
                    feature_a,

                "feature_b_index":
                    j + 1,

                "feature_b":
                    feature_b,

                "spearman_rho":
                    rho,

                "absolute_spearman_rho":
                    absolute_rho,

                "edge_threshold":
                    ABS_RHO_THRESHOLD,

                "correlated_edge":
                    correlated_edge,
            }
        )


pairs_df = pd.DataFrame(
    pair_rows
)


EXPECTED_PAIR_COUNT = (
    25
    *
    24
    //
    2
)


if len(
    pairs_df
) != EXPECTED_PAIR_COUNT:

    raise RuntimeError(
        "Expected exactly 300 "
        "unique feature pairs."
    )


edge_count = int(
    pairs_df[
        "correlated_edge"
    ].sum()
)


print(
    "Unique feature pairs:",
    len(
        pairs_df
    )
)

print(
    "|rho| >= 0.80 edges:",
    edge_count
)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

components = {}


for feature in FEATURES:

    root = union_find.find(
        feature
    )


    components.setdefault(
        root,
        []
    ).append(
        feature
    )


feature_position = {
    feature:
        index
    for index, feature
    in enumerate(
        FEATURES
    )
}


component_list = list(
    components.values()
)


# Keep member order equal to frozen feature order.
for component in component_list:

    component.sort(
        key=lambda feature:
            feature_position[
                feature
            ]
    )


# Group ordering is deterministic:
# component whose first feature appears earliest
# in the frozen 25-feature list comes first.
component_list.sort(
    key=lambda members:
        min(
            feature_position[
                member
            ]
            for member in members
        )
)


group_rows = []


for group_number, members in enumerate(
    component_list,
    start=1,
):

    group_id = (
        f"CFG_{group_number:02d}"
    )


    member_string = "|".join(
        members
    )


    group_size = len(
        members
    )


    for feature in members:

        group_rows.append(
            {
                "group_id":
                    group_id,

                "group_size":
                    group_size,

                "feature_index":
                    feature_position[
                        feature
                    ]
                    +
                    1,

                "feature":
                    feature,

                "group_members":
                    member_string,

                "is_singleton":
                    bool(
                        group_size
                        ==
                        1
                    ),

                "grouping_source":
                    "CYCLE24_TRAIN_ONLY",

                "correlation_method":
                    "SPEARMAN",

                "edge_rule":
                    "ABS_RHO_GTE_0.80",

                "feature_selection":
                    False,
            }
        )


groups_df = pd.DataFrame(
    group_rows
).sort_values(
    "feature_index"
).reset_index(
    drop=True
)


if len(
    groups_df
) != 25:

    raise RuntimeError(
        "Every frozen feature must "
        "appear exactly once."
    )


if groups_df[
    "feature"
].duplicated().any():

    raise RuntimeError(
        "A feature appears in more "
        "than one correlated group."
    )


if set(
    groups_df[
        "feature"
    ]
) != set(
    FEATURES
):

    raise RuntimeError(
        "Correlated-feature grouping "
        "does not cover all 25 features."
    )


group_count = int(
    groups_df[
        "group_id"
    ].nunique()
)


singleton_count = int(
    groups_df.loc[
        groups_df[
            "is_singleton"
        ],
        "group_id",
    ].nunique()
)


multi_feature_group_count = (
    group_count
    -
    singleton_count
)


largest_group_size = int(
    groups_df[
        "group_size"
    ].max()
)


print(
    "Connected components:",
    group_count
)

print(
    "Singleton groups:",
    singleton_count
)

print(
    "Multi-feature groups:",
    multi_feature_group_count
)

print(
    "Largest group size:",
    largest_group_size
)


# ============================================================
# VERIFY EDGE CONNECTIVITY CONSISTENCY
# ============================================================

feature_to_group = dict(
    zip(
        groups_df[
            "feature"
        ],
        groups_df[
            "group_id"
        ],
    )
)


edge_rows = pairs_df[
    pairs_df[
        "correlated_edge"
    ]
]


for _, row in edge_rows.iterrows():

    feature_a = str(
        row[
            "feature_a"
        ]
    )

    feature_b = str(
        row[
            "feature_b"
        ]
    )


    if (
        feature_to_group[
            feature_a
        ]
        !=
        feature_to_group[
            feature_b
        ]
    ):

        raise RuntimeError(
            "Threshold-crossing edge connects "
            "features assigned to different "
            "connected components."
        )


print(
    "Connected-component integrity:",
    "PASS"
)


# ============================================================
# SAVE MANIFESTS
# ============================================================

OUTPUT_PAIRS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_GROUPS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


pairs_df.to_csv(
    OUTPUT_PAIRS_FILE,
    index=False,
)


groups_df.to_csv(
    OUTPUT_GROUPS_FILE,
    index=False,
)


# ============================================================
# HASH OUTPUTS
# ============================================================

correlation_hash = sha256_file(
    OUTPUT_CORRELATION_FILE
)

pairs_hash = sha256_file(
    OUTPUT_PAIRS_FILE
)

groups_hash = sha256_file(
    OUTPUT_GROUPS_FILE
)


# ============================================================
# ROUND-TRIP VERIFICATION
# ============================================================

correlation_roundtrip = pd.read_csv(
    OUTPUT_CORRELATION_FILE,
    keep_default_na=False,
)

pairs_roundtrip = pd.read_csv(
    OUTPUT_PAIRS_FILE,
    keep_default_na=False,
)

groups_roundtrip = pd.read_csv(
    OUTPUT_GROUPS_FILE,
    keep_default_na=False,
)


if len(
    correlation_roundtrip
) != 25:

    raise RuntimeError(
        "Correlation CSV round-trip failed."
    )


if len(
    pairs_roundtrip
) != 300:

    raise RuntimeError(
        "Feature-pair CSV round-trip failed."
    )


if len(
    groups_roundtrip
) != 25:

    raise RuntimeError(
        "Feature-group CSV round-trip failed."
    )


print(
    "Correlation matrix round-trip:",
    "PASS"
)

print(
    "Feature-pair round-trip:",
    "PASS"
)

print(
    "Feature-group round-trip:",
    "PASS"
)


# ============================================================
# GROUP SUMMARY FOR CONTRACT
# ============================================================

group_summary = []


for group_id in (
    groups_df[
        "group_id"
    ].drop_duplicates()
):

    subset = groups_df[
        groups_df[
            "group_id"
        ]
        ==
        group_id
    ].sort_values(
        "feature_index"
    )


    members = subset[
        "feature"
    ].tolist()


    group_summary.append(
        {
            "group_id":
                group_id,

            "group_size":
                len(
                    members
                ),

            "members":
                members,

            "is_singleton":
                bool(
                    len(
                        members
                    )
                    ==
                    1
                ),
        }
    )


# ============================================================
# CONTRACT
# ============================================================

contract = {

    "contract_name":
        (
            "Cycle24-TRAIN Correlated "
            "SHARP Feature Groups"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CORRELATED_FEATURE_GROUPS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_9_5",

    "source_population":
        {
            "name":
                "CYCLE24_TRAIN_ONLY",

            "file":
                str(
                    CYCLE24_TRAIN_FILE
                ),

            "sha256":
                train_hash,

            "rows":
                int(
                    len(
                        train
                    )
                ),

            "entity_groups":
                train_group_count,

            "cycle24_validation_used":
                False,

            "cycle24_holdout_used":
                False,

            "cycle25_used":
                False,
        },

    "features":
        {
            "count":
                25,

            "order":
                FEATURES,

            "feature_selection_performed":
                False,

            "feature_removed":
                False,
        },

    "correlation_design":
        {
            "method":
                "SPEARMAN",

            "source_rows":
                "ALL_CYCLE24_TRAIN_ROWS",

            "pair_count":
                300,

            "absolute_rho_threshold":
                ABS_RHO_THRESHOLD,

            "edge_rule":
                "ABS(SPEARMAN_RHO) >= 0.80",

            "threshold_crossing_edges":
                edge_count,

            "graph_type":
                "UNDIRECTED",

            "grouping_method":
                "CONNECTED_COMPONENTS",

            "transitive_grouping":
                True,

            "singleton_features_preserved":
                True,

            "group_order_rule":
                (
                    "Groups ordered by earliest "
                    "member position in frozen "
                    "25-feature order."
                ),
        },

    "group_results":
        {
            "group_count":
                group_count,

            "singleton_group_count":
                singleton_count,

            "multi_feature_group_count":
                multi_feature_group_count,

            "largest_group_size":
                largest_group_size,

            "groups":
                group_summary,
        },

    "future_shap_use":
        {
            "individual_feature_analysis_retained":
                True,

            "group_level_analysis_required":
                True,

            "group_level_mean_abs_shap_rule":
                (
                    "Sum member-feature mean "
                    "absolute SHAP magnitudes."
                ),

            "groups_may_not_be_changed_using_cycle25":
                True,

            "correlation_is_not_causality":
                True,
        },

    "artifacts":
        {
            "spearman_correlation_matrix":
                {
                    "file":
                        str(
                            OUTPUT_CORRELATION_FILE
                        ),

                    "sha256":
                        correlation_hash,

                    "rows":
                        25,
                },

            "feature_pairs":
                {
                    "file":
                        str(
                            OUTPUT_PAIRS_FILE
                        ),

                    "sha256":
                        pairs_hash,

                    "rows":
                        300,

                    "threshold_crossing_edges":
                        edge_count,
                },

            "correlated_feature_groups":
                {
                    "file":
                        str(
                            OUTPUT_GROUPS_FILE
                        ),

                    "sha256":
                        groups_hash,

                    "rows":
                        25,

                    "groups":
                        group_count,
                },
        },

    "data_access":
        {
            "cycle24_train_read":
                True,

            "cycle24_validation_read":
                False,

            "cycle24_holdout_read":
                False,

            "cycle25_read":
                False,

            "model_artifacts_loaded":
                False,

            "models_retrained":
                False,

            "shap_computed":
                False,
        },

    "evaluation_state":
        {
            "marginal_shift":
                True,

            "prevalence_ar_population":
                True,

            "joint_domain_shift":
                True,

            "common_support":
                True,

            "correlated_feature_groups":
                True,

            "within_cycle24_shap_null":
                False,

            "cross_cycle_shap":
                False,

            "explanation_drift":
                False,
        },

    "provenance":
        {
            "chapter_9_1_protocol":
                {
                    "file":
                        str(
                            PROTOCOL_FILE
                        ),

                    "sha256":
                        sha256_file(
                            PROTOCOL_FILE
                        ),
                },

            "feature_registry":
                {
                    "file":
                        str(
                            FEATURE_REGISTRY_FILE
                        ),

                    "sha256":
                        actual_registry_hash,
                },

            "cycle24_train":
                {
                    "file":
                        str(
                            CYCLE24_TRAIN_FILE
                        ),

                    "sha256":
                        train_hash,
                },
        },

    "interpretation_rule":
        (
            "Strong pairwise correlations identify "
            "sets of frozen model inputs that should "
            "also be interpreted jointly in later "
            "SHAP robustness analysis. Connected "
            "components are a robustness grouping "
            "device only; they do not imply causal, "
            "physical, or interchangeable variables."
        ),

    "immutability_rule":
        (
            "After Chapter 9.5, the Spearman "
            "correlation source population, "
            "absolute-rho threshold, graph edges, "
            "connected components, and group "
            "membership must not be changed in "
            "response to Cycle25 or SHAP results."
        ),
}


OUTPUT_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with OUTPUT_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


# ============================================================
# PRINT GROUPS
# ============================================================

print(
    "\n===================================="
)

print(
    "FROZEN CORRELATED FEATURE GROUPS"
)

print(
    "===================================="
)


for group in group_summary:

    members = ", ".join(
        group[
            "members"
        ]
    )

    print(
        f"{group['group_id']} "
        f"(n={group['group_size']}): "
        f"{members}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n===================================="
)

print(
    "CHAPTER 9.5 SUMMARY"
)

print(
    "===================================="
)


print(
    "Cycle24 TRAIN rows:",
    len(
        train
    )
)

print(
    "Frozen features:",
    25
)

print(
    "Unique feature pairs:",
    300
)

print(
    "|rho| >= 0.80 edges:",
    edge_count
)

print(
    "Connected feature groups:",
    group_count
)

print(
    "Singleton groups:",
    singleton_count
)

print(
    "Multi-feature groups:",
    multi_feature_group_count
)

print(
    "Largest group size:",
    largest_group_size
)

print(
    "Cycle24 validation read:",
    False
)

print(
    "Cycle24 holdout read:",
    False
)

print(
    "Cycle25 read:",
    False
)

print(
    "Models loaded/retrained:",
    False
)

print(
    "SHAP computed:",
    False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_CORRELATION_FILE
)

print(
    OUTPUT_PAIRS_FILE
)

print(
    OUTPUT_GROUPS_FILE
)

print(
    OUTPUT_CONTRACT_FILE
)


print(
    "\n===================================="
)

print(
    "MODEL CHAPTER 9.5 COMPLETE — CORRELATED FEATURE GROUPS FROZEN"
)

print(
    "===================================="
)