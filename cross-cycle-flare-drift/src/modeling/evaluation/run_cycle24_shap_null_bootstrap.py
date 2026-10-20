# ============================================================
# MODEL CHAPTER 9.6C
# WITHIN-CYCLE24 SHAP NULL BOOTSTRAP
#
# Purpose
# -------
# Estimate ordinary within-Cycle24 variation in SHAP
# attribution structure BEFORE evaluating Cycle25 SHAP drift.
#
# Frozen design
# -------------
# Population:
#   Cycle24 independent HOLDOUT only
#
# Bootstrap unit:
#   ENTITY_GROUP_ID
#
# Replicates:
#   2000 independent A-vs-B pseudo-sample comparisons
#
# Seeds:
#   A = 29062
#   B = 29063
#
# Same bootstrap group draws are reused across:
#   - Logistic Regression
#   - Random Forest
#   - XGBoost
#
# And across strata:
#   - OVERALL
#   - TARGET_0
#   - TARGET_1
#
# Metrics:
#   - Spearman rho
#   - Kendall tau-b
#   - Top-5 overlap fraction
#   - Top-10 overlap fraction
#   - Normalized L1 distance on mean-|SHAP|
#   - Cosine distance on mean-|SHAP|
#   - Signed cosine similarity
#   - Sign agreement
#
# Featurewise null values are also retained for later
# empirical featurewise inference / FDR in Chapter 10.
#
# IMPORTANT:
#   - NO Cycle25 read
#   - NO model loading
#   - NO SHAP recomputation
#   - NO model retraining
#   - NO thresholds interpreted here
#     (thresholds belong to Chapter 9.6D)
# ============================================================


from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, kendalltau


# ============================================================
# FROZEN SETTINGS
# ============================================================

N_REPLICATES = 2000

SEED_A = 29062
SEED_B = 29063

EXPECTED_ROWS = 13072
EXPECTED_GROUPS = 176
EXPECTED_FEATURES = 25

EPSILON = 1e-12

ENTITY_GROUP_COLUMN = "ENTITY_GROUP_ID"
TARGET_COLUMN = "target"


MODEL_IDS = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


STRATA = [
    "OVERALL",
    "TARGET_0",
    "TARGET_1",
]


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


SHAP_COLUMNS = [
    f"SHAP_{feature}"
    for feature in FEATURES
]


# ============================================================
# INPUT PATHS
# ============================================================

HOLDOUT_FILE = Path(
    "artifacts/splitting/data/"
    "cycle24_holdout.parquet"
)


SHAP_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "cycle24_holdout_shap_contract.json"
)


SHAP_REGISTRY_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "cycle24_holdout_shap_registry.csv"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

NULL_DIR = Path(
    "artifacts/models/shift_controls/shap/null"
)


METRIC_REPLICATES_FILE = (
    NULL_DIR
    /
    "cycle24_shap_null_replicates.parquet"
)


FEATUREWISE_REPLICATES_FILE = (
    NULL_DIR
    /
    "cycle24_shap_null_featurewise_replicates.parquet"
)


DRAW_COUNTS_FILE = (
    NULL_DIR
    /
    "cycle24_shap_null_draw_counts.npz"
)


GROUP_ORDER_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "cycle24_shap_null_group_order.csv"
)


EXECUTION_MANIFEST_FILE = Path(
    "artifacts/models/shift_controls/manifests/"
    "cycle24_shap_null_execution_manifest.csv"
)


EXECUTION_CONTRACT_FILE = Path(
    "artifacts/models/shift_controls/contracts/"
    "cycle24_shap_null_execution_contract.json"
)


# ============================================================
# EXPECTED PROVENANCE
# ============================================================

EXPECTED_HOLDOUT_HASH = (
    "8abb3929febcde7d778c4781ac890ff5e"
    "658933c2ef2a06aa19da085e7e255fc"
)


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


def sha256_text(value: str) -> str:

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def verify_exists(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


# ============================================================
# BOOTSTRAP DRAW CREATION
# ============================================================

def create_bootstrap_counts(
    seed: int,
    n_replicates: int,
    n_groups: int,
):

    # Explicit PCG64 Generator through default_rng.
    rng = np.random.default_rng(
        seed
    )


    # Each replicate samples exactly n_groups clusters
    # WITH replacement from the complete holdout group set.
    draws = rng.choice(
        n_groups,
        size=(
            n_replicates,
            n_groups,
        ),
        replace=True,
    )


    counts = np.zeros(
        (
            n_replicates,
            n_groups,
        ),
        dtype=np.uint16,
    )


    replicate_index = np.repeat(
        np.arange(
            n_replicates,
            dtype=np.int64,
        ),
        n_groups,
    )


    np.add.at(
        counts,
        (
            replicate_index,
            draws.reshape(-1),
        ),
        1,
    )


    # Every replicate MUST contain exactly 176
    # sampled group instances.
    if not np.all(
        counts.sum(axis=1)
        ==
        n_groups
    ):

        raise RuntimeError(
            "Bootstrap draw-count integrity failed."
        )


    return counts


# ============================================================
# VECTOR METRICS
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    norm_a = float(
        np.linalg.norm(a)
    )

    norm_b = float(
        np.linalg.norm(b)
    )


    if (
        norm_a <= EPSILON
        and
        norm_b <= EPSILON
    ):

        return 1.0


    if (
        norm_a <= EPSILON
        or
        norm_b <= EPSILON
    ):

        return 0.0


    value = float(
        np.dot(a, b)
        /
        (
            norm_a
            *
            norm_b
        )
    )


    # Numerical protection only.
    return float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )


def top_k_overlap(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
) -> float:

    feature_index = np.arange(
        len(a),
        dtype=np.int64,
    )


    # Primary sort:
    # descending attribution magnitude.
    #
    # Tie break:
    # frozen feature order.
    order_a = np.lexsort(
        (
            feature_index,
            -a,
        )
    )


    order_b = np.lexsort(
        (
            feature_index,
            -b,
        )
    )


    top_a = set(
        order_a[:k].tolist()
    )

    top_b = set(
        order_b[:k].tolist()
    )


    return (
        len(
            top_a.intersection(
                top_b
            )
        )
        /
        float(k)
    )


def normalized_l1_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    numerator = float(
        np.sum(
            np.abs(
                a
                -
                b
            )
        )
    )


    denominator = float(
        np.sum(
            np.abs(a)
        )
        +
        np.sum(
            np.abs(b)
        )
    )


    if denominator <= EPSILON:

        return 0.0


    return (
        numerator
        /
        denominator
    )


def sign_agreement(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(
        np.mean(
            np.sign(a)
            ==
            np.sign(b)
        )
    )


def calculate_structure_metrics(
    mean_abs_a: np.ndarray,
    mean_abs_b: np.ndarray,
    mean_signed_a: np.ndarray,
    mean_signed_b: np.ndarray,
):

    spearman_result = spearmanr(
        mean_abs_a,
        mean_abs_b,
    )


    kendall_result = kendalltau(
        mean_abs_a,
        mean_abs_b,
        variant="b",
    )


    spearman_value = float(
        spearman_result.statistic
    )


    kendall_value = float(
        kendall_result.statistic
    )


    if not np.isfinite(
        spearman_value
    ):

        raise RuntimeError(
            "Non-finite Spearman rho."
        )


    if not np.isfinite(
        kendall_value
    ):

        raise RuntimeError(
            "Non-finite Kendall tau-b."
        )


    mean_abs_cosine_similarity = (
        cosine_similarity(
            mean_abs_a,
            mean_abs_b,
        )
    )


    return {
        "spearman_rho":
            spearman_value,

        "kendall_tau_b":
            kendall_value,

        "top5_overlap":
            top_k_overlap(
                mean_abs_a,
                mean_abs_b,
                5,
            ),

        "top10_overlap":
            top_k_overlap(
                mean_abs_a,
                mean_abs_b,
                10,
            ),

        "normalized_l1_mean_abs":
            normalized_l1_distance(
                mean_abs_a,
                mean_abs_b,
            ),

        "cosine_distance_mean_abs":
            (
                1.0
                -
                mean_abs_cosine_similarity
            ),

        "signed_cosine_similarity":
            cosine_similarity(
                mean_signed_a,
                mean_signed_b,
            ),

        "sign_agreement":
            sign_agreement(
                mean_signed_a,
                mean_signed_b,
            ),
    }


# ============================================================
# GROUP-LEVEL PRE-AGGREGATION
# ============================================================

def build_group_aggregates(
    shap_matrix: np.ndarray,
    target: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
    stratum: str,
):

    if stratum == "OVERALL":

        mask = np.ones(
            len(target),
            dtype=bool,
        )


    elif stratum == "TARGET_0":

        mask = (
            target
            ==
            0
        )


    elif stratum == "TARGET_1":

        mask = (
            target
            ==
            1
        )


    else:

        raise RuntimeError(
            f"Unknown stratum: {stratum}"
        )


    selected_shap = shap_matrix[
        mask
    ]


    selected_group_codes = group_codes[
        mask
    ]


    if len(
        selected_shap
    ) == 0:

        raise RuntimeError(
            f"Empty stratum: {stratum}"
        )


    group_abs_sum = np.zeros(
        (
            n_groups,
            EXPECTED_FEATURES,
        ),
        dtype=np.float64,
    )


    group_signed_sum = np.zeros(
        (
            n_groups,
            EXPECTED_FEATURES,
        ),
        dtype=np.float64,
    )


    group_row_count = np.zeros(
        n_groups,
        dtype=np.int64,
    )


    np.add.at(
        group_abs_sum,
        selected_group_codes,
        np.abs(
            selected_shap
        ),
    )


    np.add.at(
        group_signed_sum,
        selected_group_codes,
        selected_shap,
    )


    np.add.at(
        group_row_count,
        selected_group_codes,
        1,
    )


    if int(
        group_row_count.sum()
    ) != len(
        selected_shap
    ):

        raise RuntimeError(
            f"{stratum}: group aggregation "
            "row-count mismatch."
        )


    return (
        group_abs_sum,
        group_signed_sum,
        group_row_count,
    )


# ============================================================
# BOOTSTRAP MEANS
# ============================================================

def calculate_bootstrap_means(
    draw_counts: np.ndarray,
    group_abs_sum: np.ndarray,
    group_signed_sum: np.ndarray,
    group_row_count: np.ndarray,
):

    bootstrap_row_counts = (
        draw_counts
        @
        group_row_count
    )


    if np.any(
        bootstrap_row_counts
        <=
        0
    ):

        raise RuntimeError(
            "A bootstrap replicate produced "
            "an empty analysis stratum."
        )


    mean_abs = (
        draw_counts
        @
        group_abs_sum
    )


    mean_abs = (
        mean_abs
        /
        bootstrap_row_counts[
            :,
            None
        ]
    )


    mean_signed = (
        draw_counts
        @
        group_signed_sum
    )


    mean_signed = (
        mean_signed
        /
        bootstrap_row_counts[
            :,
            None
        ]
    )


    if not np.isfinite(
        mean_abs
    ).all():

        raise RuntimeError(
            "Non-finite bootstrap "
            "mean-|SHAP| values."
        )


    if not np.isfinite(
        mean_signed
    ).all():

        raise RuntimeError(
            "Non-finite bootstrap "
            "mean signed SHAP values."
        )


    return (
        mean_abs,
        mean_signed,
        bootstrap_row_counts,
    )


# ============================================================
# START
# ============================================================

print("=" * 76)

print(
    "MODEL CHAPTER 9.6C — "
    "WITHIN-CYCLE24 SHAP NULL BOOTSTRAP"
)

print("=" * 76)


# ============================================================
# REQUIRED INPUTS
# ============================================================

for path in [
    HOLDOUT_FILE,
    SHAP_CONTRACT_FILE,
    SHAP_REGISTRY_FILE,
]:

    verify_exists(path)


# ============================================================
# OUTPUT PROTECTION
# ============================================================

outputs = [
    METRIC_REPLICATES_FILE,
    FEATUREWISE_REPLICATES_FILE,
    DRAW_COUNTS_FILE,
    GROUP_ORDER_FILE,
    EXECUTION_MANIFEST_FILE,
    EXECUTION_CONTRACT_FILE,
]


for path in outputs:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite Chapter 9.6C "
            "null artifacts."
        )


# ============================================================
# VERIFY 9.6B FROZEN CONTRACT
# ============================================================

shap_contract = load_json(
    SHAP_CONTRACT_FILE
)


if (
    shap_contract.get("status")
    !=
    "FROZEN_CYCLE24_HOLDOUT_SHAP"
):

    raise RuntimeError(
        "Chapter 9.6B SHAP contract "
        "is not frozen."
    )


if (
    shap_contract.get("next_stage")
    !=
    "MODEL_CHAPTER_9_6C"
):

    raise RuntimeError(
        "Chapter 9.6B contract does not "
        "point to Chapter 9.6C."
    )


if (
    shap_contract[
        "evaluation_state"
    ][
        "cycle24_holdout_shap"
    ]
    is not True
):

    raise RuntimeError(
        "Cycle24 HOLDOUT SHAP is not frozen."
    )


if (
    shap_contract[
        "evaluation_state"
    ][
        "within_cycle24_shap_null"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected existing SHAP-null state."
    )


print(
    "Chapter 9.6B frozen contract:",
    "PASS"
)


# ============================================================
# VERIFY REGISTRY HASH AGAINST CONTRACT
# ============================================================

registry_hash = sha256_file(
    SHAP_REGISTRY_FILE
)


expected_registry_hash = (
    shap_contract[
        "artifacts"
    ][
        "combined_registry"
    ][
        "sha256"
    ]
)


if (
    registry_hash
    !=
    expected_registry_hash
):

    raise RuntimeError(
        "9.6B registry hash mismatch."
    )


registry = pd.read_csv(
    SHAP_REGISTRY_FILE,
    keep_default_na=False,
)


if len(
    registry
) != 3:

    raise RuntimeError(
        "Expected exactly 3 frozen "
        "SHAP registry rows."
    )


if set(
    registry[
        "candidate_id"
    ].astype(str)
) != set(
    MODEL_IDS
):

    raise RuntimeError(
        "Frozen SHAP registry "
        "candidate mismatch."
    )


print(
    "Frozen SHAP registry:",
    "PASS"
)


# ============================================================
# HOLDOUT METADATA ONLY
# ============================================================

holdout_hash = sha256_file(
    HOLDOUT_FILE
)


if (
    holdout_hash
    !=
    EXPECTED_HOLDOUT_HASH
):

    raise RuntimeError(
        "Cycle24 HOLDOUT hash mismatch."
    )


if (
    holdout_hash
    !=
    shap_contract[
        "evaluation_population"
    ][
        "sha256"
    ]
):

    raise RuntimeError(
        "Cycle24 HOLDOUT differs from "
        "9.6B frozen contract."
    )


# Important:
# only group + target metadata are loaded.
# Raw SHARP feature values are NOT required in 9.6C.
holdout = pd.read_parquet(
    HOLDOUT_FILE,
    columns=[
        ENTITY_GROUP_COLUMN,
        TARGET_COLUMN,
    ],
)


if len(
    holdout
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Cycle24 HOLDOUT row-count mismatch."
    )


target = holdout[
    TARGET_COLUMN
].to_numpy(
    dtype=np.int8
)


if not np.isin(
    target,
    [
        0,
        1,
    ],
).all():

    raise RuntimeError(
        "Unexpected target values."
    )


# ============================================================
# FROZEN ENTITY-GROUP ORDER
# ============================================================

groups = sorted(
    holdout[
        ENTITY_GROUP_COLUMN
    ].astype(
        str
    ).unique().tolist()
)


if len(
    groups
) != EXPECTED_GROUPS:

    raise RuntimeError(
        "Expected exactly 176 "
        "Cycle24 HOLDOUT ENTITY_GROUPs."
    )


group_to_index = {
    group:
        index
    for index, group
    in enumerate(groups)
}


group_codes = np.array(
    [
        group_to_index[
            value
        ]
        for value
        in holdout[
            ENTITY_GROUP_COLUMN
        ].astype(
            str
        )
    ],
    dtype=np.int64,
)


if (
    group_codes.min()
    <
    0
    or
    group_codes.max()
    >=
    EXPECTED_GROUPS
):

    raise RuntimeError(
        "ENTITY_GROUP encoding failed."
    )


group_order_hash = sha256_text(
    "\n".join(
        groups
    )
)


group_order_df = pd.DataFrame(
    {
        "group_index":
            np.arange(
                EXPECTED_GROUPS,
                dtype=np.int64,
            ),

        ENTITY_GROUP_COLUMN:
            groups,
    }
)


print(
    "Cycle24 HOLDOUT rows:",
    len(holdout)
)

print(
    "ENTITY_GROUPs:",
    len(groups)
)

print(
    "Frozen group ordering:",
    "PASS"
)


# ============================================================
# CREATE FIXED A/B BOOTSTRAP DRAWS
# ============================================================

print()

print(
    "Creating bootstrap A draws..."
)


draw_counts_a = create_bootstrap_counts(
    seed=SEED_A,
    n_replicates=N_REPLICATES,
    n_groups=EXPECTED_GROUPS,
)


print(
    "Creating bootstrap B draws..."
)


draw_counts_b = create_bootstrap_counts(
    seed=SEED_B,
    n_replicates=N_REPLICATES,
    n_groups=EXPECTED_GROUPS,
)


# A and B must not accidentally be identical.
if np.array_equal(
    draw_counts_a,
    draw_counts_b,
):

    raise RuntimeError(
        "Bootstrap A and B draws "
        "are unexpectedly identical."
    )


unique_groups_a = (
    draw_counts_a
    >
    0
).sum(
    axis=1
)


unique_groups_b = (
    draw_counts_b
    >
    0
).sum(
    axis=1
)


print(
    "Bootstrap replicates:",
    N_REPLICATES
)

print(
    "Groups sampled/replicate:",
    EXPECTED_GROUPS
)

print(
    "Seed A:",
    SEED_A
)

print(
    "Seed B:",
    SEED_B
)

print(
    "A/B draw integrity:",
    "PASS"
)


# ============================================================
# SAVE DRAW COUNTS + GROUP ORDER
# ============================================================

NULL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


GROUP_ORDER_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


EXECUTION_MANIFEST_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


EXECUTION_CONTRACT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


group_order_df.to_csv(
    GROUP_ORDER_FILE,
    index=False,
)


np.savez_compressed(
    DRAW_COUNTS_FILE,
    counts_a=draw_counts_a,
    counts_b=draw_counts_b,
)


# ============================================================
# RESULT HOLDERS
# ============================================================

metric_frames = []

featurewise_frames = []

execution_manifest_rows = []

input_shap_artifacts = []


# ============================================================
# PROCESS EACH FROZEN MODEL
# ============================================================

for candidate_id in MODEL_IDS:

    print()

    print("-" * 76)

    print(
        "NULL BOOTSTRAP:",
        candidate_id
    )

    print("-" * 76)


    row = registry[
        registry[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(
        row
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: frozen registry "
            "lookup failed."
        )


    row = row.iloc[
        0
    ]


    model_family = str(
        row[
            "model_family"
        ]
    )


    shap_path = Path(
        str(
            row[
                "shap_file"
            ]
        )
    )


    verify_exists(
        shap_path
    )


    shap_hash = sha256_file(
        shap_path
    )


    expected_shap_hash = str(
        row[
            "shap_sha256"
        ]
    )


    if (
        shap_hash
        !=
        expected_shap_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: frozen SHAP "
            "artifact hash mismatch."
        )


    # Cross-check with frozen 9.6B contract.
    contract_model_rows = [
        model_row
        for model_row
        in shap_contract[
            "models"
        ]
        if (
            model_row[
                "candidate_id"
            ]
            ==
            candidate_id
        )
    ]


    if len(
        contract_model_rows
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: frozen contract "
            "model lookup failed."
        )


    contract_model_row = (
        contract_model_rows[
            0
        ]
    )


    if (
        contract_model_row[
            "shap_artifact"
        ][
            "sha256"
        ]
        !=
        shap_hash
    ):

        raise RuntimeError(
            f"{candidate_id}: SHAP hash "
            "differs from 9.6B contract."
        )


    # ========================================================
    # LOAD FROZEN ROW-LEVEL SHAP
    # ========================================================

    shap_df = pd.read_parquet(
        shap_path
    )


    if len(
        shap_df
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            f"{candidate_id}: SHAP row-count "
            "mismatch."
        )


    if not np.array_equal(
        shap_df[
            "source_row_number"
        ].to_numpy(
            dtype=np.int64
        ),
        np.arange(
            EXPECTED_ROWS,
            dtype=np.int64,
        ),
    ):

        raise RuntimeError(
            f"{candidate_id}: SHAP row-order "
            "mismatch."
        )


    if not np.array_equal(
        shap_df[
            TARGET_COLUMN
        ].to_numpy(
            dtype=np.int8
        ),
        target,
    ):

        raise RuntimeError(
            f"{candidate_id}: target "
            "alignment mismatch."
        )


    if not np.array_equal(
        shap_df[
            ENTITY_GROUP_COLUMN
        ].astype(
            str
        ).to_numpy(),
        holdout[
            ENTITY_GROUP_COLUMN
        ].astype(
            str
        ).to_numpy(),
    ):

        raise RuntimeError(
            f"{candidate_id}: ENTITY_GROUP "
            "alignment mismatch."
        )


    shap_matrix = shap_df[
        SHAP_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )


    if shap_matrix.shape != (
        EXPECTED_ROWS,
        EXPECTED_FEATURES,
    ):

        raise RuntimeError(
            f"{candidate_id}: SHAP matrix "
            "shape mismatch."
        )


    if not np.isfinite(
        shap_matrix
    ).all():

        raise RuntimeError(
            f"{candidate_id}: non-finite "
            "SHAP values."
        )


    print(
        "Frozen SHAP artifact:",
        "PASS"
    )


    input_shap_artifacts.append(
        {
            "candidate_id":
                candidate_id,

            "model_family":
                model_family,

            "file":
                str(
                    shap_path
                ),

            "sha256":
                shap_hash,
        }
    )


    # ========================================================
    # THREE FROZEN STRATA
    # ========================================================

    for stratum in STRATA:

        print(
            f"  {stratum} ..."
        )


        (
            group_abs_sum,
            group_signed_sum,
            group_row_count,
        ) = build_group_aggregates(
            shap_matrix=shap_matrix,
            target=target,
            group_codes=group_codes,
            n_groups=EXPECTED_GROUPS,
            stratum=stratum,
        )


        # ----------------------------------------------------
        # Bootstrap A
        # ----------------------------------------------------

        (
            mean_abs_a,
            mean_signed_a,
            rows_a,
        ) = calculate_bootstrap_means(
            draw_counts=draw_counts_a,
            group_abs_sum=group_abs_sum,
            group_signed_sum=group_signed_sum,
            group_row_count=group_row_count,
        )


        # ----------------------------------------------------
        # Bootstrap B
        # ----------------------------------------------------

        (
            mean_abs_b,
            mean_signed_b,
            rows_b,
        ) = calculate_bootstrap_means(
            draw_counts=draw_counts_b,
            group_abs_sum=group_abs_sum,
            group_signed_sum=group_signed_sum,
            group_row_count=group_row_count,
        )


        # ----------------------------------------------------
        # STRUCTURE METRICS
        # ----------------------------------------------------

        metric_data = {
            "replicate_id":
                np.arange(
                    N_REPLICATES,
                    dtype=np.int64,
                ),

            "candidate_id":
                np.repeat(
                    candidate_id,
                    N_REPLICATES,
                ),

            "model_family":
                np.repeat(
                    model_family,
                    N_REPLICATES,
                ),

            "stratum":
                np.repeat(
                    stratum,
                    N_REPLICATES,
                ),

            "a_rows":
                rows_a.astype(
                    np.int64
                ),

            "b_rows":
                rows_b.astype(
                    np.int64
                ),

            "a_unique_groups":
                unique_groups_a.astype(
                    np.int64
                ),

            "b_unique_groups":
                unique_groups_b.astype(
                    np.int64
                ),
        }


        metric_names = [
            "spearman_rho",
            "kendall_tau_b",
            "top5_overlap",
            "top10_overlap",
            "normalized_l1_mean_abs",
            "cosine_distance_mean_abs",
            "signed_cosine_similarity",
            "sign_agreement",
        ]


        metric_values = {
            name:
                np.empty(
                    N_REPLICATES,
                    dtype=np.float64,
                )
            for name
            in metric_names
        }


        for replicate_id in range(
            N_REPLICATES
        ):

            result = calculate_structure_metrics(
                mean_abs_a[
                    replicate_id
                ],
                mean_abs_b[
                    replicate_id
                ],
                mean_signed_a[
                    replicate_id
                ],
                mean_signed_b[
                    replicate_id
                ],
            )


            for metric_name in metric_names:

                metric_values[
                    metric_name
                ][
                    replicate_id
                ] = result[
                    metric_name
                ]


        for metric_name in metric_names:

            if not np.isfinite(
                metric_values[
                    metric_name
                ]
            ).all():

                raise RuntimeError(
                    f"{candidate_id}/{stratum}: "
                    f"non-finite {metric_name}."
                )


            metric_data[
                metric_name
            ] = metric_values[
                metric_name
            ]


        metric_frame = pd.DataFrame(
            metric_data
        )


        metric_frames.append(
            metric_frame
        )


        # ====================================================
        # FEATUREWISE NULL REPLICATES
        # ====================================================

        delta_mean_abs = (
            mean_abs_b
            -
            mean_abs_a
        )


        abs_delta_mean_abs = np.abs(
            delta_mean_abs
        )


        delta_mean_signed = (
            mean_signed_b
            -
            mean_signed_a
        )


        abs_delta_mean_signed = np.abs(
            delta_mean_signed
        )


        featurewise_frame = pd.DataFrame(
            {
                "replicate_id":
                    np.repeat(
                        np.arange(
                            N_REPLICATES,
                            dtype=np.int64,
                        ),
                        EXPECTED_FEATURES,
                    ),

                "candidate_id":
                    np.repeat(
                        candidate_id,
                        (
                            N_REPLICATES
                            *
                            EXPECTED_FEATURES
                        ),
                    ),

                "model_family":
                    np.repeat(
                        model_family,
                        (
                            N_REPLICATES
                            *
                            EXPECTED_FEATURES
                        ),
                    ),

                "stratum":
                    np.repeat(
                        stratum,
                        (
                            N_REPLICATES
                            *
                            EXPECTED_FEATURES
                        ),
                    ),

                "feature_index":
                    np.tile(
                        np.arange(
                            1,
                            EXPECTED_FEATURES + 1,
                            dtype=np.int64,
                        ),
                        N_REPLICATES,
                    ),

                "feature":
                    np.tile(
                        np.array(
                            FEATURES,
                            dtype=object,
                        ),
                        N_REPLICATES,
                    ),

                "a_mean_abs_shap":
                    mean_abs_a.reshape(
                        -1
                    ),

                "b_mean_abs_shap":
                    mean_abs_b.reshape(
                        -1
                    ),

                "delta_mean_abs_shap":
                    delta_mean_abs.reshape(
                        -1
                    ),

                "abs_delta_mean_abs_shap":
                    abs_delta_mean_abs.reshape(
                        -1
                    ),

                "a_mean_signed_shap":
                    mean_signed_a.reshape(
                        -1
                    ),

                "b_mean_signed_shap":
                    mean_signed_b.reshape(
                        -1
                    ),

                "delta_mean_signed_shap":
                    delta_mean_signed.reshape(
                        -1
                    ),

                "abs_delta_mean_signed_shap":
                    abs_delta_mean_signed.reshape(
                        -1
                    ),
            }
        )


        if len(
            featurewise_frame
        ) != (
            N_REPLICATES
            *
            EXPECTED_FEATURES
        ):

            raise RuntimeError(
                f"{candidate_id}/{stratum}: "
                "featurewise null row-count mismatch."
            )


        featurewise_frames.append(
            featurewise_frame
        )


        execution_manifest_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "model_family":
                    model_family,

                "stratum":
                    stratum,

                "replicates":
                    N_REPLICATES,

                "features":
                    EXPECTED_FEATURES,

                "empty_a_replicates":
                    int(
                        np.sum(
                            rows_a
                            <=
                            0
                        )
                    ),

                "empty_b_replicates":
                    int(
                        np.sum(
                            rows_b
                            <=
                            0
                        )
                    ),

                "min_a_rows":
                    int(
                        rows_a.min()
                    ),

                "max_a_rows":
                    int(
                        rows_a.max()
                    ),

                "min_b_rows":
                    int(
                        rows_b.min()
                    ),

                "max_b_rows":
                    int(
                        rows_b.max()
                    ),
            }
        )


        print(
            f"  {stratum}: "
            f"{N_REPLICATES} comparisons PASS"
        )


    print(
        candidate_id,
        "COMPLETE"
    )


# ============================================================
# COMBINE METRIC REPLICATES
# ============================================================

metrics_df = pd.concat(
    metric_frames,
    ignore_index=True,
)


expected_metric_rows = (
    len(MODEL_IDS)
    *
    len(STRATA)
    *
    N_REPLICATES
)


if len(
    metrics_df
) != expected_metric_rows:

    raise RuntimeError(
        "Null metric row-count mismatch."
    )


if metrics_df.duplicated(
    subset=[
        "candidate_id",
        "stratum",
        "replicate_id",
    ]
).any():

    raise RuntimeError(
        "Duplicate null metric replicate."
    )


# ============================================================
# COMBINE FEATUREWISE REPLICATES
# ============================================================

featurewise_df = pd.concat(
    featurewise_frames,
    ignore_index=True,
)


expected_featurewise_rows = (
    len(MODEL_IDS)
    *
    len(STRATA)
    *
    N_REPLICATES
    *
    EXPECTED_FEATURES
)


if len(
    featurewise_df
) != expected_featurewise_rows:

    raise RuntimeError(
        "Featurewise null row-count mismatch."
    )


if featurewise_df.duplicated(
    subset=[
        "candidate_id",
        "stratum",
        "replicate_id",
        "feature",
    ]
).any():

    raise RuntimeError(
        "Duplicate featurewise null replicate."
    )


# ============================================================
# EXECUTION MANIFEST
# ============================================================

execution_manifest_df = pd.DataFrame(
    execution_manifest_rows
)


if len(
    execution_manifest_df
) != 9:

    raise RuntimeError(
        "Expected 9 execution-manifest rows."
    )


if (
    execution_manifest_df[
        "empty_a_replicates"
    ].sum()
    !=
    0
):

    raise RuntimeError(
        "Empty A bootstrap strata detected."
    )


if (
    execution_manifest_df[
        "empty_b_replicates"
    ].sum()
    !=
    0
):

    raise RuntimeError(
        "Empty B bootstrap strata detected."
    )


# ============================================================
# SAVE 9.6C OUTPUTS
# ============================================================

metrics_df.to_parquet(
    METRIC_REPLICATES_FILE,
    index=False,
)


featurewise_df.to_parquet(
    FEATUREWISE_REPLICATES_FILE,
    index=False,
)


execution_manifest_df.to_csv(
    EXECUTION_MANIFEST_FILE,
    index=False,
)


# ============================================================
# ROUND-TRIP VALIDATION
# ============================================================

metrics_rt = pd.read_parquet(
    METRIC_REPLICATES_FILE
)


featurewise_rt = pd.read_parquet(
    FEATUREWISE_REPLICATES_FILE
)


manifest_rt = pd.read_csv(
    EXECUTION_MANIFEST_FILE,
    keep_default_na=False,
)


if len(
    metrics_rt
) != expected_metric_rows:

    raise RuntimeError(
        "Null metrics round-trip failed."
    )


if len(
    featurewise_rt
) != expected_featurewise_rows:

    raise RuntimeError(
        "Featurewise null round-trip failed."
    )


if len(
    manifest_rt
) != 9:

    raise RuntimeError(
        "Execution manifest round-trip failed."
    )


# ============================================================
# HASH OUTPUTS
# ============================================================

metrics_hash = sha256_file(
    METRIC_REPLICATES_FILE
)


featurewise_hash = sha256_file(
    FEATUREWISE_REPLICATES_FILE
)


draw_counts_hash = sha256_file(
    DRAW_COUNTS_FILE
)


group_order_file_hash = sha256_file(
    GROUP_ORDER_FILE
)


manifest_hash = sha256_file(
    EXECUTION_MANIFEST_FILE
)


# ============================================================
# 9.6C EXECUTION CONTRACT
# ============================================================

execution_contract = {

    "contract_name":
        (
            "Within-Cycle24 SHAP Null "
            "Bootstrap Execution"
        ),

    "version":
        "1.0",

    "status":
        (
            "CYCLE24_SHAP_NULL_COMPUTED_"
            "PENDING_9_6D_AUDIT"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_9_6C",

    "source_contract":
        {
            "file":
                str(
                    SHAP_CONTRACT_FILE
                ),

            "sha256":
                sha256_file(
                    SHAP_CONTRACT_FILE
                ),

            "required_status":
                "FROZEN_CYCLE24_HOLDOUT_SHAP",
        },

    "population":
        {
            "name":
                "CYCLE24_HOLDOUT",

            "file":
                str(
                    HOLDOUT_FILE
                ),

            "sha256":
                holdout_hash,

            "rows":
                EXPECTED_ROWS,

            "entity_groups":
                EXPECTED_GROUPS,
        },

    "bootstrap_design":
        {
            "unit":
                "ENTITY_GROUP_ID",

            "group_order_rule":
                "LEXICOGRAPHIC_STRING_ASCENDING",

            "group_order_sha256":
                group_order_hash,

            "replicates":
                N_REPLICATES,

            "sampled_groups_per_replicate":
                EXPECTED_GROUPS,

            "replacement":
                True,

            "a_seed":
                SEED_A,

            "b_seed":
                SEED_B,

            "rng":
                (
                    "numpy.random.default_rng "
                    "(PCG64)"
                ),

            "sampling_operation":
                (
                    "rng.choice("
                    "n_groups, "
                    "size=(2000,n_groups), "
                    "replace=True)"
                ),

            "same_draws_across_models":
                True,

            "same_draws_across_strata":
                True,
        },

    "strata":
        [
            "OVERALL",
            "TARGET_0",
            "TARGET_1",
        ],

    "structure_metrics":
        {
            "spearman_rho":
                (
                    "Spearman correlation between "
                    "A and B mean-|SHAP| feature vectors."
                ),

            "kendall_tau_b":
                (
                    "Kendall tau-b between "
                    "A and B mean-|SHAP| feature vectors."
                ),

            "top5_overlap":
                (
                    "Intersection size of deterministic "
                    "top-5 mean-|SHAP| features divided by 5."
                ),

            "top10_overlap":
                (
                    "Intersection size of deterministic "
                    "top-10 mean-|SHAP| features divided by 10."
                ),

            "normalized_l1_mean_abs":
                (
                    "sum(abs(A-B)) / "
                    "(sum(abs(A)) + sum(abs(B)))."
                ),

            "cosine_distance_mean_abs":
                (
                    "1 - cosine_similarity("
                    "A_mean_abs, B_mean_abs)."
                ),

            "signed_cosine_similarity":
                (
                    "Cosine similarity between "
                    "A and B mean signed-SHAP vectors."
                ),

            "sign_agreement":
                (
                    "Fraction of 25 features with "
                    "identical numpy.sign of A and B "
                    "mean signed SHAP."
                ),
        },

    "featurewise_null":
        {
            "retained":
                True,

            "quantities":
                [
                    "A mean-|SHAP|",
                    "B mean-|SHAP|",
                    "signed delta mean-|SHAP| B-A",
                    "absolute delta mean-|SHAP|",
                    "A mean signed SHAP",
                    "B mean signed SHAP",
                    "signed delta mean signed SHAP B-A",
                    "absolute delta mean signed SHAP",
                ],

            "purpose":
                (
                    "Preserve the within-Cycle24 "
                    "featurewise null distribution "
                    "for later empirical featurewise "
                    "testing and FDR procedures."
                ),
        },

    "input_shap_artifacts":
        input_shap_artifacts,

    "artifacts":
        {
            "metric_replicates":
                {
                    "file":
                        str(
                            METRIC_REPLICATES_FILE
                        ),

                    "sha256":
                        metrics_hash,

                    "rows":
                        expected_metric_rows,
                },

            "featurewise_replicates":
                {
                    "file":
                        str(
                            FEATUREWISE_REPLICATES_FILE
                        ),

                    "sha256":
                        featurewise_hash,

                    "rows":
                        expected_featurewise_rows,
                },

            "draw_counts":
                {
                    "file":
                        str(
                            DRAW_COUNTS_FILE
                        ),

                    "sha256":
                        draw_counts_hash,

                    "shape_a":
                        list(
                            draw_counts_a.shape
                        ),

                    "shape_b":
                        list(
                            draw_counts_b.shape
                        ),
                },

            "group_order":
                {
                    "file":
                        str(
                            GROUP_ORDER_FILE
                        ),

                    "sha256":
                        group_order_file_hash,

                    "rows":
                        EXPECTED_GROUPS,
                },

            "execution_manifest":
                {
                    "file":
                        str(
                            EXECUTION_MANIFEST_FILE
                        ),

                    "sha256":
                        manifest_hash,

                    "rows":
                        9,
                },
        },

    "software":
        {
            "numpy":
                np.__version__,

            "pandas":
                pd.__version__,

            "scipy":
                scipy.__version__,
        },

    "scientific_guardrails":
        {
            "cycle25_read":
                False,

            "models_loaded":
                False,

            "models_retrained":
                False,

            "shap_recomputed":
                False,

            "cycle24_train_read":
                False,

            "cycle24_holdout_shap_read":
                True,

            "raw_sharp_features_required":
                False,

            "thresholds_computed":
                False,

            "drift_claims_made":
                False,
        },

    "next_stage":
        (
            "MODEL_CHAPTER_9_6D_"
            "NULL_THRESHOLDS_AND_INTEGRITY_AUDIT"
        ),

    "interpretation_rule":
        (
            "Chapter 9.6C only constructs the "
            "within-Cycle24 null distributions. "
            "No cross-cycle drift decision or null "
            "threshold interpretation is permitted "
            "until Chapter 9.6D is completed."
        ),
}


with EXECUTION_CONTRACT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        execution_contract,
        file,
        indent=2,
    )


# ============================================================
# CONTRACT ROUND-TRIP
# ============================================================

contract_rt = load_json(
    EXECUTION_CONTRACT_FILE
)


if (
    contract_rt[
        "status"
    ]
    !=
    (
        "CYCLE24_SHAP_NULL_COMPUTED_"
        "PENDING_9_6D_AUDIT"
    )
):

    raise RuntimeError(
        "9.6C contract round-trip failed."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 76)

print(
    "MODEL CHAPTER 9.6C — COMPLETE"
)

print("=" * 76)


print(
    "Bootstrap replicates:",
    N_REPLICATES
)

print(
    "Independent A/B draws:",
    "PASS"
)

print(
    "Bootstrap unit:",
    "ENTITY_GROUP_ID"
)

print(
    "Groups per replicate:",
    EXPECTED_GROUPS
)

print(
    "Models:",
    len(MODEL_IDS)
)

print(
    "Strata:",
    len(STRATA)
)

print(
    "Structure metric rows:",
    len(metrics_df)
)

print(
    "Expected structure metric rows:",
    expected_metric_rows
)

print(
    "Featurewise null rows:",
    len(featurewise_df)
)

print(
    "Expected featurewise null rows:",
    expected_featurewise_rows
)

print(
    "Empty bootstrap strata:",
    0
)

print(
    "Cycle25 read:",
    False
)

print(
    "SHAP recomputed:",
    False
)

print(
    "Models loaded:",
    False
)

print(
    "Models retrained:",
    False
)

print(
    "Thresholds interpreted:",
    False
)

print(
    "Status:",
    (
        "CYCLE24_SHAP_NULL_COMPUTED_"
        "PENDING_9_6D_AUDIT"
    )
)


print()

print(
    "Saved:",
    METRIC_REPLICATES_FILE
)

print(
    "Saved:",
    FEATUREWISE_REPLICATES_FILE
)

print(
    "Saved:",
    DRAW_COUNTS_FILE
)

print(
    "Saved:",
    GROUP_ORDER_FILE
)

print(
    "Saved:",
    EXECUTION_MANIFEST_FILE
)

print(
    "Saved:",
    EXECUTION_CONTRACT_FILE
)


print()

print(
    "NEXT:",
    (
        "Chapter 9.6D — Null thresholds "
        "+ integrity audit"
    )
)