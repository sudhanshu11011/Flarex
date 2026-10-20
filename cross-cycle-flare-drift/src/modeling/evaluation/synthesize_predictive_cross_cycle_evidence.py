# ============================================================
# MODEL CHAPTER 11.2
# PREDICTIVE CROSS-CYCLE SYNTHESIS
#
# Purpose:
#   Summarize already-frozen Chapter-8 predictive evidence
#   for final cross-cycle synthesis.
#
# Primary predictive metric:
#   TSS
#
# Important:
#   - NO new bootstrap
#   - NO new prediction
#   - NO retraining
#   - NO threshold tuning
#   - NO equivalence claim
#   - NO "stable" claim from zero-containing CI
# ============================================================


from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json

import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

MODEL_ORDER = [
    "LR_02_C0p01_BALANCED",
    "RF_04_D12_L5_BALSUB",
    "XGB_02_LR0p03_D3_BAL",
]


SCOPE_ORDER = [
    "C25_B1_2020_2021",
    "C25_B2_2022_2023",
    "C25_B3_2024_2025",
    "C25_ALL_2020_2025",
]


PRIMARY_CLASSIFICATION = (
    "NO_MEASURABLE_CHANGE_AT_95CI"
)


MODEL_SYNTHESIS_LABEL = (
    "NO_MEASURABLE_PRIMARY_TSS_CHANGE_"
    "AT_95CI_ACROSS_ALL_C25_SCOPES"
)


# ============================================================
# INPUTS
# ============================================================

CHAPTER11_INPUT_CONTRACT = Path(
    "artifacts/models/final_synthesis/contracts/"
    "chapter11_synthesis_inputs_contract.json"
)


PREDICTIVE_CONTRACT = Path(
    "artifacts/models/evaluation/contracts/"
    "cross_cycle_predictive_drift_contract.json"
)


PREDICTIVE_DRIFT = Path(
    "artifacts/models/evaluation/metrics/"
    "cross_cycle_predictive_drift.csv"
)


HOLDOUT_RESULTS = Path(
    "artifacts/models/evaluation/manifests/"
    "frozen_holdout_results.csv"
)


CYCLE25_RESULTS = Path(
    "artifacts/models/evaluation/manifests/"
    "frozen_cycle25_predictive_results.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_ROOT = Path(
    "artifacts/models/final_synthesis"
)


METRIC_DIR = (
    OUTPUT_ROOT
    /
    "metrics"
)


CONTRACT_DIR = (
    OUTPUT_ROOT
    /
    "contracts"
)


MODEL_OUTPUT = (
    METRIC_DIR
    /
    "predictive_cross_cycle_model_synthesis.csv"
)


SUMMARY_OUTPUT = (
    METRIC_DIR
    /
    "predictive_cross_cycle_summary.csv"
)


CONTRACT_OUTPUT = (
    CONTRACT_DIR
    /
    "predictive_cross_cycle_synthesis_contract.json"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def require_file(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "MODEL CHAPTER 11.2 — "
    "PREDICTIVE CROSS-CYCLE SYNTHESIS"
)

print("=" * 78)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    CHAPTER11_INPUT_CONTRACT,
    PREDICTIVE_CONTRACT,
    PREDICTIVE_DRIFT,
    HOLDOUT_RESULTS,
    CYCLE25_RESULTS,
]


for path in required_files:

    require_file(
        path
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

for path in [
    MODEL_OUTPUT,
    SUMMARY_OUTPUT,
    CONTRACT_OUTPUT,
]:

    if path.exists():

        raise RuntimeError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite frozen "
            "Chapter-11.2 outputs."
        )


# ============================================================
# VERIFY CHAPTER-11 INPUT FREEZE
# ============================================================

input_contract = load_json(
    CHAPTER11_INPUT_CONTRACT
)


if (
    input_contract.get("status")
    !=
    "FROZEN_CHAPTER11_SYNTHESIS_INPUTS"
):

    raise RuntimeError(
        "Chapter-11 synthesis inputs are not frozen."
    )


predictive_state = input_contract.get(
    "predictive_evidence_state",
    {},
)


if (
    predictive_state.get("primary_metric")
    !=
    "TSS"
):

    raise RuntimeError(
        "Unexpected primary predictive metric."
    )


if (
    int(
        predictive_state.get(
            "primary_comparisons",
            -1,
        )
    )
    != 12
):

    raise RuntimeError(
        "Unexpected number of primary comparisons."
    )


if (
    int(
        predictive_state.get(
            "no_measurable_change_at_95ci",
            -1,
        )
    )
    != 12
):

    raise RuntimeError(
        "Frozen predictive evidence state changed."
    )


if (
    predictive_state.get(
        "equivalence_claimed"
    )
    is not False
):

    raise RuntimeError(
        "Unexpected equivalence claim."
    )


print(
    "Frozen Chapter-11 input contract:",
    "PASS"
)


# ============================================================
# VERIFY PREDICTIVE CONTRACT
# ============================================================

predictive_contract = load_json(
    PREDICTIVE_CONTRACT
)


if (
    predictive_contract.get("status")
    !=
    "FROZEN_CROSS_CYCLE_PREDICTIVE_DRIFT"
):

    raise RuntimeError(
        "Predictive contract is not frozen."
    )


if (
    predictive_contract.get("primary_metric")
    !=
    "TSS"
):

    raise RuntimeError(
        "Predictive primary metric is not TSS."
    )


constraints = predictive_contract.get(
    "interpretation_constraints",
    {},
)


if (
    constraints.get(
        "zero_containing_delta_ci_is_not_equivalence"
    )
    is not True
):

    raise RuntimeError(
        "Equivalence guardrail missing."
    )


print(
    "Frozen predictive contract:",
    "PASS"
)


# ============================================================
# LOAD TABLES
# ============================================================

drift = pd.read_csv(
    PREDICTIVE_DRIFT,
    keep_default_na=False,
)


holdout = pd.read_csv(
    HOLDOUT_RESULTS,
    keep_default_na=False,
)


cycle25 = pd.read_csv(
    CYCLE25_RESULTS,
    keep_default_na=False,
)


# ============================================================
# PRIMARY TSS ONLY
# ============================================================

tss = drift[
    drift[
        "metric"
    ].astype(str).str.lower()
    ==
    "tss"
].copy()


if len(
    tss
) != 12:

    raise RuntimeError(
        "Expected exactly 12 TSS comparisons."
    )


if set(
    tss[
        "candidate_id"
    ].astype(str)
) != set(
    MODEL_ORDER
):

    raise RuntimeError(
        "TSS model set mismatch."
    )


if set(
    tss[
        "scope_id"
    ].astype(str)
) != set(
    SCOPE_ORDER
):

    raise RuntimeError(
        "TSS Cycle25 scope set mismatch."
    )


if not (
    tss[
        "predictive_change_classification"
    ].astype(str)
    ==
    PRIMARY_CLASSIFICATION
).all():

    raise RuntimeError(
        "At least one primary TSS classification changed."
    )


if (
    tss[
        "equivalence_claimed"
    ].astype(str).str.lower()
    ==
    "true"
).any():

    raise RuntimeError(
        "Equivalence unexpectedly claimed."
    )


print(
    "Primary TSS evidence integrity:",
    "PASS"
)


# ============================================================
# MODEL SYNTHESIS
# ============================================================

model_rows = []


for candidate_id in MODEL_ORDER:

    holdout_row = holdout[
        holdout[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ]


    if len(
        holdout_row
    ) != 1:

        raise RuntimeError(
            f"{candidate_id}: expected one "
            "C24 HOLDOUT row."
        )


    holdout_row = holdout_row.iloc[
        0
    ]


    model_tss = tss[
        tss[
            "candidate_id"
        ].astype(str)
        ==
        candidate_id
    ].copy()


    if len(
        model_tss
    ) != 4:

        raise RuntimeError(
            f"{candidate_id}: expected four "
            "Cycle25 TSS comparisons."
        )


    scope_rows = {}


    for scope_id in SCOPE_ORDER:

        scope = model_tss[
            model_tss[
                "scope_id"
            ].astype(str)
            ==
            scope_id
        ]


        if len(
            scope
        ) != 1:

            raise RuntimeError(
                f"{candidate_id}/{scope_id}: "
                "expected one TSS row."
            )


        scope_rows[
            scope_id
        ] = scope.iloc[
            0
        ]


    b1 = scope_rows[
        "C25_B1_2020_2021"
    ]

    b2 = scope_rows[
        "C25_B2_2022_2023"
    ]

    b3 = scope_rows[
        "C25_B3_2024_2025"
    ]

    pooled = scope_rows[
        "C25_ALL_2020_2025"
    ]


    classification_count = int(
        (
            model_tss[
                "predictive_change_classification"
            ].astype(str)
            ==
            PRIMARY_CLASSIFICATION
        ).sum()
    )


    if classification_count != 4:

        raise RuntimeError(
            f"{candidate_id}: all four "
            "primary comparisons must have "
            "NO_MEASURABLE_CHANGE_AT_95CI."
        )


    # CI containing zero is required by the
    # frozen classification state.
    for _, row in model_tss.iterrows():

        lower = float(
            row[
                "delta_ci_lower_2p5"
            ]
        )

        upper = float(
            row[
                "delta_ci_upper_97p5"
            ]
        )


        if not (
            lower
            <=
            0.0
            <=
            upper
        ):

            raise RuntimeError(
                f"{candidate_id}: classification / "
                "CI consistency failure."
            )


    model_rows.append(
        {
            "candidate_id":
                candidate_id,

            "model_family":
                str(
                    holdout_row[
                        "family"
                    ]
                ),

            "frozen_threshold":
                float(
                    holdout_row[
                        "frozen_threshold"
                    ]
                ),

            "cycle24_holdout_tss":
                float(
                    holdout_row[
                        "tss"
                    ]
                ),

            "cycle24_holdout_tss_ci_lower_95":
                float(
                    holdout_row[
                        "tss_ci_lower_95"
                    ]
                ),

            "cycle24_holdout_tss_ci_upper_95":
                float(
                    holdout_row[
                        "tss_ci_upper_95"
                    ]
                ),

            # --------------------------------------------
            # B1
            # --------------------------------------------

            "b1_tss":
                float(
                    b1[
                        "cycle25_point"
                    ]
                ),

            "b1_delta_tss":
                float(
                    b1[
                        "delta_cycle25_minus_cycle24"
                    ]
                ),

            "b1_delta_ci_lower_95":
                float(
                    b1[
                        "delta_ci_lower_2p5"
                    ]
                ),

            "b1_delta_ci_upper_95":
                float(
                    b1[
                        "delta_ci_upper_97p5"
                    ]
                ),

            "b1_primary_classification":
                str(
                    b1[
                        "predictive_change_classification"
                    ]
                ),

            # --------------------------------------------
            # B2
            # --------------------------------------------

            "b2_tss":
                float(
                    b2[
                        "cycle25_point"
                    ]
                ),

            "b2_delta_tss":
                float(
                    b2[
                        "delta_cycle25_minus_cycle24"
                    ]
                ),

            "b2_delta_ci_lower_95":
                float(
                    b2[
                        "delta_ci_lower_2p5"
                    ]
                ),

            "b2_delta_ci_upper_95":
                float(
                    b2[
                        "delta_ci_upper_97p5"
                    ]
                ),

            "b2_primary_classification":
                str(
                    b2[
                        "predictive_change_classification"
                    ]
                ),

            # --------------------------------------------
            # B3
            # --------------------------------------------

            "b3_tss":
                float(
                    b3[
                        "cycle25_point"
                    ]
                ),

            "b3_delta_tss":
                float(
                    b3[
                        "delta_cycle25_minus_cycle24"
                    ]
                ),

            "b3_delta_ci_lower_95":
                float(
                    b3[
                        "delta_ci_lower_2p5"
                    ]
                ),

            "b3_delta_ci_upper_95":
                float(
                    b3[
                        "delta_ci_upper_97p5"
                    ]
                ),

            "b3_primary_classification":
                str(
                    b3[
                        "predictive_change_classification"
                    ]
                ),

            # --------------------------------------------
            # POOLED
            # --------------------------------------------

            "pooled_tss":
                float(
                    pooled[
                        "cycle25_point"
                    ]
                ),

            "pooled_delta_tss":
                float(
                    pooled[
                        "delta_cycle25_minus_cycle24"
                    ]
                ),

            "pooled_delta_ci_lower_95":
                float(
                    pooled[
                        "delta_ci_lower_2p5"
                    ]
                ),

            "pooled_delta_ci_upper_95":
                float(
                    pooled[
                        "delta_ci_upper_97p5"
                    ]
                ),

            "pooled_primary_classification":
                str(
                    pooled[
                        "predictive_change_classification"
                    ]
                ),

            # --------------------------------------------
            # MODEL-LEVEL SYNTHESIS
            # --------------------------------------------

            "primary_tss_comparisons":
                4,

            "no_measurable_change_count":
                classification_count,

            "all_primary_tss_delta_ci_include_zero":
                True,

            "equivalence_claimed":
                False,

            "predictive_synthesis_label":
                MODEL_SYNTHESIS_LABEL,

            "predictive_stability_proven":
                False,

            "predictive_equivalence_proven":
                False,

            "model_retrained":
                False,

            "threshold_reoptimized":
                False,

            "preprocessing_refit":
                False,
        }
    )


model_synthesis = pd.DataFrame(
    model_rows
)


if len(
    model_synthesis
) != 3:

    raise RuntimeError(
        "Expected 3 predictive model synthesis rows."
    )


# ============================================================
# CROSS-MODEL SUMMARY
# ============================================================

models_with_all_no_measurable_change = int(
    (
        model_synthesis[
            "no_measurable_change_count"
        ]
        ==
        4
    ).sum()
)


if (
    models_with_all_no_measurable_change
    !=
    3
):

    raise RuntimeError(
        "Unexpected cross-model predictive state."
    )


summary = pd.DataFrame(
    [
        {
            "models_evaluated":
                3,

            "cycle25_scopes_per_model":
                4,

            "primary_tss_comparisons":
                12,

            "no_measurable_change_at_95ci":
                12,

            "models_with_no_measurable_change_across_all_scopes":
                3,

            "primary_cross_model_predictive_pattern":
                (
                    "NO_MEASURABLE_PRIMARY_TSS_CHANGE_"
                    "AT_95CI_ACROSS_ALL_MODELS_AND_SCOPES"
                ),

            "equivalence_claimed":
                False,

            "stability_proven":
                False,

            "secondary_metrics_used_for_primary_drift_classification":
                False,

            "interpretation":
                (
                    "Across LR, RF, and XGB, all four "
                    "Cycle25 TSS comparisons per model "
                    "have 95% bootstrap delta intervals "
                    "that include zero under the frozen "
                    "classification rule. Therefore no "
                    "measurable primary TSS change was "
                    "detected. This does not establish "
                    "predictive equivalence or prove "
                    "strict stability."
                ),
        }
    ]
)


# ============================================================
# SAVE CSV OUTPUTS
# ============================================================

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


for output_file in [
    MODEL_OUTPUT,
    SUMMARY_OUTPUT,
]:

    if output_file.exists():

        raise RuntimeError(
            f"Output unexpectedly exists: {output_file}"
        )


model_temp = MODEL_OUTPUT.with_name(
    MODEL_OUTPUT.stem
    +
    ".tmp.csv"
)


summary_temp = SUMMARY_OUTPUT.with_name(
    SUMMARY_OUTPUT.stem
    +
    ".tmp.csv"
)


for path in [
    model_temp,
    summary_temp,
]:

    if path.exists():

        path.unlink()


model_synthesis.to_csv(
    model_temp,
    index=False,
)


summary.to_csv(
    summary_temp,
    index=False,
)


model_rt = pd.read_csv(
    model_temp,
    keep_default_na=False,
)


summary_rt = pd.read_csv(
    summary_temp,
    keep_default_na=False,
)


if len(
    model_rt
) != 3:

    raise RuntimeError(
        "Model synthesis round-trip failed."
    )


if len(
    summary_rt
) != 1:

    raise RuntimeError(
        "Predictive summary round-trip failed."
    )


model_temp.replace(
    MODEL_OUTPUT
)


summary_temp.replace(
    SUMMARY_OUTPUT
)


model_hash = sha256_file(
    MODEL_OUTPUT
)


summary_hash = sha256_file(
    SUMMARY_OUTPUT
)


print(
    "Predictive synthesis CSV round-trip:",
    "PASS"
)


# ============================================================
# CONTRACT
# ============================================================

CONTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


contract = {

    "contract_name":
        (
            "Frozen Chapter-11 Predictive "
            "Cross-Cycle Synthesis"
        ),

    "version":
        "1.0",

    "status":
        "FROZEN_CHAPTER11_PREDICTIVE_SYNTHESIS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "chapter":
        "MODEL_CHAPTER_11_2",

    "primary_metric":
        "TSS",

    "models":
        MODEL_ORDER,

    "cycle25_scopes":
        SCOPE_ORDER,

    "evidence_state":
        {
            "models_evaluated":
                3,

            "scopes_per_model":
                4,

            "primary_comparisons":
                12,

            "no_measurable_change_at_95ci":
                12,

            "models_with_all_four_no_measurable_change":
                3,

            "equivalence_claimed":
                False,

            "stability_proven":
                False,
        },

    "model_synthesis":
        {
            "file":
                MODEL_OUTPUT.as_posix(),

            "sha256":
                model_hash,

            "rows":
                3,
        },

    "cross_model_summary":
        {
            "file":
                SUMMARY_OUTPUT.as_posix(),

            "sha256":
                summary_hash,

            "rows":
                1,

            "pattern":
                (
                    "NO_MEASURABLE_PRIMARY_TSS_CHANGE_"
                    "AT_95CI_ACROSS_ALL_MODELS_AND_SCOPES"
                ),
        },

    "scientific_guardrails":
        {
            "new_predictions_generated":
                False,

            "models_retrained":
                False,

            "thresholds_reoptimized":
                False,

            "preprocessing_refit":
                False,

            "new_bootstrap_performed":
                False,

            "new_statistical_threshold_created":
                False,

            "secondary_metrics_reclassified":
                False,

            "equivalence_test_performed":
                False,

            "equivalence_claimed":
                False,

            "strict_stability_claimed":
                False,

            "final_prediction_explanation_joint_outcome_assigned":
                False,

            "hypotheses_evaluated":
                False,
        },

    "required_interpretation":
        (
            "The frozen primary TSS comparison detected "
            "no measurable Cycle24-to-Cycle25 change at "
            "the 95% bootstrap-CI criterion in any of the "
            "12 model-by-scope comparisons. This result "
            "must not be described as proof of statistical "
            "equivalence or strict predictive stability. "
            "Secondary predictive metrics remain reported "
            "but are not used to redefine the frozen "
            "primary predictive-drift classification."
        ),

    "next_stage":
        (
            "MODEL_CHAPTER_11_3_"
            "EXPLANATION_SYNTHESIS"
        ),
}


contract_temp = CONTRACT_OUTPUT.with_name(
    CONTRACT_OUTPUT.stem
    +
    ".tmp.json"
)


if contract_temp.exists():

    contract_temp.unlink()


with contract_temp.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        contract,
        file,
        indent=2,
    )


contract_rt = load_json(
    contract_temp
)


if (
    contract_rt.get("status")
    !=
    "FROZEN_CHAPTER11_PREDICTIVE_SYNTHESIS"
):

    raise RuntimeError(
        "Predictive synthesis contract "
        "round-trip failed."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "equivalence_claimed"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected equivalence claim."
    )


if (
    contract_rt[
        "scientific_guardrails"
    ][
        "strict_stability_claimed"
    ]
    is not False
):

    raise RuntimeError(
        "Unexpected stability claim."
    )


contract_temp.replace(
    CONTRACT_OUTPUT
)


contract_hash = sha256_file(
    CONTRACT_OUTPUT
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 78)

print(
    "MODEL CHAPTER 11.2 — COMPLETE"
)

print("=" * 78)


for _, row in model_synthesis.iterrows():

    print()

    print(
        row[
            "candidate_id"
        ]
    )

    print(
        "C24 HOLDOUT TSS:",
        f"{row['cycle24_holdout_tss']:.6f}"
    )

    print(
        "B1 TSS:",
        f"{row['b1_tss']:.6f}",
        "| delta:",
        f"{row['b1_delta_tss']:.6f}"
    )

    print(
        "B2 TSS:",
        f"{row['b2_tss']:.6f}",
        "| delta:",
        f"{row['b2_delta_tss']:.6f}"
    )

    print(
        "B3 TSS:",
        f"{row['b3_tss']:.6f}",
        "| delta:",
        f"{row['b3_delta_tss']:.6f}"
    )

    print(
        "Pooled TSS:",
        f"{row['pooled_tss']:.6f}",
        "| delta:",
        f"{row['pooled_delta_tss']:.6f}"
    )

    print(
        "Primary classifications:",
        "4/4 NO_MEASURABLE_CHANGE_AT_95CI"
    )


print()

print(
    "Total primary TSS comparisons:",
    12
)

print(
    "NO_MEASURABLE_CHANGE_AT_95CI:",
    12
)

print(
    "Equivalence claimed:",
    False
)

print(
    "Strict stability claimed:",
    False
)


print()

print(
    "Cross-model predictive pattern:"
)

print(
    "NO_MEASURABLE_PRIMARY_TSS_CHANGE_"
    "AT_95CI_ACROSS_ALL_MODELS_AND_SCOPES"
)


print()

print(
    "Status:",
    "FROZEN_CHAPTER11_PREDICTIVE_SYNTHESIS"
)


print()

print(
    "Model synthesis:",
    MODEL_OUTPUT
)

print(
    "Summary:",
    SUMMARY_OUTPUT
)

print(
    "Contract:",
    CONTRACT_OUTPUT
)

print(
    "Contract SHA256:",
    contract_hash
)


print()

print(
    "NEXT:",
    "Chapter 11.3 — Explanation Synthesis"
)