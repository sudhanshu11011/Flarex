from pathlib import Path
import argparse
import hashlib
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent


# ============================================================
# DATASETS
# ============================================================

CYCLE24_DATASET = (
    ROOT
    / "data"
    / "final"
    / "dataset_v2"
    / "cycle24_dataset_v2.parquet"
)

CYCLE25_DATASET = (
    ROOT
    / "data"
    / "final"
    / "dataset_v2"
    / "cycle25_dataset_v2.parquet"
)


# ============================================================
# FROZEN MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = (
    ROOT
    / "artifacts"
    / "models"
    / "evaluation"
    / "manifests"
    / "holdout_frozen_model_registry.csv"
)


# ============================================================
# LOGISTIC SCALER
# ============================================================

LOGISTIC_SCALER_FILE = (
    ROOT
    / "artifacts"
    / "preprocessing"
    / "scalers"
    / "logistic_standard_scaler.joblib"
)


EXPECTED_SCALER_SHA256 = (
    "48a9c07301f30052f49bad2db048418d3"
    "db16951f0433002b831b59fa10a1af5"
)


# ============================================================
# FROZEN FEATURE ORDER
# ============================================================

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
# REQUIRED DATA COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "T_REC",
    "observation_time_utc",
    "HARPNUM",
    "NOAA_AR",
    "cycle",
    "target",
    "label_status",
] + FEATURES


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing:\n{path}"
        )


def load_estimator(model_file: Path):
    bundle = joblib.load(model_file)

    # Direct estimator
    if hasattr(bundle, "predict_proba"):
        return bundle, bundle

    # Dictionary bundle
    if isinstance(bundle, dict):
        preferred_keys = [
            "model",
            "estimator",
            "classifier",
            "pipeline",
        ]

        for key in preferred_keys:
            candidate = bundle.get(key)

            if hasattr(candidate, "predict_proba"):
                return bundle, candidate

        for value in bundle.values():
            if hasattr(value, "predict_proba"):
                return bundle, value

    raise RuntimeError(
        f"Could not locate estimator with predict_proba in:\n"
        f"{model_file}"
    )


def verify_model_feature_order(bundle, candidate_id):
    if not isinstance(bundle, dict):
        return

    for key in [
        "feature_order",
        "features",
        "feature_names",
    ]:
        if key in bundle:
            stored = list(bundle[key])

            if stored != FEATURES:
                raise RuntimeError(
                    f"{candidate_id}: model feature order mismatch."
                )

            return


# ============================================================
# VERIFY FROZEN ARTIFACTS
# ============================================================

def load_frozen_models():
    require_file(MODEL_REGISTRY)
    require_file(LOGISTIC_SCALER_FILE)

    registry = pd.read_csv(
        MODEL_REGISTRY,
        keep_default_na=False,
    )

    if len(registry) != 3:
        raise RuntimeError(
            "Expected exactly 3 frozen models."
        )

    expected_families = {
        "logistic",
        "random_forest",
        "xgboost",
    }

    if set(registry["family"].astype(str)) != expected_families:
        raise RuntimeError(
            "Frozen model-family set mismatch."
        )

    models = {}

    for _, row in registry.iterrows():
        family = str(row["family"])
        candidate_id = str(row["candidate_id"])

        model_path_raw = Path(str(row["model_file"]))

        if model_path_raw.is_absolute():
            model_file = model_path_raw
        else:
            model_file = ROOT / model_path_raw

        require_file(model_file)

        actual_hash = sha256_file(model_file)
        expected_hash = str(row["model_sha256"])

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{candidate_id}: model SHA256 mismatch."
            )

        bundle, estimator = load_estimator(model_file)

        verify_model_feature_order(
            bundle,
            candidate_id,
        )

        models[family] = {
            "candidate_id": candidate_id,
            "model_file": model_file,
            "estimator": estimator,
            "threshold": float(
                row["frozen_threshold"]
            ),
        }

    # ----------------------------------------
    # Verify scaler
    # ----------------------------------------

    scaler_hash = sha256_file(
        LOGISTIC_SCALER_FILE
    )

    if scaler_hash != EXPECTED_SCALER_SHA256:
        raise RuntimeError(
            "Frozen Logistic scaler SHA256 mismatch."
        )

    scaler_bundle = joblib.load(
        LOGISTIC_SCALER_FILE
    )

    if not isinstance(scaler_bundle, dict):
        raise RuntimeError(
            "Expected Logistic scaler dictionary bundle."
        )

    required_scaler_keys = {
        "scaler",
        "feature_order",
        "feature_count",
        "fit_scope",
    }

    missing = (
        required_scaler_keys
        -
        set(scaler_bundle.keys())
    )

    if missing:
        raise RuntimeError(
            f"Scaler bundle missing keys: {sorted(missing)}"
        )

    if list(
        scaler_bundle["feature_order"]
    ) != FEATURES:
        raise RuntimeError(
            "Frozen Logistic scaler feature order mismatch."
        )

    if int(
        scaler_bundle["feature_count"]
    ) != 25:
        raise RuntimeError(
            "Frozen Logistic scaler feature count mismatch."
        )

    if (
        scaler_bundle["fit_scope"]
        !=
        "CYCLE24_TRAIN_ONLY"
    ):
        raise RuntimeError(
            "Frozen Logistic scaler fit scope mismatch."
        )

    scaler = scaler_bundle["scaler"]

    if int(scaler.n_features_in_) != 25:
        raise RuntimeError(
            "Frozen Logistic scaler n_features_in_ mismatch."
        )

    return models, scaler


# ============================================================
# LOAD HISTORICAL DATA FOR DATE
# ============================================================

def load_rows_for_date(date_text: str):
    try:
        requested_date = pd.Timestamp(
            date_text
        ).date()

    except Exception as exc:
        raise ValueError(
            "Date format must be YYYY-MM-DD."
        ) from exc

    matching_frames = []

    for dataset_path in [
        CYCLE24_DATASET,
        CYCLE25_DATASET,
    ]:
        require_file(dataset_path)

        data = pd.read_parquet(
            dataset_path,
            columns=REQUIRED_COLUMNS,
        )

        observation_time = pd.to_datetime(
            data["observation_time_utc"],
            utc=True,
        )

        mask = (
            observation_time.dt.date
            ==
            requested_date
        )

        selected = data.loc[
            mask
        ].copy()

        if not selected.empty:
            selected[
                "observation_time_utc"
            ] = observation_time.loc[
                mask
            ]

            matching_frames.append(
                selected
            )

    if not matching_frames:
        return pd.DataFrame(
            columns=REQUIRED_COLUMNS
        )

    result = pd.concat(
        matching_frames,
        ignore_index=True,
    )

    result = result.sort_values(
        [
            "NOAA_AR",
            "observation_time_utc",
            "HARPNUM",
        ]
    ).reset_index(
        drop=True
    )

    return result


# ============================================================
# ACTIVE REGION SELECTION
# ============================================================

def choose_active_region(rows: pd.DataFrame):
    valid = rows.copy()

    valid["NOAA_AR"] = pd.to_numeric(
        valid["NOAA_AR"],
        errors="coerce",
    )

    valid = valid[
        valid["NOAA_AR"].notna()
        &
        (
            valid["NOAA_AR"]
            >
            0
        )
    ].copy()

    active_regions = sorted(
        valid["NOAA_AR"]
        .astype(int)
        .unique()
        .tolist()
    )

    if not active_regions:
        raise RuntimeError(
            "No valid NOAA Active Region found for this date."
        )

    print()
    print("Available NOAA Active Regions")
    print("-" * 42)

    for index, ar in enumerate(
        active_regions,
        start=1,
    ):
        ar_rows = valid[
            valid["NOAA_AR"]
            ==
            ar
        ]

        print(
            f"{index:>2}. NOAA AR {ar} "
            f"({len(ar_rows)} observations)"
        )

    print()

    while True:
        value = input(
            "Enter NOAA Active Region number: "
        ).strip()

        try:
            selected_ar = int(value)
        except ValueError:
            print(
                "Please enter a valid numeric NOAA AR."
            )
            continue

        if selected_ar not in active_regions:
            print(
                "That NOAA AR is not available for this date."
            )
            continue

        return selected_ar


# ============================================================
# OBSERVATION SELECTION
# ============================================================

def choose_observation(
    rows: pd.DataFrame,
    active_region: int,
):
    observations = rows[
        pd.to_numeric(
            rows["NOAA_AR"],
            errors="coerce",
        )
        ==
        active_region
    ].copy()

    observations = observations.sort_values(
        "observation_time_utc"
    ).reset_index(
        drop=True
    )

    if observations.empty:
        raise RuntimeError(
            "No observations found for selected NOAA AR."
        )

    print()
    print(
        f"Observations for NOAA AR {active_region}"
    )
    print("-" * 72)

    for index, row in observations.iterrows():
        timestamp = pd.Timestamp(
            row["observation_time_utc"]
        )

        print(
            f"{index + 1:>2}. "
            f"{timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"| HARPNUM={int(row['HARPNUM'])}"
        )

    print()

    while True:
        value = input(
            "Select observation number: "
        ).strip()

        try:
            selected_index = int(value)
        except ValueError:
            print(
                "Please enter a valid observation number."
            )
            continue

        if not (
            1
            <=
            selected_index
            <=
            len(observations)
        ):
            print(
                "Observation number is out of range."
            )
            continue

        return observations.iloc[
            selected_index - 1
        ]


# ============================================================
# PREDICT ONE OBSERVATION
# ============================================================

def predict_observation(
    row: pd.Series,
    models,
    scaler,
):
    feature_values = row[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    ).reshape(
        1,
        -1
    )

    if feature_values.shape != (
        1,
        25,
    ):
        raise RuntimeError(
            "Expected exactly 25 SHARP features."
        )

    if not np.isfinite(
        feature_values
    ).all():
        raise RuntimeError(
            "Observation contains missing/non-finite features."
        )

    # Logistic Regression:
    # frozen scaler transform only — never fit.
    logistic_input = scaler.transform(
        feature_values
    )

    # RF uses raw float64.
    rf_input = feature_values

    # XGBoost uses raw float32.
    xgb_input = feature_values.astype(
        np.float32,
        copy=False,
    )

    model_inputs = {
        "logistic": logistic_input,
        "random_forest": rf_input,
        "xgboost": xgb_input,
    }

    results = []

    display_names = {
        "logistic":
            "Logistic Regression",

        "random_forest":
            "Random Forest",

        "xgboost":
            "XGBoost",
    }

    for family in [
        "logistic",
        "random_forest",
        "xgboost",
    ]:
        model_info = models[family]

        estimator = model_info[
            "estimator"
        ]

        threshold = float(
            model_info[
                "threshold"
            ]
        )

        probability = float(
            estimator.predict_proba(
                model_inputs[family]
            )[0, 1]
        )

        if not np.isfinite(
            probability
        ):
            raise RuntimeError(
                f"{family}: non-finite probability."
            )

        if not (
            0.0
            <=
            probability
            <=
            1.0
        ):
            raise RuntimeError(
                f"{family}: probability outside [0,1]."
            )

        prediction = int(
            probability
            >=
            threshold
        )

        results.append(
            {
                "family":
                    family,

                "model":
                    display_names[
                        family
                    ],

                "candidate_id":
                    model_info[
                        "candidate_id"
                    ],

                "probability":
                    probability,

                "threshold":
                    threshold,

                "prediction":
                    prediction,
            }
        )

    return results


# ============================================================
# DISPLAY RESULT
# ============================================================

def print_result(
    row: pd.Series,
    results,
):
    timestamp = pd.Timestamp(
        row["observation_time_utc"]
    )

    actual_target = int(
        row["target"]
    )

    actual_text = (
        "FLARE (>= M1.0 within next 24h)"
        if actual_target == 1
        else
        "NO FLARE (< M1.0 within next 24h)"
    )

    print()
    print("=" * 72)
    print("SOLAR FLARE PREDICTION")
    print("=" * 72)

    print(
        "Cycle            :",
        row["cycle"],
    )

    print(
        "Observation UTC  :",
        timestamp.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    print(
        "NOAA Active Region:",
        int(row["NOAA_AR"]),
    )

    print(
        "HARPNUM          :",
        int(row["HARPNUM"]),
    )

    print()
    print(
        "Target: >= M1.0 flare within next 24 hours"
    )

    print()
    print("-" * 72)

    for result in results:
        prediction_text = (
            "FLARE"
            if result["prediction"] == 1
            else
            "NO FLARE"
        )

        print(
            f"{result['model']:<20} "
            f"Probability={result['probability']:.6f}  "
            f"Threshold={result['threshold']:.6f}  "
            f"Prediction={prediction_text}"
        )

    print("-" * 72)

    print()
    print(
        "Actual historical result:",
        actual_text,
    )

    print(
        "Stored label status      :",
        row["label_status"],
    )

    print()
    print(
        "NOTE: Models are reported individually. "
        "No ensemble/majority-vote decision is used."
    )

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Historical solar-flare inference using "
            "the frozen Cycle24-trained LR/RF/XGB models."
        )
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Historical observation date in YYYY-MM-DD format.",
    )

    args = parser.parse_args()

    print()
    print("=" * 72)
    print("FLAREX — FROZEN SOLAR FLARE PREDICTION")
    print("=" * 72)

    print()
    print("Verifying frozen model artifacts...")

    models, scaler = load_frozen_models()

    print("Frozen models/scaler: VERIFIED")

    print()

    if args.date:
        date_text = args.date.strip()
    else:
        date_text = input(
            "Enter historical date (YYYY-MM-DD): "
        ).strip()

    rows = load_rows_for_date(
        date_text
    )

    if rows.empty:
        print()
        print(
            "No frozen historical observation found "
            f"for {date_text}."
        )

        print(
            "Current historical datasets cover "
            "Cycle24 and Cycle25 observations only."
        )

        print(
            "Future/live dates require a separate "
            "SHARP data-fetch mode."
        )

        return 1

    cycles = sorted(
        rows["cycle"]
        .astype(str)
        .unique()
        .tolist()
    )

    print()
    print(
        f"Historical observations found: {len(rows)}"
    )

    print(
        "Cycle:",
        ", ".join(cycles),
    )

    active_region = choose_active_region(
        rows
    )

    selected_row = choose_observation(
        rows,
        active_region,
    )

    results = predict_observation(
        selected_row,
        models,
        scaler,
    )

    print_result(
        selected_row,
        results,
    )

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()

    except KeyboardInterrupt:
        print()
        print("Prediction cancelled by user.")
        exit_code = 130

    except Exception as exc:
        print()
        print("=" * 72)
        print("ERROR")
        print("=" * 72)
        print(str(exc))
        exit_code = 1

    sys.exit(exit_code)