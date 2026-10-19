from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap


MODEL_FILE = Path(
    "artifacts/models/random_forest/models/"
    "RF_04_D12_L5_BALSUB.joblib"
)

HOLDOUT_FILE = Path(
    "artifacts/splitting/data/"
    "cycle24_holdout.parquet"
)

BACKGROUND_FILE = Path(
    "artifacts/models/shift_controls/shap/background/"
    "cycle24_train_shap_background.parquet"
)


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


BACKGROUND_SIZE = 512


print("=" * 60)
print("RF FULL HOLDOUT — INTERNAL PREDICTION DIAGNOSTIC")
print("=" * 60)


# ----------------------------------------------------------
# Load frozen RF
# ----------------------------------------------------------

bundle = joblib.load(MODEL_FILE)

model = bundle["model"]

print("Model:", bundle["candidate_id"])
print("Training scope:", bundle["training_scope"])


if bundle["feature_order"] != FEATURES:
    raise RuntimeError("Feature order mismatch.")


# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------

holdout = pd.read_parquet(HOLDOUT_FILE)

background_df = pd.read_parquet(BACKGROUND_FILE)


X_holdout = holdout[FEATURES].to_numpy(
    dtype=np.float64
)

X_background = background_df[FEATURES].to_numpy(
    dtype=np.float64
)


if X_holdout.shape != (13072, 25):
    raise RuntimeError(
        f"Unexpected holdout shape: {X_holdout.shape}"
    )


if X_background.shape != (512, 25):
    raise RuntimeError(
        f"Unexpected background shape: {X_background.shape}"
    )


if not np.isfinite(X_holdout).all():
    raise RuntimeError("Holdout contains non-finite values.")


if not np.isfinite(X_background).all():
    raise RuntimeError("Background contains non-finite values.")


print("Holdout shape:", X_holdout.shape)
print("Background shape:", X_background.shape)


# ----------------------------------------------------------
# Exact frozen 512-row masker
# ----------------------------------------------------------

masker = shap.maskers.Independent(
    X_background,
    max_samples=BACKGROUND_SIZE,
)


if masker.data.shape[0] != BACKGROUND_SIZE:
    raise RuntimeError(
        "Masker did not retain all 512 background rows."
    )


explainer = shap.TreeExplainer(
    model,
    data=masker,
    feature_perturbation="interventional",
    model_output="raw",
)


if explainer.masker.data.shape[0] != BACKGROUND_SIZE:
    raise RuntimeError(
        "TreeExplainer background is not 512 rows."
    )


print(
    "TreeExplainer background rows:",
    explainer.masker.data.shape[0]
)


# ----------------------------------------------------------
# sklearn RF probability
# ----------------------------------------------------------

sklearn_output = model.predict_proba(
    X_holdout
)[:, 1]


# ----------------------------------------------------------
# SHAP internal TreeEnsemble prediction
#
# IMPORTANT:
# This does NOT compute SHAP values.
# It only asks SHAP's internal tree representation
# for its model output.
# ----------------------------------------------------------

internal_output = explainer.model.predict(
    X_holdout
)


internal_output = np.asarray(
    internal_output,
    dtype=np.float64,
)


print(
    "Internal output shape:",
    internal_output.shape
)


if internal_output.ndim == 2:

    if internal_output.shape[1] < 2:
        raise RuntimeError(
            "Unexpected RF internal output shape."
        )

    internal_positive = internal_output[:, 1]

elif internal_output.ndim == 1:

    internal_positive = internal_output

else:

    raise RuntimeError(
        f"Unexpected internal output shape: "
        f"{internal_output.shape}"
    )


# ----------------------------------------------------------
# Compare SHAP internal RF representation
# vs sklearn predict_proba
# ----------------------------------------------------------

error = np.abs(
    internal_positive
    -
    sklearn_output
)


print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print(
    "Mean absolute difference:",
    f"{error.mean():.16e}"
)

print(
    "P95 absolute difference:",
    f"{np.quantile(error, 0.95):.16e}"
)

print(
    "P99 absolute difference:",
    f"{np.quantile(error, 0.99):.16e}"
)

print(
    "Max absolute difference:",
    f"{error.max():.16e}"
)


print(
    "Rows > 1e-12:",
    int(np.sum(error > 1e-12))
)

print(
    "Rows > 1e-9:",
    int(np.sum(error > 1e-9))
)

print(
    "Rows > 1e-6:",
    int(np.sum(error > 1e-6))
)


worst_index = int(
    np.argmax(error)
)


print()
print("Worst row index:", worst_index)

print(
    "sklearn probability:",
    float(
        sklearn_output[worst_index]
    )
)

print(
    "SHAP internal probability:",
    float(
        internal_positive[worst_index]
    )
)

print(
    "absolute difference:",
    float(
        error[worst_index]
    )
)


# ----------------------------------------------------------
# Interpretation
# ----------------------------------------------------------

print()
print("=" * 60)
print("DIAGNOSTIC INTERPRETATION")
print("=" * 60)


if error.max() <= 1e-12:

    print(
        "PASS: SHAP internal RF model output matches "
        "sklearn RF essentially exactly."
    )

    print(
        "Therefore the previous additivity failure is "
        "not caused by RF model-output conversion."
    )

    print(
        "The issue is isolated to TreeSHAP attribution "
        "reconstruction / numerical sensitivity."
    )

elif error.max() <= 1e-6:

    print(
        "PASS WITH NUMERICAL WARNING: internal and "
        "sklearn RF outputs differ only at tiny "
        "floating-point scale."
    )

else:

    print(
        "WARNING: SHAP internal RF representation "
        "does not exactly reproduce sklearn "
        "predict_proba for all rows."
    )

    print(
        "Do NOT rerun full SHAP until this mismatch "
        "is investigated."
    )


print()
print("SHAP values computed:", False)
print("Cycle25 read:", False)
print("Artifacts modified:", False)

print()
print("DIAGNOSTIC COMPLETE")