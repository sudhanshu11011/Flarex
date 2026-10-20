import joblib
import numpy as np
import pandas as pd
import shap

from pathlib import Path


MODEL_FILE = Path(
    "artifacts/models/xgboost/models/"
    "XGB_02_LR0p03_D3_BAL.joblib"
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


print("=" * 70)
print("XGB INTERVENTIONAL SHAP — BOOSTER COMPATIBILITY DIAGNOSTIC")
print("=" * 70)


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

bundle = joblib.load(
    MODEL_FILE
)

model = bundle["model"]

booster = model.get_booster()


print("Candidate:", bundle["candidate_id"])
print("Training scope:", bundle["training_scope"])

print(
    "Wrapper enable_categorical:",
    model.get_params().get(
        "enable_categorical"
    )
)

print(
    "Booster feature_types:",
    booster.feature_types
)


# ============================================================
# LOAD DATA
# ============================================================

holdout = pd.read_parquet(
    HOLDOUT_FILE
)

background = pd.read_parquet(
    BACKGROUND_FILE
)


X_holdout = holdout[
    FEATURES
].to_numpy(
    dtype=np.float64
)

X_background = background[
    FEATURES
].to_numpy(
    dtype=np.float64
)


print(
    "Holdout shape:",
    X_holdout.shape
)

print(
    "Background shape:",
    X_background.shape
)


if X_holdout.shape != (
    13072,
    25,
):

    raise RuntimeError(
        "Unexpected HOLDOUT shape."
    )


if X_background.shape != (
    512,
    25,
):

    raise RuntimeError(
        "Unexpected background shape."
    )


if not np.isfinite(
    X_holdout
).all():

    raise RuntimeError(
        "Non-finite HOLDOUT features."
    )


if not np.isfinite(
    X_background
).all():

    raise RuntimeError(
        "Non-finite background features."
    )


print(
    "Numeric feature integrity:",
    "PASS"
)


# ============================================================
# COMPARE WRAPPER VS BOOSTER RAW MARGIN
# ============================================================

X_test = X_holdout[
    :256
]


wrapper_margin = model.predict(
    X_test,
    output_margin=True,
)


booster_margin = booster.inplace_predict(
    X_test,
    predict_type="margin",
)


wrapper_margin = np.asarray(
    wrapper_margin,
    dtype=np.float64,
).reshape(-1)


booster_margin = np.asarray(
    booster_margin,
    dtype=np.float64,
).reshape(-1)


prediction_error = np.abs(
    wrapper_margin
    -
    booster_margin
)


print()
print("Wrapper vs Booster raw margin")
print(
    "Mean abs error:",
    f"{prediction_error.mean():.16e}"
)

print(
    "Max abs error:",
    f"{prediction_error.max():.16e}"
)


if not np.allclose(
    wrapper_margin,
    booster_margin,
    atol=1e-12,
    rtol=1e-12,
):

    raise RuntimeError(
        "Wrapper and Booster raw margins "
        "do not match."
    )


print(
    "Wrapper vs Booster:",
    "PASS"
)


# ============================================================
# CREATE EXACT 512-ROW MASKER
# ============================================================

masker = shap.maskers.Independent(
    X_background,
    max_samples=512,
)


if masker.data.shape[0] != 512:

    raise RuntimeError(
        "Masker did not retain 512 rows."
    )


print()
print(
    "Masker background rows:",
    masker.data.shape[0]
)


# ============================================================
# IMPORTANT TEST
#
# Use the exact frozen Booster rather than sklearn wrapper.
# No model retraining.
# No tree modification.
# Same fitted trees.
# Same raw-margin output.
# Same 512-row interventional background.
# ============================================================

explainer = shap.TreeExplainer(
    booster,
    data=masker,
    feature_perturbation="interventional",
    model_output="raw",
)


print(
    "Booster TreeExplainer construction:",
    "PASS"
)


if explainer.masker.data.shape[0] != 512:

    raise RuntimeError(
        "TreeExplainer did not retain "
        "512 background rows."
    )


# ============================================================
# SMALL SHAP TEST ONLY
# ============================================================

TEST_ROWS = 16

X_small = X_holdout[
    :TEST_ROWS
]


explanation = explainer(
    X_small,
    check_additivity=True,
)


values = np.asarray(
    explanation.values,
    dtype=np.float64,
)


base_values = np.asarray(
    explanation.base_values,
    dtype=np.float64,
)


print(
    "SHAP values shape:",
    values.shape
)

print(
    "Base values shape:",
    base_values.shape
)


if values.shape != (
    TEST_ROWS,
    25,
):

    raise RuntimeError(
        "Unexpected Booster SHAP shape."
    )


if base_values.ndim == 0:

    base_vector = np.full(
        TEST_ROWS,
        float(base_values),
    )

elif base_values.shape == (
    TEST_ROWS,
):

    base_vector = base_values

elif base_values.shape == (
    TEST_ROWS,
    1,
):

    base_vector = base_values[:, 0]

else:

    raise RuntimeError(
        "Unexpected base-value shape: "
        f"{base_values.shape}"
    )


# ============================================================
# ADDITIVITY
# ============================================================

raw_margin = booster.inplace_predict(
    X_small,
    predict_type="margin",
)


raw_margin = np.asarray(
    raw_margin,
    dtype=np.float64,
).reshape(-1)


reconstructed = (
    base_vector
    +
    values.sum(
        axis=1
    )
)


error = np.abs(
    reconstructed
    -
    raw_margin
)


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    "Mean SHAP reconstruction error:",
    f"{error.mean():.16e}"
)

print(
    "Max SHAP reconstruction error:",
    f"{error.max():.16e}"
)


if not np.allclose(
    reconstructed,
    raw_margin,
    atol=1e-2,
    rtol=1e-2,
):

    raise RuntimeError(
        "Booster interventional SHAP "
        "additivity failed."
    )


print()
print(
    "Booster interventional SHAP:",
    "PASS"
)

print(
    "512-row background:",
    "PASS"
)

print(
    "Raw-margin additivity:",
    "PASS"
)

print(
    "Model retrained:",
    False
)

print(
    "Cycle25 read:",
    False
)

print(
    "Artifacts modified:",
    False
)


print()
print(
    "DIAGNOSTIC COMPLETE"
)