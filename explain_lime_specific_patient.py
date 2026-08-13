from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


# =========================================================
# FILE PATHS
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = PROJECT_DIR / "CDSS_Dashboard"
MODEL_DIR = DASHBOARD_DIR / "models"

DATA_PATH = PROJECT_DIR / "lidc_model_dataset_binary.csv"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
IMPUTER_PATH = MODEL_DIR / "imputer.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"

OUTPUT_PATH = (
    PROJECT_DIR
    / "lime_LIDC-IDRI-0001_nodule_1.html"
)


# =========================================================
# SELECT THE PATIENT AND NODULE
# =========================================================
PATIENT_ID = "LIDC-IDRI-0001"
NODULE_NUMBER = 1


# =========================================================
# LOAD DATA AND SAVED MODEL FILES
# =========================================================
df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)
feature_names = list(
    joblib.load(FEATURE_NAMES_PATH)
)


# =========================================================
# FIND THE EXACT PATIENT RECORD
# =========================================================
selected_rows = df[
    (df["patient_id"].astype(str) == PATIENT_ID)
    & (
        df["nodule_number"].astype(int)
        == NODULE_NUMBER
    )
].copy()

if selected_rows.empty:
    raise ValueError(
        f"No record was found for {PATIENT_ID}, "
        f"nodule {NODULE_NUMBER}."
    )

selected_row = selected_rows.iloc[0]


# =========================================================
# PREPARE THE FULL DATASET FOR LIME
# =========================================================
X = df[feature_names].copy()

X_imputed = imputer.transform(X)

selected_case = pd.DataFrame(
    [selected_row[feature_names].to_dict()],
    columns=feature_names,
)

selected_case_imputed = imputer.transform(
    selected_case
)[0]


# =========================================================
# CREATE THE LIME EXPLAINER
# =========================================================
explainer = LimeTabularExplainer(
    training_data=np.asarray(X_imputed),
    feature_names=feature_names,
    class_names=[
        "Low Risk",
        "Suspicious Risk",
    ],
    mode="classification",
    discretize_continuous=True,
    random_state=42,
)


# =========================================================
# GENERATE THE PATIENT-SPECIFIC EXPLANATION
# =========================================================
explanation = explainer.explain_instance(
    data_row=selected_case_imputed,
    predict_fn=model.predict_proba,
    num_features=min(
        10,
        len(feature_names),
    ),
    num_samples=5000,
)


# =========================================================
# DISPLAY PREDICTION INFORMATION
# =========================================================
prediction = int(
    model.predict(
        selected_case_imputed.reshape(1, -1)
    )[0]
)

probabilities = model.predict_proba(
    selected_case_imputed.reshape(1, -1)
)[0]

predicted_label = (
    "Suspicious Risk"
    if prediction == 1
    else "Low Risk"
)

actual_label = selected_row.get(
    "risk_group",
    "Not available",
)

print("=" * 60)
print("LIME EXPLANATION GENERATED")
print("=" * 60)
print(f"Patient ID: {PATIENT_ID}")
print(f"Nodule number: {NODULE_NUMBER}")
print(f"Recorded dataset group: {actual_label}")
print(f"Predicted class: {predicted_label}")
print(
    f"Low-risk probability: "
    f"{probabilities[0]:.4f}"
)
print(
    f"Suspicious-risk probability: "
    f"{probabilities[1]:.4f}"
)


# =========================================================
# PRINT FEATURE CONTRIBUTIONS
# =========================================================
print("\nTop LIME feature contributions:")

for feature, weight in explanation.as_list():
    print(f"{feature}: {weight:+.4f}")


# =========================================================
# SAVE THE INTERACTIVE HTML FILE
# =========================================================
explanation.save_to_file(
    str(OUTPUT_PATH)
)

print("\nSaved as:")
print(OUTPUT_PATH)