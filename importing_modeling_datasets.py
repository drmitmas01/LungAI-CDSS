from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import joblib
import numpy as np
import shap

from lightgbm import LGBMClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier







# Identify the current project directory
PROJECT_DIR = Path.cwd()

# Create the path to the processed modelling dataset
dataset_path = PROJECT_DIR / "lidc_model_dataset_binary.csv"

# Confirm dataset exists
if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

# Load dataset
df = pd.read_csv(dataset_path)

print("Dataset path:", dataset_path)
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

# -------------------------------------------------
# Exploratory Data Analysis
# -------------------------------------------------

print("\nNumber of observations:", df.shape[0])
print("Number of variables:", df.shape[1])

print("\nMissing values in each variable:")
print(df.isnull().sum())

print("\nDistribution of risk groups:")
print(df["risk_group"].value_counts())


# Visualise target-class distribution
risk_counts = df["risk_group"].value_counts()

ax = risk_counts.plot(
    kind="bar",
    figsize=(7, 5),
)

plt.title("Distribution of Lung Nodule Risk Categories")
plt.xlabel("Risk Group")
plt.ylabel("Number of Nodules")
plt.xticks(rotation=0)

for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Prepare predictors and target variable
# -------------------------------------------------

exclude_columns = [
    "patient_id",
    "nodule_number",
    "mean_malignancy_score",
    "target",
    "risk_group",
]

# Predictor variables
X = df.drop(
    columns=exclude_columns
).copy()

# Binary target variable
y = df["target"].astype(int).copy()


# -------------------------------------------------
# Training and testing split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining and testing split completed.")

print("Training predictor shape:", X_train.shape)
print("Testing predictor shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

print("\nTraining class proportions:")
print(y_train.value_counts(normalize=True).sort_index().round(4))

print("\nTesting class proportions:")
print(y_test.value_counts(normalize=True).sort_index().round(4))

print("\nPredictor matrix shape:", X.shape)
print("Target vector shape:", y.shape)

print("\nPredictor variables used:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts().sort_index())


# -------------------------------------------------
# Define machine-learning pipelines
# -------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ]),

    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]),

    "XGBoost": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                eval_metric="logloss",
            ),
        ),
    ]),

    "LightGBM": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                random_state=42,
                class_weight="balanced",
                verbosity=-1,
            ),
        ),
    ]),
}

print("\nMachine-learning pipelines created:")

for model_name in models:
    print("-", model_name)

# -------------------------------------------------
# Train and evaluate the models
# -------------------------------------------------

results = []
trained_models = {}

for model_name, model in models.items():
    print("\n" + "=" * 65)
    print("Training:", model_name)
    print("=" * 65)

    # Train model
    model.fit(X_train, y_train)

    # Retain the fitted pipeline
    trained_models[model_name] = model

    # Generate predictions
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Extract confusion-matrix counts
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    sensitivity = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "True Negative": tn,
        "False Positive": fp,
        "False Negative": fn,
        "True Positive": tp,
    })

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Low risk",
                "Suspicious risk",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("ROC-AUC:", round(roc_auc, 4))


# Create model-comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False,
).reset_index(drop=True)

print("\nModel performance comparison:")
print(
    results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1-score",
            "ROC-AUC",
        ]
    ].round(4).to_string(index=False)
)

# Save results
results_df.to_csv(
    "model_performance_results_implementation.csv",
    index=False,
)

print(
    "\nResults saved as "
    "model_performance_results_implementation.csv"
)

# -------------------------------------------------
# SHAP explainability for XGBoost
# -------------------------------------------------

xgb_pipeline = trained_models["XGBoost"]

xgb_imputer = xgb_pipeline.named_steps[
    "imputer"
]

xgb_model = xgb_pipeline.named_steps[
    "model"
]

# Apply the fitted imputer
X_train_imputed = xgb_imputer.transform(
    X_train
)

X_test_imputed = xgb_imputer.transform(
    X_test
)

X_train_imputed_df = pd.DataFrame(
    X_train_imputed,
    columns=X.columns,
    index=X_train.index,
)

X_test_imputed_df = pd.DataFrame(
    X_test_imputed,
    columns=X.columns,
    index=X_test.index,
)

# Create the SHAP explainer
shap_explainer = shap.TreeExplainer(
    xgb_model
)

shap_values = shap_explainer.shap_values(
    X_test_imputed_df
)

print("\nSHAP values generated successfully.")

# Create and save the global SHAP plot
shap.summary_plot(
    shap_values,
    X_test_imputed_df,
    show=False,
)

plt.tight_layout()

plt.savefig(
    "shap_summary_plot_implementation.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "SHAP summary plot saved as "
    "shap_summary_plot_implementation.png"
)

# -------------------------------------------------
# LIME explanation for a selected lung nodule
# -------------------------------------------------

selected_patient_id = "LIDC-IDRI-0001"
selected_nodule_number = 1

selected_rows = df[
    (
        df["patient_id"].astype(str)
        == selected_patient_id
    )
    & (
        pd.to_numeric(
            df["nodule_number"],
            errors="coerce",
        )
        == selected_nodule_number
    )
].copy()

if selected_rows.empty:
    raise ValueError(
        f"No record found for {selected_patient_id}, "
        f"nodule {selected_nodule_number}."
    )

selected_case = selected_rows[
    X.columns
].iloc[[0]]

selected_case_imputed = xgb_imputer.transform(
    selected_case
)

# Create LIME explainer
lime_explainer = LimeTabularExplainer(
    training_data=np.asarray(
        X_train_imputed_df
    ),
    feature_names=list(X.columns),
    class_names=[
        "Low risk",
        "Suspicious risk",
    ],
    mode="classification",
    discretize_continuous=True,
    random_state=42,
)

# Explain selected nodule
lime_explanation = lime_explainer.explain_instance(
    data_row=selected_case_imputed[0],
    predict_fn=xgb_model.predict_proba,
    num_features=min(10, len(X.columns)),
    num_samples=5000,
)

print("\nLIME feature contributions:")

for feature, weight in lime_explanation.as_list():
    print(f"{feature}: {weight:+.4f}")

# Save HTML explanation
lime_explanation.save_to_file(
    "lime_LIDC_IDRI_0001_implementation.html"
)

# Save image explanation
lime_figure = lime_explanation.as_pyplot_figure()

lime_figure.set_size_inches(10, 6)

plt.tight_layout()

plt.savefig(
    "lime_LIDC_IDRI_0001_implementation.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "LIME explanation saved successfully."
)

# -------------------------------------------------
# Save the trained model and feature names
# -------------------------------------------------

xgb_pipeline_path = (
    PROJECT_DIR
    / "xgboost_lidc_pipeline_implementation.joblib"
)

feature_names_path = (
    PROJECT_DIR
    / "model_feature_names_implementation.joblib"
)

joblib.dump(
    xgb_pipeline,
    xgb_pipeline_path,
)

joblib.dump(
    list(X.columns),
    feature_names_path,
)

print("\nXGBoost pipeline saved to:")
print(xgb_pipeline_path)

print("\nFeature names saved to:")
print(feature_names_path)

# -------------------------------------------------
# Verify the exported model
# -------------------------------------------------

reloaded_pipeline = joblib.load(
    xgb_pipeline_path
)

original_predictions = xgb_pipeline.predict(
    X_test
)

reloaded_predictions = reloaded_pipeline.predict(
    X_test
)

predictions_match = np.array_equal(
    original_predictions,
    reloaded_predictions,
)

print(
    "\nReloaded model predictions match "
    "the original predictions:",
    predictions_match,
)

if not predictions_match:
    raise RuntimeError(
        "The exported pipeline did not reproduce "
        "the original predictions."
    )