import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("lidc_model_dataset_binary.csv")

exclude_columns = [
    "patient_id",
    "nodule_number",
    "mean_malignancy_score",
    "target",
    "risk_group"
]

X = df.drop(columns=exclude_columns)
y = df["target"]

# Split data exactly like before
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Impute missing values
imputer = SimpleImputer(strategy="median")

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns
)

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns
)

# Train best model: XGBoost
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_imputed, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_imputed)

# SHAP summary plot
shap.summary_plot(shap_values, X_test_imputed, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.close()

# SHAP bar plot
shap.summary_plot(shap_values, X_test_imputed, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance_bar.png", dpi=300, bbox_inches="tight")
plt.close()

print("SHAP plots saved:")
print("shap_summary_plot.png")
print("shap_feature_importance_bar.png")