import joblib
import pandas as pd

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

# Save feature names
feature_names = list(X.columns)

# Handle missing values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# Train model
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X, y)

# Save everything
joblib.dump(model, "CDSS_Dashboard/models/xgboost_model.pkl")
joblib.dump(imputer, "CDSS_Dashboard/models/imputer.pkl")
joblib.dump(feature_names, "CDSS_Dashboard/models/feature_names.pkl")

print("===================================")
print("Model saved successfully!")
print("===================================")
print("xgboost_model.pkl")
print("imputer.pkl")
print("feature_names.pkl")