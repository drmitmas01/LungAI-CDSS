import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from lime.lime_tabular import LimeTabularExplainer


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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Reset y_test index so case_index works cleanly
y_test = y_test.reset_index(drop=True)

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

# Train XGBoost model
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_imputed, y_train)

# Create LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train_imputed.values,
    feature_names=list(X.columns),
    class_names=["Low risk", "Suspicious risk"],
    mode="classification",
    discretize_continuous=True
)

# Choose one test case to explain
case_index = 0

case_to_explain = X_test_imputed.iloc[case_index].values

exp = explainer.explain_instance(
    data_row=case_to_explain,
    predict_fn=model.predict_proba,
    num_features=10
)

# Save explanation
exp.save_to_file("lime_explanation_case_0.html")

prediction = model.predict(X_test_imputed.iloc[[case_index]])[0]
probability = model.predict_proba(X_test_imputed.iloc[[case_index]])[0]

print("LIME explanation saved as lime_explanation_case_0.html")
print("Actual label:", y_test.iloc[case_index])
print("Predicted label:", prediction)
print("Probability low risk:", round(probability[0], 4))
print("Probability suspicious risk:", round(probability[1], 4))