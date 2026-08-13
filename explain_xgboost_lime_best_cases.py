import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer


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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_test = y_test.reset_index(drop=True)
X_test_original = X_test.reset_index(drop=True)

imputer = SimpleImputer(strategy="median")

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns
)

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns
)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_imputed, y_train)

probs = model.predict_proba(X_test_imputed)

# Find strongest suspicious-risk prediction
suspicious_index = np.argmax(probs[:, 1])

# Find strongest low-risk prediction
low_risk_index = np.argmax(probs[:, 0])

explainer = LimeTabularExplainer(
    training_data=X_train_imputed.values,
    feature_names=list(X.columns),
    class_names=["Low risk", "Suspicious risk"],
    mode="classification",
    discretize_continuous=True
)

def explain_case(case_index, filename):
    exp = explainer.explain_instance(
        data_row=X_test_imputed.iloc[case_index].values,
        predict_fn=model.predict_proba,
        num_features=10
    )

    exp.save_to_file(filename)

    prediction = model.predict(X_test_imputed.iloc[[case_index]])[0]
    probability = model.predict_proba(X_test_imputed.iloc[[case_index]])[0]

    print("\nSaved:", filename)
    print("Case index:", case_index)
    print("Actual label:", y_test.iloc[case_index])
    print("Predicted label:", prediction)
    print("Probability low risk:", round(probability[0], 4))
    print("Probability suspicious risk:", round(probability[1], 4))


explain_case(suspicious_index, "lime_high_confidence_suspicious_case.html")
explain_case(low_risk_index, "lime_high_confidence_low_risk_case.html")