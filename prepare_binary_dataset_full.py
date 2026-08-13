import pandas as pd

# Load extracted LIDC-IDRI nodule features
df = pd.read_csv("lidc_nodule_features_full.csv")

print("Original dataset shape:", df.shape)

# Remove rows without malignancy score
df = df.dropna(subset=["mean_malignancy_score"])

# Create binary target label
# 0 = low risk
# 1 = suspicious risk
df["target"] = df["mean_malignancy_score"].apply(
    lambda x: 0 if x <= 2 else 1
)

df["risk_group"] = df["target"].map({
    0: "Low risk",
    1: "Suspicious risk"
})

# Save final ML dataset
df.to_csv("lidc_model_dataset_binary.csv", index=False)

print("Final dataset shape:", df.shape)
print(df["risk_group"].value_counts())
print("Saved as lidc_model_dataset_binary.csv")