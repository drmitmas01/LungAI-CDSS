import pandas as pd

# Read CSV
df = pd.read_csv("model_performance_results.csv")

# Save as Excel
df.to_excel("model_performance_results.xlsx", index=False)

print("Excel file created successfully!")