import numpy as np

# Fix for older pylidc code with newer NumPy versions
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

import pylidc as pl
import pandas as pd

rows = []

scans = pl.query(pl.Scan).all()
print("Total scans:", len(scans))


def safe_mean(cluster, attr):
    values = []
    for ann in cluster:
        try:
            value = getattr(ann, attr)
            if value is not None:
                values.append(float(value))
        except Exception:
            pass
    return np.mean(values) if values else np.nan


def safe_std(cluster, attr):
    values = []
    for ann in cluster:
        try:
            value = getattr(ann, attr)
            if value is not None:
                values.append(float(value))
        except Exception:
            pass
    return np.std(values) if values else np.nan


features = [
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]

skipped_scans = []

for index, scan in enumerate(scans, start=1):
    print(f"Processing scan {index}/{len(scans)}: {scan.patient_id}")

    try:
        clusters = scan.cluster_annotations()
    except Exception as e:
        print(f"Skipping {scan.patient_id}: {e}")
        skipped_scans.append(scan.patient_id)
        continue

    for nodule_number, cluster in enumerate(clusters, start=1):
        if len(cluster) == 0:
            continue

        mean_malignancy = safe_mean(cluster, "malignancy")

        row = {
            "patient_id": scan.patient_id,
            "nodule_number": nodule_number,
            "number_of_radiologists": len(cluster),
            "mean_malignancy_score": mean_malignancy,
        }

        for feature in features:
            row[feature] = safe_mean(cluster, feature)
            row[f"{feature}_std"] = safe_std(cluster, feature)

        rows.append(row)

df = pd.DataFrame(rows)

# To scan all the   scans and extract features, the following code can be used. 
for scan in scans:
    annotation_clusters = scan.cluster_annotations()

    for nodule_number, annotations in enumerate(
        annotation_clusters,
        start=1,
    ):
        features = {
            "patient_id": scan.patient_id,
            "nodule_number": nodule_number,
            "subtlety": np.mean(
                [annotation.subtlety for annotation in annotations]
            ),
            "internal_structure": np.mean(
                [
                    annotation.internalStructure
                    for annotation in annotations
                ]
            ),
            "calcification": np.mean(
                [annotation.calcification for annotation in annotations]
            ),
            "sphericity": np.mean(
                [annotation.sphericity for annotation in annotations]
            ),
            "margin": np.mean(
                [annotation.margin for annotation in annotations]
            ),
            "lobulation": np.mean(
                [annotation.lobulation for annotation in annotations]
            ),
            "spiculation": np.mean(
                [annotation.spiculation for annotation in annotations]
            ),
            "texture": np.mean(
                [annotation.texture for annotation in annotations]
            ),
            "malignancy": np.mean(
                [annotation.malignancy for annotation in annotations]
            ),
        }

df.to_csv("lidc_nodule_features_full.csv", index=False)

print("\nFeature extraction complete")
print("---------------------------")
print("Final dataset shape:", df.shape)
print("Skipped scans:", len(skipped_scans))
print(df.head())
print("Saved as lidc_nodule_features_full.csv")