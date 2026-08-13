# LungAI-CDSS

LungAI-CDSS is an Explainable Artificial Intelligence-based Clinical Decision Support System research prototype developed for CT-derived lung nodule malignancy-risk prediction.

## Dissertation Project

**Title:** Design and Development of an Explainable AI-Based Clinical Decision Support System Prototype for Early Lung Cancer Detection Using CT-Derived Lung Nodule Features

## Dataset

The project uses the LIDC-IDRI dataset containing thoracic CT scans and radiologist-annotated lung nodule characteristics.

The full LIDC-IDRI dataset is not included in this repository.

## Machine Learning Models

Four supervised machine-learning models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## Explainable AI

Model predictions were interpreted using:

- SHAP
- LIME

## LungAI-CDSS Dashboard

A Streamlit-based clinician-facing research prototype was developed to display:

- CT-derived nodule characteristics
- malignancy-risk prediction
- prediction probability
- SHAP explanations
- LIME explanations
- CT image visualisation
- nodule visualisation
- model performance information

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
