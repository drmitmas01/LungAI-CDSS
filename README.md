# LungAI-CDSS

**LungAI-CDSS** is an Explainable Artificial Intelligence (XAI)-based Clinical Decision Support System research prototype developed for CT-based lung nodule malignancy-risk prediction.

The project forms part of an MSc dissertation titled:

**Design and Development of an Explainable AI-Based Clinical Decision Support System Prototype for Early Lung Cancer Detection Using CT-Derived Lung Nodule Features**

---

## Project Overview

The aim of this project is to investigate how machine learning, explainable AI and a clinician-facing interface can be integrated into a single research prototype for lung nodule malignancy-risk assessment.

The system uses structured radiological features derived from CT-identified lung nodules and produces a binary risk classification:

* **Low Risk**
* **Suspicious Risk**

The prototype is intended to support research into interpretable AI-assisted lung nodule assessment and is not designed to replace clinical judgement.

---

## Dataset

The project uses the **LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)** dataset.

The dataset contains thoracic CT scans and lung nodule annotations produced by multiple radiologists.

Radiological characteristics used in the project include features such as:

* Subtlety
* Internal structure
* Calcification
* Sphericity
* Margin
* Lobulation
* Spiculation
* Texture

The complete LIDC-IDRI CT/DICOM dataset is **not included in this GitHub repository**.

---

## Data Preparation

The modelling dataset was generated from extracted LIDC-IDRI lung nodule features.

The binary target was defined as:

* `0` = Low Risk
* `1` = Suspicious Risk

The target was derived from the mean radiologist malignancy score.

Predictor variables were separated from identifiers and outcome-related variables to avoid target leakage.

Missing predictor values were handled using **median imputation** within the machine-learning pipelines.

---

## Machine Learning Models

Four supervised machine-learning algorithms were developed and compared:

1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**
4. **LightGBM**

The models were evaluated using an 80:20 stratified training and testing split.

---

## Model Evaluation

Model performance was assessed using:

* Accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1-score
* ROC-AUC
* Confusion Matrix
* Classification Report

These metrics were used to compare predictive performance across the four machine-learning models.

---

## Explainable Artificial Intelligence

Explainability was incorporated into the system using:

### SHAP

SHAP was used to provide:

* Global feature importance
* Feature contribution analysis
* Nodule-level prediction explanations

### LIME

LIME was used to generate local explanations for selected individual lung nodule predictions.

These techniques help demonstrate how different radiological features influence model predictions.

---

## LungAI-CDSS Dashboard

A clinician-facing research prototype was developed using **Streamlit**.

The main application file is:

```bash
app.py
```

The dashboard provides functionality for presenting:

* CT-derived lung nodule characteristics
* Predicted malignancy-risk category
* Prediction probability
* Model performance information
* SHAP explanations
* LIME explanations
* CT image visualisation
* Lung nodule visualisation

The dashboard is intended to demonstrate how machine-learning predictions and explanation outputs can be presented within a single interface.

---

## Technologies Used

The project was developed using Python and associated machine-learning and visualisation libraries, including:

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* LightGBM
* SHAP
* LIME
* matplotlib
* Plotly
* Streamlit
* pydicom
* pylidc
* joblib
* Jupyter Notebook

---

## Installation

Clone the repository or download the project files.

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Using a Python virtual environment is recommended.

---

## Running the Streamlit Application

Navigate to the project directory and run:

```bash
python -m streamlit run app.py
```

Alternatively:

```bash
streamlit run app.py
```

The application should then open in a web browser, normally at:

```text
http://localhost:8501
```

---

## Jupyter Notebook

The repository also contains the Jupyter Notebook used to demonstrate the machine-learning workflow:

```text
LungAI_CDSS_Demo.ipynb
```

The notebook includes stages such as:

* Dataset loading
* Exploratory data analysis
* Data preprocessing
* Predictor and target preparation
* Train-test splitting
* Model development
* Model evaluation
* SHAP analysis
* LIME explanation
* Model export
* Reproducibility checks

---

## Repository Contents

The repository contains files relating to:

* LIDC-IDRI feature extraction
* Dataset preparation
* Machine-learning model training
* Model evaluation
* SHAP explainability
* LIME explainability
* Model export
* Jupyter Notebook analysis
* Streamlit dashboard implementation
* Model performance outputs
* Research figures and visualisations

---

## Research Contribution

The contribution of this project is primarily **integrative rather than algorithmic**.

The project does not introduce a new machine-learning algorithm. Instead, it combines:

**CT-derived lung nodule features → comparative machine-learning models → SHAP and LIME explainability → clinician-facing Streamlit dashboard**

within a single research prototype.

---

## Limitations

The project has several important limitations:

* It uses the LIDC-IDRI research dataset rather than a live NHS clinical dataset.
* Radiologist malignancy ratings are not equivalent to uniformly confirmed histopathological diagnoses.
* External clinical validation has not been performed.
* A formal clinician usability study was not conducted.
* Comprehensive demographic fairness analysis was not possible because of limited demographic and socioeconomic information in LIDC-IDRI.
* The prototype is not integrated with an NHS Electronic Health Record system.
* The system has not undergone regulatory or clinical-safety approval.

---

## Clinical Disclaimer

**LungAI-CDSS is a research-stage academic prototype only.**

It has not been clinically validated, approved as a medical device, or authorised for use in patient care.

The software must **not** be used for:

* Clinical diagnosis
* Treatment decisions
* Patient management
* Referral decisions
* Replacement of radiological interpretation
* Replacement of multidisciplinary clinical judgement

Any future clinical implementation would require external validation, clinical evaluation, regulatory assessment and appropriate healthcare governance approval.

---

## Author

**Samuel Idowu**

MSc Health Analytics and Technologies
Centre for Applied Computer Science
University of Greater Manchester

---

## Academic Purpose

This repository contains source code and supporting materials developed as part of an MSc dissertation research project.

The repository is provided for academic, research and educational purposes.
