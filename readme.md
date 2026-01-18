# Employee Attrition Prediction

An end-to-end Machine Learning project that predicts employee attrition (Yes / No) using structured HR data.
The project includes data preprocessing, model training, hyperparameter tuning, evaluation, and deployment through an interactive Gradio web application.

---

## Project Overview

Employee attrition is a critical HR problem where organizations aim to identify employees who are at risk of leaving.
This project builds a robust ML pipeline to predict attrition and deploys it as a user-friendly web app.

---

## Repository Structure

* `Employee_Attrition_Prediction.ipynb`
  Main notebook containing:

  * Data loading and exploration
  * Data preprocessing
  * Feature engineering
  * Pipeline creation
  * Model training and cross-validation
  * Hyperparameter tuning
  * Best model selection
  * Final evaluation

* `app.py`
  Gradio-based web application with a wizard-style UI (Next / Back buttons) for predicting employee attrition.

* `employee_attrition_bundle.pkl`
  Serialized deployment bundle containing:

  * Trained sklearn pipeline
  * Feature column order
  * Numerical and categorical column lists
  * UI dropdown choices

* `transformers.py`
  Custom sklearn transformers used for feature engineering and outlier handling.

* `requirements.txt`
  Python dependencies required to run the notebook and web app.

* `.gitignore`
  Ignore rules for virtual environments, cache files, and local artifacts.

---

## Machine Learning Workflow

### 1. Data Loading

* Load employee attrition dataset
* Validate shape and preview rows

### 2. Data Preprocessing

* Missing value imputation
* Categorical encoding (OneHotEncoder)
* Numerical feature scaling
* Outlier handling using IQR clipping
* Feature engineering using custom transformers

### 3. Pipeline Creation

* Unified sklearn Pipeline using ColumnTransformer

### 4. Model Training

* Logistic Regression as baseline and final model

### 5. Cross-Validation

* Stratified cross-validation for robustness

### 6. Hyperparameter Tuning

* GridSearchCV for optimal hyperparameters

### 7. Best Model Selection

* Model selected based on ROC-AUC and PR-AUC

### 8. Model Evaluation

* Accuracy
* ROC-AUC
* PR-AUC
* Confusion Matrix
* Classification Report

### 9. Deployment

* Model serialized using pickle
* Deployed via Gradio
* Hosted on Hugging Face Spaces

---

## Running the Project Locally

### Step 1: Create Virtual Environment (Windows)

```bash
python -m venv .venv
.\.venv\Scripts\Activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Gradio App

```bash
python app.py
```

Open the local URL displayed in the terminal.

---

## Gradio Application Features

* Wizard-style interface with **Next** and **Back** navigation
* Inputs grouped into:

  * Personal
  * Job
  * Compensation
  * Satisfaction
  * Experience
  * Other
* Prediction output includes:

  * Attrition prediction (Yes / No)
  * Probability score
  * Risk band (Low / Medium / High)
  * Summary of user inputs

---

## Hugging Face Spaces Deployment Note

To avoid conflict with Hugging Face's `transformers` library:

* Rename `transformers.py` → `custom_transformers.py`
* Update import in `app.py`:

```python
from custom_transformers import RatioFeatureEngineer, IQRClipper
```

Optionally add:

```python
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
```

---

## Author

**Mohammad Arifur Rahman**
Employee Attrition Prediction Project

---

If you want next:

* 🔹 **Short README for Hugging Face Space**
* 🔹 **Professional GitHub description (1–2 lines)**
* 🔹 **Badges + demo screenshot section**
* 🔹 **Resume-ready project description**

Just tell me.
