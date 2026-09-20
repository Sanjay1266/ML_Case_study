# Machine Learning Case Study (23CSE301 Capstone Project)

This repository contains end-to-end Machine Learning pipelines covering both **Classification** and **Regression** tracks, including comprehensive data audits, exploratory data analysis (EDA), leakage-free preprocessing pipelines, model training, hyperparameter tuning, and comparative performance evaluations.

---

## Project Structure

```text
ML_Case_study/
├── app/                        # Application / deployment interface
├── data/
│   ├── bank-full.csv           # Bank marketing dataset (Classification)
│   └── OList-Ecommerce.csv     # Brazilian E-Commerce dataset (Regression)
├── models/                     # Saved model checkpoints and artifacts
├── notebooks/
│   ├── classification.ipynb    # Classification track (5 models)
│   └── regression.ipynb        # Regression track (10 models)
├── LICENSE                     # Project license
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Tracks & Models Evaluated

### 1. Classification Track (`notebooks/classification.ipynb`)
- **Dataset**: Bank Marketing (`data/bank-full.csv`)
- **Target**: Term Deposit Subscription (`y`: `yes` / `no`)
- **Algorithms**:
  1. Logistic Regression
  2. K-Nearest Neighbors (KNN)
  3. Gaussian Naive Bayes
  4. Decision Tree Classifier
  5. Support Vector Machine (SVC)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

### 2. Regression Track (`notebooks/regression.ipynb`)
- **Dataset**: OList Brazilian E-Commerce (`data/OList-Ecommerce.csv`)
- **Target**: Product Price (`price` - continuous)
- **Algorithms**:
  1. Linear Regression
  2. Ridge Regression
  3. Lasso Regression
  4. ElasticNet
  5. Decision Tree Regressor
  6. Random Forest Regressor
  7. Gradient Boosting Regressor
  8. Support Vector Regressor (SVR)
  9. K-Nearest Neighbors Regressor (KNN)
  10. Polynomial Regression
- **Evaluation Metrics**: $R^2$ Score, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)

---

## Quick Start (Local Setup)

### 1. Clone Repository & Navigate
```bash
git clone https://github.com/Sanjay1266/ML_Case_study.git
cd ML_Case_study
```

### 2. Create & Activate Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Notebooks
Launch Jupyter Notebook or JupyterLab:
```bash
# To run classification
jupyter notebook notebooks/classification.ipynb

# To run regression
jupyter notebook notebooks/regression.ipynb
```
*Alternatively, open any notebook directly in VS Code and select the `venv` Python kernel.*

---

## Contributors

| Name | Roll Number | GitHub |
| :--- | :--- | :--- |
| **DARISA NAGA JYOTHI** | `CB.SC.U4CSE24214` | [@jyothidarisa](https://github.com/jyothidarisa) |
| **Sanjay S** | `CB.SC.U4CSE24249` | [@Sanjay1266](https://github.com/Sanjay1266) |
| **MONITHAA P** | `CB.SC.U4CSE24262` | [@Monithaa-P](https://github.com/Monithaa-P) |
