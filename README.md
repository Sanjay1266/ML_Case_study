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
  6. Random Forest Classifier
  7. AdaBoost Classifier
  8. Gradient Boosting Classifier
  9. Bagging Classifier
  10. Multi-Layer Perceptron (MLP) Classifier
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score (Weighted), ROC-AUC, Confusion Matrix
- **Interactive Interface**: In-notebook `ipywidgets` GUI (Section 25) & standalone Streamlit app (`app/streamlit_app.py`)

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
# To run classification (10 algorithms + in-notebook GUI)
jupyter notebook notebooks/classification.ipynb

# To run regression
jupyter notebook notebooks/regression.ipynb
```
*Alternatively, open any notebook directly in VS Code and select the `venv` Python kernel.*

### 5. Run the Standalone Streamlit Web Application
Launch the interactive Term Deposit Subscription Predictor:
```bash
streamlit run app/streamlit_app.py
```
This opens the web interface in your browser at `http://localhost:8501`, allowing you to:
- Test customer predictions across all 10 classification models.
- Adjust customer demographic, financial, and marketing contact parameters.
- Inspect real-time probability estimates and customize decision thresholds.
- View comparative benchmark tables and feature importance insights.

---

## Contributors

| Name | Roll Number | GitHub |
| :--- | :--- | :--- |
| **DARISA NAGA JYOTHI** | `CB.SC.U4CSE24214` | [@jyothidarisa](https://github.com/jyothidarisa) |
| **Sanjay S** | `CB.SC.U4CSE24249` | [@Sanjay1266](https://github.com/Sanjay1266) |
| **MONITHAA P** | `CB.SC.U4CSE24262` | [@Monithaa-P](https://github.com/Monithaa-P) |
