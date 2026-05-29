# 🏦 Bank Customer Churn Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clientsafe.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning project that predicts customer churn (attrition) for a retail bank. Built with an **XGBoost Classifier** optimized via **RandomizedSearchCV**, trained on SMOTE-resampled data to handle class imbalance, and deployed as an interactive **Streamlit web application**.

### 🔗 **[Try the Live App →](https://clientsafe.streamlit.app/)**

---

## ✨ Features

- **Real-time churn prediction** — Adjust customer attributes via sidebar controls and get instant risk assessment
- **Feature importance visualization** — Interactive Plotly bar chart showing which factors drive churn the most
- **Probability breakdown** — View both churn and retention probabilities with styled result cards
- **Responsive UI** — Wide layout with sidebar inputs and a two-column results dashboard

---

## 📁 Repository Structure

```text
Bank-Churn-Prediction/
├── .gitignore                      # Git ignore rules for Python, IDEs, and secrets
├── .streamlit/
│   └── config.toml                 # Streamlit theme and server configuration
├── LICENSE                         # MIT License
├── README.md                       # Documentation (this file)
├── requirements.txt                # Python dependencies
├── app.py                          # Streamlit web application
├── train.py                        # Model training and artifact generation script
├── data/
│   ├── Churn_Modelling.csv         # Raw customer churn dataset (10,000 records)
│   └── bank_churn_data.xlsx        # Full dataset with model predictions exported
├── models/
│   ├── best_model.pkl              # Serialized best XGBoost model
│   └── scaler.pkl                  # Serialized MinMaxScaler
├── assets/
│   ├── pic_1.png                   # Image for Streamlit sidebar
│   ├── pic_2.png                   # Image for Streamlit header
│   └── feature_importance.xlsx     # Feature importances exported from the model
└── notebooks/
    └── Predicting_churn.ipynb      # Jupyter notebook with EDA and prototyping
```

---

## ⚡ Quick Start (Local Development)

### 1. Clone the Repository
```bash
git clone https://github.com/arijit1204/Bank-Churn-Prediction.git
cd Bank-Churn-Prediction
```

### 2. Set Up Virtual Environment & Install Dependencies

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```
The app opens in your browser at `http://localhost:8501`.

---

## 🛠️ Model Training & Artifact Generation

To retrain the model, modify hyperparameters, or regenerate all artifacts:
```bash
python train.py
```

### Training Pipeline Steps:
| Step | Description |
|------|-------------|
| 1 | Load raw dataset `data/Churn_Modelling.csv` |
| 2 | One-hot encode categorical features (`Geography`, `Gender`, `HasCrCard`, `IsActiveMember`) |
| 3 | Scale numeric features (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`) with `MinMaxScaler` |
| 4 | Split data (80/20) and apply **SMOTE** oversampling on training set |
| 5 | Run **RandomizedSearchCV** (3-fold CV, 50 iterations) to find optimal XGBoost hyperparameters |
| 6 | Export predictions to `data/bank_churn_data.xlsx` and feature importances to `assets/feature_importance.xlsx` |
| 7 | Save trained model and scaler to `models/` |

---

## 🚀 Model Details & Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | XGBoost Classifier |
| **Class Imbalance Handling** | SMOTE oversampling |
| **Hyperparameter Tuning** | RandomizedSearchCV (F1-score) |
| **Test Accuracy** | ~84.95% |

### Features Used
Credit Score · Age · Tenure · Account Balance · Number of Products · Estimated Salary · Geography (France / Germany / Spain) · Gender · Credit Card Ownership · Active Member Status

### Key Predictors
Age · Number of Products · Active Membership Status · Account Balance

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| **ML Framework** | XGBoost, scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Class Imbalance** | imbalanced-learn (SMOTE) |
| **Visualization** | Plotly |
| **Web App** | Streamlit |
| **Deployment** | Streamlit Community Cloud |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/arijit1204">Arijit Dutta</a>
</p>
