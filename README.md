# Bank Customer Churn Predictor

This repository contains a machine learning project that predicts customer churn (attrition) for a retail bank. It features an **XGBoost Classifier** optimized via **RandomizedSearchCV** and trained on resampled data using **SMOTE** to handle class imbalance. The final model is deployed as an interactive, user-friendly **Streamlit web application**.

---

## 📁 Repository Structure

```text
├── .gitignore                      # Git ignore rules for Python and IDEs
├── README.md                       # Documentation (this file)
├── requirements.txt                # Python dependencies
├── app.py                          # Streamlit web application
├── train.py                        # Model training and artifact generation script
├── data/
│   ├── Churn_Modelling.csv         # Raw customer churn dataset
│   └── bank_churn_data.xlsx        # Full dataset with model predictions exported
├── models/
│   ├── best_model.pkl              # Serialized best XGBoost model
│   └── scaler.pkl                  # Serialized MinMaxScaler
├── assets/
│   ├── pic_1.png                   # Image for Streamlit sidebar
│   ├── pic_2.png                   # Image for Streamlit header
│   └── feature_importance.xlsx     # Feature importances exported from the model
└── notebooks/
    └── Predicting_churn.ipynb      # Original Jupyter notebook with EDA and prototyping
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/arijit1204/Bank-Churn-Prediction.git
cd Bank-Churn-Prediction
```

### 2. Set Up Virtual Environment & Install Dependencies
Create a virtual environment to manage dependencies locally:

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
Launch the interactive web application to perform predictions:
```bash
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`.

---

## 🛠️ Model Training & Artifact Generation

If you want to retrain the model, modify hyperparameters, or regenerate the pickles and evaluation excels, run the training pipeline:
```bash
python train.py
```

### What `train.py` does:
1. Loads the raw dataset `data/Churn_Modelling.csv`.
2. Encodes categorical columns (`Geography`, `Gender`, `HasCrCard`, `IsActiveMember`).
3. Scales continuous numerical features (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`) using `MinMaxScaler`.
4. Splits data (80% train, 20% test) and applies **SMOTE** (Synthetic Minority Over-sampling Technique) to resolve class imbalance in training set.
5. Performs **RandomizedSearchCV** (3-fold cross-validation) to find optimal XGBoost hyperparameters (tuning F1-score).
6. Exports predictions to `data/bank_churn_data.xlsx` and feature importance rankings to `assets/feature_importance.xlsx`.
7. Saves the trained model and scaler to the `models/` directory.

---

## 🚀 Model Details & Performance
- **Algorithm:** Extreme Gradient Boosting (XGBoost Classifier)
- **Features Used:** Credit Score, Age, Tenure, Account Balance, Number of Products, Estimated Salary, Geography (France/Germany/Spain), Gender, Credit Card Ownership, Active Member Status.
- **Handling Imbalance:** SMOTE oversampling
- **Evaluation Metric:** F1-score optimization
- **Accuracy:** ~84.95% on test partition
- **Key Predictors:** Age, Number of Products, Active Membership status, Account Balance.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
