import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def train_model():
    print("Step 1: Loading raw dataset...")
    raw_csv_path = os.path.join("data", "Churn_Modelling.csv")
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Missing {raw_csv_path}. Please place it in the data folder.")
    
    raw_raw = pd.read_csv(raw_csv_path, encoding="latin1")
    
    print("Step 2: Selecting relevant features...")
    features_to_keep = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited'
    ]
    raw_raw_v = raw_raw[features_to_keep]
    
    print("Step 3: One-hot encoding categorical variables...")
    new_raw_data = pd.get_dummies(raw_raw_v, columns=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])
    
    # Ensure all one-hot encoded columns are present in a standard order
    # (matches features defined in Streamlit app)
    expected_one_hot_columns = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary",
        "Geography_France", "Geography_Germany", "Geography_Spain",
        "Gender_Female", "Gender_Male",
        "HasCrCard_0", "HasCrCard_1",
        "IsActiveMember_0", "IsActiveMember_1",
        "Exited"
    ]
    
    # Fill missing one-hot columns with False if any don't exist
    for col in expected_one_hot_columns:
        if col not in new_raw_data.columns:
            new_raw_data[col] = False
            
    # Rearrange columns to expected order
    new_raw_data = new_raw_data[expected_one_hot_columns]
    
    print("Step 4: Scaling numeric columns...")
    scale_vars = ['CreditScore', 'EstimatedSalary', 'Tenure', 'Balance', 'Age', 'NumOfProducts']
    scaler = MinMaxScaler()
    new_raw_data[scale_vars] = scaler.fit_transform(new_raw_data[scale_vars])
    
    # Features (X) and Target (y)
    X = new_raw_data.drop(columns=["Exited"])
    y = new_raw_data["Exited"]
    
    print("Step 5: Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Step 6: Applying SMOTE oversampling to the training set...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("Step 7: Re-running RandomizedSearchCV hyperparameter tuning...")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 1, 5],
        "reg_lambda": [1, 10, 50],
    }
    
    xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=50,
        scoring="f1",
        cv=3,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    
    random_search.fit(X_train_resampled, y_train_resampled)
    print("Best Hyperparameters found:", random_search.best_params_)
    
    best_model = random_search.best_estimator_
    best_model.fit(X_train_resampled, y_train_resampled)
    
    # Verify performance
    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Model test accuracy: {test_acc:.2%}")
    
    # Predict on entire dataset
    all_df_predict = best_model.predict(X)
    all_df_predict_prob = best_model.predict_proba(X)
    
    print("Step 8: Exporting predictions to data/bank_churn_data.xlsx...")
    raw_raw['Exited Prediction'] = all_df_predict
    raw_raw['Exited Prediction Probability'] = all_df_predict_prob[:, 1]
    
    os.makedirs("data", exist_ok=True)
    raw_raw.to_excel(os.path.join("data", "bank_churn_data.xlsx"), index=False)
    
    print("Step 9: Exporting feature importances to assets/feature_importance.xlsx...")
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    feature_importance['Feature Importance Score'] = feature_importance['Importance'].round(4)
    
    os.makedirs("assets", exist_ok=True)
    feature_importance.to_excel(os.path.join("assets", "feature_importance.xlsx"), index=False)
    
    print("Step 10: Saving model and scaler pickles...")
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "best_model.pkl"), 'wb') as file:
        pickle.dump(best_model, file)
    with open(os.path.join("models", "scaler.pkl"), 'wb') as file:
        pickle.dump(scaler, file)
        
    print("Training process completed successfully! All files generated and saved.")

if __name__ == "__main__":
    train_model()
