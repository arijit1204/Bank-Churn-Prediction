import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import os

# Set Streamlit layout to wide
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and scaler
@st.cache_resource
def load_assets():
    model_path = os.path.join("models", "best_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.info("Make sure to run 'python train.py' to generate model artifacts first.")
    st.stop()

# Sidebar Setup
sidebar_image_path = os.path.join("assets", "pic_1.png")
if os.path.exists(sidebar_image_path):
    st.sidebar.image(sidebar_image_path, use_column_width=True)
else:
    st.sidebar.warning("Sidebar image not found. Ensure assets/pic_1.png exists.")

st.sidebar.header("Customer Profile Input")

# Collect User Inputs using clean UI controls
credit_score = st.sidebar.slider("Credit Score", min_value=350, max_value=850, value=600, step=1)
age = st.sidebar.slider("Age (Years)", min_value=18, max_value=100, value=30, step=1)
tenure = st.sidebar.slider("Tenure (Years with Bank)", min_value=0, max_value=10, value=2, step=1)
balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=8000.0, step=500.0)
num_of_products = st.sidebar.selectbox("Number of Products", options=[1, 2, 3, 4], index=1)
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", min_value=0.0, max_value=300000.0, value=60000.0, step=1000.0)

geography = st.sidebar.selectbox("Geography", options=["France", "Germany", "Spain"], index=0)
gender = st.sidebar.radio("Gender", options=["Female", "Male"], index=0)
has_cr_card = st.sidebar.selectbox("Has Credit Card?", options=["Yes", "No"], index=0)
is_active_member = st.sidebar.selectbox("Is Active Member?", options=["Yes", "No"], index=0)

# Map selectboxes to one-hot columns (exact standard expected by model)
geography_france = 1.0 if geography == "France" else 0.0
geography_germany = 1.0 if geography == "Germany" else 0.0
geography_spain = 1.0 if geography == "Spain" else 0.0

gender_female = 1.0 if gender == "Female" else 0.0
gender_male = 1.0 if gender == "Male" else 0.0

has_cr_card_0 = 1.0 if has_cr_card == "No" else 0.0
has_cr_card_1 = 1.0 if has_cr_card == "Yes" else 0.0

is_active_member_0 = 1.0 if is_active_member == "No" else 0.0
is_active_member_1 = 1.0 if is_active_member == "Yes" else 0.0

# Prepare raw user inputs dict (exactly in the sequence expected by the model columns)
user_inputs = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "EstimatedSalary": estimated_salary,
    "Geography_France": geography_france,
    "Geography_Germany": geography_germany,
    "Geography_Spain": geography_spain,
    "Gender_Female": gender_female,
    "Gender_Male": gender_male,
    "HasCrCard_0": has_cr_card_0,
    "HasCrCard_1": has_cr_card_1,
    "IsActiveMember_0": is_active_member_0,
    "IsActiveMember_1": is_active_member_1
}

# Convert input dict to a DataFrame
input_data = pd.DataFrame([user_inputs])

# Scale numeric variables
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]
input_data_scaled = input_data.copy()
input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

# App Header
header_image_path = os.path.join("assets", "pic_2.png")
if os.path.exists(header_image_path):
    st.image(header_image_path, use_column_width=True)
else:
    st.warning("Header image not found. Ensure assets/pic_2.png exists.")

st.markdown("<h1 style='text-align: center; color: #1e3d59;'>Bank Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #17b978;'>Predict the likelihood of customer attrition and optimize retention campaigns</p>", unsafe_allow_html=True)
st.markdown("---")

# Page Columns
left_col, right_col = st.columns([1, 1])

# Left Column: Feature Importance
with left_col:
    st.subheader("📊 Feature Importance Analysis")
    st.markdown("See which factors are most critical in determining customer churn.")
    
    importance_path = os.path.join("assets", "feature_importance.xlsx")
    if os.path.exists(importance_path):
        feature_importance_df = pd.read_excel(importance_path, usecols=["Feature", "Feature Importance Score"])
        
        # Plot the feature importance bar chart
        fig = px.bar(
            feature_importance_df.sort_values(by="Feature Importance Score", ascending=True),
            x="Feature Importance Score",
            y="Feature",
            orientation="h",
            labels={"Feature Importance Score": "Importance Score", "Feature": "Factors"},
            color="Feature Importance Score",
            color_continuous_scale=px.colors.sequential.Tealgrn,
            height=500
        )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance data not found. Run training script to generate.")

# Right Column: Prediction Results
with right_col:
    st.subheader("🔍 Prediction Control Center")
    st.write("Click the button below to analyze the customer based on the selected attributes in the sidebar.")
    
    predict_btn = st.button("🔮 Calculate Churn Risk", use_container_width=True)
    
    if predict_btn:
        # Get predictions
        probabilities = model.predict_proba(input_data_scaled)[0]
        prediction = model.predict(input_data_scaled)[0]
        
        churn_prob = probabilities[1]
        retain_prob = probabilities[0]
        
        st.markdown("---")
        st.subheader("Results Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Churn Probability", value=f"{churn_prob:.2%}")
        with col2:
            st.metric(label="Retention Probability", value=f"{retain_prob:.2%}")
            
        # Display styled output based on predicted class
        if prediction == 1:
            st.error(f"### Output: **Churn Alert (Likely to Attrit)**")
            st.markdown(
                f"<div style='background-color:#ffe6e6; padding:15px; border-radius:5px; border-left: 6px solid #ff4d4d; color: #800000;'>"
                f"⚠️ **Warning:** The model predicts a high churn risk of **{churn_prob:.1%}** for this customer profile. "
                f"It is recommended to engage this customer with special offers, waivers, or active relationship management support."
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.success(f"### Output: **Retained (Likely to Stay)**")
            st.markdown(
                f"<div style='background-color:#e6ffec; padding:15px; border-radius:5px; border-left: 6px solid #2eb85c; color: #00591e;'>"
                f"✅ **Good News:** The customer profile shows a low churn risk, with a retention probability of **{retain_prob:.1%}**. "
                f"Normal account engagement and standard support levels are sufficient for keeping this customer satisfied."
                f"</div>",
                unsafe_allow_html=True
            )
