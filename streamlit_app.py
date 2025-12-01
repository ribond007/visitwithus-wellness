# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from huggingface_hub import hf_hub_download
from pathlib import Path
import os

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "your-hf-username/visitwithus-wellness-model")
HF_TOKEN = os.getenv("HF_TOKEN", None)

@st.cache_resource
def load_model():
    # downloads file from HF model repo
    try:
        local = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.pkl", repo_type="model", token=HF_TOKEN)
    except Exception as e:
        st.error("Failed to download model from HF. Ensure HF_MODEL_REPO and HF_TOKEN are set.")
        raise e
    with open(local, "rb") as f:
        model = pickle.load(f)
    return model

st.title("VisitWithUs — Wellness Package Purchase Prediction")
st.write("Fill customer details and click Predict.")

model = load_model()

# Minimal set of input fields (add rest as required)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
monthly_income = st.number_input("MonthlyIncome", min_value=0, value=30000)
num_person = st.number_input("NumberOfPersonVisiting", min_value=1, value=2)
gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
passport = st.selectbox("Passport (1=Yes)", [1, 0])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": monthly_income,
        "NumberOfPersonVisiting": num_person,
        "Gender": gender,
        "Passport": passport
    }])
    try:
        prob = model.predict_proba(input_df)[:,1][0]
        label = int(prob > 0.5)
        st.metric("Probability of purchase", f"{prob:.3f}")
        st.write("Predicted Label (0 = No, 1 = Yes):", label)
    except Exception as e:
        st.error("Error during prediction — check that model expects same feature names and preprocessing.")
        st.exception(e)
