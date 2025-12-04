# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="VisitWithUs — Wellness Prediction", layout="centered")

st.title("VisitWithUs — Wellness Package Purchase Prediction")
st.write("Lightweight demo UI. If the model isn't available, you'll see an error message below.")

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "RishiBond/visitwithus-wellness-model")
HF_TOKEN = os.getenv("HF_TOKEN", None)

@st.cache_resource
def load_model_from_hf():
    # Try unauthenticated download first (works if model repo is public)
    try:
        local_file = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="best_model.pkl",
            repo_type="model"
        )
        with open(local_file, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e_public:
        # If public download failed, try with token if provided (for private repos)
        if HF_TOKEN:
            try:
                local_file = hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    filename="best_model.pkl",
                    repo_type="model",
                    token=HF_TOKEN
                )
                with open(local_file, "rb") as f:
                    model = pickle.load(f)
                return model, None
            except Exception as e_token:
                return None, f"Public download failed: {e_public}; Authenticated download failed: {e_token}"
        else:
            return None, f"Public download failed: {e_public}. If the model is private, either make it public or set HF_TOKEN."

model, model_error = load_model_from_hf()

# Sidebar input controls
st.sidebar.header("Customer input")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Unknown"])
monthly_income = st.sidebar.number_input("MonthlyIncome", min_value=0, value=30000)
num_person = st.sidebar.number_input("NumberOfPersonVisiting", min_value=1, value=2)
passport = st.sidebar.selectbox("Passport (1 = Yes, 0 = No)", [1, 0])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "MonthlyIncome": monthly_income,
        "NumberOfPersonVisiting": num_person,
        "Passport": passport
    }])

    if model is None:
        st.error("Model not loaded.")
        st.info("Reason: " + (model_error or "Unknown"))
    else:
        try:
            prob = model.predict_proba(input_df)[:, 1][0]
            st.success(f"Predicted purchase probability: {prob:.3f}")
            st.write("Predicted class (1 = Will Buy, 0 = Won't Buy):", int(prob > 0.5))
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            st.write("Most likely the model expects more/different features. Check your notebook feature list.")
