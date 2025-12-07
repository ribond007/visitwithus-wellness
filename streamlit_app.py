# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="VisitWithUs — Wellness Prediction", layout="centered")

st.title("VisitWithUs — Wellness Package Purchase Prediction")
st.write("Lightweight demo UI. If the model isn't available, you'll see an error message below.")

# Hardcode working model repo (model is public)
MODEL_REPO = "RishiBond/visitwithus-wellness-model-docker"
HF_DATASET_REPO = "RishiBond/visitwithus-wellness-data"


@st.cache_resource
def load_model_from_hf():
    """Download and load the best model from HF Model Hub."""
    try:
        local_file = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="best_model.pkl",
            repo_type="model",
        )
        with open(local_file, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def get_template_row():
    """
    Load tourism.csv from HF dataset hub and return
    a single-row DataFrame with ALL feature columns.
    We will modify only a few columns based on user input.
    """
    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="tourism.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(path)

    # Use all columns except target as features
    feature_cols = [c for c in df.columns if c != "ProdTaken"]
    row = df[feature_cols].iloc[0:1].copy()
    return row


model, model_error = load_model_from_hf()

# Sidebar input controls
st.sidebar.header("Customer input")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
monthly_income = st.sidebar.number_input("MonthlyIncome", min_value=0, value=30000)
num_person = st.sidebar.number_input("NumberOfPersonVisiting", min_value=1, value=2)
passport = st.sidebar.selectbox("Passport (1 = Yes, 0 = No)", [1, 0])

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded.")
        st.info("Reason: " + (model_error or "Unknown"))
    else:
        try:
            # Start from a full feature template row
            input_df = get_template_row()

            # Overwrite only the fields we collect
            input_df.loc[:, "Age"] = age
            input_df.loc[:, "Gender"] = gender
            input_df.loc[:, "MonthlyIncome"] = monthly_income
            input_df.loc[:, "NumberOfPersonVisiting"] = num_person
            input_df.loc[:, "Passport"] = passport

            # Predict
            prob = model.predict_proba(input_df)[:, 1][0]
            st.success(f"Predicted purchase probability: {prob:.3f}")
            st.write("Predicted class (1 = Will Buy, 0 = Won't Buy):", int(prob > 0.5))
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            st.write("Most likely the model expects different features. Check your notebook feature list.")
