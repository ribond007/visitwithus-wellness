# VisitWithUs – Wellness Package Purchase Prediction (MLOps Pipeline)

A complete end-to-end **MLOps project** to automate customer purchase prediction for the *Wellness Tourism Package* offered by **VisitWithUs**.

This project integrates:

- **Hugging Face Dataset Hub** (data versioning)
- **Model training + hyperparameter tuning**
- **Experiment tracking** (MLflow)
- **Model registry on Hugging Face Model Hub**
- **Deployment using Streamlit on Hugging Face Spaces**
- **CI/CD with GitHub Actions**

---

##  Project Goal

To predict whether a customer will purchase the Wellness Tourism Package *before contacting them, allowing smarter marketing decisions and improved customer targeting.

---

## Repository Structure
visitwithus-wellness/
│
├── data/
│ └── tourism.csv
│
├── model/
│ └── best_model.pkl (after CI/CD run)
│
├── scripts/
│ ├── train_pipeline.py
│ ├── upload_to_hf.py
│ └── utils.py
│
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
│
├── .github/
│ └── workflows/
│ └── pipeline.yml
│
└── README.md

---

##  Features Implemented

###  Data Registration
- Dataset uploaded to Hugging Face Dataset Hub  
- Automated data pull during training

###  Data Cleaning & Feature Engineering
- Missing value handling  
- Categorical encoding  
- New derived features (optional)
- Train-Test split  

###  Model Building
Models trained:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- Ensemble (optional for higher marks)

###  Experiment Tracking
- Using **MLflow**  
- Logged parameters, metrics, artifacts

###  Model Evaluation
- ROC-AUC  
- F1, Precision, Recall  
- Confusion Matrix  
- SHAP explanations  

###  Model Deployment
- Streamlit frontend  
- Hugging Face Space  
- Dockerfile for containerized serving  

###  CI/CD Pipeline
Using **GitHub Actions**:

- Pull dataset from HF  
- Run training  
- Evaluate model  
- Push best model to HF Model Hub  
- Automatic updates on new commits  

---

##  Important Links (To Be Added After Deployment)

| Component | Link |
|----------|------|
|  GitHub Repo | https://github.com/<your-username>/visitwithus-wellness |
|  HF Dataset | https://huggingface.co/datasets/<your-username>/visitwithus-wellness-data |
|  HF Model | https://huggingface.co/<your-username>/visitwithus-wellness-model |
|  HF Space (Streamlit App) | https://huggingface.co/spaces/<your-username>/visitwithus-wellness-app |
|  Notebook (HTML) | *Add link after upload* |

---

##  How to Run Locally

### 1. Clone the Repo
```bash
git clone https://github.com/<username>/visitwithus-wellness.git
cd visitwithus-wellness
