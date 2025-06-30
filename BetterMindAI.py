import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# 🚀 Load and preprocess data
@st.cache_data
def load_and_train():
    df = pd.read_csv('healthcare-dataset-stroke-data(in) (2).csv')
    df = df.drop(columns=['id'])
    df = df.dropna(subset=['stroke'])

    df['bmi'] = df['bmi'].replace(0, np.nan)
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)

    y = df['stroke']
    X = df.drop(columns=['stroke'])

    X = pd.get_dummies(X, columns=['gender', 'Residence_type', 'smoking_status'], drop_first=True)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, y)

    return model, X.columns

# 🔁 Load model
model, model_columns = load_and_train()

# 🧠 Prediction function
def predict_stroke(patient_dict):
    input_df = pd.DataFrame([patient_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    return pred, prob

# 🌐 Streamlit UI
st.title("🩺 Stroke Risk Predictor")

st.markdown("### Enter Patient Information:")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 50)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])

with col2:
    work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    Residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

if st.button("Predict Stroke Risk"):
    patient_data = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender': gender.lower(),
        'Residence_type': Residence_type,
        'smoking_status': smoking_status
    }

    pred, prob = predict_stroke(patient_data)

    st.subheader("🧠 Prediction Result:")
    st.success("Stroke" if pred == 1 else "No Stroke")
    st.info(f"Probability of Stroke: {prob:.4f}")
