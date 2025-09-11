import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========================
# 🎯 Load the Trained Model
# ========================
# Replace "best_model.pkl" with the file where you saved your trained model
model = joblib.load("best_model_logistic_regression.pkl")
scaler = joblib.load("scaler (1).pkl")

# ========================
# 🎨 Streamlit UI
# ========================
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether a patient is **Diabetic** or **Non-Diabetic** based on health metrics.")

# ========================
# 📥 User Inputs
# ========================
st.sidebar.header("Enter Patient Data")

pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=1000, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)

# ========================
# 📊 Prepare Data
# ========================
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])

# Scale data (important to use same scaler from training)
input_data_scaled = scaler.transform(input_data)

# ========================
# 🤖 Make Prediction
# ========================
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data_scaled)[0]
    proba = model.predict_proba(input_data_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The patient is **Diabetic** with probability {proba:.2f}")
    else:
        st.success(f"✅ The patient is **Non-Diabetic** with probability {1-proba:.2f}")

# ========================
# ℹ️ Footer
# ========================
st.markdown("""
---
Made with ❤️ using **Streamlit** and Machine Learning  
""")

