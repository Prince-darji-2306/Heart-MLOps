import joblib
import streamlit as st
# import mlflow
# import mlflow.pyfunc
import pandas as pd

# For Reference of Local Direct Deployment from production.
# Set MLflow tracking URI
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load model using alias
# MODEL_NAME = "HeartDiseaseCatBoost-V1"
# model = mlflow.pyfunc.load_model(
#     f"models:/{MODEL_NAME}@production"
# )

model = joblib.load("models/catboost_model.pkl")

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below:")

# Example input fields (modify according to your dataset)
age              = st.slider("Age", 20, 90, 50)
sex              = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
chest_pain       = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
bp               = st.slider("BP", 80, 200, 120)
cholesterol      = st.slider("Cholesterol", 100, 600, 200)
fbs              = st.selectbox("FBS over 120", [0, 1])
ekg              = st.selectbox("EKG Results", [0, 1, 2])
max_hr           = st.slider("Max HR", 60, 220, 150)
exercise_angina  = st.selectbox("Exercise Angina", [0, 1])
st_depression    = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
slope_of_st      = st.selectbox("Slope of ST", [1, 2, 3])
vessels_fluro    = st.selectbox("Number of Vessels Fluro", [0, 1, 2, 3])
thallium         = st.selectbox("Thallium", [3, 6, 7])

if st.button("Predict"):

    input_data = pd.DataFrame([{
    "id" : 0, "Age": age, "Sex": sex, "Chest pain type": chest_pain,
    "BP": bp, "Cholesterol": cholesterol, "FBS over 120": fbs,
    "EKG results": ekg, "Max HR": max_hr, "Exercise angina": exercise_angina,
    "ST depression": st_depression, "Slope of ST": slope_of_st,
    "Number of vessels fluro": vessels_fluro, "Thallium": thallium
    }])

    prediction = model.predict(input_data)
    if prediction[0] == 'Presence':
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")