import streamlit as st
import numpy as np
import joblib

# --- MODEL LOADING ---
# Load the trained machine learning model from the file.
# @st.cache_resource ensures this is done only once.
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# --- USER INTERFACE ---
st.title("Heart Disease Prediction")
st.markdown("Enter the patient's details below to predict the likelihood of heart disease.")

if model is None:
    st.error("Model file not found. Please ensure `model.joblib` is in the same directory.")
else:
    # Create columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=100, value=52)
        sex = st.selectbox('Sex', options=[(1, 'Male'), (0, 'Female')], format_func=lambda x: x[1])[0]
        cp = st.selectbox('Chest Pain Type (CP)', options=[
            (0, 'Typical Angina'), (1, 'Atypical Angina'), 
            (2, 'Non-anginal Pain'), (3, 'Asymptomatic')
        ], format_func=lambda x: x[1])[0]
        trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=125)
        chol = st.number_input('Serum Cholesterol (chol)', min_value=100, max_value=600, value=212)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[(1, 'True'), (0, 'False')], format_func=lambda x: x[1])[0]
        
    with col2:
        restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', options=[
            (0, 'Normal'), (1, 'ST-T wave abnormality'), (2, 'Probable or definite left ventricular hypertrophy')
        ], format_func=lambda x: x[1])[0]
        thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=168)
        exang = st.selectbox('Exercise Induced Angina (exang)', options=[(1, 'Yes'), (0, 'No')], format_func=lambda x: x[1])[0]
        oldpeak = st.number_input('ST depression induced by exercise (oldpeak)', min_value=0.0, max_value=6.2, value=1.0, step=0.1)
        slope = st.selectbox('Slope of the peak exercise ST segment (slope)', options=[
            (0, 'Upsloping'), (1, 'Flat'), (2, 'Downsloping')
        ], format_func=lambda x: x[1])[0]
        ca = st.selectbox('Number of major vessels colored by fluoroscopy (ca)', options=[0, 1, 2, 3, 4])
        thal = st.selectbox('Thalassemia (thal)', options=[
            (0, 'Normal'), (1, 'Fixed defect'), (2, 'Reversible defect'), (3, 'Unknown')
        ], format_func=lambda x: x[1])[0]

    # --- PREDICTION LOGIC ---
    if st.button('**Predict**', use_container_width=True):
        # Prepare the feature array in the correct order
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        ]])

        # Make a prediction
        prediction = model.predict(features)
        
        st.write("---")
        st.subheader("Prediction Result")

        # Display the result
        if prediction[0] == 1:
            st.error('**High probability of Heart Disease**', icon="💔")
            st.warning("This is a prediction based on the provided data and does not constitute medical advice. Please consult a healthcare professional.", icon="⚠️")
        else:
            st.success('**Low probability of Heart Disease**', icon="💖")
            st.info("This is a prediction based on the provided data. Continue to maintain a healthy lifestyle and consult a doctor for regular check-ups.", icon="ℹ️")

